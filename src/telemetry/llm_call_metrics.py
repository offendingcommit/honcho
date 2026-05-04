"""Per-`honcho_llm_call` observability.

Captures one structured log line + Prometheus sample set per LLM call,
covering every feature (deriver, dialectic, dream, summary). Designed to
be the *only* instrumentation point inside `honcho_llm_call`; subsystem-
specific token counters in `prometheus.metrics` continue to work in parallel.

Outcome classification distinguishes "the model didn't converge"
(`error_max_iterations`) from "the infra broke" (timeout / validation /
other) so dashboards and alerts can target each independently.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.exceptions import ValidationException
from src.telemetry.prometheus import LLMCallOutcome, prometheus_metrics

if TYPE_CHECKING:
    from src.config import ModelConfig
    from src.llm.types import HonchoLLMCallResponse

logger = logging.getLogger("honcho.llm.call")


# Track names used by callers, mapped to clean Prom label values. Anything
# not in the table falls through `_normalize_track_name` (lower + underscores).
_FEATURE_LABEL_MAP: dict[str, str] = {
    "Minimal Deriver": "deriver",
    "Dialectic Agent": "dialectic",
    "Dialectic Agent Stream": "dialectic_stream",
}


def normalize_feature_label(track_name: str | None, trace_name: str | None) -> str:
    """Map caller's `track_name`/`trace_name` to a low-cardinality Prom label.

    Prefers explicit `_FEATURE_LABEL_MAP` matches, then snake-cases the
    track_name (e.g. ``"Dreamer/deduction"`` → ``"dream_deduction"``),
    falling back to trace_name, then to ``"unknown"``.
    """
    if track_name and track_name in _FEATURE_LABEL_MAP:
        return _FEATURE_LABEL_MAP[track_name]
    raw = track_name or trace_name
    if not raw:
        return "unknown"
    # "Dreamer/Deduction" → "dreamer_deduction" → "dream_deduction"
    s = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    s = s.replace("dreamer_", "dream_")
    return s or "unknown"


@dataclass
class _CallState:
    """Mutable observation collected across the lifetime of one call."""

    feature: str
    primary_provider: str
    primary_model: str
    has_backup: bool
    backup_provider: str | None = None
    backup_model: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    final_provider: str = ""
    final_model: str = ""
    attempts: int = 1
    iterations: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    tool_calls: int = 0
    used_backup: bool = False
    outcome: LLMCallOutcome = LLMCallOutcome.SUCCESS
    error_class: str | None = None


def _classify_error(exc: BaseException) -> LLMCallOutcome:
    """Map an exception to a coarse outcome bucket."""
    if isinstance(exc, ValidationException):
        return LLMCallOutcome.ERROR_VALIDATION
    if isinstance(exc, TimeoutError):
        return LLMCallOutcome.ERROR_TIMEOUT
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return LLMCallOutcome.ERROR_TIMEOUT
    if "pydantic" in name and "validation" in name:
        return LLMCallOutcome.ERROR_VALIDATION
    return LLMCallOutcome.ERROR_OTHER


@contextmanager
def observe_llm_call(
    *,
    track_name: str | None,
    trace_name: str | None,
    runtime_model_config: ModelConfig,
) -> Iterator[_CallState]:
    """Wrap one `honcho_llm_call` invocation with metrics + structured logging.

    The caller mutates the yielded `_CallState` (sets `attempts`, `iterations`,
    accumulates tokens, etc.) over the call's lifetime; on exit we emit the
    Prometheus samples and a single logfmt line. Metric errors are swallowed
    inside `prometheus_metrics`; this wrapper never raises.
    """
    fb = runtime_model_config.fallback
    state = _CallState(
        feature=normalize_feature_label(track_name, trace_name),
        primary_provider=str(runtime_model_config.transport),
        primary_model=str(runtime_model_config.model),
        has_backup=fb is not None,
        backup_provider=str(fb.transport) if fb is not None else None,
        backup_model=str(fb.model) if fb is not None else None,
    )
    # Default the "winning" provider/model to primary; the caller updates
    # these post-success based on the actual AttemptPlan that returned.
    state.final_provider = state.primary_provider
    state.final_model = state.primary_model

    try:
        yield state
    except BaseException as exc:
        state.outcome = _classify_error(exc)
        state.error_class = type(exc).__name__
        _emit(state)
        raise
    else:
        # Success — caller is responsible for setting iterations / final
        # provider / outcome (via `finalize_success` below) before exit.
        _emit(state)


def finalize_success(
    state: _CallState,
    *,
    response: HonchoLLMCallResponse[Any] | None,
    final_provider: str | None,
    final_model: str | None,
    attempts: int,
    iterations: int | None,
    has_backup: bool,
) -> None:
    """Populate `state` from a successful response and pick the outcome bucket.

    Called by `honcho_llm_call` right before the context manager exits when
    no exception was raised. `iterations` is None for tool-less calls.
    """
    state.attempts = max(1, attempts)
    state.iterations = iterations
    if final_provider:
        state.final_provider = final_provider
    if final_model:
        state.final_model = final_model
    state.used_backup = (
        has_backup
        and state.final_model != state.primary_model
        and state.backup_model is not None
        and state.final_model == state.backup_model
    )
    if response is not None:
        state.input_tokens = int(response.input_tokens or 0)
        state.output_tokens = int(response.output_tokens or 0)
        state.cache_creation_input_tokens = int(
            response.cache_creation_input_tokens or 0
        )
        state.cache_read_input_tokens = int(response.cache_read_input_tokens or 0)
        state.tool_calls = len(response.tool_calls_made or [])

    if state.used_backup:
        state.outcome = LLMCallOutcome.SUCCESS_VIA_BACKUP
    elif state.attempts > 1:
        state.outcome = LLMCallOutcome.SUCCESS_AFTER_RETRY
    else:
        state.outcome = LLMCallOutcome.SUCCESS


def mark_max_iterations(state: _CallState, iterations: int) -> None:
    """Mark a tool-loop call that hit `max_tool_iterations` without converging.

    Called when execute_tool_loop returns from the synthesis fallback path
    rather than from natural convergence. The call still returned content,
    but the model didn't decide to stop on its own — different reliability
    signal than a clean success.
    """
    state.iterations = iterations
    state.outcome = LLMCallOutcome.ERROR_MAX_ITERATIONS


def _emit(state: _CallState) -> None:
    duration = time.monotonic() - state.started_at
    outcome_value = state.outcome.value

    prometheus_metrics.record_llm_call(
        feature=state.feature,
        provider=state.final_provider,
        model=state.final_model,
        outcome=outcome_value,
        duration_seconds=duration,
    )
    if state.input_tokens:
        prometheus_metrics.record_llm_tokens(
            feature=state.feature,
            provider=state.final_provider,
            model=state.final_model,
            token_type="input",
            count=state.input_tokens,
        )
    if state.output_tokens:
        prometheus_metrics.record_llm_tokens(
            feature=state.feature,
            provider=state.final_provider,
            model=state.final_model,
            token_type="output",
            count=state.output_tokens,
        )
    if state.cache_read_input_tokens:
        prometheus_metrics.record_llm_tokens(
            feature=state.feature,
            provider=state.final_provider,
            model=state.final_model,
            token_type="cache_read",
            count=state.cache_read_input_tokens,
        )
    if state.cache_creation_input_tokens:
        prometheus_metrics.record_llm_tokens(
            feature=state.feature,
            provider=state.final_provider,
            model=state.final_model,
            token_type="cache_creation",
            count=state.cache_creation_input_tokens,
        )
    if state.iterations is not None:
        prometheus_metrics.observe_llm_iterations(
            feature=state.feature,
            outcome=outcome_value,
            iterations=state.iterations,
        )
    if state.used_backup and state.backup_provider and state.backup_model:
        prometheus_metrics.record_llm_backup_used(
            feature=state.feature,
            primary_provider=state.primary_provider,
            primary_model=state.primary_model,
            backup_provider=state.backup_provider,
            backup_model=state.backup_model,
        )

    # One structured logfmt line per call. Quote-free values keep `| logfmt`
    # parsing in Loki/Grafana straightforward.
    iter_value = state.iterations if state.iterations is not None else 0
    err_part = f" error_class={state.error_class}" if state.error_class else ""
    line = (
        f"honcho.llm.call feature={state.feature}"
        f" provider={state.final_provider} model={state.final_model}"
        f" outcome={outcome_value} latency_ms={int(duration * 1000)}"
        f" attempts={state.attempts}"
        f" used_backup={'true' if state.used_backup else 'false'}"
        f" input_tokens={state.input_tokens}"
        f" output_tokens={state.output_tokens}"
        f" cache_read_tokens={state.cache_read_input_tokens}"
        f" cache_creation_tokens={state.cache_creation_input_tokens}"
        f" tool_calls={state.tool_calls} iterations={iter_value}"
        f"{err_part}"
    )
    logger.info(line)


__all__ = [
    "finalize_success",
    "mark_max_iterations",
    "normalize_feature_label",
    "observe_llm_call",
]
