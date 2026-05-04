"""Prometheus metrics for Honcho."""

from __future__ import annotations

import logging
from enum import Enum
from typing import cast, final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings

disable_created_metrics()

logger = logging.getLogger(__name__)


class NamespacedCounter(Counter):
    def labels(self, **kwargs: str) -> NamespacedCounter:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class NamespacedGauge(Gauge):
    def labels(self, **kwargs: str) -> NamespacedGauge:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class NamespacedHistogram(Histogram):
    def labels(self, **kwargs: str) -> NamespacedHistogram:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class TokenTypes(Enum):
    INPUT = "input"
    OUTPUT = "output"


class DeriverTaskTypes(Enum):
    INGESTION = "ingestion"
    SUMMARY = "summary"


class DeriverComponents(Enum):
    PROMPT = "prompt"
    MESSAGES = "messages"
    PREVIOUS_SUMMARY = "previous_summary"
    OUTPUT_TOTAL = "output_total"


class DialecticComponents(Enum):
    TOTAL = "total"


class LLMCallOutcome(Enum):
    """Terminal outcome of a single `honcho_llm_call`.

    Distinguishes "model didn't converge" (max_iterations) from "infra broke"
    (timeout/validation/other) so dashboards and alerts can target each
    independently. `success_via_backup` is its own bucket so silent failover
    rate is observable without parsing logs.
    """

    SUCCESS = "success"
    SUCCESS_AFTER_RETRY = "success_after_retry"
    SUCCESS_VIA_BACKUP = "success_via_backup"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_VALIDATION = "error_validation"
    ERROR_MAX_ITERATIONS = "error_max_iterations"
    ERROR_OTHER = "error_other"


api_requests_counter = NamespacedCounter(
    "api_requests",
    "Total API requests",
    ["namespace", "method", "endpoint", "status_code"],
)

api_request_duration_histogram = NamespacedHistogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["namespace", "method", "endpoint", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

messages_created_counter = NamespacedCounter(
    "messages_created",
    "Total messages created",
    ["namespace", "workspace_name", "session_name"],
)

session_context_requests_counter = NamespacedCounter(
    "session_context_requests",
    "Total session context requests",
    ["namespace", "workspace_name", "session_name"],
)

session_search_requests_counter = NamespacedCounter(
    "session_search_requests",
    "Total session search requests",
    ["namespace", "workspace_name", "session_name"],
)

dialectic_calls_counter = NamespacedCounter(
    "dialectic_calls",
    "Total dialectic calls",
    ["namespace", "workspace_name", "reasoning_level"],
)

deriver_queue_items_processed_counter = NamespacedCounter(
    "deriver_queue_items_processed",
    "Total deriver queue items processed",
    ["namespace", "workspace_name", "task_type"],
)

deriver_queue_items_enqueued_counter = NamespacedCounter(
    "deriver_queue_items_enqueued",
    "Total deriver queue items enqueued",
    ["namespace", "workspace_name", "task_type"],
)

deriver_tokens_processed_counter = NamespacedCounter(
    "deriver_tokens_processed",
    "Total tokens processed by the deriver",
    ["namespace", "task_type", "token_type", "component"],
)

dialectic_tokens_processed_counter = NamespacedCounter(
    "dialectic_tokens_processed",
    "Total tokens processed by the dialectic",
    ["namespace", "token_type", "component", "reasoning_level"],
)

dreamer_tokens_processed_counter = NamespacedCounter(
    "dreamer_tokens_processed",
    "Total tokens processed by the dreamer",
    ["namespace", "specialist_name", "token_type"],
)

deriver_active_workers_gauge = NamespacedGauge(
    "deriver_active_workers",
    "Current number of deriver workers actively processing work units",
    ["namespace"],
)

deriver_queue_depth_gauge = NamespacedGauge(
    "deriver_queue_depth",
    "Current queue depth by workspace, task type, and state",
    ["namespace", "workspace_name", "task_type", "state"],
)

deriver_queue_oldest_age_gauge = NamespacedGauge(
    "deriver_queue_oldest_age_seconds",
    "Age in seconds of the oldest queue item by workspace, task type, and state",
    ["namespace", "workspace_name", "task_type", "state"],
)

deriver_queue_error_backlog_gauge = NamespacedGauge(
    "deriver_queue_error_backlog",
    "Current count of errored queue items retained in the queue table",
    ["namespace", "workspace_name", "task_type"],
)

deriver_queue_errors_counter = NamespacedCounter(
    "deriver_queue_errors",
    "Total deriver queue item processing errors",
    ["namespace", "workspace_name", "task_type"],
)

deriver_queue_item_latency_histogram = NamespacedHistogram(
    "deriver_queue_item_latency_seconds",
    "Queue item latency from enqueue to terminal state",
    ["namespace", "workspace_name", "task_type", "outcome"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

sessions_active_gauge = NamespacedGauge(
    "sessions_active",
    "Current number of active sessions by workspace",
    ["namespace", "workspace_name"],
)

session_last_message_age_gauge = NamespacedGauge(
    "session_last_message_age_seconds",
    "Age in seconds since the last message in an active session",
    ["namespace", "workspace_name", "session_name"],
)

session_queue_depth_gauge = NamespacedGauge(
    "session_queue_depth",
    "Current queue depth by workspace, session, and state",
    ["namespace", "workspace_name", "session_name", "state"],
)

session_queue_oldest_age_gauge = NamespacedGauge(
    "session_queue_oldest_age_seconds",
    "Age in seconds of the oldest queue item by workspace, session, and state",
    ["namespace", "workspace_name", "session_name", "state"],
)

# ---- Per-LLM-call observability ---------------------------------------------
# Cardinality budget: feature ~6, provider ~4, model ~10, outcome 7 → ~1.7k
# series cap. Deliberately no workspace_name label here: the question these
# answer is "is this model effective for this feature", not "is workspace X
# slow". Per-workspace LLM behavior shows up in dialectic_calls + token
# counters which already carry workspace_name.

llm_calls_counter = NamespacedCounter(
    "llm_calls",
    "Total honcho_llm_call invocations by feature, provider, model, outcome",
    ["namespace", "feature", "provider", "model", "outcome"],
)

llm_call_duration_histogram = NamespacedHistogram(
    "llm_call_duration_seconds",
    "End-to-end honcho_llm_call latency (includes retries and backup failover)",
    ["namespace", "feature", "provider", "model", "outcome"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

# Distinct from the existing deriver/dialectic/dreamer token counters:
# this one carries provider+model so we can answer "tokens through gemini
# vs glm-5.1" without bouncing through Langfuse.
llm_tokens_counter = NamespacedCounter(
    "llm_tokens",
    "LLM tokens by feature/provider/model/token_type",
    ["namespace", "feature", "provider", "model", "token_type"],
)

llm_tool_calls_counter = NamespacedCounter(
    "llm_tool_calls",
    "Individual tool invocations within an LLM tool loop",
    ["namespace", "feature", "tool_name", "outcome"],
)

llm_iterations_histogram = NamespacedHistogram(
    "llm_iterations",
    "Tool-loop iterations consumed per call (1 = no tool calls)",
    ["namespace", "feature", "outcome"],
    buckets=(1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50),
)

llm_backup_used_counter = NamespacedCounter(
    "llm_backup_used",
    "Counts when a call's retry chain switched from primary to backup provider",
    [
        "namespace",
        "feature",
        "primary_provider",
        "primary_model",
        "backup_provider",
        "backup_model",
    ],
)


@final
class PrometheusMetrics:
    _instance: PrometheusMetrics | None = None

    def __new__(cls) -> PrometheusMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _handle_metric_error(self, method_name: str, error: Exception) -> None:
        import sentry_sdk

        sentry_sdk.capture_exception(error)
        logger.warning(
            "Failed to record Prometheus metric in %s: %s", method_name, str(error)
        )

    def record_api_request(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: str,
    ) -> None:
        try:
            api_requests_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_api_request", e)

    def record_api_request_duration(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: str,
        duration_seconds: float,
    ) -> None:
        try:
            api_request_duration_histogram.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).observe(duration_seconds)
        except Exception as e:
            self._handle_metric_error("record_api_request_duration", e)

    def record_messages_created(
        self,
        *,
        count: int,
        workspace_name: str,
        session_name: str,
    ) -> None:
        try:
            messages_created_counter.labels(
                workspace_name=workspace_name,
                session_name=session_name,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_messages_created", e)

    def record_session_context_request(
        self,
        *,
        workspace_name: str,
        session_name: str,
    ) -> None:
        try:
            session_context_requests_counter.labels(
                workspace_name=workspace_name,
                session_name=session_name,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_session_context_request", e)

    def record_session_search_request(
        self,
        *,
        workspace_name: str,
        session_name: str,
    ) -> None:
        try:
            session_search_requests_counter.labels(
                workspace_name=workspace_name,
                session_name=session_name,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_session_search_request", e)

    def record_dialectic_call(
        self,
        *,
        workspace_name: str,
        reasoning_level: str,
    ) -> None:
        try:
            dialectic_calls_counter.labels(
                workspace_name=workspace_name,
                reasoning_level=reasoning_level,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_dialectic_call", e)

    def record_deriver_queue_item(
        self,
        *,
        count: int,
        workspace_name: str,
        task_type: str,
    ) -> None:
        try:
            deriver_queue_items_processed_counter.labels(
                workspace_name=workspace_name,
                task_type=task_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_deriver_queue_item", e)

    def record_deriver_queue_item_enqueued(
        self,
        *,
        count: int,
        workspace_name: str,
        task_type: str,
    ) -> None:
        try:
            deriver_queue_items_enqueued_counter.labels(
                workspace_name=workspace_name,
                task_type=task_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_deriver_queue_item_enqueued", e)

    def record_deriver_tokens(
        self,
        *,
        count: int,
        task_type: str,
        token_type: str,
        component: str,
    ) -> None:
        try:
            deriver_tokens_processed_counter.labels(
                task_type=task_type,
                token_type=token_type,
                component=component,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_deriver_tokens", e)

    def record_dialectic_tokens(
        self,
        *,
        count: int,
        token_type: str,
        component: str,
        reasoning_level: str,
    ) -> None:
        try:
            dialectic_tokens_processed_counter.labels(
                token_type=token_type,
                component=component,
                reasoning_level=reasoning_level,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_dialectic_tokens", e)

    def record_dreamer_tokens(
        self,
        *,
        count: int,
        specialist_name: str,
        token_type: str,
    ) -> None:
        try:
            dreamer_tokens_processed_counter.labels(
                specialist_name=specialist_name,
                token_type=token_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_dreamer_tokens", e)

    def set_deriver_active_workers(self, *, count: int) -> None:
        try:
            deriver_active_workers_gauge.labels().set(count)
        except Exception as e:
            self._handle_metric_error("set_deriver_active_workers", e)

    def set_deriver_queue_depth(
        self,
        *,
        workspace_name: str,
        task_type: str,
        state: str,
        count: int,
    ) -> None:
        try:
            deriver_queue_depth_gauge.labels(
                workspace_name=workspace_name,
                task_type=task_type,
                state=state,
            ).set(count)
        except Exception as e:
            self._handle_metric_error("set_deriver_queue_depth", e)

    def set_deriver_queue_oldest_age(
        self,
        *,
        workspace_name: str,
        task_type: str,
        state: str,
        age_seconds: float,
    ) -> None:
        try:
            deriver_queue_oldest_age_gauge.labels(
                workspace_name=workspace_name,
                task_type=task_type,
                state=state,
            ).set(age_seconds)
        except Exception as e:
            self._handle_metric_error("set_deriver_queue_oldest_age", e)

    def set_deriver_queue_error_backlog(
        self,
        *,
        workspace_name: str,
        task_type: str,
        count: int,
    ) -> None:
        try:
            deriver_queue_error_backlog_gauge.labels(
                workspace_name=workspace_name,
                task_type=task_type,
            ).set(count)
        except Exception as e:
            self._handle_metric_error("set_deriver_queue_error_backlog", e)

    def record_deriver_queue_error(
        self,
        *,
        workspace_name: str,
        task_type: str,
    ) -> None:
        try:
            deriver_queue_errors_counter.labels(
                workspace_name=workspace_name,
                task_type=task_type,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_deriver_queue_error", e)

    def observe_deriver_queue_item_latency(
        self,
        *,
        workspace_name: str,
        task_type: str,
        outcome: str,
        latency_seconds: float,
    ) -> None:
        try:
            deriver_queue_item_latency_histogram.labels(
                workspace_name=workspace_name,
                task_type=task_type,
                outcome=outcome,
            ).observe(latency_seconds)
        except Exception as e:
            self._handle_metric_error("observe_deriver_queue_item_latency", e)

    def set_sessions_active(
        self,
        *,
        workspace_name: str,
        count: int,
    ) -> None:
        try:
            sessions_active_gauge.labels(workspace_name=workspace_name).set(count)
        except Exception as e:
            self._handle_metric_error("set_sessions_active", e)

    def set_session_last_message_age(
        self,
        *,
        workspace_name: str,
        session_name: str,
        age_seconds: float,
    ) -> None:
        try:
            session_last_message_age_gauge.labels(
                workspace_name=workspace_name,
                session_name=session_name,
            ).set(age_seconds)
        except Exception as e:
            self._handle_metric_error("set_session_last_message_age", e)

    def set_session_queue_depth(
        self,
        *,
        workspace_name: str,
        session_name: str,
        state: str,
        count: int,
    ) -> None:
        try:
            session_queue_depth_gauge.labels(
                workspace_name=workspace_name,
                session_name=session_name,
                state=state,
            ).set(count)
        except Exception as e:
            self._handle_metric_error("set_session_queue_depth", e)

    def set_session_queue_oldest_age(
        self,
        *,
        workspace_name: str,
        session_name: str,
        state: str,
        age_seconds: float,
    ) -> None:
        try:
            session_queue_oldest_age_gauge.labels(
                workspace_name=workspace_name,
                session_name=session_name,
                state=state,
            ).set(age_seconds)
        except Exception as e:
            self._handle_metric_error("set_session_queue_oldest_age", e)

    def record_llm_call(
        self,
        *,
        feature: str,
        provider: str,
        model: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        try:
            llm_calls_counter.labels(
                feature=feature,
                provider=provider,
                model=model,
                outcome=outcome,
            ).inc()
            llm_call_duration_histogram.labels(
                feature=feature,
                provider=provider,
                model=model,
                outcome=outcome,
            ).observe(duration_seconds)
        except Exception as e:
            self._handle_metric_error("record_llm_call", e)

    def record_llm_tokens(
        self,
        *,
        feature: str,
        provider: str,
        model: str,
        token_type: str,
        count: int,
    ) -> None:
        if count <= 0:
            return
        try:
            llm_tokens_counter.labels(
                feature=feature,
                provider=provider,
                model=model,
                token_type=token_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_llm_tokens", e)

    def record_llm_tool_call(
        self,
        *,
        feature: str,
        tool_name: str,
        outcome: str,
    ) -> None:
        try:
            llm_tool_calls_counter.labels(
                feature=feature,
                tool_name=tool_name,
                outcome=outcome,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_llm_tool_call", e)

    def observe_llm_iterations(
        self,
        *,
        feature: str,
        outcome: str,
        iterations: int,
    ) -> None:
        try:
            llm_iterations_histogram.labels(
                feature=feature,
                outcome=outcome,
            ).observe(iterations)
        except Exception as e:
            self._handle_metric_error("observe_llm_iterations", e)

    def record_llm_backup_used(
        self,
        *,
        feature: str,
        primary_provider: str,
        primary_model: str,
        backup_provider: str,
        backup_model: str,
    ) -> None:
        try:
            llm_backup_used_counter.labels(
                feature=feature,
                primary_provider=primary_provider,
                primary_model=primary_model,
                backup_provider=backup_provider,
                backup_model=backup_model,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_llm_backup_used", e)


prometheus_metrics = PrometheusMetrics()


async def metrics_endpoint(_request: Request) -> Response:
    if not settings.METRICS.ENABLED:
        return Response("Metrics are disabled", status_code=404)
    try:
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        return Response("Failed to generate metrics", status_code=500)
