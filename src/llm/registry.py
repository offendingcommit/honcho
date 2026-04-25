"""Single owner of provider runtime objects: clients, backends, history adapters.

Consolidates wiring that previously lived in both `src/llm/__init__.py` and
`src/utils/clients.py`. Everything that touches provider SDKs at runtime
(default client construction, override client caching, backend selection,
history adapter selection) lives here now.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, assert_never

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from src.config import ModelConfig, ModelTransport, settings
from src.exceptions import ValidationException

from .backend import ProviderBackend
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.openai import OpenAIBackend
from .credentials import default_transport_api_key
from .history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    HistoryAdapter,
    OpenAIHistoryAdapter,
)
from .types import ProviderClient


def _cf_gateway_default_headers() -> dict[str, str] | None:
    """Build the cf-aig-authorization header dict, or None if disabled.

    Gateway-layer auth (cfut_ token) rides alongside the upstream provider
    key in Authorization. When set, every OpenAI-compatible client in this
    module sends it so CF AI Gateway can authenticate requests before
    forwarding them to the upstream provider.
    """
    token = settings.LLM.CF_GATEWAY_AUTH_TOKEN
    if not token:
        return None
    return {"cf-aig-authorization": f"Bearer {token}"}


@lru_cache(maxsize=1)
def get_anthropic_client() -> AsyncAnthropic:
    """Default Anthropic client built from settings.LLM.ANTHROPIC_API_KEY."""
    return AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """Default OpenAI client built from settings.LLM.OPENAI_API_KEY.

    Honors ``LLM_OPENAI_BASE_URL`` to route default OpenAI traffic through a
    proxy/gateway, and ``LLM_CF_GATEWAY_AUTH_TOKEN`` to inject the
    cf-aig-authorization header on every request.
    """
    kwargs: dict[str, Any] = {"api_key": settings.LLM.OPENAI_API_KEY}
    if settings.LLM.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.LLM.OPENAI_BASE_URL
    headers = _cf_gateway_default_headers()
    if headers is not None:
        kwargs["default_headers"] = headers
    return AsyncOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Default Gemini client built from settings.LLM.GEMINI_API_KEY."""
    return genai.Client(api_key=settings.LLM.GEMINI_API_KEY)


# Bounded cache — in practice the (base_url, api_key) key space is small
# and process-scoped, but maxsize=128 keeps worst-case memory predictable.
@lru_cache(maxsize=128)
def get_openai_override_client(
    base_url: str | None, api_key: str | None
) -> AsyncOpenAI:
    """OpenAI client for a specific (base_url, api_key) pair. Cached by key."""
    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    headers = _cf_gateway_default_headers()
    if headers is not None:
        kwargs["default_headers"] = headers
    return AsyncOpenAI(**kwargs)


@lru_cache(maxsize=128)
def get_anthropic_override_client(
    base_url: str | None,
    api_key: str | None,
) -> AsyncAnthropic:
    """Anthropic client for a specific (base_url, api_key) pair. Cached by key."""
    return AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=600.0)


@lru_cache(maxsize=128)
def get_gemini_override_client(
    base_url: str | None, api_key: str | None
) -> genai.Client:
    """Gemini client for a specific (base_url, api_key) pair. Cached by key."""
    http_options = genai_types.HttpOptions(base_url=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_options)


# Module-level default-client registry, populated at import time. Tests patch
# this dict via `patch.dict(CLIENTS, {...})` to inject mock provider clients.
CLIENTS: dict[ModelTransport, ProviderClient] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    CLIENTS["anthropic"] = AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )

if settings.LLM.OPENAI_API_KEY:
    _openai_default_kwargs: dict[str, Any] = {"api_key": settings.LLM.OPENAI_API_KEY}
    if settings.LLM.OPENAI_BASE_URL:
        _openai_default_kwargs["base_url"] = settings.LLM.OPENAI_BASE_URL
    _cf_headers = _cf_gateway_default_headers()
    if _cf_headers is not None:
        _openai_default_kwargs["default_headers"] = _cf_headers
    CLIENTS["openai"] = AsyncOpenAI(**_openai_default_kwargs)

if settings.LLM.GEMINI_API_KEY:
    CLIENTS["gemini"] = genai.client.Client(
        api_key=settings.LLM.GEMINI_API_KEY,
    )


def client_for_model_config(
    provider: ModelTransport,
    model_config: ModelConfig,
) -> ProviderClient:
    """Resolve the provider client for a ModelConfig.

    Fast path: no overrides → reuse the module-level default client from
    CLIENTS (the test-mockable seam). Otherwise route through the cached
    override factories.
    """
    if model_config.api_key is None and model_config.base_url is None:
        existing_client = CLIENTS.get(provider)
        if existing_client is not None:
            return existing_client

    api_key = model_config.api_key or default_transport_api_key(provider)
    base_url = model_config.base_url
    if not api_key:
        raise ValidationException(f"Missing API key for {provider} model config")

    if provider == "anthropic":
        return get_anthropic_override_client(base_url, api_key)
    if provider == "openai":
        return get_openai_override_client(base_url, api_key)
    if provider == "gemini":
        return get_gemini_override_client(base_url, api_key)
    assert_never(provider)


def backend_for_provider(
    provider: ModelTransport,
    client: ProviderClient,
) -> ProviderBackend:
    """Wrap a raw provider SDK client in the matching ProviderBackend adapter."""
    if provider == "anthropic":
        return AnthropicBackend(client)
    if provider == "openai":
        return OpenAIBackend(client)
    if provider == "gemini":
        return GeminiBackend(client)
    assert_never(provider)


def history_adapter_for_provider(provider: ModelTransport) -> HistoryAdapter:
    """Provider-appropriate HistoryAdapter for assistant/tool message formatting."""
    if provider == "anthropic":
        return AnthropicHistoryAdapter()
    if provider == "gemini":
        return GeminiHistoryAdapter()
    return OpenAIHistoryAdapter()


def get_backend(config: ModelConfig) -> ProviderBackend:
    """High-level one-shot backend factory: ModelConfig → ProviderBackend.

    Delegates client resolution to ``client_for_model_config``, which owns
    the CLIENTS fast-path and the missing-API-key validation. Both the
    production path (via ``honcho_llm_call_inner``) and the live-test path
    (via this function) now construct clients through the same helper, so
    validation behavior stays consistent.
    """
    client = client_for_model_config(config.transport, config)
    return backend_for_provider(config.transport, client)


__all__ = [
    "CLIENTS",
    "backend_for_provider",
    "client_for_model_config",
    "get_anthropic_client",
    "get_anthropic_override_client",
    "get_backend",
    "get_gemini_client",
    "get_gemini_override_client",
    "get_openai_client",
    "get_openai_override_client",
    "history_adapter_for_provider",
]
