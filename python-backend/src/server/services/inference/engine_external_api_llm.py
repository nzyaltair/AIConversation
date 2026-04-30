"""
External API LLM inference engine.

Calls any OpenAI-compatible API (DeepSeek, OpenAI, local proxy, etc.) via the
openai SDK. Configuration (base_url, api_key, model, reasoning_effort) is
passed per-request through **kwargs — no env vars are read by the engine itself.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from typing import Generator
from urllib.parse import urlparse

from server.services.inference import register_engine
from server.services.inference.base import LlmEngine, ChatResult

logger = logging.getLogger(__name__)

# Whitelist of allowed API hosts (prevents SSRF to internal networks)
_ALLOWED_HOSTS = frozenset({"api.deepseek.com", "api.openai.com"})

# Generic error message shown to users (avoids leaking API provider details)
_GENERIC_API_ERROR_MSG = (
    "External API call failed. Please check your API key, "
    "base URL, and model ID in Settings."
)


def _validate_base_url(raw: str) -> str:
    """Validate and sanitize the external API base URL.

    Blocks private IPs, loopback, and link-local addresses to prevent SSRF.
    Allows any public hostname since users may use custom proxies.
    """
    url = raw.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("base_url must start with http:// or https://")

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        raise ValueError("base_url must include a valid hostname")

    # Block bare IP addresses that are private/loopback/link-local
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise ValueError("Localhost addresses are not allowed in base_url")

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not a bare IP — hostname is fine (DNS-resolved addresses are
        # acceptable since users are intentionally configuring their own API).
        addr = None

    if addr is not None and (addr.is_private or addr.is_loopback or addr.is_link_local):
        raise ValueError(f"Private/internal addresses are not allowed: {hostname}")

    return url


@register_engine("llm", "external")
class ExternalApiLlmEngine(LlmEngine):
    """LLM engine that delegates to any OpenAI-compatible external API.

    The engine does NOT read environment variables. All connection parameters
    (base_url, api_key, model) are supplied via ``generate(**kwargs)`` so that
    users can configure them per-session from the frontend.
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._openai_clients: dict[str, object] = {}
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------
    async def load(self) -> None:
        """Verify the openai SDK is importable. No network calls are made."""
        try:
            import openai  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "openai SDK is not installed. Run: pip install openai"
            )
        self._loaded = True
        logger.info("External API LLM engine loaded (variant=%s)", self.variant)

    async def unload(self) -> None:
        for client in self._openai_clients.values():
            try:
                client.close()
            except Exception:
                pass
        self._openai_clients.clear()
        self._loaded = False

    def _get_client(self, base_url: str, api_key: str):
        """Get or create a cached OpenAI client keyed by (base_url, api_key).

        Thread-safe: protected by a lock since generate() runs in executor threads.
        """
        cache_key = f"{base_url}|{api_key}"
        with self._client_lock:
            if cache_key not in self._openai_clients:
                from openai import OpenAI
                self._openai_clients[cache_key] = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=30.0,
                    max_retries=1,
                )
            return self._openai_clients[cache_key]

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: object,
    ):
        """Run chat completion against the configured external API.

        Expected kwargs (from frontend ``api_config``):
        - ``base_url``: API endpoint URL (e.g. "https://api.deepseek.com")
        - ``api_key``: API key / bearer token
        - ``model``: model identifier (e.g. "deepseek-v4-flash")
        - ``reasoning_effort`` (optional): "low" | "medium" | "high" — only
          passed when ``enable_thinking`` is True
        - ``enable_thinking`` (optional): bool — gates whether
          ``reasoning_effort`` is forwarded to the API
        """
        self._ensure_loaded()

        base_url = str(kwargs.get("base_url", "https://api.deepseek.com"))
        api_key = str(kwargs.get("api_key", ""))
        model = str(kwargs.get("model", self.variant))
        reasoning_effort = kwargs.get("reasoning_effort", None)
        enable_thinking = kwargs.get("enable_thinking", None)

        if not api_key:
            raise RuntimeError(
                "[API_KEY_MISSING] API key is required. "
                "Please configure it in the Settings panel."
            )

        # Validate base_url to prevent SSRF
        base_url = _validate_base_url(base_url)

        client = self._get_client(base_url, api_key)

        # Convert internal message format to API format
        api_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        extra_params: dict = {}
        # Only pass reasoning_effort when thinking is enabled (per DeepSeek API spec)
        if enable_thinking and reasoning_effort is not None and str(reasoning_effort) not in ("", "none"):
            extra_params["reasoning_effort"] = str(reasoning_effort)

        if stream:
            return self._stream_generate(
                client, model, api_messages, max_tokens,
                temperature, top_p, extra_params,
            )

        return self._sync_generate(
            client, model, api_messages, max_tokens,
            temperature, top_p, extra_params,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("ExternalApiLlmEngine is not loaded. Call load() first.")

    def _sync_generate(
        self,
        client,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_params: dict,
    ) -> ChatResult:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **extra_params,
            )
        except Exception:
            logger.exception("External API sync call failed")
            return ChatResult(
                id=f"error-{int(time.time())}",
                created=int(time.time()),
                model=model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[API Error] {_GENERIC_API_ERROR_MSG}",
                    },
                    "finish_reason": "error",
                }],
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        # DeepSeek-R1 returns reasoning_content in the top-level message
        thinking = getattr(choice.message, "reasoning_content", None) or ""

        return ChatResult(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "thinking": thinking,
                },
                "finish_reason": choice.finish_reason or "stop",
            }],
        )

    def _stream_generate(
        self,
        client,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_params: dict,
    ) -> Generator[dict, None, None]:
        created = int(time.time())
        chunk_id = f"chatcmpl-{created}"

        try:
            stream_response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **extra_params,
            )

            for chunk in stream_response:
                choices = chunk.choices
                if not choices:
                    continue

                delta = choices[0].delta
                content_delta = delta.content or ""
                # DeepSeek-R1 streams reasoning_content in the delta
                thinking_delta = getattr(delta, "reasoning_content", None) or ""

                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content_delta,
                            "thinking": thinking_delta,
                        },
                        "finish_reason": choices[0].finish_reason,
                    }],
                }

        except Exception:
            logger.exception("External API stream call failed")
            yield {"error": _GENERIC_API_ERROR_MSG}
            return

        # Final done chunk (compatible with existing _stream_completion)
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
