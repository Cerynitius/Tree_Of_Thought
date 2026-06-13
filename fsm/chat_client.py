"""HTTP client and requester adapters for the local chat backend."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

from .errors import (
    ChatBackendError,
    ChatBackendResponseError,
    ChatBackendTransportError,
    TRANSIENT_HTTP_STATUS_CODES,
)


DEFAULT_CHAT_API_URL = "http://localhost:1234/api/v1/chat"


ChatRequester = Callable[[str, dict[str, Any], float], Any]


class LocalChatAPIClient:
    """Thin client for the local chat API exposed at ``/api/v1/chat``."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_CHAT_API_URL,
        timeout: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        requester: Optional[ChatRequester] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative.")

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._requester = requester or self._default_requester

    def chat(self, *, model: str, system_prompt: str, input_text: str) -> Any:
        normalized_model = str(model).strip()
        if not normalized_model:
            raise ValueError("model must be a non-empty string.")

        normalized_input_text = str(input_text).strip()
        if not normalized_input_text:
            raise ValueError("input_text must be a non-empty string.")

        payload = {
            "model": normalized_model,
            "system_prompt": "" if system_prompt is None else str(system_prompt),
            "input": normalized_input_text,
        }
        for attempt_index in range(self.max_retries + 1):
            try:
                response = self._requester(self.base_url, payload, self.timeout)
            except Exception as exc:
                normalized_exc = self._normalize_request_exception(exc)
                if attempt_index >= self.max_retries or not self._should_retry(normalized_exc):
                    if normalized_exc is exc:
                        raise
                    raise normalized_exc from exc
                self._sleep_before_retry(attempt_index)
                continue

            if response is None:
                raise ChatBackendResponseError("Chat backend returned no response body.")
            if isinstance(response, (bytes, bytearray)):
                response = response.decode("utf-8", errors="replace")
            if isinstance(response, str) and not response.strip():
                raise ChatBackendResponseError("Chat backend returned an empty response body.")
            return response

        raise ChatBackendTransportError("Chat backend request exhausted all retry attempts.")

    def _normalize_request_exception(self, exc: Exception) -> Exception:
        if isinstance(exc, ChatBackendError):
            return exc
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return ChatBackendTransportError(
                f"Timed out after {self.timeout:.1f}s waiting for chat backend at {self.base_url}"
            )
        if isinstance(exc, HTTPError):
            details = exc.read().decode("utf-8", errors="replace")
            return ChatBackendTransportError(
                f"Chat backend returned HTTP {exc.code}: {details}",
                status_code=exc.code,
                response_body=details,
            )
        if isinstance(exc, URLError):
            return ChatBackendTransportError(
                f"Failed to reach chat backend at {self.base_url}: {exc.reason}"
            )
        return exc

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, ChatBackendTransportError):
            return exc.status_code is None or exc.status_code in TRANSIENT_HTTP_STATUS_CODES
        return False

    def _sleep_before_retry(self, attempt_index: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(self.retry_backoff_seconds * (2**attempt_index))

    def _default_requester(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise ChatBackendTransportError(
                f"Timed out after {timeout:.1f}s waiting for chat backend at {url}"
            ) from exc
        except requests.RequestException as exc:
            raise ChatBackendTransportError(
                f"Failed to reach chat backend at {url}: {exc}"
            ) from exc

        raw = response.text
        if response.status_code >= 400:
            raise ChatBackendTransportError(
                f"Chat backend returned HTTP {response.status_code}: {raw}",
                status_code=response.status_code,
                response_body=raw,
            )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def make_openai_requester(api_key: str) -> "ChatRequester":
    """Return a requester that speaks the OpenAI messages format with Bearer auth.

    Converts the internal {system_prompt, input} payload to {messages:[...]}
    so any OpenAI-compatible API (OpenRouter, OpenAI, etc.) works out of the box.
    """
    def _requester(url: str, payload: dict[str, Any], timeout: float) -> Any:
        messages: list[dict[str, str]] = []
        sp = payload.get("system_prompt", "")
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": payload.get("input", "")})
        oai_payload = {
            "model": payload.get("model", ""),
            "messages": messages,
        }
        body = json.dumps(oai_payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        request = Request(url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    return _requester


__all__ = [
    "ChatRequester",
    "DEFAULT_CHAT_API_URL",
    "LocalChatAPIClient",
    "make_openai_requester",
]
