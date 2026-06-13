"""Error taxonomy for the local chat backend."""

from __future__ import annotations


TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ChatBackendError(RuntimeError):
    """Base class for local chat backend failures."""


class ChatBackendTransportError(ChatBackendError):
    """Transport-level or HTTP failures when calling the local chat backend."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ChatBackendResponseError(ChatBackendError):
    """Raised when the local chat backend returns an unusable payload."""


__all__ = [
    "ChatBackendError",
    "ChatBackendResponseError",
    "ChatBackendTransportError",
    "TRANSIENT_HTTP_STATUS_CODES",
]
