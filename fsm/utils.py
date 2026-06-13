"""Internal helpers shared across the ToT FSM modules."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from enum import Enum
from typing import Any

from pydantic import BaseModel

# Shared meta-task guidance for the strategy-scan phase; scheduler and backend
# must present the same instruction or route planning drifts between layers.
META_TASK_STRATEGY_SCAN_GUIDANCE = (
    "Analyze the next-step strategy space at planning level. "
    "Make one route-local planning claim only, keep all other routes deferred, and do not solve the final answer yet."
)


def _model_field_names(model_type: type[BaseModel]) -> set[str]:
    try:
        return set(model_type.model_fields)
    except AttributeError:
        return set(model_type.__fields__)


def _build_model(model_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    """Construct a schema-locked Pydantic model with v1/v2 compatibility."""

    unexpected = set(payload) - _model_field_names(model_type)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Unexpected fields for {model_type.__name__}: {names}")
    try:
        return model_type.model_validate(payload)
    except AttributeError:
        return model_type.parse_obj(payload)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model with v1/v2 compatibility."""

    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        try:
            value = value.model_dump()
        except AttributeError:
            value = value.dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _normalize_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_signature_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_hash(value: Any) -> str:
    normalized = _normalize_signature_value(value)
    encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _serialize_blob(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _deserialize_blob(blob: str) -> Any:
    return pickle.loads(base64.b64decode(blob.encode("ascii")))


__all__ = [
    "META_TASK_STRATEGY_SCAN_GUIDANCE",
    "_build_model",
    "_deserialize_blob",
    "_model_dump",
    "_model_field_names",
    "_normalize_signature_value",
    "_serialize_blob",
    "_stable_hash",
]