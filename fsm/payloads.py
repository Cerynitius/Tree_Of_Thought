"""Self-contained payload extraction and coercion helpers for backend responses."""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any, Optional

from pydantic import BaseModel

from .errors import ChatBackendResponseError
from .utils import _model_field_names


def _serialize_raw_response_for_repair(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response
    try:
        return json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(raw_response)


def _truncate_compact_text(value: Any, max_chars: int) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            candidates.append(stripped[first_newline + 1:last_fence].strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    candidates.extend(_extract_balanced_json_object_candidates(stripped))

    valid_payloads: list[tuple[int, int, dict[str, Any]]] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            valid_payloads.append((len(loaded), len(candidate), dict(loaded)))
    if valid_payloads:
        return max(valid_payloads, key=lambda item: (item[0], item[1]))[2]
    raise ValueError("Chat backend response did not contain a valid JSON object payload.")


def _extract_balanced_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: Optional[int] = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = text[start_index:index + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start_index = None

    return candidates


def _content_to_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        preferred_parts: list[str] = []
        fallback_parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                fallback_parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_parts: list[str] = []
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                item_parts.append(text.strip())
            nested_content = item.get("content")
            if isinstance(nested_content, str) and nested_content.strip():
                item_parts.append(nested_content.strip())
            elif isinstance(nested_content, list):
                nested_text = _content_to_text(nested_content)
                if isinstance(nested_text, str) and nested_text.strip():
                    item_parts.append(nested_text.strip())
            if not item_parts:
                continue

            item_type = str(item.get("type", "")).strip().lower()
            target_parts = preferred_parts if item_type and item_type not in {"reasoning", "analysis"} else fallback_parts
            target_parts.extend(item_parts)
        if preferred_parts:
            return "\n".join(preferred_parts)
        if fallback_parts:
            return "\n".join(fallback_parts)
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            text = _coerce_string_scalar(item)
            if text:
                items.append(text)
        return items
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    text = _coerce_string_scalar(value)
    return [text] if text else []


def _coerce_string_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in (
            "action",
            "step",
            "first_step",
            "current_step",
            "selected_task",
            "step_focus",
            "thought_step",
            "objective",
            "title",
            "name",
            "summary",
            "description",
            "reason",
            "guidance",
            "text",
            "content",
        ):
            if key not in value:
                continue
            text = _coerce_string_scalar(value.get(key))
            if text:
                return text

        parts: list[str] = []
        for nested_value in value.values():
            text = _coerce_string_scalar(nested_value)
            if text and text not in parts:
                parts.append(text)
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            text = _coerce_string_scalar(item)
            if text and text not in parts:
                parts.append(text)
        return "; ".join(parts)
    return str(value).strip()


def _dedupe_string_sequence(values: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _coerce_string_scalar(value)
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _dedupe_structured_reasoning_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        signature = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _coerce_structured_reasoning_item(value: Any, *, default_status: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        label = ""
        for key in (
            "label",
            "action",
            "selected_task",
            "step_focus",
            "step",
            "title",
            "name",
            "summary",
            "description",
            "text",
            "content",
            "current_step",
            "first_step",
        ):
            label = _coerce_string_scalar(value.get(key))
            if label:
                break
        if label:
            normalized["label"] = label

        for target_key, source_keys in (
            ("route_family", ("route_family", "route", "family", "perspective")),
            ("step_type", ("step_type", "task_type", "phase", "mode")),
            ("target_quantity", ("target_quantity", "target", "quantity", "focus")),
            (
                "correction_mode",
                (
                    "correction_mode",
                    "correction_style",
                    "correction_family",
                    "closure_strategy",
                    "error_model",
                    "parameterization",
                ),
            ),
            (
                "correction_target",
                (
                    "correction_target",
                    "correction_quantity",
                    "correction_term",
                    "correction_focus",
                    "target_correction",
                    "deferred_correction",
                ),
            ),
            ("guidance", ("guidance", "current_step_guidance", "description", "rationale")),
            ("status", ("status", "selection", "role")),
            ("slot", ("slot", "reasoning_slot", "distribution_slot")),
        ):
            for source_key in source_keys:
                text = _coerce_string_scalar(value.get(source_key))
                if text:
                    normalized[target_key] = text
                    break

        for target_key, source_keys in (
            ("governing_models", ("governing_models", "used_models", "models")),
            ("assumptions", ("assumptions",)),
            ("deferred_terms", ("deferred_terms", "deferred_tasks", "pending_terms", "corrections")),
            ("completion_signals", ("completion_signals",)),
        ):
            aggregated: list[str] = []
            for source_key in source_keys:
                aggregated.extend(_coerce_string_list(value.get(source_key)))
            if aggregated:
                normalized[target_key] = list(dict.fromkeys(aggregated))

        priority = value.get("priority")
        if isinstance(priority, (int, float)):
            normalized["priority"] = priority

        if default_status and not normalized.get("status"):
            normalized["status"] = default_status

        if not normalized.get("label"):
            fallback_label = _coerce_string_scalar(value)
            if fallback_label:
                normalized["label"] = fallback_label
        return normalized

    text = _coerce_string_scalar(value)
    if not text:
        return {}
    normalized = {"label": text}
    if default_status:
        normalized["status"] = default_status
    return normalized


def _coerce_structured_reasoning_list(value: Any, *, default_status: str = "") -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [
            _coerce_structured_reasoning_item(item, default_status=default_status)
            for item in value
        ]
    else:
        items = [_coerce_structured_reasoning_item(value, default_status=default_status)]
    return _dedupe_structured_reasoning_items([item for item in items if item])


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped: None} if stripped else {}
    if not isinstance(value, list):
        return {}

    normalized: dict[str, Any] = {}
    for item in value:
        if isinstance(item, dict):
            normalized.update({str(key): nested_value for key, nested_value in item.items()})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized[str(item[0])] = item[1]
            continue
        text = str(item).strip()
        if text:
            normalized[text] = None
    return normalized


def _coerce_optional_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _coerce_optional_number_or_none(value: Any) -> Any:
    coerced = _coerce_optional_number(value)
    if isinstance(coerced, str):
        return None
    return coerced


def _coerce_evaluation_field_aliases(
    normalized: dict[str, Any],
    field_names: set[str],
) -> dict[str, Any]:
    expected_fields = {
        "domain_consistency",
        "variable_grounding",
        "contextual_relevance",
        "simplicity_hint",
        "score",
        "reason",
        "hard_rule_violations",
    }
    if not {"domain_consistency", "contextual_relevance"}.issubset(field_names):
        return normalized

    for alias in list(normalized):
        normalized_alias = str(alias).strip().lower()
        if normalized_alias in expected_fields:
            continue
        match = get_close_matches(normalized_alias, sorted(expected_fields), n=1, cutoff=0.72)
        if not match:
            continue
        normalized.setdefault(match[0], normalized.get(alias))
        normalized.pop(alias, None)
    return normalized


def _normalize_chat_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        wrapper_keys = {"choices", "content", "data", "message", "output", "response", "text"}
        if not (set(response) & wrapper_keys):
            return dict(response)
        for key in ("output", "response", "content", "text", "data"):
            value = response.get(key)
            if isinstance(value, dict):
                return dict(value)
            content_text = _content_to_text(value)
            if content_text is not None:
                return _extract_json_payload(content_text)
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content", message.get("text"))
            if isinstance(content, dict):
                return dict(content)
            content_text = _content_to_text(content)
            if content_text is not None:
                return _extract_json_payload(content_text)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content_text = _content_to_text(message.get("content", message.get("text")))
                    if content_text is not None:
                        return _extract_json_payload(content_text)
                for key in ("output", "content", "text"):
                    value = first.get(key)
                    if isinstance(value, dict):
                        return dict(value)
                    content_text = _content_to_text(value)
                    if content_text is not None:
                        return _extract_json_payload(content_text)
    if isinstance(response, str):
        return _extract_json_payload(response)
    raise TypeError("Chat backend response must be a dictionary or string payload.")


__all__ = [
    "_coerce_evaluation_field_aliases",
    "_coerce_mapping",
    "_coerce_optional_number",
    "_coerce_optional_number_or_none",
    "_coerce_string_list",
    "_coerce_string_scalar",
    "_coerce_structured_reasoning_item",
    "_coerce_structured_reasoning_list",
    "_content_to_text",
    "_dedupe_string_sequence",
    "_dedupe_structured_reasoning_items",
    "_extract_balanced_json_object_candidates",
    "_extract_json_payload",
    "_normalize_chat_payload",
    "_serialize_raw_response_for_repair",
    "_truncate_compact_text",
]
