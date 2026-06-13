from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from benchmarks import direct_9b_benchmark_payload


DEFAULT_API_URL = "http://localhost:1234/api/v1/chat"
DEFAULT_MODEL = "qwen3.5-9b-mlx"


@dataclass(frozen=True)
class ExpectedValue:
    label: str
    value: float
    abs_tol: float
    rel_tol: float = 0.03


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    topic: str
    prompt: str
    expected_values: tuple[ExpectedValue, ...]
    reference_answer: str
    freeform_output: bool = False
    tool_invocations: tuple[dict[str, Any], ...] = ()


def _expected_value_from_payload(payload: dict[str, Any]) -> ExpectedValue:
    return ExpectedValue(
        label=str(payload["label"]),
        value=float(payload["value"]),
        abs_tol=float(payload["abs_tol"]),
        rel_tol=float(payload.get("rel_tol", 0.03)),
    )


def _case_from_payload(payload: dict[str, Any]) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=str(payload["case_id"]),
        topic=str(payload["topic"]),
        prompt=str(payload["prompt"]),
        expected_values=tuple(_expected_value_from_payload(item) for item in payload.get("expected_values", [])),
        reference_answer=str(payload["reference_answer"]),
        freeform_output=bool(payload.get("freeform_output", False)),
        tool_invocations=tuple(
            dict(item)
            for item in payload.get("tool_invocations", [])
            if isinstance(item, dict)
        ),
    )


_BENCHMARK_PAYLOAD = direct_9b_benchmark_payload({})
SYSTEM_PROMPT = str(_BENCHMARK_PAYLOAD["system_prompt"])
FREEFORM_SYSTEM_PROMPT = str(_BENCHMARK_PAYLOAD["freeform_system_prompt"])
SUITES: dict[str, tuple[BenchmarkCase, ...]] = {
    str(suite_name): tuple(_case_from_payload(item) for item in suite_cases)
    for suite_name, suite_cases in dict(_BENCHMARK_PAYLOAD["suites"]).items()
}
DEFAULT_SUITE = str(_BENCHMARK_PAYLOAD.get("default_suite") or next(iter(SUITES)))


def build_user_prompt(case: BenchmarkCase, *, freeform_output: bool = False) -> str:
    if freeform_output or case.freeform_output:
        return (
            f"Problem id: {case.case_id}\n"
            f"Topic: {case.topic}\n"
            f"Problem: {case.prompt}\n\n"
            "Solve this as a normal free-response problem. Do not use JSON or any fixed schema. "
            "End with a clearly labeled final answer."
        )
    return (
        f"Problem id: {case.case_id}\n"
        f"Topic: {case.topic}\n"
        f"Problem: {case.prompt}\n\n"
        "Solve the problem. Return only JSON, for example: "
        "{\"final_answer\": \"...\", \"concise_solution\": \"...\"}."
    )


def call_local_model(
    api_url: str,
    model: str,
    case: BenchmarkCase,
    timeout: float,
    *,
    freeform_output: bool = False,
) -> tuple[dict[str, Any], float]:
    use_freeform = freeform_output or case.freeform_output
    payload = {
        "model": model,
        "system_prompt": FREEFORM_SYSTEM_PROMPT if use_freeform else SYSTEM_PROMPT,
        "input": build_user_prompt(case, freeform_output=freeform_output),
    }
    started_at = time.perf_counter()
    response = requests.post(
        api_url,
        json=payload,
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    latency = time.perf_counter() - started_at
    response.raise_for_status()
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    return normalize_chat_response(body), latency


def normalize_chat_response(body: Any) -> dict[str, Any]:
    text = extract_response_text(body)
    parsed = extract_json_object(text)
    if parsed is None:
        return {"final_answer": text.strip(), "concise_solution": "", "raw_response": body}
    parsed.setdefault("raw_response", body)
    return parsed


def extract_response_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        preferred_parts: list[str] = []
        fallback_parts: list[str] = []
        for item in body:
            item_text = ""
            item_type = ""
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                item_text = extract_response_text(item.get("content", item.get("text", ""))).strip()
            elif isinstance(item, str):
                item_text = item.strip()
            if not item_text:
                continue
            if item_type and item_type not in {"reasoning", "analysis"}:
                preferred_parts.append(item_text)
            else:
                fallback_parts.append(item_text)
        if preferred_parts:
            return "\n".join(preferred_parts)
        if fallback_parts:
            return "\n".join(fallback_parts)
    if isinstance(body, dict):
        for key in ("output", "response", "content", "text", "data"):
            value = body.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                nested = extract_response_text(value)
                if nested:
                    return nested
        message = body.get("message")
        if isinstance(message, dict):
            nested = extract_response_text(message.get("content", message.get("text", "")))
            if nested:
                return nested
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    nested = extract_response_text(message.get("content", message.get("text", "")))
                    if nested:
                        return nested
                for key in ("output", "content", "text"):
                    nested = extract_response_text(first.get(key, ""))
                    if nested:
                        return nested
        try:
            return json.dumps(body, ensure_ascii=False)
        except TypeError:
            return str(body)
    return str(body)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            candidates.append(stripped[first_newline + 1:last_fence].strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def score_case(case: BenchmarkCase, model_payload: dict[str, Any]) -> dict[str, Any]:
    answer_text = str(model_payload.get("final_answer") or model_payload.get("answer") or "").strip()
    if not answer_text:
        answer_text = extract_response_text(model_payload.get("raw_response", model_payload))
    scoring_text = extract_scoring_answer_text(answer_text)
    numbers = extract_numbers(scoring_text)
    matches: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    for expected in case.expected_values:
        best_match: tuple[int, float] | None = None
        for index, candidate in enumerate(numbers):
            if index in used_indices:
                continue
            allowed_error = max(expected.abs_tol, abs(expected.value) * expected.rel_tol)
            error = abs(candidate - expected.value)
            if error <= allowed_error and (best_match is None or error < best_match[1]):
                best_match = (index, error)
        if best_match is None:
            matches.append(
                {
                    "label": expected.label,
                    "expected": expected.value,
                    "matched": None,
                    "ok": False,
                }
            )
            continue
        used_indices.add(best_match[0])
        matches.append(
            {
                "label": expected.label,
                "expected": expected.value,
                "matched": numbers[best_match[0]],
                "error": best_match[1],
                "ok": True,
            }
        )

    ok = all(match["ok"] for match in matches)
    return {
        "case_id": case.case_id,
        "topic": case.topic,
        "ok": ok,
        "matches": matches,
        "model_final_answer": scoring_text,
        "reference_answer": case.reference_answer,
    }


def extract_scoring_answer_text(text: str) -> str:
    """Isolate the explicit final-answer region before number matching.

    Long responses are never scanned wholesale: matching any number in a wide
    tail lets intermediate values (or tree metadata) collide with expected
    values and inflate accuracy. Prefer text after an explicit answer marker;
    otherwise fall back to the last few lines only.
    """

    stripped = str(text).strip()
    if len(stripped) <= 1000:
        return stripped
    markers = list(
        re.finditer(
            r"(?i)(?:final\s+answer|final\s+result)\b\s*[:：=]?|(?:\banswer\b|\bresult\b)\s*[:：=]",
            stripped,
        )
    )
    if markers:
        tail = stripped[markers[-1].start() :].strip()
        if tail:
            return tail[:600]
    final_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if final_lines:
        return "\n".join(final_lines[-3:])[:600]
    return stripped[-300:].strip()


def extract_numbers(text: str) -> list[float]:
    normalized = text.replace("−", "-").replace("×", "*")
    number_pattern = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    values: list[float] = []
    for match in number_pattern.finditer(normalized):
        raw = match.group(0)
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = selected_cases(args)
    rows: list[dict[str, Any]] = []
    correct = 0
    started_at = time.perf_counter()

    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.case_id} ...", flush=True)
        try:
            model_payload, latency = call_local_model(
                args.api_url,
                args.model,
                case,
                args.timeout,
                freeform_output=args.freeform_output,
            )
            scored = score_case(case, model_payload)
            scored["latency_seconds"] = round(latency, 3)
            scored["error"] = ""
        except Exception as exc:  # noqa: BLE001 - benchmark should continue after one failed request.
            scored = {
                "case_id": case.case_id,
                "topic": case.topic,
                "ok": False,
                "matches": [],
                "model_final_answer": "",
                "reference_answer": case.reference_answer,
                "latency_seconds": None,
                "error": str(exc),
            }
        rows.append(scored)
        if scored["ok"]:
            correct += 1
        status = "OK" if scored["ok"] else "FAIL"
        latency_text = "n/a" if scored["latency_seconds"] is None else f"{scored['latency_seconds']:.3f}s"
        print(f"    {status} latency={latency_text}")
        if args.verbose or not scored["ok"]:
            print(f"    model: {scored['model_final_answer']}")
            print(f"    ref:   {scored['reference_answer']}")
            if scored.get("error"):
                print(f"    error: {scored['error']}")

    elapsed = time.perf_counter() - started_at
    accuracy = correct / len(cases) if cases else 0.0
    return {
        "model": args.model,
        "api_url": args.api_url,
        "case_count": len(cases),
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 3),
        "suite": args.suite,
        "freeform_output": bool(args.freeform_output),
        "rows": rows,
    }


def selected_cases(args: argparse.Namespace) -> tuple[BenchmarkCase, ...]:
    cases = SUITES[args.suite]
    if args.case_id:
        selected = tuple(case for case in cases if case.case_id == args.case_id)
        if not selected:
            raise ValueError(f"Unknown case id for suite {args.suite}: {args.case_id}")
        return selected
    if args.limit:
        return cases[: args.limit]
    return cases


def regrade_saved_summary(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.regrade_input, "r", encoding="utf-8") as input_file:
        saved = json.load(input_file)

    cases_by_id = {case.case_id: case for suite_cases in SUITES.values() for case in suite_cases}
    rows: list[dict[str, Any]] = []
    correct = 0

    for saved_row in saved.get("rows", []):
        case_id = str(saved_row.get("case_id", ""))
        case = cases_by_id.get(case_id)
        if case is None:
            rows.append(
                {
                    "case_id": case_id,
                    "topic": str(saved_row.get("topic", "")),
                    "ok": False,
                    "matches": [],
                    "model_final_answer": str(saved_row.get("model_final_answer", "")),
                    "reference_answer": str(saved_row.get("reference_answer", "")),
                    "latency_seconds": saved_row.get("latency_seconds"),
                    "error": "unknown case_id in saved results",
                }
            )
            continue

        raw_answer = saved_row.get("model_final_answer", "")
        try:
            raw_body: Any = json.loads(str(raw_answer))
        except json.JSONDecodeError:
            raw_body = raw_answer
        model_payload = normalize_chat_response(raw_body)
        scored = score_case(case, model_payload)
        scored["latency_seconds"] = saved_row.get("latency_seconds")
        scored["error"] = str(saved_row.get("error", ""))
        rows.append(scored)
        if scored["ok"]:
            correct += 1

    case_count = len(rows)
    accuracy = correct / case_count if case_count else 0.0
    return {
        "model": str(saved.get("model", args.model)),
        "api_url": str(saved.get("api_url", args.api_url)),
        "case_count": case_count,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": saved.get("elapsed_seconds"),
        "suite": str(saved.get("suite", args.suite)),
        "freeform_output": bool(saved.get("freeform_output", args.freeform_output)),
        "rows": rows,
        "regraded_from": args.regrade_input,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct local-model benchmark against the local chat endpoint."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", choices=sorted(SUITES), default=DEFAULT_SUITE)
    parser.add_argument("--case-id", default="", help="Run one case from the selected suite by id.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="Number of cases to run; 0 runs all cases.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--regrade-input", default="", help="Regrade a saved benchmark JSON without calling the model.")
    parser.add_argument("--freeform-output", action="store_true", help="Ask for natural prose instead of JSON for every selected case.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict-exit", action="store_true", help="Exit non-zero if any answer is graded incorrect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = regrade_saved_summary(args) if args.regrade_input else run_benchmark(args)
    print("\nDIRECT_9B_BENCHMARK_SUMMARY")
    print(f"model={summary['model']}")
    print(f"api_url={summary['api_url']}")
    print(f"suite={summary.get('suite', args.suite)}")
    print(f"correct={summary['correct']}/{summary['case_count']}")
    print(f"accuracy={summary['accuracy']:.1%}")
    print(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, ensure_ascii=False, indent=2)
        print(f"wrote={args.output}")
    if args.strict_exit and summary["correct"] != summary["case_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
