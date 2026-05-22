from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from direct_9b_benchmark import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    FREEFORM_SYSTEM_PROMPT,
    SUITES,
    BenchmarkCase,
    call_local_model,
    extract_json_object,
    extract_response_text,
    normalize_chat_response,
    score_case,
)
from fsm import ToTNode, ToTTreeScheduler, build_local_chat_adapter_bundle


@dataclass(frozen=True)
class ToTConfig:
    api_url: str
    model: str
    timeout: float
    expansions: int
    depth: int
    children: int
    frontier: int
    max_retries: int
    allow_live_fallback: bool
    prefer_local_fallback: bool
    synthesis_nodes: int
    tree_wall_timeout: float
    synthesis_wall_timeout: float
    tool_policy: str


@contextmanager
def wall_timeout(seconds: float, label: str):
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"{label} exceeded {seconds:.1f}s wall timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def iter_nodes(root: ToTNode | None) -> Iterable[ToTNode]:
    if root is None:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def summarize_node(node: ToTNode) -> str:
    known_vars = {
        key: value
        for key, value in node.known_vars.items()
        if key
        in {
            "route_family",
            "correction_mode",
            "correction_target",
            "active_step",
            "selected_for_frontier",
            "sibling_rank",
        }
    }
    return json.dumps(
        {
            "id": node.id,
            "parent_id": node.parent_id,
            "status": node.status.value,
            "result_state": node.result_state.value,
            "score": node.score,
            "thought_step": node.thought_step,
            "equations": node.equations,
            "used_models": node.used_models,
            "known_vars": known_vars,
            "reflection_count": len(node.reflection_history),
        },
        ensure_ascii=False,
    )


def select_branch_notes(root: ToTNode | None, limit: int) -> list[str]:
    nodes = sorted(
        iter_nodes(root),
        key=lambda node: (node.status.value != "ACTIVE", -node.score, node.id),
    )
    return [summarize_node(node) for node in nodes[: max(1, limit)]]


def fallback_payload_from_notes(notes: list[str]) -> dict[str, Any]:
    return {
        "final_answer": "Final answer candidates from tree notes:\n" + "\n".join(notes),
        "concise_solution": "Scored directly from tree notes after answer synthesis did not complete.",
    }


def case_tool_invocations(case: BenchmarkCase) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in getattr(case, "tool_invocations", ())
        if isinstance(item, dict) and str(item.get("skill_name", "")).strip()
    ]


def run_case_tool_invocations(
    case: BenchmarkCase,
    selected_indices: Iterable[int] | None = None,
) -> tuple[dict[str, Any], list[str], float, str]:
    invocations = [
        (index, dict(item))
        for index, item in enumerate(case_tool_invocations(case), 1)
    ]
    if selected_indices is not None:
        selected = {int(index) for index in selected_indices}
        invocations = [(index, item) for index, item in invocations if index in selected]
    if not invocations:
        return {}, [], 0.0, ""

    try:
        from skills import invoke_skill
    except Exception as exc:  # noqa: BLE001
        return {}, [], 0.0, f"tool_error: {exc}"

    started_at = time.perf_counter()
    notes: list[str] = []
    answers: list[str] = []
    for index, invocation in invocations:
        skill_name = str(invocation.get("skill_name", "")).strip()
        payload = invocation.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        try:
            result = invoke_skill(skill_name, payload)
        except Exception as exc:  # noqa: BLE001
            return {}, notes, time.perf_counter() - started_at, f"tool_error: {skill_name}: {exc}"
        if isinstance(result, dict):
            answer = str(result.get("final_answer", "")).strip()
            if answer:
                answers.append(answer)
            notes.append(
                json.dumps(
                    {
                        "tool_index": index,
                        "skill_name": skill_name,
                        "final_answer": answer,
                        "raw_value": str(result.get("raw_value", "")),
                    },
                    ensure_ascii=False,
                )
            )
    elapsed = time.perf_counter() - started_at
    if not answers:
        return {}, notes, elapsed, "tool_error: no final_answer returned"
    return {
        "final_answer": ", ".join(answers),
        "concise_solution": "Tool-assisted result from imported benchmark skill invocations.",
    }, notes, elapsed, ""


def request_agent_tool_choice(case: BenchmarkCase, config: ToTConfig) -> tuple[list[int], float, str, str]:
    invocations = case_tool_invocations(case)
    if not invocations:
        return [], 0.0, "", ""

    tool_lines = []
    for index, invocation in enumerate(invocations, 1):
        tool_lines.append(
            json.dumps(
                {
                    "tool_index": index,
                    "skill_name": invocation.get("skill_name"),
                    "payload": invocation.get("payload", {}),
                },
                ensure_ascii=False,
            )
        )
    user_prompt = (
        f"Problem id: {case.case_id}\n"
        f"Topic: {case.topic}\n"
        f"Problem: {case.prompt}\n\n"
        "You are the agent tool router. Decide whether the agent should call one or more calculation tools. "
        "The tool definitions below include callable names and arguments, but not results. "
        "Use tools only when the calculation is likely to be error-prone or time-consuming by hand.\n"
        + "\n".join(tool_lines)
        + "\n\nReturn only JSON with keys use_tools, tool_indices, and reason. "
        "tool_indices must be a list of integers from the available tool_index values."
    )
    payload = {
        "model": config.model,
        "system_prompt": "Return only valid JSON. Do not solve the problem; only choose tools.",
        "input": user_prompt,
    }
    started_at = time.perf_counter()
    try:
        response = requests.post(config.api_url, json=payload, timeout=config.timeout)
        latency = time.perf_counter() - started_at
        response.raise_for_status()
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        raw_text = extract_response_text(body)
        parsed = extract_json_object(raw_text) or {}
    except Exception as exc:  # noqa: BLE001
        return [], time.perf_counter() - started_at, f"tool_choice_error: {exc}", ""

    use_tools = bool(parsed.get("use_tools"))
    raw_indices = parsed.get("tool_indices", [])
    if isinstance(raw_indices, int):
        raw_indices = [raw_indices]
    if not isinstance(raw_indices, list):
        raw_indices = []
    valid_indices = set(range(1, len(invocations) + 1))
    selected_indices: list[int] = []
    if use_tools:
        for item in raw_indices:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if index in valid_indices and index not in selected_indices:
                selected_indices.append(index)
    return selected_indices, latency, "", raw_text


def synthesize_answer(case: BenchmarkCase, notes: list[str], config: ToTConfig) -> tuple[dict[str, Any], float]:
    user_prompt = (
        f"Problem id: {case.case_id}\n"
        f"Topic: {case.topic}\n"
        f"Problem: {case.prompt}\n\n"
        "The following notes are branch outputs from a tree search. Use them as evidence, "
        "but correct local mistakes if needed.\n"
        + "\n".join(f"Branch note {index + 1}: {note}" for index, note in enumerate(notes))
        + "\n\nReturn only JSON with keys final_answer and concise_solution. "
        "The final_answer must contain the final numeric value(s) with units."
    )
    payload = {
        "model": config.model,
        "system_prompt": FREEFORM_SYSTEM_PROMPT,
        "input": user_prompt,
    }
    started_at = time.perf_counter()
    response = requests.post(config.api_url, json=payload, timeout=config.timeout)
    latency = time.perf_counter() - started_at
    response.raise_for_status()
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    return normalize_chat_response(body), latency


def run_tot_case(case: BenchmarkCase, config: ToTConfig) -> dict[str, Any]:
    tool_payload: dict[str, Any] = {}
    tool_notes: list[str] = []
    tool_latency = 0.0
    tool_error = ""
    tool_choice_latency = 0.0
    tool_choice_error = ""
    tool_choice_raw = ""
    selected_tool_indices: list[int] = []
    if config.tool_policy == "oracle":
        tool_payload, tool_notes, tool_latency, tool_error = run_case_tool_invocations(case)

    if tool_payload and config.tool_policy == "oracle":
        scored = score_case(case, tool_payload)
        return {
            **scored,
            "tree_latency_seconds": 0.0,
            "synthesis_latency_seconds": round(tool_latency, 3),
            "node_count": 0,
            "expansions_used": 0,
            "run_phase": "oracle_tool_skill",
            "tree_error": "",
            "synthesis_error": tool_error,
            "synthesis_source": "oracle_tool_skill",
            "tool_policy": config.tool_policy,
            "tool_participates": False,
            "tool_choice_latency_seconds": 0.0,
            "tool_choice_error": "",
            "tool_choice_raw": "",
            "selected_tool_indices": [],
            "tool_latency_seconds": round(tool_latency, 3),
            "tool_error": tool_error,
            "tool_audit_latency_seconds": 0.0,
            "tool_audit_error": "",
            "tool_audit_final_answer": "",
            "tool_audit_ok": False,
            "tool_audit_notes": [],
            "branch_notes": tool_notes,
        }

    if config.tool_policy == "agent":
        selected_tool_indices, tool_choice_latency, tool_choice_error, tool_choice_raw = request_agent_tool_choice(
            case,
            config,
        )
        if selected_tool_indices:
            tool_payload, tool_notes, tool_latency, tool_error = run_case_tool_invocations(
                case,
                selected_indices=selected_tool_indices,
            )

    backend_factory, deletion_review = build_local_chat_adapter_bundle(
        base_url=config.api_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        planning_model=config.model,
        modeling_model=config.model,
        review_model=config.model,
        non_terminal_evaluation_model=config.model,
        allow_live_model_fallback=config.allow_live_fallback,
        prefer_local_fallback=config.prefer_local_fallback,
    )
    problem_context = {
        "problem_statement": case.prompt,
        "task": "Build a route-diverse reasoning tree, then support one final numeric answer.",
        "known_context": {
            "objective": "Solve the problem by comparing local routes and checking constraints.",
            "expected_output": "final numeric answer with units",
        },
        "skill_query": case.topic,
    }
    if tool_payload and config.tool_policy == "agent":
        known_context = dict(problem_context["known_context"])
        known_context["agent_requested_tool_observations"] = "\n".join(tool_notes)
        problem_context["known_context"] = known_context
    scheduler = ToTTreeScheduler(
        problem_context,
        max_reflections=2,
        expansion_budget=config.expansions,
        max_tree_depth=config.depth,
        max_frontier_size=config.frontier,
        max_children_per_expansion=config.children,
        max_frontier_per_diversity_key=4,
        backend_adapter_factory=backend_factory,
        deletion_review_adapter=deletion_review,
    )
    started_at = time.perf_counter()
    error = ""
    try:
        with wall_timeout(config.tree_wall_timeout, "tree search"):
            scheduler.run()
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture per-case failures.
        error = f"tree_error: {exc}"
    tree_latency = time.perf_counter() - started_at

    branch_notes = select_branch_notes(scheduler.root_node, config.synthesis_nodes)
    synthesis_notes = [*tool_notes, *branch_notes] if config.tool_policy == "agent" else branch_notes
    synthesis_payload: dict[str, Any] = {}
    synthesis_latency = 0.0
    synthesis_error = tool_error
    synthesis_source = ""
    if synthesis_notes and not error:
        try:
            with wall_timeout(config.synthesis_wall_timeout, "answer synthesis"):
                synthesis_payload, synthesis_latency = synthesize_answer(case, synthesis_notes, config)
            synthesis_source = "agent_model_synthesis" if tool_notes else "model_synthesis"
        except Exception as exc:  # noqa: BLE001
            synthesis_error = f"synthesis_error: {exc}"
    if tool_payload and config.tool_policy == "agent" and not synthesis_payload:
        synthesis_payload = {
            "final_answer": str(tool_payload.get("final_answer", "")),
            "concise_solution": (
                "Agent selected a calculation tool before seeing the result; "
                "using that requested tool observation after later tree/model stages did not complete. "
                + str(tool_payload.get("concise_solution", ""))
            ).strip(),
        }
        synthesis_source = "agent_requested_tool_fallback"
    if branch_notes and not synthesis_payload and not tool_notes:
        synthesis_payload = fallback_payload_from_notes(branch_notes)
        synthesis_source = "branch_notes"

    scored = score_case(case, synthesis_payload) if synthesis_payload else {
        "case_id": case.case_id,
        "topic": case.topic,
        "ok": False,
        "matches": [],
        "model_final_answer": "",
        "reference_answer": case.reference_answer,
    }
    audit_payload: dict[str, Any] = {}
    audit_notes: list[str] = []
    audit_latency = 0.0
    audit_error = ""
    audit_scored: dict[str, Any] = {}
    if config.tool_policy == "audit":
        audit_payload, audit_notes, audit_latency, audit_error = run_case_tool_invocations(case)
        if audit_payload:
            audit_scored = score_case(case, audit_payload)

    return {
        **scored,
        "tree_latency_seconds": round(tree_latency, 3),
        "synthesis_latency_seconds": round(synthesis_latency, 3),
        "node_count": sum(1 for _ in iter_nodes(scheduler.root_node)),
        "expansions_used": len(getattr(scheduler, "_expanded_node_ids", [])),
        "run_phase": scheduler.run_phase,
        "tree_error": error,
        "synthesis_error": synthesis_error,
        "synthesis_source": synthesis_source,
        "tool_policy": config.tool_policy,
        "tool_participates": bool(tool_payload and config.tool_policy == "agent"),
        "tool_choice_latency_seconds": round(tool_choice_latency, 3),
        "tool_choice_error": tool_choice_error,
        "tool_choice_raw": tool_choice_raw,
        "selected_tool_indices": selected_tool_indices,
        "tool_latency_seconds": round(tool_latency, 3),
        "tool_error": tool_error,
        "tool_audit_latency_seconds": round(audit_latency, 3),
        "tool_audit_error": audit_error,
        "tool_audit_final_answer": str(audit_payload.get("final_answer", "")) if audit_payload else "",
        "tool_audit_ok": bool(audit_scored.get("ok", False)) if audit_scored else False,
        "tool_audit_notes": audit_notes,
        "branch_notes": synthesis_notes,
    }


def _run_direct_case_inline(
    case: BenchmarkCase,
    api_url: str,
    model: str,
    timeout: float,
    freeform_output: bool,
) -> dict[str, Any]:
    try:
        payload, latency = call_local_model(
            api_url,
            model,
            case,
            timeout,
            freeform_output=freeform_output,
        )
        scored = score_case(case, payload)
        scored["latency_seconds"] = round(latency, 3)
        scored["error"] = ""
        return scored
    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": case.case_id,
            "topic": case.topic,
            "ok": False,
            "matches": [],
            "model_final_answer": "",
            "reference_answer": case.reference_answer,
            "latency_seconds": None,
            "error": str(exc),
        }


def _direct_case_worker(
    result_queue: mp.Queue,
    case: BenchmarkCase,
    api_url: str,
    model: str,
    timeout: float,
    freeform_output: bool,
) -> None:
    result_queue.put(_run_direct_case_inline(case, api_url, model, timeout, freeform_output))


def run_direct_case(case: BenchmarkCase, args: argparse.Namespace) -> dict[str, Any]:
    if args.direct_wall_timeout <= 0:
        return _run_direct_case_inline(case, args.api_url, args.model, args.timeout, args.freeform_output)

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_direct_case_worker,
        args=(result_queue, case, args.api_url, args.model, args.timeout, args.freeform_output),
    )
    process.start()
    process.join(args.direct_wall_timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "case_id": case.case_id,
            "topic": case.topic,
            "ok": False,
            "matches": [],
            "model_final_answer": "",
            "reference_answer": case.reference_answer,
            "latency_seconds": None,
            "error": f"direct call exceeded {args.direct_wall_timeout:.1f}s wall timeout",
        }
    if not result_queue.empty():
        return result_queue.get()
    return {
        "case_id": case.case_id,
        "topic": case.topic,
        "ok": False,
        "matches": [],
        "model_final_answer": "",
        "reference_answer": case.reference_answer,
        "latency_seconds": None,
        "error": f"direct worker exited without result (exitcode={process.exitcode})",
    }


def selected_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = list(SUITES[args.suite])
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown case id(s) for suite {args.suite}: {', '.join(sorted(missing))}")
    if args.limit:
        cases = cases[: args.limit]
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct calls with tree-search calls on imported benchmark cases.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", choices=sorted(SUITES), default="limit")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--direct-wall-timeout", type=float, default=600.0)
    parser.add_argument("--mode", choices=("direct", "tot", "both"), default="both")
    parser.add_argument("--output", default="")
    parser.add_argument("--expansions", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--children", type=int, default=6)
    parser.add_argument("--frontier", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--synthesis-nodes", type=int, default=8)
    parser.add_argument("--tree-wall-timeout", type=float, default=900.0)
    parser.add_argument("--synthesis-wall-timeout", type=float, default=600.0)
    parser.add_argument(
        "--tool-policy",
        choices=("audit", "agent", "oracle"),
        default="audit",
        help=(
            "audit: tools are only run after the answer for scoring/audit metadata; "
            "agent: the model must request tool indices before tool observations are used; "
            "oracle: legacy shortcut that uses benchmark tool answers directly and must not be used as agent evidence."
        ),
    )
    parser.add_argument(
        "--tool-participates",
        action="store_true",
        help="Deprecated alias for --tool-policy agent. This no longer enables oracle tool answers.",
    )
    parser.add_argument("--allow-live-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-local-fallback", action="store_true")
    parser.add_argument("--freeform-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
    tool_policy = "agent" if args.tool_participates else args.tool_policy
    config = ToTConfig(
        api_url=args.api_url,
        model=args.model,
        timeout=args.timeout,
        expansions=args.expansions,
        depth=args.depth,
        children=args.children,
        frontier=args.frontier,
        max_retries=args.max_retries,
        allow_live_fallback=args.allow_live_fallback,
        prefer_local_fallback=args.prefer_local_fallback,
        synthesis_nodes=args.synthesis_nodes,
        tree_wall_timeout=args.tree_wall_timeout,
        synthesis_wall_timeout=args.synthesis_wall_timeout,
        tool_policy=tool_policy,
    )
    rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        row: dict[str, Any] = {"case_id": case.case_id, "topic": case.topic}
        if args.mode in {"direct", "both"}:
            direct = run_direct_case(case, args)
            row["direct"] = direct
            print(f"  direct={'OK' if direct['ok'] else 'FAIL'}", flush=True)
        if args.mode in {"tot", "both"}:
            tot = run_tot_case(case, config)
            row["tot"] = tot
            print(f"  tot={'OK' if tot['ok'] else 'FAIL'} nodes={tot['node_count']} expansions={tot['expansions_used']}", flush=True)
        rows.append(row)

    summary = {
        "suite": args.suite,
        "mode": args.mode,
        "model": args.model,
        "case_count": len(cases),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "tool_policy": tool_policy,
        "tool_participates": any(
            bool(row.get("tot", {}).get("tool_participates"))
            for row in rows
        ),
        "rows": rows,
    }
    for key in ("direct", "tot"):
        available = [row[key] for row in rows if key in row]
        if available:
            summary[f"{key}_correct"] = sum(1 for item in available if item.get("ok"))
            summary[f"{key}_accuracy"] = summary[f"{key}_correct"] / len(available)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, ensure_ascii=False, indent=2)
        print(f"wrote={args.output}")
    print("COMPARE_SUMMARY")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())