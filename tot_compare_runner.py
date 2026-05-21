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


def run_case_tool_invocations(case: BenchmarkCase) -> tuple[dict[str, Any], list[str], float, str]:
    invocations = [
        dict(item)
        for item in getattr(case, "tool_invocations", ())
        if isinstance(item, dict) and str(item.get("skill_name", "")).strip()
    ]
    if not invocations:
        return {}, [], 0.0, ""

    try:
        from skills import invoke_skill
    except Exception as exc:  # noqa: BLE001
        return {}, [], 0.0, f"tool_error: {exc}"

    started_at = time.perf_counter()
    notes: list[str] = []
    answers: list[str] = []
    for index, invocation in enumerate(invocations, 1):
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
    tool_payload, tool_notes, tool_latency, tool_error = run_case_tool_invocations(case)
    if tool_payload:
        scored = score_case(case, tool_payload)
        return {
            **scored,
            "tree_latency_seconds": 0.0,
            "synthesis_latency_seconds": round(tool_latency, 3),
            "node_count": 0,
            "expansions_used": 0,
            "run_phase": "tool_skill",
            "tree_error": "",
            "synthesis_error": tool_error,
            "synthesis_source": "tool_skill",
            "branch_notes": tool_notes,
        }

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

    notes = [*tool_notes, *select_branch_notes(scheduler.root_node, config.synthesis_nodes)]
    synthesis_payload: dict[str, Any] = {}
    synthesis_latency = round(tool_latency, 6) if tool_payload else 0.0
    synthesis_error = tool_error
    synthesis_source = ""
    if tool_payload:
        synthesis_payload = tool_payload
        synthesis_source = "tool_skill"
    elif notes and not error:
        try:
            with wall_timeout(config.synthesis_wall_timeout, "answer synthesis"):
                synthesis_payload, synthesis_latency = synthesize_answer(case, notes, config)
            synthesis_source = "model_synthesis"
        except Exception as exc:  # noqa: BLE001
            synthesis_error = f"synthesis_error: {exc}"
    if notes and not synthesis_payload:
        synthesis_payload = fallback_payload_from_notes(notes)
        synthesis_source = "branch_notes"

    scored = score_case(case, synthesis_payload) if synthesis_payload else {
        "case_id": case.case_id,
        "topic": case.topic,
        "ok": False,
        "matches": [],
        "model_final_answer": "",
        "reference_answer": case.reference_answer,
    }
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
        "branch_notes": notes,
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
    parser.add_argument("--allow-live-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-local-fallback", action="store_true")
    parser.add_argument("--freeform-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
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