"""Validation harness for hard-rule pruning changes.

Drives the running tot_api against real benchmark physics problems and, per case,
measures two things:

  * correct_answer_pruned -- a node that reached the full correct answer was
    killed with a PRUNED_BY_RULE status (the regression we care about).
  * solved -- a surviving (non-pruned) node reached the full correct answer.

Run the same subset before and after a logic change and diff the two rates.
Benchmark expected values stay in benchmarks.py; this harness only reads them
for scoring and never feeds them to the system under test.

Usage:
    python validate_pruning.py --suites boundary ap1 em traps \
        --per-case-timeout 180 --output reports/pruning_before.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from typing import Any

from direct_9b_benchmark import SUITES, extract_numbers

DEFAULT_BASE = "http://127.0.0.1:8000"


def _post(base: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base: str, path: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(base: str, path: str, timeout: float) -> None:
    req = urllib.request.Request(base + path, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        # Best-effort cleanup: a session may be mid-slice (lock held) so the
        # delete can block/time out. Never let cleanup crash the run.
        pass


def _walk(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return out
    out.append(node)
    for child in node.get("children") or []:
        out.extend(_walk(child))
    return out


def _node_numbers(node: dict[str, Any]) -> list[float]:
    """Numbers from a node's answer-bearing content only.

    Deliberately excludes the blind scan of every known_var value: givens such as
    ``g = 9.8`` live there and collide with expected answers (e.g. a normal force
    that also happens to be 9.8 N), inflating the solve count. Answers surface in
    equations (``d = 7.2 m``) or in explicit answer fields.
    """
    parts: list[str] = [str(eq) for eq in (node.get("equations") or [])]
    kv = node.get("known_vars") or {}
    for key in ("final_answer", "candidate_answer", "answer", "result"):
        if kv.get(key) is not None:
            parts.append(str(kv[key]))
    return extract_numbers(" ; ".join(parts))


def _node_grounding_violations(node: dict[str, Any]) -> list[str]:
    """Boundary 'not grounded' violations on a pruned node -- the fix's target."""
    if not _is_pruned(node):
        return []
    kv = node.get("known_vars") or {}
    return [
        v
        for v in (kv.get("hard_rule_violations") or [])
        if "not grounded" in str(v).lower()
    ]


def _value_is_numeric(value: Any) -> bool:
    text = str(value.get("value") if isinstance(value, dict) and "value" in value else value).strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _node_suppressed_boundary_entries(node: dict[str, Any]) -> list[str]:
    """Boundary entries the fix exempts (non-numeric value) -- over-suppression audit.

    These are what the source-level fix silently drops from grounding. Surfacing
    them lets us check none is actually a real numeric constraint in disguise.
    """
    kv = node.get("known_vars") or {}
    checked = (kv.get("hard_rule_check") or {}).get("checked") or {}
    bc = checked.get("boundary_conditions")
    if not isinstance(bc, dict):
        return []
    return [f"{k} = {v}" for k, v in bc.items() if not _value_is_numeric(v)]


def _node_reaches_answer(node: dict[str, Any], case: Any) -> bool:
    """True iff the node's numbers cover every expected value within tolerance."""
    numbers = _node_numbers(node)
    if not numbers:
        return False
    for expected in case.expected_values:
        tol = max(expected.abs_tol, abs(expected.value) * expected.rel_tol)
        if not any(abs(n - expected.value) <= tol for n in numbers):
            return False
    return True


def _is_pruned(node: dict[str, Any]) -> bool:
    return "PRUN" in str(node.get("status") or "").upper()


def evaluate_tree(state: dict[str, Any], case: Any) -> dict[str, Any]:
    nodes = _walk(state.get("root"))
    pruned_correct: list[dict[str, Any]] = []
    solved_nodes: list[dict[str, Any]] = []
    grounding_violations: list[str] = []
    suppressed: set[str] = set()
    for n in nodes:
        grounding_violations.extend(_node_grounding_violations(n))
        suppressed.update(_node_suppressed_boundary_entries(n))
        if not _node_reaches_answer(n, case):
            continue
        if _is_pruned(n):
            pruned_correct.append(n)
        else:
            solved_nodes.append(n)
    return {
        "node_count": len(nodes),
        "solved": bool(solved_nodes),
        "correct_answer_pruned": bool(pruned_correct),
        # The direct fix metric: nodes killed for an ungrounded boundary axis.
        "grounding_prunes": len(grounding_violations),
        "grounding_violation_samples": sorted(set(grounding_violations))[:5],
        # Over-suppression audit: boundary entries the fix exempted from grounding.
        "suppressed_boundary_samples": sorted(suppressed)[:20],
    }


def _scheduler_config(max_expansions: int) -> dict[str, Any]:
    # Mirror the API defaults but cap lifetime expansions so each case runs to
    # completion in bounded time. run_on_create is False so there is no background
    # auto-run thread -- the case is driven by a single synchronous /run call,
    # which keeps the runs serial and contention-free.
    return {
        "depth_preset": "medium",
        "max_reflections": 2,
        "max_tree_depth": 8,
        "max_frontier_size": 16,
        "max_children_per_expansion": 2,
        "max_live_children_per_batch": 2,
        "max_total_expansions": max_expansions,
        "use_local_root_proposal": True,
        "use_local_root_evaluation": True,
        "use_local_child_proposal": True,
        "use_local_child_evaluation": True,
        "max_frontier_per_diversity_key": 4,
        "children_key": "children",
    }


_EMPTY_EVAL = {
    "node_count": 0,
    "solved": False,
    "correct_answer_pruned": False,
    "grounding_prunes": 0,
    "grounding_violation_samples": [],
}


def run_case(
    base: str,
    case: Any,
    max_expansions: int,
    per_case_timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    created = _post(
        base,
        "/api/tot/sessions",
        {
            "run_on_create": True,
            "problem_context": {"problem_statement": case.prompt},
            "scheduler": _scheduler_config(max_expansions),
        },
        timeout=60,
    )
    sid = created["session_id"]
    deadline = time.monotonic() + per_case_timeout
    result = dict(_EMPTY_EVAL)
    status = "busy"
    error = ""
    try:
        while True:
            try:
                state = _get(base, f"/api/tot/sessions/{sid}", timeout=30)["state"]
            except Exception as exc:
                error = str(exc)[:200]
                status = "error"
                break
            run_state = state["run_state"]
            status = run_state["status"]
            error = run_state["last_error"]
            result = evaluate_tree(state, case)
            # Early exit once the answer node exists (solved or wrongly pruned);
            # the grounding-prune count is captured from the same snapshot.
            if result["solved"] or result["correct_answer_pruned"]:
                break
            if status in ("ready", "error"):
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                break
            time.sleep(poll_interval)
    finally:
        # Short, non-blocking cleanup: the capped background run (<= max_expansions)
        # finishes on its own shortly after, so we never wait on the session lock.
        _delete(base, f"/api/tot/sessions/{sid}", timeout=8)
    return {
        "case_id": case.case_id,
        "topic": case.topic,
        "final_status": status,
        "error": error[:200],
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--suites", nargs="+", default=["boundary", "ap1", "em", "traps"])
    parser.add_argument("--problems-file", default="", help="JSON list of fresh problems; overrides --suites.")
    parser.add_argument("--max-expansions", type=int, default=12, help="Lifetime expansion cap per case.")
    parser.add_argument("--per-case-timeout", type=float, default=140.0, help="Wall-clock cap per case before giving up.")
    parser.add_argument("--poll-interval", type=float, default=4.0)
    parser.add_argument("--concurrency", type=int, default=1, help="Run this many cases in parallel (needs server TOT_MAX_ACTIVE_AUTO_RUNS >= this).")
    parser.add_argument("--limit", type=int, default=0, help="Cap cases per suite (0 = all).")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cases: list[Any] = []
    if args.problems_file:
        from types import SimpleNamespace

        with open(args.problems_file, encoding="utf-8") as fh:
            raw = json.load(fh)
        for item in raw:
            expected = [
                SimpleNamespace(
                    label=str(e.get("label", "")),
                    value=float(e["value"]),
                    abs_tol=float(e.get("abs_tol", 0.0)),
                    rel_tol=float(e.get("rel_tol", 0.0)),
                )
                for e in item.get("expected_values", [])
            ]
            cases.append(
                SimpleNamespace(
                    case_id=str(item["case_id"]),
                    topic=str(item.get("topic", "")),
                    prompt=str(item["prompt"]),
                    expected_values=expected,
                )
            )
        source = args.problems_file
    else:
        for suite in args.suites:
            if suite not in SUITES:
                raise SystemExit(f"unknown suite: {suite} (have {sorted(SUITES)})")
            suite_cases = list(SUITES[suite])
            if args.limit:
                suite_cases = suite_cases[: args.limit]
            cases.extend(suite_cases)
        source = f"suites={args.suites}"

    conc = max(1, int(args.concurrency))
    print(f"validating {len(cases)} cases ({source}) (max_expansions={args.max_expansions}, concurrency={conc})\n", flush=True)

    def work(item):
        i, case = item
        t0 = time.monotonic()
        row = run_case(args.base, case, args.max_expansions, args.per_case_timeout, args.poll_interval)
        row["seconds"] = round(time.monotonic() - t0, 1)
        row["_i"] = i
        return row

    rows: list[dict[str, Any]] = []
    done = 0
    items = list(enumerate(cases, 1))
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futures = [ex.submit(work, it) for it in items]
        for fut in cf.as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            flag = "solved" if row["solved"] else ("PRUNED-CORRECT" if row["correct_answer_pruned"] else "no-answer")
            print(
                f"[{done:>2}/{len(cases)}] {row['case_id']:<40} {flag:<14} "
                f"gnd_prunes={row['grounding_prunes']:<2} status={row['final_status']:<7} "
                f"nodes={row['node_count']:<3} {row['seconds']}s",
                flush=True,
            )
            if row["error"]:
                print(f"        error: {row['error']}", flush=True)
    rows.sort(key=lambda r: r.get("_i", 0))

    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    pruned_correct = sum(r["correct_answer_pruned"] for r in rows)
    grounding_prunes = sum(r["grounding_prunes"] for r in rows)
    errored = sum(1 for r in rows if r["error"])
    suppressed_all = sorted({s for r in rows for s in r.get("suppressed_boundary_samples", [])})
    summary = {
        "source": source,
        "max_expansions": args.max_expansions,
        "cases": n,
        "solved": solved,
        "solved_rate": round(solved / n, 3) if n else 0.0,
        "correct_answer_pruned": pruned_correct,
        "grounding_prunes_total": grounding_prunes,
        "errored_cases": errored,
        "suppressed_boundary_entries": suppressed_all,
        "rows": rows,
    }
    print(
        f"\nSUMMARY  cases={n}  solved={solved} ({summary['solved_rate']})  "
        f"correct-answer-pruned={pruned_correct}  grounding-prunes(total)={grounding_prunes}  "
        f"errored={errored}",
        flush=True,
    )
    print(f"distinct suppressed boundary entries (over-suppression audit): {len(suppressed_all)}", flush=True)
    for s in suppressed_all[:40]:
        print(f"  - {s}", flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
