"""Regression tests for benchmark isolation and grading integrity."""

import unittest

from benchmarks import (
    benchmark_physics_solver,
    benchmark_symbolic_solver,
    direct_9b_benchmark_payload,
)
from direct_9b_benchmark import (
    BenchmarkCase,
    ExpectedValue,
    extract_scoring_answer_text,
    score_case,
)
from fsm import ToTNode
from skills import PUBLIC_SKILL_NAMES, SKILL_REGISTRY, search_skills
from tot_compare_runner import fallback_payload_from_tree

BENCHMARK_ONLY_NAMES = (
    "direct_9b_benchmark_payload",
    "benchmark_symbolic_solver",
    "benchmark_physics_solver",
)


class BenchmarkIsolationTests(unittest.TestCase):
    def test_benchmark_content_is_not_registered_as_runtime_skills(self) -> None:
        for name in BENCHMARK_ONLY_NAMES:
            self.assertNotIn(name, SKILL_REGISTRY)
            self.assertNotIn(name, PUBLIC_SKILL_NAMES)

    def test_skill_search_cannot_discover_benchmark_solvers(self) -> None:
        matches = search_skills("benchmark")
        matched_names = {str(entry.get("name", "")) for entry in matches}
        for name in BENCHMARK_ONLY_NAMES:
            self.assertNotIn(name, matched_names)

    def test_benchmark_tool_invocations_execute_through_benchmarks_module(self) -> None:
        payload = direct_9b_benchmark_payload({})
        solvers = {
            "benchmark_symbolic_solver": benchmark_symbolic_solver,
            "benchmark_physics_solver": benchmark_physics_solver,
        }
        executed = 0
        for suite_cases in payload["suites"].values():
            for case in suite_cases:
                for invocation in case.get("tool_invocations", []):
                    solver = solvers[invocation["skill_name"]]
                    result = solver(dict(invocation["payload"]))
                    self.assertIsInstance(result, dict)
                    self.assertTrue(str(result.get("final_answer", "")).strip())
                    executed += 1
        self.assertGreater(executed, 0)


class GradingIntegrityTests(unittest.TestCase):
    @staticmethod
    def _case(expected: float, abs_tol: float = 0.05) -> BenchmarkCase:
        return BenchmarkCase(
            case_id="grading-regression",
            topic="grading",
            prompt="irrelevant",
            expected_values=(ExpectedValue(label="x", value=expected, abs_tol=abs_tol),),
            reference_answer="x = expected",
        )

    def test_marker_tail_is_used_for_long_responses(self) -> None:
        text = ("intermediate value 3.0 appears here. " * 40) + "\nFinal answer: x = 7.25"
        region = extract_scoring_answer_text(text)
        self.assertIn("7.25", region)
        self.assertNotIn("intermediate", region)

    def test_long_response_without_marker_scans_only_final_lines(self) -> None:
        body_lines = [f"step {index}: trying candidate {index}.0" for index in range(60)]
        text = "\n".join([*body_lines, "so the requested quantity equals 7.25"])
        region = extract_scoring_answer_text(text)
        self.assertIn("7.25", region)
        self.assertNotIn("step 5:", region)

    def test_number_buried_mid_response_no_longer_matches(self) -> None:
        filler = "\n".join(f"derivation line {index} with no digits here" for index in range(80))
        text = "the value 3.0 shows up early\n" + filler + "\nno conclusion was reached"
        scored = score_case(self._case(3.0), {"final_answer": text})
        self.assertFalse(scored["ok"])

    def test_number_in_final_line_still_matches(self) -> None:
        filler = "\n".join(f"derivation line {index} with no digits here" for index in range(80))
        text = filler + "\ntherefore x = 3.0"
        scored = score_case(self._case(3.0), {"final_answer": text})
        self.assertTrue(scored["ok"])

    def test_fallback_payload_ignores_node_scores_and_metadata(self) -> None:
        root = ToTNode(thought_step="root", score=3.0)
        child = ToTNode(parent_id=root.id, thought_step="child", score=3.0)
        root.children.append(child)

        self.assertEqual(fallback_payload_from_tree(root), {})

        child.known_vars["candidate_answer"] = "x = 7.25"
        payload = fallback_payload_from_tree(root)
        self.assertIn("7.25", payload["final_answer"])
        self.assertNotIn("3.0", payload["final_answer"])

        scored = score_case(self._case(3.0), payload)
        self.assertFalse(scored["ok"])


if __name__ == "__main__":
    unittest.main()
