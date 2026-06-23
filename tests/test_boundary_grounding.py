"""Deterministic tests for advisory boundary-grounding.

Boundary-grounding ("X is not grounded in equations") proved to be an unreliable
signal across the benchmark corpus and a fresh problem set: this model routinely
writes boundary keys (``v_initial``, ``v_final``, ``T``) that do not textually
match its own compressed equation symbols, and encodes real numeric constraints
with units (``T = 20 C``). Both make the grounding check false-positive on
legitimate boundary conditions.

The fix makes boundary-grounding *advisory*: the violation is still produced (for
observability, in ``ignored_violations``) but never prunes, in any context. Gated
by TOT_BOUNDARY_GROUNDING_FIX (default on); these assume default-on.

Advisory is scoped to grounding only -- a different boundary check (value depends
on the constrained axis) is still surfaced as an effective violation.
"""
import unittest

from fsm import NodeBuilderFSM, NodeStatus, ToTNode

GROUND_KEY = "Boundary condition key is not grounded in equations or known variables:"
GROUND_AXIS = "Boundary condition axis is not grounded in equations or known variables:"


def _terminal_route_context(*, boundary_conditions, equations, route_family="route-b"):
    return {
        "meta_task": {
            "first_step": "identify governing relation",
            "step_ordering": [
                "identify governing relation",
                "choose one active correction or closure",
                "express the target quantity in known variables",
            ],
        },
        "meta_task_progress": {
            "current_step_index": 2,
            "current_step": "express the target quantity in known variables",
            "phase": "incremental_refinement",
            "current_step_guidance": "Execute only this task: express the target quantity in known variables.",
            "previous_steps": ["identify governing relation", "choose one active correction or closure"],
            "remaining_steps": [],
            "is_terminal_step": True,
            "selected_route_family": route_family,
        },
        "route_focus": {"label": f"{route_family} route", "route_family": route_family},
        "proposal": {
            "thought_step": "Collapse the selected route into the final target quantity.",
            "equations": equations,
            "used_models": ["Kinematics"],
            "boundary_conditions": boundary_conditions,
        },
        "calculation": {},
        "evaluation": {"score": 8.0},
    }


class AdvisoryGroundingTests(unittest.TestCase):
    def _run(self, **ctx_kwargs):
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context=_terminal_route_context(**ctx_kwargs),
            max_reflections=1,
        )
        return fsm.run()

    def _check(self, node):
        return node.known_vars["hard_rule_check"]

    def test_descriptive_key_is_ignored_not_pruned(self):
        node = self._run(
            boundary_conditions={"surface": "rough segment"},
            equations=["m a = m g sin(theta) - mu_k N"],
        )
        check = self._check(node)
        self.assertNotEqual(node.status, NodeStatus.PRUNED_BY_RULE)
        self.assertTrue(check["passed"])
        self.assertIn(f"{GROUND_KEY} surface", check["ignored_violations"])

    def test_finding2_unit_bearing_constraint_is_ignored_not_over_suppressed(self):
        # T = 20 C is a real numeric constraint (units broke the old value heuristic).
        # Advisory logs it and moves on -- not silently dropped, not pruned.
        node = self._run(
            boundary_conditions={"T": "20 C"},
            equations=["q = -k A dT/dx"],
        )
        check = self._check(node)
        self.assertNotEqual(node.status, NodeStatus.PRUNED_BY_RULE)
        self.assertTrue(check["passed"])
        self.assertIn(f"{GROUND_KEY} T", check["ignored_violations"])

    def test_finding1_numeric_keyed_boundary_condition_is_ignored_not_pruned(self):
        # v_initial / v_final are real BCs whose key names do not match the equation
        # symbols (v0); these previously still pruned. Now advisory.
        node = self._run(
            boundary_conditions={"v_initial": 5.0, "v_final": 0},
            equations=["a = g*(sin(theta) + mu*cos(theta))", "d = v0**2/(2*a)"],
        )
        check = self._check(node)
        self.assertNotEqual(node.status, NodeStatus.PRUNED_BY_RULE)
        self.assertTrue(check["passed"])
        self.assertIn(f"{GROUND_KEY} v_initial", check["ignored_violations"])
        self.assertIn(f"{GROUND_KEY} v_final", check["ignored_violations"])

    def test_live_bug_consumed_axis_is_ignored(self):
        node = self._run(
            boundary_conditions={"v_f = 0": "active_boundary_condition"},
            equations=["d = h / mu_k", "d = 7.2 m"],
        )
        check = self._check(node)
        self.assertNotEqual(node.status, NodeStatus.PRUNED_BY_RULE)
        self.assertIn(f"{GROUND_AXIS} v_f", check["ignored_violations"])

    def test_advisory_is_scoped_to_grounding_only(self):
        # A non-grounding boundary check (value depends on the constrained axis) is
        # NOT advisory -- it must still surface as an effective violation.
        node = self._run(
            boundary_conditions={"x = 0": "u(x,t)"},
            equations=["u_t = D*u_xx"],
        )
        self.assertIn(
            "Boundary condition value depends on constrained axis: x = 0",
            self._check(node)["violations"],
        )


if __name__ == "__main__":
    unittest.main()
