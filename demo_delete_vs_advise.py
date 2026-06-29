"""Show 'deletes' vs 'advises' on the SAME node the browser would produce.

The browser POSTs a problem; the scheduler turns it into nodes; each node runs
through NodeBuilderFSM (exactly what this drives). We feed one node that carries a
boundary condition whose name doesn't match its equation symbols -- the exact thing
the grounding check trips on -- and print the node object the API returns (and the
browser renders), in HARD mode (deletes) vs ADVISORY mode (advises).

Mode comes from TOT_BOUNDARY_GROUNDING_FIX (read at import), so run this twice.
"""
import os
from fsm import NodeBuilderFSM, NodeStatus, ToTNode

# This mirrors what the scheduler builds for a terminal route step from a browser
# problem like "...how far does the block travel before stopping?" The model wrote a
# boundary condition keyed "v_f = 0" but the equation uses d/h/mu_k -- names don't match.
problem_context = {
    "meta_task": {"first_step": "identify governing relation",
                  "step_ordering": ["identify governing relation",
                                    "choose one active correction or closure",
                                    "express the target quantity in known variables"]},
    "meta_task_progress": {"current_step_index": 2,
                           "current_step": "express the target quantity in known variables",
                           "phase": "incremental_refinement", "is_terminal_step": True,
                           "selected_route_family": "route-b"},
    "route_focus": {"label": "route-b route", "route_family": "route-b"},
    "proposal": {"thought_step": "Collapse the route into the final distance.",
                 "equations": ["d = h / mu_k", "d = 7.2 m"],
                 "used_models": ["Kinematics"],
                 "boundary_conditions": {"v_f = 0": "active_boundary_condition"}},
    "calculation": {},
    "evaluation": {"score": 8.0},
}

mode = "ADVISORY (advises)" if os.getenv("TOT_BOUNDARY_GROUNDING_FIX", "1") != "0" else "HARD (deletes)"
node = NodeBuilderFSM(parent_node=ToTNode(id="parent-1"),
                      problem_context=problem_context, max_reflections=1).run()
hrc = node.known_vars.get("hard_rule_check", {})

print(f"================ MODE: {mode} ================")
print(f"node.status            = {node.status.value}")
print(f"result_state           = {getattr(node.result_state, 'value', node.result_state)}")
print(f"hard_rule_check.passed = {hrc.get('passed')}")
print(f"FATAL violations       = {hrc.get('violations')}")
print(f"IGNORED (advised)      = {hrc.get('ignored_violations')}")
print(f"the equation it derived: {node.equations}")
deleted = str(node.status.value).upper().startswith("PRUNED")
print(f"\n-> browser would render this node as: {'PRUNED (deleted, removed from the tree)' if deleted else 'ACTIVE'}"
      + ("" if deleted else f"  with an 'ignored noise' badge x{len(hrc.get('ignored_violations') or [])}"))
