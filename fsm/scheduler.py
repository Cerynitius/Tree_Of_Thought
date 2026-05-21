"""Tree-level scheduler built on top of the single-node FSM."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .backend import (
    DeleteNodeReviewDecision,
    DeleteNodeReviewRequest,
    NodeDeletionReviewAdapter,
    ReasoningBackendAdapter,
)
from .builder import NodeBuilderFSM
from .models import NodeSnapshot, NodeStatus, ToTNode
from .utils import _deserialize_blob, _model_field_names, _serialize_blob, _stable_hash


class TreeSchedulerState(BaseModel):
    """Persistent scheduler snapshot.

    The state payload is stored as a trusted local pickle-backed blob so that
    SymPy expressions and recursive Pydantic node trees survive round-trips.
    """

    version: int = 1
    snapshot_blob: str


class ToTTreeScheduler:
    """Tree-level harness that expands and ranks sibling nodes.

    The scheduler reuses ``NodeBuilderFSM`` for each node and adds the missing
    tree-level behavior on top:

    - sibling ranking
    - frontier management with depth-biased priority
    - global expansion-budget control
    - diversity control
    - state deduplication and loop suppression
    - duplicate-branch merge
    - persistence and resume support

    Child candidates are provided deterministically through the ``children`` key
    in each node problem context. Each child context may itself contain nested
    ``children`` entries for deeper expansion.
    """

    META_TASK_STRATEGY_SCAN_GUIDANCE = (
        "Analyze the next-step strategy space at planning level. "
        "Make one route-local planning claim only, keep all other routes deferred, and do not solve the final answer yet."
    )

    def __init__(
        self,
        root_problem_context: dict[str, Any],
        *,
        max_reflections: int = 2,
        expansion_budget: int = 8,
        max_tree_depth: int = 8,
        max_frontier_size: int = 16,
        max_children_per_expansion: int = 6,
        max_frontier_per_diversity_key: int = 4,
        children_key: str = "children",
        backend_adapter_factory: Optional[Callable[[dict[str, Any]], ReasoningBackendAdapter]] = None,
        deletion_review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> None:
        if expansion_budget < 0:
            raise ValueError("expansion_budget must be non-negative.")
        if max_tree_depth < 0:
            raise ValueError("max_tree_depth must be non-negative.")
        if max_frontier_size < 1:
            raise ValueError("max_frontier_size must be at least 1.")
        if max_children_per_expansion < 1:
            raise ValueError("max_children_per_expansion must be at least 1.")
        if max_frontier_per_diversity_key < 1:
            raise ValueError("max_frontier_per_diversity_key must be at least 1.")

        self.root_problem_context = root_problem_context
        self.max_reflections = max_reflections
        self.expansion_budget = expansion_budget
        self.max_tree_depth = max_tree_depth
        self.max_frontier_size = max_frontier_size
        self.max_children_per_expansion = max_children_per_expansion
        self.max_frontier_per_diversity_key = max_frontier_per_diversity_key
        self.children_key = children_key
        self.backend_adapter_factory = backend_adapter_factory
        self.deletion_review_adapter = deletion_review_adapter

        self.root_node: Optional[ToTNode] = None
        self._frontier: list[dict[str, Any]] = []
        self._expansion_log: list[dict[str, Any]] = []
        self._expanded_node_ids: list[str] = []
        self._node_index: dict[str, ToTNode] = {}
        self._signature_registry: dict[str, str] = {}
        self._problem_context_prepared = False
        self.target_expansion_budget = expansion_budget
        self.run_status = "idle"
        self.run_phase = "created"
        self.last_error = ""
        self.auto_run_requested = False

    def _set_run_state(
        self,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        if status is not None:
            self.run_status = str(status)
        if phase is not None:
            self.run_phase = str(phase)
        if last_error is not None:
            self.last_error = str(last_error)

    def _prepare_root_problem_context_if_needed(self) -> None:
        if self._problem_context_prepared:
            return
        if self.backend_adapter_factory is None:
            self._problem_context_prepared = True
            return

        self._set_run_state(status="busy", phase="preparing-meta-task", last_error="")
        backend_adapter = self.backend_adapter_factory(self.root_problem_context)
        self.root_problem_context = backend_adapter.prepare_problem_context(deepcopy(self.root_problem_context))
        self._problem_context_prepared = True

    def _final_run_phase(self) -> str:
        if self.run_status == "error":
            return "error"
        if self.root_node is None:
            return "created"
        if not self._frontier:
            return "frontier-empty"
        if len(self._expanded_node_ids) >= self.expansion_budget:
            return "awaiting-next-step"
        return "idle"

    def run(self) -> dict[str, Any]:
        """Build the root node and expand the tree under scheduler constraints."""
        self._set_run_state(status="busy", phase="preparing-meta-task", last_error="")

        try:
            self._prepare_root_problem_context_if_needed()

            if self.root_node is None:
                self._set_run_state(status="busy", phase="building-root")
                root_node = self._build_node(parent_node=None, problem_context=self.root_problem_context)
                self.root_node = root_node
                self._initialize_root_node(root_node)
                if self._can_expand_below_depth(0) and self._is_expandable(root_node, self.root_problem_context):
                    self._frontier.append(
                        {
                            "node": root_node,
                            "problem_context": self.root_problem_context,
                            "depth": 0,
                        }
                    )
                    retained_ids = self._rebalance_frontier()
                    root_node.known_vars["selected_for_frontier"] = root_node.id in retained_ids

            while self._frontier and len(self._expanded_node_ids) < self.expansion_budget:
                self._set_run_state(status="busy", phase="expanding-frontier")
                current = self._frontier.pop(0)
                parent_node = current["node"]
                problem_context = current["problem_context"]
                depth = current["depth"]

                child_contexts = self._extract_child_contexts(problem_context)
                if not child_contexts:
                    continue

                self._expanded_node_ids.append(parent_node.id)
                built_children: list[tuple[ToTNode, dict[str, Any]]] = []
                for child_context in child_contexts:
                    child_node = self._build_node(parent_node=parent_node, problem_context=child_context)
                    built_children.append((child_node, child_context))

                ranked_children = sorted(
                    built_children,
                    key=lambda item: self._node_ranking_key(item[0]),
                )

                expandable_candidates: list[tuple[ToTNode, dict[str, Any]]] = []
                sibling_ranking: list[dict[str, Any]] = []

                for rank, (child_node, child_context) in enumerate(ranked_children, start=1):
                    scheduler_action = self._apply_scheduler_controls(child_node, parent_node)
                    priority = self._node_priority(child_node)
                    expandable = (
                        scheduler_action is None
                        and self._can_expand_below_depth(depth + 1)
                        and self._is_expandable(child_node, child_context)
                    )
                    child_node.known_vars["sibling_rank"] = rank
                    child_node.known_vars["sibling_priority"] = priority
                    child_node.known_vars["scheduler_action"] = scheduler_action or "expanded"
                    child_node.known_vars["expandable"] = expandable
                    child_node.known_vars["selected_for_frontier"] = False
                    sibling_ranking.append(
                        {
                            "node_id": child_node.id,
                            "rank": rank,
                            "status": child_node.status.value,
                            "priority": priority,
                            "score": child_node.score,
                            "expandable": expandable,
                            "scheduler_action": scheduler_action or "expanded",
                            "diversity_key": child_node.known_vars.get("diversity_key"),
                        }
                    )
                    if expandable:
                        expandable_candidates.append((child_node, child_context))

                parent_node.known_vars["sibling_ranking"] = sibling_ranking

                frontier_candidate_budget = self._frontier_candidate_budget(problem_context)
                frontier_candidates = expandable_candidates[:frontier_candidate_budget]
                for child_node, child_context in frontier_candidates:
                    self._frontier.append(
                        {
                            "node": child_node,
                            "problem_context": child_context,
                            "depth": depth + 1,
                        }
                    )

                retained_ids = self._rebalance_frontier()
                for child_node, _ in ranked_children:
                    child_node.known_vars["selected_for_frontier"] = child_node.id in retained_ids

                parent_node.known_vars["selected_child_ids"] = [
                    child_node.id
                    for child_node, _ in ranked_children
                    if child_node.id in retained_ids
                ]
                self._expansion_log.append(
                    {
                        "parent_id": parent_node.id,
                        "depth": depth,
                        "expanded": True,
                        "child_ids": [child.id for child, _ in ranked_children],
                        "frontier_candidate_ids": [
                            child.id
                            for child, _ in frontier_candidates
                        ],
                        "retained_frontier_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.id in retained_ids
                        ],
                        "pruned_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.status != NodeStatus.ACTIVE
                        ],
                        "duplicate_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.known_vars.get("scheduler_action") == "merged-duplicate"
                        ],
                        "loop_suppressed_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.known_vars.get("scheduler_action") == "suppressed-loop"
                        ],
                        "budget_remaining": self.expansion_budget - len(self._expanded_node_ids),
                        "frontier_size_after": len(self._frontier),
                    }
                )
        except Exception as exc:
            self._set_run_state(status="error", phase="error", last_error=str(exc))
            raise

        self._set_run_state(status="ready", phase=self._final_run_phase(), last_error="")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the current scheduler state without mutating the tree."""

        meta_task = self.root_problem_context.get("meta_task", {})
        return {
            "root": self.root_node,
            "meta_task": dict(meta_task) if isinstance(meta_task, dict) else {},
            "frontier": self._frontier_snapshot(),
            "expansion_log": list(self._expansion_log),
            "expanded_node_ids": list(self._expanded_node_ids),
            "expansions_used": len(self._expanded_node_ids),
            "expansion_budget": self.expansion_budget,
            "max_tree_depth": self.max_tree_depth,
            "target_expansion_budget": self.target_expansion_budget,
            "remaining_budget": self.expansion_budget - len(self._expanded_node_ids),
            "run_state": {
                "status": self.run_status,
                "phase": self.run_phase,
                "problem_context_prepared": self._problem_context_prepared,
                "auto_run_requested": self.auto_run_requested,
                "target_expansion_budget": self.target_expansion_budget,
                "last_error": self.last_error,
            },
        }

    def delete_node(
        self,
        node_id: str,
        *,
        reason: str,
        requested_by: str = "frontend",
        review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> dict[str, Any]:
        """Delete a node subtree only after an AI review approves the operation."""

        if self.root_node is None:
            raise RuntimeError("Cannot delete a node before the tree has been built.")

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("Deletion reason must be provided for AI review.")

        target_node = self._node_index.get(node_id)
        if target_node is None:
            raise KeyError(f"Unknown node id: {node_id}")
        if target_node.parent_id is None:
            raise ValueError("Deleting the root node is not supported.")

        effective_review_adapter = review_adapter or self.deletion_review_adapter
        if effective_review_adapter is None:
            raise ValueError("AI deletion review adapter is required before deleting a node.")

        parent_node = self._node_index.get(target_node.parent_id)
        if parent_node is None:
            raise RuntimeError("Failed to resolve the parent node for deletion.")

        steering_route_focus = self._route_focus_from_node(target_node)
        if not steering_route_focus:
            steering_route_focus = self._route_focus_from_node(parent_node)

        review_request = DeleteNodeReviewRequest(
            requested_by=requested_by,
            reason=normalized_reason,
            current_root_id=self.root_node.id,
            current_frontier_size=len(self._frontier),
            target_node=self._node_snapshot(target_node),
            parent_node=self._node_snapshot(parent_node),
            descendant_count=max(0, len(self._collect_subtree_nodes(target_node)) - 1),
            is_frontier_node=any(entry["node"].id == node_id for entry in self._frontier),
            is_expanded_node=node_id in self._expanded_node_ids,
        )
        review = self._build_model(
            DeleteNodeReviewDecision,
            self._normalize_review_payload(
                effective_review_adapter.review_delete_node(review_request)
            ),
        )

        if not review.approved:
            return {
                "deleted": False,
                "node_id": node_id,
                "parent_id": parent_node.id,
                "deleted_node_ids": [],
                "review": self._model_dump(review),
                "frontier": self._frontier_snapshot(),
                "steering_route_focus": steering_route_focus,
            }

        deleted_nodes = self._collect_subtree_nodes(target_node)
        deleted_node_ids = [node.id for node in deleted_nodes]
        parent_node.children = [child for child in parent_node.children if child.id != node_id]

        self._frontier = [
            entry for entry in self._frontier if entry["node"].id not in deleted_node_ids
        ]
        self._expanded_node_ids = [
            expanded_id for expanded_id in self._expanded_node_ids if expanded_id not in deleted_node_ids
        ]

        self._resync_runtime_state()
        self._expansion_log.append(
            {
                "event": "delete-node",
                "requested_by": requested_by,
                "node_id": node_id,
                "parent_id": parent_node.id,
                "deleted_node_ids": deleted_node_ids,
                "review": self._model_dump(review),
                "frontier_size_after": len(self._frontier),
            }
        )

        return {
            "deleted": True,
            "node_id": node_id,
            "parent_id": parent_node.id,
            "deleted_node_ids": deleted_node_ids,
            "review": self._model_dump(review),
            "frontier": self._frontier_snapshot(),
            "steering_route_focus": steering_route_focus,
        }

    def apply_steering_prompt(
        self,
        *,
        parent_node_id: str,
        prompt: str,
        requested_by: str = "frontend",
        source_deleted_node_ids: Optional[list[str]] = None,
        route_focus_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.root_node is None:
            raise RuntimeError("Cannot steer before the tree has been built.")

        normalized_prompt = str(prompt or "").strip()
        if not normalized_prompt:
            raise ValueError("Steering prompt must be provided.")

        parent_node = self._node_index.get(parent_node_id)
        if parent_node is None:
            raise KeyError(f"Unknown parent node id: {parent_node_id}")

        parent_depth = self._node_depth(parent_node)
        if not self._can_expand_below_depth(parent_depth):
            result = {
                "applied": False,
                "parent_id": parent_node_id,
                "prompt": normalized_prompt,
                "reason": "selected parent is already at max tree depth",
            }
            self._expansion_log.append(
                {
                    "event": "steer-node",
                    "requested_by": requested_by,
                    "parent_id": parent_node_id,
                    "applied": False,
                    "reason": result["reason"],
                    "frontier_size_after": len(self._frontier),
                }
            )
            return result

        steer_context = self._build_steering_frontier_context(
            parent_node=parent_node,
            prompt=normalized_prompt,
            parent_depth=parent_depth,
            requested_by=requested_by,
            source_deleted_node_ids=list(source_deleted_node_ids or []),
            route_focus_override=route_focus_override,
        )
        try:
            current_priority = float(parent_node.known_vars.get("expansion_priority", 1.0))
        except (TypeError, ValueError):
            current_priority = 1.0
        parent_node.known_vars["expansion_priority"] = max(current_priority, 2.0)
        self._frontier.append(
            {
                "node": parent_node,
                "problem_context": steer_context,
                "depth": parent_depth,
            }
        )
        retained_ids = self._rebalance_frontier()
        self._refresh_selection_metadata(self.root_node, retained_ids)
        parent_node.known_vars["selected_for_frontier"] = parent_node.id in retained_ids
        parent_node.known_vars["last_steering_prompt"] = normalized_prompt

        result = {
            "applied": parent_node.id in retained_ids,
            "parent_id": parent_node_id,
            "prompt": normalized_prompt,
            "frontier_size_after": len(self._frontier),
        }
        self._expansion_log.append(
            {
                "event": "steer-node",
                "requested_by": requested_by,
                "parent_id": parent_node_id,
                "source_deleted_node_ids": list(source_deleted_node_ids or []),
                "prompt": normalized_prompt,
                "applied": result["applied"],
                "frontier_size_after": len(self._frontier),
            }
        )
        self._set_run_state(status="ready", phase="steering-ready", last_error="")
        return result

    def _build_node(self, parent_node: Optional[ToTNode], problem_context: dict[str, Any]) -> ToTNode:
        fsm = NodeBuilderFSM(
            parent_node=parent_node,
            problem_context=problem_context,
            max_reflections=self.max_reflections,
            backend_adapter=self._make_backend_adapter(problem_context),
        )
        node = fsm.run()
        route_focus = problem_context.get("route_focus")
        if isinstance(route_focus, dict):
            route_family = str(route_focus.get("route_family") or route_focus.get("label") or "").strip()
            if route_family and not str(node.known_vars.get("route_family", "")).strip():
                node.known_vars["route_family"] = route_family
            for key in ("correction_mode", "correction_target"):
                value = str(route_focus.get(key, "")).strip()
                if value and not str(node.known_vars.get(key, "")).strip():
                    node.known_vars[key] = value
            slot = str(route_focus.get("slot", "")).strip()
            if slot and not str(node.known_vars.get("distributed_reasoning_slot", "")).strip():
                node.known_vars["distributed_reasoning_slot"] = slot
        operator_steering = problem_context.get("operator_steering")
        if isinstance(operator_steering, dict) and operator_steering:
            node.known_vars["operator_steering"] = dict(operator_steering)
            prompt = str(operator_steering.get("prompt", "")).strip()
            if prompt:
                node.known_vars["operator_steering_prompt"] = prompt
        meta_task_progress = problem_context.get("meta_task_progress")
        if isinstance(meta_task_progress, dict):
            selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
            if selected_route_family and not str(node.known_vars.get("route_family", "")).strip():
                node.known_vars["route_family"] = selected_route_family
            for progress_key, known_var_key in (
                ("selected_correction_mode", "correction_mode"),
                ("selected_correction_target", "correction_target"),
            ):
                value = str(meta_task_progress.get(progress_key, "")).strip()
                if value and not str(node.known_vars.get(known_var_key, "")).strip():
                    node.known_vars[known_var_key] = value
        self._node_index[node.id] = node
        self._append_node_events(node)
        return node

    def _append_node_events(self, node: ToTNode) -> None:
        raw_events = node.known_vars.get("node_event_log")
        if not isinstance(raw_events, list):
            return

        for item in raw_events:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry.setdefault("node_id", node.id)
            entry.setdefault("parent_id", node.parent_id)
            self._expansion_log.append(entry)

    def _make_backend_adapter(self, problem_context: dict[str, Any]) -> Optional[ReasoningBackendAdapter]:
        if self.backend_adapter_factory is None:
            return None
        return self.backend_adapter_factory(problem_context)

    def _initialize_root_node(self, root_node: ToTNode) -> None:
        signature = self._compute_state_signature(root_node)
        root_node.known_vars["state_signature"] = signature
        root_node.known_vars["diversity_key"] = self._compute_diversity_key(root_node)
        root_node.known_vars.setdefault("scheduler_action", "root")
        if root_node.status == NodeStatus.ACTIVE:
            self._signature_registry[signature] = root_node.id

    def _extract_child_contexts(self, problem_context: dict[str, Any]) -> list[dict[str, Any]]:
        payload = problem_context.get(self.children_key, [])
        if not isinstance(payload, list):
            payload = []
        explicit_children = [dict(item) for item in payload if isinstance(item, dict)]
        if explicit_children:
            return explicit_children
        return self._synthesize_meta_task_child_contexts(problem_context)

    def _synthesize_meta_task_child_contexts(self, problem_context: dict[str, Any]) -> list[dict[str, Any]]:
        if any(key in problem_context for key in ("proposal", "calculation", "evaluation", "reflection")):
            return []

        problem_statement = str(problem_context.get("problem_statement", "")).strip()
        meta_task = problem_context.get("meta_task")
        meta_task_progress = problem_context.get("meta_task_progress")
        if not problem_statement:
            return []
        if not isinstance(meta_task, dict) or not meta_task:
            return []
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return []

        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        if not step_ordering:
            return []

        try:
            current_step_index = int(meta_task_progress.get("current_step_index", 0))
        except (TypeError, ValueError):
            current_step_index = 0
        current_step_index = max(0, min(current_step_index, len(step_ordering) - 1))
        if current_step_index >= len(step_ordering) - 1:
            return []

        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        route_focus = problem_context.get("route_focus")
        if not selected_route_family and isinstance(route_focus, dict):
            selected_route_family = str(
                route_focus.get("route_family") or route_focus.get("label") or ""
            ).strip()

        raw_route_options = meta_task.get("route_options", [])
        if current_step_index == 0 and not selected_route_family and isinstance(raw_route_options, list):
            route_options = [dict(item) for item in raw_route_options if isinstance(item, dict) and item]
            if route_options:
                child_contexts: list[dict[str, Any]] = []
                route_surface_budget = self._route_surface_budget(problem_context)
                for index, route_option in enumerate(route_options[:route_surface_budget]):
                    child_context = deepcopy(problem_context)
                    child_context.pop(self.children_key, None)
                    child_context["meta_task"] = dict(meta_task)
                    child_context["meta_task_progress"] = self._build_meta_task_progress(
                        meta_task,
                        step_index=current_step_index,
                    )
                    route_family = str(route_option.get("route_family") or route_option.get("label") or "").strip()
                    route_label = str(route_option.get("label") or route_family or f"route {index + 1}").strip()
                    route_guidance = str(route_option.get("guidance", "")).strip()
                    route_focus = dict(route_option)
                    route_focus.setdefault("label", route_label)
                    route_focus.setdefault("route_family", route_family or route_label)
                    route_focus.setdefault("slot", str(index))
                    child_context["route_focus"] = route_focus
                    child_progress = dict(child_context["meta_task_progress"])
                    child_progress["selected_route_family"] = route_focus["route_family"]
                    child_progress["distributed_reasoning_slot"] = str(index)
                    child_progress["current_step"] = f"route-local scan: {route_focus['route_family']}"
                    correction_mode = str(route_focus.get("correction_mode", "")).strip()
                    if correction_mode:
                        child_progress["selected_correction_mode"] = correction_mode
                    correction_target = str(route_focus.get("correction_target", "")).strip()
                    if correction_target:
                        child_progress["selected_correction_target"] = correction_target
                    refined_guidance = self._build_route_strategy_scan_guidance(route_focus)
                    if route_guidance:
                        refined_guidance = f"{refined_guidance} Route-specific focus: {route_guidance}".strip()
                    correction_guidance_parts: list[str] = []
                    if correction_mode:
                        correction_guidance_parts.append(
                            f"use the {correction_mode} correction framing"
                        )
                    if correction_target:
                        correction_guidance_parts.append(
                            f"treat {correction_target} as the active correction quantity"
                        )
                    if correction_guidance_parts:
                        refined_guidance = (
                            f"{refined_guidance} Correction-specific focus: {'; '.join(correction_guidance_parts)}."
                        ).strip()
                    child_progress["current_step_guidance"] = refined_guidance
                    child_context["meta_task_progress"] = child_progress
                    child_context["auto_generated_child"] = True
                    child_contexts.append(child_context)
                if child_contexts:
                    return child_contexts

        divergent_contexts = self._synthesize_route_local_divergent_child_contexts(
            problem_context=problem_context,
            meta_task=meta_task,
            meta_task_progress=meta_task_progress,
            step_ordering=step_ordering,
            current_step_index=current_step_index,
            selected_route_family=selected_route_family,
        )
        if divergent_contexts:
            return divergent_contexts

        child_context = self._build_progress_child_context(
            problem_context=problem_context,
            meta_task=meta_task,
            meta_task_progress=meta_task_progress,
            step_index=current_step_index + 1,
        )
        child_context["auto_generated_child"] = True
        return [child_context]

    def _build_progress_child_context(
        self,
        *,
        problem_context: dict[str, Any],
        meta_task: dict[str, Any],
        meta_task_progress: dict[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        child_context = deepcopy(problem_context)
        child_context.pop(self.children_key, None)
        child_context["meta_task"] = dict(meta_task)
        child_context["meta_task_progress"] = self._build_meta_task_progress(
            meta_task,
            step_index=step_index,
        )
        inherited_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        if inherited_route_family:
            child_context["meta_task_progress"]["selected_route_family"] = inherited_route_family
        inherited_slot = str(meta_task_progress.get("distributed_reasoning_slot", "")).strip()
        if inherited_slot:
            child_context["meta_task_progress"]["distributed_reasoning_slot"] = inherited_slot
        inherited_correction_mode = str(meta_task_progress.get("selected_correction_mode", "")).strip()
        if inherited_correction_mode:
            child_context["meta_task_progress"]["selected_correction_mode"] = inherited_correction_mode
        inherited_correction_target = str(meta_task_progress.get("selected_correction_target", "")).strip()
        if inherited_correction_target:
            child_context["meta_task_progress"]["selected_correction_target"] = inherited_correction_target
        return child_context

    def _synthesize_route_local_divergent_child_contexts(
        self,
        *,
        problem_context: dict[str, Any],
        meta_task: dict[str, Any],
        meta_task_progress: dict[str, Any],
        step_ordering: list[str],
        current_step_index: int,
        selected_route_family: str,
    ) -> list[dict[str, Any]]:
        if not selected_route_family:
            return []
        next_step_index = current_step_index + 1
        if next_step_index >= len(step_ordering):
            return []

        surface_budget = self._route_surface_budget(problem_context)
        if surface_budget <= 1:
            return []

        route_focus = problem_context.get("route_focus")
        route_focus = dict(route_focus) if isinstance(route_focus, dict) else {}
        route_focus = self._merge_matching_route_option(
            meta_task=meta_task,
            selected_route_family=selected_route_family,
            route_focus=route_focus,
        )
        variants = self._build_route_local_divergence_variants(
            route_family=selected_route_family,
            route_focus=route_focus,
            meta_task_progress=meta_task_progress,
            next_step=step_ordering[next_step_index],
        )

        child_contexts: list[dict[str, Any]] = []
        for variant_index, variant in enumerate(variants[:surface_budget]):
            child_context = self._build_progress_child_context(
                problem_context=problem_context,
                meta_task=meta_task,
                meta_task_progress=meta_task_progress,
                step_index=next_step_index,
            )
            child_progress = dict(child_context.get("meta_task_progress", {}))
            child_progress["selected_route_family"] = selected_route_family
            child_progress["current_step_guidance"] = str(variant["guidance"])
            child_progress["distributed_reasoning_slot"] = self._build_divergent_slot(
                meta_task_progress=meta_task_progress,
                route_focus=route_focus,
                next_step_index=next_step_index,
                variant_index=variant_index,
            )
            child_progress["branch_variant"] = str(variant["label"])
            child_progress["selected_correction_mode"] = str(variant["correction_mode"])
            child_progress["selected_correction_target"] = str(variant["correction_target"])

            child_route_focus = dict(route_focus)
            child_route_focus.setdefault("route_family", selected_route_family)
            child_route_focus["slot"] = child_progress["distributed_reasoning_slot"]
            child_route_focus["branch_variant"] = str(variant["label"])
            child_route_focus["correction_mode"] = str(variant["correction_mode"])
            child_route_focus["correction_target"] = str(variant["correction_target"])
            child_route_focus["guidance"] = str(variant["guidance"])

            child_context["meta_task_progress"] = child_progress
            child_context["route_focus"] = child_route_focus
            child_context["branch_divergence"] = {
                "variant": str(variant["label"]),
                "route_family": selected_route_family,
                "step_index": next_step_index,
            }
            child_context["auto_generated_child"] = True
            child_contexts.append(child_context)
        return child_contexts

    def _merge_matching_route_option(
        self,
        *,
        meta_task: dict[str, Any],
        selected_route_family: str,
        route_focus: dict[str, Any],
    ) -> dict[str, Any]:
        merged_focus = dict(route_focus)
        for item in meta_task.get("route_options", []):
            if not isinstance(item, dict):
                continue
            route_family = str(item.get("route_family") or item.get("label") or "").strip()
            if route_family != selected_route_family:
                continue
            merged = dict(item)
            merged.update(merged_focus)
            return merged
        return merged_focus

    def _build_divergent_slot(
        self,
        *,
        meta_task_progress: dict[str, Any],
        route_focus: dict[str, Any],
        next_step_index: int,
        variant_index: int,
    ) -> str:
        base_slot = str(
            meta_task_progress.get("distributed_reasoning_slot")
            or route_focus.get("slot")
            or route_focus.get("route_family")
            or "route"
        ).strip()
        return f"{base_slot}.{next_step_index}.{variant_index}"

    def _build_route_local_divergence_variants(
        self,
        *,
        route_family: str,
        route_focus: dict[str, Any],
        meta_task_progress: dict[str, Any],
        next_step: str,
    ) -> list[dict[str, str]]:
        inherited_mode = str(
            meta_task_progress.get("selected_correction_mode")
            or route_focus.get("correction_mode")
            or "route-local continuation"
        ).strip()
        inherited_target = str(
            meta_task_progress.get("selected_correction_target")
            or route_focus.get("correction_target")
            or "active route detail"
        ).strip()
        route_label = str(route_focus.get("label") or route_family or "selected route").strip()

        variants = [
            {
                "label": "route continuation",
                "correction_mode": inherited_mode,
                "correction_target": inherited_target,
                "guidance": (
                    f"Continue only the {route_label} branch for the next checkpoint: {next_step}. "
                    f"Keep the current {inherited_mode} framing and refine exactly one local detail."
                ),
            },
            {
                "label": "alternative closure",
                "correction_mode": f"alternative closure for {route_family}",
                "correction_target": inherited_target,
                "guidance": (
                    f"Fork the {route_label} branch by testing one alternative closure for {inherited_target}. "
                    "Do not merge it back into the main continuation yet."
                ),
            },
            {
                "label": "boundary or constraint",
                "correction_mode": "constraint-first refinement",
                "correction_target": "active boundary or constraint",
                "guidance": (
                    f"Fork the {route_label} branch around one boundary condition, admissibility rule, or constraint needed for {next_step}."
                ),
            },
            {
                "label": "limiting regime",
                "correction_mode": "limiting-regime refinement",
                "correction_target": "dominant regime or scale",
                "guidance": (
                    f"Fork the {route_label} branch around one limiting case, scale estimate, or dominant regime before solving {next_step}."
                ),
            },
            {
                "label": "variable grounding",
                "correction_mode": "variable-grounding refinement",
                "correction_target": "missing variable or relation",
                "guidance": (
                    f"Fork the {route_label} branch around one missing variable, relation, or parameter definition required by {next_step}."
                ),
            },
            {
                "label": "consistency check",
                "correction_mode": "consistency-check refinement",
                "correction_target": "local consistency condition",
                "guidance": (
                    f"Fork the {route_label} branch around one dimensional, sign, limit, or consistency check for {next_step}."
                ),
            },
        ]

        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for variant in variants:
            key = (
                str(variant["label"]),
                str(variant["correction_mode"]),
                str(variant["correction_target"]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)
        return deduped

    def _build_steering_frontier_context(
        self,
        *,
        parent_node: ToTNode,
        prompt: str,
        parent_depth: int,
        requested_by: str,
        source_deleted_node_ids: list[str],
        route_focus_override: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        base_context = self._base_steering_problem_context()
        meta_task = base_context.get("meta_task")
        if not isinstance(meta_task, dict) or not meta_task:
            meta_task = self._build_default_steering_meta_task()

        route_focus = dict(route_focus_override) if isinstance(route_focus_override, dict) else {}
        if not route_focus:
            route_focus = self._route_focus_from_node(parent_node)
        route_family = str(route_focus.get("route_family") or route_focus.get("label") or "operator-steered").strip()
        route_focus.setdefault("label", route_family)
        route_focus.setdefault("route_family", route_family)
        route_focus["branch_variant"] = "operator-steered"
        route_focus["correction_mode"] = "operator steering"
        route_focus["correction_target"] = self._truncate_text(prompt, 120)
        route_focus["guidance"] = self._build_steering_guidance(prompt)

        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        if step_ordering:
            child_step_index = max(0, min(parent_depth, len(step_ordering) - 1))
        else:
            child_step_index = 0

        child_context = deepcopy(base_context)
        child_context.pop(self.children_key, None)
        child_context["meta_task"] = dict(meta_task)
        child_context["meta_task_progress"] = self._build_meta_task_progress(
            meta_task,
            step_index=child_step_index,
        )
        child_context["meta_task_progress"]["selected_route_family"] = route_focus["route_family"]
        child_context["meta_task_progress"]["distributed_reasoning_slot"] = self._build_steering_slot(
            parent_node=parent_node,
            child_step_index=child_step_index,
        )
        child_context["meta_task_progress"]["branch_variant"] = "operator-steered"
        child_context["meta_task_progress"]["selected_correction_mode"] = "operator steering"
        child_context["meta_task_progress"]["selected_correction_target"] = self._truncate_text(prompt, 120)
        child_context["meta_task_progress"]["current_step_guidance"] = self._build_steering_guidance(prompt)
        child_context["route_focus"] = route_focus
        child_context["steering_prompt"] = prompt
        child_context["operator_steering"] = {
            "prompt": prompt,
            "requested_by": requested_by,
            "parent_id": parent_node.id,
            "source_deleted_node_ids": list(source_deleted_node_ids),
        }
        child_context["thought_step"] = f"Operator-steered branch: {self._truncate_text(prompt, 180)}"
        child_context["auto_generated_child"] = True

        parent_context = deepcopy(base_context)
        parent_context.pop("proposal", None)
        parent_context.pop("calculation", None)
        parent_context.pop("evaluation", None)
        parent_context.pop("reflection", None)
        parent_context["meta_task"] = dict(meta_task)
        parent_context[self.children_key] = [child_context]
        parent_context["steering_prompt"] = prompt
        parent_context["operator_steering"] = dict(child_context["operator_steering"])
        return parent_context

    def _base_steering_problem_context(self) -> dict[str, Any]:
        base_context = deepcopy(self.root_problem_context)
        for key in ("children", "proposal", "calculation", "evaluation", "reflection", "orchestrator_task"):
            base_context.pop(key, None)
        return base_context

    def _build_default_steering_meta_task(self) -> dict[str, Any]:
        return {
            "objective": "Continue the tree from an operator steering prompt.",
            "minimal_subproblems": [
                "apply operator steering",
                "refine the replacement branch",
                "verify local consistency",
            ],
            "step_ordering": [
                "apply operator steering",
                "refine the replacement branch",
                "verify local consistency",
            ],
            "first_step": "apply operator steering",
            "completion_signals": [
                "operator steering applied",
                "replacement branch explored",
                "local consistency checked",
            ],
            "route_options": [],
            "step_blueprints": [],
        }

    def _build_steering_guidance(self, prompt: str) -> str:
        return (
            f"Operator steering prompt: {self._truncate_text(prompt, 320)}. "
            "Continue from the parent branch, avoid recreating the deleted subtree, and make exactly one local replacement step."
        )

    def _build_steering_slot(self, *, parent_node: ToTNode, child_step_index: int) -> str:
        base_slot = str(
            parent_node.known_vars.get("distributed_reasoning_slot")
            or parent_node.known_vars.get("route_family")
            or parent_node.id
        ).strip()
        return f"{base_slot}.steer.{child_step_index}.{len(self._expansion_log)}"

    def _route_focus_from_node(self, node: ToTNode) -> dict[str, Any]:
        route_focus: dict[str, Any] = {}
        for source_key, target_key in (
            ("route_family", "route_family"),
            ("correction_mode", "correction_mode"),
            ("correction_target", "correction_target"),
            ("distributed_reasoning_slot", "slot"),
        ):
            value = str(node.known_vars.get(source_key, "")).strip()
            if value:
                route_focus[target_key] = value
        if node.used_models:
            route_focus["governing_models"] = [str(item) for item in node.used_models if str(item).strip()]
        if route_focus.get("route_family") and not route_focus.get("label"):
            route_focus["label"] = str(route_focus["route_family"])
        return route_focus

    def _node_depth(self, node: ToTNode) -> int:
        depth = 0
        current = node
        while current.parent_id is not None:
            parent = self._node_index.get(current.parent_id)
            if parent is None:
                break
            depth += 1
            current = parent
        return depth

    def _truncate_text(self, text: str, limit: int) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 1)].rstrip() + "..."

    def _build_route_strategy_scan_guidance(self, route_focus: dict[str, Any]) -> str:
        route_family = str(route_focus.get("route_family") or route_focus.get("label") or "this").strip()
        return (
            f"Stay at planning level and look only at the {route_family} route. "
            "Make one tiny route-local step only, keep every other route deferred, and do not solve the final answer yet."
        )

    def _build_meta_task_progress(self, meta_task: dict[str, Any], *, step_index: int) -> dict[str, Any]:
        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        first_step = str(meta_task.get("first_step", "")).strip()
        if step_ordering:
            normalized_index = max(0, min(step_index, len(step_ordering) - 1))
            current_step = step_ordering[normalized_index]
            previous_steps = step_ordering[:normalized_index]
            remaining_steps = step_ordering[normalized_index + 1 :]
        else:
            normalized_index = max(0, step_index)
            current_step = first_step
            previous_steps = []
            remaining_steps = []

        phase = "strategy_scan" if normalized_index == 0 else "incremental_refinement"
        if phase == "strategy_scan":
            current_step_guidance = self.META_TASK_STRATEGY_SCAN_GUIDANCE
        else:
            current_step_guidance = (
                f"Refine only the current subproblem: {current_step}. "
                "Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pending fixes deferred."
            )

        return {
            "current_step_index": normalized_index,
            "current_step": current_step,
            "current_step_guidance": current_step_guidance,
            "previous_steps": previous_steps,
            "remaining_steps": remaining_steps,
            "total_steps": len(step_ordering),
            "phase": phase,
            "is_terminal_step": bool(step_ordering) and normalized_index >= len(step_ordering) - 1,
            "route_options": [dict(item) for item in meta_task.get("route_options", []) if isinstance(item, dict)],
            "step_blueprints": [dict(item) for item in meta_task.get("step_blueprints", []) if isinstance(item, dict)],
        }

    def _apply_scheduler_controls(self, node: ToTNode, parent_node: ToTNode) -> Optional[str]:
        signature = self._compute_state_signature(node)
        node.known_vars["state_signature"] = signature
        node.known_vars["diversity_key"] = self._compute_diversity_key(node)

        ancestor_signature_map = self._ancestor_signature_map(parent_node)
        if signature in ancestor_signature_map:
            node.known_vars["merged_into_node_id"] = ancestor_signature_map[signature]
            node.known_vars["suppressed_by_scheduler"] = "loop"
            return "suppressed-loop"

        if node.status != NodeStatus.ACTIVE:
            return None

        canonical_id = self._signature_registry.get(signature)
        if canonical_id is not None and canonical_id != node.id:
            canonical_node = self._node_index.get(canonical_id)
            node.known_vars["merged_into_node_id"] = canonical_id
            node.known_vars["suppressed_by_scheduler"] = "duplicate"
            if canonical_node is not None:
                canonical_node.known_vars.setdefault("merged_duplicate_node_ids", []).append(node.id)
                canonical_node.known_vars.setdefault("merged_duplicate_parent_ids", []).append(parent_node.id)
                canonical_node.known_vars["merged_duplicate_count"] = len(
                    canonical_node.known_vars.get("merged_duplicate_node_ids", [])
                )
                canonical_node.score = max(canonical_node.score, node.score)
                canonical_priority = self._node_priority(canonical_node)
                duplicate_priority = self._node_priority(node)
                canonical_node.known_vars["expansion_priority"] = max(
                    canonical_priority,
                    duplicate_priority,
                )
            return "merged-duplicate"

        self._signature_registry[signature] = node.id
        return None

    def _compute_state_signature(self, node: ToTNode) -> str:
        signature_payload = {
            "equations": sorted(str(item) for item in node.equations),
            "used_models": sorted(str(item) for item in node.used_models),
            "quantities": {str(key): value for key, value in node.quantities.items()},
            "boundary_conditions": {
                str(key): value for key, value in node.boundary_conditions.items()
            },
        }
        route_family = str(node.known_vars.get("route_family", "")).strip()
        correction_mode = str(node.known_vars.get("correction_mode", "")).strip()
        correction_target = str(node.known_vars.get("correction_target", "")).strip()
        distributed_reasoning_slot = str(node.known_vars.get("distributed_reasoning_slot", "")).strip()
        if route_family or correction_mode or correction_target or distributed_reasoning_slot:
            signature_payload["route_family"] = route_family
            signature_payload["correction_mode"] = correction_mode
            signature_payload["correction_target"] = correction_target
            signature_payload["distributed_reasoning_slot"] = distributed_reasoning_slot
        return _stable_hash(signature_payload)

    def _compute_diversity_key(self, node: ToTNode) -> str:
        route_family = str(node.known_vars.get("route_family", "")).strip()
        correction_mode = str(node.known_vars.get("correction_mode", "")).strip()
        correction_target = str(node.known_vars.get("correction_target", "")).strip()
        distributed_reasoning_slot = str(node.known_vars.get("distributed_reasoning_slot", "")).strip()
        boundary_axes = []
        for key in node.boundary_conditions:
            text = str(key)
            boundary_axes.append(text.split("=", 1)[0].strip() if "=" in text else text)
        used_models = sorted(str(item) for item in node.used_models)
        if route_family or correction_mode or correction_target:
            diversity_payload = {
                "route_family": route_family,
                "correction_mode": correction_mode,
                "correction_target": correction_target,
                "slot": distributed_reasoning_slot,
                "used_models": used_models,
            }
        elif used_models or boundary_axes:
            diversity_payload = {
                "used_models": used_models,
                "boundary_axes": sorted(boundary_axes),
            }
        else:
            diversity_payload = {
                "equation_heads": [str(item)[:80] for item in node.equations[:2]],
            }
        return _stable_hash(diversity_payload)[:16]

    def _ancestor_signature_map(self, node: ToTNode) -> dict[str, str]:
        out: dict[str, str] = {}
        current = node
        while current is not None:
            signature = current.known_vars.get("state_signature")
            if signature is not None:
                out[str(signature)] = current.id
            if current.parent_id is None:
                break
            current = self._node_index.get(current.parent_id)
        return out

    def _is_expandable(self, node: ToTNode, problem_context: dict[str, Any]) -> bool:
        return node.status == NodeStatus.ACTIVE and bool(self._extract_child_contexts(problem_context))

    def _can_expand_below_depth(self, node_depth: int) -> bool:
        return node_depth < self.max_tree_depth

    def _node_priority(self, node: ToTNode) -> float:
        if node.status != NodeStatus.ACTIVE:
            return 0.0
        raw_priority = node.known_vars.get("expansion_priority", 1.0)
        try:
            return round(float(raw_priority), 4)
        except (TypeError, ValueError):
            return 0.0

    def _node_ranking_key(self, node: ToTNode) -> tuple[Any, ...]:
        return (
            0 if node.status == NodeStatus.ACTIVE else 1,
            -self._node_priority(node),
            -float(node.score),
            len(node.reflection_history),
            len(node.equations),
            node.id,
        )

    def _frontier_entry_key(self, entry: dict[str, Any]) -> tuple[Any, ...]:
        return (-int(entry["depth"]), *self._node_ranking_key(entry["node"]))

    def _route_surface_budget(self, problem_context: Optional[dict[str, Any]] = None) -> int:
        if self.expansion_budget - len(self._expanded_node_ids) <= 0:
            return 0
        if self._is_root_strategy_scan_route_surface(problem_context):
            return max(1, self.max_frontier_size)
        return max(
            1,
            min(
                self.max_children_per_expansion,
                self.max_frontier_size,
            ),
        )

    def _frontier_candidate_budget(self, problem_context: Optional[dict[str, Any]] = None) -> int:
        if self._is_root_strategy_scan_route_surface(problem_context):
            return max(1, self.max_frontier_size)
        # Cap only the sibling slice for this expansion. Frontier capacity is
        # enforced later in _rebalance_frontier so lower-scored but distinct
        # route families or correction modes can still compete for retention.
        return max(1, self.max_children_per_expansion)

    def _is_root_strategy_scan_route_surface(self, problem_context: Optional[dict[str, Any]]) -> bool:
        if not isinstance(problem_context, dict) or not problem_context:
            return False
        meta_task = problem_context.get("meta_task")
        meta_task_progress = problem_context.get("meta_task_progress")
        if not isinstance(meta_task, dict) or not isinstance(meta_task_progress, dict):
            return False
        raw_route_options = meta_task.get("route_options", [])
        if not isinstance(raw_route_options, list) or not any(isinstance(item, dict) and item for item in raw_route_options):
            return False
        try:
            current_step_index = int(meta_task_progress.get("current_step_index", 0))
        except (TypeError, ValueError):
            current_step_index = 0
        if current_step_index != 0:
            return False
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        if selected_route_family:
            return False
        route_focus = problem_context.get("route_focus")
        if isinstance(route_focus, dict) and any(str(route_focus.get(key, "")).strip() for key in ("route_family", "label")):
            return False
        return True

    def _try_retain_frontier_entry(
        self,
        entry: dict[str, Any],
        retained: list[dict[str, Any]],
        diversity_counts: dict[str, int],
    ) -> bool:
        diversity_key = str(entry["node"].known_vars.get("diversity_key", entry["node"].id))
        if diversity_counts.get(diversity_key, 0) >= self.max_frontier_per_diversity_key:
            entry["node"].known_vars["selected_for_frontier"] = False
            entry["node"].known_vars["suppressed_by_scheduler"] = "diversity-cap"
            return False
        if len(retained) >= self.max_frontier_size:
            entry["node"].known_vars["selected_for_frontier"] = False
            return False
        retained.append(entry)
        diversity_counts[diversity_key] = diversity_counts.get(diversity_key, 0) + 1
        entry["node"].known_vars["suppressed_by_scheduler"] = ""
        return True

    def _rebalance_frontier(self) -> set[str]:
        self._frontier.sort(key=self._frontier_entry_key)
        retained: list[dict[str, Any]] = []
        diversity_counts: dict[str, int] = {}
        deferred_entries: list[dict[str, Any]] = []

        for entry in self._frontier:
            diversity_key = str(entry["node"].known_vars.get("diversity_key", entry["node"].id))
            if diversity_counts.get(diversity_key, 0) == 0 and len(retained) < self.max_frontier_size:
                self._try_retain_frontier_entry(entry, retained, diversity_counts)
                continue
            deferred_entries.append(entry)

        for entry in deferred_entries:
            if len(retained) >= self.max_frontier_size:
                entry["node"].known_vars["selected_for_frontier"] = False
                continue
            self._try_retain_frontier_entry(entry, retained, diversity_counts)

        self._frontier = retained
        return {entry["node"].id for entry in self._frontier}

    def _frontier_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": entry["node"].id,
                "parent_id": entry["node"].parent_id,
                "depth": entry["depth"],
                "priority": self._node_priority(entry["node"]),
                "score": entry["node"].score,
                "status": entry["node"].status.value,
                "state_signature": entry["node"].known_vars.get("state_signature"),
                "diversity_key": entry["node"].known_vars.get("diversity_key"),
                "route_family": entry["node"].known_vars.get("route_family"),
                "correction_mode": entry["node"].known_vars.get("correction_mode"),
                "correction_target": entry["node"].known_vars.get("correction_target"),
                "distributed_reasoning_slot": entry["node"].known_vars.get("distributed_reasoning_slot"),
                "needs_deeper_reasoning": bool(
                    entry["node"].known_vars.get("needs_deeper_reasoning", False)
                ),
                "child_context_count": len(self._extract_child_contexts(entry["problem_context"])),
            }
            for entry in self._frontier
        ]

    def save_state(self, file_path: str) -> None:
        """Persist scheduler state for trusted local resume."""

        snapshot = {
            "root_problem_context": self.root_problem_context,
            "root_node": self.root_node,
            "frontier": self._frontier,
            "expansion_log": self._expansion_log,
            "expanded_node_ids": self._expanded_node_ids,
            "signature_registry": self._signature_registry,
            "max_reflections": self.max_reflections,
            "expansion_budget": self.expansion_budget,
            "max_tree_depth": self.max_tree_depth,
            "max_frontier_size": self.max_frontier_size,
            "max_children_per_expansion": self.max_children_per_expansion,
            "max_frontier_per_diversity_key": self.max_frontier_per_diversity_key,
            "children_key": self.children_key,
        }
        state = TreeSchedulerState(snapshot_blob=_serialize_blob(snapshot))
        path = Path(file_path)
        try:
            payload = state.model_dump_json(indent=2)
        except AttributeError:
            payload = state.json(indent=2)
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def from_state_file(
        cls,
        file_path: str,
        *,
        backend_adapter_factory: Optional[Callable[[dict[str, Any]], ReasoningBackendAdapter]] = None,
        deletion_review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> "ToTTreeScheduler":
        """Restore a scheduler from a saved state snapshot."""

        text = Path(file_path).read_text(encoding="utf-8")
        try:
            state = TreeSchedulerState.model_validate_json(text)
        except AttributeError:
            state = TreeSchedulerState.parse_raw(text)

        snapshot = _deserialize_blob(state.snapshot_blob)
        scheduler = cls(
            root_problem_context=snapshot["root_problem_context"],
            max_reflections=snapshot["max_reflections"],
            expansion_budget=snapshot["expansion_budget"],
            max_tree_depth=snapshot.get("max_tree_depth", 8),
            max_frontier_size=snapshot["max_frontier_size"],
            max_children_per_expansion=snapshot["max_children_per_expansion"],
            max_frontier_per_diversity_key=snapshot["max_frontier_per_diversity_key"],
            children_key=snapshot["children_key"],
            backend_adapter_factory=backend_adapter_factory,
            deletion_review_adapter=deletion_review_adapter,
        )
        scheduler.root_node = snapshot["root_node"]
        scheduler._frontier = snapshot["frontier"]
        scheduler._expansion_log = snapshot["expansion_log"]
        scheduler._expanded_node_ids = snapshot["expanded_node_ids"]
        scheduler._signature_registry = snapshot["signature_registry"]
        if scheduler.root_node is not None:
            scheduler._rebuild_node_index(scheduler.root_node)
        return scheduler

    def _rebuild_node_index(self, node: ToTNode) -> None:
        self._node_index[node.id] = node
        for child in node.children:
            self._rebuild_node_index(child)

    def _node_snapshot(self, node: ToTNode) -> NodeSnapshot:
        return NodeSnapshot(
            id=node.id,
            parent_id=node.parent_id,
            thought_step=node.thought_step,
            equations=list(node.equations),
            known_vars=dict(node.known_vars),
            used_models=list(node.used_models),
            quantities=dict(node.quantities),
            boundary_conditions=dict(node.boundary_conditions),
            status=node.status,
            fsm_state=node.fsm_state,
            result_state=node.result_state,
            score=node.score,
            reflection_history=list(node.reflection_history),
        )

    def _normalize_review_payload(
        self,
        payload: BaseModel | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(payload, BaseModel):
            return self._model_dump(payload)
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError("Deletion review adapter must return a Pydantic model or a dictionary payload.")

    def _build_model(self, model_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
        unexpected = set(payload) - _model_field_names(model_type)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected fields for {model_type.__name__}: {names}")
        try:
            return model_type.model_validate(payload)
        except AttributeError:
            return model_type.parse_obj(payload)

    def _model_dump(self, model: BaseModel) -> dict[str, Any]:
        try:
            return model.model_dump()
        except AttributeError:
            return model.dict()

    def _collect_subtree_nodes(self, node: ToTNode) -> list[ToTNode]:
        collected = [node]
        for child in node.children:
            collected.extend(self._collect_subtree_nodes(child))
        return collected

    def _resync_runtime_state(self) -> None:
        self._node_index = {}
        self._signature_registry = {}
        if self.root_node is None:
            self._frontier = []
            return

        self._clear_scheduler_metadata(self.root_node)
        self._sync_existing_tree(self.root_node, parent_node=None)
        retained_ids = self._rebalance_frontier()
        self._refresh_selection_metadata(self.root_node, retained_ids)

    def _clear_scheduler_metadata(self, node: ToTNode) -> None:
        for key in [
            "diversity_key",
            "expandable",
            "merged_duplicate_count",
            "merged_duplicate_node_ids",
            "merged_duplicate_parent_ids",
            "merged_into_node_id",
            "scheduler_action",
            "selected_child_ids",
            "selected_for_frontier",
            "sibling_priority",
            "sibling_ranking",
            "state_signature",
            "suppressed_by_scheduler",
        ]:
            node.known_vars.pop(key, None)
        for child in node.children:
            self._clear_scheduler_metadata(child)

    def _sync_existing_tree(
        self,
        node: ToTNode,
        parent_node: Optional[ToTNode],
    ) -> None:
        self._node_index[node.id] = node
        node.known_vars["state_signature"] = self._compute_state_signature(node)
        node.known_vars["diversity_key"] = self._compute_diversity_key(node)

        if parent_node is None:
            node.known_vars["scheduler_action"] = "root"
            if node.status == NodeStatus.ACTIVE:
                self._signature_registry[node.known_vars["state_signature"]] = node.id
        else:
            scheduler_action = self._apply_scheduler_controls(node, parent_node)
            node.known_vars["scheduler_action"] = scheduler_action or "expanded"

        ranked_children = sorted(node.children, key=self._node_ranking_key)
        sibling_ranking: list[dict[str, Any]] = []
        for rank, child in enumerate(ranked_children, start=1):
            child.known_vars["sibling_rank"] = rank
            child.known_vars["sibling_priority"] = self._node_priority(child)
            self._sync_existing_tree(child, parent_node=node)
            sibling_ranking.append(
                {
                    "node_id": child.id,
                    "rank": rank,
                    "status": child.status.value,
                    "priority": self._node_priority(child),
                    "score": child.score,
                    "expandable": False,
                    "scheduler_action": child.known_vars.get("scheduler_action", "expanded"),
                    "diversity_key": child.known_vars.get("diversity_key"),
                }
            )
        if sibling_ranking:
            node.known_vars["sibling_ranking"] = sibling_ranking

    def _refresh_selection_metadata(self, node: ToTNode, frontier_ids: set[str]) -> None:
        node.known_vars["selected_for_frontier"] = node.id in frontier_ids
        if node.children:
            node.known_vars["selected_child_ids"] = [
                child.id for child in node.children if child.id in frontier_ids
            ]
            sibling_ranking = node.known_vars.get("sibling_ranking", [])
            refreshed_sibling_ranking: list[dict[str, Any]] = []
            for entry in sibling_ranking:
                child = self._node_index.get(entry.get("node_id"))
                if child is None:
                    continue
                updated = dict(entry)
                updated["priority"] = self._node_priority(child)
                updated["score"] = child.score
                updated["scheduler_action"] = child.known_vars.get("scheduler_action", updated.get("scheduler_action", "expanded"))
                updated["expandable"] = child.id in frontier_ids
                refreshed_sibling_ranking.append(updated)
            node.known_vars["sibling_ranking"] = refreshed_sibling_ranking
        for child in node.children:
            self._refresh_selection_metadata(child, frontier_ids)


__all__ = ["ToTTreeScheduler", "TreeSchedulerState"]