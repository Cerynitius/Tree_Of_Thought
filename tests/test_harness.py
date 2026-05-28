import json
import unittest
from io import BytesIO
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError

from fsm import (
    DEFAULT_CHAT_API_URL,
    DEFAULT_MODELING_MODEL,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    DEFAULT_PLANNING_MODEL,
    DEFAULT_REVIEW_MODEL,
    ChatBackendResponseError,
    ChatBackendTransportError,
    DeleteNodeReviewDecision,
    DeleteNodeReviewRequest,
    DeterministicContextBackendAdapter,
    NodeBuilderFSM,
    NodeSnapshot,
    NodeDeletionReviewAdapter,
    NodeResultState,
    NodeStatus,
    LocalChatDeletionReviewAdapter,
    LocalChatDualModelBackendAdapter,
    ProposalRequest,
    ReflectionRequest,
    ReasoningBackendAdapter,
    ToTTreeScheduler,
    ToTNode,
    EvaluationRequest,
    FSMState,
    build_local_chat_adapter_bundle,
)
from skills import PUBLIC_SKILL_NAMES, SKILL_REGISTRY, invoke_skill, search_skills


class InvalidProposalBackendAdapter(ReasoningBackendAdapter):
    name = "invalid-proposal"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        return {
            "thought_step": "invalid extra field",
            "equations": ["eq0"],
            "unexpected": True,
        }

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        return {"score": 8.0}

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        return {"equations": ["eq1"]}


class MetaTaskStepBackendAdapter(ReasoningBackendAdapter):
    name = "meta-task-step"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        progress = dict(request.problem_context.get("meta_task_progress", {}))
        step_index = int(progress.get("current_step_index", 0))
        current_step = str(progress.get("current_step", f"step {step_index}"))
        if step_index == 0:
            return {
                "thought_step": "Compare the route-a path against the segmented segmented route and defer the actual derivation to later refinements.",
                "equations": [
                    "candidate route A: route-balance balance",
                    "candidate route B: segmented route-b chain",
                ],
                "known_vars": {},
                "used_models": ["Route Balance Model", "Segment Balance Model"],
                "quantities": {},
                "boundary_conditions": {},
            }
        return {
            "thought_step": f"Refine only the current subproblem: {current_step}.",
            "equations": [current_step],
            "known_vars": {},
            "used_models": ["Route Balance Model"],
            "quantities": {},
            "boundary_conditions": {},
        }

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        del request
        return {"score": 8.0}

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        del request
        return {"equations": ["reflected_step"]}


class RepeatedChildProposalBackendAdapter(ReasoningBackendAdapter):
    name = "repeated-child-proposal"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        del request
        return {
            "thought_step": "Add the state transition condition x_next = f(x) as the sole state-transition relation for the state-transition route.",
            "equations": ["v = wr"],
            "known_vars": {},
            "used_models": ["State-transition relation"],
            "quantities": {},
            "boundary_conditions": {},
        }

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        del request
        return {"score": 8.0}

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        del request
        return {
            "thought_step": "Add the horizontal update closure a = ug and expose u as the active control parameter for the state-transition route.",
            "equations": ["a = ug"],
            "known_vars": {
                "active_control_parameter": "u",
            },
            "used_models": ["State-transition relation"],
            "quantities": {
                "mu": "control coefficient",
            },
            "boundary_conditions": {},
        }


class RouteFocusedMetaTaskBackendAdapter(MetaTaskStepBackendAdapter):
    name = "route-focused-meta-task"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        progress = dict(request.problem_context.get("meta_task_progress", {}))
        route_focus = dict(request.problem_context.get("route_focus", {}))
        route_family = str(
            route_focus.get("route_family")
            or progress.get("selected_route_family")
            or "generic-route"
        )
        correction_mode = str(
            route_focus.get("correction_mode")
            or progress.get("selected_correction_mode")
            or ""
        )
        step_index = int(progress.get("current_step_index", 0))
        current_step = str(progress.get("current_step", f"step {step_index}"))
        if step_index == 0:
            governing_model = {
                "route_a": "Route Balance Model",
                "route-b": "Segment Balance Model",
            }.get(route_family, "Route Balance Model")
            correction_suffix = f" using the {correction_mode} correction mode" if correction_mode else ""
            return {
                "thought_step": f"Route scan: {route_family} route only{correction_suffix}. Name one governing relation or one decisive assumption and defer every other route.",
                "equations": [f"{route_family}: route-local seed:{correction_mode or 'default'}"],
                "known_vars": {
                    "route_family": route_family,
                    "correction_mode": correction_mode,
                },
                "used_models": [governing_model],
                "quantities": {},
                "boundary_conditions": {},
            }

        governing_model = {
            "route_a": "Route Balance Model",
            "route-b": "Segment Balance Model",
        }.get(route_family, "Route Balance Model")
        correction_suffix = f" using the {correction_mode} correction mode" if correction_mode else ""
        return {
            "thought_step": f"Refine only the current subproblem: {current_step} via the {route_family} route{correction_suffix}.",
            "equations": [f"{route_family}: {current_step}:{correction_mode or 'default'}"],
            "known_vars": {
                "route_family": route_family,
                "correction_mode": correction_mode,
            },
            "used_models": [governing_model],
            "quantities": {},
            "boundary_conditions": {},
        }


class RouteNeutralMetaTaskBackendAdapter(MetaTaskStepBackendAdapter):
    name = "route-neutral-meta-task"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        progress = dict(request.problem_context.get("meta_task_progress", {}))
        step_index = int(progress.get("current_step_index", 0))
        if step_index == 0:
            return {
                "thought_step": "Route scan placeholder. Keep the branch local and defer every other route.",
                "equations": ["route-neutral seed"],
                "known_vars": {},
                "used_models": ["Generic route model"],
                "quantities": {},
                "boundary_conditions": {},
            }
        return {
            "thought_step": "Refine the currently selected route only.",
            "equations": ["route-neutral refinement"],
            "known_vars": {},
            "used_models": ["Generic route model"],
            "quantities": {},
            "boundary_conditions": {},
        }


class WideRouteFocusedMetaTaskBackendAdapter(RouteFocusedMetaTaskBackendAdapter):
    name = "wide-route-focused-meta-task"


class RejectDeleteReviewAdapter(NodeDeletionReviewAdapter):
    name = "reject-delete"

    def review_delete_node(self, request: DeleteNodeReviewRequest) -> dict[str, object]:
        return {
            "approved": False,
            "reason": f"Rejected deletion for {request.target_node.id}",
            "risk_level": "high",
        }


class ApproveDeleteReviewAdapter(NodeDeletionReviewAdapter):
    name = "approve-delete"

    def review_delete_node(self, request: DeleteNodeReviewRequest) -> DeleteNodeReviewDecision:
        return DeleteNodeReviewDecision(
            approved=True,
            reason=f"Approved deletion for {request.target_node.id}",
            risk_level="medium",
        )


class CapturingChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        self.calls.append({
            "url": url,
            "payload": dict(payload),
            "timeout": timeout,
        })
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            return {"output": json.dumps({"compressed": input_payload["value"]})}
        if stage == "meta-analysis":
            return {
                "output": json.dumps(
                    {
                        "objective": "derive a minimal tree plan",
                        "givens": ["problem statement available"],
                        "unknowns": ["target quantity"],
                        "minimal_subproblems": ["identify governing relation", "solve for target quantity"],
                        "step_ordering": ["identify governing relation", "solve for target quantity"],
                        "first_step": "identify governing relation",
                        "completion_signals": ["target quantity expressed in known variables"],
                        "route_options": [
                            {
                                "label": "route-a path",
                                "route_family": "route_a",
                                "governing_models": ["Route Balance Model"],
                                "assumptions": ["losses deferred"],
                                "correction_mode": "lossless baseline first",
                                "correction_target": "deferred correction",
                            },
                            {
                                "label": "route-b route",
                                "route_family": "route-b",
                                "governing_models": ["Segment Balance Model"],
                                "assumptions": ["segment relations analyzed separately"],
                                "correction_mode": "piecewise-relation closure",
                                "correction_target": "segment transition relation",
                            },
                        ],
                        "step_blueprints": [
                            {
                                "label": "identify governing relation",
                                "step_type": "strategy_scan",
                                "target_quantity": "governing relation",
                                "correction_mode": "compare deferred-loss closures",
                            },
                            {
                                "label": "solve for target quantity",
                                "step_type": "incremental_refinement",
                                "target_quantity": "target quantity",
                                "correction_target": "deferred correction",
                            },
                        ],
                    }
                )
            }
        if stage == "orchestrator":
            latest_critique = str(input_payload["request"].get("latest_critique", "")).strip()
            selected_task = (
                f"Address only this critique: {latest_critique}"
                if latest_critique
                else "Identify the governing relation only and defer all derivations."
            )
            guidance = (
                f"Execute only the selected task: {selected_task}"
                if not latest_critique
                else f"Execute only the selected task and nothing else: {selected_task}."
            )
            return {
                "output": json.dumps(
                    {
                        "step_focus": "identify governing relation",
                        "current_step_guidance": guidance,
                        "task_breakdown": [
                            "identify governing relation",
                            "name the deferred correction terms",
                            "leave algebraic derivation for a later step",
                        ],
                        "selected_task": selected_task,
                        "deferred_tasks": [
                            "derive the target quantity",
                            "apply later correction terms",
                        ],
                        "completion_signals": [
                            "one local task completed",
                            "later tasks explicitly deferred",
                        ],
                        "selected_route_family": "route_a",
                        "candidate_tasks": [
                            {
                                "label": selected_task,
                                "route_family": "route_a",
                                "status": "selected",
                                "correction_mode": "lossless baseline first",
                                "correction_target": "deferred correction",
                            },
                            {
                                "label": "Check the route-b route only at planning level",
                                "route_family": "route-b",
                                "status": "candidate",
                                "correction_mode": "piecewise-relation closure",
                                "correction_target": "segment transition relation",
                            },
                        ],
                    }
                )
            }
        if stage == "proposal":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "model next step",
                        "equations": ["eq0"],
                        "known_vars": {},
                        "used_models": ["diffusion"],
                        "quantities": {},
                        "boundary_conditions": {},
                    }
                )
            }
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.8,
                        "variable_grounding": 0.7,
                        "contextual_relevance": 0.9,
                        "simplicity_hint": 0.6,
                        "reason": "looks good",
                        "hard_rule_violations": [],
                    }
                )
            }
        if stage == "reflect":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "refined step",
                        "equations": ["eq1"],
                        "used_models": ["diffusion"],
                        "quantities": {},
                        "boundary_conditions": {},
                    }
                )
            }
        if stage == "delete-review":
            return {
                "output": json.dumps(
                    {
                        "approved": True,
                        "reason": "safe to delete",
                        "risk_level": "low",
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class FlakyChatRequester:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        self.calls += 1
        if self.calls == 1:
            raise URLError("temporary backend outage")
        return {
            "output": json.dumps(
                {
                    "thought_step": "retry success",
                    "equations": ["eq_retry"],
                    "known_vars": {},
                    "used_models": [],
                    "quantities": {},
                    "boundary_conditions": {},
                }
            )
        }


class ModelNotFoundThenFallbackChatRequester:
    def __init__(self, missing_model: str) -> None:
        self.missing_model = str(missing_model)
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        model = str(payload["model"])
        self.calls.append(model)
        if model == self.missing_model:
            body = json.dumps(
                {
                    "error": {
                        "message": f"Invalid model identifier \"{self.missing_model}\".",
                        "type": "invalid_request",
                        "param": "model",
                        "code": "model_not_found",
                    }
                }
            ).encode("utf-8")
            raise HTTPError(
                url=DEFAULT_CHAT_API_URL,
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=BytesIO(body),
            )
        return {
            "output": json.dumps(
                {
                    "thought_step": "fallback success",
                    "equations": ["eq_fallback"],
                    "known_vars": {},
                    "used_models": [],
                    "quantities": {},
                    "boundary_conditions": {},
                }
            )
        }


class InvalidEvaluationThenReviewFallbackChatRequester:
    def __init__(self, *, invalid_model: str, fallback_model: str) -> None:
        self.invalid_model = str(invalid_model)
        self.fallback_model = str(fallback_model)
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        model = str(payload["model"])
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(f"{stage}:{model}")
        if model == self.invalid_model:
            if stage not in {"evaluate", "repair"}:
                raise AssertionError(f"Unexpected stage: {stage}")
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": "The node is locally consistent.",
                        "variable_grounding": "Known variables are referenced correctly.",
                        "contextual_relevance": "The step is on-task for the selected route.",
                        "reason": "invalid light-model payload",
                        "hard_rule_violations": [],
                    }
                )
            }
        if model == self.fallback_model:
            if stage not in {"evaluate", "repair"}:
                raise AssertionError(f"Unexpected stage: {stage}")
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.79,
                        "variable_grounding": 0.73,
                        "contextual_relevance": 0.86,
                        "simplicity_hint": 0.48,
                        "reason": "review-model fallback success",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected model: {model}")


class TimeoutChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        raise TimeoutError("timed out")


class ReviewTimeoutThenProposalFallbackChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        model = str(payload["model"])
        self.calls.append(model)
        if model == "qwen/qwen3-1.7b":
            raise TimeoutError("timed out")
        return {
            "output": json.dumps(
                {
                    "thought_step": "fallback proposal success",
                    "equations": ["eq_fallback_proposal"],
                    "known_vars": {},
                    "used_models": [],
                    "quantities": {},
                    "boundary_conditions": {},
                }
            )
        }


class EvaluationTransportThenReviewFallbackChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        model = str(payload["model"])
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(f"{stage}:{model}")
        if model == "qwen2.5-0.5b-instruct-mlx":
            body = json.dumps(
                {
                    "error": {
                        "message": "The number of tokens to keep from the initial prompt is greater than the context length.",
                        "type": "internal_error",
                        "code": "unknown",
                        "param": None,
                    }
                }
            ).encode("utf-8")
            raise HTTPError(
                url=DEFAULT_CHAT_API_URL,
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=BytesIO(body),
            )
        if model == DEFAULT_REVIEW_MODEL:
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.81,
                        "variable_grounding": 0.77,
                        "contextual_relevance": 0.88,
                        "simplicity_hint": 0.42,
                        "reason": "review-model transport fallback success",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected model: {model}")


class ListContentChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "thought_step": "list content success",
                                        "equations": ["eq_list"],
                                        "known_vars": {},
                                        "used_models": [],
                                        "quantities": {},
                                        "boundary_conditions": {},
                                    }
                                ),
                            }
                        ]
                    }
                }
            ]
        }


class TypoEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.8,
                        "variable_grounding": 0.7,
                        "contextual_relevance": 0.9,
                        "simplity_hint": 0.6,
                        "reason": "looks good",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class BlankSimplicityEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.8,
                        "variable_grounding": 0.7,
                        "contextual_relevance": 0.9,
                        "simplicity_hint": "",
                        "reason": "blank simplicity is acceptable",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class SimplyHintEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.75,
                        "variable_grounding": 0.65,
                        "contextual_relevance": 0.85,
                        "simply_hint": 0.55,
                        "reason": "alternate typo alias",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class SimplificationHintEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.72,
                        "variable_grounding": 0.66,
                        "contextual_relevance": 0.81,
                        "simplification_hint": 0.51,
                        "reason": "longer typo alias",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class ContextalRelevanceEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.71,
                        "variable_grounding": 0.64,
                        "contextal_relevance": 0.82,
                        "reason": "fuzzy relevance alias",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class TextualSimplicityHintEvaluationChatRequester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        stage = json.loads(str(payload["input"]))["stage"]
        self.calls.append(stage)
        if stage == "evaluate":
            return {
                "output": json.dumps(
                    {
                        "domain_consistency": 0.78,
                        "variable_grounding": 0.68,
                        "contextual_relevance": 0.84,
                        "simplicity_hint": "Simplify by assuming theta stays small in the initial pass.",
                        "reason": "textual hint should not fail validation",
                        "hard_rule_violations": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class ReasoningAndMessageChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        return {
            "output": [
                {
                    "type": "reasoning",
                    "content": (
                        "I am considering {\"thought_step\": \"discard this reasoning sample\"} "
                        "before returning the final payload."
                    ),
                },
                {
                    "type": "message",
                    "content": json.dumps(
                        {
                            "thought_step": "message segment success",
                            "equations": ["eq_message"],
                            "known_vars": {},
                            "used_models": [],
                            "quantities": {},
                            "boundary_conditions": {},
                        }
                    ),
                },
            ]
        }


class MixedTextProposalChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        return {
            "output": (
                "Draft branch {\"thought_step\": \"discard this partial draft\"}.\n"
                "Final payload follows.\n"
                "{\"thought_step\": \"mixed text success\", \"equations\": [\"eq_mixed\"], \"known_vars\": {}, \"used_models\": [], \"quantities\": {}, \"boundary_conditions\": {}}"
            )
        }


class MixedTextRepairChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        self.calls.append(dict(payload))
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "proposal":
            return {"output": "This answer is malformed and needs repair."}
        if stage == "repair":
            return {
                "output": (
                    "Repair note {\"thought_step\": \"discard this repair sketch\"}.\n"
                    "Use this object instead.\n"
                    "{\"thought_step\": \"repair mixed text success\", \"equations\": [\"eq_repair_mixed\"], \"known_vars\": {}, \"used_models\": [\"single_step_model\"], \"quantities\": {}, \"boundary_conditions\": {}}"
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class SchemaAdjacentProposalChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        return {
            "output": json.dumps(
                {
                    "thought_step": "coerced proposal",
                    "equations": "eq_schema_adjacent",
                    "known_vars": ["g", "mu"],
                    "used_models": "closure_model",
                    "quantities": ["v_t", "rho_s"],
                    "boundary_conditions": ["small Reynolds number"],
                }
            )
        }


class StructuredMetaAnalysisChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        input_payload = json.loads(str(payload["input"]))
        if input_payload["stage"] != "meta-analysis":
            raise AssertionError(f"Unexpected stage: {input_payload['stage']}")
        return {
            "output": json.dumps(
                {
                    "objective": "derive a broad session plan",
                    "givens": ["problem statement available"],
                    "unknowns": ["target value"],
                    "minimal_subproblems": [
                        {
                            "action": "Compare route families",
                            "description": "Route partition versus route-b route.",
                        },
                        {"action": "Refine the state constraint"},
                    ],
                    "step_ordering": [
                        {
                            "action": "Compare route families",
                            "description": "Route partition versus route-b route.",
                        },
                        {"action": "Refine the state constraint"},
                    ],
                    "first_step": {
                        "action": "Compare route families",
                        "description": "Route partition versus route-b route.",
                    },
                    "completion_signals": [
                        {"action": "Target expressed in known variables"}
                    ],
                }
            )
        }


class RepairingProposalChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        self.calls.append(dict(payload))
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "proposal":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "too many fields",
                        "equations": ["eq_bad"],
                        "known_vars": {},
                        "used_models": [],
                        "quantities": {},
                        "boundary_conditions": {},
                        "unexpected": True,
                    }
                )
            }
        if stage == "repair":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "repaired proposal",
                        "equations": ["eq_repaired"],
                        "known_vars": {},
                        "used_models": ["single_step_model"],
                        "quantities": {},
                        "boundary_conditions": {},
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class EmptyChatRequester:
    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> str:
        del url, payload, timeout
        return "   "


class StaticErrorBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


class OrchestratorOverflowChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del timeout
        self.calls.append({"url": url, "payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            return {"output": json.dumps({"compressed": input_payload["value"]})}
        if stage == "orchestrator":
            error_body = json.dumps(
                {
                    "error": {
                        "message": "The number of tokens to keep from the initial prompt is greater than the context length.",
                        "type": "internal_error",
                    }
                }
            ).encode("utf-8")
            raise HTTPError(url, 500, "context overflow", None, StaticErrorBody(error_body))
        if stage == "proposal":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "fallback proposal",
                        "equations": ["eq_fallback"],
                        "known_vars": {},
                        "used_models": ["route_a"],
                        "quantities": {},
                        "boundary_conditions": {},
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class OrchestratorTransientFailureChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del timeout
        self.calls.append({"url": url, "payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            return {"output": json.dumps({"compressed": input_payload["value"]})}
        if stage == "orchestrator":
            raise URLError("temporary orchestrator outage")
        if stage == "proposal":
            return {
                "output": json.dumps(
                    {
                        "thought_step": "transient fallback proposal",
                        "equations": ["eq_transient_fallback"],
                        "known_vars": {},
                        "used_models": ["route_a"],
                        "quantities": {},
                        "boundary_conditions": {},
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class MetaAnalysisOverflowRetryChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.meta_analysis_calls = 0

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del timeout
        self.calls.append({"url": url, "payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            return {"output": json.dumps({"compressed": input_payload["value"]})}
        if stage != "meta-analysis":
            raise AssertionError(f"Unexpected stage: {stage}")
        self.meta_analysis_calls += 1
        if self.meta_analysis_calls == 1:
            error_body = json.dumps(
                {
                    "error": {
                        "message": "The number of tokens to keep from the initial prompt is greater than the context length.",
                        "type": "internal_error",
                    }
                }
            ).encode("utf-8")
            raise HTTPError(url, 500, "context overflow", None, StaticErrorBody(error_body))
        return {
            "output": json.dumps(
                {
                    "objective": "derive a minimal tree plan",
                    "givens": ["problem statement available"],
                    "unknowns": ["target quantity"],
                    "minimal_subproblems": ["identify governing relation", "solve for target quantity"],
                    "step_ordering": ["identify governing relation", "solve for target quantity"],
                    "first_step": "identify governing relation",
                    "completion_signals": ["target quantity expressed in known variables"],
                    "route_options": [],
                    "step_blueprints": [],
                }
            )
        }


class MetaAnalysisOverflowAlwaysChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del timeout
        self.calls.append({"url": url, "payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            return {"output": json.dumps({"compressed": input_payload["value"]})}
        if stage != "meta-analysis":
            raise AssertionError(f"Unexpected stage: {stage}")
        error_body = json.dumps(
            {
                "error": {
                    "message": "The number of tokens to keep from the initial prompt is greater than the context length.",
                    "type": "internal_error",
                }
            }
        ).encode("utf-8")
        raise HTTPError(url, 500, "context overflow", None, StaticErrorBody(error_body))


class InvalidMetaAnalysisPayloadChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> object:
        del url, timeout
        self.calls.append({"payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "meta-analysis":
            return {"output": "definitely not json"}
        if stage == "repair":
            return {"output": "still not json"}
        raise AssertionError(f"Unexpected stage: {stage}")


class TimeoutMetaAnalysisChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> object:
        del url, timeout
        self.calls.append({"payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "meta-analysis":
            raise TimeoutError("meta-analysis timeout")
        raise AssertionError(f"Unexpected stage: {stage}")


class CompressionAwareMetaAnalysisChatRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, timeout
        self.calls.append({"payload": dict(payload)})
        input_payload = json.loads(str(payload["input"]))
        stage = input_payload["stage"]
        if stage == "compress-context":
            compressed_value = (
                "compressed problem statement"
                if input_payload.get("value_type") == "text"
                else input_payload.get("value")
            )
            return {"output": json.dumps({"compressed": compressed_value})}
        if stage == "meta-analysis":
            return {
                "output": json.dumps(
                    {
                        "objective": "derive a minimal tree plan",
                        "givens": ["problem statement available"],
                        "unknowns": ["target quantity"],
                        "minimal_subproblems": ["identify governing relation", "solve for target quantity"],
                        "step_ordering": ["identify governing relation", "solve for target quantity"],
                        "first_step": "identify governing relation",
                        "completion_signals": ["target quantity expressed in known variables"],
                        "route_options": [],
                        "step_blueprints": [],
                    }
                )
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class SkillInvokerTests(unittest.TestCase):
    def test_registry_matches_public_names(self) -> None:
        self.assertEqual(set(SKILL_REGISTRY), set(PUBLIC_SKILL_NAMES))

    def test_invoke_skill_supports_params_dict(self) -> None:
        result = invoke_skill("partition_function", {"energies": [0, 1]})
        self.assertIn("Z", result)
        self.assertIn("U", result)

    def test_invoke_skill_supports_zero_arg(self) -> None:
        zero_arg_matches = search_skills("matrix", call_style="zero_arg", limit=1)
        self.assertTrue(zero_arg_matches)
        result = invoke_skill(zero_arg_matches[0]["name"])
        self.assertTrue(result)

    def test_invoke_skill_supports_direct_args(self) -> None:
        result = invoke_skill(
            "vector_divergence",
            {
                "args": [[1, 0, 0]],
                "kwargs": {"coord_system": "cartesian"},
            },
        )
        self.assertEqual(result, 0)

    def test_search_skills_finds_topic_matches(self) -> None:
        matches = search_skills("gradient")
        names = {item["name"] for item in matches}
        self.assertIn("vector_gradient", names)

    def test_domain_plugin_bundle_infers_builtin_plugin(self) -> None:
        result = invoke_skill(
            "tot_domain_plugin_bundle",
            {
                "problem_statement": "A general cross-disciplinary system has multiple constraints.",
            },
        )

        self.assertEqual(result["selection_mode"], "fallback")
        plugin_labels = {item["label"] for item in result["selected_plugins"]}
        self.assertIn("General Scientific Reasoning", plugin_labels)
        latex_values = {item["latex"] for item in result["representative_formulas"]}
        self.assertIn(r"f(x) = 0", latex_values)
        self.assertIn("Knowledge scope:", result["prompt_fragment"])

    def test_domain_plugin_bundle_accepts_custom_plugins(self) -> None:
        result = invoke_skill(
            "tot_domain_plugin_bundle",
            {
                "problem_context": {
                    "problem_statement": "Find the equilibrium market price.",
                    "domain_plugins": [
                        {
                            "name": "microeconomics",
                            "label": "Microeconomics",
                            "summary": "Custom plugin for supply-demand equilibrium reasoning.",
                            "knowledge_scope": ["supply curve", "demand curve", "equilibrium condition"],
                            "representative_formulas": [
                                {
                                    "latex": r"Q_s(p) = Q_d(p)",
                                    "meaning": "Market-clearing equilibrium condition.",
                                }
                            ],
                            "route_seed_options": [
                                {
                                    "label": "market-clearing route",
                                    "route_family": "market-clearing",
                                    "governing_models": ["Supply-demand equilibrium"],
                                    "guidance": "Choose one equilibrium relation before adding elasticities or shocks.",
                                    "correction_mode": "equilibrium-first scan",
                                    "correction_target": "market-clearing condition",
                                }
                            ],
                        }
                    ],
                }
            },
        )

        self.assertEqual(result["selection_mode"], "custom")
        self.assertEqual(result["selected_plugins"][0]["label"], "Microeconomics")
        self.assertEqual(result["representative_formulas"][0]["latex"], r"Q_s(p) = Q_d(p)")
        self.assertEqual(result["route_seed_options"][0]["route_family"], "market-clearing")

    def test_domain_plugin_bundle_prefers_explicit_skill_templates(self) -> None:
        result = invoke_skill(
            "tot_domain_plugin_bundle",
            {
                "skill_names": ["partition_function"],
                "problem_statement": "Estimate a aggregate observable.",
            },
        )

        self.assertEqual(result["selection_mode"], "explicit")
        self.assertEqual(result["selected_skills"][0]["skill_name"], "partition_function")
        self.assertEqual(result["recommended_skills"], ["partition_function"])
        latex_values = {item["latex"] for item in result["representative_formulas"]}
        self.assertIn(r"Z = \sum_i e^{-\beta E_i}", latex_values)
        route_families = {item["route_family"] for item in result["route_seed_options"]}
        self.assertEqual(route_families, {"partition-function"})

    def test_domain_plugin_bundle_changes_formulas_when_skill_changes(self) -> None:
        first = invoke_skill(
            "tot_domain_plugin_bundle",
            {
                "domain_plugins": [
                    {
                        "name": "alpha-domain",
                        "label": "Alpha Domain",
                        "representative_formulas": [{"latex": "A(x)=0", "meaning": "alpha relation"}],
                    }
                ]
            },
        )
        second = invoke_skill(
            "tot_domain_plugin_bundle",
            {
                "domain_plugins": [
                    {
                        "name": "beta-domain",
                        "label": "Beta Domain",
                        "representative_formulas": [{"latex": "B(x)=0", "meaning": "beta relation"}],
                    }
                ]
            },
        )

        first_formulas = {item["latex"] for item in first["representative_formulas"]}
        second_formulas = {item["latex"] for item in second["representative_formulas"]}
        self.assertIn("A(x)=0", first_formulas)
        self.assertIn("B(x)=0", second_formulas)
        self.assertNotEqual(first_formulas, second_formulas)

    def test_validation_plugin_bundle_prefers_explicit_skill_validators(self) -> None:
        result = invoke_skill(
            "tot_validation_plugin_bundle",
            {
                "skill_names": ["partition_function"],
            },
        )

        self.assertEqual(result["selection_mode"], "explicit")
        self.assertEqual(result["selected_validators"][0]["skill_name"], "partition_function")
        self.assertIn(
            "beta",
            result["hard_rule_params"]["required_any_context_patterns"],
        )

    def test_validation_plugin_bundle_normalizes_nested_validation_rules(self) -> None:
        result = invoke_skill(
            "tot_validation_plugin_bundle",
            {
                "domain_plugins": [
                    {
                        "name": "microeconomics",
                        "label": "Microeconomics",
                        "validation_rules": {
                            "equations": {
                                "require_any_patterns": ["Q_s", "Q_d"],
                                "forbid_patterns": ["F = m a"],
                            },
                            "models": {
                                "require_exact": ["Supply-demand equilibrium"],
                                "require_any_patterns": ["elasticity", "market-clearing"],
                            },
                            "context": {
                                "require_all_patterns": ["market", "equilibrium"],
                                "forbid_patterns": ["transform"],
                            },
                            "variables": {
                                "require_known": ["p"],
                                "nonzero": ["p"],
                            },
                            "flags": {
                                "require_equations": True,
                                "semantic_boundary_checks": False,
                            },
                            "violations": {
                                "append": ["custom plugin check"],
                            },
                        },
                    }
                ],
            },
        )

        self.assertEqual(result["selection_mode"], "custom")
        self.assertEqual(result["hard_rule_params"]["required_any_equation_patterns"], ["Q_s", "Q_d"])
        self.assertEqual(result["hard_rule_params"]["required_models"], ["Supply-demand equilibrium"])
        self.assertEqual(result["hard_rule_params"]["required_any_model_patterns"], ["elasticity", "market-clearing"])
        self.assertEqual(result["hard_rule_params"]["required_all_context_patterns"], ["market", "equilibrium"])
        self.assertEqual(result["hard_rule_params"]["required_known_vars"], ["p"])
        self.assertEqual(result["hard_rule_params"]["nonzero_var_names"], ["p"])
        self.assertFalse(result["hard_rule_params"]["semantic_boundary_checks"])
        self.assertEqual(result["hard_rule_params"]["custom_violations"], ["custom plugin check"])

    def test_hard_rule_check_validates_models_and_boundary_conditions(self) -> None:
        result = invoke_skill(
            "tot_hard_rule_check",
            {
                "equations": ["u_t = D u_xx"],
                "used_models": ["1D diffusion", "continuum approximation"],
                "boundary_conditions": {"x=0": "u(0,t)=u0", "x=L": "u(L,t)=0"},
                "required_models": ["1D diffusion"],
                "required_model_patterns": ["continuum"],
                "required_boundary_condition_keys": ["x=0"],
                "required_boundary_condition_patterns": ["u(L,t)=0"],
            },
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["violations"], [])

    def test_hard_rule_check_catches_semantic_boundary_violation(self) -> None:
        result = invoke_skill(
            "tot_hard_rule_check",
            {
                "equations": ["u_t = D u_xx"],
                "boundary_conditions": {"x=0": "u(x,t)"},
            },
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "Boundary condition value depends on constrained axis: x=0",
            result["violations"],
        )

    def test_hard_rule_check_uses_selected_skill_validation_plugins(self) -> None:
        result = invoke_skill(
            "tot_hard_rule_check",
            {
                "skill_names": ["partition_function"],
                "equations": ["F = m a"],
            },
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "No context matches any required pattern: partition | beta | ln z | z = | boltzmann",
            result["violations"],
        )
        self.assertEqual(result["checked"]["validation_plugin_selection_mode"], "explicit")
        self.assertEqual(
            result["checked"]["validation_plugins"][0]["skill_names"],
            ["partition_function"],
        )

    def test_hard_rule_check_supports_nested_validation_rule_groups(self) -> None:
        result = invoke_skill(
            "tot_hard_rule_check",
            {
                "domain_plugins": [
                    {
                        "name": "microeconomics",
                        "label": "Microeconomics",
                        "validation_rules": {
                            "equations": {
                                "require_any_patterns": ["Q_s", "Q_d"],
                            },
                            "context": {
                                "require_all_patterns": ["market", "equilibrium"],
                            },
                        },
                    }
                ],
                "equations": ["F = m a"],
                "thought_step": "Use a relation balance.",
            },
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "No equation matches any required pattern: Q_s | Q_d",
            result["violations"],
        )
        self.assertIn("No context matches required pattern: market", result["violations"])
        self.assertIn("No context matches required pattern: equilibrium", result["violations"])
        self.assertEqual(result["checked"]["required_any_equation_patterns"], ["Q_s", "Q_d"])
        self.assertEqual(result["checked"]["required_all_context_patterns"], ["market", "equilibrium"])

    def test_hard_rule_check_uses_thought_step_for_context_patterns(self) -> None:
        result = invoke_skill(
            "tot_hard_rule_check",
            {
                "domain_plugins": [
                    {
                        "name": "microeconomics",
                        "label": "Microeconomics",
                        "validation_rules": {
                            "equations": {
                                "require_any_patterns": ["Q_s", "Q_d"],
                            },
                            "context": {
                                "require_all_patterns": ["market", "equilibrium"],
                            },
                        },
                    }
                ],
                "equations": ["Q_s(p) = Q_d(p)"],
                "thought_step": "Use the market equilibrium condition.",
            },
        )

        self.assertTrue(result["passed"])

    def test_stage_prompt_contract_marks_proposal_as_single_step(self) -> None:
        result = invoke_skill("tot_stage_prompt_contract", {"stage": "proposal"})

        self.assertTrue(result["single_step"])
        self.assertIn("exactly one minimal next-step candidate", result["prompt_fragment"])
        self.assertIn("child must add exactly one explicit local delta beyond the parent", result["prompt_fragment"])
        self.assertEqual(
            result["required_keys"],
            [
                "thought_step",
                "equations",
                "known_vars",
                "used_models",
                "quantities",
                "boundary_conditions",
            ],
        )

    def test_stage_prompt_contract_meta_analysis_mentions_correction_divergence(self) -> None:
        result = invoke_skill("tot_stage_prompt_contract", {"stage": "meta-analysis"})

        self.assertFalse(result["single_step"])
        self.assertIn("alternative correction quantities or closure choices", result["prompt_fragment"])
        self.assertIn("short and atomic", result["prompt_fragment"])
        self.assertIn("correction_mode", result["prompt_fragment"])

    def test_stage_prompt_contract_meta_analysis_injects_plugin_formulas(self) -> None:
        result = invoke_skill(
            "tot_stage_prompt_contract",
            {
                "stage": "meta-analysis",
                "problem_context": {
                    "problem_statement": "A general cross-disciplinary system has multiple constraints.",
                },
            },
        )

        self.assertIn("General Scientific Reasoning", result["prompt_fragment"])
        self.assertIn("Representative LaTeX formulas:", result["prompt_fragment"])
        self.assertIn(r"f(x) = 0", result["prompt_fragment"])

    def test_stage_prompt_contract_meta_analysis_injects_selected_skill_templates(self) -> None:
        result = invoke_skill(
            "tot_stage_prompt_contract",
            {
                "stage": "meta-analysis",
                "problem_context": {
                    "problem_statement": "Estimate a aggregate observable.",
                    "skill_names": ["partition_function"],
                },
            },
        )

        self.assertIn("Selected skill partition_function", result["prompt_fragment"])
        self.assertIn("Formula templates:", result["prompt_fragment"])
        self.assertIn(r"Z = \sum_i e^{-\beta E_i}", result["prompt_fragment"])

    def test_stage_prompt_contract_defines_orchestrator_schema(self) -> None:
        result = invoke_skill("tot_stage_prompt_contract", {"stage": "orchestrator"})

        self.assertFalse(result["single_step"])
        self.assertIn("strictly decompose", result["prompt_fragment"])
        self.assertIn("smallest executable micro tasks", result["prompt_fragment"])
        self.assertIn("do not receive the full problem statement", result["prompt_fragment"])
        self.assertIn("correction_mode", result["prompt_fragment"])
        self.assertEqual(
            result["required_keys"],
            [
                "step_focus",
                "current_step_guidance",
                "task_breakdown",
                "selected_task",
                "deferred_tasks",
                "completion_signals",
            ],
        )

    def test_stage_prompt_contract_marks_evaluation_as_problem_redacted(self) -> None:
        result = invoke_skill("tot_stage_prompt_contract", {"stage": "evaluation"})

        self.assertIn("do not receive the full problem statement", result["prompt_fragment"])
        self.assertIn("currently selected subtask", result["prompt_fragment"])


class CompressionProbeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, *, model: str, system_prompt: str, input_text: str) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "input_text": input_text,
            }
        )
        return {"output": json.dumps({"compressed": "unexpected"})}


class NodeHarnessTests(unittest.TestCase):
    def test_structured_meta_analysis_is_extended_to_minimum_depth(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=CapturingChatRequester())

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Compare route families before refining the constraint.",
                "meta_task": {
                    "objective": "derive a broad session plan",
                    "minimal_subproblems": [
                        {"action": "Compare route families"},
                        {"action": "Refine the state constraint"},
                    ],
                    "step_ordering": [
                        {"action": "Compare route families"},
                        {"action": "Refine the state constraint"},
                    ],
                    "first_step": {"action": "Compare route families"},
                    "completion_signals": [
                        {"action": "Target expressed in known variables"},
                    ],
                },
            }
        )

        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            [
                "Compare route families",
                "Refine the state constraint",
                "identify governing relation",
                "choose one active correction or closure",
                "express the target quantity in known variables",
                "isolate the remaining unknown or boundary condition",
                "introduce one missing constitutive relation or closure",
                "reduce to one solvable local relation",
            ],
        )
        self.assertEqual(prepared["meta_task"]["first_step"], "Compare route families")
        self.assertFalse(prepared["meta_task_progress"]["is_terminal_step"])

    def test_compact_reasoning_request_does_not_call_compression_model(self) -> None:
        client = CompressionProbeClient()
        backend = LocalChatDualModelBackendAdapter(client=client)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Find the target value of a generic process with delayed closure.",
                "task": "Use the modeling model to propose the next reasoning step, then score each step for domain consistency and variable grounding.",
                "notes": [
                    "Branch early across plausible route families before committing to one.",
                    "Keep each node local and atomic.",
                ],
                "known_context": {
                    "objective": "Derive and prune a useful reasoning tree.",
                    "expected_output": "concise, structured, and domain-valid intermediate steps",
                },
            }
        )

        backend._build_compact_reasoning_request(
            stage="proposal",
            request=ProposalRequest(
                attempt_index=0,
                problem_context=prepared,
                current_node=NodeSnapshot(id="node-1"),
            ),
        )

        self.assertEqual(client.calls, [])

    def test_local_chat_backend_repairs_invalid_payload_once(self) -> None:
        requester = RepairingProposalChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["thought_step"], "repaired proposal")
        self.assertEqual(proposal["equations"], ["eq_repaired"])
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(json.loads(str(requester.calls[1]["input"]))["stage"], "repair")

    def test_local_chat_backend_uses_explicit_stage_payloads_without_live_call(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "proposal": {
                "thought_step": "explicit proposal",
                "equations": ["F = m * a"],
            },
            "evaluation": {
                "score": 9.1,
                "reason": "explicit score",
            },
            "reflection": {
                "thought_step": "explicit reflection",
                "equations": ["F_refined = m * a"],
            },
        }

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1"),
            )
        )
        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1"),
            )
        )
        reflection = backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1"),
                latest_critique="try again",
            )
        )

        self.assertEqual(proposal["equations"], ["F = m * a"])
        self.assertEqual(evaluation["score"], 9.1)
        self.assertEqual(reflection["equations"], ["F_refined = m * a"])
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_local_reflection_for_low_scoring_explicit_payloads(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "proposal": {
                "thought_step": "explicit proposal",
                "equations": ["F = m * a"],
            },
            "evaluation": {
                "score": 5.1,
                "reason": "explicit low score",
            },
        }

        reflection = backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="low score"),
                latest_critique="score below threshold",
            )
        )

        self.assertTrue(reflection["known_vars"]["local_model_fallback"])
        self.assertEqual(reflection["known_vars"]["fallback_stage"], "reflection")
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_fallback_reflection_skips_orchestrator_live_call(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            allow_live_model_fallback=True,
        )

        reflection = backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                latest_critique="score below threshold",
                problem_context={
                    "problem_statement": "Repair one low-scoring explicit candidate.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Stay local and revise only one relation.",
                        "phase": "incremental_refinement",
                        "selected_route_family": "benchmark",
                        "is_terminal_step": False,
                    },
                },
                current_node=NodeSnapshot(id="node-1", thought_step="low score"),
            )
        )

        self.assertTrue(reflection["known_vars"]["local_model_fallback"])
        self.assertEqual(reflection["known_vars"]["fallback_stage"], "reflection")
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_prepares_meta_task_locally_and_propagates_to_children(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Find the target value.",
                "children": [{"problem_statement": "Consider the closure relation first."}],
            }
        )

        self.assertIn("meta_task", prepared)
        self.assertIn("meta_task_progress", prepared)
        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(prepared["meta_task_progress"]["current_step_index"], 0)
        self.assertEqual(prepared["meta_task_progress"]["current_step"], "identify governing relation")
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        self.assertIn("strategy space", prepared["meta_task_progress"]["current_step_guidance"])
        self.assertIn("route-local planning claim", prepared["meta_task_progress"]["current_step_guidance"])
        self.assertEqual(
            prepared["children"][0]["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        self.assertEqual(prepared["children"][0]["meta_task_progress"]["current_step_index"], 1)
        self.assertEqual(
            prepared["children"][0]["meta_task_progress"]["current_step"],
            "choose one active correction or closure",
        )
        self.assertEqual(prepared["children"][0]["meta_task_progress"]["phase"], "incremental_refinement")
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_fast_local_meta_analysis_for_simple_problem_statement(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Find the target value.",
            }
        )

        self.assertIn("meta_task", prepared)
        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        route_families = {
            str(item.get("route_family", ""))
            for item in prepared["meta_task"].get("route_options", [])
        }
        self.assertTrue({"dependency", "constraint", "invariant", "scaling"}.issubset(route_families))
        self.assertIn("decomposition", route_families)
        self.assertIn("extremal", route_families)
        self.assertGreaterEqual(len(prepared["meta_task"].get("route_options", [])), 6)
        self.assertEqual(
            prepared["meta_task"]["route_options"][0]["correction_mode"],
            "dependency-first scan",
        )
        self.assertIn(
            "constraint-first scan",
            {
                str(item.get("correction_mode", ""))
                for item in prepared["meta_task"].get("route_options", [])
            },
        )
        self.assertIn(
            "Pick exactly one route option and make one tiny route-local claim.",
            prepared["meta_task"]["step_blueprints"][0]["guidance"],
        )
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_fast_local_meta_analysis_for_frontend_default_context(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "task": "Use the modeling model to propose the next reasoning step, then score each step for domain consistency and variable grounding.",
                "notes": [
                    "The frontend polls the live scheduler state and renders it as an ASCII tree.",
                    "Node deletion is routed through the backend review model before the subtree is removed.",
                ],
                "known_context": {
                    "objective": "Derive and prune a useful reasoning tree.",
                    "expected_output": "concise, structured, and domain-valid intermediate steps",
                },
                "problem_statement": "Find the target value.",
            }
        )

        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_custom_domain_plugin_routes_for_meta_analysis(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Find the equilibrium market price.",
                "domain_plugins": [
                    {
                        "name": "microeconomics",
                        "label": "Microeconomics",
                        "knowledge_scope": ["supply curve", "demand curve", "equilibrium condition"],
                        "representative_formulas": [
                            {
                                "latex": r"Q_s(p) = Q_d(p)",
                                "meaning": "Market-clearing equilibrium condition.",
                            }
                        ],
                        "route_seed_options": [
                            {
                                "label": "market-clearing route",
                                "route_family": "market-clearing",
                                "governing_models": ["Supply-demand equilibrium"],
                                "guidance": "Choose one equilibrium relation before adding shocks.",
                                "correction_mode": "equilibrium-first scan",
                                "correction_target": "market-clearing condition",
                            }
                        ],
                    }
                ],
            }
        )

        route_families = {
            str(item.get("route_family", ""))
            for item in prepared["meta_task"].get("route_options", [])
        }
        self.assertEqual(route_families, {"market-clearing"})
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_explicit_skill_routes_for_meta_analysis(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Estimate the target relation.",
                "skill_names": ["partition_function"],
            }
        )

        route_families = {
            str(item.get("route_family", ""))
            for item in prepared["meta_task"].get("route_options", [])
        }
        self.assertEqual(route_families, {"partition-function"})
        self.assertIn(
            "partition_function skill only",
            prepared["meta_task"]["route_options"][0]["guidance"],
        )
        self.assertEqual(requester.calls, [])

    def test_strategy_scan_proposal_keeps_modeling_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "problem_statement": "Find the target value for a generic process with delayed closure.",
            "meta_task": {
                "objective": "Find the target value for a generic process with delayed closure.",
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
                "completion_signals": ["selected route grounded in known variables"],
            },
            "meta_task_progress": {
                "current_step_index": 0,
                "current_step": "identify governing relation",
                "current_step_guidance": "Choose one governing route only.",
                "remaining_steps": [
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
                "phase": "strategy_scan",
                "is_terminal_step": False,
            },
        }

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="inspect one route"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), DEFAULT_MODELING_MODEL)
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "proposal")

    def test_route_focused_incremental_refinement_proposal_uses_review_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "problem_statement": "Find the target value for a generic process with delayed closure.",
            "meta_task": {
                "objective": "Find the target value for a generic process with delayed closure.",
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
                "completion_signals": ["target quantity expressed in known variables"],
            },
            "meta_task_progress": {
                "current_step_index": 1,
                "current_step": "choose one active correction or closure",
                "current_step_guidance": "Choose one correction term only.",
                "remaining_steps": ["express the target quantity in known variables"],
                "phase": "incremental_refinement",
                "is_terminal_step": False,
                "selected_route_family": "route_a",
                "selected_correction_mode": "multi-state correction map",
                "selected_correction_target": "staged residual correction",
            },
            "route_focus": {
                "route_family": "route_a",
                "correction_mode": "multi-state correction map",
                "correction_target": "staged residual correction",
            },
        }

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="choose one correction"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), DEFAULT_REVIEW_MODEL)
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "proposal")

    def test_terminal_incremental_refinement_proposal_keeps_modeling_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "problem_statement": "Find the target value for a generic process with delayed closure.",
            "meta_task": {
                "objective": "Find the target value for a generic process with delayed closure.",
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
                "completion_signals": ["target quantity expressed in known variables"],
            },
            "meta_task_progress": {
                "current_step_index": 2,
                "current_step": "express the target quantity in known variables",
                "current_step_guidance": "Express the target quantity using the selected route.",
                "remaining_steps": [],
                "phase": "incremental_refinement",
                "is_terminal_step": True,
                "selected_route_family": "route_a",
                "selected_correction_mode": "multi-state correction map",
                "selected_correction_target": "staged residual correction",
            },
            "route_focus": {
                "route_family": "route_a",
                "correction_mode": "multi-state correction map",
                "correction_target": "staged residual correction",
            },
        }

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="close the chosen route"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), DEFAULT_MODELING_MODEL)
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "proposal")

    def test_non_terminal_evaluation_uses_light_review_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "meta_task": {
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
            },
            "meta_task_progress": {
                "current_step_index": 1,
                "current_step": "choose one active correction or closure",
                "current_step_guidance": "Choose one correction term only.",
                "remaining_steps": ["express the target quantity in known variables"],
                "phase": "incremental_refinement",
                "is_terminal_step": False,
                "selected_route_family": "route_a",
            },
        }

        backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="refine one correction"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), DEFAULT_NON_TERMINAL_EVALUATION_MODEL)
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "evaluate")

    def test_non_terminal_evaluation_can_use_configured_small_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            non_terminal_evaluation_model="qwen/qwen3-1.7b@4bit",
        )
        problem_context = {
            "meta_task": {
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
            },
            "meta_task_progress": {
                "current_step_index": 1,
                "current_step": "choose one active correction or closure",
                "current_step_guidance": "Choose one correction term only.",
                "remaining_steps": ["express the target quantity in known variables"],
                "phase": "incremental_refinement",
                "is_terminal_step": False,
                "selected_route_family": "route_a",
            },
        }

        backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="refine one correction"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), "qwen/qwen3-1.7b@4bit")
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "evaluate")

    def test_terminal_evaluation_keeps_review_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
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
                "current_step_guidance": "Write the target relation only.",
                "remaining_steps": [],
                "phase": "incremental_refinement",
                "is_terminal_step": True,
                "selected_route_family": "route_a",
            },
        }

        backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="close the route"),
            )
        )

        self.assertEqual(str(requester.calls[-1]["payload"]["model"]), DEFAULT_REVIEW_MODEL)
        self.assertEqual(json.loads(str(requester.calls[-1]["payload"]["input"]))["stage"], "evaluate")

    def test_non_terminal_evaluation_falls_back_to_review_model_on_invalid_light_payload(self) -> None:
        invalid_evaluation_model = "qwen2.5-0.5b-instruct-mlx"
        requester = InvalidEvaluationThenReviewFallbackChatRequester(
            invalid_model=invalid_evaluation_model,
            fallback_model=DEFAULT_REVIEW_MODEL,
        )
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            non_terminal_evaluation_model=invalid_evaluation_model,
        )
        problem_context = {
            "meta_task": {
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
            },
            "meta_task_progress": {
                "current_step_index": 1,
                "current_step": "choose one active correction or closure",
                "current_step_guidance": "Choose one correction term only.",
                "remaining_steps": ["express the target quantity in known variables"],
                "phase": "incremental_refinement",
                "is_terminal_step": False,
                "selected_route_family": "route_a",
            },
        }

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="refine one correction"),
            )
        )

        self.assertEqual(
            requester.calls,
            [
                f"evaluate:{invalid_evaluation_model}",
                f"repair:{invalid_evaluation_model}",
                f"evaluate:{DEFAULT_REVIEW_MODEL}",
            ],
        )
        self.assertEqual(evaluation["domain_consistency"], 0.79)

    def test_non_terminal_evaluation_falls_back_to_review_model_on_transport_error(self) -> None:
        requester = EvaluationTransportThenReviewFallbackChatRequester()
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            non_terminal_evaluation_model="qwen2.5-0.5b-instruct-mlx",
            max_retries=0,
        )
        problem_context = {
            "meta_task": {
                "first_step": "identify governing relation",
                "step_ordering": [
                    "identify governing relation",
                    "choose one active correction or closure",
                    "express the target quantity in known variables",
                ],
            },
            "meta_task_progress": {
                "current_step_index": 1,
                "current_step": "choose one active correction or closure",
                "current_step_guidance": "Choose one correction term only.",
                "remaining_steps": ["express the target quantity in known variables"],
                "phase": "incremental_refinement",
                "is_terminal_step": False,
                "selected_route_family": "route_a",
            },
        }

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=NodeSnapshot(id="node-1", thought_step="refine one correction"),
            )
        )

        self.assertEqual(
            requester.calls,
            [
                "evaluate:qwen2.5-0.5b-instruct-mlx",
                f"evaluate:{DEFAULT_REVIEW_MODEL}",
            ],
        )
        self.assertEqual(evaluation["domain_consistency"], 0.81)

    def test_local_chat_backend_uses_local_fallback_meta_analysis_for_long_problem_statement(self) -> None:
        requester = CompressionAwareMetaAnalysisChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Determine the target value from a very long prompt. " + ("x" * 5000),
            }
        )

        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        route_families = {
            str(item.get("route_family", ""))
            for item in prepared["meta_task"].get("route_options", [])
        }
        self.assertTrue({"dependency", "constraint", "invariant", "scaling"}.issubset(route_families))
        self.assertIn("decomposition", route_families)
        self.assertIn("extremal", route_families)
        self.assertGreaterEqual(len(prepared["meta_task"].get("route_options", [])), 6)
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_local_fallback_meta_analysis_for_large_problem_context(self) -> None:
        requester = MetaAnalysisOverflowRetryChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Determine the target value from a very long prompt. " + ("x" * 8000),
                "children": [
                    {"problem_statement": "child one context"},
                    {"problem_statement": "child two context"},
                ],
                "known_context": {"large_hint": "y" * 6000},
            }
        )

        self.assertIn("meta_task", prepared)
        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        self.assertTrue(prepared["meta_task"].get("route_options"))
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_prepare_problem_context_skips_overflow_meta_analysis_requester(self) -> None:
        requester = MetaAnalysisOverflowAlwaysChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Determine the target value from a very long prompt. " + ("x" * 8000),
            }
        )
        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        self.assertTrue(prepared["meta_task"].get("route_options"))
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_regime_map_route_for_answer_change_motion_prompt(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Compare two process variants and identify the parameter regime in which the answer changes.",
                "domain_plugins": [
                    {
                        "name": "generic-regime-test",
                        "label": "Generic Regime Test",
                        "route_seed_options": [
                            {
                                "label": "route-a path",
                                "route_family": "route_a",
                                "governing_models": ["Route Balance Model"],
                                "guidance": "Name one baseline relation only.",
                                "correction_mode": "baseline-first scan",
                                "correction_target": "baseline relation",
                            },
                            {
                                "label": "regime-map route",
                                "route_family": "regime-map",
                                "governing_models": ["Regime comparison"],
                                "guidance": "Identify one parameter threshold where the answer could flip before committing to a derivation.",
                                "correction_mode": "regime-selection scan",
                                "correction_target": "answer-flip boundary",
                            },
                        ],
                    }
                ],
            }
        )

        route_options = prepared["meta_task"].get("route_options", [])
        route_families = [str(item.get("route_family", "")) for item in route_options]
        self.assertEqual(route_families.count("route_a"), 1)
        self.assertIn("regime-map", route_families)
        regime_route = next(item for item in route_options if item.get("route_family") == "regime-map")
        self.assertEqual(regime_route["correction_target"], "answer-flip boundary")
        self.assertIn("parameter threshold", regime_route["guidance"])
        self.assertEqual(requester.calls, [])

    def test_strategy_scan_proposal_bypasses_orchestrator_model_without_selected_route(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Find the target value.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "route_options": [
                            {
                                "label": "route-b route",
                                "route_family": "route-b",
                                "correction_mode": "full-relation inventory",
                                "correction_target": "active relation term",
                            },
                            {
                                "label": "closure-c route",
                                "route_family": "closure-c",
                                "correction_mode": "closure-family scan",
                                "correction_target": "closure rule",
                            },
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 0,
                        "current_step": "identify governing relation",
                        "current_step_guidance": "Analyze the next-step strategy space broadly.",
                        "phase": "strategy_scan",
                        "remaining_steps": [
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                    },
                },
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        staged_calls = [json.loads(str(call["payload"]["input"]))["stage"] for call in requester.calls]
        self.assertEqual(staged_calls, ["proposal"])
        proposal_request = json.loads(str(requester.calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_task"],
            "Analyze the next-step strategy space broadly.",
        )

    def test_route_selected_strategy_scan_proposal_bypasses_orchestrator_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Find the target value.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "route_options": [
                            {
                                "label": "route-a path",
                                "route_family": "route_a",
                                "correction_mode": "lossless baseline first",
                                "correction_target": "deferred correction",
                            }
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 0,
                        "current_step": "route-local scan: route_a",
                        "current_step_guidance": "Stay at planning level and look only at the route-a path.",
                        "phase": "strategy_scan",
                        "remaining_steps": [
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "selected_route_family": "route_a",
                        "selected_correction_mode": "lossless baseline first",
                        "selected_correction_target": "deferred correction",
                    },
                    "route_focus": {
                        "label": "route-a path",
                        "route_family": "route_a",
                        "correction_mode": "lossless baseline first",
                        "correction_target": "deferred correction",
                    },
                },
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        staged_calls = [json.loads(str(call["payload"]["input"]))["stage"] for call in requester.calls]
        self.assertEqual(staged_calls, ["proposal"])
        proposal_request = json.loads(str(requester.calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_route_family"],
            "route_a",
        )
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_task"],
            "Stay at planning level and look only at the route-a path.",
        )

    def test_selected_route_incremental_refinement_proposal_bypasses_orchestrator_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Find the target value.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "route_options": [
                            {
                                "label": "route-b route",
                                "route_family": "route-b",
                                "correction_mode": "minimal closure first",
                                "correction_target": "neglected relation term",
                            }
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Refine only the current subproblem: choose one active correction or closure.",
                        "phase": "incremental_refinement",
                        "remaining_steps": ["express the target quantity in known variables"],
                        "selected_route_family": "route-b",
                        "selected_correction_mode": "minimal closure first",
                        "selected_correction_target": "neglected relation term",
                    },
                    "route_focus": {
                        "label": "route-b route with minimal closure",
                        "route_family": "route-b",
                        "correction_mode": "minimal closure first",
                        "correction_target": "neglected relation term",
                    },
                },
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        staged_calls = [json.loads(str(call["payload"]["input"]))["stage"] for call in requester.calls]
        self.assertEqual(staged_calls, ["proposal"])
        proposal_request = json.loads(str(requester.calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_route_family"],
            "route-b",
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task_progress"]["selected_correction_mode"],
            "minimal closure first",
        )

    def test_selected_route_incremental_refinement_without_correction_metadata_bypasses_orchestrator_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Compare two route-local closures.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Stay on the selected route and add one local correction only.",
                        "phase": "incremental_refinement",
                        "remaining_steps": ["express the target quantity in known variables"],
                        "selected_route_family": "state-transition",
                        "is_terminal_step": False,
                    },
                },
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        staged_calls = [json.loads(str(call["payload"]["input"]))["stage"] for call in requester.calls]
        self.assertEqual(staged_calls, ["proposal"])
        proposal_request = json.loads(str(requester.calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_route_family"],
            "state-transition",
        )

    def test_selected_route_incremental_reflection_with_latest_critique_bypasses_orchestrator_model(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                latest_critique="Rewrite thought_step so it names exactly one new local delta.",
                problem_context={
                    "problem_statement": "Repair one route-local semantic delta.",
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Stay on the selected route and add one local correction only.",
                        "phase": "incremental_refinement",
                        "remaining_steps": ["express the target quantity in known variables"],
                        "selected_route_family": "state-transition",
                        "is_terminal_step": False,
                    },
                    "route_focus": {
                        "label": "state-transition route",
                        "route_family": "state-transition",
                        "correction_mode": "piecewise-state propagation",
                        "correction_target": "state transition",
                    },
                },
                current_node=NodeSnapshot(id="node-1", thought_step="repeat the parent wording"),
            )
        )

        staged_calls = [json.loads(str(call["payload"]["input"]))["stage"] for call in requester.calls]
        self.assertEqual(staged_calls, ["reflect"])
        reflect_request = json.loads(str(requester.calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            reflect_request["problem_context"]["orchestrator_task"]["selected_task"],
            "Rewrite thought_step so it names exactly one new local delta.",
        )

    def test_local_chat_backend_prepare_problem_context_skips_invalid_meta_analysis_requester(self) -> None:
        requester = InvalidMetaAnalysisPayloadChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Find the target value.",
                "children": [{"problem_statement": "Consider closure-rule choices first."}],
            }
        )

        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        self.assertEqual(requester.calls, [])
        self.assertEqual(len(requester.calls), 0)

    def test_local_chat_backend_prepare_problem_context_skips_timeout_meta_analysis_requester(self) -> None:
        requester = TimeoutMetaAnalysisChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Compare two modeling routes before selecting one.",
                "children": [{"problem_statement": "keep one route deferred"}],
            }
        )

        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(prepared["meta_task_progress"]["phase"], "strategy_scan")
        self.assertEqual(requester.calls, [])
        self.assertEqual(len(requester.calls), 0)

    def test_local_chat_backend_prepare_problem_context_bypasses_structured_meta_analysis_requester(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        prepared = backend.prepare_problem_context(
            {
                "problem_statement": "Compare route families before refining the constraint.",
            }
        )

        self.assertEqual(prepared["meta_task"]["first_step"], "identify governing relation")
        self.assertEqual(
            prepared["meta_task"]["step_ordering"],
            list(LocalChatDualModelBackendAdapter.LOCAL_META_ANALYSIS_STEP_ORDERING),
        )
        self.assertEqual(prepared["meta_task_progress"]["current_step"], "identify governing relation")
        self.assertEqual(requester.calls, [])

    def test_local_chat_backend_uses_single_step_prompt_contract_for_proposal_and_reflection(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        node_snapshot = NodeSnapshot(
            id="node-1",
            known_vars={"trace_blob": "x" * 1200},
            reflection_history=["older reflection", "latest reflection"],
        )
        problem_context = {
            "meta_task": {
                "first_step": "identify governing relation",
                "step_ordering": ["identify governing relation", "solve for target quantity"],
                "route_options": [
                    {
                        "label": "route-a path",
                        "route_family": "route_a",
                        "correction_mode": "lossless baseline first",
                        "correction_target": "deferred correction",
                    },
                    {
                        "label": "route-b route",
                        "route_family": "route-b",
                        "correction_mode": "piecewise-relation closure",
                        "correction_target": "segment transition relation",
                    },
                ],
                "step_blueprints": [
                    {
                        "label": "identify governing relation",
                        "step_type": "strategy_scan",
                        "correction_mode": "compare deferred-loss closures",
                    },
                    {
                        "label": "solve for target quantity",
                        "step_type": "incremental_refinement",
                        "correction_target": "deferred correction",
                    },
                ],
            },
            "meta_task_progress": {
                "current_step_index": 0,
                "current_step": "identify governing relation",
                "current_step_guidance": "Analyze the next-step strategy space broadly.",
                "phase": "strategy_scan",
                "remaining_steps": ["solve for target quantity"],
            },
            "children": [{"problem_statement": "child context should be counted only"}],
            "known_context": {"large_hint": "y" * 800},
        }

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )
        backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )

        staged_calls = list(requester.calls)
        staged_stages = [json.loads(str(call["payload"]["input"]))["stage"] for call in staged_calls]
        self.assertEqual(staged_stages, ["proposal", "reflect"])
        proposal_prompt = str(staged_calls[0]["payload"]["system_prompt"])
        reflection_prompt = str(staged_calls[1]["payload"]["system_prompt"])
        proposal_request = json.loads(str(staged_calls[0]["payload"]["input"]))["request"]
        self.assertIn("exactly one minimal next-step candidate", proposal_prompt)
        self.assertIn("request.problem_context.orchestrator_task.selected_task", proposal_prompt)
        self.assertIn("request.problem_context.meta_task_progress.current_step_guidance", proposal_prompt)
        self.assertIn("phase is strategy_scan", proposal_prompt)
        self.assertIn("phase is incremental_refinement", proposal_prompt)
        self.assertIn("do not compare many routes inside one node", proposal_prompt)
        self.assertIn("must add exactly one explicit local delta beyond the parent", proposal_prompt)
        self.assertIn("thought_step itself must name that new local delta", proposal_prompt)
        self.assertIn("active_control_parameter", proposal_prompt)
        self.assertIn("exactly one local revision step", reflection_prompt)
        self.assertIn("stay route-local and atomic", reflection_prompt)
        self.assertIn("repair that by adding exactly one explicit local delta", reflection_prompt)
        self.assertIn("thought_step itself must name that delta", reflection_prompt)
        self.assertIn("request.problem_context.orchestrator_task.selected_task", reflection_prompt)
        self.assertIn("Do not add any other top-level keys.", proposal_prompt)
        self.assertIn("Use {} for empty object fields", proposal_prompt)
        self.assertIn("single short paragraph", proposal_prompt)
        self.assertIn("under about 160 characters", proposal_prompt)
        self.assertIn("one phrase or sentence only", proposal_prompt)
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_task"],
            "Analyze the next-step strategy space broadly.",
        )
        selected_label = proposal_request["problem_context"]["orchestrator_task"]["selected_task"]
        self.assertEqual(
            sum(
                1
                for item in proposal_request["problem_context"]["orchestrator_task"]["candidate_tasks"]
                if item.get("label") == selected_label
            ),
            1,
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task_progress"]["current_step_guidance"],
            "Execute only this task: Analyze the next-step strategy space broadly.",
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task"]["route_options"][0]["route_family"],
            "route_a",
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task"]["route_options"][0]["correction_target"],
            "deferred correction",
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task"]["step_blueprints"][0]["step_type"],
            "strategy_scan",
        )
        self.assertNotIn(
            "selected_correction_mode",
            proposal_request["problem_context"]["meta_task_progress"],
        )
        self.assertEqual(proposal_request["problem_context"]["child_context_count"], 1)
        self.assertNotIn("children", proposal_request["problem_context"])
        self.assertNotIn("known_context", proposal_request["problem_context"])
        self.assertEqual(proposal_request["current_node"]["known_vars"]["known_var_keys"], ["trace_blob"])
        self.assertNotIn("trace_blob", proposal_request["current_node"]["known_vars"])
        self.assertEqual(proposal_request["schema_id"], "proposal.v1")

    def test_only_modeling_requests_receive_full_problem_statement(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        problem_context = {
            "problem_statement": "Find the target value for a generic process with delayed closure.",
            "meta_task": {
                "objective": "Find the target value for a generic process with delayed closure.",
                "first_step": "identify governing relation",
                "step_ordering": ["identify governing relation", "solve for target quantity"],
                "completion_signals": ["selected route grounded in known variables"],
            },
            "meta_task_progress": {
                "current_step_index": 0,
                "current_step": "identify governing relation",
                "current_step_guidance": "Choose one governing route only.",
                "remaining_steps": ["solve for target quantity"],
                "phase": "strategy_scan",
            },
        }
        node_snapshot = NodeSnapshot(id="node-1", thought_step="inspect one route")

        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )
        backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )

        staged_calls = list(requester.calls)
        staged_stages = [json.loads(str(call["payload"]["input"]))["stage"] for call in staged_calls]
        self.assertEqual(staged_stages, ["proposal", "evaluate"])
        proposal_request = json.loads(str(staged_calls[0]["payload"]["input"]))["request"]
        evaluation_request = json.loads(str(staged_calls[1]["payload"]["input"]))["request"]

        self.assertEqual(
            proposal_request["problem_context"]["problem_statement"],
            "Find the target value for a generic process with delayed closure.",
        )
        self.assertEqual(
            proposal_request["problem_context"]["meta_task"]["objective"],
            "Find the target value for a generic process with delayed closure.",
        )
        self.assertNotIn("problem_statement", evaluation_request["problem_context"])
        self.assertNotIn("objective", evaluation_request["problem_context"]["meta_task"])

    def test_local_chat_backend_maps_timeout_to_transport_error(self) -> None:
        backend = LocalChatDualModelBackendAdapter(
            requester=TimeoutChatRequester(),
            max_retries=0,
        )

        with self.assertRaisesRegex(ChatBackendTransportError, "Timed out after"):
            backend.propose(
                ProposalRequest(
                    attempt_index=0,
                    problem_context={},
                    current_node=NodeSnapshot(id="node-1"),
                )
            )

    def test_local_chat_backend_retries_transient_transport_failure(self) -> None:
        requester = FlakyChatRequester()
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            max_retries=1,
            retry_backoff_seconds=0.0,
        )

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(requester.calls, 2)
        self.assertEqual(proposal["equations"], ["eq_retry"])

    def test_local_chat_backend_falls_back_when_model_is_unavailable(self) -> None:
        missing_model = "missing/local-qwen-model"
        requester = ModelNotFoundThenFallbackChatRequester(missing_model=missing_model)
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            modeling_model=missing_model,
            max_retries=0,
        )

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(requester.calls[0], missing_model)
        self.assertEqual(requester.calls[1], DEFAULT_REVIEW_MODEL)
        self.assertEqual(proposal["thought_step"], "fallback success")

    def test_route_focused_proposal_falls_back_after_review_model_timeout(self) -> None:
        requester = ReviewTimeoutThenProposalFallbackChatRequester()
        backend = LocalChatDualModelBackendAdapter(
            requester=requester,
            review_model="qwen/qwen3-1.7b",
            max_retries=0,
        )

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "meta_task": {
                        "first_step": "identify governing relation",
                        "step_ordering": [
                            "identify governing relation",
                            "choose one active correction or closure",
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Stay on one selected route only.",
                        "phase": "incremental_refinement",
                        "remaining_steps": [],
                        "selected_route_family": "route_a",
                        "selected_correction_mode": "lossless baseline first",
                        "selected_correction_target": "residual work",
                    },
                    "route_focus": {
                        "route_family": "route_a",
                        "correction_mode": "lossless baseline first",
                        "correction_target": "residual work",
                    },
                },
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(requester.calls[0], "qwen/qwen3-1.7b")
        self.assertEqual(requester.calls[1], DEFAULT_REVIEW_MODEL)
        self.assertEqual(proposal["thought_step"], "fallback proposal success")
        self.assertEqual(proposal["equations"], ["eq_fallback_proposal"])

    def test_local_chat_backend_accepts_list_content_wrapper(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=ListContentChatRequester())

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["thought_step"], "list content success")

    def test_local_chat_backend_accepts_typo_simplity_hint_without_repair(self) -> None:
        requester = TypoEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertEqual(evaluation["simplicity_hint"], 0.6)

    def test_local_chat_backend_treats_blank_simplicity_hint_as_none(self) -> None:
        requester = BlankSimplicityEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertIsNone(evaluation["simplicity_hint"])

    def test_local_chat_backend_accepts_typo_simply_hint_without_repair(self) -> None:
        requester = SimplyHintEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertEqual(evaluation["simplicity_hint"], 0.55)

    def test_local_chat_backend_accepts_typo_simplification_hint_without_repair(self) -> None:
        requester = SimplificationHintEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertEqual(evaluation["simplicity_hint"], 0.51)

    def test_local_chat_backend_accepts_typo_contextal_relevance_without_repair(self) -> None:
        requester = ContextalRelevanceEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertEqual(evaluation["contextual_relevance"], 0.82)

    def test_local_chat_backend_treats_textual_simplicity_hint_as_none(self) -> None:
        requester = TextualSimplicityHintEvaluationChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    }
                },
                current_node=NodeSnapshot(id="node-1", thought_step="evaluate one correction"),
            )
        )

        self.assertEqual(requester.calls, ["evaluate"])
        self.assertIsNone(evaluation["simplicity_hint"])

    def test_local_chat_backend_prefers_message_segment_over_reasoning_trace(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=ReasoningAndMessageChatRequester())

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["thought_step"], "message segment success")
        self.assertEqual(proposal["equations"], ["eq_message"])

    def test_local_chat_backend_extracts_final_json_object_from_mixed_text(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=MixedTextProposalChatRequester())

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["thought_step"], "mixed text success")
        self.assertEqual(proposal["equations"], ["eq_mixed"])

    def test_local_chat_backend_extracts_final_json_object_during_repair(self) -> None:
        requester = MixedTextRepairChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["thought_step"], "repair mixed text success")
        self.assertEqual(proposal["equations"], ["eq_repair_mixed"])
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(json.loads(str(requester.calls[1]["input"]))["stage"], "repair")

    def test_local_chat_backend_coerces_schema_adjacent_collection_fields(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=SchemaAdjacentProposalChatRequester())

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=NodeSnapshot(id="node-1"),
            )
        )

        self.assertEqual(proposal["equations"], ["eq_schema_adjacent"])
        self.assertEqual(proposal["used_models"], ["closure_model"])
        self.assertEqual(proposal["known_vars"], {"g": None, "mu": None})
        self.assertEqual(proposal["quantities"], {"v_t": None, "rho_s": None})
        self.assertEqual(proposal["boundary_conditions"], {"small Reynolds number": None})

    def test_local_chat_backend_rejects_empty_response(self) -> None:
        backend = LocalChatDualModelBackendAdapter(requester=EmptyChatRequester())

        with self.assertRaises(ChatBackendResponseError):
            backend.evaluate(
                EvaluationRequest(
                    attempt_index=0,
                    problem_context={},
                    current_node=NodeSnapshot(id="node-1"),
                )
            )

    def test_local_chat_backend_uses_modeling_and_review_models(self) -> None:
        requester = CapturingChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)
        node_snapshot = NodeSnapshot(
            id="node-1",
            thought_step="t" * 900,
            equations=["eq0", "eq1", "eq2", "eq3", "eq4", "eq5", "eq6"],
            known_vars={"debug_trace": "z" * 1500},
            used_models=["m0", "m1", "m2", "m3", "m4", "m5", "m6"],
        )
        problem_context = {
            "meta_task": {"step_ordering": ["identify governing relation", "solve for target quantity"]},
            "meta_task_progress": {
                "current_step_index": 0,
                "current_step": "identify governing relation",
                "current_step_guidance": "Analyze the next-step strategy space broadly.",
                "phase": "strategy_scan",
                "remaining_steps": ["solve for target quantity"],
            },
            "children": [{"problem_statement": "child one"}, {"problem_statement": "child two"}],
        }

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )
        evaluation = backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
            )
        )
        reflection = backend.reflect(
            ReflectionRequest(
                attempt_index=0,
                problem_context=problem_context,
                current_node=node_snapshot,
                latest_critique="variables are insufficiently grounded",
            )
        )

        self.assertEqual(proposal["thought_step"], "model next step")
        self.assertEqual(evaluation["reason"], "looks good")
        self.assertEqual(reflection["thought_step"], "refined step")
        staged_calls = list(requester.calls)
        staged_stages = [json.loads(str(call["payload"]["input"]))["stage"] for call in staged_calls]
        self.assertEqual(staged_stages, ["proposal", "evaluate", "orchestrator", "reflect"])
        self.assertEqual(
            [call["payload"]["model"] for call in staged_calls],
            [
                DEFAULT_MODELING_MODEL,
                DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
                DEFAULT_PLANNING_MODEL,
                DEFAULT_MODELING_MODEL,
            ],
        )
        proposal_request = json.loads(str(staged_calls[0]["payload"]["input"]))["request"]
        evaluation_prompt = str(staged_calls[1]["payload"]["system_prompt"])
        evaluation_request = json.loads(str(staged_calls[1]["payload"]["input"]))["request"]
        orchestrator_request = json.loads(str(staged_calls[2]["payload"]["input"]))["request"]
        self.assertEqual(proposal_request["problem_context"]["child_context_count"], 2)
        self.assertNotIn("children", proposal_request["problem_context"])
        self.assertIn("Use JSON numbers, not strings, for numeric fields", evaluation_prompt)
        self.assertIn("single short paragraph", evaluation_prompt)
        self.assertNotIn("problem_statement", orchestrator_request["problem_context"])
        self.assertNotIn("objective", orchestrator_request["problem_context"]["meta_task"])
        self.assertEqual(evaluation_request["problem_context"]["child_context_count"], 2)
        self.assertEqual(evaluation_request["current_node"]["known_vars"]["known_var_keys"], ["debug_trace"])
        self.assertEqual(len(evaluation_request["current_node"]["equations"]), 6)
        self.assertEqual(len(evaluation_request["current_node"]["used_models"]), 6)
        self.assertEqual(evaluation_request["schema_id"], "evaluation.v1")
        for call in requester.calls:
            self.assertEqual(call["url"], DEFAULT_CHAT_API_URL)
            self.assertEqual(set(call["payload"]), {"model", "system_prompt", "input"})

    def test_node_snapshot_flattens_nested_equation_lists(self) -> None:
        fsm = NodeBuilderFSM(parent_node=None, problem_context={})
        node = ToTNode.model_construct(
            id="node-1",
            parent_id=None,
            thought_step="test node",
            equations=[["F = m * a", "refined_attempt_1"], "v = a t"],
            known_vars={},
            used_models=[["Segment Balance Model"], "State-transition"],
            quantities={},
            boundary_conditions={},
            status=NodeStatus.ACTIVE,
            fsm_state=FSMState.EVALUATE,
            score=0.0,
            reflection_history=[["first critique"], "second critique"],
            children=[],
        )

        snapshot = fsm._node_snapshot(node)

        self.assertEqual(snapshot.equations, ["F = m * a", "refined_attempt_1", "v = a t"])
        self.assertEqual(snapshot.used_models, ["Segment Balance Model", "State-transition"])
        self.assertEqual(snapshot.reflection_history, ["first critique", "second critique"])

    def test_local_chat_bundle_uses_review_model_for_delete_review(self) -> None:
        requester = CapturingChatRequester()
        backend_factory, delete_adapter = build_local_chat_adapter_bundle(requester=requester)
        node_snapshot = NodeSnapshot(id="node-1")

        backend = backend_factory({})
        backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={},
                current_node=node_snapshot,
            )
        )
        decision = delete_adapter.review_delete_node(
            DeleteNodeReviewRequest(
                requested_by="frontend",
                reason="cleanup",
                target_node=node_snapshot,
            )
        )

        self.assertTrue(decision["approved"])
        self.assertIsInstance(delete_adapter, LocalChatDeletionReviewAdapter)
        self.assertEqual(requester.calls[0]["payload"]["model"], DEFAULT_MODELING_MODEL)
        self.assertEqual(requester.calls[1]["payload"]["model"], DEFAULT_REVIEW_MODEL)

    def test_local_chat_delete_review_uses_local_fallback_when_enabled(self) -> None:
        calls = []

        def unavailable_requester(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
            calls.append({"url": url, "payload": dict(payload), "timeout": timeout})
            raise ChatBackendTransportError("delete review model unavailable")

        delete_adapter = LocalChatDeletionReviewAdapter(
            requester=unavailable_requester,
            allow_live_model_fallback=True,
        )

        decision = delete_adapter.review_delete_node(
            DeleteNodeReviewRequest(
                requested_by="frontend",
                reason="operator prune and steer",
                target_node=NodeSnapshot(id="node-1"),
                descendant_count=1,
            )
        )

        self.assertTrue(decision["approved"])
        self.assertEqual(decision["risk_level"], "medium")
        self.assertIn("operator prune and steer", decision["reason"])
        self.assertEqual(calls, [])

    def test_local_chat_bundle_propagates_custom_non_terminal_evaluation_model(self) -> None:
        requester = CapturingChatRequester()
        backend_factory, _delete_adapter = build_local_chat_adapter_bundle(
            requester=requester,
            non_terminal_evaluation_model="qwen/qwen3-1.7b@4bit",
        )
        backend = backend_factory({})

        backend.evaluate(
            EvaluationRequest(
                attempt_index=0,
                problem_context={
                    "meta_task_progress": {
                        "current_step_index": 1,
                        "current_step": "choose one active correction or closure",
                        "current_step_guidance": "Choose one correction term only.",
                        "remaining_steps": ["express the target quantity in known variables"],
                        "phase": "incremental_refinement",
                        "is_terminal_step": False,
                    },
                },
                current_node=NodeSnapshot(id="node-1", thought_step="refine one correction"),
            )
        )

        self.assertEqual(requester.calls[-1]["payload"]["model"], "qwen/qwen3-1.7b@4bit")

    def test_orchestrator_guidance_overrides_meta_task_progress_for_hard_rule_checks(self) -> None:
        strict_guidance = "Compare only the segment route-b route and defer every correction term."
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": ["compare route families", "refine segment relation"],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space broadly.",
                    "phase": "strategy_scan",
                    "remaining_steps": ["refine segment relation"],
                },
                "proposal": {
                    "thought_step": strict_guidance,
                    "equations": ["candidate route: segment relation balance"],
                    "known_vars": {
                        "orchestrator_task": {
                            "step_focus": "compare route families",
                            "current_step_guidance": strict_guidance,
                            "task_breakdown": [
                                "compare the segment route-b route",
                                "defer the correction terms",
                            ],
                            "selected_task": "compare the segment route-b route only",
                            "deferred_tasks": ["derive the target relation"],
                            "completion_signals": ["one route isolated"],
                        }
                    },
                    "used_models": ["Segment Balance Model"],
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(
            node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["current_step_guidance"],
            strict_guidance,
        )

    def test_orchestrator_context_overflow_falls_back_to_local_task(self) -> None:
        requester = OrchestratorOverflowChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester)

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Stress the orchestrator with a large state tree.",
                    "meta_task": {
                        "first_step": "compare route families",
                        "step_ordering": [
                            "compare route families",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "completion_signals": ["one local task completed"],
                    },
                    "meta_task_progress": {
                        "current_step_index": 2,
                        "current_step": "express the target quantity in known variables",
                        "current_step_guidance": "Express the target quantity without reopening route selection.",
                        "remaining_steps": [],
                        "phase": "incremental_refinement",
                        "is_terminal_step": True,
                    },
                    "children": [{"problem_statement": "child one"}, {"problem_statement": "child two"}],
                },
                current_node=NodeSnapshot(
                    id="node-1",
                    known_vars={
                        "hard_rule_check": {
                            "checked": {"trace": "x" * 5000},
                            "violations": ["v" * 2000],
                        }
                    },
                ),
            )
        )

        self.assertEqual(proposal["thought_step"], "fallback proposal")
        orchestrator_calls = [
            call
            for call in requester.calls
            if json.loads(str(call["payload"]["input"]))["stage"] == "orchestrator"
        ]
        proposal_calls = [
            call
            for call in requester.calls
            if json.loads(str(call["payload"]["input"]))["stage"] == "proposal"
        ]
        self.assertGreaterEqual(len(orchestrator_calls), 2)
        self.assertEqual(len(proposal_calls), 1)
        first_orchestrator_request = json.loads(str(orchestrator_calls[0]["payload"]["input"]))["request"]
        self.assertEqual(first_orchestrator_request["problem_context"]["child_context_count"], 2)
        proposal_request = json.loads(str(proposal_calls[0]["payload"]["input"]))["request"]
        self.assertEqual(
            proposal_request["problem_context"]["orchestrator_task"]["selected_task"],
            "Express the target quantity without reopening route selection.",
        )
        self.assertEqual(proposal_request["problem_context"]["child_context_count"], 2)

    def test_orchestrator_transient_failure_falls_back_to_local_task(self) -> None:
        requester = OrchestratorTransientFailureChatRequester()
        backend = LocalChatDualModelBackendAdapter(requester=requester, max_retries=0)

        proposal = backend.propose(
            ProposalRequest(
                attempt_index=0,
                problem_context={
                    "problem_statement": "Close the selected route after choosing the correction term.",
                    "meta_task": {
                        "first_step": "compare route families",
                        "step_ordering": [
                            "compare route families",
                            "choose one active correction or closure",
                            "express the target quantity in known variables",
                        ],
                        "completion_signals": ["one local task completed"],
                    },
                    "meta_task_progress": {
                        "current_step_index": 2,
                        "current_step": "express the target quantity in known variables",
                        "current_step_guidance": "Express the target quantity without reopening route selection.",
                        "remaining_steps": [],
                        "phase": "incremental_refinement",
                        "is_terminal_step": True,
                    },
                    "children": [{"problem_statement": "child one"}, {"problem_statement": "child two"}],
                },
                current_node=NodeSnapshot(id="node-1", thought_step="close the route"),
            )
        )

        self.assertEqual(proposal["thought_step"], "transient fallback proposal")
        self.assertEqual(proposal["equations"], ["eq_transient_fallback"])
        orchestrator_calls = [
            call
            for call in requester.calls
            if json.loads(str(call["payload"]["input"]))["stage"] == "orchestrator"
        ]
        self.assertGreaterEqual(len(orchestrator_calls), 1)

    def test_hard_rule_check_uses_registry_invoker_trace(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "proposal": {"equations": ["E = T + V"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["E = T + V"]}
                },
                "evaluation": {"score": 8.0},
            },
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertIn("hard_rule_check", node.known_vars)
        self.assertEqual(node.known_vars["hard_rule_check"]["trace"]["skill_name"], "tot_hard_rule_check")
        self.assertEqual(node.known_vars["hard_rule_check"]["trace"]["call_style"], "params_dict")

    def test_low_score_without_rule_violation_stays_active(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "proposal": {"equations": ["eq0"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["eq0"]}
                },
                "evaluation": {
                    "domain_consistency": 0.35,
                    "variable_grounding": 0.4,
                    "contextual_relevance": 0.5,
                },
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["needs_deeper_reasoning"])
        self.assertFalse(node.known_vars["evaluation_passed"])
        self.assertIn("expansion_priority", node.known_vars)

    def test_successful_evaluation_uses_positive_default_reason(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "proposal": {"equations": ["eq0"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["eq0"]}
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["evaluation_passed"])
        self.assertEqual(
            node.known_vars["evaluation_breakdown"]["reason"],
            "Weighted evaluation passed the acceptance threshold.",
        )

    def test_semantic_delta_critique_caps_weighted_score_below_threshold(self) -> None:
        critique = (
            "Non-terminal child hid its new local delta only in structured fields while repeating the parent thought_step."
        )
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "evaluation": {"score": 8.5},
            },
            max_reflections=0,
        )
        fsm.node.equations = ["eq0"]
        fsm.node.fsm_state = FSMState.EVALUATE
        fsm.node.known_vars["semantic_delta_critique"] = critique

        fsm._handle_evaluate()

        self.assertFalse(fsm.node.known_vars["evaluation_passed"])
        self.assertLess(fsm.node.score, fsm.SCORE_THRESHOLD)
        self.assertEqual(fsm.node.score, 5.75)
        self.assertIn(critique, fsm.node.known_vars["evaluation_breakdown"]["reason"])
        self.assertTrue(fsm.node.known_vars["needs_deeper_reasoning"])

    def test_recoverable_rule_violation_caps_weighted_score_below_threshold(self) -> None:
        recoverable_violation = "Missing required boundary condition key: x=L"
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "evaluation": {
                    "score": 8.5,
                    "hard_rule_violations": [recoverable_violation],
                },
            },
            max_reflections=0,
        )
        fsm.node.equations = ["eq0"]
        fsm.node.fsm_state = FSMState.EVALUATE

        fsm._handle_evaluate()

        self.assertFalse(fsm.node.known_vars["evaluation_passed"])
        self.assertLess(fsm.node.score, fsm.SCORE_THRESHOLD)
        self.assertEqual(fsm.node.score, 5.75)
        self.assertEqual(fsm.node.fsm_state, FSMState.REFLECT)
        self.assertIn(recoverable_violation, fsm.node.known_vars["evaluation_breakdown"]["reason"])

    def test_rule_violation_prunes_node(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "proposal": {"equations": ["eq0"]},
                "calculation": {
                    "skill_params": {"forbidden_equation_patterns": ["eq0"]}
                },
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.PRUNED_BY_RULE)
        self.assertEqual(node.result_state, NodeResultState.DROP)
        self.assertIn("hard_rule_violations", node.known_vars)

    def test_terminal_pass_marks_node_as_finalize_only_on_final_result(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "express target quantity",
                    "step_ordering": ["express target quantity"],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "express target quantity",
                    "current_step_guidance": "Write the final target relation.",
                    "phase": "incremental_refinement",
                    "total_steps": 1,
                    "is_terminal_step": True,
                },
                "proposal": {"equations": ["eq0"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["eq0"]}
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.SOLVED)
        self.assertEqual(node.result_state, NodeResultState.FINALIZE)

    def test_root_rule_violation_goes_to_reflection(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "proposal": {
                    "thought_step": "Start from a forbidden first draft.",
                    "equations": ["eq0"],
                },
                "calculation": {
                    "skill_params": {"forbidden_equation_patterns": ["eq0"]}
                },
                "reflection": {
                    "thought_step": "Replace the forbidden root draft with a revised seed branch.",
                    "equations": ["eq1"],
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.thought_step, "Replace the forbidden root draft with a revised seed branch.")
        self.assertEqual(node.equations, ["eq1"])
        self.assertEqual(len(node.reflection_history), 1)
        self.assertNotIn("hard_rule_violations", node.known_vars)

    def test_root_skill_validation_mismatch_goes_to_reflection(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "skill_names": ["partition_function"],
                "proposal": {
                    "thought_step": "Use a segmented-style relation first.",
                    "equations": ["F = m a"],
                },
                "calculation": {},
                "reflection": {
                    "thought_step": "Switch to a partition-function state sum.",
                    "equations": ["Z = exp(-beta E0)"],
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.thought_step, "Switch to a partition-function state sum.")
        self.assertEqual(node.equations, ["Z = exp(-beta E0)"])
        self.assertEqual(
            node.known_vars["hard_rule_check"]["checked"]["validation_plugin_selection_mode"],
            "explicit",
        )
        self.assertEqual(
            node.known_vars["hard_rule_check"]["checked"]["validation_plugins"][0]["skill_names"],
            ["partition_function"],
        )
        self.assertEqual(
            node.reflection_history,
            ["No context matches any required pattern: partition | beta | ln z | z = | boltzmann"],
        )
        self.assertNotIn("hard_rule_violations", node.known_vars)

    def test_root_nested_validation_rule_group_mismatch_goes_to_reflection(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "domain_plugins": [
                    {
                        "name": "microeconomics",
                        "label": "Microeconomics",
                        "validation_rules": {
                            "equations": {
                                "require_any_patterns": ["Q_s", "Q_d"],
                            },
                            "context": {
                                "require_all_patterns": ["market", "equilibrium"],
                            },
                        },
                    }
                ],
                "proposal": {
                    "thought_step": "Use a relation balance first.",
                    "equations": ["F = m a"],
                },
                "calculation": {},
                "reflection": {
                    "thought_step": "Use the market equilibrium condition.",
                    "equations": ["Q_s(p) = Q_d(p)"],
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.thought_step, "Use the market equilibrium condition.")
        self.assertEqual(node.equations, ["Q_s(p) = Q_d(p)"])
        self.assertIn(
            "No equation matches any required pattern: Q_s | Q_d",
            node.reflection_history[0],
        )
        self.assertNotIn("hard_rule_violations", node.known_vars)

    def test_meta_task_scope_allows_comparative_strategy_scan_without_reflection(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "compare plausible next-step solution routes without computing the answer",
                    "step_ordering": [
                        "compare plausible next-step solution routes without computing the answer",
                        "apply the route_a method on the segment only",
                        "propagate the result across the horizontal segment",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare plausible next-step solution routes without computing the answer",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level: compare plausible governing routes, identify the main laws/models each route would use, mark the key unknowns and deferred correction terms, and decide what should be refined next. Do not solve for the final answer yet.",
                    "remaining_steps": [
                        "apply the route_a method on the segment only",
                        "propagate the result across the horizontal segment",
                    ],
                },
                "proposal": {
                    "thought_step": "Apply the route_a method on the segment only, then propagate the result across the next segment to solve for the final value.",
                    "equations": ["relation_out = relation_in - correction"],
                    "used_models": ["Route Balance Model"],
                },
                "calculation": {"skill_params": {"enforce_meta_task_step_scope": True}},
                "reflection": {
                    "thought_step": "Compare an segment-only route-a path against a staged horizontal-residual route and defer the actual final-value derivation to the next refinement step.",
                    "equations": [
                        "candidate route A: segment route-balance balance",
                        "candidate route B: staged residual bookkeeping after the segment",
                    ],
                    "used_models": ["Route Balance Model"],
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertNotIn("recoverable_rule_violations", node.known_vars)
        self.assertNotIn("hard_rule_violations", node.known_vars)
        self.assertEqual(node.thought_step, "Apply the route_a method on the segment only, then propagate the result across the next segment to solve for the final value.")
        meta_scope = node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]
        self.assertTrue(meta_scope["comparative_strategy_scan"])
        self.assertEqual(meta_scope["violations"], [])
        self.assertEqual(len(node.reflection_history), 0)

    def test_root_strategy_scan_ignores_review_hard_rule_false_positives(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "identify governing relation",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Execute only this task: Analyze the next-step strategy space at planning level. Make one route-local planning claim only, keep all other routes deferred, and do not solve the final answer yet.",
                    "remaining_steps": ["choose one active correction or closure"],
                },
                "proposal": {
                    "thought_step": "Select the route-a path and state that the route-balance theorem governs the motion for both bodies.",
                    "equations": ["ΔK + ΔU = W_fric"],
                    "used_models": ["Route Balance Model"],
                },
                "calculation": {},
                "evaluation": {
                    "domain_consistency": 1.0,
                    "variable_grounding": 1.0,
                    "contextual_relevance": 1.0,
                    "simplicity_hint": 1.0,
                    "reason": "The selected task aligns with the problem's route_a-based framework and follows the required step-by-step strategy.",
                    "hard_rule_violations": [
                        "task execution exceeds single-step constraint",
                        "no actual route-local claim produced",
                    ],
                },
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["evaluation_passed"])
        self.assertNotIn("hard_rule_violations", node.known_vars)
        self.assertEqual(node.reflection_history, [])
        self.assertEqual(
            node.known_vars["ignored_review_rule_violations"],
            [
                "task execution exceeds single-step constraint",
                "no actual route-local claim produced",
            ],
        )

    def test_route_focused_strategy_scan_child_ignores_review_hard_rule_false_positives(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "route-local scan: route_a",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Execute only this task: Stay at planning level and look only at the route-a path.",
                    "remaining_steps": ["choose one active correction or closure"],
                    "selected_route_family": "route_a",
                },
                "route_focus": {
                    "label": "route-a path",
                    "route_family": "route_a",
                    "correction_mode": "lossless baseline first",
                    "correction_target": "residual term",
                },
                "proposal": {
                    "thought_step": "Use the route-balance theorem as the governing relation for both bodies.",
                    "used_models": ["Route Balance Model"],
                },
                "calculation": {},
                "evaluation": {
                    "domain_consistency": 1.0,
                    "variable_grounding": 1.0,
                    "contextual_relevance": 1.0,
                    "simplicity_hint": 1.0,
                    "reason": "The route-local route_a claim stays at planning level.",
                    "hard_rule_violations": [
                        "step_focus mismatch with current task",
                        "no solving final answer; only one route-local step executed",
                    ],
                },
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["evaluation_passed"])
        self.assertEqual(node.equations, [])
        self.assertEqual(
            node.known_vars["ignored_review_rule_violations"],
            [
                "step_focus mismatch with current task",
                "no solving final answer; only one route-local step executed",
            ],
        )

    def test_route_focused_incremental_refinement_child_ignores_review_hard_rule_false_positives(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "phase": "incremental_refinement",
                    "current_step_guidance": "Execute only this task: Refine only the current subproblem: choose one active correction or closure.",
                    "previous_steps": ["identify governing relation"],
                    "remaining_steps": ["express the target quantity in known variables"],
                    "selected_route_family": "route-b",
                },
                "route_focus": {
                    "label": "route-b route",
                    "route_family": "route-b",
                    "correction_mode": "direct relation inventory",
                    "correction_target": "active relation term",
                },
                "proposal": {
                    "thought_step": "Add the route-b closure for the segment as the active correction.",
                    "equations": ["a = g*sin(theta)/(1 + I/(m*r^2))"],
                },
                "calculation": {},
                "evaluation": {
                    "domain_consistency": 1.0,
                    "variable_grounding": 1.0,
                    "contextual_relevance": 1.0,
                    "simplicity_hint": 1.0,
                    "reason": "The correction stays local to the selected route.",
                    "hard_rule_violations": [
                        "duplicate task selection",
                        "deferred task overreach",
                    ],
                },
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["evaluation_passed"])
        self.assertEqual(
            node.known_vars["ignored_review_rule_violations"],
            ["duplicate task selection", "deferred task overreach"],
        )

    def test_state_transition_boundary_label_noise_does_not_prune_nonterminal_route_step(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "phase": "incremental_refinement",
                    "current_step_guidance": "Execute only this task: Refine only the current subproblem: choose one active correction or closure.",
                    "previous_steps": ["identify governing relation"],
                    "remaining_steps": ["express the target quantity in known variables"],
                    "is_terminal_step": False,
                    "selected_route_family": "state-transition",
                },
                "route_focus": {
                    "label": "state-transition route",
                    "route_family": "state-transition",
                    "correction_mode": "piecewise-state propagation",
                    "correction_target": "state transition",
                },
                "proposal": {
                    "thought_step": "Claim that on the controlled segment, the update is uniform with a = ug.",
                    "equations": ["x = v_i^2 / (2ug)"],
                    "used_models": ["State-transition relation"],
                    "boundary_conditions": {
                        "segment_type": "horizontal rough",
                        "motion_type": "uniformly decelerated",
                    },
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["hard_rule_check"]["passed"])
        self.assertEqual(
            node.known_vars["hard_rule_check"]["ignored_violations"],
            [
                "Boundary condition key is not grounded in equations or known variables: segment_type",
                "Boundary condition key is not grounded in equations or known variables: motion_type",
            ],
        )
        self.assertNotIn("hard_rule_violations", node.known_vars)

    def test_relation_balance_boundary_label_noise_does_not_prune_nonterminal_route_step(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "phase": "incremental_refinement",
                    "current_step_guidance": "Execute only this task: Refine only the current subproblem: choose one active correction or closure.",
                    "previous_steps": ["identify governing relation"],
                    "remaining_steps": ["express the target quantity in known variables"],
                    "is_terminal_step": False,
                    "selected_route_family": "route-b",
                },
                "route_focus": {
                    "label": "route-b route",
                    "route_family": "route-b",
                    "correction_mode": "direct relation inventory",
                    "correction_target": "active relation term",
                },
                "proposal": {
                    "thought_step": "Claim the segment relation balance with residual as the active correction.",
                    "equations": ["m a = m g sin(theta) - mu_k N"],
                    "used_models": ["Segment Balance Model"],
                    "boundary_conditions": {
                        "surface": "rough segment",
                        "motion": "translational",
                    },
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["hard_rule_check"]["passed"])
        self.assertEqual(
            node.known_vars["hard_rule_check"]["ignored_violations"],
            [
                "Boundary condition key is not grounded in equations or known variables: surface",
                "Boundary condition key is not grounded in equations or known variables: motion",
            ],
        )
        self.assertNotIn("hard_rule_violations", node.known_vars)

    def test_nonterminal_child_without_semantic_delta_is_reflected_before_calculation(self) -> None:
        parent_node = ToTNode(
            id="parent-1",
            thought_step="Add the state transition condition x_next = f(x) as the sole state-transition relation for the state-transition route.",
            equations=["v = wr"],
            known_vars={
                "route_family": "state-transition",
                "correction_mode": "piecewise-state propagation",
                "correction_target": "state transition",
            },
            used_models=["State-transition relation"],
        )
        fsm = NodeBuilderFSM(
            parent_node=parent_node,
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "current_step_guidance": "Choose exactly one local correction.",
                    "phase": "incremental_refinement",
                    "remaining_steps": ["express the target quantity in known variables"],
                    "is_terminal_step": False,
                    "selected_route_family": "state-transition",
                },
                "route_focus": {
                    "label": "state-transition route",
                    "route_family": "state-transition",
                    "correction_mode": "piecewise-state propagation",
                    "correction_target": "state transition",
                },
            },
            max_reflections=1,
            backend_adapter=RepeatedChildProposalBackendAdapter(),
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertEqual(
            node.thought_step,
            "Add the horizontal update closure a = ug and expose u as the active control parameter for the state-transition route.",
        )
        self.assertEqual(node.known_vars["active_control_parameter"], "u")
        self.assertEqual(node.quantities["mu"], "control coefficient")
        self.assertIn(
            "Non-terminal proposal repeated its parent without adding one explicit correction",
            node.reflection_history[0],
        )
        self.assertEqual(node.known_vars["node_event_log"][0]["trigger"], "missing-semantic-delta")

    def test_nonterminal_child_without_semantic_delta_is_soft_pruned_when_retry_budget_is_zero(self) -> None:
        parent_node = ToTNode(
            id="parent-1",
            thought_step="Add the state transition condition x_next = f(x) as the sole state-transition relation for the state-transition route.",
            equations=["v = wr"],
            known_vars={
                "route_family": "state-transition",
                "correction_mode": "piecewise-state propagation",
                "correction_target": "state transition",
            },
            used_models=["State-transition relation"],
        )
        fsm = NodeBuilderFSM(
            parent_node=parent_node,
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "current_step_guidance": "Choose exactly one local correction.",
                    "phase": "incremental_refinement",
                    "remaining_steps": ["express the target quantity in known variables"],
                    "is_terminal_step": False,
                    "selected_route_family": "state-transition",
                },
                "route_focus": {
                    "label": "state-transition route",
                    "route_family": "state-transition",
                    "correction_mode": "piecewise-state propagation",
                    "correction_target": "state transition",
                },
            },
            max_reflections=0,
            backend_adapter=RepeatedChildProposalBackendAdapter(),
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.PRUNED_BY_SLM)
        self.assertEqual(node.result_state, NodeResultState.DROP)
        self.assertFalse(node.known_vars["evaluation_passed"])
        self.assertFalse(node.known_vars["needs_deeper_reasoning"])
        self.assertEqual(node.known_vars["expansion_priority"], 0.0)
        self.assertIn("semantic_delta_critique", node.known_vars)

    def test_route_local_strategy_scan_boundary_label_noise_does_not_prune_root_route(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "route-local scan: regime-map",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Execute only this task: Stay at planning level and look only at the regime-map route. Make one tiny route-local step only.",
                    "previous_steps": [],
                    "remaining_steps": [
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                    "selected_route_family": "regime-map",
                },
                "route_focus": {
                    "label": "regime-map route",
                    "route_family": "regime-map",
                    "correction_mode": "regime-selection scan",
                    "correction_target": "answer-flip boundary",
                },
                "proposal": {
                    "thought_step": "Identify the transition between segment-dominated and horizontal-residual-dominated regimes as a potential answer-flip threshold.",
                    "boundary_conditions": {
                        "active_boundary_condition": "compare state resistance on segment vs control residual work on horizontal segment",
                    },
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=1,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.result_state, NodeResultState.PASS)
        self.assertTrue(node.known_vars["hard_rule_check"]["passed"])
        self.assertEqual(
            node.known_vars["hard_rule_check"]["ignored_violations"],
            [
                "Boundary condition key is not grounded in equations or known variables: active_boundary_condition",
            ],
        )

    def test_strategy_scan_planning_only_proposal_does_not_inject_placeholder_equation(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": ["identify governing relation", "choose one active correction or closure"],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "identify governing relation",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Execute only this task: Analyze the next-step strategy space at planning level.",
                    "remaining_steps": ["choose one active correction or closure"],
                },
                "proposal": {
                    "thought_step": "Select the governing model only and defer algebra.",
                    "used_models": ["Route Balance Model"],
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.equations, [])
        self.assertTrue(node.known_vars["hard_rule_check"]["passed"])
        self.assertNotIn("recoverable_rule_violations", node.known_vars)

    def test_choose_correction_step_does_not_inject_placeholder_equation(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=ToTNode(id="parent-1"),
            problem_context={
                "meta_task": {
                    "first_step": "identify governing relation",
                    "step_ordering": [
                        "identify governing relation",
                        "choose one active correction or closure",
                        "express the target quantity in known variables",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 1,
                    "current_step": "choose one active correction or closure",
                    "phase": "incremental_refinement",
                    "current_step_guidance": "Execute only this task: Refine only the current subproblem: choose one active correction or closure.",
                    "previous_steps": ["identify governing relation"],
                    "remaining_steps": ["express the target quantity in known variables"],
                    "selected_route_family": "regime-map",
                },
                "route_focus": {
                    "label": "regime-map route",
                    "route_family": "regime-map",
                    "correction_mode": "regime-selection scan",
                    "correction_target": "answer-flip boundary",
                },
                "proposal": {
                    "thought_step": "Add one answer-flip boundary parameter and defer the final relation.",
                    "known_vars": {"eta_crit": "1"},
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.equations, [])
        self.assertTrue(node.known_vars["hard_rule_check"]["passed"])

    def test_meta_task_scope_keeps_single_step_branch_active(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "compare the most plausible governing routes for the segment segment",
                    "step_ordering": [
                        "compare the most plausible governing routes for the segment segment",
                        "solve for the value at the bottom of the segment",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare the most plausible governing routes for the segment segment",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level: compare plausible governing routes, identify the main laws/models each route would use, mark the key unknowns and deferred correction terms, and decide what should be refined next. Do not solve for the final answer yet.",
                    "remaining_steps": ["solve for the value at the bottom of the segment"],
                },
                "proposal": {
                    "thought_step": "Compare an route-a path against a route-b route for the segment, note that residual is the first deferred correction, and leave the actual value derivation for the next refinement step.",
                    "equations": ["candidate route A: route-balance balance", "candidate route B: tangential relation balance"],
                    "used_models": ["Route Balance Model", "Segment Balance Model"],
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["phase"], "strategy_scan")
        self.assertIn(
            "strategy space",
            node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["current_step_guidance"],
        )
        self.assertEqual(node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["current_step_index"], 0)
        self.assertEqual(node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["future_step_matches"], [])

    def test_strategy_scan_route_summary_with_descriptive_boundary_context_stays_active(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "meta_task": {
                    "first_step": "compare two route families before committing to a derivation",
                    "step_ordering": [
                        "compare two route families before committing to a derivation",
                        "derive the segment-stage value expression",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare two route families before committing to a derivation",
                    "phase": "strategy_scan",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level: compare plausible governing routes, identify the main laws/models each route would use, mark the key unknowns and deferred correction terms, and decide what should be refined next. Do not solve for the final answer yet.",
                    "remaining_steps": ["derive the segment-stage value expression"],
                },
                "proposal": {
                    "thought_step": "Strategy Scan Phase: Comparing two primary solution routes and deferring the detailed derivation until a later refinement step.",
                    "equations": [
                        "Route route (proposed): v_f² = 2gh_eff - 2u_k gL",
                        "segmented route (proposed): v_edge² = v_bottom² - 2u_k gL, then v_f² = v_edge² + 2gH",
                    ],
                    "used_models": ["Route Balance Model", "Segment Balance Model"],
                    "boundary_conditions": {
                        "Initial state: v=0 at position s=0": None,
                        "Final state: symbolic impact condition retained for later synthesis": None,
                    },
                },
                "calculation": {},
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertEqual(node.known_vars["hard_rule_check"]["checked"]["semantic_boundary_violations"], [])
        self.assertTrue(node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["comparative_strategy_scan"])
        self.assertEqual(node.known_vars["hard_rule_check"]["checked"]["meta_task_step_scope"]["future_step_matches"], [])

    def test_missing_required_boundary_rule_goes_to_reflection(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={
                "proposal": {
                    "equations": ["u_t = D u_xx"],
                    "used_models": ["1D diffusion"],
                    "boundary_conditions": {"x=0": "u(0,t)=u0"},
                },
                "calculation": {
                    "skill_params": {
                        "required_models": ["1D diffusion"],
                        "required_boundary_condition_keys": ["x=L"],
                    }
                },
                "reflection": {
                    "thought_step": "Add the missing boundary condition at x=L before proceeding.",
                    "equations": ["u_t = D u_xx"],
                    "used_models": ["1D diffusion"],
                    "boundary_conditions": {
                        "x=0": "u(0,t)=u0",
                        "x=L": "u(L,t)=uL",
                    },
                },
                "evaluation": {"score": 8.0},
            },
            max_reflections=0,
        )

        node = fsm.run()

        self.assertEqual(node.status, NodeStatus.ACTIVE)
        self.assertNotIn("recoverable_rule_violations", node.known_vars)
        self.assertIn("Missing required boundary condition key: x=L", node.reflection_history)
        self.assertEqual(node.boundary_conditions["x=L"], "u(L,t)=uL")

    def test_backend_adapter_output_is_schema_locked(self) -> None:
        fsm = NodeBuilderFSM(
            parent_node=None,
            problem_context={},
            backend_adapter=InvalidProposalBackendAdapter(),
        )

        with self.assertRaises(ValueError):
            fsm.run()


class TreeSchedulerTests(unittest.TestCase):
    def test_tree_scheduler_synthesizes_meta_task_children_without_explicit_children(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "problem_statement": "Advance through the meta-task one step at a time.",
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": [
                        "compare route families",
                        "refine bottom value",
                        "refine edge value",
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level: compare plausible governing routes, identify the main laws/models each route would use, mark the key unknowns and deferred correction terms, and decide what should be refined next. Do not solve for the final answer yet.",
                    "previous_steps": [],
                    "remaining_steps": ["refine bottom value", "refine edge value"],
                    "total_steps": 3,
                    "phase": "strategy_scan",
                    "is_terminal_step": False,
                },
            },
            expansion_budget=2,
            max_reflections=0,
            backend_adapter_factory=lambda _problem_context: MetaTaskStepBackendAdapter(),
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(result["expansions_used"], 2)
        self.assertEqual(len(root.children), 1)
        self.assertTrue(root.children[0].known_vars.get("selected_for_frontier"))
        self.assertEqual(root.children[0].thought_step, "Refine only the current subproblem: refine bottom value.")
        self.assertEqual(len(root.children[0].children), 1)
        self.assertEqual(root.children[0].children[0].thought_step, "Refine only the current subproblem: refine edge value.")
        self.assertEqual(result["frontier"], [])
        self.assertEqual(result["expansion_log"][0]["retained_frontier_ids"], [root.children[0].id])

    def test_tree_scheduler_synthesizes_route_focused_children_for_strategy_scan(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "problem_statement": "Branch into multiple route families before committing to one.",
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": [
                        "compare route families",
                        "refine bottom value",
                        "refine edge value",
                    ],
                    "route_options": [
                        {"label": "route-a path", "route_family": "route_a"},
                        {"label": "route-b route", "route_family": "route-b"},
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level.",
                    "previous_steps": [],
                    "remaining_steps": ["refine bottom value", "refine edge value"],
                    "total_steps": 3,
                    "phase": "strategy_scan",
                    "is_terminal_step": False,
                },
            },
            expansion_budget=2,
            max_frontier_size=4,
            max_children_per_expansion=2,
            max_reflections=0,
            backend_adapter_factory=lambda _problem_context: RouteFocusedMetaTaskBackendAdapter(),
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(result["expansions_used"], 2)
        self.assertEqual(len(root.children), 2)
        route_families = {str(child.known_vars.get("route_family", "")) for child in root.children}
        self.assertEqual(route_families, {"route_a", "route-b"})
        self.assertTrue(all("route only" in child.thought_step for child in root.children))
        frontier_route_families = {entry["node"].known_vars.get("route_family") for entry in scheduler._frontier}
        self.assertEqual(frontier_route_families, {"route_a", "route-b"})
        expanded_route_children = [child for child in root.children if child.children]
        self.assertEqual(len(expanded_route_children), 1)
        self.assertEqual(
            expanded_route_children[0].children[0].known_vars.get("route_family"),
            expanded_route_children[0].known_vars.get("route_family"),
        )
        diversity_keys = {str(child.known_vars.get("diversity_key", "")) for child in root.children}
        self.assertEqual(len(diversity_keys), 2)

    def test_tree_scheduler_keeps_distinct_root_routes_when_backend_seed_matches(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "problem_statement": "Branch into multiple route families before committing to one.",
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": [
                        "compare route families",
                        "refine bottom value",
                        "refine edge value",
                    ],
                    "route_options": [
                        {"label": "route-a path", "route_family": "route_a"},
                        {"label": "route-b route", "route_family": "route-b"},
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level.",
                    "previous_steps": [],
                    "remaining_steps": ["refine bottom value", "refine edge value"],
                    "total_steps": 3,
                    "phase": "strategy_scan",
                    "is_terminal_step": False,
                },
            },
            expansion_budget=2,
            max_frontier_size=4,
            max_children_per_expansion=2,
            max_reflections=0,
            backend_adapter_factory=lambda _problem_context: RouteNeutralMetaTaskBackendAdapter(),
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(len(root.children), 2)
        self.assertEqual(
            {str(child.known_vars.get("route_family", "")) for child in root.children},
            {"route_a", "route-b"},
        )
        self.assertEqual(
            [str(child.known_vars.get("scheduler_action", "")) for child in root.children],
            ["expanded", "expanded"],
        )

    def test_tree_scheduler_root_strategy_scan_uses_frontier_capacity_for_wide_fanout(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "problem_statement": "Branch into many route families before committing to one.",
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": [
                        "compare route families",
                        "refine bottom value",
                    ],
                    "route_options": [
                        {"label": "route-a path", "route_family": "route_a"},
                        {"label": "route-b route", "route_family": "route-b"},
                        {"label": "momentum route", "route_family": "momentum"},
                        {"label": "scaling route", "route_family": "scaling"},
                        {"label": "constraint route", "route_family": "constraint"},
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space broadly.",
                    "previous_steps": [],
                    "remaining_steps": ["refine bottom value"],
                    "total_steps": 2,
                    "phase": "strategy_scan",
                    "is_terminal_step": False,
                },
            },
            expansion_budget=5,
            max_frontier_size=5,
            max_children_per_expansion=2,
            max_reflections=0,
            backend_adapter_factory=lambda _problem_context: WideRouteFocusedMetaTaskBackendAdapter(),
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(len(root.children), 5)
        self.assertEqual(
            {str(child.known_vars.get("route_family", "")) for child in root.children},
            {"route_a", "route-b", "momentum", "scaling", "constraint"},
        )

    def test_tree_scheduler_limits_route_synthesis_to_surface_budget(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "problem_statement": "Keep branching within the available surface budget.",
                "meta_task": {
                    "first_step": "compare route families",
                    "step_ordering": [
                        "compare route families",
                        "refine bottom value",
                    ],
                    "route_options": [
                        {"label": "route-a path", "route_family": "route_a"},
                        {"label": "route-b route", "route_family": "route-b"},
                        {"label": "momentum route", "route_family": "momentum"},
                    ],
                },
                "meta_task_progress": {
                    "current_step_index": 0,
                    "current_step": "compare route families",
                    "current_step_guidance": "Analyze the next-step strategy space broadly.",
                    "previous_steps": [],
                    "remaining_steps": ["refine bottom value"],
                    "total_steps": 2,
                    "phase": "strategy_scan",
                    "is_terminal_step": False,
                },
            },
            expansion_budget=1,
            max_frontier_size=4,
            max_children_per_expansion=3,
            max_reflections=0,
            backend_adapter_factory=lambda _problem_context: RouteFocusedMetaTaskBackendAdapter(),
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(result["expansions_used"], 1)
        self.assertEqual(len(root.children), 3)
        self.assertEqual(
            {str(child.known_vars.get("route_family", "")) for child in root.children},
            {"route_a", "route-b", "momentum"},
        )
        self.assertEqual(
            {str(entry["node"].known_vars.get("route_family", "")) for entry in scheduler._frontier},
            {"route_a", "route-b", "momentum"},
        )

    def test_tree_scheduler_gap_fills_missing_route_family_before_repeating_one(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_route_a_a"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {"route_family": "route_a"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_a"]}},
                        "evaluation": {"score": 9.5},
                        "children": [{"proposal": {"equations": ["grand_route_a_a"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_a"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_route_a_b"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {"route_family": "route_a"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_b"]}},
                        "evaluation": {"score": 9.4},
                        "children": [{"proposal": {"equations": ["grand_route_a_b"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_b"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_relation"],
                            "used_models": ["Segment Balance Model"],
                            "known_vars": {"route_family": "route-b"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_relation"]}},
                        "evaluation": {"score": 7.0},
                        "children": [{"proposal": {"equations": ["grand_relation"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_relation"]}}, "evaluation": {"score": 8.0}}],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=2,
            max_children_per_expansion=3,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        retained_ids = set(result["expansion_log"][0]["retained_frontier_ids"])
        retained_route_families = {
            str(child.known_vars.get("route_family", ""))
            for child in root.children
            if child.id in retained_ids
        }

        self.assertEqual(retained_route_families, {"route_a", "route-b"})

    def test_tree_scheduler_logs_full_candidate_slice_before_diversity_rebalance(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_route_a_a"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {"route_family": "route_a"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_a"]}},
                        "evaluation": {"score": 9.5},
                        "children": [{"proposal": {"equations": ["grand_route_a_a"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_a"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_route_a_b"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {"route_family": "route_a"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_b"]}},
                        "evaluation": {"score": 9.4},
                        "children": [{"proposal": {"equations": ["grand_route_a_b"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_b"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_relation"],
                            "used_models": ["Segment Balance Model"],
                            "known_vars": {"route_family": "route-b"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_relation"]}},
                        "evaluation": {"score": 7.0},
                        "children": [{"proposal": {"equations": ["grand_relation"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_relation"]}}, "evaluation": {"score": 8.0}}],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=2,
            max_children_per_expansion=3,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        child_relation = next(child for child in root.children if child.equations == ["child_relation"])
        expansion_entry = result["expansion_log"][0]

        self.assertEqual(len(expansion_entry["frontier_candidate_ids"]), 3)
        self.assertIn(child_relation.id, expansion_entry["frontier_candidate_ids"])
        self.assertEqual(len(expansion_entry["retained_frontier_ids"]), 2)

    def test_tree_scheduler_keeps_distinct_correction_modes_within_same_route_family(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_route_a_linear_a"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {
                                "route_family": "route_a",
                                "correction_mode": "linear delayed closure closure",
                                "correction_target": "delayed closure coefficient",
                            },
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_linear_a"]}},
                        "evaluation": {"score": 9.5},
                        "children": [{"proposal": {"equations": ["grand_route_a_linear_a"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_linear_a"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_route_a_linear_b"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {
                                "route_family": "route_a",
                                "correction_mode": "linear delayed closure closure",
                                "correction_target": "delayed closure coefficient",
                            },
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_linear_b"]}},
                        "evaluation": {"score": 9.4},
                        "children": [{"proposal": {"equations": ["grand_route_a_linear_b"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_linear_b"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_route_a_quadratic"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {
                                "route_family": "route_a",
                                "correction_mode": "quadratic delayed closure closure",
                                "correction_target": "delayed closure coefficient",
                            },
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_quadratic"]}},
                        "evaluation": {"score": 7.0},
                        "children": [{"proposal": {"equations": ["grand_route_a_quadratic"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_quadratic"]}}, "evaluation": {"score": 8.0}}],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=2,
            max_children_per_expansion=3,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        retained_ids = set(result["expansion_log"][0]["retained_frontier_ids"])
        retained_correction_modes = {
            str(child.known_vars.get("correction_mode", ""))
            for child in root.children
            if child.id in retained_ids
        }

        self.assertEqual(retained_correction_modes, {"linear delayed closure closure", "quadratic delayed closure closure"})

    def test_tree_scheduler_rebalances_against_full_candidate_slice_with_existing_frontier_entries(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_route_a_root"],
                            "used_models": ["Route Balance Model"],
                            "known_vars": {"route_family": "route_a"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_route_a_root"]}},
                        "evaluation": {"score": 9.5},
                        "children": [
                            {
                                "proposal": {
                                    "equations": ["grand_route_a_a"],
                                    "used_models": ["Route Balance Model"],
                                    "known_vars": {"route_family": "route_a"},
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_a"]}},
                                "evaluation": {"score": 9.6},
                                "children": [{"proposal": {"equations": ["great_route_a_a"]}, "calculation": {"skill_params": {"required_equation_patterns": ["great_route_a_a"]}}, "evaluation": {"score": 8.0}}],
                            },
                            {
                                "proposal": {
                                    "equations": ["grand_route_a_b"],
                                    "used_models": ["Route Balance Model"],
                                    "known_vars": {"route_family": "route_a"},
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_route_a_b"]}},
                                "evaluation": {"score": 9.5},
                                "children": [{"proposal": {"equations": ["great_route_a_b"]}, "calculation": {"skill_params": {"required_equation_patterns": ["great_route_a_b"]}}, "evaluation": {"score": 8.0}}],
                            },
                            {
                                "proposal": {
                                    "equations": ["grand_relation"],
                                    "used_models": ["Segment Balance Model"],
                                    "known_vars": {"route_family": "route-b"},
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_relation"]}},
                                "evaluation": {"score": 7.0},
                                "children": [{"proposal": {"equations": ["great_relation"]}, "calculation": {"skill_params": {"required_equation_patterns": ["great_relation"]}}, "evaluation": {"score": 8.0}}],
                            },
                        ],
                    },
                    {
                        "proposal": {
                            "equations": ["child_momentum_root"],
                            "used_models": ["Linear Momentum"],
                            "known_vars": {"route_family": "momentum"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_momentum_root"]}},
                        "evaluation": {"score": 8.8},
                        "children": [{"proposal": {"equations": ["grand_momentum"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_momentum"]}}, "evaluation": {"score": 8.0}}],
                    },
                ],
            },
            expansion_budget=2,
            max_frontier_size=2,
            max_children_per_expansion=3,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        child_route_a = next(child for child in root.children if child.equations == ["child_route_a_root"])
        grand_relation = next(child for child in child_route_a.children if child.equations == ["grand_relation"])
        expansion_entry = next(
            entry for entry in result["expansion_log"] if entry.get("parent_id") == child_route_a.id
        )

        self.assertEqual(len(expansion_entry["frontier_candidate_ids"]), 3)
        self.assertIn(grand_relation.id, expansion_entry["frontier_candidate_ids"])
        self.assertEqual(len(result["frontier"]), 2)

    def test_tree_scheduler_logs_pruned_nodes_in_activity_log(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["eq-root"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["eq-root"]},
                },
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["eq-bad"]},
                        "calculation": {
                            "skill_params": {"forbidden_equation_patterns": ["eq-bad"]},
                        },
                    }
                ],
            },
            expansion_budget=1,
            max_reflections=0,
        )

        result = scheduler.run()
        pruned_entries = [entry for entry in result["expansion_log"] if entry.get("event") == "pruned-by-rule"]

        self.assertEqual(len(pruned_entries), 1)
        self.assertEqual(pruned_entries[0]["source_state"], "calculate")
        self.assertIn("Equation matches forbidden pattern: eq-bad", pruned_entries[0]["violations"])
        self.assertEqual(pruned_entries[0]["status"], NodeStatus.PRUNED_BY_RULE.value)

    def test_tree_scheduler_logs_root_bounce_back_in_activity_log(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {
                    "thought_step": "Start from a forbidden root draft.",
                    "equations": ["eq0"],
                },
                "calculation": {
                    "skill_params": {"forbidden_equation_patterns": ["eq0"]},
                },
                "reflection": {
                    "thought_step": "Replace the forbidden root draft with a revised seed branch.",
                    "equations": ["eq1"],
                },
                "evaluation": {"score": 8.0},
            },
            expansion_budget=0,
            max_reflections=0,
        )

        result = scheduler.run()
        bounce_entries = [entry for entry in result["expansion_log"] if entry.get("event") == "bounce-to-reflection"]

        self.assertEqual(len(bounce_entries), 1)
        self.assertEqual(bounce_entries[0]["source_state"], "calculate")
        self.assertEqual(bounce_entries[0]["trigger"], "root-rule-violation")
        self.assertIn("Equation matches forbidden pattern: eq0", bounce_entries[0]["violations"])
        self.assertEqual(result["root"].status, NodeStatus.ACTIVE)

    def test_tree_scheduler_ranks_siblings_and_respects_budget(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["root_eq"]}
                },
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_a"],
                            "used_models": ["diffusion"],
                            "boundary_conditions": {"x=0": "fixed"},
                        },
                        "calculation": {
                            "skill_params": {
                                "required_equation_patterns": ["child_a"],
                                "required_models": ["diffusion"],
                                "required_boundary_condition_keys": ["x=0"],
                            }
                        },
                        "evaluation": {"score": 9.0},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_a"]},
                                "calculation": {
                                    "skill_params": {"required_equation_patterns": ["grand_a"]}
                                },
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    },
                    {
                        "proposal": {"equations": ["child_b"]},
                        "calculation": {
                            "skill_params": {"required_equation_patterns": ["child_b"]}
                        },
                        "evaluation": {
                            "domain_consistency": 0.35,
                            "variable_grounding": 0.4,
                            "contextual_relevance": 0.5,
                        },
                        "children": [
                            {
                                "proposal": {"equations": ["grand_b"]},
                                "calculation": {
                                    "skill_params": {"required_equation_patterns": ["grand_b"]}
                                },
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    },
                    {
                        "proposal": {
                            "equations": ["child_c"],
                            "used_models": ["bad_model"],
                        },
                        "calculation": {
                            "skill_params": {
                                "required_equation_patterns": ["child_c"],
                                "forbidden_models": ["bad_model"],
                            }
                        },
                        "evaluation": {"score": 9.0},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_c"]},
                                "calculation": {
                                    "skill_params": {"required_equation_patterns": ["grand_c"]}
                                },
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    },
                ],
            },
            expansion_budget=2,
            max_frontier_size=1,
            max_children_per_expansion=2,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]

        self.assertEqual(result["expansions_used"], 2)
        self.assertEqual(result["expanded_node_ids"][0], root.id)
        self.assertEqual(len(root.children), 3)

        ranked_children = sorted(root.children, key=lambda child: child.known_vars["sibling_rank"])
        self.assertEqual(ranked_children[0].status, NodeStatus.ACTIVE)
        self.assertTrue(ranked_children[0].known_vars["selected_for_frontier"])
        self.assertFalse(ranked_children[1].known_vars["selected_for_frontier"])
        self.assertEqual(ranked_children[2].status, NodeStatus.PRUNED_BY_RULE)
        self.assertEqual(result["expanded_node_ids"][1], ranked_children[0].id)
        self.assertEqual(root.known_vars["selected_child_ids"], [ranked_children[0].id])

    def test_tree_scheduler_prefers_deeper_frontier_nodes_for_next_expansion(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["root_eq"]}
                },
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_a"]},
                        "calculation": {
                            "skill_params": {"required_equation_patterns": ["child_a"]}
                        },
                        "evaluation": {"score": 9.3},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_a"]},
                                "calculation": {
                                    "skill_params": {"required_equation_patterns": ["grand_a"]}
                                },
                                "evaluation": {"score": 9.9},
                                "children": [
                                    {
                                        "proposal": {"equations": ["great_a"]},
                                        "calculation": {
                                            "skill_params": {"required_equation_patterns": ["great_a"]}
                                        },
                                        "evaluation": {"score": 8.4},
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "proposal": {"equations": ["child_b"]},
                        "calculation": {
                            "skill_params": {"required_equation_patterns": ["child_b"]}
                        },
                        "evaluation": {"score": 6.2},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_b"]},
                                "calculation": {
                                    "skill_params": {"required_equation_patterns": ["grand_b"]}
                                },
                                "evaluation": {"score": 7.8},
                            }
                        ],
                    },
                ],
            },
            expansion_budget=3,
            max_frontier_size=2,
            max_children_per_expansion=2,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        child_a = next(child for child in root.children if child.equations == ["child_a"])
        child_b = next(child for child in root.children if child.equations == ["child_b"])
        grand_a = child_a.children[0]

        self.assertEqual(result["expanded_node_ids"], [root.id, child_a.id, grand_a.id])
        self.assertEqual(result["frontier"][0]["depth"], 1)
        self.assertEqual(result["frontier"][0]["node_id"], child_b.id)

    def test_tree_scheduler_enrelations_diversity_cap(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "equations": ["child_a"],
                            "used_models": ["diffusion"],
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_a"]}},
                        "evaluation": {"score": 9.0},
                        "children": [{"proposal": {"equations": ["grand_a"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_a"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_b"],
                            "used_models": ["diffusion"],
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_b"]}},
                        "evaluation": {"score": 8.9},
                        "children": [{"proposal": {"equations": ["grand_b"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_b"]}}, "evaluation": {"score": 8.0}}],
                    },
                    {
                        "proposal": {
                            "equations": ["child_c"],
                            "used_models": ["wave"],
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_c"]}},
                        "evaluation": {"score": 8.7},
                        "children": [{"proposal": {"equations": ["grand_c"]}, "calculation": {"skill_params": {"required_equation_patterns": ["grand_c"]}}, "evaluation": {"score": 8.0}}],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=2,
            max_children_per_expansion=3,
            max_frontier_per_diversity_key=1,
            max_reflections=0,
        )

        result = scheduler.run()
        frontier_ids = [entry["node_id"] for entry in result["frontier"]]
        root = result["root"]
        selected_models = {
            tuple(sorted(child.used_models))
            for child in root.children
            if child.id in frontier_ids
        }

        self.assertEqual(len(result["frontier"]), 2)
        self.assertEqual(selected_models, {("diffusion",), ("wave",)})

    def test_tree_scheduler_merges_duplicates_and_suppresses_loops(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"], "used_models": ["root_model"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["dup_eq"], "used_models": ["dup_model"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["dup_eq"]}},
                        "evaluation": {"score": 9.0},
                        "children": [],
                    },
                    {
                        "proposal": {"equations": ["dup_eq"], "used_models": ["dup_model"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["dup_eq"]}},
                        "evaluation": {"score": 8.5},
                        "children": [],
                    },
                    {
                        "proposal": {"equations": ["root_eq"], "used_models": ["root_model"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                        "evaluation": {"score": 8.2},
                        "children": [],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=3,
            max_children_per_expansion=3,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        by_rank = sorted(root.children, key=lambda child: child.known_vars["sibling_rank"])

        self.assertEqual(by_rank[0].known_vars["scheduler_action"], "expanded")
        self.assertEqual(by_rank[1].known_vars["scheduler_action"], "merged-duplicate")
        self.assertEqual(by_rank[2].known_vars["scheduler_action"], "suppressed-loop")
        self.assertEqual(by_rank[1].known_vars["merged_into_node_id"], by_rank[0].id)
        self.assertEqual(by_rank[2].known_vars["merged_into_node_id"], root.id)

    def test_tree_scheduler_merges_repeated_divergent_logic_across_slots(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {
                    "equations": ["root_eq"],
                    "used_models": ["root_model"],
                    "thought_step": "root route",
                },
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "thought_step": "apply the boundary constraint at x = 0",
                            "equations": ["u(0) = 0"],
                            "used_models": ["diffusion"],
                            "known_vars": {"active_boundary_condition": "u(0)=0"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["u(0) = 0"]}},
                        "evaluation": {"score": 9.0},
                        "route_focus": {
                            "route_family": "constraint",
                            "correction_mode": "constraint-first refinement",
                            "correction_target": "active boundary or constraint",
                            "slot": "constraint.1.0",
                        },
                        "children": [],
                    },
                    {
                        "proposal": {
                            "thought_step": "apply the boundary constraint at x = 0",
                            "equations": ["u(0) = 0"],
                            "used_models": ["diffusion"],
                            "known_vars": {"active_boundary_condition": "u(0)=0"},
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["u(0) = 0"]}},
                        "evaluation": {"score": 8.7},
                        "route_focus": {
                            "route_family": "constraint",
                            "correction_mode": "consistency-check refinement",
                            "correction_target": "local consistency condition",
                            "slot": "constraint.1.1",
                        },
                        "children": [],
                    },
                ],
            },
            expansion_budget=1,
            max_frontier_size=2,
            max_children_per_expansion=2,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        by_rank = sorted(root.children, key=lambda child: child.known_vars["sibling_rank"])

        self.assertEqual(by_rank[0].known_vars["scheduler_action"], "expanded")
        self.assertEqual(by_rank[1].known_vars["scheduler_action"], "merged-duplicate")
        self.assertEqual(by_rank[1].known_vars["merged_into_node_id"], by_rank[0].id)

    def test_tree_scheduler_strictly_merges_semantically_repeated_non_root_leaf_divergence(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {
                    "thought_step": "root route",
                    "equations": ["root_eq"],
                    "used_models": ["root_model"],
                },
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "thought_step": "select the constraint route",
                            "equations": ["child_eq"],
                            "used_models": ["diffusion"],
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 9.0},
                        "children": [
                            {
                                "proposal": {
                                    "thought_step": "apply the boundary constraint at x = 0 before closing the route",
                                    "equations": ["u(0) = 0"],
                                    "known_vars": {"active_boundary_condition": "u(0)=0"},
                                    "used_models": ["diffusion"],
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["u(0) = 0"]}},
                                "evaluation": {"score": 9.0},
                                "meta_task_progress": {
                                    "current_step_index": 2,
                                    "phase": "incremental_refinement",
                                },
                                "route_focus": {
                                    "route_family": "constraint",
                                    "correction_mode": "constraint-first refinement",
                                    "correction_target": "active boundary or constraint",
                                    "slot": "constraint.2.0",
                                },
                            },
                            {
                                "proposal": {
                                    "thought_step": "enforce the x=0 boundary condition before closing the branch",
                                    "equations": ["u(0)=0"],
                                    "known_vars": {"active_boundary_condition": "u(0) = 0"},
                                    "used_models": ["diffusion"],
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["u(0)=0"]}},
                                "evaluation": {"score": 8.8},
                                "meta_task_progress": {
                                    "current_step_index": 2,
                                    "phase": "incremental_refinement",
                                },
                                "route_focus": {
                                    "route_family": "constraint",
                                    "correction_mode": "consistency-check refinement",
                                    "correction_target": "local consistency condition",
                                    "slot": "constraint.2.1",
                                },
                            },
                        ],
                    }
                ],
            },
            expansion_budget=2,
            max_frontier_size=2,
            max_children_per_expansion=2,
            max_reflections=0,
        )

        result = scheduler.run()
        child = result["root"].children[0]
        by_rank = sorted(child.children, key=lambda grandchild: grandchild.known_vars["sibling_rank"])

        self.assertEqual(by_rank[0].known_vars["scheduler_action"], "expanded")
        self.assertEqual(by_rank[1].known_vars["scheduler_action"], "merged-semantic-duplicate")
        self.assertEqual(by_rank[1].known_vars["merged_into_node_id"], by_rank[0].id)

    def test_tree_scheduler_merges_semantic_duplicates_without_slot_marker(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {
                    "thought_step": "root route",
                    "equations": ["root_eq"],
                    "used_models": ["root_model"],
                },
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {
                            "thought_step": "select the constraint route",
                            "equations": ["child_eq"],
                            "used_models": ["diffusion"],
                        },
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 9.0},
                        "children": [
                            {
                                "proposal": {
                                    "thought_step": "enforce boundary condition at x zero before closure",
                                    "equations": ["u(0)=0"],
                                    "known_vars": {"active_boundary_condition": "u(0)=0"},
                                    "used_models": ["diffusion"],
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["u(0)=0"]}},
                                "evaluation": {"score": 8.9},
                                "meta_task_progress": {
                                    "current_step_index": 2,
                                    "phase": "incremental_refinement",
                                },
                                "route_focus": {
                                    "route_family": "constraint",
                                    "correction_mode": "constraint-first refinement",
                                    "correction_target": "active boundary or constraint",
                                },
                            },
                            {
                                "proposal": {
                                    "thought_step": "enforce boundary condition at x = 0 before closure",
                                    "equations": ["u(0) = 0"],
                                    "known_vars": {"active_boundary_condition": "u(0) = 0"},
                                    "used_models": ["diffusion"],
                                },
                                "calculation": {"skill_params": {"required_equation_patterns": ["u(0) = 0"]}},
                                "evaluation": {"score": 8.7},
                                "meta_task_progress": {
                                    "current_step_index": 2,
                                    "phase": "incremental_refinement",
                                },
                                "route_focus": {
                                    "route_family": "constraint",
                                    "correction_mode": "consistency-check refinement",
                                    "correction_target": "local consistency condition",
                                },
                            },
                        ],
                    }
                ],
            },
            expansion_budget=2,
            max_frontier_size=2,
            max_children_per_expansion=2,
            max_reflections=0,
        )

        result = scheduler.run()
        child = result["root"].children[0]
        by_rank = sorted(child.children, key=lambda grandchild: grandchild.known_vars["sibling_rank"])

        self.assertEqual(by_rank[0].known_vars["scheduler_action"], "expanded")
        self.assertEqual(by_rank[1].known_vars["scheduler_action"], "merged-semantic-duplicate")
        self.assertEqual(by_rank[1].known_vars["merged_into_node_id"], by_rank[0].id)

    def test_tree_scheduler_can_persist_and_resume(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_eq"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 9.0},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_eq"]},
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_eq"]}},
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    }
                ],
            },
            expansion_budget=1,
            max_frontier_size=1,
            max_children_per_expansion=1,
            max_reflections=0,
        )

        first = scheduler.run()
        self.assertEqual(first["expansions_used"], 1)
        self.assertEqual(len(first["frontier"]), 1)

        with TemporaryDirectory() as temp_dir:
            state_path = f"{temp_dir}/scheduler_state.json"
            scheduler.save_state(state_path)
            resumed = ToTTreeScheduler.from_state_file(state_path)
            resumed.expansion_budget = 2
            second = resumed.run()

        self.assertEqual(second["expansions_used"], 2)
        self.assertEqual(second["remaining_budget"], 0)
        self.assertEqual(len(second["expanded_node_ids"]), 2)

    def test_delete_node_requires_ai_review(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_eq"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 8.0},
                    }
                ],
            },
            expansion_budget=1,
            max_frontier_size=1,
            max_children_per_expansion=1,
            max_reflections=0,
        )

        result = scheduler.run()
        child_id = result["root"].children[0].id

        with self.assertRaises(ValueError):
            scheduler.delete_node(child_id, reason="frontend cleanup")

    def test_delete_node_rejected_review_does_not_mutate_tree(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_eq"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 8.0},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_eq"]},
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_eq"]}},
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    }
                ],
            },
            expansion_budget=1,
            max_frontier_size=1,
            max_children_per_expansion=1,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        child_id = root.children[0].id
        response = scheduler.delete_node(
            child_id,
            reason="frontend cleanup",
            review_adapter=RejectDeleteReviewAdapter(),
        )

        self.assertFalse(response["deleted"])
        self.assertEqual(len(root.children), 1)
        self.assertEqual(response["frontier"][0]["node_id"], child_id)

    def test_delete_node_approved_review_updates_tree_state(self) -> None:
        scheduler = ToTTreeScheduler(
            root_problem_context={
                "proposal": {"equations": ["root_eq"]},
                "calculation": {"skill_params": {"required_equation_patterns": ["root_eq"]}},
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_eq"]},
                        "calculation": {"skill_params": {"required_equation_patterns": ["child_eq"]}},
                        "evaluation": {"score": 8.0},
                        "children": [
                            {
                                "proposal": {"equations": ["grand_eq"]},
                                "calculation": {"skill_params": {"required_equation_patterns": ["grand_eq"]}},
                                "evaluation": {"score": 8.0},
                            }
                        ],
                    }
                ],
            },
            expansion_budget=1,
            max_frontier_size=1,
            max_children_per_expansion=1,
            max_reflections=0,
        )

        result = scheduler.run()
        root = result["root"]
        child_id = root.children[0].id
        response = scheduler.delete_node(
            child_id,
            reason="frontend cleanup",
            review_adapter=ApproveDeleteReviewAdapter(),
        )

        self.assertTrue(response["deleted"])
        self.assertEqual(response["deleted_node_ids"], [child_id])
        self.assertEqual(root.children, [])
        self.assertEqual(response["frontier"], [])
        self.assertNotIn(child_id, scheduler._node_index)


if __name__ == "__main__":
    unittest.main()