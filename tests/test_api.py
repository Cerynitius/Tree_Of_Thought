import os
import unittest
import time

from fastapi.testclient import TestClient

from fsm import (
    ChatBackendTransportError,
    DeterministicContextBackendAdapter,
    EvaluationRequest,
    NodeDeletionReviewAdapter,
    ProposalRequest,
    ReasoningBackendAdapter,
    ReflectionRequest,
    ToTTreeScheduler,
)
from tests.test_harness import ApproveDeleteReviewAdapter, MetaTaskStepBackendAdapter
from tot_api import (
    DEFAULT_PROBLEM_CONTEXT_DRAFT,
    ChatBackendConfig,
    SchedulerSessionStore,
    create_app,
)


def deterministic_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        return DeterministicContextBackendAdapter(problem_context)

    return backend_factory, ApproveDeleteReviewAdapter()


CHAT_BACKEND_ENV_KEYS = (
    "CHAT_BASE_URL",
    "CHAT_TIMEOUT",
    "PLANNING_MODEL",
    "MODELING_MODEL",
    "REVIEW_MODEL",
    "NON_TERMINAL_EVALUATION_MODEL",
    "ALLOW_LIVE_MODEL_FALLBACK",
    "PREFER_LOCAL_FALLBACK",
)


def clear_chat_backend_env(test: unittest.TestCase) -> None:
    """Remove local .env overrides so ChatBackendConfig tests see built-in defaults."""

    for key in CHAT_BACKEND_ENV_KEYS:
        original = os.environ.pop(key, None)
        if original is not None:
            test.addCleanup(os.environ.__setitem__, key, original)


class FailingBackendAdapter(ReasoningBackendAdapter):
    name = "failing-backend"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        del request
        raise ChatBackendTransportError("local model backend unavailable")

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        del request
        raise ChatBackendTransportError("local model backend unavailable")

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        del request
        raise ChatBackendTransportError("local model backend unavailable")


class PreparingBackendAdapter(DeterministicContextBackendAdapter):
    name = "preparing-backend"

    def prepare_problem_context(self, problem_context: dict[str, object]) -> dict[str, object]:
        prepared = dict(problem_context)
        prepared["meta_task"] = {
            "objective": "prepared once during create_session",
            "step_ordering": ["first", "second"],
        }
        return prepared


class SlowPreparingBackendAdapter(DeterministicContextBackendAdapter):
    name = "slow-preparing-backend"

    def prepare_problem_context(self, problem_context: dict[str, object]) -> dict[str, object]:
        time.sleep(0.25)
        prepared = dict(problem_context)
        prepared["meta_task"] = {
            "objective": "slow background preparation",
            "step_ordering": ["first", "second"],
        }
        return prepared


class SlowProgressBackendAdapter(DeterministicContextBackendAdapter):
    name = "slow-progress-backend"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        time.sleep(0.12)
        return super().propose(request)

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        time.sleep(0.12)
        return super().evaluate(request)

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        return super().reflect(request)


class VerySlowProgressBackendAdapter(DeterministicContextBackendAdapter):
    name = "very-slow-progress-backend"

    def propose(self, request: ProposalRequest) -> dict[str, object]:
        time.sleep(0.25)
        return super().propose(request)

    def evaluate(self, request: EvaluationRequest) -> dict[str, object]:
        time.sleep(0.25)
        return super().evaluate(request)

    def reflect(self, request: ReflectionRequest) -> dict[str, object]:
        return super().reflect(request)


class NoOpDeletionReviewAdapter(NodeDeletionReviewAdapter):
    name = "noop-delete-review"

    def review_delete_node(self, request):
        del request
        return {"approved": True, "reason": "ok", "risk_level": "low"}


def failing_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        del problem_context
        return FailingBackendAdapter()

    return backend_factory, NoOpDeletionReviewAdapter()


def preparing_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        return PreparingBackendAdapter(problem_context)

    return backend_factory, ApproveDeleteReviewAdapter()


def slow_preparing_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        return SlowPreparingBackendAdapter(problem_context)

    return backend_factory, ApproveDeleteReviewAdapter()


def slow_progress_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        return SlowProgressBackendAdapter(problem_context)

    return backend_factory, ApproveDeleteReviewAdapter()


def very_slow_progress_adapter_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        return VerySlowProgressBackendAdapter(problem_context)

    return backend_factory, ApproveDeleteReviewAdapter()


def meta_task_auto_bundle(_config: ChatBackendConfig):
    def backend_factory(problem_context: dict[str, object]):
        del problem_context
        return MetaTaskStepBackendAdapter()

    return backend_factory, ApproveDeleteReviewAdapter()


class ToTAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=deterministic_adapter_bundle,
            )
        )

    def test_frontend_index_is_served(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ToT Terminal", response.text)
        self.assertIn("problemPromptInput", response.text)
        self.assertIn("resultsPanelTitle", response.text)
        self.assertIn("candidateResults", response.text)
        self.assertIn("/static/app.js", response.text)

    def test_chat_backend_config_defaults_to_live_only_local_qwen_settings(self) -> None:
        clear_chat_backend_env(self)
        config = ChatBackendConfig()

        self.assertEqual(config.timeout, 600.0)
        self.assertFalse(config.allow_live_model_fallback)
        self.assertFalse(config.prefer_local_fallback)

    def test_chat_backend_config_defaults_to_local_qwen_model_for_all_roles(self) -> None:
        clear_chat_backend_env(self)
        config = ChatBackendConfig()

        self.assertEqual(config.planning_model, "qwen3.5-9b-mlx")
        self.assertEqual(config.modeling_model, "qwen3.5-9b-mlx")
        self.assertEqual(config.review_model, "qwen3.5-9b-mlx")
        self.assertEqual(config.non_terminal_evaluation_model, "qwen3.5-9b-mlx")

    def test_defaults_endpoint_returns_frontend_drafts(self) -> None:
        response = self.client.get("/api/tot/defaults")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["problem_context"], DEFAULT_PROBLEM_CONTEXT_DRAFT)
        self.assertEqual(payload["scheduler"]["depth_preset"], "medium")
        self.assertEqual(payload["scheduler"]["max_tree_depth"], 8)
        self.assertEqual(payload["scheduler"]["max_live_children_per_batch"], 2)
        self.assertTrue(payload["scheduler"]["use_local_root_proposal"])
        self.assertTrue(payload["scheduler"]["use_local_root_evaluation"])
        self.assertTrue(payload["scheduler"]["use_local_child_proposal"])
        self.assertTrue(payload["scheduler"]["use_local_child_evaluation"])
        self.assertEqual(payload["scheduler"]["children_key"], "children")
        self.assertNotIn("expansion_budget", payload["scheduler"])

    def test_active_session_cap_reacquire_respects_fifo_wait_order(self) -> None:
        store = SchedulerSessionStore(max_active_auto_runs=1)
        first_session_id = store.create(ToTTreeScheduler(root_problem_context={"problem_statement": "first"}))
        second_session_id = store.create(ToTTreeScheduler(root_problem_context={"problem_statement": "second"}))

        self.assertTrue(store.try_acquire_auto_run_slot(first_session_id))
        self.assertFalse(store.try_acquire_auto_run_slot(second_session_id))

        store.release_auto_run_slot(first_session_id)

        self.assertFalse(store.try_acquire_auto_run_slot(first_session_id))
        self.assertTrue(store.try_acquire_auto_run_slot(second_session_id))

        store.release_auto_run_slot(second_session_id)

        self.assertTrue(store.try_acquire_auto_run_slot(first_session_id))

    def test_create_session_injects_reasoning_depth_preset_into_backend_context(self) -> None:
        captured = {}

        def recording_bundle(config: ChatBackendConfig):
            del config

            def backend_factory(problem_context: dict[str, object]):
                captured["problem_context"] = dict(problem_context)
                return DeterministicContextBackendAdapter(problem_context)

            return backend_factory, ApproveDeleteReviewAdapter()

        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=recording_bundle,
            )
        )

        create = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "scheduler": {
                    "depth_preset": "high",
                },
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                },
            },
        )

        self.assertEqual(create.status_code, 200)
        session_id = create.json()["session_id"]
        run = client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 200)
        self.assertEqual(captured["problem_context"]["reasoning_depth_preset"], "high")

    def test_create_session_backfills_problem_statement_from_problem_prompt(self) -> None:
        captured = {}

        def recording_bundle(config: ChatBackendConfig):
            del config

            def backend_factory(problem_context: dict[str, object]):
                captured["problem_context"] = dict(problem_context)
                return DeterministicContextBackendAdapter(problem_context)

            return backend_factory, ApproveDeleteReviewAdapter()

        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=recording_bundle,
            )
        )

        create = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "problem_prompt": "Solve the spring-block problem.",
                "problem_context": {
                    "task": "task fallback",
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                },
            },
        )

        self.assertEqual(create.status_code, 200)
        session_id = create.json()["session_id"]
        run = client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 200)
        self.assertEqual(
            captured["problem_context"]["problem_statement"],
            "Solve the spring-block problem.",
        )

    def test_create_session_accepts_custom_non_terminal_evaluation_model(self) -> None:
        captured = {}

        def recording_bundle(config: ChatBackendConfig):
            captured["config"] = config
            return deterministic_adapter_bundle(config)

        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=recording_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "backend": {
                    "non_terminal_evaluation_model": "qwen/qwen3-1.7b@4bit",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            captured["config"].non_terminal_evaluation_model,
            "qwen/qwen3-1.7b@4bit",
        )


    def test_create_session_and_get_state(self) -> None:
        response = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("session_id", payload)
        self.assertIsNone(payload["state"]["root"])
        self.assertFalse(payload["state"]["run_state"]["problem_context_prepared"])

        run = self.client.post(f"/api/tot/sessions/{payload['session_id']}/run")
        self.assertEqual(run.status_code, 200)
        self.assertEqual(run.json()["state"]["root"]["equations"], ["root_eq"])
        self.assertNotIn("remaining_budget", run.json()["state"])
        self.assertNotIn("target_expansion_budget", run.json()["state"].get("run_state", {}))

        session_response = self.client.get(f"/api/tot/sessions/{payload['session_id']}")
        self.assertEqual(session_response.status_code, 200)
        self.assertEqual(session_response.json()["state"]["root"]["equations"], ["root_eq"])

    def test_run_session_runs_to_completion_and_expands(self) -> None:
        create = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {"equations": ["child_eq"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq"]}
                            },
                            "evaluation": {"score": 8.0},
                        }
                    ],
                },
            },
        )
        session_id = create.json()["session_id"]

        run = self.client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 200)
        state = run.json()["state"]
        self.assertEqual(state["expansions_used"], 1)
        self.assertEqual(len(state["root"]["children"]), 1)

    def test_run_session_honors_max_tree_depth(self) -> None:
        create = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "scheduler": {"max_tree_depth": 1},
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {"equations": ["child_eq"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq"]}
                            },
                            "evaluation": {"score": 8.0},
                            "children": [
                                {
                                    "proposal": {"equations": ["grand_eq"]},
                                    "calculation": {
                                        "skill_params": {"required_equation_patterns": ["grand_eq"]}
                                    },
                                    "evaluation": {"score": 8.0},
                                }
                            ],
                        }
                    ],
                },
            },
        )
        session_id = create.json()["session_id"]

        run = self.client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 200)
        state = run.json()["state"]
        self.assertEqual(state["max_tree_depth"], 1)
        self.assertEqual(state["expansions_used"], 1)
        self.assertEqual(len(state["root"]["children"]), 1)
        self.assertEqual(state["root"]["children"][0]["children"], [])
        self.assertEqual(state["frontier"], [])

    def test_run_session_stops_at_max_total_expansions_cap(self) -> None:
        create = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "scheduler": {"max_total_expansions": 1},
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {"equations": ["child_eq"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq"]}
                            },
                            "evaluation": {"score": 8.0},
                            "children": [
                                {
                                    "proposal": {"equations": ["grand_eq"]},
                                    "calculation": {
                                        "skill_params": {"required_equation_patterns": ["grand_eq"]}
                                    },
                                    "evaluation": {"score": 8.0},
                                }
                            ],
                        }
                    ],
                },
            },
        )
        session_id = create.json()["session_id"]

        run = self.client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 200)
        state = run.json()["state"]
        self.assertEqual(state["max_total_expansions"], 1)
        self.assertEqual(state["expansions_used"], 1)
        self.assertEqual(len(state["frontier"]), 1)
        self.assertEqual(state["run_state"]["status"], "ready")

        rerun = self.client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(rerun.status_code, 200)
        rerun_state = rerun.json()["state"]
        self.assertEqual(rerun_state["expansions_used"], 1)
        self.assertEqual(len(rerun_state["frontier"]), 1)

    def test_delete_node_endpoint_reviews_then_deletes(self) -> None:
        create = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {"equations": ["child_eq"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq"]}
                            },
                            "evaluation": {"score": 8.0},
                        }
                    ],
                },
                "scheduler": {"max_children_per_expansion": 1},
            },
        )
        payload = create.json()
        session_id = payload["session_id"]
        run = self.client.post(f"/api/tot/sessions/{session_id}/run")
        child_id = run.json()["state"]["root"]["children"][0]["id"]

        delete = self.client.request(
            "DELETE",
            f"/api/tot/sessions/{session_id}/nodes/{child_id}",
            json={"reason": "frontend cleanup", "requested_by": "ui"},
        )

        self.assertEqual(delete.status_code, 200)
        response = delete.json()
        self.assertTrue(response["deleted"])
        self.assertEqual(response["deleted_node_ids"], [child_id])
        self.assertEqual(response["state"]["root"]["children"], [])

    def test_delete_node_endpoint_can_steer_and_continue(self) -> None:
        create = self.client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "scheduler": {
                    "max_tree_depth": 1,
                    "max_frontier_size": 3,
                    "max_children_per_expansion": 3,
                },
                "problem_context": {
                    "problem_statement": "Replace a deleted branch with an operator-steered continuation.",
                    "meta_task": {
                        "objective": "operator steer regression",
                        "first_step": "apply operator steering",
                        "step_ordering": [
                            "apply operator steering",
                            "verify replacement branch",
                        ],
                        "completion_signals": ["operator steering applied"],
                    },
                    "meta_task_progress": {
                        "current_step_index": 0,
                        "current_step": "apply operator steering",
                        "current_step_guidance": "prepare a replacement branch",
                        "previous_steps": [],
                        "remaining_steps": ["verify replacement branch"],
                        "total_steps": 2,
                        "phase": "strategy_scan",
                        "is_terminal_step": False,
                    },
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {
                                "thought_step": "old branch",
                                "equations": ["old_eq"],
                                "known_vars": {"route_family": "constraint"},
                                "used_models": ["Constraint relation"],
                            },
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["old_eq"]}
                            },
                            "evaluation": {"score": 8.0},
                        }
                    ],
                },
            },
        )
        session_id = create.json()["session_id"]
        run = self.client.post(f"/api/tot/sessions/{session_id}/run")
        old_child_id = run.json()["state"]["root"]["children"][0]["id"]

        steer_prompt = "avoid the deleted path and focus on boundary constraints"
        delete = self.client.request(
            "DELETE",
            f"/api/tot/sessions/{session_id}/nodes/{old_child_id}",
            json={
                "reason": "replace with operator steering",
                "requested_by": "ui",
                "steer_prompt": steer_prompt,
                "run_after_delete": True,
            },
        )

        self.assertEqual(delete.status_code, 200)
        response = delete.json()
        self.assertTrue(response["deleted"])
        self.assertTrue(response["steering"]["applied"])
        self.assertEqual(response["steering"]["prompt"], steer_prompt)
        self.assertEqual(response["state"]["run_state"]["status"], "ready")
        self.assertEqual(response["state"]["frontier"], [])

        replacement_children = response["state"]["root"]["children"]
        self.assertEqual(len(replacement_children), 1)
        replacement = replacement_children[0]
        self.assertNotEqual(replacement["id"], old_child_id)
        self.assertIn(steer_prompt, replacement["known_vars"]["operator_steering_prompt"])
        self.assertIn("Operator-steered branch", replacement["thought_step"])

    def test_create_session_maps_backend_failure_to_502(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=failing_adapter_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "scheduler": {
                    "use_local_root_proposal": False,
                    "use_local_root_evaluation": False,
                },
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]

        run = client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(run.status_code, 502)
        self.assertIn("local model backend unavailable", run.json()["detail"])

    def test_background_run_failure_surfaces_in_session_state(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=failing_adapter_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": True,
                "scheduler": {
                    "max_children_per_expansion": 3,
                    "max_live_children_per_batch": 1,
                    "use_local_root_proposal": False,
                    "use_local_root_evaluation": False,
                },
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]

        state = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = client.get(f"/api/tot/sessions/{session_id}").json()["state"]
            if state["run_state"]["status"] == "error":
                break
            time.sleep(0.01)

        self.assertIsNotNone(state)
        self.assertEqual(state["run_state"]["status"], "error")
        self.assertIn("local model backend unavailable", state["run_state"]["last_error"])
        self.assertFalse(state["run_state"]["auto_run_requested"])

    def test_create_session_prepares_problem_context_once(self) -> None:
        store = SchedulerSessionStore()
        client = TestClient(
            create_app(
                session_store=store,
                adapter_bundle_factory=preparing_adapter_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": False,
                "problem_context": {
                    "problem_statement": "Build the tree step by step.",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"]["meta_task"], {})
        session_id = response.json()["session_id"]
        scheduler = store.get(session_id)
        self.assertEqual(scheduler.root_problem_context.get("meta_task"), None)

        run = client.post(f"/api/tot/sessions/{session_id}/run")

        self.assertEqual(
            run.json()["state"]["meta_task"]["objective"],
            "prepared once during create_session",
        )
        self.assertEqual(
            scheduler.root_problem_context["meta_task"]["objective"],
            "prepared once during create_session",
        )

    def test_create_session_can_expand_without_explicit_children_when_meta_task_progress_exists(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=meta_task_auto_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": True,
                "problem_context": {
                    "problem_statement": "Advance through the meta-task one step at a time.",
                    "meta_task": {
                        "first_step": "compare route families",
                        "step_ordering": [
                            "compare route families",
                            "refine first target",
                            "refine second target",
                        ],
                    },
                    "meta_task_progress": {
                        "current_step_index": 0,
                        "current_step": "compare route families",
                        "current_step_guidance": "Analyze the next-step strategy space as broadly as possible while staying at planning level: compare plausible governing routes, identify the main laws/models each route would use, mark the key unknowns and deferred correction terms, and decide what should be refined next. Do not solve for the final answer yet.",
                        "previous_steps": [],
                        "remaining_steps": ["refine first target", "refine second target"],
                        "total_steps": 3,
                        "phase": "strategy_scan",
                        "is_terminal_step": False,
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        create_state = response.json()["state"]
        self.assertIsNone(create_state["root"])
        self.assertTrue(create_state["run_state"]["auto_run_requested"])

        session_id = response.json()["session_id"]
        state = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = client.get(f"/api/tot/sessions/{session_id}").json()["state"]
            if state["expansions_used"] >= 2 and state["root"]:
                break
            time.sleep(0.01)

        self.assertIsNotNone(state)
        self.assertEqual(state["expansions_used"], 2)
        self.assertEqual(len(state["root"]["children"]), 1)
        self.assertEqual(
            state["root"]["children"][0]["thought_step"],
            "Refine only the current subproblem: refine first target.",
        )
        self.assertEqual(
            state["root"]["children"][0]["children"][0]["thought_step"],
            "Refine only the current subproblem: refine second target.",
        )

    def test_get_session_stays_responsive_while_background_run_holds_lock(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=slow_preparing_adapter_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": True,
                "problem_context": {
                    "problem_statement": "Keep session polling responsive during slow background preparation.",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]

        time.sleep(0.05)
        started = time.monotonic()
        poll = client.get(f"/api/tot/sessions/{session_id}")
        elapsed = time.monotonic() - started

        self.assertEqual(poll.status_code, 200)
        self.assertLess(elapsed, 0.15)
        state = poll.json()["state"]
        self.assertEqual(state["run_state"]["status"], "busy")
        self.assertEqual(state["run_state"]["phase"], "preparing-meta-task")

    def test_background_run_publishes_child_progress_before_completion(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(),
                adapter_bundle_factory=slow_progress_adapter_bundle,
            )
        )

        response = client.post(
            "/api/tot/sessions",
            json={
                "run_on_create": True,
                "scheduler": {
                    "use_local_child_proposal": False,
                    "use_local_child_evaluation": False,
                },
                "problem_context": {
                    "proposal": {"equations": ["root_eq"]},
                    "calculation": {
                        "skill_params": {"required_equation_patterns": ["root_eq"]}
                    },
                    "evaluation": {"score": 8.0},
                    "children": [
                        {
                            "proposal": {"equations": ["child_eq_1"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq_1"]}
                            },
                            "evaluation": {"score": 8.0},
                        },
                        {
                            "proposal": {"equations": ["child_eq_2"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq_2"]}
                            },
                            "evaluation": {"score": 8.0},
                        },
                        {
                            "proposal": {"equations": ["child_eq_3"]},
                            "calculation": {
                                "skill_params": {"required_equation_patterns": ["child_eq_3"]}
                            },
                            "evaluation": {"score": 8.0},
                        },
                    ],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]

        saw_child_while_busy = False
        saw_in_flight_expansion = False
        final_state = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            final_state = client.get(f"/api/tot/sessions/{session_id}").json()["state"]
            root = final_state.get("root") or {}
            children = root.get("children") or []
            if final_state["run_state"]["status"] == "busy" and len(children) >= 1:
                saw_child_while_busy = True
                in_flight_expansion = final_state["run_state"].get("in_flight_expansion")
                if isinstance(in_flight_expansion, dict):
                    self.assertEqual(in_flight_expansion.get("parent_id"), root.get("id"))
                    self.assertEqual(in_flight_expansion.get("expected_child_count"), 3)
                    self.assertGreaterEqual(int(in_flight_expansion.get("built_child_count", 0)), 1)
                    self.assertGreaterEqual(int(in_flight_expansion.get("remaining_child_count", 0)), 1)
                    saw_in_flight_expansion = True
            if final_state["run_state"]["status"] != "busy" and len(children) == 3:
                break
            time.sleep(0.02)

        self.assertTrue(saw_child_while_busy)
        self.assertTrue(saw_in_flight_expansion)
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["run_state"]["status"], "ready")
        self.assertIsNone(final_state["run_state"].get("in_flight_expansion"))
        self.assertEqual(len(final_state["root"]["children"]), 3)

    def test_background_run_active_session_cap_queues_second_session_until_slot_frees(self) -> None:
        client = TestClient(
            create_app(
                session_store=SchedulerSessionStore(max_active_auto_runs=1),
                adapter_bundle_factory=very_slow_progress_adapter_bundle,
            )
        )

        payload = {
            "run_on_create": True,
            "scheduler": {
                "max_children_per_expansion": 1,
                "max_live_children_per_batch": 1,
                "use_local_child_proposal": False,
                "use_local_child_evaluation": False,
            },
            "problem_context": {
                "proposal": {"equations": ["root_eq"]},
                "calculation": {
                    "skill_params": {"required_equation_patterns": ["root_eq"]}
                },
                "evaluation": {"score": 8.0},
                "children": [
                    {
                        "proposal": {"equations": ["child_eq_1"]},
                        "calculation": {
                            "skill_params": {"required_equation_patterns": ["child_eq_1"]}
                        },
                        "evaluation": {"score": 8.0},
                    }
                ],
            },
        }

        first = client.post("/api/tot/sessions", json=payload)
        self.assertEqual(first.status_code, 200)
        time.sleep(0.02)
        second = client.post("/api/tot/sessions", json=payload)
        self.assertEqual(second.status_code, 200)

        second_session_id = second.json()["session_id"]
        saw_queued_second = False
        final_second = None
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            final_second = client.get(f"/api/tot/sessions/{second_session_id}").json()["state"]
            phase = final_second["run_state"].get("phase")
            if phase == "queued-active-slot":
                saw_queued_second = True
            if final_second["run_state"].get("status") == "ready":
                break
            time.sleep(0.02)

        self.assertTrue(saw_queued_second)
        self.assertIsNotNone(final_second)
        self.assertEqual(final_second["run_state"]["status"], "ready")
        self.assertGreaterEqual(final_second["expansions_used"], 1)


class SessionStoreEvictionTests(unittest.TestCase):
    """The in-memory store must bound retention so abandoned sessions cannot leak."""

    class _FakeScheduler:
        def __init__(self, status: str = "ready") -> None:
            self.run_status = status

        def snapshot(self) -> dict:
            return {"run_state": {"status": self.run_status}}

    def test_lru_cap_evicts_oldest_idle_sessions(self) -> None:
        store = SchedulerSessionStore(max_sessions=3)
        ids = [store.create(self._FakeScheduler("ready")) for _ in range(6)]
        with store._lock:
            self.assertLessEqual(len(store._sessions), 3)
        self.assertIn(ids[-1], store._sessions)   # newest kept
        self.assertNotIn(ids[0], store._sessions)  # oldest evicted

    def test_busy_sessions_are_never_evicted(self) -> None:
        store = SchedulerSessionStore(max_sessions=2)
        busy = store.create(self._FakeScheduler("busy"))
        for _ in range(8):
            store.create(self._FakeScheduler("ready"))
        self.assertIn(busy, store._sessions)  # busy survives despite the cap

    def test_ttl_evicts_idle_sessions(self) -> None:
        store = SchedulerSessionStore(max_sessions=None, session_ttl_seconds=0.05)
        old = store.create(self._FakeScheduler("ready"))
        time.sleep(0.08)
        fresh = store.create(self._FakeScheduler("ready"))  # create triggers the TTL sweep
        self.assertNotIn(old, store._sessions)
        self.assertIn(fresh, store._sessions)


class PayloadLeniencyTests(unittest.TestCase):
    """Strict build rejects stray fields (which triggers a repair call); the
    lenient last-resort salvages recognized fields so a persistently-noisy model
    degrades gracefully instead of hard-failing the node."""

    def test_strict_rejects_but_lenient_salvages_noise(self) -> None:
        from fsm.models import ReflectionPayload
        from fsm.utils import _build_model, _build_model_lenient

        payload = {
            "thought_step": "reflect on the step",
            "equations": ["v_f^2 = v_i^2 + 2*a*d"],
            "error": "backend noise",        # unknown
            "raw_response": "```json ...```", # unknown
            "reasoning": "chatter",           # unknown
        }
        # strict raises -> this is what triggers the backend repair path
        with self.assertRaises(ValueError):
            _build_model(ReflectionPayload, payload)
        # lenient drops the noise and keeps the recognized fields
        model = _build_model_lenient(ReflectionPayload, payload)
        self.assertEqual(model.thought_step, "reflect on the step")
        self.assertEqual(model.equations, ["v_f^2 = v_i^2 + 2*a*d"])


if __name__ == "__main__":
    unittest.main()