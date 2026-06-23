"""FastAPI application that exposes the ToT scheduler over HTTP."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import Any, Callable, Literal, Optional, TypeVar
from uuid import uuid4

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fsm import (
    DEFAULT_CHAT_API_URL,
    DEFAULT_MODELING_MODEL,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    DEFAULT_PLANNING_MODEL,
    DEFAULT_REVIEW_MODEL,
    ChatBackendError,
    DeterministicContextBackendAdapter,
    NodeDeletionReviewAdapter,
    ReasoningBackendAdapter,
    ToTTreeScheduler,
    build_local_chat_adapter_bundle,
)

SchedulerOperationResult = TypeVar("SchedulerOperationResult")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

DEFAULT_PROBLEM_CONTEXT_DRAFT: dict[str, Any] = {
    "task": (
        "Use the modeling model to propose the next reasoning step, then score each step for domain consistency "
        "and variable grounding."
    ),
    "notes": [
        "The frontend polls the live scheduler state and renders it as an ASCII tree.",
        "Node deletion is routed through the backend review model before the subtree is removed.",
    ],
    "known_context": {
        "objective": "Derive and prune a useful reasoning tree.",
        "expected_output": "concise, structured, and domain-valid intermediate steps",
    },
}

AdapterBundleFactory = Callable[
    ["ChatBackendConfig"],
    tuple[Callable[[dict[str, Any]], ReasoningBackendAdapter], NodeDeletionReviewAdapter],
]


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    stripped = raw_value.strip()
    if not stripped:
        return default
    value = int(stripped)
    if value <= 0:
        return None
    return value


class ChatBackendConfig(BaseModel):
    base_url: str = Field(default_factory=lambda: os.getenv("CHAT_BASE_URL", DEFAULT_CHAT_API_URL))
    planning_model: str = Field(default_factory=lambda: os.getenv("PLANNING_MODEL", DEFAULT_PLANNING_MODEL))
    modeling_model: str = Field(default_factory=lambda: os.getenv("MODELING_MODEL", DEFAULT_MODELING_MODEL))
    review_model: str = Field(default_factory=lambda: os.getenv("REVIEW_MODEL", DEFAULT_REVIEW_MODEL))
    non_terminal_evaluation_model: str = Field(default_factory=lambda: os.getenv("NON_TERMINAL_EVALUATION_MODEL", DEFAULT_NON_TERMINAL_EVALUATION_MODEL))
    timeout: float = Field(default_factory=lambda: float(os.getenv("CHAT_TIMEOUT", "600")), gt=0.0)
    allow_live_model_fallback: bool = Field(
        default_factory=lambda: _env_bool("ALLOW_LIVE_MODEL_FALLBACK", False),
        description="Allow local deterministic fallback only after live model transport failures.",
    )
    prefer_local_fallback: bool = Field(
        default_factory=lambda: _env_bool("PREFER_LOCAL_FALLBACK", False),
        description="Use local deterministic fallback before live model calls. Keep false for fair model-vs-ToT comparisons.",
    )


class SchedulerConfig(BaseModel):
    depth_preset: Literal["low", "medium", "high"] = "medium"
    max_reflections: int = Field(default=2, ge=0)
    max_tree_depth: int = Field(default=8, ge=0)
    max_frontier_size: int = Field(default=16, ge=1)
    max_children_per_expansion: int = Field(default=3, ge=1)
    max_live_children_per_batch: int = Field(default=2, ge=1)
    max_total_expansions: Optional[int] = Field(
        default_factory=lambda: _env_optional_int("TOT_MAX_TOTAL_EXPANSIONS", 64),
        ge=1,
        description="Hard ceiling on lifetime expansions per session; null (or env value <= 0) means unlimited.",
    )
    use_local_root_proposal: bool = True
    use_local_root_evaluation: bool = True
    use_local_child_proposal: bool = True
    use_local_child_evaluation: bool = True
    max_frontier_per_diversity_key: int = Field(default=4, ge=1)
    children_key: str = "children"


class CreateSessionRequest(BaseModel):
    problem_prompt: str = Field(
        default="",
        description="Optional plain-text problem statement used to backfill problem_context.problem_statement when missing.",
    )
    problem_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context merged with problem_statement. See /api/tot/defaults for the UI draft shape.",
    )
    backend: ChatBackendConfig = Field(default_factory=ChatBackendConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    run_on_create: bool = True


class FrontendDefaultsResponse(BaseModel):
    problem_context: dict[str, Any] = Field(
        default_factory=lambda: deepcopy(DEFAULT_PROBLEM_CONTEXT_DRAFT)
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class DeleteNodeRequest(BaseModel):
    reason: str
    requested_by: str = "frontend"
    steer_prompt: str = ""
    run_after_delete: bool = False


class SessionStateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class DeleteNodeResponse(BaseModel):
    session_id: str
    deleted: bool
    node_id: str
    parent_id: Optional[str] = None
    deleted_node_ids: list[str] = Field(default_factory=list)
    review: dict[str, Any] = Field(default_factory=dict)
    steering: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class SchedulerSessionStore:
    """In-memory session store for live ToT schedulers."""

    def __init__(self, *, max_active_auto_runs: Optional[int] = 3) -> None:
        self._sessions: dict[str, ToTTreeScheduler] = {}
        self._session_locks: dict[str, RLock] = {}
        self._session_snapshots: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self.max_active_auto_runs = (
            None if max_active_auto_runs is None else max(1, int(max_active_auto_runs))
        )
        self._active_auto_run_sessions: set[str] = set()
        self._auto_run_wait_queue: deque[str] = deque()
        self._auto_run_condition = Condition(self._lock)

    @staticmethod
    def _freeze_snapshot(scheduler: ToTTreeScheduler) -> dict[str, Any]:
        return deepcopy(scheduler.snapshot())

    def create(self, scheduler: ToTTreeScheduler) -> str:
        session_id = uuid4().hex
        snapshot = self._freeze_snapshot(scheduler)
        with self._lock:
            self._sessions[session_id] = scheduler
            self._session_locks[session_id] = RLock()
            self._session_snapshots[session_id] = snapshot
        return session_id

    def get(self, session_id: str) -> ToTTreeScheduler:
        with self._lock:
            scheduler = self._sessions.get(session_id)
        if scheduler is None:
            raise KeyError(session_id)
        return scheduler

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            scheduler = self._sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
            cached_snapshot = deepcopy(self._session_snapshots.get(session_id, {}))
        if scheduler is None or session_lock is None:
            raise KeyError(session_id)
        if not session_lock.acquire(blocking=False):
            return jsonable_encoder(self._merge_snapshot_with_live_run_state(cached_snapshot, scheduler))
        try:
            snapshot = self._freeze_snapshot(scheduler)
        finally:
            session_lock.release()

        with self._lock:
            self._session_snapshots[session_id] = snapshot
        return jsonable_encoder(deepcopy(snapshot))

    def execute(
        self,
        session_id: str,
        operation: Callable[[ToTTreeScheduler], SchedulerOperationResult],
    ) -> SchedulerOperationResult:
        with self._lock:
            scheduler = self._sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
        if scheduler is None or session_lock is None:
            raise KeyError(session_id)
        with session_lock:
            result = operation(scheduler)
            snapshot = self._freeze_snapshot(scheduler)
        with self._lock:
            self._session_snapshots[session_id] = snapshot
        return result

    def publish_snapshot(self, session_id: str, scheduler: ToTTreeScheduler) -> None:
        snapshot = self._freeze_snapshot(scheduler)
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            self._session_snapshots[session_id] = snapshot

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session_lock = self._session_locks.get(session_id)
        if session_lock is None:
            return False
        with session_lock:
            with self._auto_run_condition:
                deleted = self._sessions.pop(session_id, None) is not None
                self._session_locks.pop(session_id, None)
                self._session_snapshots.pop(session_id, None)
                self._active_auto_run_sessions.discard(session_id)
                self._remove_from_auto_run_wait_queue(session_id)
                self._auto_run_condition.notify_all()
                return deleted

    def _remove_from_auto_run_wait_queue(self, session_id: str) -> None:
        if not self._auto_run_wait_queue:
            return
        self._auto_run_wait_queue = deque(
            queued_session_id
            for queued_session_id in self._auto_run_wait_queue
            if queued_session_id != session_id
        )

    def _enqueue_auto_run_waiter(self, session_id: str) -> None:
        if session_id in self._active_auto_run_sessions:
            return
        if session_id in self._auto_run_wait_queue:
            return
        self._auto_run_wait_queue.append(session_id)

    def try_acquire_auto_run_slot(self, session_id: str) -> bool:
        with self._auto_run_condition:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            if self.max_active_auto_runs is None:
                self._remove_from_auto_run_wait_queue(session_id)
                self._active_auto_run_sessions.add(session_id)
                return True
            if session_id in self._active_auto_run_sessions:
                return True
            if self._auto_run_wait_queue and self._auto_run_wait_queue[0] != session_id:
                self._enqueue_auto_run_waiter(session_id)
                return False
            if len(self._active_auto_run_sessions) < self.max_active_auto_runs:
                self._remove_from_auto_run_wait_queue(session_id)
                self._active_auto_run_sessions.add(session_id)
                return True
            self._enqueue_auto_run_waiter(session_id)
            return False

    def wait_for_auto_run_slot(self, session_id: str, timeout: float = 0.05) -> None:
        with self._auto_run_condition:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            self._auto_run_condition.wait(timeout=timeout)
            if session_id not in self._sessions:
                raise KeyError(session_id)

    def release_auto_run_slot(self, session_id: str) -> None:
        with self._auto_run_condition:
            removed = session_id in self._active_auto_run_sessions
            self._active_auto_run_sessions.discard(session_id)
            if removed:
                self._auto_run_condition.notify_all()

    def mark_auto_run_waiting(self, session_id: str) -> None:
        self.execute(
            session_id,
            lambda scheduler: getattr(scheduler, "_set_run_state", lambda **_: None)(
                status="busy",
                phase="queued-active-slot",
                last_error="",
            )
            or scheduler,
        )

    def _merge_snapshot_with_live_run_state(
        self,
        cached_snapshot: dict[str, Any],
        scheduler: ToTTreeScheduler,
    ) -> dict[str, Any]:
        snapshot = deepcopy(cached_snapshot) if cached_snapshot else {}
        in_flight_expansion = getattr(scheduler, "_in_flight_expansion", None)
        snapshot["run_state"] = {
            "status": str(getattr(scheduler, "run_status", "idle")),
            "phase": str(getattr(scheduler, "run_phase", "created")),
            "problem_context_prepared": bool(getattr(scheduler, "_problem_context_prepared", False)),
            "auto_run_requested": bool(getattr(scheduler, "auto_run_requested", False)),
            "last_error": str(getattr(scheduler, "last_error", "")),
            "in_flight_expansion": deepcopy(in_flight_expansion)
            if isinstance(in_flight_expansion, dict)
            else None,
        }
        return snapshot


def _prune_budget_fields(state: dict[str, Any]) -> dict[str, Any]:
    public_state = deepcopy(state)
    public_state.pop("expansion_budget", None)
    public_state.pop("remaining_budget", None)
    public_state.pop("target_expansion_budget", None)
    run_state = public_state.get("run_state")
    if isinstance(run_state, dict):
        run_state.pop("target_expansion_budget", None)
    return public_state


def _raise_backend_http_error(exc: ChatBackendError) -> None:
    raise HTTPException(status_code=502, detail=str(exc)) from exc


def _get_session_or_404(app: FastAPI, session_id: str) -> ToTTreeScheduler:
    try:
        return app.state.session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _get_session_state_or_404(app: FastAPI, session_id: str) -> dict[str, Any]:
    try:
        return _serialize_session_state(app.state.session_store, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _execute_session_or_404(
    app: FastAPI,
    session_id: str,
    operation: Callable[[ToTTreeScheduler], SchedulerOperationResult],
) -> SchedulerOperationResult:
    try:
        return app.state.session_store.execute(session_id, operation)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _default_adapter_bundle_factory(
    config: ChatBackendConfig,
) -> tuple[Callable[[dict[str, Any]], ReasoningBackendAdapter], NodeDeletionReviewAdapter]:
    return build_local_chat_adapter_bundle(
        base_url=config.base_url,
        timeout=config.timeout,
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        planning_model=config.planning_model,
        modeling_model=config.modeling_model,
        review_model=config.review_model,
        non_terminal_evaluation_model=config.non_terminal_evaluation_model,
        allow_live_model_fallback=config.allow_live_model_fallback,
        prefer_local_fallback=config.prefer_local_fallback,
    )


def _serialize_session_state(store: SchedulerSessionStore, session_id: str) -> dict[str, Any]:
    return _prune_budget_fields(store.snapshot(session_id))


def _run_scheduler_with_progress(
    scheduler: ToTTreeScheduler,
    additional_budget: int,
    progress_callback: Optional[Callable[[ToTTreeScheduler], None]] = None,
) -> ToTTreeScheduler:
    scheduler.target_expansion_budget = max(
        int(getattr(scheduler, "target_expansion_budget", scheduler.expansion_budget)),
        scheduler.expansion_budget + additional_budget,
    )
    scheduler.expansion_budget += additional_budget
    scheduler.run(
        progress_callback=(None if progress_callback is None else lambda: progress_callback(scheduler))
    )
    return scheduler


def _run_scheduler_until_complete(
    session_store: SchedulerSessionStore,
    session_id: str,
) -> dict[str, Any]:
    def publish_progress(scheduler: ToTTreeScheduler) -> None:
        session_store.publish_snapshot(session_id, scheduler)

    def has_remaining_scheduler_work(state: dict[str, Any]) -> bool:
        if state.get("frontier"):
            return True
        run_state = state.get("run_state") or {}
        return isinstance(run_state.get("in_flight_expansion"), dict)

    def has_in_flight_expansion(state: dict[str, Any]) -> bool:
        run_state = state.get("run_state") or {}
        return isinstance(run_state.get("in_flight_expansion"), dict)

    def total_expansion_cap_reached(state: dict[str, Any]) -> bool:
        cap = state.get("max_total_expansions")
        if not isinstance(cap, int) or cap <= 0:
            return False
        return int(state.get("expansions_used", 0)) >= cap

    def execute_auto_run_slice(
        operation: Callable[[ToTTreeScheduler], dict[str, Any]],
    ) -> dict[str, Any]:
        waiting_marked = False
        while True:
            if session_store.try_acquire_auto_run_slot(session_id):
                break
            if not waiting_marked:
                session_store.mark_auto_run_waiting(session_id)
                waiting_marked = True
            session_store.wait_for_auto_run_slot(session_id)

        try:
            return session_store.execute(session_id, operation)
        finally:
            session_store.release_auto_run_slot(session_id)

    state = execute_auto_run_slice(
        lambda scheduler: _run_scheduler_with_progress(
            scheduler,
            0,
            progress_callback=publish_progress,
        ).snapshot()
    )
    while has_remaining_scheduler_work(state):
        cap_reached = total_expansion_cap_reached(state)
        if cap_reached and not has_in_flight_expansion(state):
            # The lifetime expansion ceiling is exhausted; only finish a parked
            # batch (additional budget 0), never start a new expansion.
            break
        additional_budget = 0 if cap_reached else 1
        state = execute_auto_run_slice(
            lambda scheduler, additional_budget=additional_budget: _run_scheduler_with_progress(
                scheduler,
                additional_budget,
                progress_callback=publish_progress,
            ).snapshot()
        )
    return state


def _progressive_auto_run_session(
    session_store: SchedulerSessionStore,
    session_id: str,
) -> None:
    try:
        _run_scheduler_until_complete(session_store, session_id)
    except Exception as exc:
        try:
            session_store.execute(
                session_id,
                lambda scheduler: getattr(scheduler, "_set_run_state", lambda **_: None)(
                    status="error",
                    phase="error",
                    last_error=str(exc),
                ) or scheduler,
            )
        except KeyError:
            pass
    finally:
        try:
            session_store.execute(
                session_id,
                lambda scheduler: setattr(scheduler, "auto_run_requested", False) or scheduler,
            )
        except KeyError:
            pass


def _start_progressive_auto_run(
    session_store: SchedulerSessionStore,
    session_id: str,
) -> None:
    Thread(
        target=_progressive_auto_run_session,
        args=(session_store, session_id),
        daemon=True,
    ).start()


def _delete_scheduler_node(
    scheduler: ToTTreeScheduler,
    *,
    node_id: str,
    request: DeleteNodeRequest,
) -> dict[str, Any]:
    result = scheduler.delete_node(
        node_id,
        reason=request.reason,
        requested_by=request.requested_by,
    )
    steer_prompt = str(request.steer_prompt or "").strip()
    if result.get("deleted") and steer_prompt:
        result["steering"] = scheduler.apply_steering_prompt(
            parent_node_id=str(result.get("parent_id", "")),
            prompt=steer_prompt,
            requested_by=request.requested_by,
            source_deleted_node_ids=list(result.get("deleted_node_ids", [])),
            route_focus_override=(
                dict(result.get("steering_route_focus", {}))
                if isinstance(result.get("steering_route_focus"), dict)
                else {}
            ),
        )
    else:
        result["steering"] = {}
    return result


def create_app(
    *,
    session_store: Optional[SchedulerSessionStore] = None,
    adapter_bundle_factory: Optional[AdapterBundleFactory] = None,
) -> FastAPI:
    app = FastAPI(title="ToT API", version="0.1.0")
    app.state.session_store = session_store or SchedulerSessionStore(
        max_active_auto_runs=_env_optional_int("TOT_MAX_ACTIVE_AUTO_RUNS", 3)
    )
    app.state.adapter_bundle_factory = adapter_bundle_factory or _default_adapter_bundle_factory
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/tot/defaults", response_model=FrontendDefaultsResponse)
    def get_defaults() -> FrontendDefaultsResponse:
        return FrontendDefaultsResponse()

    @app.post("/api/tot/sessions", response_model=SessionStateResponse)
    def create_session(request: CreateSessionRequest) -> SessionStateResponse:
        try:
            backend_factory, deletion_review_adapter = app.state.adapter_bundle_factory(request.backend)
            root_problem_context = deepcopy(request.problem_context)
            problem_prompt = str(request.problem_prompt or "").strip()
            problem_statement = str(root_problem_context.get("problem_statement", "")).strip()
            if not problem_statement:
                task_fallback = str(root_problem_context.get("task", "")).strip()
                objective_fallback = str(root_problem_context.get("objective", "")).strip()
                problem_statement = problem_prompt or task_fallback or objective_fallback
            if problem_statement:
                root_problem_context["problem_statement"] = problem_statement
            root_problem_context["reasoning_depth_preset"] = request.scheduler.depth_preset
            scheduler = ToTTreeScheduler(
                root_problem_context=root_problem_context,
                max_reflections=request.scheduler.max_reflections,
                expansion_budget=0,
                max_tree_depth=request.scheduler.max_tree_depth,
                max_frontier_size=request.scheduler.max_frontier_size,
                max_children_per_expansion=request.scheduler.max_children_per_expansion,
                max_live_children_per_batch=request.scheduler.max_live_children_per_batch,
                use_local_root_proposal=request.scheduler.use_local_root_proposal,
                use_local_root_evaluation=request.scheduler.use_local_root_evaluation,
                use_local_child_proposal=request.scheduler.use_local_child_proposal,
                use_local_child_evaluation=request.scheduler.use_local_child_evaluation,
                max_frontier_per_diversity_key=request.scheduler.max_frontier_per_diversity_key,
                children_key=request.scheduler.children_key,
                backend_adapter_factory=backend_factory,
                deletion_review_adapter=deletion_review_adapter,
                max_total_expansions=request.scheduler.max_total_expansions,
            )
            scheduler.auto_run_requested = bool(request.run_on_create)
            scheduler.run_status = "busy" if request.run_on_create else "idle"
            scheduler.run_phase = "queued" if request.run_on_create else "created"
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        except (ValueError, TypeError) as exc:
            # Bad scheduler bounds or malformed problem_context: report as a client
            # error instead of leaking a raw 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session_id = app.state.session_store.create(scheduler)
        state = _serialize_session_state(app.state.session_store, session_id)
        if request.run_on_create:
            _start_progressive_auto_run(
                app.state.session_store,
                session_id,
            )
        return SessionStateResponse(session_id=session_id, state=state)

    @app.get("/api/tot/sessions/{session_id}", response_model=SessionStateResponse)
    def get_session(session_id: str) -> SessionStateResponse:
        return SessionStateResponse(session_id=session_id, state=_get_session_state_or_404(app, session_id))

    @app.post("/api/tot/sessions/{session_id}/run", response_model=SessionStateResponse)
    def run_session(session_id: str) -> SessionStateResponse:
        try:
            _run_scheduler_until_complete(app.state.session_store, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        return SessionStateResponse(session_id=session_id, state=_get_session_state_or_404(app, session_id))

    @app.delete("/api/tot/sessions/{session_id}/nodes/{node_id}", response_model=DeleteNodeResponse)
    def delete_node(session_id: str, node_id: str, request: DeleteNodeRequest) -> DeleteNodeResponse:
        try:
            result = _execute_session_or_404(
                app,
                session_id,
                lambda current_scheduler: _delete_scheduler_node(
                    current_scheduler,
                    node_id=node_id,
                    request=request,
                ),
            )
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            if request.run_after_delete and bool(result["deleted"]):
                state = _prune_budget_fields(_run_scheduler_until_complete(app.state.session_store, session_id))
            else:
                # Serialize through the session store so the read happens under the
                # session lock instead of racing a concurrent auto-run thread.
                state = _serialize_session_state(app.state.session_store, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)

        return DeleteNodeResponse(
            session_id=session_id,
            deleted=bool(result["deleted"]),
            node_id=str(result["node_id"]),
            parent_id=(str(result.get("parent_id")) if result.get("parent_id") is not None else None),
            deleted_node_ids=list(result["deleted_node_ids"]),
            review=dict(result["review"]),
            steering=dict(result.get("steering", {})),
            state=state,
        )

    @app.delete("/api/tot/sessions/{session_id}", response_model=SessionDeleteResponse)
    def delete_session(session_id: str) -> SessionDeleteResponse:
        deleted = app.state.session_store.delete(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return SessionDeleteResponse(session_id=session_id, deleted=True)

    return app


app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()