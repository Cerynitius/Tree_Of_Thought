# Tree_Of_Thought

<video src="https://github.com/user-attachments/assets/34a0f5e0-1d53-4231-a2f7-dc1f987f57d6" controls width="100%"></video>

Tree_Of_Thought is an external Tree-of-Thought reasoning system for structured problem solving.

Instead of relying on one opaque model completion, this project turns reasoning into an explicit, inspectable, controllable tree with live state, scoring, pruning, and deterministic tool support.

It combines:

- a FastAPI service that manages long-lived reasoning sessions
- a browser UI for creating sessions, inspecting nodes, steering, and pruning branches
- a node-level FSM and tree scheduler for controlled branch growth
- a SymPy-backed skill layer for exact symbolic and analytical computation
- role-split model routing for planning, modeling, review, and non-terminal evaluation


## Why A Tree Beats Built-In CoT

Built-in CoT is useful, but it has hard limits when you care about reliability, auditability, or branch diversity.

- Built-in CoT is hidden. You usually see only the final answer, not a persistent reasoning structure you can inspect, compare, rank, or edit.
- Built-in CoT is mostly linear. Once a single completion drifts, alternative paths are lost. A tree keeps multiple viable routes alive at the same time.
- Built-in CoT is difficult to control step-by-step. This system constrains each node to one local move instead of letting a model solve too much at once.
- Built-in CoT is hard to debug. Here every node has explicit status, score, route metadata, review output, and frontier state.
- Built-in CoT does not naturally support operator intervention. This system lets you inspect a node, delete a subtree, steer the follow-up prompt, reconnect to a session, or continue expansion later.

The practical result is reasoning with persistence, structure, and control.


## Why An External System Helps

The biggest gain comes from externalizing reasoning out of a single model pass.

- State becomes durable. Sessions, nodes, frontier entries, and run phases live outside any one completion.
- Reasoning becomes reproducible. The same scheduler settings and backend settings can be replayed and regression-tested.
- Models become swappable. Planning, modeling, review, and evaluation can each use different models with different cost and latency profiles.
- Deterministic checks become first-class. Hard rules, symbolic math, and structured post-processing do not depend on a model remembering every constraint.
- Human oversight becomes possible. The system exposes the tree over HTTP and in a browser UI instead of trapping everything inside a prompt.
- System-level optimization becomes possible. You can tune frontier size, depth, reflection limits, deletion policy, and model routing independently from the prompt text.

In short, externalization turns reasoning from a hidden behavior into a real software system.


## Main Ideas

- Role-split model routing. Planning, modeling, review, and non-terminal evaluation are separated rather than collapsed into one all-purpose model call.
- Route-local incremental refinement. Non-terminal nodes are expected to add exactly one new local delta instead of paraphrasing the parent or jumping ahead.
- Parent-child semantic-delta enforcement. The system checks whether a child is meaningfully different from its parent and soft-prunes unresolved duplicates.
- FSM-governed node lifecycle. Proposal, calculation, evaluation, reflection, and finalization are modeled as explicit states rather than ad hoc prompt retries.
- Lightweight intermediate evaluation plus stronger review. Non-terminal nodes can be scored cheaply while terminal or deletion-sensitive decisions still go through stronger review paths.
- Review-gated subtree deletion. Branch removal flows through backend review before deletion.
- Deterministic skill integration. Symbolic computation lives in `skills.py`, `skill_registry.md`, and `skills.md`. Benchmark problems, expected values, and per-case tool solutions live in `benchmarks.py`, outside the runtime skill registry, so the system under test cannot discover benchmark answers through skill search.
- Live inspectable frontier. The scheduler exposes the current tree, frontier selection, candidate answers, and expansion state for operator debugging.


## System Overview

The current architecture uses four reasoning roles:

- planning model: route selection and orchestration
- modeling model: propose or revise one local next step
- review model: review, deletion review, and terminal evaluation
- non-terminal evaluation model: lightweight scoring for intermediate nodes

The scheduler keeps multiple branches alive, enforces route-local refinement, and exposes the evolving tree through the web UI and HTTP API.


## Requirements

- Conda or another Python environment manager
- Access to the configured planning, modeling, review, and evaluation models

The provided environment file installs the Python dependencies used by the API, scheduler, tests, and symbolic skill layer.


## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate tot
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate tot
```


## Run The App

Start the FastAPI server:

```bash
python tot_api.py
```

Then open:

```text
http://127.0.0.1:8000/
```

The UI is served from the same process and loads the terminal-style tree explorer from `frontend/`.


## Basic Workflow

1. Start the local chat backend.
2. Start `tot_api.py`.
3. Open `http://127.0.0.1:8000/`.
4. Enter a problem statement.
5. Create a session.
6. Inspect the tree, frontier, final result panel, and node details.
7. Delete or steer branches when you want to redirect the search.
8. Run the session again to continue until the frontier is exhausted.

The UI also supports reconnecting to an existing session id, polling controls, node deletion through backend review, steering after deletion, and recommended model presets.


## API Surface

Main endpoints:

- `GET /` - serve the frontend
- `GET /health` - lightweight health check
- `POST /api/tot/sessions` - create a session
- `GET /api/tot/sessions/{session_id}` - fetch current state
- `POST /api/tot/sessions/{session_id}/run` - run until the current frontier is exhausted
- `DELETE /api/tot/sessions/{session_id}/nodes/{node_id}` - delete a subtree after review
- `DELETE /api/tot/sessions/{session_id}` - delete a session

A minimal session-creation example:

```bash
curl -X POST http://127.0.0.1:8000/api/tot/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "run_on_create": true,
    "problem_context": {
      "problem_statement": "Compare several viable solution routes before refining the best branch."
    }
  }'
```


## Repository Layout

- `tot_api.py` - FastAPI app, session store, route handlers, and frontend serving
- `fsm/` - backend adapters (`backend.py`), HTTP client (`chat_client.py`), error taxonomy (`errors.py`), payload coercion (`payloads.py`), node FSM (`builder.py`), models, and tree scheduler
- `frontend/` - browser UI for tree inspection and session control
- `skills.py` - runtime skill registry, hard-rule checking, and ToT prompt/plugin layer
- `physics_skills.py` - generic SymPy computation skills (mechanics, EM, quantum, thermo, relativity, optics, fluids)
- `benchmarks.py` - benchmark fixtures (problems, expected values, per-case tool solutions) kept outside the skill registry
- `skill_registry.md` - human-readable map from problem classes to skill names
- `skills.md` - skill calling conventions and usage guidance
- `tests/` - API, scheduler, FSM, and backend regression tests
- `environment.yml` - conda environment definition


## Testing

Run the API tests:

```bash
python -m unittest tests.test_api
```

Run the main harness regression suite:

```bash
python -m unittest tests.test_harness -v
```


## Operational Notes

- Session state is stored in memory, not in a database.
- Session creation returns a session id immediately; deeper expansion can continue in the background when `run_on_create` is enabled.
- Each session stops expanding once it reaches `max_total_expansions` (default 64, configurable per session or via `TOT_MAX_TOTAL_EXPANSIONS`; unlimited when null). This bounds total model calls per session.
- Local chat runs live-first by default. `allow_live_model_fallback` permits deterministic local fallback only after transport failures, while `prefer_local_fallback` or `PREFER_LOCAL_FALLBACK=1` restores the older fast fallback-first behavior.
- If you change backend code, restart `tot_api.py` so the running server picks up the new behavior.
- Non-terminal evaluation is intentionally lighter-weight than terminal review.
- Node deletion is review-gated on the backend before a subtree is removed.


## Status

The repository is suitable for a controlled beta workflow: the API and FSM regression suites are in place, frontend and backend defaults are aligned, and the system is intended to be run locally against a compatible model backend.
