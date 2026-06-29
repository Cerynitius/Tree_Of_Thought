"""Live: POST the exact browser payload, poll the growing tree, show deletes/advises."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"

# exactly what frontend/app.js buildCreateSessionPayload() sends
PAYLOAD = {
    "problem_context": {
        "problem_statement": ("A 2.0 kg block slides from rest down a frictionless incline of height "
                              "1.8 m, then onto a rough horizontal surface with kinetic friction "
                              "coefficient mu_k = 0.25. How far does it travel before stopping? "
                              "Use g = 9.8 m/s^2."),
        "reasoning_depth_preset": "medium",
    },
    "backend": {"base_url": "http://localhost:1234/api/v1/chat",
                "planning_model": "qwen3.6-35b-a3b-ud-mlx", "modeling_model": "qwen3.6-35b-a3b-ud-mlx",
                "review_model": "qwen3.6-35b-a3b-ud-mlx", "non_terminal_evaluation_model": "qwen3.6-35b-a3b-ud-mlx",
                "timeout": 300},
    "scheduler": {"depth_preset": "medium", "max_tree_depth": 8, "max_total_expansions": 12,
                  "max_children_per_expansion": 2, "max_live_children_per_batch": 2},
    "run_on_create": True,
}


def post(path, body, t=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read().decode())

def get(path, t=30):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=t).read().decode())

def walk(n, acc=None):
    acc = [] if acc is None else acc
    if isinstance(n, dict):
        acc.append(n)
        for c in n.get("children") or []:
            walk(c, acc)
    return acc

def grounding_events(state):
    deleted, advised = [], []
    for n in walk(state.get("root")):
        kv = n.get("known_vars") or {}
        hrc = kv.get("hard_rule_check") or {}
        fatal = list(kv.get("hard_rule_violations") or []) + list(hrc.get("violations") or [])
        ign = list(hrc.get("ignored_violations") or [])
        if str(n.get("status", "")).upper().startswith("PRUNED") and any("not grounded" in str(v) for v in fatal):
            deleted.append((n, [v for v in fatal if "not grounded" in str(v)]))
        if any("not grounded" in str(v) for v in ign):
            advised.append((n, [v for v in ign if "not grounded" in str(v)]))
    return deleted, advised


print(f"\n================ LIVE: {LABEL} ================", flush=True)
sid = post("/api/tot/sessions", PAYLOAD)["session_id"]
print(f"POSTed browser payload -> session {sid}", flush=True)
deadline = time.monotonic() + 600
hit = None
while time.monotonic() < deadline:
    try:
        state = get(f"/api/tot/sessions/{sid}")["state"]
    except Exception as e:
        print("poll error:", str(e)[:80], flush=True); time.sleep(5); continue
    nodes = walk(state.get("root"))
    pruned = sum(1 for n in nodes if str(n.get("status", "")).upper().startswith("PRUNED"))
    deleted, advised = grounding_events(state)
    rs = state["run_state"]
    print(f"  tree: {len(nodes):>2} nodes, {pruned} pruned | grounding deleted={len(deleted)} advised={len(advised)} | status={rs['status']}", flush=True)
    if deleted or advised:
        hit = (deleted, advised); break
    if rs["status"] in ("ready", "error"):
        break
    time.sleep(5)

deleted, advised = hit if hit else grounding_events(get(f"/api/tot/sessions/{sid}")["state"])
print("\n  --- what the browser receives & renders ---", flush=True)
if deleted:
    n, v = deleted[0]
    print(f"  DELETED node {n.get('id')}: status={n.get('status')}  eq={n.get('equations')}", flush=True)
    print(f"    fatal grounding violation: {v[0]}", flush=True)
    print(f"    -> browser renders: PRUNED (greyed, removed from viable tree)", flush=True)
if advised:
    n, v = advised[0]
    hrc = (n.get('known_vars') or {}).get('hard_rule_check') or {}
    print(f"  ADVISED node {n.get('id')}: status={n.get('status')}  eq={n.get('equations')}", flush=True)
    print(f"    ignored (advised) violation: {v[0]}", flush=True)
    print(f"    -> browser renders: ACTIVE/SOLVED with 'ignored noise x{len(hrc.get('ignored_violations') or [])}' badge", flush=True)
if not deleted and not advised:
    print("  (no grounding event surfaced within budget on this run)", flush=True)

try:
    req = urllib.request.Request(BASE + f"/api/tot/sessions/{sid}", method="DELETE")
    urllib.request.urlopen(req, timeout=15).read()
except Exception:
    pass
