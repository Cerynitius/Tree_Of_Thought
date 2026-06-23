"""Disentangle throughput vs algorithmic non-termination.

Runs ONE physics problem with a small expansion budget but NO wall-clock cut
(synchronous /run), then inspects whether any route reached a FINALIZED terminal
node with an extracted answer -- and how long the budget took.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:8000"
PROBLEM = ("A 4.0 kg block slides along a horizontal surface at 6.0 m/s and comes to rest due to "
           "kinetic friction with coefficient mu_k = 0.20. How far does it travel before stopping? "
           "Use g = 9.8 m/s^2.")  # expected d = v^2/(2 mu g) = 9.18 m

def post(path, body, timeout):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen(BASE+path, timeout=timeout) as r:
        return json.loads(r.read().decode())

cfg = {"depth_preset":"medium","max_reflections":2,"max_tree_depth":8,
       "max_frontier_size":8,"max_children_per_expansion":2,"max_live_children_per_batch":2,
       "max_total_expansions":10,"use_local_root_proposal":True,"use_local_root_evaluation":True,
       "use_local_child_proposal":True,"use_local_child_evaluation":True,
       "max_frontier_per_diversity_key":4,"children_key":"children"}

created = post("/api/tot/sessions",
               {"run_on_create":False,"problem_context":{"problem_statement":PROBLEM},"scheduler":cfg},
               timeout=60)
sid = created["session_id"]
print(f"session {sid}; running synchronously (budget=10, no wall-clock cut)...", flush=True)

t0 = time.monotonic()
err = ""
try:
    state = post(f"/api/tot/sessions/{sid}/run", {}, timeout=1800)["state"]
except Exception as e:
    err = str(e)[:200]
    state = get(f"/api/tot/sessions/{sid}")["state"]
elapsed = time.monotonic() - t0

rs = state["run_state"]
print(f"\n/run finished in {elapsed:.0f}s  status={rs['status']} phase={rs['phase']} err={(err or rs['last_error'])[:160]}")
print(f"expansions_used={state['expansions_used']}/{state.get('max_total_expansions')}")

def walk(n, d=0, acc=None):
    if acc is None: acc=[]
    if not n: return acc
    acc.append((d,n))
    for c in n.get("children") or []: walk(c, d+1, acc)
    return acc

nodes = walk(state.get("root"))
from collections import Counter
fsm = Counter(n.get("fsm_state") for _,n in nodes)
status = Counter(n.get("status") for _,n in nodes)
maxd = max((d for d,_ in nodes), default=0)
print(f"nodes={len(nodes)} max_depth={maxd}")
print(f"fsm_state: {dict(fsm)}")
print(f"status: {dict(status)}")

# any terminal/answer node?
print("\n--- nodes carrying an answer field ---")
any_ans = False
for d,n in nodes:
    kv = n.get("known_vars") or {}
    ans = kv.get("final_answer") or kv.get("candidate_answer") or kv.get("answer") or kv.get("result")
    term = kv.get("is_terminal_step") or (kv.get("meta_task_progress") or {}).get("is_terminal_step")
    if ans or term:
        any_ans = True
        print(f"  d{d} fsm={n.get('fsm_state')} status={n.get('status')} terminal={bool(term)} answer={str(ans)[:80]}")
        eqs = n.get("equations") or []
        if eqs: print(f"      eq: {'; '.join(str(e) for e in eqs)[:100]}")
if not any_ans:
    print("  (NONE — no node reached a terminal/answer state)")

blob = json.dumps(state)
print(f"\ncontains '9.18'={('9.18' in blob)}  contains '9.2'={('9.2' in blob)}  deepest step reached / terminal? max_depth={maxd}")
post_del = urllib.request.Request(BASE+f"/api/tot/sessions/{sid}", method="DELETE")
try: urllib.request.urlopen(post_del, timeout=15).read()
except Exception: pass
