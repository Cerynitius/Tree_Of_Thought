"""Clean idle-model concurrency benchmark for the local 9B backend.

Measures wall-clock for N identical requests run sequentially vs concurrently.
speedup ~1x  => backend serializes (no gain from parallel cases)
speedup >1.5x => backend batches (parallelizing the case loop would help)
"""
import json, time, urllib.request
import concurrent.futures as cf

URL = "http://localhost:1234/api/v1/chat"
MODEL = "qwen/qwen3.5-9b"
N = 4

def call(i):
    body = json.dumps({
        "model": MODEL,
        "system_prompt": "You are concise.",
        "input": f"In exactly 4 sentences, explain Newton's second law of motion. (variant {i})",
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return time.monotonic() - t

print(f"warmup: {call(0):.1f}s", flush=True)

t = time.monotonic()
seq = [call(i) for i in range(1, N + 1)]
seq_wall = time.monotonic() - t

t = time.monotonic()
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    conc = list(ex.map(call, range(N + 1, 2 * N + 1)))
conc_wall = time.monotonic() - t

print(f"\nsequential {N} reqs: wall={seq_wall:.1f}s  (avg per-req {sum(seq)/N:.1f}s)")
print(f"concurrent {N} reqs: wall={conc_wall:.1f}s  (avg per-req {sum(conc)/N:.1f}s)")
print(f"\nSPEEDUP = {seq_wall/conc_wall:.2f}x   (>~1.5x => batching helps; ~1.0x => serial)")
