# Active-Session Cap 9-Session Validation

Date: 2026-05-30
Model: qwen3.6-35b-a3b-ud-mlx
Observation window: 360 seconds
Backend change under test: active auto-run slice cap with `max_active_auto_runs=3`
Scheduler fast paths under test: `use_local_child_proposal=true`, `use_local_child_evaluation=true`, `max_live_children_per_batch=2`

## Goal

Check whether `9 concurrent` can recover to roughly `~1 expansion/session` after adding backend admission control.

## Prior No-Cap Baseline

From [reports/concurrency_isolation_latest.json](reports/concurrency_isolation_latest.json):
- total_sessions: 9
- ready_sessions: 0
- avg_expansions: 0.00
- avg_frontier: 0.00
- avg_nodes: 1.00

Interpretation:
- In the same 360-second window, the no-cap run effectively stayed root-only.

## New Active-Cap Result

Summary:
- total_sessions: 9
- ready_sessions: 2
- busy_sessions: 7
- avg_expansions: 13.44
- avg_frontier: 1.56
- avg_nodes: 9.67
- timeout_error_sessions: 0

This clears the target by average throughput: `13.44 expansions / 9 sessions` is well above `~1 expansion/session`.

## Heartbeat Signal

At 30s:
- `busy:building-root = 3`
- `busy:queued-active-slot = 6`

At 120s:
- `ready:frontier-empty = 1`
- `busy:expanding-frontier = 1`
- `busy:building-root = 2`
- `busy:queued-active-slot = 5`

At 360s:
- `ready:frontier-empty = 2`
- `busy:expanding-frontier = 2`
- `busy:building-root = 1`
- `busy:queued-active-slot = 4`

Interpretation:
- The cap is active and visible in state: queued sessions explicitly report `phase=queued-active-slot` instead of all hammering the model backend together.
- Aggregate work progresses again instead of collapsing to root-only stalls.

## Per-Session End State at 360s

| problem | preset | status | phase | expansions | frontier | nodes |
| --- | --- | --- | --- | ---: | ---: | ---: |
| spring_friction | low | ready | frontier-empty | 27 | 0 | 19 |
| spring_friction | medium | busy | expanding-frontier | 32 | 5 | 22 |
| spring_friction | high | busy | expanding-frontier | 35 | 9 | 27 |
| incline_friction | low | busy | queued-active-slot | 0 | 0 | 0 |
| incline_friction | medium | busy | queued-active-slot | 0 | 0 | 0 |
| incline_friction | high | busy | building-root | 0 | 0 | 0 |
| probability_same_color | low | ready | frontier-empty | 27 | 0 | 19 |
| probability_same_color | medium | busy | queued-active-slot | 0 | 0 | 0 |
| probability_same_color | high | busy | queued-active-slot | 0 | 0 | 0 |

## Conclusion

The active-session cap succeeded on the metric you asked for: the 9-session run recovered well past `~1 expansion/session` on average in the same 360-second observation window.

What changed structurally:
- backend auto-runs no longer all execute at once; they acquire execution in capped slices;
- non-root child `PROPOSE` and `EVALUATE` no longer require live calls by default, so the sessions that do acquire slices make much more progress per slice.

What is still not solved:
- fairness is weak. The current slice admission control restores throughput, but it is not FIFO or round-robin, so a few sessions can keep reacquiring slices while others remain queued.

Practical implication:
- the backend bottleneck is now much more manageable under 9-session pressure, but the next backend step should be fair scheduling of slice admission, not just capped admission.
