# Backend Concurrency Isolation Report

Date: 2026-05-29

Environment:
- API: http://127.0.0.1:8000
- Live model: qwen3.6-35b-a3b-ud-mlx
- Backend timeout: 180 seconds
- Observation window: 360 seconds
- Scheduler knobs: max_live_children_per_batch=2, use_local_child_evaluation=true
- Grouped runs restarted the API server between each 3-session batch to remove leftover in-flight work.

## Scenario A: 9 Concurrent Sessions

Summary:
- total_sessions: 9
- busy_sessions: 9
- ready_sessions: 0
- error_sessions: 0
- timeout_error_sessions: 0
- avg_expansions: 0.00
- avg_frontier: 0.00
- avg_nodes: 1.00

Per-session snapshot at 360s:

| problem | preset | status | phase | expansions | frontier | nodes | last_error |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| spring_friction | low | busy | expanding-frontier | 0 | 0 | 1 | |
| spring_friction | medium | busy | expanding-frontier | 0 | 0 | 1 | |
| spring_friction | high | busy | expanding-frontier | 0 | 0 | 1 | |
| incline_friction | low | busy | expanding-frontier | 0 | 0 | 1 | |
| incline_friction | medium | busy | expanding-frontier | 0 | 0 | 1 | |
| incline_friction | high | busy | expanding-frontier | 0 | 0 | 1 | |
| probability_same_color | low | busy | expanding-frontier | 0 | 0 | 1 | |
| probability_same_color | medium | busy | expanding-frontier | 0 | 0 | 1 | |
| probability_same_color | high | busy | expanding-frontier | 0 | 0 | 1 | |

Observation:
- In 6 minutes, all 9 live sessions remained stuck at root-only state. None produced a logged child expansion.

## Scenario B: 3 x 3 Grouped Runs

Aggregate summary across all 9 grouped sessions:
- total_sessions: 9
- busy_sessions: 9
- ready_sessions: 0
- error_sessions: 0
- timeout_error_sessions: 0
- avg_expansions: 0.89
- avg_frontier: 1.44
- avg_nodes: 3.89

### Group 1: spring_friction

Summary:
- avg_expansions: 1.00
- avg_frontier: 1.67
- avg_nodes: 4.00

| preset | status | phase | expansions | frontier | nodes |
| --- | --- | --- | ---: | ---: | ---: |
| low | busy | expanding-frontier | 1 | 1 | 4 |
| medium | busy | expanding-frontier | 1 | 2 | 4 |
| high | busy | expanding-frontier | 1 | 2 | 4 |

### Group 2: incline_friction

Summary:
- avg_expansions: 1.00
- avg_frontier: 1.67
- avg_nodes: 4.00

| preset | status | phase | expansions | frontier | nodes |
| --- | --- | --- | ---: | ---: | ---: |
| low | busy | expanding-frontier | 1 | 1 | 4 |
| medium | busy | expanding-frontier | 1 | 2 | 4 |
| high | busy | expanding-frontier | 1 | 2 | 4 |

### Group 3: probability_same_color

Summary:
- avg_expansions: 0.67
- avg_frontier: 1.00
- avg_nodes: 3.67

| preset | status | phase | expansions | frontier | nodes |
| --- | --- | --- | ---: | ---: | ---: |
| low | busy | expanding-frontier | 1 | 1 | 3 |
| medium | busy | expanding-frontier | 1 | 2 | 4 |
| high | busy | expanding-frontier | 0 | 0 | 4 |

Observation:
- Every grouped batch advanced beyond root-only state.
- Most grouped sessions produced at least one logged expansion and multiple child nodes in the same 6-minute window.

## Conclusion

The dominant bottleneck is live model-service throughput under concurrency, not scheduler bookkeeping by itself.

Evidence:
- The scheduler code path was the same in both scenarios.
- Changing only the number of simultaneously active sessions from 9 to 3 changed average progress from root-only (`avg_expansions=0.00`, `avg_nodes=1.00`) to clear child creation (`avg_expansions=0.89`, `avg_nodes=3.89`).
- No grouped run finished within 6 minutes, so scheduler-side throughput still needs work, but the first-order collapse happens when too many live sessions compete for the backend at once.

Immediate implication:
- The next throughput win is likely to come from reducing live root/proposal pressure or enforcing an active-session cap, not from further snapshot or bookkeeping changes alone.
