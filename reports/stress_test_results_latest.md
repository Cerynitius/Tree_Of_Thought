# Live Frontend Stress Test Results

Generated at: 2026-05-29 17:05:00
UI/API: http://127.0.0.1:8000
Method: frontend page automation only. Each session was created through the page preset buttons plus CREATE & RUN.
Live backend: http://localhost:1234/api/v1/chat
Live model: qwen3.6-35b-a3b-ud-mlx
Timeout seconds: 180

## Current Scheduler Shape

- Root route scan now respects the same child surface cap as deeper expansions.
- Single-expansion child surface is capped at `low=2`, `medium=3`, `high=4`.
- Live child builds are additionally batched with `max_live_children_per_batch=2` for all presets.
- Busy snapshots expose `in_flight_expansion`, so the frontend no longer collapses early child batches into a false `frontier=0, expansions=0` stall view.

## Comparison Summary

| slice | observation window | busy | error | ready | timeout errors | avg expansions | avg frontier | avg nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-patch baseline | 10m | 4 | 5 | 0 | 5 | 9.33 | 4.22 | 18.00 |
| post-batched live cap | 17.0m | 5 | 4 | 0 | 4 | 0.22 | 0.22 | 1.89 |

## Delta

| metric | baseline | post-batch | delta |
|---|---:|---:|---:|
| timeout errors | 5/9 | 4/9 | -1 |
| timeout error rate | 55.56% | 44.44% | -11.11 pp |
| avg expansions | 9.33 | 0.22 | -97.64% |
| avg frontier | 4.22 | 0.22 | -94.79% |
| avg nodes | 18.00 | 1.89 | -89.50% |

## Current Snapshot Rows

| problem | preset | session_id | status | phase | expansions | frontier | nodes | in-flight | last_error |
|---|---|---|---|---|---:|---:|---:|---|---|
| spring_friction | low | 6db17f44d40e44908ebc1a5cbab4b05e | busy | expanding-frontier | 1 | 1 | 3 | 0/2 built, 2 remaining |  |
| spring_friction | medium | 203bbc4e85e146eaadb8fd4bdde638c3 | error | error | 0 | 0 | 1 | - | timeout in `proposal` |
| spring_friction | high | d3e4454a8a1f492b8ecb3a626aef646d | busy | expanding-frontier | 0 | 0 | 3 | 2/3 built, 1 remaining |  |
| incline_friction | low | f651e81d7dd84f799921a888d047216c | error | error | 0 | 0 | 1 | - | timeout in `proposal` |
| incline_friction | medium | 49368025b20d48a69a2dc50b910ddd27 | error | error | 0 | 0 | 0 | - | timeout in `evaluation` |
| incline_friction | high | 0c00f273006b4e1daea23da408cf46bd | error | error | 0 | 0 | 0 | - | timeout in `evaluation` |
| probability_same_color | low | 92da5ef710334f3bbdeca2719ad06482 | busy | expanding-frontier | 1 | 1 | 3 | 0/2 built, 2 remaining |  |
| probability_same_color | medium | 5e9ea9a0b4eb4d5ab54d837d564bb126 | busy | expanding-frontier | 0 | 0 | 3 | 2/3 built, 1 remaining |  |
| probability_same_color | high | 2656367be6884536b66ed19066442f00 | busy | expanding-frontier | 0 | 0 | 3 | 2/4 built, 2 remaining |  |

## Findings

1. The batched child-build cap reduces timeout count under a longer live observation window, but only slightly.
Even after stretching the window from the original 10 minutes to about 17 minutes, timeout errors dropped from `5/9` to `4/9`.

2. The timeout improvement is being bought almost entirely by slowing tree throughput.
Average expansions fell from `9.33` to `0.22`, average frontier from `4.22` to `0.22`, and average nodes from `18.00` to `1.89`.

3. The new behavior is qualitatively different from the old frontier blow-up pattern.
Spring high and probability medium/high are no longer exploding into `frontier=15/22`; instead they are still sitting in visible early child batches like `2/3 built` or `2/4 built`.

4. Online concurrency is still the limiting factor.
`incline_friction` now fails across all three presets, and no session reached `ready` by the current observation point.

5. Snapshot visibility is fixed even when throughput is poor.
The frontend now surfaces `in-flight x/y` directly, so early-batch work is no longer misreported as a dead scheduler.

## Bottom Line

- The new continuation path does what it was intended to do: it narrows the live child batch and keeps early work visible.
- It does not yet solve live throughput. Under 9 concurrent frontend-created sessions, the system still spends most of its time inside the first expansion batch and continues to hit 180-second model timeouts.
- In short: `slightly fewer timeouts, dramatically less throughput`.

## Notes

- The pre-patch baseline row is preserved from the earlier 2026-05-29 live frontend report.
- The current post-batch row comes from the same 9-session frontend run after a clean server restart and a structured API capture at approximately 17 minutes.
- An intermediate 10m/15m Playwright polling script completed, but its large-file artifact collapsed to an accessibility snapshot rather than structured JSON; the final structured capture above was taken directly from the same session set.
