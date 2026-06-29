# Boundary-grounding investigation — findings

## TL;DR
A deterministic SymPy "hard-rule" step-verifier was pruning **correct** physics
derivations: its boundary-condition grounding check vetoed nodes whose BC keys/axes
didn't textually match the node's equation symbols. The fix demotes boundary
grounding to **advisory** (logged, never fatal), gated by `TOT_BOUNDARY_GROUNDING_FIX`
(default on). Advisory beats hard gating by **+24pp (9B) / +29pp (35B)** physics
solve rate, and a real-data analysis shows the grounding check caught **zero genuine
errors** — so advisory is the right default and a "smart" discriminating gate, though
feasible, is unnecessary.

## A/B: advisory vs. hard gating (15 fresh physics problems, 3 paired reps/model)

| Model | Advisory solve | Hard-gate solve | Δ | correct-pruned (advisory / hard) |
|---|---|---|---|---|
| Qwen3.5-9B (dense)     | 93.3% [86.7–100] | 68.9% [66.7–73.3] | **+24.4 pp** | 0 / 5,5,3 |
| Qwen3.6-35B-a3b (MoE)  | 100% [100–100]   | 71.1% [66.7–73.3] | **+28.9 pp** | 0 / 3,3,4 |

- Ranges non-overlapping on both models; worst advisory run beats best hard run.
- Mechanism consistent: hard gating prunes correct answers (BC-heavy cases —
  projectile, pendulum, thin-lens, Doppler, spring, flywheel); advisory prunes none.
- The effect **generalizes and slightly strengthens** on the stronger model →
  "hard symbolic gating is net-negative" is not merely a weak-model artifact.

## Completion was a measurement artifact, not algorithmic failure
An earlier "rarely completes (1/75)" reading was wrong: the harness capped cases at
140–240s but the 9B needs ~250–480s/case. Given adequate wall-clock the system solves
at ~93%. (See `new_problems_truesolverate.json`.)

## Real-data grounding-precision analysis (`experiments/analyze_grounding.py`)
Instrumented the grounding check to capture every real `(equations, known_vars,
boundary_conditions)` context where it fired, across 57 real physics problem contexts:

- **59 grounding fires; 95% were on descriptive (non-numeric) BCs** (e.g. "level
  ground implies y=0", "isothermal = T=300K const") — unambiguous false negatives.
- A prototype "smart" predicate (non-numeric-exempt + suffix/alias normalization)
  **eliminates 97%** of fires while **retaining 4/4 adversarial garbage detection**.
- The single non-numeric-exempt rule does ~95% of the work; name-normalization
  handled ~1 of 3 numeric cases.
- **Grounding caught 0 genuine errors** — every fire was a false negative. The 2
  residual numeric flags (`x(0)=0`, `v_initial=0`) are legitimate BCs the node's own
  equations expressed with a different symbol (lineage-recoverable), not garbage.

### Decision
A smart discriminating gate is feasible (97% recovery + adversarial retention) but
**unnecessary**: grounding's observed precision on real runs is ~0, so advisory
recovers 100% of false negatives for free with no observed downside. Ship advisory;
keep `TOT_BOUNDARY_GROUNDING_FIX=0` as a kill-switch to restore route-local gating.

## Concurrency note
The local backend (LM Studio) batches short requests ~1.5x, but splits its context
window across parallel slots — ToT's large prompts overflow under concurrency
("Context size exceeded"). Serial execution is required for this workload.
