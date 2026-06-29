# experiments/

Reproducibility scripts for the results in [`../PAPER.md`](../PAPER.md) and
[`../reports/FINDINGS.md`](../reports/FINDINGS.md). Run them **from the repo root**
(each script puts the repo root on `sys.path`).

| Script | What it does | Needs a running `tot_api.py`? |
|--------|--------------|-------------------------------|
| `validate_pruning.py` | Drive the ToT API over a problem set (benchmark `--suites` or a `--problems-file`); report solve rate, correct-answer-pruned, and a grounding-prune audit. | yes |
| `orchestrate_ab.sh` | Paired advisory-vs-hard A/B campaign; restarts the server per condition, writes `reports/ab_*.json`. | manages its own |
| `new_physics_problems.json` | The 15 held-out physics problems used in the A/B. | — |
| `demo_delete_vs_advise.py` | Deterministic demonstrator: the same node is `PRUNED_BY_RULE` under hard gating vs `SOLVED` (violation logged) under advisory. Set `TOT_BOUNDARY_GROUNDING_FIX=0` for the hard run. | no (offline) |
| `analyze_grounding.py` + `grounding_contexts.jsonl` | The grounding-precision analysis over 57 real captured contexts (59 fires, 0 true positives, 97% recoverable). | no (offline) |
| `concurrency_probe.py`, `diagnose_completion.py` | One-off diagnostics behind the throughput / concurrency findings. | yes |

Examples:

```bash
python experiments/demo_delete_vs_advise.py                 # advisory (default)
TOT_BOUNDARY_GROUNDING_FIX=0 python experiments/demo_delete_vs_advise.py   # hard gating
python experiments/analyze_grounding.py                     # uses the shipped data
bash experiments/orchestrate_ab.sh                          # full A/B (slow; local model)
```
