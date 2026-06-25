#!/bin/bash
# Collect REAL grounding-fire contexts across many physics problems, to test the
# proposed smart-gate rules offline before implementing them. Advisory mode (the
# grounding check still fires and logs; nodes aren't pruned, so more contexts are
# explored). Server reads TOT_GROUNDING_LOG and appends every fire.
set -u
cd /Users/ruixiuzhang/Desktop/Tree_Of_Thought-main
LOG=/tmp/grounding_contexts.jsonl
: > "$LOG"

pkill -9 -f tot_api.py 2>/dev/null; sleep 3
TOT_BOUNDARY_GROUNDING_FIX=1 TOT_MAX_ACTIVE_AUTO_RUNS=1 TOT_GROUNDING_LOG="$LOG" \
  nohup python tot_api.py > /tmp/tot_server_collect.log 2>&1 &
for i in $(seq 1 20); do
  code=$(curl -s -m 2 http://127.0.0.1:8000/health -o /dev/null -w "%{http_code}" 2>/dev/null)
  [ "$code" = "200" ] && break; sleep 1
done
echo "server up; grounding log -> $LOG"

echo "########## benchmark suites (boundary ap1 em traps) ##########"
python -u validate_pruning.py --suites boundary ap1 em traps --max-expansions 12 \
  --concurrency 1 --per-case-timeout 600 --poll-interval 6 --output reports/collect_suites.json

echo "########## fresh problems ##########"
python -u validate_pruning.py --problems-file new_physics_problems.json --max-expansions 12 \
  --concurrency 1 --per-case-timeout 600 --poll-interval 6 --output reports/collect_fresh.json

echo "contexts collected: $(wc -l < "$LOG" | tr -d ' ')"
echo COLLECT_DONE
