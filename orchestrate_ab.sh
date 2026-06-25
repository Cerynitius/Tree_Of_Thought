#!/bin/bash
# Paired, interleaved A/B campaign: advisory vs hard boundary-grounding, N reps.
# Each rep runs advisory then hard (interleaved to avoid temporal drift), saving a
# separate report per run so partial results survive interruption.
set -u
cd /Users/ruixiuzhang/Desktop/Tree_Of_Thought-main
PROB=new_physics_problems.json
REPS=3

CONC=1  # serial: concurrency split LM Studio's context window -> ToT prompts overflowed (HTTP 500)

start_server() {  # $1 = TOT_BOUNDARY_GROUNDING_FIX value
  pkill -9 -f tot_api.py 2>/dev/null
  sleep 3
  TOT_BOUNDARY_GROUNDING_FIX=$1 TOT_MAX_ACTIVE_AUTO_RUNS=$CONC nohup python tot_api.py > /tmp/tot_server_ab.log 2>&1 &
  for i in $(seq 1 20); do
    code=$(curl -s -m 2 http://127.0.0.1:8000/health -o /dev/null -w "%{http_code}" 2>/dev/null)
    [ "$code" = "200" ] && return 0
    sleep 1
  done
  echo "!! server failed to start (fix=$1)"
  return 1
}

run() {  # $1 = output name
  # Budget-bound: high per-case timeout so concurrency-inflated latency never
  # fakes a wall-clock timeout; cases finish on solved/ready, not the clock.
  python -u validate_pruning.py --problems-file "$PROB" --max-expansions 12 \
    --concurrency 1 --per-case-timeout 600 --poll-interval 6 --output "reports/ab_$1.json"
}

for r in $(seq 1 $REPS); do
  echo "########## REP $r : ADVISORY (fix=1) ##########"
  start_server 1 && run "advisory_35b_rep${r}"
  echo "########## REP $r : HARD (fix=0) ##########"
  start_server 0 && run "hard_35b_rep${r}"
done

# leave the server in the good (advisory) state
pkill -9 -f tot_api.py 2>/dev/null; sleep 3
TOT_BOUNDARY_GROUNDING_FIX=1 nohup python tot_api.py > /tmp/tot_server.log 2>&1 &
echo "ALL DONE"
