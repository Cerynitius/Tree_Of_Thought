#!/bin/bash
# One-rep advisory-vs-hard A/B over benchmark SUITES -- a headroom test on harder
# problems (vs the near-saturating held-out set). Run from anywhere.
#   Usage: bash experiments/run_suite_ab.sh "hard limit" [budget] [per_case_timeout]
set -u
cd "$(dirname "$0")/.."
SUITES="${1:-hard limit}"
BUDGET="${2:-16}"
TO="${3:-700}"
PREFIX="${4:-hardbench}"   # output -> reports/${PREFIX}_{advisory,hard}.json

start() {  # $1 = TOT_BOUNDARY_GROUNDING_FIX
  pkill -9 -f tot_api.py 2>/dev/null; sleep 3
  TOT_BOUNDARY_GROUNDING_FIX=$1 nohup python tot_api.py > /tmp/tot_server_hb.log 2>&1 &
  for i in $(seq 1 20); do
    [ "$(curl -s -m 2 http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}' 2>/dev/null)" = "200" ] && return 0
    sleep 1
  done
  echo "!! server failed to start"; return 1
}

run() {  # $1 = condition (advisory|hard)
  python -u experiments/validate_pruning.py --suites $SUITES --max-expansions "$BUDGET" \
    --per-case-timeout "$TO" --concurrency 1 --poll-interval 6 --output "reports/${PREFIX}_$1.json"
}

echo "##### ADVISORY (suites: $SUITES, budget $BUDGET, prefix $PREFIX) #####"; start 1 && run advisory
echo "##### HARD #####"; start 0 && run hard

# leave the server on advisory
pkill -9 -f tot_api.py 2>/dev/null; sleep 3
TOT_BOUNDARY_GROUNDING_FIX=1 nohup python tot_api.py > /tmp/tot_server.log 2>&1 &
echo "HARDBENCH DONE"
