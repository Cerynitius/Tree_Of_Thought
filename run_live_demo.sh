#!/bin/bash
set -u
cd /Users/ruixiuzhang/Desktop/Tree_Of_Thought-main
start() {  # $1 = TOT_BOUNDARY_GROUNDING_FIX
  pkill -9 -f tot_api.py 2>/dev/null; sleep 3
  TOT_BOUNDARY_GROUNDING_FIX=$1 nohup python tot_api.py > /tmp/tot_server_live.log 2>&1 &
  for i in $(seq 1 15); do
    [ "$(curl -s -m 2 http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}' 2>/dev/null)" = "200" ] && return 0
    sleep 1
  done
}
echo "##### HARD mode (deletes) #####"; start 0; python3 run_live_demo.py "HARD (deletes)"
echo ""; echo "##### ADVISORY mode (advises) #####"; start 1; python3 run_live_demo.py "ADVISORY (advises)"
echo ""; echo "ALL DONE (server left on advisory)"
