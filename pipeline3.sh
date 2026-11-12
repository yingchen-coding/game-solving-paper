#!/bin/bash
# Queued after pipeline2: re-test the paraphrase signal at n=20, plus opposite prompt and Dweck praise teachers (in the moment).
cd "$(dirname "$0")"
P=.venv/bin/python
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
log() { echo "[$(date '+%F %T')] $*"; }
until [ -f runs/pipeline2.log ] && grep -q "pipeline2 done" runs/pipeline2.log; do sleep 60; done
for s in paraphrase opposite selfcheck; do
  log "sys_$s"; $P evaluate.py --policy mlx:$Q --system $s --out runs/sys_$s --n $N > /dev/null 2>&1
done
for t in praise_process praise_ability; do
  log "moment_$t"; $P evaluate.py --policy mlx:$Q --teacher $t --out runs/moment_$t --n $N > /dev/null 2>&1
done
log "pipeline3 done"
