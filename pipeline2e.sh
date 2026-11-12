#!/bin/bash
# pipeline2e: + replication on fresh seeds. pipeline2d: + near-transfer evals. pipeline2c: pipeline2b plus the check-as-action condition. Re-prioritised queue (2026-11-11 16:55). Emotion teachers (encourage/yell/remind) did nothing or hurt in the moment,
# so their "sticks" rounds are dropped. Information conditions that target the root cause (acting against evidence)
# go first: coach, expert check-rules playbook, worked-example SFT (Sweller), check-prompts, then dream and stick rounds.
cd "$(dirname "$0")"
P=.venv/bin/python
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "evaluate.py|mlx_lm lora|collect.py|dream.py" > /dev/null; do sleep 30; done
train() {  # data adapter
  [ -f "$2/adapters.safetensors" ] && return 0
  $P -m mlx_lm lora --model $Q --train --data "$1" --adapter-path "$2" --iters 200 --batch-size 1 \
     --num-layers 8 --learning-rate 2e-5 --mask-prompt --max-seq-length 2048 --grad-checkpoint \
     --steps-per-eval 100 --val-batches 4 --save-every 20 > "$2.log" 2>&1
  $P pick_adapter.py "$2" "$2.log" >> "$2.log"
}
ev() { log "$1"; $P evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; }

ev stick_worked --adapter adapters/q7_worked
# Check as an ACTION (CHECK tool before every commit), anchored to a tool the held-out tasks also have.
ev q7_checktool --check-tool
grep -qE "keeping final|using checkpoint|before any" adapters/q7_workedcheck.log || $P pick_adapter.py adapters/q7_workedcheck adapters/q7_workedcheck.log >> adapters/q7_workedcheck.log
ev stick_workedcheck --adapter adapters/q7_workedcheck --check-tool
# Near transfer (Lock, Machine: a training game's skill under a new surface), never practised.
NEAR="--envs lock,machine"
ev near_plain $NEAR
ev near_strategy $NEAR --system strategy
ev near_ei_r1 $NEAR --adapter adapters/q7_ei_r1
ev near_worked $NEAR --adapter adapters/q7_worked
ev near_checktool $NEAR --check-tool
ev near_workedcheck $NEAR --adapter adapters/q7_workedcheck --check-tool
# Pre-registered replication of C2/C3 (CONCLUSIONS.md) on fresh seeds 10020-10059.
R="--seed0 10020 --n 40 --envs oddcoin,hiddenrule"
for c in "rep_plain" "rep_strategy --system strategy" "rep_expert --playbook playbooks/expert.txt" "rep_yell --teacher yell" "rep_encourage --teacher encourage"; do
  set -- $c; name=$1; shift
  log "$name"; $P evaluate.py --policy mlx:$Q "$@" $R --out runs/$name > /dev/null 2>&1
done
for s in selfcheck paraphrase opposite; do ev sys_$s --system $s; done

log "dream"; $P dream.py --policy mlx:$Q runs/q7_collect_r1 --out runs/q7_dream > runs/q7_dream.log 2>&1
$P build_sft.py runs/q7_collect_r1 runs/q7_dream --out data/q7_dream
log "train dream"; train data/q7_dream adapters/q7_dream
ev stick_dream --adapter adapters/q7_dream

for t in coach hint; do
  log "practise $t"; $P collect.py --policy mlx:$Q --temp 0.8 --seeds 0:30 --k 2 --teacher $t --out runs/q7_collect_$t > /dev/null 2>&1
  $P build_sft.py runs/q7_collect_$t --out data/q7_$t
  log "train $t"; train data/q7_$t adapters/q7_$t
  ev stick_$t --adapter adapters/q7_$t
done
for t in praise_process praise_ability; do ev moment_$t --teacher $t; done
log "pipeline2e done"
