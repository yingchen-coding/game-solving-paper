#!/bin/bash
# pipeline2g = pipeline2f + diverse practice first. pipeline2f: runs alongside a second experimenter's jobs. Code is frozen in frozen/ (git HEAD c0ebdae,
# max_tokens 256) so every condition stays comparable with the 13 finished ones; the live tree now has 640.
# Only one 7B fits in 16 GB, so every step waits until no other model process is running.
cd "$(dirname "$0")"
P=.venv/bin/python
S=frozen
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
PAT="evaluate.py|mlx_lm lora|mlx_lm.lora|collect.py|dream.py|train_lora.py|reflect.py"
log() { echo "[$(date '+%F %T')] $*"; }
gate() { while true; do
  while pgrep -f "$PAT" > /dev/null; do sleep 20; done
  sleep $((5 + RANDOM % 15)); pgrep -f "$PAT" > /dev/null || return 0; done; }
train() {
  [ -f "$2/adapters.safetensors" ] && return 0
  gate; $P -m mlx_lm lora --model $Q --train --data "$1" --adapter-path "$2" --iters 200 --batch-size 1 \
     --num-layers 8 --learning-rate 2e-5 --mask-prompt --max-seq-length 2048 --grad-checkpoint \
     --steps-per-eval 100 --val-batches 4 --save-every 20 > "$2.log" 2>&1
  $P pick_adapter.py "$2" "$2.log" >> "$2.log"; }
ev() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P $S/evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; }
evr() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P $S/evaluate.py --policy mlx:$Q "${@:2}" --seed0 10020 --n 40 --envs oddcoin,hiddenrule --out runs/$1 > /dev/null 2>&1; }

# Variability of practice: check-before-commit on 8 games (4 + 4 new) vs 4, same 240 episodes and 651 examples.
train data/q7_diverse adapters/q7_diverse
ev stick_diverse --adapter adapters/q7_diverse --check-tool
ev near_diverse --envs lock,machine --adapter adapters/q7_diverse --check-tool
# Pre-registered replication of C2/C3 (CONCLUSIONS.md) on fresh seeds 10020-10059.
evr rep_plain
evr rep_expert --playbook playbooks/expert.txt
evr rep_yell --teacher yell
evr rep_strategy --system strategy
evr rep_encourage --teacher encourage
ev stick_workedcheck --adapter adapters/q7_workedcheck --check-tool
NEAR="--envs lock,machine"
ev near_plain $NEAR
ev near_strategy $NEAR --system strategy
ev near_ei_r1 $NEAR --adapter adapters/q7_ei_r1
ev near_worked $NEAR --adapter adapters/q7_worked
ev near_workedcheck $NEAR --adapter adapters/q7_workedcheck --check-tool
ev near_checktool $NEAR --check-tool
for s in selfcheck paraphrase opposite; do ev sys_$s --system $s; done
log "pipeline2g done"
