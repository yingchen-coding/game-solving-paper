#!/bin/bash
# Main run on Qwen2.5-7B-Instruct 4-bit. Order: baselines -> (A) self-reflection playbook ->
# (B) expert-iteration fine-tune -> (C) both. Eval = held-out seeds (>= 10000) of every env, n=20.
# Learning uses TRAIN_GAMES seeds < 10000 only. Resumable: finished episodes are skipped.
cd "$(dirname "$0")"
P=.venv/bin/python
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f smoke7b_think > /dev/null; do sleep 30; done

log "baseline plain";    $P evaluate.py --policy mlx:$Q --out runs/q7_plain --n $N > /dev/null 2>&1
log "baseline strategy"; $P evaluate.py --policy mlx:$Q --system strategy --out runs/q7_strategy --n $N > /dev/null 2>&1

log "A: reflect"
[ -f runs/q7_reflect/playbook.txt ] || $P reflect.py --policy mlx:$Q --rounds 3 --per-game 3 --out runs/q7_reflect > runs/q7_reflect.log 2>&1
log "A: eval"; $P evaluate.py --policy mlx:$Q --playbook runs/q7_reflect/playbook.txt --out runs/q7_playbook --n $N > /dev/null 2>&1

log "B: collect"; $P collect.py --policy mlx:$Q --temp 0.8 --seeds 0:30 --k 2 --out runs/q7_collect_r1 > runs/q7_collect_r1.log 2>&1
$P build_sft.py runs/q7_collect_r1 --out data/q7_ei_r1
mkdir -p adapters
if [ ! -f adapters/q7_ei_r1/adapters.safetensors ]; then
  log "B: train"
  $P -m mlx_lm lora --model $Q --train --data data/q7_ei_r1 --adapter-path adapters/q7_ei_r1 --iters 200 --batch-size 1 \
     --num-layers 8 --learning-rate 5e-5 --mask-prompt --max-seq-length 2048 --grad-checkpoint \
     --steps-per-eval 100 --val-batches 4 --save-every 100 > adapters/q7_ei_r1.log 2>&1
fi
log "B: eval"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_ei_r1 --out runs/q7_ei_r1 --n $N > /dev/null 2>&1

log "C: eval"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_ei_r1 --playbook runs/q7_reflect/playbook.txt --out runs/q7_ei_playbook --n $N > /dev/null 2>&1
log "pipeline done"
