#!/bin/bash
# Runs after run_baselines.sh: SFT-on-oracle control, then expert iteration (2 rounds), each evaluated
# on held-out seeds of every env with the PLAIN prompt. Resumable: every step skips finished work.
set -e
cd "$(dirname "$0")"
P=.venv/bin/python
Q3=mlx-community/Qwen2.5-3B-Instruct-4bit
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f run_baselines.sh > /dev/null; do sleep 60; done

train() {  # data_dir adapter_dir iters
  [ -f "$2/adapters.safetensors" ] && return 0
  log "train $2 on $1"
  $P -m mlx_lm lora --model $Q3 --train --data "$1" --adapter-path "$2" --iters "$3" --batch-size 2 \
     --num-layers 16 --learning-rate 1e-4 --mask-prompt --max-seq-length 3072 --grad-checkpoint \
     --steps-per-eval 100 --val-batches 4 --save-every 100 > "$2.log" 2>&1
}
mkdir -p adapters

# control: imitation of the scripted expert
train data/oracle_v2 adapters/sft_oracle 400
log "eval sft_oracle"; $P evaluate.py --policy mlx:$Q3 --adapter adapters/sft_oracle --out runs/q3_sft_oracle --n 20 > /dev/null 2>&1

# expert iteration round 1: sample from base
log "collect r1"; $P collect.py --policy mlx:$Q3 --temp 0.8 --seeds 0:40 --k 2 --out runs/collect_r1 > /dev/null 2>&1
$P build_sft.py runs/collect_r1 --out data/ei_r1
train data/ei_r1 adapters/ei_r1 300
log "eval ei_r1"; $P evaluate.py --policy mlx:$Q3 --adapter adapters/ei_r1 --out runs/q3_ei_r1 --n 20 > /dev/null 2>&1

# round 2: sample from the round-1 policy, train on both rounds
log "collect r2"; $P collect.py --policy mlx:$Q3 --adapter adapters/ei_r1 --temp 0.8 --seeds 40:80 --k 2 --out runs/collect_r2 > /dev/null 2>&1
$P build_sft.py runs/collect_r1 runs/collect_r2 --out data/ei_r2
train data/ei_r2 adapters/ei_r2 400
log "eval ei_r2"; $P evaluate.py --policy mlx:$Q3 --adapter adapters/ei_r2 --out runs/q3_ei_r2 --n 20 > /dev/null 2>&1
log "pipeline done"
