#!/bin/bash
# Resume of pipeline.sh after the first LoRA run diverged to NaN (lr 5e-5, see pick_adapter.py): retrain at lr 2e-5
# with frequent checkpoints, then the B and C evals, then release pipeline2.
cd "$(dirname "$0")"
P=.venv/bin/python
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
log() { echo "[$(date '+%F %T')] $*"; }
log "B: train (lr 2e-5)"
$P -m mlx_lm lora --model $Q --train --data data/q7_ei_r1 --adapter-path adapters/q7_ei_r1 --iters 200 --batch-size 1 \
   --num-layers 8 --learning-rate 2e-5 --mask-prompt --max-seq-length 2048 --grad-checkpoint \
   --steps-per-eval 100 --val-batches 4 --save-every 20 > adapters/q7_ei_r1.log 2>&1
$P pick_adapter.py adapters/q7_ei_r1 adapters/q7_ei_r1.log >> adapters/q7_ei_r1.log || { log "B: no finite adapter"; exit 1; }
log "B: eval"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_ei_r1 --out runs/q7_ei_r1 --n $N > /dev/null 2>&1
log "C: eval"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_ei_r1 --playbook runs/q7_reflect/playbook.txt --out runs/q7_ei_playbook --n $N > /dev/null 2>&1
log "pipeline done"
