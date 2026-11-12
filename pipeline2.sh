#!/bin/bash
# "How to raise a problem solver" runs (HYPOTHESES.md), queued after pipeline.sh. 7B, eval seeds >= 10000, n=20.
# Stage 1 = teacher present at test (in-the-moment). Stage 2 = practise with the teacher, keep wins, strip the
# teacher, fine-tune, test WITHOUT the teacher (does the skill stick?). Practice volume equals Practice (H1):
# train seeds 0:30, k=2.
cd "$(dirname "$0")"
P=.venv/bin/python
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "pipeline done" runs/pipeline_q7.log; do sleep 60; done
train() {  # data adapter
  [ -f "$2/adapters.safetensors" ] && return 0
  $P -m mlx_lm lora --model $Q --train --data "$1" --adapter-path "$2" --iters 200 --batch-size 1 \
     --num-layers 8 --learning-rate 2e-5 --mask-prompt --max-seq-length 2048 --grad-checkpoint \
     --steps-per-eval 100 --val-batches 4 --save-every 20 > "$2.log" 2>&1
  $P pick_adapter.py "$2" "$2.log" >> "$2.log"
}
for t in encourage yell remind hint coach; do
  log "moment $t"; $P evaluate.py --policy mlx:$Q --teacher $t --out runs/moment_$t --n $N > /dev/null 2>&1
done

log "dream"; $P dream.py --policy mlx:$Q runs/q7_collect_r1 --out runs/q7_dream > runs/q7_dream.log 2>&1
$P build_sft.py runs/q7_collect_r1 runs/q7_dream --out data/q7_dream
log "dream train"; train data/q7_dream adapters/q7_dream
log "dream eval"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_dream --out runs/stick_dream --n $N > /dev/null 2>&1

for t in coach yell encourage remind hint; do
  log "practise $t"; $P collect.py --policy mlx:$Q --temp 0.8 --seeds 0:30 --k 2 --teacher $t --out runs/q7_collect_$t > /dev/null 2>&1
  $P build_sft.py runs/q7_collect_$t --out data/q7_$t
  log "train $t"; train data/q7_$t adapters/q7_$t
  log "stick $t"; $P evaluate.py --policy mlx:$Q --adapter adapters/q7_$t --out runs/stick_$t --n $N > /dev/null 2>&1
done

log "H8 expert playbook"; $P evaluate.py --policy mlx:$Q --playbook playbooks/expert.txt --out runs/h8_expert --n $N > /dev/null 2>&1
for mode in taxonomy contrastive; do
  [ -f runs/q7_reflect_$mode/playbook.txt ] || $P reflect.py --policy mlx:$Q --mode $mode --rounds 3 --per-game 3 --out runs/q7_reflect_$mode > runs/q7_reflect_$mode.log 2>&1
  $P evaluate.py --policy mlx:$Q --playbook runs/q7_reflect_$mode/playbook.txt --out runs/h_$mode --n $N > /dev/null 2>&1
done
log "pipeline2 done"
