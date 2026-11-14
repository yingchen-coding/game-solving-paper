#!/bin/bash
# pipeline2k = 2j reordered around problem-solving capability (Ying 2026-11-12: focus on this), plus self-explanation
# (Chi et al. 1989): the model writes its own reasoning for the expert's moves. Emotion replications go last.
# pipeline2j = 2i + analogical comparison prompt (Gentner, Loewenstein & Thompson 2003): two solved cases of
# different kinds plus their shared method, vs expert rules alone (h8_expert), a prompt-only test of what education
# research says produces transfer.
# pipeline2i = 2h + process-verified worked examples (thoughts show every check's arithmetic) with CHECK before commit,
# on 8 games vs 4 (same 240 episodes / 651 examples): stick_diverse showed the tool habit transfers in form but the
# model narrates checks it never computed.
# pipeline2h = 2g with LoRA via train_lora.py (grad clipping + NaN skip): mlx_lm lora went NaN at iter 30 on q7_diverse.
# Both check-habit adapters retrained the same way so 4 vs 8 games differ only in the games.
# pipeline2g = pipeline2f + diverse practice first. pipeline2f: runs alongside a second experimenter's jobs. Code is frozen in frozen/ (git HEAD c0ebdae,
# max_tokens 256) so every condition stays comparable with the 13 finished ones; the live tree now has 640.
# Only one 7B fits in 16 GB, so every step waits until no other model process is running.
cd "$(dirname "$0")"
P=.venv/bin/python
S=frozen
Q=mlx-community/Qwen2.5-7B-Instruct-4bit
N=20
PAT="\.venv/bin/python.*(evaluate\.py|mlx_lm|collect\.py|dream\.py|train_lora\.py|reflect\.py)"
log() { echo "[$(date '+%F %T')] $*"; }
gate() {  # the other experimenter's queue starts its next job at once, so poll fast; yield_turn below gives it its turn back
  while pgrep -f "$PAT" > /dev/null; do sleep 2; done; }
yield_turn() { sleep 90; }
train() {
  [ -f "$2/adapters.safetensors" ] && return 0
  gate; log "train $2"; $P $S/train_lora.py --model $Q --data "$1" --adapter-path "$2" --iters 200 --lr 2e-5 \
     --num-layers 8 --max-seq-length 2048 --save-every 20 --steps-per-eval 50 --val-batches 4 > "$2.log" 2>&1; yield_turn; }
ev() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P $S/evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
evr() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P $S/evaluate.py --policy mlx:$Q "${@:2}" --seed0 10020 --n 40 --envs oddcoin,hiddenrule --out runs/$1 > /dev/null 2>&1; yield_turn; }

# Variability of practice: check-before-commit on 8 games (4 + 4 new) vs 4, same 240 episodes and 651 examples.
train data/q7_diverse adapters/q7_diverse
train data/q7_workedcheck adapters/q7_workedcheck2
ev stick_diverse --adapter adapters/q7_diverse --check-tool
ev stick_workedcheck2 --adapter adapters/q7_workedcheck2 --check-tool
# The three decisive tests first (Ying 2026-11-12): process-verified practice, analogy, self-explanation.
train data/q7_procdiverse adapters/q7_procdiverse
ev stick_procdiverse --adapter adapters/q7_procdiverse --check-tool
ev analogy --playbook playbooks/analogy.txt
[ -s data/q7_selfexplain/stats.json ] || { gate; log "selfexplain data"; $P $S/selfexplain_collect.py data/q7_procdiverse --out data/q7_selfexplain > runs/q7_selfexplain.log 2>&1; yield_turn; }
train data/q7_selfexplain adapters/q7_selfexplain
ev stick_selfexplain --adapter adapters/q7_selfexplain --check-tool
train data/q7_proc4 adapters/q7_proc4
ev stick_proc4 --adapter adapters/q7_proc4 --check-tool
# Fresh-seed replication of the self-explanation effect (found after looking at the data): seeds 10020-10039, all tasks
ev rep_selfexplain --adapter adapters/q7_selfexplain --check-tool --seed0 10020
ev rep_plain_all --seed0 10020
NEAR="--envs lock,machine"
ev near_plain $NEAR
ev near_procdiverse $NEAR --adapter adapters/q7_procdiverse --check-tool
ev near_selfexplain $NEAR --adapter adapters/q7_selfexplain --check-tool
ev near_analogy $NEAR --playbook playbooks/analogy.txt
ev near_proc4 $NEAR --adapter adapters/q7_proc4 --check-tool
ev near_diverse $NEAR --adapter adapters/q7_diverse --check-tool
ev near_workedcheck2 $NEAR --adapter adapters/q7_workedcheck2 --check-tool
ev near_strategy $NEAR --system strategy
ev near_ei_r1 $NEAR --adapter adapters/q7_ei_r1
ev near_worked $NEAR --adapter adapters/q7_worked
ev near_checktool $NEAR --check-tool
# Pre-registered replication of C2 (guidance vs expertise) on fresh seeds 10020-10059.
evr rep_plain
evr rep_expert --playbook playbooks/expert.txt
evr rep_strategy --system strategy
for s in selfcheck paraphrase opposite; do ev sys_$s --system $s; done
evr rep_yell --teacher yell
evr rep_encourage --teacher encourage
log "pipeline2k done"
