#!/bin/bash
# pipeline2u = 2t + the arithmetic layer retrained at lr 1e-5 after divergence.
# pipeline2t = 2s + the signed-arithmetic layer (make_arith.py).
# pipeline2s = 2r + the application-problem layer (make_applied.py).
# pipeline2r = 2q + lenient-parser rescoring of the textbook model on Lock/Machine.
# pipeline2q = 2p with the format-preserving textbook (q7_theory_act) in place of near_theory.
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
evd() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P diag_evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
# 512-token replies: does self-explanation fail on new surfaces because its reasoning is cut off at 256 tokens?
evd near_selfexplain_t512 --envs lock,machine --adapter adapters/q7_selfexplain --check-tool
evd near_plain_t512 --envs lock,machine
# 教理论 (Ying 2026-11-14): teach the theory school-style, lessons + exercises, no game wording; games never trained
ev theory_prompt --playbook playbooks/theory.txt
ev near_theory_prompt --envs lock,machine --playbook playbooks/theory.txt
train data/q7_theory adapters/q7_theory
ev stick_theory --adapter adapters/q7_theory
# The Q&A textbook made every game reply drop its 'Action:' line (stick_theory: 100% invalid), so retrain with the
# exercises answered in the agent's Thought/Action format (verb STATE, used by no game); near_theory dropped as moot.
train data/q7_theory_act adapters/q7_theory_act
ev stick_theory_act --adapter adapters/q7_theory_act
ev near_theory_act --envs lock,machine --adapter adapters/q7_theory_act
# near_theory_act: every Machine move invalid because 'Action: INPUT 0' sits mid-line; rescore with a lenient parser
evl() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P lenient_evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
evl near_theory_act_lenient --envs lock,machine --adapter adapters/q7_theory_act
evl near_plain_lenient --envs machine
# Gap fill (应用题): the textbook model picked the right inputs (20/20) but could not compute from readings under new
# names; Chapter 6 + 420 application problems with renamed functions, readings as a log, explicit balance pans.
train data/q7_theory_app adapters/q7_theory_app
ev stick_theory_app --adapter adapters/q7_theory_app
ev near_theory_app --envs lock,machine --adapter adapters/q7_theory_app
evl near_theory_app_lenient --envs machine --adapter adapters/q7_theory_app
# Gap fill 2 (signed arithmetic): with Chapter 6 the model collects the readings and names the chapter but computes
# a wrongly in 8 of 13 written HiddenRule solutions, always at a double negative; Chapter 7 + 200 drills + clean form.
train data/q7_theory_arith adapters/q7_theory_arith
ev stick_theory_arith --adapter adapters/q7_theory_arith
evl near_theory_arith_lenient --envs lock,machine --adapter adapters/q7_theory_arith
# q7_theory_arith diverged: train loss 0.13 at iter 110 rose to 0.94 at 150 and the final adapter was kept (4 val
# batches); Machine replies became 'PPPP...', GuessNumber fell to 9/20. Retrain at half the learning rate with 16 val
# batches so the kept checkpoint is chosen on more than a few examples.
train_lo() {
  [ -f "$2/adapters.safetensors" ] && return 0
  gate; log "train $2 (lr 1e-5)"; $P $S/train_lora.py --model $Q --data "$1" --adapter-path "$2" --iters 200 --lr 1e-5 \
     --num-layers 8 --max-seq-length 2048 --save-every 20 --steps-per-eval 25 --val-batches 16 > "$2.log" 2>&1; yield_turn; }
train_lo data/q7_theory_arith adapters/q7_theory_arith_lr1
ev stick_theory_arith_lr1 --adapter adapters/q7_theory_arith_lr1
evl near_theory_arith_lr1_lenient --envs lock,machine --adapter adapters/q7_theory_arith_lr1
# 授人以渔: one skill under many wordings vs the same data repeated, tested on a wording never trained (Machine)
train data/q7_distill_vary adapters/q7_distill_vary
train data/q7_distill_dup adapters/q7_distill_dup
ev vary_hm --envs hiddenrule,machine --adapter adapters/q7_distill_vary
ev dup_hm --envs hiddenrule,machine --adapter adapters/q7_distill_dup
# 化陌生为熟悉: map the new names onto a known problem before acting (prompt only)
ev near_familiar --envs lock,machine --playbook playbooks/familiar.txt
ev near_distill_familiar --envs lock,machine --adapter adapters/q7_distill --playbook playbooks/familiar.txt
NEAR="--envs lock,machine"
ev near_plain $NEAR
ev near_selfexplain $NEAR --adapter adapters/q7_selfexplain --check-tool
ev rep_selfexplain --adapter adapters/q7_selfexplain --check-tool --seed0 10020
ev rep_plain_all --seed0 10020
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
log "pipeline2u done"
