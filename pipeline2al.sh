#!/bin/bash
# pipeline2aa = 2z + C20, GRPO with per-seed randomised wording.
# pipeline2z = 2y + fresh-seed replication of the cross-skill Parcels result.
# pipeline2y = 2x + working-prompt mechanism test and the one-wording dose condition.
# pipeline2x = 2w + rejection-sampling control for GRPO.
# pipeline2w = 2v + C19, outcome-reward RL (GRPO) on the original wording, tested on the twins.
# pipeline2v = 2u + C18, varied wordings on a second skill (OddCoin -> Parcels).
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
PAT="\.venv/bin/python.*(evaluate\.py|mlx_lm|collect\.py|dream\.py|train_lora\.py|reflect\.py|grpo\.py|make_selfvary\.py)"
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
# C18: does 授人以渔 generalise to a second skill? Varied wordings of HiddenRule AND OddCoin (vs the same rows
# repeated), tested on Parcels, OddCoin's held-out twin (twin_envs.py), alongside Machine.
evt() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P twin_evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
evt twin_plain --envs parcels
evt twin_distill --envs parcels --adapter adapters/q7_distill
train data/q7_distill_vary2 adapters/q7_distill_vary2
evt twin_vary2 --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_distill_vary2
train data/q7_distill_dup2 adapters/q7_distill_dup2
evt twin_dup2 --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_distill_dup2
evt twin_vary --envs parcels --adapter adapters/q7_distill_vary
# C19 (Ying 2026-11-15: read DeepSeek, find how transfer is learned): R1 trains with outcome-reward RL (GRPO), and
# "SFT memorizes, RL generalizes" (Chu et al. 2025). Every condition here so far is supervised. GRPO on HiddenRule and
# OddCoin in their original wording only, seeds 1000+, then test on Machine and Parcels.
grpo_train() {  # gate first: an orphaned grpo.py from a replaced queue may still be writing this adapter
  gate; [ -f "$1/adapters.safetensors" ] && return 0
  log "grpo $1"; $P grpo.py --envs "${2:-hiddenrule,oddcoin}" --iters 60 --group 4 --out "$1" > "$1.log" 2>&1; yield_turn; }
# C18 gap: on Machine the repeated-data model writes its working in 8/20 final replies (152 chars) and the varied one
# in 19/20 (521 chars). If skipped working is the mechanism, asking for the working should rescue the repeated model.
ev near_dup_workout --envs machine --adapter adapters/q7_distill_dup --playbook playbooks/workout.txt
ev near_plain_workout --envs machine --playbook playbooks/workout.txt
grpo_train adapters/q7_grpo
evt twin_grpo --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_grpo
# Replication on fresh seeds 10020-10059 (found after looking): HiddenRule-only varied wordings lifted Parcels to 19/20
# against 12/20 for distillation with identical OddCoin data; varying OddCoin too gave 15.
evtr() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P twin_evaluate.py --policy mlx:$Q "${@:2}" --seed0 10020 --n 40 --out runs/$1 > /dev/null 2>&1; yield_turn; }
evtr rep_parcels_distill --envs parcels --adapter adapters/q7_distill
evtr rep_parcels_vary --envs parcels --adapter adapters/q7_distill_vary
evtr rep_parcels_vary2 --envs parcels --adapter adapters/q7_distill_vary2
evtr rep_parcels_dup2 --envs parcels --adapter adapters/q7_distill_dup2
# C20: GRPO on the original wording improved the practised games (HiddenRule 4->10, OddCoin 16->18) but not the
# twins (Machine 13->10, Parcels 9->10). Same RL with the wording redrawn per seed (domain randomisation), pools
# excluding both twins' wording.
grpo_train adapters/q7_grpo_rand randhiddenrule,randoddcoin
evt twin_grpo_rand --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_grpo_rand
# C21: Parcels has only 21 possible puzzles; seeds 10000-10019 hold 13 and 10020-10059 hold 19, and greedy decoding
# replays a repeat identically. Pooled over 20 distinct puzzles: distill 16, vary 19, vary2 15, dup2 15 (vary vs
# distill paired 4-1, p = .375). Re-test every model once on all 21 puzzles (parcels21, seeds 10080-10100).
ev21() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P twin_evaluate.py --policy mlx:$Q "${@:2}" --envs parcels21 --seed0 10080 --n 21 --out runs/$1 > /dev/null 2>&1; yield_turn; }
ev21 p21_plain
ev21 p21_distill --adapter adapters/q7_distill
ev21 p21_vary --adapter adapters/q7_distill_vary
ev21 p21_dup2 --adapter adapters/q7_distill_dup2
# C21 gap: losing claims name a parcel already ruled out (distill 6/6 failures, dup2 10/13, vary2 7/13), mostly after
# unequal pans; 14% of the distilled OddCoin weighings are unequal. Same rows as dup2 with every OddCoin row
# replaced by a scripted expert that writes reading -> ruled out -> still possible, equal pans.
train data/q7_coinclean adapters/q7_coinclean
ev21 p21_coinclean --adapter adapters/q7_coinclean
evt twin_coinclean --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_coinclean
# C22: randomised-wording GRPO fell to HiddenRule 0, OddCoin 14, Machine 12, Parcels 6 (plain 4/16/13/9). Invalid moves
# 2 -> 198 over 80 games: 'Action: ACTION QUERY 0', 'Action: Action: INPUT 0', action mid-line. Diagnostic: same
# adapter with a parser that forgives placement and doubled labels. Fix: R1's format reward (--format-penalty 1).
evl2() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P lenient_twin_evaluate.py --policy mlx:$Q "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
evl2 twin_grpo_rand_lenient --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_grpo_rand
# C21 void: data/q7_coinclean stored whole episodes, and the trainer trains only each row's last reply, so only the
# ANSWER turns were learned (replies never wrote the expert's "Still possible"). Rebuilt as per-turn prefixes.
train data/q7_coinclean2 adapters/q7_coinclean2
ev21 p21_coinclean2 --adapter adapters/q7_coinclean2
evt twin_coinclean2 --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_coinclean2
# C22 diagnostic result: with a parser that forgives doubled/misplaced Action labels, randomised-wording GRPO scores
# Machine 18/20 (plain 13, original-wording GRPO 10) and Parcels 20/20 = 13/13 distinct (plain 7/13, 6-0, p = .031).
# Found after looking, so confirm on fresh problems with the same lenient parser for every model: all 21 Parcels
# puzzles, and Machine on 40 fresh seeds (39 distinct).
evl21() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P lenient_twin_evaluate.py --policy mlx:$Q "${@:2}" --envs parcels21 --seed0 10080 --n 21 --out runs/$1 > /dev/null 2>&1; yield_turn; }
evlm() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P lenient_twin_evaluate.py --policy mlx:$Q "${@:2}" --envs machine --seed0 10020 --n 40 --out runs/$1 > /dev/null 2>&1; yield_turn; }
evl21 p21L_grpo_rand --adapter adapters/q7_grpo_rand
evl21 p21L_plain
evl21 p21L_grpo --adapter adapters/q7_grpo
evlm repL_machine_grpo_rand --adapter adapters/q7_grpo_rand
evlm repL_machine_plain
evlm repL_machine_grpo --adapter adapters/q7_grpo
# C23 (Ying 2026-11-16, after a news piece on Anthropic's "Closing the Loop"): can the student write its own varied
# wordings? Qwen proposes 20 HiddenRule wordings (temp 1.0), a mechanical check filters them, and the C17 dataset is
# rebuilt with them. Compare Machine with vary_hm (experimenter's wordings, 20/20) and dup_hm (repeats, 6/20).
[ -s data/q7_selfvary/train.jsonl ] || { gate; log "make_selfvary"; $P make_selfvary.py data/q7_distill --k 4 --out data/q7_selfvary > runs/make_selfvary.log 2>&1; yield_turn; }
train data/q7_selfvary adapters/q7_selfvary
ev selfvary_hm --envs hiddenrule,machine --adapter adapters/q7_selfvary
# C23 result: Qwen's own wordings (narrow: one noun, two letters, one verb) give Machine 18/20, against 6/20 for
# repeats (13-1) and 20/20 for the experimenter's. How little variety is enough? G12: four copies in ONE fixed new
# wording. Moved ahead of the Llama and format-reward runs because it is cheap and answers that question.
train data/q7_distill_one adapters/q7_distill_one
ev one_hm --envs hiddenrule,machine --adapter adapters/q7_distill_one
# one_hm: HiddenRule 0/20, Machine 7/20, 140 invalid moves ("Action:Thought:..." loops). Its adapter is the
# iteration-49 checkpoint (best of a 26-row validation set), while vary/dup kept iteration 149. Same model at
# iteration 140, to tell the dose effect from an early-checkpoint format break.
ev one_i140_hm --envs hiddenrule,machine --adapter adapters/q7_distill_one_i140
# C24 (Ying 2026-11-16: second model family, local only, no paid API): replicate C17 on Llama-3.1-8B-Instruct-4bit.
# Same data (varied wordings vs the same rows repeated), same LoRA settings, HiddenRule + Machine. The download is
# ~4.5 GB; skip (and log it) under 10 GB free: at 7.8 GB free a 23 MB adapter save already failed (q7_selfvary).
L=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
roomy() { [ "$(df -g ~ | awk 'NR==2{print $4}')" -ge 10 ] || { log "skip $1: under 10 GB free"; return 1; }; }
evq() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P $S/evaluate.py --policy mlx:$L "${@:2}" --out runs/$1 --n $N > /dev/null 2>&1; yield_turn; }
trainq() { [ -f "$2/adapters.safetensors" ] && return 0
  gate; log "train $2"; $P $S/train_lora.py --model $L --data "$1" --adapter-path "$2" --iters 200 --lr 2e-5 \
     --num-layers 8 --max-seq-length 2048 --save-every 20 --steps-per-eval 50 --val-batches 4 > "$2.log" 2>&1; yield_turn; }
if [ -s runs/llama_plain_hm/summary.json ] || roomy llama; then
  evq llama_plain_hm --envs hiddenrule,machine
  trainq data/q7_distill_vary adapters/llama_distill_vary
  trainq data/q7_distill_dup adapters/llama_distill_dup
  evq llama_vary_hm --envs hiddenrule,machine --adapter adapters/llama_distill_vary
  evq llama_dup_hm --envs hiddenrule,machine --adapter adapters/llama_distill_dup
fi
grpo_fmt() { gate; [ -f "$1/adapters.safetensors" ] && return 0
  log "grpo $1"; $P grpo.py --envs "$2" --iters 60 --group 4 --format-penalty 1 --out "$1" > "$1.log" 2>&1; yield_turn; }
grpo_fmt adapters/q7_grpo_randfmt randhiddenrule,randoddcoin
evt twin_grpo_randfmt --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_grpo_randfmt
# C22 format reward result (strict parser): HiddenRule 11, OddCoin 19, Parcels 18/20 (11/13 distinct), Machine
# 13/20 (plain 13). Invalid moves 198 -> 19. Fresh-problem tests with the strict, frozen parser for every model.
evs21() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P twin_evaluate.py --policy mlx:$Q "${@:2}" --envs parcels21 --seed0 10080 --n 21 --out runs/$1 > /dev/null 2>&1; yield_turn; }
evsm() { [ -s runs/$1/summary.json ] && return 0; gate; log "$1"; $P twin_evaluate.py --policy mlx:$Q "${@:2}" --envs machine --seed0 10020 --n 40 --out runs/$1 > /dev/null 2>&1; yield_turn; }
evs21 p21_grpo_randfmt --adapter adapters/q7_grpo_randfmt
evs21 p21_grpo --adapter adapters/q7_grpo
evsm repS_machine_grpo_randfmt --adapter adapters/q7_grpo_randfmt
evsm repS_machine_plain
evsm repS_machine_grpo --adapter adapters/q7_grpo
# Control for C19: the same budget of sampled episodes (seeds 1000-1029 x 2 games x 4, temp 0.9) from the plain
# model, keeping only wins and fine-tuning on them (rejection sampling, one round): online updates and the
# push-down on losing episodes are what GRPO adds.
[ -s runs/collect_rft/episodes.jsonl ] && [ "$(wc -l < runs/collect_rft/episodes.jsonl)" -ge 240 ] || { gate; log "collect_rft"; $P $S/collect.py --policy mlx:$Q --temp 0.9 --seeds 1000:1030 --k 4 --games hiddenrule,oddcoin --out runs/collect_rft > runs/collect_rft.log 2>&1; yield_turn; }
[ -s data/q7_rft/train.jsonl ] || $P $S/build_sft.py runs/collect_rft --out data/q7_rft > runs/build_rft.log 2>&1
train data/q7_rft adapters/q7_rft
evt twin_rft --envs oddcoin,parcels,hiddenrule,machine --adapter adapters/q7_rft
# G12: one alternative wording (all four copies the same) vs four random ones vs repeats, HiddenRule -> Machine.
train data/q7_distill_one adapters/q7_distill_one
ev one_hm --envs hiddenrule,machine --adapter adapters/q7_distill_one
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
log "pipeline2al done"
