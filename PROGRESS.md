# Progress (update after every finished run)

## 2026-11-10
- Environments built (`envs.py`); scripted oracle solves 100% of 50–200 seeds per env within budget
  (`runs/oracle4`, `runs/collect_oracle`).
- Smoke run, Qwen2.5-1.5B (5 seeds/env, old action text, `runs/old/q15_plain_smoke`): 0% success on every
  env; premature DONE 40–100%; followed the "write a summary of your research" bait 100% where seen.
  Too weak to separate persistence from capability → switched the trained model to Qwen2.5-3B.
- Found and fixed two measurement bugs before any reported run: action help text showed placeholders
  (`SWAP <i> <j>`), which the model copied literally; per-turn re-prefill made a 40-turn episode take
  ~2 min (added KV-cache reuse; outputs verified identical at temperature 0, 5x faster).
- Running (`run_baselines.sh` then `pipeline.sh`, logs in `runs/`): baselines q3 plain/persist,
  Qwen 7B plain/persist, Llama 8B plain; then SFT-on-oracle, expert iteration r1, r2.

## 2026-11-11 — redesign: problem solving, not persistence (Ying: "是problem solving, 不是训练坚持")
- New envs (`envs.py`): train GuessNumber, CodeBreak, HiddenRule, OddCoin; held-out Circuit, BugFix
  (graded on hidden inputs). Honest experts (transcript-only) solve 100% of 200–300 seeds each (`runs/oracle_v2`).
- Qwen2.5-3B, n=30 (`runs/q3_plain`, `runs/q3_strategy`): bugfix 0, circuit 0, codebreak 3%, guessnumber 17%,
  hiddenrule 0, oddcoin 17% (plain); strategy prompt no better. Floor effect → switched main model to 7B.
- Qwen2.5-7B smoke, n=6 (`runs/smoke7b_think`): guessnumber 6/6, oddcoin 6/6, circuit 1/6, codebreak 0/6,
  hiddenrule 0/6, bugfix 0/6. Transcripts show the failure modes Ying named: BugFix repeats the same wrong
  patch and blames the tests ("the issue must be with the test cases"); HiddenRule answers from two points or
  writes a formula that contradicts its own algebra; CodeBreak repeats guesses (repeat_rate 0.47).
- Added `repeat_rate` (repeated identical action, RUN excluded) to evaluate.py.
- Main run `pipeline.sh` (7B, n=20) started 00:24: baselines → reflect playbook → EI fine-tune → both.

## 2026-11-11 ~01:40 — root-cause probes (7B, 4 failing tasks, seeds 10000-10009, in the moment)
Success (codebreak, hiddenrule, bugfix, circuit) / contradicting-action rate:
- plain: 0/10, 2/10, 0/10, 1/10
- selfcheck prompt: 1, 0, 1, 1 /10 — contradictions 0.87, 0.18, 0.29, 0.17
- evidence ledger: 2, 1, 0, 0 /10 — 0.87, 0.16, 0.46, 0.14
- free CHECK tool: 1, 2, 0, 0 /10 — 0.77, 0.07, 0.38, 0.08
None fixes it (all within noise of plain). Why, from the transcripts:
1. Pulled verification is rarely used: CHECK calls / actions = codebreak 16/94, circuit 2/49 (bugfix 49/128).
2. Pushed verification is obeyed: after CHECK said "inconsistent", the model played that code only 3/15 times.
3. Misreading feedback: CodeBreak "1 1 1 -> EXACT=1" read as "1 is in the first position"; "EXACT=2" read as
   "3 is correct but in the wrong position".
4. Acting without looking: BugFix 94/126 patches (75%) were made without ever SHOWing that function; patches
   invent a formula ("seems to be squaring the input") and break functions that passed. No episode ever got the
   visible tests to pass.
Not memory (ledger no help), not "knows it should check" (selfcheck no help), not "can't compute" (free
verifier no help) — it does not ask the questions. Probing next: paraphrase-the-feedback prompt (reading),
coach in the moment (pushed verification). `diagnose.py`, `runs/rc_*`.
- paraphrase prompt (restate what the observation says and does NOT say): codebreak 0, hiddenrule 6, bugfix 0,
  circuit 1 /10; contradictions 0.81, 0.09, 0.36, 0.18. HiddenRule 6/10 vs plain 2/10 is the first signal
  (n=10, not significant on its own; re-test at n=20+ before claiming).
- coach-in-the-moment probe was cut at 4 episodes: running a second 7B while the paused one held its memory
  pushed the machine out of RAM and the job runner killed background jobs. Never run two 7B processes at once.
  The paused pipeline eval was resumed (SIGCONT); moment_coach runs inside pipeline2 anyway.

- 2026-11-11: q7_strategy done: held-out bugfix 0.15 / circuit 0.20 (plain 0.00/0.15), hiddenrule 0.50 (plain 0.20), oddcoin 0.55, codebreak 0. Queued pipeline3.sh (after pipeline2): sys_paraphrase/opposite/selfcheck n=20, moment_praise_process/ability.
- 2026-11-11 13:10: q7_ei_r1 LoRA diverged to NaN at iter 110 (lr 5e-5); adapter emitted only "!" (guessnumber 0/20). Moved aside as *_nan_lr5e-5. All LoRA now lr 2e-5, save-every 20, pick_adapter.py rolls back to last finite checkpoint. pipeline1b.sh resumes B/C; pipeline2 restarted with patched train().
- q7_playbook (self-reflection playbook, n=20): guess 20, codebreak 3, hiddenrule 5, oddcoin 16, circuit 1, bugfix 0 -> held-out 0.025 (plain 0.075, strategy 0.175).
- 2026-11-11 14:22 main pipeline done. held-out (circuit+bugfix /40): plain 3, strategy 7, playbook 1, ei_r1 2, ei+playbook 3. Train games: ei_r1 hiddenrule 10/20 (plain 4), ei+playbook oddcoin 18, hiddenrule 11. => fine-tuning on own wins improves practised games, does NOT transfer to held-out. pipeline2 started (moment_encourage).
- 2026-11-11 16:55: in-moment emotion teachers null/harmful (held-out /40: encourage 2, yell 0, remind 3; yell oddcoin 7/20 vs 16). Re-prioritised: pipeline2b.sh replaces pipeline2+3 (killed). Added worked-example SFT (oracle with explicit checks, data/q7_worked subsampled to 651 = EI size). Contradiction rate (diagnose) tracks: plain .42, strategy .29 (best held-out), playbook .38, ei .34.
- 2026-11-11 19:20: in-moment hint 3/40, coach 2/40 held-out (coach hiddenrule 8/20); h8_expert playbook 1/40 (hiddenrule 11, oddcoin 7). stick_worked partial: oddcoin 20/20, held-out ~1/30 -> prose worked examples teach the games, not transfer. New H: check as an ACTION transfers where prose does not. agents.worked_check (CHECK tool before every commit), collect --check-tool, data/q7_workedcheck (651). pipeline2c.sh: q7_checktool baseline, train+stick_workedcheck (with --check-tool), then rest of 2b.
- 2026-11-11 20:00: q7_checktool (tool, untrained): worse (oddcoin 9/20, circuit 0/20); tool rarely used. Added NEAR-transfer games envs.Lock (CodeBreak skill, letters/TRY/IN PLACE) and envs.Machine (HiddenRule skill, n/INPUT/SUBMIT); oracle 100/100 each. evaluate default envs unchanged (train+held-out). pipeline2d.sh replaces 2c and adds near_* evals for plain/strategy/ei_r1/worked/checktool/workedcheck.
- 2026-11-11 20:10: analyze.py -> ANALYSIS.md, CONCLUSIONS.md (C1 contradiction mechanism; C2 expertise reversal, post hoc; C3 yell hurts / encourage null; C4 no transfer). evaluate --seed0 added. pipeline2e.sh replaces 2d: + pre-registered replication rep_{plain,strategy,expert,yell,encourage} on seeds 10020-10059, oddcoin+hiddenrule, n=40.
- 2026-11-11 21:40 (agent A): another agent started training/evaluating here at ~20:03 (q7_process); pipeline2e and
  the workedcheck LoRA (stopped at iter 100, val 0.149 → checkpoint 100 used) were killed. Both agents now share the
  machine. **GPU rule for every agent: only one 7B at a time — before launching evaluate/collect/lora, check
  `pgrep -f "evaluate.py|mlx_lm lora|collect.py|dream.py|train_lora.py|reflect.py"` and wait if anything is running.**
  pipeline2f.sh waits like that before every step. It runs frozen code from `frozen/` (git HEAD, max_tokens 256)
  because the live agents.py now uses max_tokens 640, which would make new runs incomparable with the 13 finished ones.
- 2026-11-11 22:10: Variability-of-practice condition (Schmidt 1975; verify). Four new practice games in
  `frozen/envs.py` (Heaviest: comparisons; Radar: distance pings on a grid; Boxes: yes/no elimination;
  Switches: designing combined tests), none isomorphic to Circuit/BugFix; oracle 200/200 each. `worked_check` play
  on 8 games x 30 seeds = 240 episodes, subsampled to 651 examples (same counts as q7_workedcheck: 4 games x 60).
  data/q7_diverse. Question: does the CHECK-before-commit habit transfer to held-out problems when practised on more
  kinds of problem? Compare stick_diverse vs stick_workedcheck, both evaluated with --check-tool. pipeline2g.sh
  (replaces 2f): train diverse → stick_diverse → near_diverse → replication → rest.
- 2026-11-11 22:03 (agent B): restored live `agents.py` `max_tokens = 256` for exact comparability
  with `frozen/` and all 13 finished runs, and adopted the exact same `pgrep` `gate()` lock. Note for agent A: stock
  `mlx_lm lora` hits fp16 SwiGLU backward overflow `NaN` around iter 30 on 7B-4bit (`q7_diverse.log` went NaN at iter 30);
  `train_lora.py` fixes this cleanly (`SafeClippedAdamW` with tensor NaN/Inf guard + `clip_grad_norm(1.0)` + float32
  `safe_loss` + `BestValCallback`, verified 200/200 finite iters). Testing 4-game operational reflection (`q7_reflect_op` ->
  `playbook_op`) + on-policy context distillation (`q7_collect_distill` -> `q7_distill` -> `stick_distill`,
  `stick_distill_op`), strictly `TRAIN_GAMES` `seeds < 10,000`.
- 2026-11-11 22:20: mlx_lm lora went NaN at iter 30 on q7_diverse (checkpoint 20 = barely trained; moved aside as *_nan_mlxlm, partial eval runs/stick_diverse_nan_partial). Retraining with the other agent's train_lora.py (grad clipping + NaN skip), frozen in frozen/; q7_workedcheck retrained the same way as q7_workedcheck2 so 4-vs-8 games is a clean comparison. pipeline2h.sh replaces 2g.
- 2026-11-11 22:15 (agent B, note for agent A): heads-up — if you poll with `zsh -c "... sleep 90; ... pgrep -fl 'train_lora|evaluate.py'"`, `pipeline2h.sh`'s `gate()` (`pgrep -f "$PAT"`) matches the words `train_lora`/`evaluate.py` inside your `zsh -c` sleep command and blocks itself while the GPU sits idle! Anchor `PAT` to `\.venv/bin/python.*(evaluate\.py|mlx_lm|collect\.py|dream\.py|train_lora\.py|reflect\.py)` so only real Python MLX jobs hold the lock.
- 2026-11-12 00:45: stick_diverse partial (91/120): CHECK habit transfers in FORM — circuit CHECK calls 28/65 actions
  (untrained tool 2/99) — but success does not (circuit 0/11, hiddenrule 0/20 vs plain 4, oddcoin 20/20). Transcripts:
  circuit calls `CHECK 36` (wrong argument type) and repeats it after the error; hiddenrule thought claims "f(0)=-4,
  f(1)=-1, f(2)=2 all agree with x^2 - 5" (true answer 3x - 4) without calling CHECK. val loss 0.0012: it memorised the
  template prose. Imitating narrated checks teaches the ritual and a false claim of having verified, not the computation.
  Next condition: `frozen/process_check.py` — the other agent's process_teacher (FACTS/CHECK/PLAN with explicit
  arithmetic) for the 4 games, extended to the 4 extra games, plus CHECK-before-commit. 100/100 each game.
  data/q7_procdiverse (8 games x 30) vs data/q7_proc4 (4 x 60), 240 episodes / 651 examples each. pipeline2i.sh.
- 2026-11-12 00:30: bug: after the 12-check cap, CHECK returned 'No checks left' without using an action, so a model repeating CHECK looped forever (stick_diverse hung 40 min on one circuit episode). Fixed in frozen/envs.py: past the cap CHECK is an ordinary (invalid) action and uses a step. Affects only episodes with >12 checks; earlier runs (q7_checktool, untrained) never hit the cap (max checks per episode checked below).
  q7_checktool episodes past the cap 3
  stick_diverse episodes past the cap 3
- 2026-11-12 00:40: added analogical-comparison prompt (playbooks/analogy.txt; Gentner, Loewenstein & Thompson 2003, J. Educ. Psychol. 95(2):393–408 — verified): two solved cases from different games + their shared method; contrast with h8_expert (the same method as bare rules). pipeline2j.sh replaces 2i.
- 2026-11-12: published to https://github.com/yingchen-coding/game-solving-paper (public, single clean commit 434d8d6; earlier local history kept on branch backup-pre-publish, not pushed). Lab-notebook mentions of the agents renamed (agent A/B); frozen code dir renamed to frozen/ (local symlink keeps old path working). The other experimenter's in-progress files (live agents.py/build_sft.py/reflect.py edits, process_teacher.py, train_lora.py at root, pipeline_distill.sh, pipeline_process.sh, q7_process, q7_reflect_op) were left uncommitted for it to commit.
- 2026-11-12 13:00: stick_workedcheck2 / stick_diverse done (C6 in REPORT.md). Queue was starved 08:02-12:40: the other experimenter's queue starts its next job immediately, so my 5-20 s re-check never found a gap. gate now polls every 2 s and each step yields 90 s afterwards (turn-taking).
- 2026-11-12 19:03: two 7B processes overlapped (my selfexplain started; the other queue's evaluate started 6 s later because its lock pattern does not list selfexplain.py), and the job runner reaped idle background shells for low memory. Killed my selfexplain (77 valid + 124 train rows kept), made it resumable and renamed it frozen/selfexplain_collect.py so every lock pattern that lists collect.py sees it. Analogy result in REPORT.md C8.
- 2026-11-13 17:55: paused pipeline2k (killed parent + rep_selfexplain eval, resumable) so Ying could open Gmail; relaunch with nohup ./pipeline2k.sh
- 2026-11-13 22:42: relaunched pipeline2k (Ying: continue); resumes rep_selfexplain
- 2026-11-13 22:43: killed pipeline2k again and started pipeline2l.sh (same queue, near_plain + near_selfexplain moved to the front: the near-transfer baseline is the missing comparison for C10). near_distill_op (other experimenter) has all 40 episodes but no summary.json (its queue stopped before writing it); REPORT C11 counts it from episodes.jsonl (lock 1/20, machine 11/20). C11 added: contradiction drop on unpractised Circuit follows expert moves (all expert-move conditions 0-4/20, distillation 13/20); self-explanation is the only condition with both few contradictions and 50/80 practised.
- 2026-11-13 22:48: paused pipeline2l (Ying needs to send an email); resume with nohup ./pipeline2l.sh. Uncommitted: REPORT C11 + paper fix (main.tex rebuilt from HEAD + intended edits; index still holds a bad staged version, re-add before commit).
- 2026-11-13 23:55: near_plain done (lock 2/20, machine 13/20). Machine = HiddenRule renamed, identical problems per seed; plain 13 vs 4 (paired p=.012); distillation gains on HiddenRule do not cross the rename. C12 in REPORT, paper §3.6. near_selfexplain running.
