# Paper plan: games as playground and eval arena for LLM problem solving — read this first

Owner: Ying Chen. Target: arXiv preprint (cs.LG / cs.CL), Ying as author; same publish route as
`frustration-paper` (public GitHub repo under yingchen-coding, Zenodo DOI, arXiv package) — only after
results exist and `/pr-review` plus the personal-info scan pass.
Folder: `~/Documents/game-solving-paper/` (renamed from persistence-paper on 2026-11-10). Started 2026-11-10.

## Rules
- Reply to Ying in Chinese. Code, file names and paper text in English.
- Every number comes from a script here that can be rerun. Negative results are reported as they are.
- Learning (weights or playbook) uses ONLY `envs.TRAIN_GAMES`, seeds < 10,000. Held-out problems
  (`envs.HELD_OUT`) and all eval seeds (>= 10,000) are never learned from.

## The question, in her words
"A coding agent cannot work through one PR, it is not sharp enough to get to the root cause, forgets the goal."
"There are so many games — can I use games to tune or RL the problem-solving capability of an LLM, so
it can solve new problems like a human?" "It is problem solving, not training persistence."
"Or train an agent whose system prompt lets it introspect and learn, or fine-tune a small model — I want a
method that improves LLM problem-solving." "Games are the playground and the eval arena."

## Design
Playground (learning) — problems that need reasoning from feedback, each with an exact verifier:
GuessNumber (search), CodeBreak / Mastermind (deduction), HiddenRule (induction of a formula from
queries), OddCoin (information-efficient tests). Eval arena = held-out seeds of those (near transfer)
+ held-out problem types never learned from: Circuit (localize a faulty pipeline stage by probing —
diagnosis) and BugFix (failing tests → read → patch → rerun; graded on HIDDEN inputs so hard-coding
fails — the shape of a real PR).

Agent format: think briefly, then `Action: ...`. Honest scripted experts (see only the transcript) solve
100% of 200 seeds per env within budget, so every failure is the model's.

Methods (Qwen2.5-3B-Instruct 4-bit, MLX, M4 16 GB):
- base, plain prompt; base + hand-written strategy prompt (inference-time control)
- A. self-reflection learning (`reflect.py`): play → see outcome + revealed answer → write a general lesson
  → rewrite a ≤8-rule playbook in its own system prompt; 3 rounds. No weight change.
- B. expert iteration / rejection-sampling RL (`collect.py` → `build_sft.py` → `mlx_lm lora`): sample own
  play at T=0.8, keep verifier-passed episodes, LoRA-train on them.
- C. A + B.
Claim needed: A/B/C beat the strategy prompt on the held-out problem types, not just on game seeds.

## Prior work (checked 2026-11-10)
SPIRAL (2506.24119): self-play RL on games → math/reasoning gains (accuracy on static benchmarks; not
interactive held-out problems). PushBench (2605.23574): persistence, inference-time controllers. Multi-domain
RL (2507.17512): puzzle RL transfers inconsistently to coding. RAGEN (2504.20073): multi-turn RL collapse.
Reflection/playbook line (checked 2026-11-10): ACE (2510.04618, ICLR 2026; playbook context via
generate/reflect/curate), ERL (2603.24639; heuristics transfer across tasks better than trajectories, beats
ExpeL), Memory Transfer Learning (2604.14004; meta-knowledge dominates, algorithmic strategy transfer
marginal), Knowledge-Centric Self-Improvement (2607.19592), Procedural Memory Distillation (2607.01480),
Stratagem (game-learned patterns stay anchored in game semantics). Our angle: the SAME game playground feeds
both weight learning (RL) and context learning (playbook), compared head-to-head on held-out problem types
that include debugging, with honest experts proving solvability. Still to read in full before writing.

## Status
`PROGRESS.md`. Older persistence design kept in `old/` and `runs_persistence_v0/` (not used).
