# Teaching LLMs to Solve Problems: Which Learning-Science Findings Carry Over?

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Can an LLM's problem-solving capability be taught so that it transfers to problems it never practised?
This repository uses small interactive games as the practice field and the exam, and tests, one by one,
the classic findings from human learning (worked examples, expertise reversal, pressure, encouragement,
variability of practice, analogical comparison, self-explanation) on Qwen2.5-7B-Instruct (4-bit, MLX).
The interesting results are where the model behaves *unlike* a human learner.

**Status: work in progress.** Some findings below were found after looking at the data and are being
re-tested on fresh seeds (pre-registered in `CONCLUSIONS.md`). Treat them as preliminary.

## Findings so far
1. **Failure mechanism: acting against evidence the model already has.** An episode with even one action that
   contradicts earlier feedback almost never succeeds: coin weighing 24% vs 92% without, black-box function
   0% vs 83%, bug fixing 1% vs 57% (p < 1e-5, 13 conditions pooled).
2. **Expertise reversal (as in people; post hoc, replicating).** How-to guidance helps where the model is weak
   (black-box function 4 → 11/20) and hurts where it is strong (coin weighing 16 → 7/20), through more answers
   that contradict its own weighings.
3. **Pressure hurts, encouragement does nothing.** Yelling: coin weighing 16 → 7/20; encouragement: no change.
4. **Practice improves practised games and nothing else.** No condition beats the plain model (3/40) on the
   never-practised problems.
5. **Unlike people: imitating narrated checking teaches the claim, not the check.** After fine-tuning on expert
   play that calls a free CHECK tool before committing, the model uses the tool on unpractised problems
   (28/65 actions vs 2/99 untrained) but does not solve more of them, and writes "all observations agree"
   for answers it never checked.

Numbers come from `analyze.py` (→ `ANALYSIS.md`); interpretation and the people-vs-LLM table are in
`CONCLUSIONS.md`; the running log is `PROGRESS.md`; `REPORT.md` is a progress summary (in Chinese).

## Problems
| Kind | Problems | Role |
|---|---|---|
| Practice | GuessNumber, CodeBreak, HiddenRule, OddCoin | may be trained on |
| Extra practice | Heaviest, Radar, Boxes, Switches | variability-of-practice conditions |
| Near transfer | Lock, Machine | same skill as a practice game, new surface |
| Far transfer | Circuit, BugFix | never trained on |

Every verifier is exact; evaluation seeds start at 10000 and are never trained on.

## Layout
- `envs.py`, `agents.py`, `evaluate.py`, `collect.py`, `build_sft.py`, `diagnose.py`, `analyze.py`: the original code.
- `frozen/`: the code used for the later conditions, frozen so every run stays comparable (extra practice games,
  `process_check.py`, `selfexplain.py`, `train_lora.py`).
- `playbooks/`: prompt-side conditions (expert rules, analogical comparison).
- `data/`: fine-tuning sets. `runs/`: episode transcripts and summaries for every condition.
- `pipeline*.sh`: the exact run order of each experiment queue.

## Reproduce
Apple Silicon with 16 GB, Python 3.12, `mlx-lm`.
```bash
python -m venv .venv && .venv/bin/pip install mlx-lm
.venv/bin/python evaluate.py --policy mlx:mlx-community/Qwen2.5-7B-Instruct-4bit --out runs/plain --n 20
.venv/bin/python analyze.py   # regenerates ANALYSIS.md from runs/
```
Only one 7B process fits in 16 GB; the pipelines wait for the GPU before each step.

## License
MIT, see [LICENSE](LICENSE).
