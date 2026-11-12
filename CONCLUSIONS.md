# Conclusions so far (2026-11-11)

Every number comes from `analyze.py` → `ANALYSIS.md` (rerunnable). Qwen2.5-7B-Instruct 4-bit, greedy decoding,
20 problems per task per condition, evaluation seeds never trained on. 13 conditions finished.

## C1. The failure is acting against evidence the model already has — the strongest finding
In every task, an episode with even one action that contradicts earlier feedback almost never succeeds:
coin weighing 24% vs 92% without, black-box function 0% vs 83%, bug fixing 1% vs 57% (all p < 1e-5, pooled over
13 conditions). Across conditions, a task's contradiction rate tracks its success (Spearman −0.82 coin
weighing, −0.78 black-box function).
**Caveat:** partly by construction where the contradiction *is* the final answer (black-box function, pipeline
fault). The non-trivial part: the model has the evidence in its context and still answers against it.

## C2. Telling the model how to work helps where it is a novice and hurts where it is already good
(Expertise-reversal effect: Kalyuga, Ayres, Chandler & Sweller 2003, *Educational Psychologist* 38(1):23–31 — verified.) Against the light-touch controls (encourage, remind,
hint pooled):
| | plain | strategy prompt | expert rules | controls |
|---|---|---|---|---|
| black-box function (novice, 20%) | 4/20 | **10/20** (p=.006) | **11/20** (p=.002) | 10/60 |
| coin weighing (good, 80%) | 16/20 | 11/20 (p=.10) | **7/20** (p=.002) | 45/60 |
It hurts through the same mechanism as C1: with expert rules, 12/20 coin-weighing answers name a coin the
model's own weighings had ruled out (plain: 3/20).
**Caveat:** found after looking at the data (post hoc). Must be confirmed on new seeds before it is a claim.

## C3. Pressure hurts, cheering does nothing
Yell: coin weighing 16 → 7/20 (p=.010), and 13/20 answers contradict the model's own weighings (plain 3/20).
Encourage: no change on any task (39 vs 40 of 80 practised, 2 vs 3 of 40 never practised).
Same direction as Yerkes–Dodson in people: pressure harms the harder task. Single run; replicate on new seeds.

## C4. Practice makes the practised game better and nothing else (no transfer yet)
- Fine-tuning on its own wins + playbook: practised games 40 → 52/80 (+15 points, CI [0, +30], p=.08 — not
  significant after correction). Worked examples: coin weighing 20/20 with zero contradictions.
- Problems it never practised (pipeline fault, bug fixing): nothing beats plain (3/40). Best is the strategy
  prompt, 7/40, p=.31.
- Giving it a free CHECK tool, untrained, makes it worse (34 vs 40 of 80 practised): it rarely calls it, and
  when it does it often answers against the result.

## What this means for the paper
The memorable line is not "X works" but: **"LLMs fail new problems by ignoring evidence they already have;
guidance only helps where the model is a novice, pressure makes it worse, and practice does not transfer."**
That is useful to practitioners: (1) measure contradiction rate, not just success; (2) don't put generic
"how to solve" rules in prompts for tasks the model already does well; (3) never use pressure wording.

## What is still needed before writing
1. **Replicate C2 and C3 on new seeds** (10020–10059, pre-registered): strategy, expert rules, yell,
   encourage, plain. ~4 h.
2. **Near-transfer test** (Lock, Machine: same skill, new surface) — queued in `pipeline2d.sh`. Decides whether
   C4 is "no transfer at all" or "transfer of skill but not across task types".
3. **Check-as-action training** — 4 games (q7_workedcheck) vs 8 games (q7_diverse, same 240 episodes / 651
   examples): does the check habit transfer when practised on more kinds of problem? (Variability of practice:
   Schmidt 1975, *Psychological Review* 82:225–260 — verified; evidence in people is mixed, Van Rossum 1990.)
4. **Power:** 20 per task only detects ~±35-point changes; held-out claims need ≥ 60 per task.

## Framing (2026-11-12, Ying): teaching people vs teaching LLMs
Title idea: *"Teaching LLMs to Solve Problems: Which Learning-Science Findings Carry Over?"* The object is
problem-solving capability: success on practised problems, transfer to unpractised ones, and the failure mechanism.
Each row is one classic finding from human learning, operationalised the same way on the model.

| Finding in people | Prediction | LLM result | Same / different | Status |
|---|---|---|---|---|
| Worked-example effect (Sweller) | studying solved examples beats unguided practice | coin weighing 20/20, zero contradictions (plain 16/20) | same, on the practised task | done, n=20 |
| Expertise reversal (Kalyuga et al. 2003) | guidance helps novices, hurts experts | weak task 4→11/20, strong task 16→7/20 | same | post hoc, replicating on new seeds |
| Pressure harms hard tasks (Yerkes–Dodson) | yelling hurts | oddcoin 16→7/20, answers against own evidence | same effect, different cause: no affect, so the wording must act as distraction | replicating |
| Encouragement / praise | small or none in large replications | none | same | replicating |
| Far transfer is rare (Thorndike & Woodworth 1901) | practice transfers only to near tasks | practised 40→52/80, never-practised 3/40 unchanged | same | near-transfer test queued |
| Learners use a free check when given one | — | pulled CHECK rarely used (2/99 circuit); pushed check obeyed | different | done |
| Imitating narrated checking | people may skip the check | model learns to *claim* it checked without computing (false "all agree") | different, new | done |
| Analogical comparison (Gentner et al. 2003) | comparing two cases transfers better than rules | queued (analogy vs expert rules) | ? | queued |
| Self-explanation (Chi 1989) | explaining in your own words transfers | not yet run | ? | to build |
| Variability of practice (Schmidt 1975) | varied practice transfers better | 8 vs 4 games queued | ? | running |

What makes it useful: practitioners borrow pedagogy for prompting and fine-tuning by intuition; this tells them which
borrowings hold and which fail, with the mechanism (contradiction rate) measured. Needed for generality: repeat the
key rows on a second model family (e.g. Llama-3.1-8B), since one 4-bit 7B is a single learner.
