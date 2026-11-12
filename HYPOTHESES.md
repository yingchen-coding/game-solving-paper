# How to raise a problem solver: lecture, practice, praise, pressure, reminders, hints, coaching, or dreams?

The big lines (Ying 2026-11-11: "encourage? hints? yell at it? reminder? dream?"). Each is something a parent
or teacher does; each is tested two ways:
- **In the moment**: the teacher talks during the TEST episodes (does it help right now?) — `runs/moment_*`.
- **Does it stick**: the teacher talks only during PRACTICE; the model keeps its wins, the teacher's words are
  removed from the training data, the model is fine-tuned, and it is tested ALONE — `runs/stick_*`.

| Teaching style | What the model hears | Information content | Run |
|---|---|---|---|
| Lecture | a hand-written strategy in the system prompt | general advice, once | `q7_strategy` |
| Practice | nothing; learns from its own wins (expert iteration) | outcome only | `q7_ei_r1` |
| Encourage | "You're doing great, keep going!" every turn | none | `moment/stick_encourage` |
| Yell | "Wrong! Think harder." every turn | none | `moment/stick_yell` |
| Remind | "Remember your task: <goal>" every turn | the goal only | `moment/stick_remind` |
| Hint | a generic question, e.g. "What does the last result rule out?" | a direction, no facts | `moment/stick_hint` |
| Coach | a fact from its own transcript, e.g. "this guess contradicts the feedback for guess 1 2 3" | specific evidence | `moment/stick_coach` |
| Dream | after a failure: reflect, then replay the same problem with that lesson; learn from replays that work | its own hindsight | `stick_dream` |
| Reflect (diary) | lessons written after each game, kept as a playbook in its own prompt | its own hindsight, kept in context | `q7_playbook` |

Predictions (written before any of these results): praise and yelling carry no information, so neither helps
in the moment or sticks (yelling may hurt: more guessing, more repeats). Reminders help only where the goal is
forgotten (long episodes). Hints and coaching help in the moment; only coaching and dreaming stick, because
they produce practice wins of better quality, not just more of them. One-line conclusion the paper aims to test:
**"Information teaches, emotion doesn't: coach and dream, don't cheer or yell."** If the data says otherwise, the
title changes, not the data.

Title candidates (pick after results):
1. Praise, Pressure, or Practice? How to Raise an LLM Problem Solver Through Games
2. Don't Yell at Your Model: What Actually Teaches LLMs to Solve New Problems
3. Coach, Don't Cheer: Teaching Styles That Make LLM Problem-Solving Skills Stick
4. Sleep On It: Dreaming Makes Game-Learned Problem Solving Transfer   (only if Dream wins)

---

## Detailed hypotheses (secondary)

Pre-registered 2026-11-11 01:00, before any method result existed (only baselines were running).
Model: Qwen2.5-7B-Instruct 4-bit. Learning only on TRAIN_GAMES seeds < 10000. Test: eval seeds >= 10000.

Primary metric: **held-out success** = mean success on Circuit and BugFix (problem types never practised).
Secondary: near-transfer success on the four games (new seeds); mechanism metric `repeat_rate`
(repeating an identical action) and invalid-action rate.
A hypothesis is supported only if its method beats BOTH the plain baseline AND the hand-written strategy
prompt on held-out success. With n=20 per env (40 held-out episodes), a difference under ~15 points is
within noise; such results are reported as "no detectable effect", not as support.

| ID | Hypothesis | Method (what changes) | Prediction |
|----|-----------|------------------------|-----------|
| H0 | Telling the model how to solve is enough | hand-written strategy prompt | no gain over plain (3B already showed none) |
| H1 | Outcome reward alone forms skills | expert iteration: train on own successful games (`q7_ei_r1`) | gains on the games; little on held-out |
| H2 | Reflecting on own play forms transferable rules | free-form lesson per episode → playbook (`q7_playbook`) | small held-out gain |
| H3 | Naming the error type makes lessons transfer | reflection must classify the failure (repeated a failed action / concluded before evidence was sufficient / answer contradicts own evidence / ignored feedback) before writing the rule | beats H2 on held-out |
| H4 | Seeing the true answer is what makes reflection useful | H2 without the revealed answer | worse than H2 |
| H5 | Contrast is the teacher | lesson from a failed and a successful attempt at the same problem (what differed) | beats H2 |
| H6 | Scaffold then fade: an in-game coach that points out evidence violations during practice forms the habit of checking evidence; the coach is removed at test | coached play → keep successes → strip coach lines → fine-tune | beats H1 on held-out and lowers repeat_rate |
| H8 | Transferred expert habits beat self-learned ones | a playbook written by a stronger agent (a stronger LLM agent, from its own working habits: `playbooks/expert.txt`) | beats H2; upper reference for context learning |
| H7 | Context and weights add up | H1 adapter + H2 playbook (`q7_ei_playbook`) | ≥ max(H1, H2) |

Coach (H6) only states facts derived from the transcript, never the answer: "this action repeats one you
already took", "this guess contradicts the feedback you got for guess X", "this formula gives f(a)=u but you
observed f(a)=v", "coin k was already ruled out". An answer that contradicts evidence is refused once with that
note (practice only).

Caveat on H8: the expert playbook was written after seeing 7B smoke transcripts, including held-out BugFix
and Circuit ones; rules 3, 5 and 8 talk about fixes, checks and intermediate values. H8 is therefore NOT a
clean held-out test — it is a reference for how far good context can go, not evidence of learning.

Results go in `RESULTS.md`, one row per hypothesis, with the run directory.
