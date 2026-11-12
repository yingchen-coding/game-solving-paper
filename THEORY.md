# Theory lenses (Ying 2026-11-11: borrow education theory, psychology, educational psychology, linguistics, philosophy)

Root-cause finding that all lenses must explain (`diagnose.py runs/q7_plain`): failure tracks how often the
model acts AGAINST evidence it already has — Mastermind 81% of guesses contradict earlier feedback (0% solved),
BugFix 42% of patches fail the tests just shown (0%), HiddenRule 15/20 answers contradict queried points (20%),
GuessNumber 0% contradictions (100%). It forms conjectures; it does not test them.
All citations below are from memory and must be checked against the original before they go in the paper.

## Philosophy — how knowledge grows
- **Popper, conjectures and refutations**: knowledge grows by trying to refute a conjecture. The model makes
  conjectures and never tries to refute them. → "consider the opposite" prompt (`opposite`), CHECK tool.
- **Peirce, abduction → deduction → induction**: our tasks cover all three (Circuit/BugFix = abduction to the
  best explanation; CodeBreak = deduction; HiddenRule = induction). Predicts the check matters most where a
  conjecture must be tested against many observations.
- **Socratic elenchus**: refute by questioning → `hint` teacher (generic questions).

## Mathematics education — Pólya, *How to Solve It*
Understand → plan → carry out → **look back** (check the result). The model skips step 4. → `selfcheck`
prompt, CHECK tool, worked examples whose every answer is preceded by a check (`agents.worked_example`).

## Educational psychology — feedback and scaffolding
- **Hattie & Timperley, four levels of feedback**: task (is it right), process (how to do it), self-regulation
  (monitor yourself), self (praise of the person). Self-level feedback is the least effective; Kluger & DeNisi
  found about a third of feedback interventions lower performance. Mapping: encourage / yell / praise_ability =
  self level; coach = task level; hint = process level; selfcheck = self-regulation level.
- **Dweck, praise for effort/process vs ability** (Mueller & Dweck): ability praise makes learners avoid
  challenge. → `praise_process` vs `praise_ability` teachers.
- **Vygotsky ZPD; Wood, Bruner & Ross scaffolding, then fading** → coach during practice only, faded at test (H6).
- **Sweller, worked-example effect** → fine-tune on explained expert play (`worked` policy).
- **Kapur, productive failure** → Dream: fail first, reflect, retry.
- **Chi, self-explanation; Flavell, metacognition** → diary/playbook, error-typed reflection.

## Cognitive psychology — named failure modes (for the paper's error taxonomy)
- **Einstellung effect** (Luchins' water jars): repeating a set approach → BugFix re-applies the same wrong patch.
- **Premature closure / jumping to conclusions**: HiddenRule answers after two points.
- **Confirmation bias** (Wason 2-4-6): tests that confirm instead of discriminate.
- **External attribution** (Weiner): "the issue must be with the test cases".
- **Dual process** (Kahneman): fast System-1 answers without the slow check.
- **Yerkes–Dodson**: pressure helps simple tasks and hurts complex ones → predicts yelling hurts more on the hard
  tasks than on GuessNumber.
- Debiasing that works in humans: **consider the opposite** (Lord, Lepper & Preston) → `opposite` prompt.

## Linguistics — how evidence is worded
- **Negation is hard for language models** (e.g. negated probes): "Lower." / "MISPLACED=2" / "ruled out" are
  relative or negative constraints. → the ledger restates all evidence each turn; a further test: rephrase
  feedback as positive constraints ("possible numbers: 51–100").
- **Inner speech** (Vygotsky private speech) and **argument structure** (Toulmin: claim, evidence, warrant):
  making the model state claim + evidence per observation is the `selfcheck` prompt.
- Ying's own line (frustration paper): understand through linguistic cues, not attention — here, whether the
  model *reads* the feedback it was given.

## Title directions this suggests
- **Conjectures Without Refutations: Why LLM Agents Fail New Problems, and Which Teaching Methods Fix It**
- **LLMs Skip Pólya's Fourth Step**
- **Coach, Don't Cheer** (if task-level feedback wins and self-level loses, as Hattie & Timperley predict)
