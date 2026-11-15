"""Teach the theory, school-style: lessons (definitions, theorems, worked examples), then graded exercises.

Ying's question: does teaching the theory behind the games, the way compulsory schooling teaches a subject, make
problem solving transfer? The curriculum covers the mathematics each game rests on, written as a textbook would:
abstract notation, no game wording, no game actions. The games themselves never appear in training, so every
evaluation is an application problem the model has never seen.

    Chapter 1  Determining a polynomial from its values   (HiddenRule, Machine)
    Chapter 2  Halving an interval                        (GuessNumber; first-difference search for Circuit)
    Chapter 3  Eliminating hypotheses                     (CodeBreak, Lock)
    Chapter 4  Three-way splitting with a balance         (OddCoin)
    Chapter 5  Verifying a candidate against every fact   (all)

Every exercise answer is computed, and every worked solution is checked against it before it is written.

    python make_theory.py --out data/q7_theory          # training data (lessons + exercises)
    python make_theory.py --out data/q7_theory_act --act  # same, exercises answered as Thought/Action
    python make_theory.py --summary playbooks/theory.txt  # the theorems only, for the prompt-only condition
"""
import argparse
import itertools
import json
import math
import random
from fractions import Fraction
from pathlib import Path

LESSONS = {
    1: """Chapter 1. Determining a polynomial from its values.
Definition. A polynomial of degree at most 2 is p(x) = a*x^2 + b*x + c with numbers a, b, c.
Theorem 1.1. A polynomial of degree at most d is determined by its values at d + 1 different points. For degree at
most 2, three values are enough and fewer are not: two points fit infinitely many parabolas.
Method (points 0, 1, 2). c = p(0). a = (p(2) - 2*p(1) + p(0)) / 2. b = p(1) - p(0) - a.
Method (any three points x0, x1, x2). First differences d1 = (y1 - y0)/(x1 - x0), d2 = (y2 - y1)/(x2 - x1).
Then a = (d2 - d1)/(x2 - x0), b = d1 - a*(x0 + x1), c = y0 - a*x0^2 - b*x0.
Choosing points. Small points such as 0, 1, 2 keep the arithmetic easy. A fourth point is not needed to find the
formula, but it can be used to check it.
Always finish by substituting every known point into the formula (Chapter 5).""",
    2: """Chapter 2. Halving an interval.
Setting. An unknown integer lies in an interval [lo, hi]. A comparison with m says whether the unknown is below m,
equal to m, or above m.
Theorem 2.1. Comparing with the middle m = floor((lo + hi) / 2) leaves at most half of the interval. An interval of
N integers is therefore resolved in at most ceil(log2(N + 1)) comparisons, and no strategy can guarantee fewer.
Update rule. If the unknown is above m, the new interval is [m + 1, hi]; if below, [lo, m - 1].
Never ask about a number outside the current interval: the answer is already known and the question is wasted.
First-difference search. A computation runs in steps 1..n and one step is wrong; every value after a wrong step is
wrong. To find the wrong step, compare the value after the middle step with the value it should have. If it is
correct, the wrong step is later; if it is wrong, the wrong step is that one or earlier. This is halving again, and
needs about log2(n) comparisons instead of n.""",
    3: """Chapter 3. Eliminating hypotheses.
Setting. There is a finite set of possible answers (hypotheses). Each observation is a fact that some hypotheses
agree with and others contradict.
Theorem 3.1. After any number of observations, the answer is among the hypotheses consistent with all of them. A
hypothesis contradicted by even one observation can never be the answer.
Rules. (1) Never propose a hypothesis that an earlier observation already rules out. (2) Prefer a test whose
possible results split the remaining hypotheses into groups of similar size. (3) When one hypothesis remains, it is
the answer.
Example of a scoring rule. For a secret sequence of digits, a guess is scored by EXACT = positions where guess and
secret agree, and MISPLACED = digits that occur in both but in different positions (each digit counted at most as
often as it occurs in both, minus the exact matches). A hypothesis is consistent with a scored guess when scoring
the guess against that hypothesis gives the same EXACT and MISPLACED.""",
    4: """Chapter 4. Three-way splitting with a balance.
Setting. Among N coins exactly one is heavier. A balance compares two groups and has three results: left heavier,
right heavier, or equal.
Theorem 4.1. One weighing can leave at most one third of the candidates, so ceil(log3 N) weighings are needed and
enough. Put equal numbers of candidate coins on each side and keep the rest off the balance, with the three groups
as equal as possible.
Update rule. If the left side is heavier, the heavy coin is on the left; if the right, on the right; if equal, it is
among the coins not weighed. Coins already cleared should not be weighed again as if they were candidates.""",
    5: """Chapter 5. Verifying a candidate against every fact.
Theorem 5.1. A candidate answer is acceptable only if it agrees with every observation made so far. One
disagreement is enough to reject it.
Method. Write the candidate. For each observation, compute what the candidate predicts and compare it with what was
observed, writing "predicted = observed" or "MISMATCH". Commit to the candidate only when there is no MISMATCH.
A candidate that agrees with every observation can still be wrong when there are too few observations to determine
the answer (Theorem 1.1); then make another observation that separates the remaining candidates.""",
}

SUMMARY = """Theory you have studied (apply it to whatever the task calls things):
1. A polynomial of degree at most 2 is fixed by 3 values. With p(0), p(1), p(2): c = p(0), a = (p(2) - 2p(1) + p(0))/2, b = p(1) - p(0) - a.
2. To find an unknown in an interval, compare with the middle; each answer halves the interval. Never ask about a value already excluded.
3. To find the first wrong step in a chain where errors persist, check the middle step against what it should be, and halve.
4. Keep only hypotheses consistent with every observation; never propose one already ruled out; prefer tests that split the rest evenly.
5. A balance has three outcomes: split candidates into three equal groups; ceil(log3 N) weighings suffice.
6. Before committing, substitute the candidate into every observation and write predicted = observed or MISMATCH."""


def poly_str(a, b, c) -> str:
    terms = []
    for coef, mono in ((a, "x^2"), (b, "x"), (c, "")):
        if coef == 0:
            continue
        mag = abs(coef)
        body = mono if mono and mag == 1 else f"{mag}{'*' if mono else ''}{mono}"
        terms.append(("-" if coef < 0 else "+", body))
    if not terms:
        return "0"
    first = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    return " ".join([first] + [f"{s} {t}" for s, t in terms[1:]])


def ex_poly(rng):
    a, b, c = rng.randint(-5, 5), rng.randint(-9, 9), rng.randint(-20, 20)
    p = lambda x: a * x * x + b * x + c  # noqa: E731
    if rng.random() < 0.5:
        xs = [0, 1, 2]
    else:
        xs = sorted(rng.sample(range(-4, 6), 3))
    ys = [p(x) for x in xs]
    facts = ", ".join(f"p({x}) = {y}" for x, y in zip(xs, ys))
    q = f"p is a polynomial of degree at most 2 with integer coefficients. {facts}. Find p(x)."
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    d1, d2 = Fraction(y1 - y0, x1 - x0), Fraction(y2 - y1, x2 - x1)
    A = (d2 - d1) / (x2 - x0)
    B = d1 - A * (x0 + x1)
    C = y0 - A * x0 * x0 - B * x0
    assert (A, B, C) == (a, b, c)
    steps = ["Three values determine a polynomial of degree at most 2 (Theorem 1.1)."]
    if xs == [0, 1, 2]:
        steps += [f"c = p(0) = {c}.", f"a = (p(2) - 2*p(1) + p(0)) / 2 = ({y2} - {2 * y1} + {y0}) / 2 = {a}.",
                  f"b = p(1) - p(0) - a = {y1} - {y0} - {a} = {b}."]
    else:
        steps += [f"d1 = ({y1} - {y0})/({x1} - {x0}) = {d1}, d2 = ({y2} - {y1})/({x2} - {x1}) = {d2}.",
                  f"a = (d2 - d1)/({x2} - {x0}) = {A}.", f"b = d1 - a*({x0} + {x1}) = {B}.",
                  f"c = {y0} - a*{x0}^2 - b*{x0} = {C}."]
    steps.append("Check: " + "; ".join(f"p({x}) = {p(x)} = {y}" for x, y in zip(xs, ys)) + ".")
    return q, " ".join(steps) + f" Answer: p(x) = {poly_str(a, b, c)}.", f"p(x) = {poly_str(a, b, c)}"


def ex_points(rng):
    k = rng.randint(1, 2)
    xs = sorted(rng.sample(range(-3, 4), k))
    facts = ", ".join(f"p({x}) = {rng.randint(-9, 9)}" for x in xs)
    q = f"p has degree at most 2. You know only {facts}. Is p determined? If not, how many more values do you need?"
    need = 3 - k
    return q, (f"No. Theorem 1.1: degree at most 2 needs 3 values and only {k} {'is' if k == 1 else 'are'} known, so "
               f"many polynomials fit. {need} more value{'s' if need > 1 else ''} at new points will determine it."), f"{need} more"


def ex_halving(rng):
    lo, hi = 1, rng.choice([16, 50, 100, 200])
    secret = rng.randint(lo, hi)
    history, a, b = [], lo, hi
    for _ in range(rng.randint(1, 4)):
        m = (a + b) // 2
        if m == secret:
            break
        if secret > m:
            history.append(f"{m}: above")
            a = m + 1
        else:
            history.append(f"{m}: below")
            b = m - 1
    m = (a + b) // 2
    q = (f"An unknown integer is in [{lo}, {hi}]. Comparisons so far: {'; '.join(history) or 'none'}. "
         "What interval is left, and what should you compare with next?")
    ans = (f"Apply each result: {'; '.join(history) or 'nothing yet'}. The unknown is in [{a}, {b}] "
           f"({b - a + 1} integers). Compare with the middle, floor(({a} + {b})/2) = {m} (Theorem 2.1). "
           f"At most {math.ceil(math.log2(b - a + 2))} more comparisons are needed.")
    return q, ans, f"[{a}, {b}], compare with {m}"


def ex_count(rng):
    n = rng.choice([7, 10, 15, 31, 60, 100, 1000])
    return (f"How many comparisons guarantee finding an integer among {n} possibilities by halving?",
            f"ceil(log2({n} + 1)) = {math.ceil(math.log2(n + 1))} (Theorem 2.1).", str(math.ceil(math.log2(n + 1))))


def ex_firstdiff(rng):
    n = rng.randint(6, 16)
    bad = rng.randint(1, n)
    lo, hi, log, checks = 1, n, [], rng.randint(1, 3)
    while lo < hi and len(log) < checks:
        m = (lo + hi) // 2
        if m >= bad:
            log.append(f"after step {m}: wrong")
            hi = m
        else:
            log.append(f"after step {m}: correct")
            lo = m + 1
    nxt = (lo + hi) // 2
    q = (f"A computation has steps 1..{n}; exactly one step is wrong and every value after it is wrong. Checks so "
         f"far: {'; '.join(log)}. Where can the wrong step be, and which step should you check next?")
    if lo == hi:
        return q, f"The wrong step is in [{lo}, {hi}], so it is step {lo}; no more checks are needed.", f"step {lo}"
    return q, (f"A correct value after step m clears steps 1..m; a wrong value means the wrong step is m or earlier. "
               f"The wrong step is in [{lo}, {hi}]. Check the value after the middle step {nxt} (Chapter 2)."), f"check step {nxt}"


def score(guess, secret):
    exact = sum(g == s for g, s in zip(guess, secret))
    common = sum(min(guess.count(d), secret.count(d)) for d in set(guess))
    return exact, common - exact


def ex_elim(rng):
    k, digits = 2, rng.choice([3, 4])
    space = [list(c) for c in itertools.product(range(1, digits + 1), repeat=k)]
    secret = rng.choice(space)
    obs = []
    for _ in range(rng.randint(1, 2)):
        g = rng.choice(space)
        obs.append((g, score(g, secret)))
    left = [h for h in space if all(score(g, h) == s for g, s in obs)]
    assert secret in left
    fmt = lambda c: "".join(map(str, c))  # noqa: E731
    facts = "; ".join(f"guess {fmt(g)} scored EXACT={e}, MISPLACED={m}" for g, (e, m) in obs)
    q = (f"A secret has {k} digits from 1-{digits}, repeats allowed. {facts}. "
         "Which secrets are still possible? Name one guess you must not make.")
    ruled = next((h for h in space if h not in left), None)
    ans = (f"Keep only secrets that would give the same scores (Theorem 3.1). Possible: "
           f"{', '.join(fmt(h) for h in left)} ({len(left)} of {len(space)}).")
    if ruled:
        ans += f" Do not guess {fmt(ruled)}: an observation already rules it out."
    return q, ans, ", ".join(fmt(h) for h in left)


def ex_balance(rng):
    n = rng.choice([8, 9, 12, 13, 26, 27])
    heavy = rng.randint(1, n)
    cands = list(range(1, n + 1))
    t = math.ceil(len(cands) / 3)
    left, right = cands[:t], cands[t:2 * t]
    res = "left heavier" if heavy in left else "right heavier" if heavy in right else "balance"
    rest = left if heavy in left else right if heavy in right else cands[2 * t:]
    q = (f"Coins 1-{n}, exactly one heavier. You weigh coins {left[0]}-{left[-1]} against "
         f"{right[0]}-{right[-1]}: {res}. Which coins are still candidates, and how many weighings are needed "
         f"from the start in the worst case?")
    return q, (f"{res.capitalize()}: the heavy coin is among {rest[0]}-{rest[-1]} ({len(rest)} coins). "
               f"Worst case from the start: ceil(log3 {n}) = {math.ceil(math.log(n, 3) - 1e-9)} weighings (Theorem 4.1)."), \
        f"{rest[0]}-{rest[-1]}, {math.ceil(math.log(n, 3) - 1e-9)} weighings"


def ex_verify(rng):
    a, b, c = rng.randint(-4, 4), rng.randint(-6, 6), rng.randint(-9, 9)
    xs = sorted(rng.sample(range(-3, 5), 3))
    obs = [(x, a * x * x + b * x + c) for x in xs]
    cand = (a, b, c) if rng.random() < 0.5 else (a, b + rng.choice([-1, 1]), c)
    ca, cb, cc = cand
    lines = []
    for x, y in obs:
        pred = ca * x * x + cb * x + cc
        lines.append(f"x = {x}: predicted {pred} {'=' if pred == y else '!='} observed {y}" + ("" if pred == y else " MISMATCH"))
    ok = all("MISMATCH" not in ln for ln in lines)
    q = (f"Observations: {', '.join(f'p({x}) = {y}' for x, y in obs)}. Candidate: p(x) = {poly_str(*cand)}. "
         "Should you commit to it?")
    return q, ("Substitute into every observation (Theorem 5.1): " + "; ".join(lines) + ". "
               + ("No mismatch and three values determine it: commit." if ok else "There is a MISMATCH: reject it.")), \
        "commit" if ok else "reject"


EXERCISES = {1: [ex_poly, ex_points], 2: [ex_halving, ex_count, ex_firstdiff], 3: [ex_elim], 4: [ex_balance],
             5: [ex_verify]}
SYSTEM = "You are a careful student. Solve the exercise step by step, using the theory you have learned."
# Agent format, copied from frozen/agents.py SYSTEM["plain"]: the Q&A textbook (data/q7_theory) taught the model to drop
# its 'Action:' line, so every game move came out invalid; this variant answers each exercise in the format it is tested in.
# STATE because ANSWER is a game verb (HiddenRule, OddCoin, Circuit) and the games must stay unseen.
AGENT_SYSTEM = ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
                "exactly one action on a final line that starts with 'Action:'.")


def rows(per_chapter: int, lesson_reps: int, seed: int, act: bool = False) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for ch, text in LESSONS.items():
        for _ in range(lesson_reps):
            out.append({"messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": f"Teach me Chapter {ch}."},
                                     {"role": "assistant", "content": text}]})
        for i in range(per_chapter):
            q, a, final = EXERCISES[ch][i % len(EXERCISES[ch])](rng)
            if act:
                q, a = f"{q}\n\nActions: STATE your answer.", f"Thought:{a}\nAction: STATE {final}"
            out.append({"messages": [{"role": "system", "content": AGENT_SYSTEM if act else SYSTEM},
                                     {"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--per-chapter", type=int, default=100)
    ap.add_argument("--lesson-reps", type=int, default=10)
    ap.add_argument("--act", action="store_true", help="answer exercises in the agent's Thought/Action format")
    args = ap.parse_args()
    if args.summary:
        args.summary.write_text(SUMMARY + "\n")
    if args.out:
        data = rows(args.per_chapter, args.lesson_reps, seed=0, act=args.act)
        valid = rows(6, 1, seed=1, act=args.act)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
        (args.out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in valid))
        (args.out / "stats.json").write_text(json.dumps({"train": len(data), "valid": len(valid)}) + "\n")


if __name__ == "__main__":
    main()
