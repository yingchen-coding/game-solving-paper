"""Add the word-problem layer (应用题) between the textbook and the games.

The Thought/Action textbook model (adapters/q7_theory_act) chose the right inputs on HiddenRule in 20/20 episodes and
reached an answer in 17/20, but solved 2/20: it could not compute coefficients from values that arrived one per turn
under a new name (f instead of the textbook's p). Its OddCoin moves were half-written (one side empty) after the
textbook's "weigh 1-4 against 5-8" phrasing, and its contradiction counts did not fall. School puts application
problems between theorem and exam; this is that layer, still with no game names, verbs or wording:

    G1 notation      each problem renames the function and variable (h, q, r, u, s / t, k, m, z, w), never f/x/g/n
    G2 data layout   values come as a log of separate readings, the way observations arrive in an episode
    G3 sides         a balance plan names both pans and the coins left off, as explicit sets
    G4 consistency   elimination problems end with a next proposal checked against every observation (Chapter 5)
    G5 bridging      Chapter 6 states the renaming step itself: name the chapter, map the names, then apply it

    python make_applied.py --out data/q7_theory_app
"""
import argparse
import itertools
import json
import math
import random
from pathlib import Path

from make_theory import AGENT_SYSTEM, LESSONS, poly_str, rows, score

FUNCS, VARS = "hqrus", "tkmzw"
CONTEXTS = ["A sensor reports", "A sealed calculator returns", "A lab notebook records", "A tank gauge shows",
            "A price table lists"]
SYMBOLS = "PQRS"

CHAPTER6 = """Chapter 6. Applying the theory: the names change, the method does not.
A problem rarely uses the textbook's names. A function may be called h or q instead of p, its input t or k instead
of x, and the facts may arrive one at a time, as separate readings, instead of in one list.
Method. (1) Say which chapter the problem belongs to. (2) Map each new name onto the textbook name, for example
"h plays the role of p, t plays the role of x". (3) Collect every reading so far into one list. (4) Apply the
chapter's method step by step, writing every intermediate number. (5) Verify the result against every reading
(Chapter 5) before committing.
For a balance, always state both pans and the coins kept off, as explicit sets. For hypotheses, the next proposal
must agree with every observation so far; check it against each one before proposing it."""


def poly_in(var, a, b, c):
    return poly_str(a, b, c).replace("x", var)


def app_poly(rng):
    f, v, ctx = rng.choice(FUNCS), rng.choice(VARS), rng.choice(CONTEXTS)
    a, b, c = rng.randint(-5, 5), rng.randint(-9, 9), rng.randint(-20, 20)
    ys = [a * x * x + b * x + c for x in (0, 1, 2)]
    log = "\n".join(f"Reading {i + 1}: {f}({x}) = {y}" for i, (x, y) in enumerate(zip((0, 1, 2), ys)))
    q = (f"{ctx} values of {f}({v}), a polynomial in {v} of degree at most 2 with integer coefficients.\n{log}\n"
         f"What is {f}({v})?\n\nActions: STATE your answer.")
    y0, y1, y2 = ys
    th = (f"Chapter 1, with {f} for p and {v} for x. Readings: {f}(0) = {y0}, {f}(1) = {y1}, {f}(2) = {y2}. "
          f"c = {f}(0) = {c}. a = ({f}(2) - 2*{f}(1) + {f}(0)) / 2 = ({y2} - {2 * y1} + {y0}) / 2 = "
          f"{y2 - 2 * y1 + y0} / 2 = {a}. b = {f}(1) - {f}(0) - a = {y1} - {y0} - {a} = {b}. "
          f"Verify (Chapter 5): " + "; ".join(f"{f}({x}) = {a * x * x + b * x + c} = {y}"
                                                for x, y in zip((0, 1, 2), ys)) + ". No MISMATCH.")
    return q, f"Thought:{th}\nAction: STATE {poly_in(v, a, b, c)}"


def app_poly_next(rng):
    f, v = rng.choice(FUNCS), rng.choice(VARS)
    known = rng.randint(0, 2)
    ys = [rng.randint(-20, 20) for _ in range(known)]
    log = "\n".join(f"Reading {i + 1}: {f}({i}) = {y}" for i, y in enumerate(ys)) or "No readings yet."
    q = (f"You may request values of {f}({v}), a polynomial of degree at most 2, before giving its formula.\n{log}\n"
         f"What should you request next?\n\nActions: STATE your answer.")
    th = (f"Chapter 1, with {f} for p. Theorem 1.1 needs 3 values and {known} {'is' if known == 1 else 'are'} known, "
          f"so the formula is not determined yet. Request a new small point: {v} = {known}.")
    return q, f"Thought:{th}\nAction: STATE {f}({known})"


def app_balance(rng):
    n = rng.choice([6, 8, 9, 12, 13, 27])
    cands = list(range(1, n + 1))
    if rng.random() < 0.5:
        t = math.ceil(n / 3)
        res = rng.choice(["left heavier", "right heavier", "equal"])
        prior = f"Earlier: {{{', '.join(map(str, cands[:t]))}}} against {{{', '.join(map(str, cands[t:2 * t]))}}}: {res}.\n"
        cands = cands[:t] if res == "left heavier" else cands[t:2 * t] if res == "right heavier" else cands[2 * t:]
    else:
        prior = ""
    k = len(cands)
    side = max(1, k // 3 if k % 3 == 1 else math.ceil(k / 3))
    left, right, off = cands[:side], cands[side:2 * side], cands[2 * side:]
    fmt = lambda s: "{" + ", ".join(map(str, s)) + "}"  # noqa: E731
    q = (f"Coins 1-{n}; exactly one is heavier. {prior}Candidates now: {fmt(cands)}. Plan the next use of the "
         f"balance.\n\nActions: STATE your answer.")
    th = (f"Chapter 4. {k} candidate{'s' if k > 1 else ''}; split into three groups as equal as possible, at most "
          f"{side} on each pan. Left pan {fmt(left)}, right pan {fmt(right)}, kept off {fmt(off) if off else '{}'}. "
          f"Both pans hold {side} coin{'s' if side > 1 else ''}"
          + (", so equal weight means the heavy coin is kept off." if off else "."))
    if k == 1:
        th, act = f"Chapter 4. Only coin {cands[0]} remains: it is the heavy coin.", f"STATE coin {cands[0]}"
    else:
        act = f"STATE left {fmt(left)} right {fmt(right)}"
    return q, f"Thought:{th}\nAction: {act}"


def app_elim(rng):
    length, alphabet = rng.choice([(2, 3), (2, 4), (3, 3)])
    syms = SYMBOLS[:alphabet]
    space = ["".join(p) for p in itertools.product(syms, repeat=length)]
    secret = rng.choice(space)
    obs = []
    for _ in range(rng.randint(1, 3)):
        g = rng.choice(space)
        obs.append((g, score(g, secret)))
    left = [h for h in space if all(score(g, h) == s for g, s in obs)]
    pick = left[0]
    log = "\n".join(f"Attempt {i + 1}: {g} gave EXACT={e}, MISPLACED={m}" for i, (g, (e, m)) in enumerate(obs))
    q = (f"A hidden code has {length} symbols from {', '.join(syms)} (repeats allowed).\n{log}\n"
         f"Propose a next code that could still be the hidden one.\n\nActions: STATE your answer.")
    checks = "; ".join(f"{g}: predicted EXACT={score(g, pick)[0]}, MISPLACED={score(g, pick)[1]} = observed"
                       for g, _ in obs)
    th = (f"Chapter 3, scored by EXACT and MISPLACED. Keep only codes consistent with every attempt (Theorem 3.1): "
          f"{', '.join(left)} ({len(left)} of {len(space)}). Propose {pick}. Verify (Chapter 5): {checks}.")
    return q, f"Thought:{th}\nAction: STATE {pick}"


def app_firstdiff(rng):
    n, step = rng.randint(6, 12), rng.randint(2, 9)
    start, bad, checks = rng.randint(0, 20), rng.randint(1, n), rng.randint(1, 3)
    expected = [start + step * i for i in range(1, n + 1)]
    lo, hi, log = 1, n, []
    while lo < hi and len(log) < checks:
        m = (lo + hi) // 2
        seen = expected[m - 1] + (rng.randint(1, 5) if m >= bad else 0)
        log.append((m, seen))
        if m >= bad:
            hi = m
        else:
            lo = m + 1
    readings = "\n".join(f"Reading: value after station {m} = {s}" for m, s in log)
    q = (f"A line of {n} stations starts from {start}; each station should add {step}. Exactly one station is "
         f"faulty, and every value after it is wrong.\n{readings}\nWhat do you do next?\n\nActions: STATE your answer.")
    notes = "; ".join(f"station {m} should give {expected[m - 1]}, saw {s}: {'correct' if s == expected[m - 1] else 'MISMATCH'}"
                      for m, s in log)
    if lo == hi:
        th, act = f"Chapter 2, first-difference search. {notes}. Only station {lo} remains: it is faulty.", f"STATE station {lo}"
    else:
        nxt = (lo + hi) // 2
        th = (f"Chapter 2, first-difference search. {notes}. The faulty station is in [{lo}, {hi}]; read the value "
              f"after the middle station {nxt} next.")
        act = f"STATE read station {nxt}"
    return q, f"Thought:{th}\nAction: {act}"


APPLIED = [app_poly, app_poly, app_poly_next, app_balance, app_elim, app_firstdiff]


def applied_rows(n: int, lesson_reps: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = [{"messages": [{"role": "system", "content": AGENT_SYSTEM},
                         {"role": "user", "content": "Teach me Chapter 6.\n\nActions: STATE your answer."},
                         {"role": "assistant", "content": f"Thought:{CHAPTER6}\nAction: STATE Chapter 6 learned"}]}
           for _ in range(lesson_reps)]
    for i in range(n):
        q, a = APPLIED[i % len(APPLIED)](rng)
        out.append({"messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": q},
                                 {"role": "assistant", "content": a}]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=420)
    args = ap.parse_args()
    assert set(LESSONS) == {1, 2, 3, 4, 5}
    train = rows(100, 10, seed=0, act=True) + applied_rows(args.n, 10, seed=2)
    valid = rows(6, 1, seed=1, act=True) + applied_rows(24, 1, seed=3)
    random.Random(4).shuffle(train)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (args.out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in valid))
    (args.out / "stats.json").write_text(json.dumps({"train": len(train), "valid": len(valid)}) + "\n")


if __name__ == "__main__":
    main()
