"""Fill the arithmetic gap left after the application layer (data/q7_theory_app).

With Chapter 6 the model names the chapter (20/20), collects three readings (19/20 on Machine) and reaches an answer
(20/20), but solves 4/20 Machine and 1/20 HiddenRule. On HiddenRule 13 final replies write the coefficient formula
with numbers; in 8 the step that goes wrong is a = (f(2) - 2*f(1) + f(0)) / 2, and every one of those has a double
negative, e.g. "(-15 - -12 + 3) / 2 = -6 / 2" (it is 0) or "2*f(1)" for f(1) = -1 written as -6. The textbook and the
application problems wrote exactly that form, "y2 - 2*y1 + y0" with raw signed numbers substituted. School teaches
signed arithmetic before it teaches formulas, so this adds it:

    Chapter 7    rules for signed numbers: subtracting a negative adds; add one term at a time
    drills       200 signed sums of the shape the formulas produce, each worked one term at a time
    clean form   every polynomial worked solution (textbook and application) now computes 2*p(1) on its own,
                 rewrites "- -" as "+", and shows each partial sum; the verification step is spelled out the same way

Everything else in data/q7_theory_app is kept, so the comparison with it isolates the arithmetic layer.

    python make_arith.py --out data/q7_theory_arith
"""
import argparse
import json
import random
import re
from pathlib import Path

import make_applied
import make_theory
from make_applied import CONTEXTS, FUNCS, VARS, poly_in
from make_theory import AGENT_SYSTEM, poly_str

CHAPTER7 = """Chapter 7. Signed numbers, one term at a time.
Rule 1. Subtracting a negative number adds its size: 5 - (-3) = 5 + 3 = 8. Never leave "- -" in a line; rewrite it as "+".
Rule 2. Adding a negative number subtracts its size: 5 + (-3) = 5 - 3 = 2.
Rule 3. Multiply before adding: in y2 - 2*y1 + y0, first compute 2*y1 on its own line, then substitute.
Rule 4. Add one term at a time and write every partial sum: -15 + 12 + 3 is -15 + 12 = -3, then -3 + 3 = 0.
Rule 5. Halve only at the end, and check that the total is even before halving an integer coefficient."""


def signed(t: int) -> str:
    return f"+ {t}" if t >= 0 else f"- {-t}"


def running_sum(terms: list[int]) -> tuple[str, int]:
    """'a + b - c; a + b = s1; s1 - c = s2', the form Chapter 7 teaches, and the total."""
    expr = f"{terms[0]} " + " ".join(signed(t) for t in terms[1:])
    acc, steps = terms[0], []
    for t in terms[1:]:
        steps.append(f"{acc} {signed(t)} = {acc + t}")
        acc += t
    return (expr + "; " + "; ".join(steps)) if steps else expr, acc


def coefficients(f: str, ys: list[int]) -> tuple[str, tuple[int, int, int]]:
    y0, y1, y2 = ys
    two = 2 * y1
    s_a, tot = running_sum([y2, -two, y0])
    a = tot // 2
    s_b, b = running_sum([y1, -y0, -a])
    text = (f"c = {f}(0) = {y0}. 2*{f}(1) = 2*({y1}) = {two}. {f}(2) - 2*{f}(1) + {f}(0) = {s_a}. "
            f"{tot} is even, a = {tot} / 2 = {a}. b = {f}(1) - {f}(0) - a = {s_b}.")
    assert (a, b) == ((y2 - 2 * y1 + y0) // 2, y1 - y0 - (y2 - 2 * y1 + y0) // 2)
    return text, (a, b, y0)


def verification(f: str, abc: tuple[int, int, int], ys: list[int]) -> str:
    a, b, c = abc
    parts = []
    for x, y in zip((0, 1, 2), ys):
        s, tot = running_sum([a * x * x, b * x, c])
        assert tot == y
        parts.append(f"{f}({x}) = {a * x * x} {signed(b * x)} {signed(c)}: {s.split('; ', 1)[-1]}, observed {y}")
    return "Verify (Chapter 5): " + "; ".join(parts) + ". No MISMATCH."


def app_poly_clean(rng):
    f, v, ctx = rng.choice(FUNCS), rng.choice(VARS), rng.choice(CONTEXTS)
    a, b, c = rng.randint(-5, 5), rng.randint(-9, 9), rng.randint(-20, 20)
    ys = [a * x * x + b * x + c for x in (0, 1, 2)]
    log = "\n".join(f"Reading {i + 1}: {f}({x}) = {y}" for i, (x, y) in enumerate(zip((0, 1, 2), ys)))
    q = (f"{ctx} values of {f}({v}), a polynomial in {v} of degree at most 2 with integer coefficients.\n{log}\n"
         f"What is {f}({v})?\n\nActions: STATE your answer.")
    work, abc = coefficients(f, ys)
    assert abc == (a, b, c)
    th = (f"Chapter 1, with {f} for p and {v} for x. Readings: {f}(0) = {ys[0]}, {f}(1) = {ys[1]}, {f}(2) = {ys[2]}. "
          f"{work} {verification(f, abc, ys)}")
    return q, f"Thought:{th}\nAction: STATE {poly_in(v, a, b, c)}"


_ex_poly = make_theory.ex_poly


def ex_poly_clean(rng):
    if rng.random() < 0.5:
        while True:  # three arbitrary points: divided differences, unchanged; the 0, 1, 2 case is rewritten below
            q, text, final = _ex_poly(rng)
            if "2*p(1)" not in text:
                return q, re.sub(r" - -(\d)", r" + \1", text), final  # Chapter 7, rule 1
    a, b, c = rng.randint(-5, 5), rng.randint(-9, 9), rng.randint(-20, 20)
    ys = [a * x * x + b * x + c for x in (0, 1, 2)]
    q = ("p is a polynomial of degree at most 2 with integer coefficients. "
         f"p(0) = {ys[0]}, p(1) = {ys[1]}, p(2) = {ys[2]}. Find p(x).")
    work, abc = coefficients("p", ys)
    text = f"Three values determine a polynomial of degree at most 2 (Theorem 1.1). {work} {verification('p', abc, ys)}"
    return q, text + f" Answer: p(x) = {poly_str(a, b, c)}.", f"p(x) = {poly_str(a, b, c)}"


def drill(rng):
    y2, y1, y0 = (rng.randint(-40, 40) for _ in range(3))
    if (y2 - 2 * y1 + y0) % 2:
        y0 += 1
    two = 2 * y1
    s, tot = running_sum([y2, -two, y0])
    q = (f"Compute step by step: ({y2}) - 2*({y1}) + ({y0}), then halve it.\n\nActions: STATE your answer.")
    th = f"Chapter 7. 2*({y1}) = {two}. ({y2}) - ({two}) + ({y0}) = {s}. {tot} is even, half is {tot // 2}."
    return q, f"Thought:{th}\nAction: STATE {tot // 2}"


def arith_rows(n: int, lesson_reps: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    lesson = {"messages": [{"role": "system", "content": AGENT_SYSTEM},
                           {"role": "user", "content": "Teach me Chapter 7.\n\nActions: STATE your answer."},
                           {"role": "assistant", "content": f"Thought:{CHAPTER7}\nAction: STATE Chapter 7 learned"}]}
    out = [lesson] * lesson_reps
    for _ in range(n):
        q, a = drill(rng)
        out.append({"messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": q},
                                 {"role": "assistant", "content": a}]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    make_theory.EXERCISES[1][0] = ex_poly_clean
    make_applied.APPLIED[:] = [app_poly_clean if fn is make_applied.app_poly else fn for fn in make_applied.APPLIED]
    train = (make_theory.rows(100, 10, seed=0, act=True) + make_applied.applied_rows(420, 10, seed=2)
             + arith_rows(200, 10, seed=5))
    valid = (make_theory.rows(6, 1, seed=1, act=True) + make_applied.applied_rows(24, 1, seed=3)
             + arith_rows(12, 1, seed=6))
    random.Random(4).shuffle(train)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (args.out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in valid))
    (args.out / "stats.json").write_text(json.dumps({"train": len(train), "valid": len(valid)}) + "\n")


if __name__ == "__main__":
    main()
