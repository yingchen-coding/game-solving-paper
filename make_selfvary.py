"""Closing the loop, small: the model writes the varied wordings itself.

The strongest result here (C17) needs someone to write new surface wordings for HiddenRule: four renamed copies of
every example lift Machine, the renamed twin, from 6/20 (same data repeated) to 20/20. Those pools were written by
the experimenter. This asks Qwen, the model being trained, to propose them, keeps only proposals that pass a
mechanical check, and builds the same dataset with them, so the comparison with data/q7_distill_vary (written by
the experimenter) and data/q7_distill_dup (repeats) asks whether the step a teacher did can be done by the student.

The check is what a verifier can test without knowing the answer: every field has the right shape, no field reuses
the original wording, Machine's wording (machine, g, n, INPUT, SUBMIT) or the reserved ASK, the two verbs differ,
and the function letter differs from the variable. Rejected proposals are counted, not repaired.

    python make_selfvary.py data/q7_distill --k 4 --out data/q7_selfvary
"""
import argparse
import json
import random
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from make_vary import is_hiddenrule, rename

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
PROMPT = """A puzzle is written like this: "A hidden function f(x) is a polynomial in x ... Actions: QUERY x \
(see f(x)), ANSWER expr in x". I want the same puzzle in different words. Propose one new wording as JSON with
these keys:
  "noun": a short English name for the thing being probed (2 words at most, lower case), instead of "hidden function"
  "f": one letter used as the function name, instead of f
  "x": one lower-case letter used as the variable, instead of x
  "probe": one upper-case English verb for asking for a value, instead of QUERY
  "final": one upper-case English verb for giving the final formula, instead of ANSWER
Reply with the JSON object only."""
BANNED_WORDS = {"hidden function", "machine", "function"}
BANNED_LETTERS = {"f", "x", "g", "n", "a", "b", "c"}  # a, b, c name the coefficients in the worked solutions
BANNED_VERBS = {"QUERY", "ANSWER", "INPUT", "SUBMIT", "ASK"}


def check(w: dict) -> str | None:
    """Why a proposal is rejected, or None if it passes."""
    if set(w) != {"noun", "f", "x", "probe", "final"} or not all(isinstance(v, str) for v in w.values()):
        return "keys"
    if not re.fullmatch(r"[a-z]+( [a-z]+)?", w["noun"]) or w["noun"] in BANNED_WORDS:
        return "noun"
    if not re.fullmatch(r"[A-Za-z]", w["f"]) or w["f"].lower() in BANNED_LETTERS:
        return "f"
    if not re.fullmatch(r"[a-z]", w["x"]) or w["x"] in BANNED_LETTERS or w["x"] == w["f"].lower():
        return "x"
    for verb in ("probe", "final"):
        if not re.fullmatch(r"[A-Z]{3,10}", w[verb]) or w[verb] in BANNED_VERBS:
            return verb
    if w["probe"] == w["final"]:
        return "same verbs"
    return None


def propose(n: int, seed: int) -> tuple[list[dict], dict]:
    model, tok = load(MODEL)
    sampler = make_sampler(temp=1.0)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                     tokenize=False)
    kept, seen, rejected = [], set(), {}
    mx.random.seed(seed)
    for _ in range(4 * n):
        text = generate(model, tok, prompt=prompt, max_tokens=120, sampler=sampler)
        hit = re.search(r"\{.*\}", text, flags=re.S)
        try:
            w = json.loads(hit[0]) if hit else None
        except json.JSONDecodeError:
            w = None
        why = "json" if not isinstance(w, dict) else check(w)
        key = json.dumps(w, sort_keys=True) if why is None else None
        if why is None and key in seen:
            why = "duplicate"
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        seen.add(key)
        kept.append(w)
        if len(kept) == n:
            break
    return kept, rejected


def build(src: Path, wordings: list[dict], k: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in (src / "train.jsonl").open()]
    rng = random.Random(seed)
    out = list(rows)
    for row in filter(is_hiddenrule, rows):
        for _ in range(k):
            w = rng.choice(wordings)
            out.append({"messages": [{**m, "content": rename(m["content"], w)} for m in row["messages"]]})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--wordings", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "wordings.json"
    if cache.exists():
        record = json.loads(cache.read_text())
    else:
        kept, rejected = propose(args.wordings, args.seed)
        record = {"kept": kept, "rejected": rejected}
        cache.write_text(json.dumps(record, indent=1) + "\n")
    assert record["kept"], f"no proposal passed the check: {record['rejected']}"
    rows = build(args.src, record["kept"], args.k, args.seed)
    with (args.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (args.out / "valid.jsonl").write_text((args.src / "valid.jsonl").read_text())
    (args.out / "stats.json").write_text(json.dumps({"rows": len(rows), "k": args.k, "wordings": len(record["kept"]),
                                                     "rejected": record["rejected"]}) + "\n")


if __name__ == "__main__":
    main()
