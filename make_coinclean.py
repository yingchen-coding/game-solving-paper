"""Replace the distilled OddCoin examples with clean worked episodes that write the candidate set after every reading.

On the Parcels replication seeds (10020-10059, 19 distinct puzzles) every trained model still loses some games.
Replaying each failure through the environment: the losing claims name a parcel the readings had already ruled out
(distillation 6 of 6 failed episodes, dup2 10 of 13, vary2 7 of 13), usually after an unequal weighing such as
"SCALE 1 2 / 3", whose result the next thought then misreads. The distilled data teaches that move: 50 of its 364
OddCoin weighings (14%) put unequal counts on the pans, and dup2/vary2 repeat each such row five times.

This keeps every non-OddCoin row of the source and swaps each OddCoin row for an episode played by a scripted
expert on a fresh practice seed. The expert sees only the transcript. Every thought (1) restates the reading,
(2) says which coins that rules out, (3) lists the coins still possible, then (4) splits them into equal pans with
the rest aside. The wording is the original OddCoin wording, so a comparison with dup2 (same row count, same
HiddenRule rows) isolates what the OddCoin examples teach.

    python make_coinclean.py data/q7_distill_dup2 --out data/q7_coinclean
"""
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import agents  # noqa: E402
import envs  # noqa: E402
from make_vary import is_oddcoin  # noqa: E402


def span(coins: list[int]) -> str:
    return " ".join(map(str, coins))


def expert(messages, env) -> str:
    """Ternary search with equal pans; the thought carries the bookkeeping the failures skipped."""
    n = env.n
    cands = list(range(1, n + 1))
    notes = []
    for m in messages:
        if m["role"] != "user":
            continue
        hit = re.match(r"([\d ]+) vs ([\d ]+): (LEFT is heavier|RIGHT is heavier|BALANCED)", m["content"])
        if not hit:
            continue
        left, right = [int(c) for c in hit[1].split()], [int(c) for c in hit[2].split()]
        if hit[3] == "BALANCED":
            out = left + right
            cands = [c for c in cands if c not in out]
            notes = [f"Reading: {span(left)} vs {span(right)} BALANCED, so every coin on the balance is normal "
                     f"and {span(out)} are ruled out."]
        else:
            heavy = left if hit[3].startswith("LEFT") else right
            cands = [c for c in cands if c in heavy]
            notes = [f"Reading: {hit[3]}, so the heavy coin is on the {hit[3].split()[0].lower()} pan "
                     f"({span(heavy)}); every coin not on that pan is ruled out."]
    left_actions = env.max_steps - env.rec.steps
    notes.append(f"Still possible: {span(cands)} ({len(cands)} coin{'s' if len(cands) > 1 else ''}, "
                 f"{left_actions} action{'s' if left_actions > 1 else ''} left).")
    if len(cands) == 1:
        notes.append(f"Only coin {cands[0]} is left, so it is the heavy coin.")
        return "Thought:" + " ".join(notes) + f"\n\nAction: ANSWER {cands[0]}"
    k = len(cands)
    side = max(1, k // 3 if k % 3 == 1 else math.ceil(k / 3))
    left, right, aside = cands[:side], cands[side:2 * side], cands[2 * side:]
    notes.append(f"Put {side} on each pan, equal counts so a tilt points at the heavy coin"
                 + (f", and keep {span(aside)} aside." if aside else "."))
    return "Thought:" + " ".join(notes) + f"\n\nAction: WEIGH {span(left)} | {span(right)}"


def build(src: Path, seed0: int) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in (src / "train.jsonl").open()]
    out, seed, swapped = [], seed0, 0
    for row in rows:
        if not is_oddcoin(row):
            out.append(row)
            continue
        ep = agents.run_episode(envs.OddCoin(seed), expert)
        assert ep["success"] and ep["invalid"] == 0, f"expert failed on seed {seed}"
        out.append({"messages": ep["messages"][:-1]})  # the closing observation is not trained on
        seed, swapped = seed + 1, swapped + 1
    assert seed < 10_000, "practice seeds must stay below the evaluation seeds"
    random.Random(0).shuffle(out)
    return out, swapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, swapped = build(args.src, args.seed0)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (args.out / "valid.jsonl").write_text((args.src / "valid.jsonl").read_text())
    (args.out / "stats.json").write_text(json.dumps({"src": str(args.src), "rows": len(rows),
                                                     "oddcoin_swapped": swapped}) + "\n")


if __name__ == "__main__":
    main()
