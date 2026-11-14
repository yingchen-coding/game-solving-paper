"""授人以渔 test: does training on one skill under many wordings let it survive a wording never seen?

Takes the distillation data and adds K copies of every HiddenRule example. In `vary` mode each copy renames the
task's surface (function name, variable, verbs, the noun for the box) from pools that exclude both the original
wording and Machine's (g, n, INPUT, SUBMIT, machine), so Machine stays a held-out wording. In `dup` mode the copies
are verbatim, giving a control with the same amount of HiddenRule data.

    python make_vary.py data/q7_distill --mode vary --k 4 --out data/q7_distill_vary
"""
import argparse
import json
import random
import re
from pathlib import Path

NOUNS = ["secret rule", "black box", "oracle", "device", "sealed formula"]
FNAMES = ["h", "p", "r", "q", "F"]
VARS = ["t", "k", "m", "z", "w"]
PROBES = ["PROBE", "TEST", "ASK", "SAMPLE", "MEASURE"]
FINALS = ["REPORT", "DECLARE", "GIVE", "STATE", "COMMIT"]


def is_hiddenrule(row: dict) -> bool:
    return "A hidden function f(x)" in row["messages"][1]["content"]


def rename(text: str, w: dict) -> str:
    text = text.replace("hidden function", w["noun"])
    text = re.sub(r"(?<![A-Za-z])f(?=\()", w["f"], text)
    text = re.sub(r"(?<![A-Za-z])([abc]?)x(?![A-Za-z])", lambda m: m.group(1) + w["x"], text)
    text = text.replace("QUERY", w["probe"]).replace("Query", w["probe"].title())
    return text.replace("ANSWER", w["final"])


def build(src: Path, mode: str, k: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in (src / "train.jsonl").open()]
    rng = random.Random(seed)
    out = list(rows)
    for row in filter(is_hiddenrule, rows):
        for _ in range(k):
            if mode == "dup":
                out.append(row)
                continue
            w = {"noun": rng.choice(NOUNS), "f": rng.choice(FNAMES), "x": rng.choice(VARS),
                 "probe": rng.choice(PROBES), "final": rng.choice(FINALS)}
            out.append({"messages": [{**m, "content": rename(m["content"], w)} for m in row["messages"]]})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--mode", choices=["vary", "dup"], required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = build(args.src, args.mode, args.k, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (args.out / "valid.jsonl").write_text((args.src / "valid.jsonl").read_text())
    (args.out / "stats.json").write_text(json.dumps({"mode": args.mode, "k": args.k, "rows": len(rows)}) + "\n")


if __name__ == "__main__":
    main()
