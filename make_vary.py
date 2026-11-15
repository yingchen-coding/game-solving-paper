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
# OddCoin pools exclude the held-out twin's wording (twin_envs.Parcels: parcel, SCALE, '/', CLAIM, pan sinks, LEVEL).
ITEMS = ["token", "weight", "ball", "block", "marble"]
WEIGHS = ["BALANCE", "PLACE", "HOLD", "MATCH", "SET"]
SEPS = ["|", "vs", "against"]
TILTS = [("LEFT is heavier", "RIGHT is heavier", "BALANCED"), ("LEFT side drops", "RIGHT side drops", "EVEN"),
         ("LEFT goes down", "RIGHT goes down", "STEADY"), ("LEFT outweighs", "RIGHT outweighs", "EQUAL")]


def is_hiddenrule(row: dict) -> bool:
    return "A hidden function f(x)" in row["messages"][1]["content"]


def rename(text: str, w: dict) -> str:
    text = text.replace("hidden function", w["noun"])
    text = re.sub(r"(?<![A-Za-z])f(?=\()", w["f"], text)
    text = re.sub(r"(?<![A-Za-z])([abc]?)x(?![A-Za-z])", lambda m: m.group(1) + w["x"], text)
    text = text.replace("QUERY", w["probe"]).replace("Query", w["probe"].title())
    return text.replace("ANSWER", w["final"])


def is_oddcoin(row: dict) -> bool:
    return "coins numbered" in row["messages"][1]["content"]


def rename_coin(text: str, w: dict) -> str:
    text = re.sub(r"\bcoin(s?)\b", lambda m: w["item"] + m.group(1), text)
    text = re.sub(r"\bCoin(s?)\b", lambda m: w["item"].title() + m.group(1), text)
    text = text.replace("WEIGH", w["weigh"]).replace("Weigh", w["weigh"].title())
    text = re.sub(r"(?<=\d) \| (?=\d)", f" {w['sep']} ", text).replace("left | right", f"left {w['sep']} right")
    for old, new in zip(TILTS[0], w["tilt"]):
        text = text.replace(old, new)
    return text.replace("ANSWER", w["final"])


def build(src: Path, mode: str, k: int, seed: int, skills: tuple[str, ...] = ("hiddenrule",)) -> list[dict]:
    rows = [json.loads(line) for line in (src / "train.jsonl").open()]
    rng = random.Random(seed)
    out = list(rows)
    for row in filter(is_hiddenrule, rows) if "hiddenrule" in skills else ():
        for _ in range(k):
            if mode == "dup":
                out.append(row)
                continue
            w = {"noun": rng.choice(NOUNS), "f": rng.choice(FNAMES), "x": rng.choice(VARS),
                 "probe": rng.choice(PROBES), "final": rng.choice(FINALS)}
            out.append({"messages": [{**m, "content": rename(m["content"], w)} for m in row["messages"]]})
    for row in filter(is_oddcoin, rows) if "oddcoin" in skills else ():
        for _ in range(k):
            if mode == "dup":
                out.append(row)
                continue
            w = {"item": rng.choice(ITEMS), "weigh": rng.choice(WEIGHS), "sep": rng.choice(SEPS),
                 "tilt": rng.choice(TILTS[1:]), "final": rng.choice(FINALS)}
            out.append({"messages": [{**m, "content": rename_coin(m["content"], w)} for m in row["messages"]]})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--mode", choices=["vary", "dup"], required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skills", default="hiddenrule", help="comma list of hiddenrule, oddcoin")
    args = ap.parse_args()
    rows = build(args.src, args.mode, args.k, args.seed, tuple(args.skills.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (args.out / "valid.jsonl").write_text((args.src / "valid.jsonl").read_text())
    (args.out / "stats.json").write_text(json.dumps({"mode": args.mode, "k": args.k, "skills": args.skills, "rows": len(rows)}) + "\n")


if __name__ == "__main__":
    main()
