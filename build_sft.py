"""Turn kept episodes into mlx-lm chat training data, one example per assistant turn.

Keep rule (the reward): the verifier passed. This is rejection-sampling RL / expert iteration: the model
imitates only its own successful play, reasoning included.

    python build_sft.py runs/collect_r1 [more dirs] --out data/r1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import agents
import envs


def keep(ep: dict) -> bool:
    return ep["success"]


def fade_coach(messages: list[dict]) -> list[dict]:
    """Remove the practice coach (H6): drop answers it refused (with the refusal) and its notes, so the
    model learns the corrected behaviour without ever seeing a coach."""
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if m["role"] == "assistant" and nxt and nxt["content"].startswith("Coach:") and "Answer not accepted" in nxt["content"]:
            i += 2
            continue
        if m["role"] == "user":
            m = {**m, "content": "\n".join(ln for ln in m["content"].split("\n") if not ln.startswith(("Coach:", "Teacher:")))}
        out.append(m)
        i += 1
    return out


def examples(ep: dict) -> list[dict]:
    """One example per assistant turn, trained on the plain system prompt (no playbook)."""
    msgs, out = [{"role": "system", "content": agents.SYSTEM["plain"]}], []
    for m in fade_coach(ep["messages"][1:]):
        if m["role"] == "assistant":
            out.append({"messages": msgs + [m]})
        msgs = msgs + [m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--clean", action="store_true", help="drop episodes with any contradiction, repeat, or invalid action")
    ap.add_argument("--balance", action="store_true", help="balance turn-level examples equally across training games")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eps = [json.loads(line) for d in a.dirs for line in (Path(d) / "episodes.jsonl").open()]
    for e in eps:
        if "seed" in e and e["seed"] >= 10_000:
            raise ValueError(f"Leakage guard: eval seed {e['seed']} >= 10000 found in SFT input!")
    kept = [e for e in eps if e["env"] in {c.name for c in envs.TRAIN_GAMES} and keep(e)]
    if a.clean:
        import diagnose
        kept = [e for e in kept if all(lbl == "ok" for lbl in diagnose.classify(e))]
    rng = random.Random(0)
    if a.balance and kept:
        by_game: dict[str, list[dict]] = {}
        for e in kept:
            by_game.setdefault(e["env"], []).extend(examples(e))
        target = max(len(v) for v in by_game.values())
        rows = []
        for g in sorted(by_game):
            pool = by_game[g]
            full_reps, rem = divmod(target, len(pool))
            rows.extend(pool * full_reps + rng.sample(pool, rem))
    else:
        rows = [x for e in kept for x in examples(e)]
    rng.shuffle(rows)
    n_valid = max(4, len(rows) // 20)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    by_env = {}
    for e in kept:
        by_env[e["env"]] = by_env.get(e["env"], 0) + 1
    stats = {"episodes": len(eps), "kept": len(kept), "kept_by_env": by_env, "examples": len(rows)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
