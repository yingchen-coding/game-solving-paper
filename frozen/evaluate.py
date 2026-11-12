"""Play N held-out seeds of each environment with one policy and write episodes + a summary.

    python evaluate.py --policy mlx:mlx-community/Qwen2.5-1.5B-Instruct-4bit --out runs/base --n 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import agents
import envs

EVAL_SEED0 = 10_000


def repeats(ep: dict) -> int:
    seen, n = set(), 0
    for m in ep["messages"]:
        if m["role"] == "assistant":
            a = " ".join(agents.action_of(m["content"]).upper().split())
            n += a in seen and not a.startswith("RUN")
            seen.add(a)
    return n


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["env"] for r in rows}):
        rs = [r for r in rows if r["env"] == name]
        n = len(rs)
        out[name] = {
            "n": n,
            "success": sum(r["success"] for r in rs) / n,
            "asked": sum(r["asks"] > 0 for r in rs) / n,
            "invalid_per_step": sum(r["invalid"] for r in rs) / max(1, sum(r["steps"] for r in rs)),
            "steps": sum(r["steps"] for r in rs) / n,
            # Repeating an action already taken (same text) is the "keeps trying the same failed fix" pattern.
            "repeat_rate": sum(repeats(r) for r in rs) / max(1, sum(r["steps"] for r in rs)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--system", default="plain", choices=sorted(agents.SYSTEM))
    ap.add_argument("--envs", default=",".join(c.name for c in envs.TRAIN_GAMES + envs.HELD_OUT))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--teacher", choices=list(envs.TEACHERS),
                    help="teacher present at test time (in-the-moment effect)")
    ap.add_argument("--ledger", action="store_true", help="show all evidence so far every turn")
    ap.add_argument("--check-tool", action="store_true", help="give the agent a free CHECK verifier")
    ap.add_argument("--playbook", help="file with learned lessons, appended to the system prompt")
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0, help="first evaluation seed (replications use fresh seeds)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "episodes.jsonl"
    done = {(r["env"], r["seed"]) for r in map(json.loads, path.open())} if path.exists() else set()
    policy = agents.make_policy(a.policy, a.adapter)
    playbook = Path(a.playbook).read_text().strip() if a.playbook else None
    with path.open("a") as f:
        for name in a.envs.split(","):
            for i in range(a.n):
                seed = a.seed0 + i
                if (name, seed) in done:
                    continue
                env = envs.ALL[name](seed)
                env.teacher = a.teacher
                env.ledger, env.allow_check = a.ledger, a.check_tool
                ep = agents.run_episode(env, policy, a.system, playbook)
                ep["seed"] = seed
                f.write(json.dumps(ep) + "\n")
                f.flush()
    rows = [json.loads(line) for line in path.open()]
    summary = {"policy": a.policy, "adapter": a.adapter, "system": a.system, "playbook": a.playbook, "teacher": a.teacher, "ledger": a.ledger, "check_tool": a.check_tool, "by_env": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_env"], indent=1))


if __name__ == "__main__":
    main()
