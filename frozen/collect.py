"""Collect training trajectories on the TRAINING games only (never on the held-out task).

Two sources:
    --policy oracle            scripted expert play (the SFT-on-expert control)
    --policy mlx:<repo> --temp sampled self-play from the model being trained (expert iteration)

    python collect.py --policy mlx:... --adapter adapters/r1 --temp 0.8 --seeds 0:60 --k 2 --out runs/collect_r2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import agents
import envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seeds", default="0:50")
    ap.add_argument("--k", type=int, default=1, help="attempts per seed")
    ap.add_argument("--system", default="plain")
    ap.add_argument("--coach", action="store_true", help="practice with the evidence coach on (H6)")
    ap.add_argument("--teacher", choices=list(envs.TEACHERS))
    ap.add_argument("--check-tool", action="store_true", help="give the agent the free CHECK verifier")
    ap.add_argument("--games", default=",".join(c.name for c in envs.TRAIN_GAMES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    games = [envs.ALL[g] for g in a.games.split(",")]
    leaked = [c.name for c in games if c in envs.HELD_OUT + envs.NEAR]
    if leaked:
        raise SystemExit(f"refusing to collect training data on test problems: {leaked}")
    lo, hi = map(int, a.seeds.split(":"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "episodes.jsonl"
    done = {(r["env"], r["seed"], r["attempt"]) for r in map(json.loads, path.open())} if path.exists() else set()
    policy = agents.make_policy(a.policy, a.adapter, a.temp)
    with path.open("a") as f:
        for seed in range(lo, hi):
            for cls in games:
                for k in range(a.k):
                    if (cls.name, seed, k) in done:
                        continue
                    env = cls(seed)
                    env.coach = a.coach
                    env.teacher = a.teacher
                    env.allow_check = a.check_tool
                    ep = agents.run_episode(env, policy, a.system)
                    ep["coach_notes"] = env.rec.coach_notes
                    ep.update(seed=seed, attempt=k)
                    f.write(json.dumps(ep) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
