"""Dream: sleep on a failed problem, then learn from the dreams that work.

For every FAILED practice episode: the model reflects on it (transcript + outcome + revealed answer) and
writes one lesson; then it replays the SAME problem from the start against the real environment, with that
lesson in its system prompt (sampled, up to --tries times). Replays that the verifier passes are saved as
episodes. build_sft.py rebuilds every example with the plain system prompt, so the lesson is faded out: the
model is trained to act as its dream-self did, without being told why.

Unlike STaR-style rationalisation, the answer never enters the actions directly: the replay interacts with
the real environment and must earn every observation.

    python dream.py --policy mlx:... runs/q7_collect_r1 --out runs/q7_dream
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm.sample_utils import make_sampler

import agents
import envs
import reflect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "episodes.jsonl"
    done = {(r["env"], r["seed"]) for r in map(json.loads, path.open())} if path.exists() else set()
    log = (out / "dreams.jsonl").open("a")
    player = agents.make_policy(a.policy)
    greedy, sampled = player.sampler, make_sampler(temp=0.8)
    failed = {}
    for d in a.dirs:
        for line in (Path(d) / "episodes.jsonl").open():
            ep = json.loads(line)
            if not ep["success"] and ep["env"] in {c.name for c in envs.TRAIN_GAMES}:
                failed.setdefault((ep["env"], ep["seed"]), ep)
    with path.open("a") as f:
        for (name, seed), ep in sorted(failed.items()):
            if (name, seed) in done:
                continue
            env = envs.ALL[name](seed)
            player.sampler = greedy
            msg = reflect.LESSON_PROMPT.format(transcript=reflect.compact(ep), outcome="FAILED", reveal=env.reveal())
            lesson = agents.complete(player, [{"role": "user", "content": msg}], 120)
            lesson = lesson.split("Lesson:", 1)[-1].strip().split("\n")[0]
            player.sampler = sampled
            solved = False
            for t in range(a.tries):
                dream = agents.run_episode(envs.ALL[name](seed), player, "plain", lesson)
                if dream["success"]:
                    dream.update(seed=seed, attempt=t, lesson=lesson)
                    f.write(json.dumps(dream) + "\n")
                    f.flush()
                    solved = True
                    break
            log.write(json.dumps({"env": name, "seed": seed, "lesson": lesson, "dream_solved": solved}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
