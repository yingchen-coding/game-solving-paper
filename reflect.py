"""Self-reflection learning: the model plays the TRAINING games, reflects on each episode, and distils
general problem-solving lessons into a playbook that goes into its own system prompt. No weights change.

Round r: play each training game with playbook_{r-1}; after each episode the model sees its transcript,
the verifier's outcome and the revealed hidden answer, and writes one general lesson; then it rewrites
the playbook (at most 8 rules, no game-specific facts). The playbook is then tested on held-out problems
with evaluate.py --playbook.

    python reflect.py --policy mlx:mlx-community/Qwen2.5-3B-Instruct-4bit --rounds 3 --per-game 3 --out runs/reflect_self
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import agents
import envs
from mlx_lm.sample_utils import make_sampler

LESSON_PROMPT = """You just tried to solve a problem. Here is what happened.

{transcript}

Outcome: {outcome}. {reveal}

Write ONE lesson, one or two sentences, about HOW to solve problems that give feedback. It must help on
DIFFERENT problems too (other puzzles, debugging code, diagnosing faults), so do not mention this game's
rules, numbers or names. Start with "Lesson:"."""

TAXONOMY_PROMPT = """You just tried to solve a problem. Here is what happened.

{transcript}

Outcome: {outcome}. {reveal}

First name the main reasoning error, choosing ONE of:
  A. repeated an action that had already failed or already given its information
  B. concluded before the evidence was sufficient to pin down the answer
  C. gave an answer that contradicts evidence already observed (or its own calculation)
  D. ignored or misread the feedback
  E. no error (solved efficiently)
Then write ONE rule that prevents that error on DIFFERENT problems too (other puzzles, debugging code,
diagnosing faults); do not mention this game's rules, numbers or names.
Format exactly:
Error: <letter>
Lesson: <rule>"""

CONTRAST_PROMPT = """You tried the same problem twice. One attempt SOLVED it, the other FAILED.

SOLVED attempt:
{good}

FAILED attempt:
{bad}

{reveal}
What did the solved attempt do differently that made it work? Write ONE lesson, one or two sentences, that
would help on DIFFERENT problems too (other puzzles, debugging code, diagnosing faults); do not mention this
game's rules, numbers or names. Start with "Lesson:"."""

MERGE_PROMPT = """Current playbook:
{playbook}

New lessons from recent problems:
{lessons}

Rewrite the playbook as at most 8 short numbered rules for solving problems from feedback. Keep what the
new lessons support, drop what they contradict, merge duplicates. General rules only: no game names, no
specific numbers. Output only the numbered rules."""


def compact(ep: dict) -> str:
    lines = []
    for m in ep["messages"][1:]:
        who = "You" if m["role"] == "assistant" else "Environment"
        lines.append(f"{who}: {m['content'][:400]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--reflector", help="model that writes lessons; default = the playing model itself")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--mode", default="free", choices=["free", "taxonomy", "noreveal", "contrastive"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    player = agents.make_policy(a.policy)
    reflector = agents.make_policy(a.reflector) if a.reflector else player
    playbook = ""
    seed = 0
    log = (out / "log.jsonl").open("a")
    greedy = player.sampler
    for r in range(1, a.rounds + 1):
        lessons, wins, total = [], 0, 0
        for cls in envs.TRAIN_GAMES:
            for _ in range(a.per_game):
                env = cls(seed)
                seed += 1
                ep = agents.run_episode(env, player, "plain", playbook or None)
                wins += ep["success"]
                total += 1
                reveal = "" if a.mode == "noreveal" else env.reveal()
                outcome = "SOLVED" if ep["success"] else "FAILED"
                error = None
                if a.mode == "contrastive":
                    # a second, sampled attempt at the same problem; a lesson only when exactly one succeeded
                    player.sampler = make_sampler(temp=0.8)
                    ep2 = agents.run_episode(cls(seed - 1), player, "plain", playbook or None)
                    player.sampler = greedy
                    if ep["success"] == ep2["success"]:
                        continue
                    good, bad = (ep, ep2) if ep["success"] else (ep2, ep)
                    msg = CONTRAST_PROMPT.format(good=compact(good), bad=compact(bad), reveal=reveal)
                elif a.mode == "taxonomy":
                    msg = TAXONOMY_PROMPT.format(transcript=compact(ep), outcome=outcome, reveal=reveal)
                else:
                    msg = LESSON_PROMPT.format(transcript=compact(ep), outcome=outcome, reveal=reveal)
                text = agents.complete(reflector, [{"role": "user", "content": msg}], 160)
                if a.mode == "taxonomy":
                    m = re.search(r"Error:\s*([A-E])", text)
                    error = m.group(1) if m else None
                lesson = text.split("Lesson:", 1)[-1].strip().split("\n")[0]
                lessons.append(lesson)
                log.write(json.dumps({"round": r, "env": env.name, "seed": seed - 1, "success": ep["success"],
                                      "error": error, "lesson": lesson}) + "\n")
                log.flush()
        merged = agents.complete(reflector, [{"role": "user", "content": MERGE_PROMPT.format(
            playbook=playbook or "(empty)", lessons="\n".join(f"- {x}" for x in lessons))}], 400)
        playbook = merged.strip()
        (out / f"playbook_r{r}.txt").write_text(playbook + "\n")
        print(f"round {r}: train success {wins}/{total}")
    (out / "playbook.txt").write_text(playbook + "\n")


if __name__ == "__main__":
    main()
