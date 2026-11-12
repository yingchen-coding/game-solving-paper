"""How the model uses the free CHECK tool: how often, whether the call is well formed, and whether it obeys the verdict.

Usage: check_use.py RUN [RUN ...]   (run dirs under runs/, evaluated with --check-tool)
Prints, per task: CHECK calls / actions, malformed calls, negative verdicts, and how often the model then commits
the very candidate the check rejected.
"""
import json
import sys
from collections import Counter
from pathlib import Path

MALFORMED = ("Check: give", "could not evaluate")
NEGATIVE = ("inconsistent", "fails", "disagree", "does not", "not consistent")


def action(content: str) -> str:
    lines = [x for x in content.splitlines() if x.startswith("Action:")]
    return lines[-1][len("Action:"):].strip() if lines else ""


def tally(run: str) -> dict:
    out: dict[str, Counter] = {}
    for line in (Path("runs") / run / "episodes.jsonl").open():
        ep = json.loads(line)
        c = out.setdefault(ep["env"], Counter())
        msgs = ep["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            act = action(m["content"])
            c["actions"] += 1
            if not act.upper().startswith("CHECK"):
                continue
            c["checks"] += 1
            verdict = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
            if verdict.startswith(MALFORMED) or any(s in verdict for s in MALFORMED):
                c["malformed"] += 1
                continue
            if not any(s in verdict for s in NEGATIVE):
                continue
            c["negative"] += 1
            nxt = next((action(x["content"]) for x in msgs[i + 2:]
                        if x["role"] == "assistant" and not action(x["content"]).upper().startswith("CHECK")), None)
            cand = act.split(None, 1)[1].replace(" ", "") if " " in act else ""
            if nxt is not None and cand and cand in nxt.replace(" ", ""):
                c["commits_rejected"] += 1
    return out


def main():
    print("| run | task | CHECK / actions | malformed | negative verdicts | then commits the rejected candidate |")
    print("|---|---|---|---|---|---|")
    for run in sys.argv[1:]:
        for env, c in tally(run).items():
            if c["checks"]:
                print(f"| {run} | {env} | {c['checks']}/{c['actions']} | {c['malformed']} | {c['negative']} | "
                      f"{c['commits_rejected']} |")


if __name__ == "__main__":
    main()
