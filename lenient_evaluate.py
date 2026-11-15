"""Run frozen/evaluate.py with an action parser that also accepts 'Action:' in mid-line.

The frozen parser takes an action only from a line that starts with 'Action:'. The Thought/Action textbook model
(adapters/q7_theory_act) writes the right Machine move on the same line as its thought ("... (Theorem 1.1).
Action: INPUT 0"), so every move is scored invalid. This wrapper changes only the parser, to tell a line-break
habit apart from missing skill.
"""
import os
import re
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import agents  # noqa: E402


def lenient_action_of(reply: str) -> str:
    hits = re.findall(r"(?i)\**action\**\s*:\s*(.+)$", reply, flags=re.MULTILINE)
    return hits[-1].strip() if hits else reply.strip()


agents.action_of = lenient_action_of
sys.argv[0] = "frozen/evaluate.py"
runpy.run_path("frozen/evaluate.py", run_name="__main__")
