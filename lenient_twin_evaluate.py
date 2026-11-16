"""Twin evaluation with a parser that forgives where the action sits and repeated 'Action' labels.

The randomised-wording GRPO adapter (adapters/q7_grpo_rand) fails mostly on format, not moves: 'Action:' mid-line
after the thought, or the label doubled ('Action: ACTION QUERY 0', 'Action: Action: INPUT 0'). The frozen parser
scores all of those invalid. This changes only the parser, to tell format drift apart from lost skill.
"""
import os
import re
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import agents  # noqa: E402
import twin_envs  # noqa: E402,F401

LABEL = re.compile(r"^(?:[*`> ]*(?:thought\s*:\s*)?action\b[*`]*\s*:?\s*)+", re.I)


def lenient_action_of(reply: str) -> str:
    hits = re.findall(r"(?i)\**action\**\s*:\s*(.+)$", reply, flags=re.MULTILINE)
    return LABEL.sub("", hits[-1].strip()) if hits else reply.strip()


agents.action_of = lenient_action_of
sys.argv[0] = "frozen/evaluate.py"
runpy.run_path("frozen/evaluate.py", run_name="__main__")
