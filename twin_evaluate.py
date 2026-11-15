"""Run frozen/evaluate.py with the held-out twins in twin_envs.py registered; the parser and policy are unchanged."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_envs  # noqa: E402,F401

sys.argv[0] = "frozen/evaluate.py"
runpy.run_path("frozen/evaluate.py", run_name="__main__")
