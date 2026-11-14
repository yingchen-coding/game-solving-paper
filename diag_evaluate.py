"""Run frozen/evaluate.py with a larger reply budget, to tell truncated reasoning apart from missing skill.

The frozen evaluator caps every reply at 256 tokens. A model trained on long explanations can run out of tokens
before it writes its Action line, which the evaluator scores as an invalid move. This wrapper changes only that cap.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import agents  # noqa: E402

MAX_TOKENS = int(os.environ.get("DIAG_MAX_TOKENS", "512"))
agents.MLXPolicy.__init__.__defaults__ = (None, 0.0, MAX_TOKENS)
sys.argv[0] = "frozen/evaluate.py"
runpy.run_path("frozen/evaluate.py", run_name="__main__")
