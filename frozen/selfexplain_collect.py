"""Self-explanation data (Chi et al. 1989): the model explains, in its own words, why the expert's next move is right.

For each worked-example training row the model sees the transcript so far and the expert's action, and writes the
Thought itself. The training example keeps the model's Thought and the expert's Action. Contrast with imitating the
expert's prose (q7_procdiverse): same states, same actions, only the authorship of the reasoning differs.

    python selfexplain_collect.py data/q7_procdiverse --out data/q7_selfexplain

Resumable: <split>.done records how many source rows were processed, so a killed run continues where it stopped.
The file name ends in collect.py so every experimenter's one-model-at-a-time lock sees it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ASK = ("The correct next action is:\nAction: {act}\nIn your own words, explain from the evidence above why this action "
       "is the right one. If it commits to an answer, check that answer against every observation, writing out the "
       "arithmetic. Reply with your Thought only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    a = ap.parse_args()
    model, tok = load(a.model)
    sampler = make_sampler(temp=0.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "kept": 0}
    for split in ("valid", "train"):
        done_file = out / f"{split}.done"
        done = int(done_file.read_text()) if done_file.exists() else 0
        with (out / f"{split}.jsonl").open("a") as dst:
            for i, line in enumerate((Path(a.src) / f"{split}.jsonl").open()):
                if i < done:
                    continue
                row = json.loads(line)
                ctx, expert = row["messages"][:-1], row["messages"][-1]["content"]
                act = expert.rsplit("Action:", 1)[1].strip()
                prompt = tok.apply_chat_template(ctx + [{"role": "user", "content": ASK.format(act=act)}],
                                                 add_generation_prompt=True, tokenize=False) + "Thought:"
                text = generate(model, tok, prompt=prompt, max_tokens=220, sampler=sampler)
                thought = re.split(r"(?im)^\s*\**action\**\s*:", text)[0].strip()
                if len(thought) >= 20:
                    dst.write(json.dumps({"messages": ctx + [{"role": "assistant",
                                                              "content": f"Thought: {thought}\nAction: {act}"}]}) + "\n")
                    dst.flush()
                done_file.write_text(str(i + 1))
    for split in ("valid", "train"):
        src_rows = sum(1 for _ in (Path(a.src) / f"{split}.jsonl").open())
        kept = sum(1 for _ in (out / f"{split}.jsonl").open())
        stats["rows"] += src_rows
        stats["kept"] += kept
    (out / "stats.json").write_text(json.dumps(stats))
    print(stats)


if __name__ == "__main__":
    main()
