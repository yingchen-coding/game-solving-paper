"""Group-relative policy gradient (GRPO-style) on the practice games, with the verifier's outcome as the only reward.

DeepSeek-R1 trains reasoning with reinforcement learning from verifiable rewards (GRPO), and Chu et al. (2025, "SFT
Memorizes, RL Generalizes") find that outcome-reward RL carries to rule and surface variants where supervised
fine-tuning memorises. Every training condition so far in this repo is supervised; all of them stayed with the
wording they were trained on. This tests the other recipe on the same model: practise HiddenRule and OddCoin in
their original wording only, then evaluate on Machine and Parcels, their renamed twins, which are never played here.

Each iteration plays G episodes of one practice seed with sampling, scores each by success, and raises the
log-probability of the replies in better-than-average episodes and lowers it in worse ones (advantage = reward minus
the group mean, divided by the group std). Groups where every episode scores the same carry no signal and are
skipped. No KL term: the LoRA rank and learning rate are small and the run is short. Seeds 1000+ are used; evaluation
uses 10000+.

    python grpo.py --envs hiddenrule,oddcoin --iters 60 --group 4 --out adapters/q7_grpo
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import agents  # noqa: E402
import envs  # noqa: E402
from train_lora import SafeClippedAdamW  # noqa: E402


def policy_for(model, tok, temp: float, max_tokens: int):
    """The frozen MLXPolicy's generation loop, driven by the model being trained instead of a fresh load."""
    pol = agents.MLXPolicy.__new__(agents.MLXPolicy)
    pol.model, pol.tok, pol.sampler, pol.max_tokens = model, tok, make_sampler(temp=temp), max_tokens
    pol.use_cache, pol.prefix = True, "Thought:"
    pol._env, pol._ctx, pol._cache = None, [], None
    return pol


def tokens_and_mask(tok, messages):
    """Token ids of the whole episode and a 0/1 mask marking the model's own reply tokens."""
    def enc(msgs, gen=False):  # text then encode: some tokenizer versions return a BatchEncoding when tokenizing
        return tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False),
                          add_special_tokens=False)

    ids, mask = [], []
    for k, m in enumerate(messages):
        upto = enc(messages[:k + 1])
        if m["role"] == "assistant":
            start = len(enc(messages[:k], gen=True))
            assert upto[:len(ids)] == ids, "chat template is not prefix-stable"
            mask += [0] * (start - len(ids)) + [1] * (len(upto) - start)
        else:
            mask += [0] * (len(upto) - len(ids))
        ids = upto
    return ids, mask


def episode_loss(model, ids, mask, adv):
    x = mx.array(ids[:-1])[None]
    y = mx.array(ids[1:])[None]
    m = mx.array(mask[1:], dtype=mx.float32)[None]
    logp = -nn.losses.cross_entropy(model(x).astype(mx.float32), y)
    return -adv * (logp * m).sum() / mx.maximum(m.sum(), mx.array(1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--envs", default="hiddenrule,oddcoin")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.seed0 + a.iters < 10_000, "practice seeds must stay below the evaluation seeds"

    mx.random.seed(0)
    model, tok = load(a.model)
    model.freeze()
    lora = {"rank": a.rank, "dropout": 0.0, "scale": 20.0}
    linear_to_lora_layers(model, a.num_layers, lora, use_dora=False)
    a.out.mkdir(parents=True, exist_ok=True)
    save_config({"fine_tune_type": "lora", "num_layers": a.num_layers, "lora_parameters": lora, "model": a.model},
                a.out / "adapter_config.json")
    opt = SafeClippedAdamW(learning_rate=a.lr, max_grad_norm=1.0)
    grad_fn = nn.value_and_grad(model, episode_loss)
    names = a.envs.split(",")
    log = (a.out / "grpo_log.jsonl").open("a")
    played = (a.out / "episodes.jsonl").open("a")  # every sampled episode, for the rejection-sampling comparison

    for it in range(a.iters):
        t0 = time.time()
        name, seed = names[it % len(names)], a.seed0 + it
        pol = policy_for(model, tok, a.temp, a.max_tokens)
        eps = [agents.run_episode(envs.ALL[name](seed), pol) for _ in range(a.group)]
        r = [float(e["success"]) for e in eps]
        for k, e in enumerate(eps):
            played.write(json.dumps({**e, "seed": seed, "attempt": k, "iter": it}) + "\n")
        played.flush()
        mean = sum(r) / len(r)
        std = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
        row = {"iter": it, "env": name, "seed": seed, "rewards": r, "updated": std > 0}
        if std > 0:
            acc = None
            for e, x in zip(eps, r):
                ids, mask = tokens_and_mask(tok, e["messages"][:-1])  # the final observation is not the model's
                _, g = grad_fn(model, ids, mask, (x - mean) / std)
                acc = g if acc is None else tree_map(lambda p, q: p + q, acc, g)
            opt.update(model, tree_map(lambda p: p / len(eps), acc))
            mx.eval(model.parameters(), opt.state)
        row["secs"] = round(time.time() - t0, 1)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if (it + 1) % 10 == 0 or it + 1 == a.iters:
            mx.save_safetensors(str(a.out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))


if __name__ == "__main__":
    main()
