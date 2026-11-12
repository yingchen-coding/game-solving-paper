"""Numerically stable fp16 LoRA trainer for 4-bit Qwen2.5 on Apple Silicon (MLX).

Keeps exact fp16 4-bit dequantization while preventing NaN weight corruption via:
1. float32 cross-entropy loss with non-zero mask denominator guard.
2. NaN/Inf gradient skip + global L2 norm clipping (max_grad_norm=1.0) BEFORE touching AdamW state or model weights.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, evaluate, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config


class SafeClippedAdamW(optim.AdamW):
    def __init__(self, learning_rate: float, max_grad_norm: float = 1.0, weight_decay: float = 0.0):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm

    def apply_gradients(self, gradients, parameters):
        clean = tree_map(lambda g: mx.where(mx.isnan(g) | mx.isinf(g), mx.zeros_like(g), g), gradients)
        clipped, _ = optim.clip_grad_norm(clean, max_norm=self.max_grad_norm)
        return super().apply_gradients(clipped, parameters)


def safe_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs).astype(mx.float32)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mx.maximum(mask.sum(), mx.array(1))
    return ce.sum() / ntoks, ntoks


def load_filtered_chat(path: Path, tok, max_seq_length: int):
    raw = [json.loads(line) for line in path.open()]
    ds = ChatDataset(raw, tok, mask_prompt=True)
    kept = []
    for item in raw:
        toks, off = ds.process(item)
        if off < min(len(toks), max_seq_length) - 8:
            kept.append(item)
    return CacheDataset(ChatDataset(kept, tok, mask_prompt=True)), len(raw) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--iters", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=45)
    ap.add_argument("--steps-per-eval", type=int, default=45)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    model, tok = load(a.model)
    model.freeze()

    lora_params = {"rank": a.rank, "dropout": 0.0, "scale": a.scale}
    linear_to_lora_layers(model, a.num_layers, lora_params, use_dora=False)

    data_dir = Path(a.data)
    train_ds, drop_tr = load_filtered_chat(data_dir / "train.jsonl", tok, a.max_seq_length)
    val_ds, drop_va = load_filtered_chat(data_dir / "valid.jsonl", tok, a.max_seq_length)
    print(f"Loaded train={len(train_ds)} (dropped {drop_tr}), valid={len(val_ds)} (dropped {drop_va})")

    out_dir = Path(a.adapter_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {"fine_tune_type": "lora", "num_layers": a.num_layers, "lora_parameters": lora_params, "model": a.model},
        out_dir / "adapter_config.json",
    )

    targs = TrainingArgs(
        batch_size=a.batch_size,
        iters=a.iters,
        val_batches=a.val_batches,
        steps_per_report=10,
        steps_per_eval=a.steps_per_eval,
        steps_per_save=a.save_every,
        adapter_file=str(out_dir / "adapters.safetensors"),
        max_seq_length=a.max_seq_length,
        grad_checkpoint=True,
        grad_accumulation_steps=a.grad_accum,
        clear_cache_threshold=512 * 1024 * 1024,
    )
    opt = SafeClippedAdamW(learning_rate=a.lr, max_grad_norm=1.0)

    class BestValCallback(TrainingCallback):
        def __init__(self):
            self.best_val = float("inf")
            self.best_iter = -1

        def on_val_loss_report(self, val_info: dict):
            vl = float(val_info["val_loss"])
            it = int(val_info["iteration"])
            if math.isfinite(vl) and vl < self.best_val and it > 0:
                self.best_val = vl
                self.best_iter = it
                weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(str(out_dir / "best_adapters.safetensors"), weights)

    cb = BestValCallback()
    train(model=model, optimizer=opt, train_dataset=train_ds, val_dataset=val_ds, args=targs, loss=safe_loss, training_callback=cb)
    final_val = float(evaluate(model=model, dataset=val_ds, loss=safe_loss, batch_size=a.batch_size,
                               num_batches=a.val_batches, max_seq_length=a.max_seq_length, iterate_batches=iterate_batches))
    print(f"Final step {a.iters} val_loss={final_val:.4f} (best earlier: {cb.best_val:.4f} at iter {cb.best_iter})")
    if math.isfinite(final_val) and final_val <= cb.best_val:
        print(f"Keeping final step {a.iters} adapter (val_loss={final_val:.4f}).")
    elif (out_dir / "best_adapters.safetensors").exists():
        shutil.copy(out_dir / "best_adapters.safetensors", out_dir / "adapters.safetensors")
        print(f"Restored best adapter from iter {cb.best_iter} (val_loss={cb.best_val:.4f}).")
    print("Training complete.")


if __name__ == "__main__":
    main()
