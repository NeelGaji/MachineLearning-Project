from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from src.model import GPT, GPTConfig, count_parameters


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def cosine_lr(it: int, max_steps: int, lr: float, warmup_steps: int, min_lr: float) -> float:
    if it < warmup_steps:
        return lr * (it + 1) / max(1, warmup_steps)
    if it >= max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(data, args.batch_size, args.block_size, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=("cuda" in device and args.dtype == "bf16")):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out
def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def make_optimizer(model, args):
    if args.use_mup:
        try:
            import mup
            return mup.MuAdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        except Exception as e:
            raise RuntimeError("µP run requested but mup optimizer could not be created. Install mup and check base shapes.") from e
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--model_name", type=str, default="tiny")
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--out_dir", type=Path, default=Path("runs/debug"))
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--use_mup", action="store_true")
    p.add_argument("--base_shapes", type=Path, default=None)
    p.add_argument("--fallback", action="store_true", help="Use fallback_models from config")


    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--max_steps", type=int, default=None, help="Override config max_steps; 0 means one epoch")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = load_yaml(args.config)
    common = cfgs["common"]
    group = "fallback_models" if args.fallback else "models"
    mcfg = cfgs[group][args.model_name]

    with open(args.data_dir / "stats.json") as f:
        stats = json.load(f)
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()

    user_max_steps = args.max_steps
    for k, v in common.items():
        if k != "max_steps":
            setattr(args, k, v)
    args.max_steps = int(common.get("max_steps", 0)) if user_max_steps is None else int(user_max_steps)
    args.block_size = int(common["block_size"])

    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")

    # One epoch means roughly train_tokens/(batch_size*block_size) optimizer steps sampled from the full stream.
    steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.block_size))
    max_steps = steps_per_epoch if args.max_steps == 0 else int(args.max_steps)
    warmup_steps = max(1, int(common.get("warmup_frac", 0.03) * max_steps))
    min_lr = args.learning_rate * float(common.get("min_lr_frac", 0.1))

    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=int(mcfg["n_layer"]),
        n_head=int(mcfg["n_head"]),
        n_embd=int(mcfg["n_embd"]),
        d_ff=int(mcfg["d_ff"]),
        dropout=float(common["dropout"]),
        bias=bool(common.get("bias", False)),
        use_mup=args.use_mup,
    )
    model = GPT(gpt_cfg).to(args.device)

    if args.use_mup:
        try:
            import mup
            if args.base_shapes is not None:
                mup.load_base_shapes(model, str(args.base_shapes))
            else:
                print("WARNING: µP without base_shapes. For final results, run scripts/make_mup_base_shapes.py and pass --base_shapes.")
        except Exception as e:
            raise RuntimeError("Failed to apply µP base shapes.") from e

    n_params = count_parameters(model)
    print(f"model={args.model_name} use_mup={args.use_mup} params={n_params:,} steps={max_steps} device={args.device}")

    if args.compile:
        model = torch.compile(model)

    optimizer = make_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 autocast does not require scaling

    if args.eval_only:
        losses = estimate_loss(model, train_data, val_data, args, args.device)
        print(losses)
        return

    metrics_path = args.out_dir / "metrics.jsonl"
    best_val = float("inf")
    start = time.time()
    tokens_seen = 0

    for it in range(max_steps):
        lr = cosine_lr(it, max_steps, args.learning_rate, warmup_steps, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        X, Y = get_batch(train_data, args.batch_size, args.block_size, args.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=("cuda" in args.device and args.dtype == "bf16")):
            _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += args.batch_size * args.block_size

        if it == 0 or (it + 1) % args.eval_interval == 0 or it == max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, args, args.device)
            elapsed = time.time() - start
            tok_s = tokens_seen / max(elapsed, 1e-9)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() and "cuda" in args.device else 0.0
            row = {
                "iter": it + 1,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "params": n_params,
                "tokens_seen": tokens_seen,
                "tok_per_sec": tok_s,
                "peak_mem_gb": peak_gb,
                "elapsed_sec": elapsed,
            }
            print(json.dumps(row))
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict() if not args.compile else model._orig_mod.state_dict(),
                    "config": gpt_cfg.__dict__,
                    "model_name": args.model_name,
                    "params": n_params,
                    "iter": it + 1,
                    "val_loss": best_val,
                }
                torch.save(ckpt, args.out_dir / "best.pt")

    summary = {
        "model_name": args.model_name,
        "use_mup": args.use_mup,
        "params": n_params,
        "best_val_loss": best_val,
        "final_iter": max_steps,
        "tokens_seen": tokens_seen,
        "epoch_equiv": tokens_seen / max(1, len(train_data)),
        "elapsed_sec": time.time() - start,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() and "cuda" in args.device else 0.0,
        "config": gpt_cfg.__dict__,
        "data_stats": stats,
        "learning_rate": args.learning_rate,
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
