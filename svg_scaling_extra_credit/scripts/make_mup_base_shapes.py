"""Create µP base shape file.

Use a base model and a delta model that differ in width. The resulting .bsh file is
loaded before µP training so mup can assign width-aware LR multipliers.

Example:
python scripts/make_mup_base_shapes.py --data_dir data/svg_bpe4096 --out mup_shapes.bsh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--out", type=Path, default=Path("mup_shapes.bsh"))
    p.add_argument("--base_width", type=int, default=128)
    p.add_argument("--delta_width", type=int, default=192)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head_base", type=int, default=4)
    p.add_argument("--n_head_delta", type=int, default=6)
    args = p.parse_args()

    import mup

    with open(args.config) as f:
        cfgs = yaml.safe_load(f)
    common = cfgs["common"]
    tok = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    vocab = tok.get_vocab_size()

    base_cfg = GPTConfig(vocab_size=vocab, block_size=common["block_size"], n_layer=args.n_layer,
                         n_head=args.n_head_base, n_embd=args.base_width, d_ff=4*args.base_width,
                         dropout=common["dropout"], bias=common.get("bias", False), use_mup=True)
    delta_cfg = GPTConfig(vocab_size=vocab, block_size=common["block_size"], n_layer=args.n_layer,
                          n_head=args.n_head_delta, n_embd=args.delta_width, d_ff=4*args.delta_width,
                          dropout=common["dropout"], bias=common.get("bias", False), use_mup=True)
    base = GPT(base_cfg)
    delta = GPT(delta_cfg)
    mup.make_base_shapes(base, delta, savefile=str(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
