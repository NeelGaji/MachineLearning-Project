from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.model import GPT, GPTConfig

DEFAULT_PREFIX = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
PREFIXES = {
    "unconditional": DEFAULT_PREFIX,
    "face": DEFAULT_PREFIX + '<circle cx="100" cy="100" r="70" fill="#ffd66b"/><circle cx="75" cy="85" r="8" fill="#000"/>',
    "open_path": DEFAULT_PREFIX + '<path d="M50 50 L150 50 L150 150" fill="none" stroke="#111" stroke-width="10"/>',
    "one_shape_group": DEFAULT_PREFIX + '<g transform="translate(20 20)"><rect x="30" y="30" width="80" height="80" rx="12" fill="#4a90e2"/>',
    "star_start": DEFAULT_PREFIX + '<path d="M100 25 L120 75 L175 75" fill="#f5a623"/>',
}


def trim_to_svg(text: str) -> str:
    low = text.lower()
    end = low.find("</svg>")
    if end != -1:
        return text[: end + len("</svg>")]
    return text + "</svg>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("outputs/samples"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--prefix_name", choices=list(PREFIXES), default="unconditional")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(args.tokenizer))
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prefix = PREFIXES[args.prefix_name]
    prefix_ids = tok.encode(prefix).ids
    records = []
    for i in range(args.n):
        idx = torch.tensor([prefix_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        text = tok.decode(out[0].tolist())
        svg = trim_to_svg(text)
        name = f"{args.prefix_name}_t{args.temperature}_sample{i:02d}.svg"
        (args.out_dir / name).write_text(svg, encoding="utf-8")
        records.append({"file": name, "prefix_name": args.prefix_name, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p, "prefix": prefix, "svg": svg})
    with open(args.out_dir / f"{args.prefix_name}_manifest.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} SVGs to {args.out_dir}")


if __name__ == "__main__":
    main()
