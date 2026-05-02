from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render_svg(svg: str, png_path: Path) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=200, output_height=200)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True, help="e.g., data/svg_bpe4096/train_svgs.jsonl")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/rendered_dataset_examples"))
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = sorted(rows, key=lambda r: r["tokens"])
    picks = []
    if rows:
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            picks.append(rows[int(q * (len(rows) - 1))])
    picks = picks[: args.n]
    manifest = []
    for i, r in enumerate(picks):
        png = args.out_dir / f"example_{i:02d}_{r['tokens']}tok.png"
        ok = render_svg(r["svg"], png)
        manifest.append({"png": png.name, "tokens": r["tokens"], "chars": r["chars"], "render_ok": ok, "svg": r["svg"]})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote examples to {args.out_dir}")


if __name__ == "__main__":
    main()
