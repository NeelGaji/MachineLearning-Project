from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def token_hist(lengths_csv: Path, out: Path):
    df = pd.read_csv(lengths_csv)
    plt.figure(figsize=(7, 4))
    plt.hist(df["token_length"], bins=50)
    plt.xlabel("Token length per SVG")
    plt.ylabel("Number of SVGs")
    plt.title("SVG token length distribution")
    plt.tight_layout()
    plt.savefig(out, dpi=200)


def image_grid(image_dir: Path, out: Path, cols: int = 5, max_images: int = 20):
    imgs = sorted(list(image_dir.glob("*.png")))[:max_images]
    if not imgs:
        raise RuntimeError(f"No PNGs found in {image_dir}")
    pil = [Image.open(p).convert("RGB").resize((200, 200)) for p in imgs]
    rows = math.ceil(len(pil) / cols)
    grid = Image.new("RGB", (cols * 200, rows * 200), "white")
    for i, im in enumerate(pil):
        grid.paste(im, ((i % cols) * 200, (i // cols) * 200))
    grid.save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths_csv", type=Path, default=None)
    p.add_argument("--hist_out", type=Path, default=Path("outputs/token_length_hist.png"))
    p.add_argument("--image_dir", type=Path, default=None)
    p.add_argument("--grid_out", type=Path, default=Path("outputs/generated_grid.png"))
    args = p.parse_args()
    args.hist_out.parent.mkdir(parents=True, exist_ok=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    if args.lengths_csv:
        token_hist(args.lengths_csv, args.hist_out)
        print("Wrote", args.hist_out)
    if args.image_dir:
        image_grid(args.image_dir, args.grid_out)
        print("Wrote", args.grid_out)


if __name__ == "__main__":
    main()
