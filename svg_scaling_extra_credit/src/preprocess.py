"""Prepare SVG LM data.

Pipeline:
1. Load one or more HuggingFace SVG datasets.
2. Extract SVG strings robustly from common columns.
3. Clean and XML-validate SVGs.
4. Train a ByteLevel BPE tokenizer.
5. Tokenize each SVG with BOS/EOS, filter by token length, split by file.
6. Write train/val/test .bin arrays plus dataset statistics.

Example:
python -m src.preprocess \
  --datasets starvector/svg-icons-simple starvector/svg-emoji-simple \
  --out data/svg_bpe4096 \
  --vocab_size 4096 --max_token_len 1024 --min_chars 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from datasets import load_dataset
from lxml import etree
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
XML_DECL_RE = re.compile(r"<\?xml.*?\?>", re.DOTALL | re.IGNORECASE)
DOCTYPE_RE = re.compile(r"<!DOCTYPE.*?>", re.DOTALL | re.IGNORECASE)
META_RE = re.compile(r"<(metadata|title|desc)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
WS_RE = re.compile(r"\s+")
FLOAT_RE = re.compile(r"(?<![A-Za-z])[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def clean_svg(svg: str, precision: int = 1) -> str:
    svg = svg.strip()
    svg = XML_DECL_RE.sub("", svg)
    svg = DOCTYPE_RE.sub("", svg)
    svg = COMMENT_RE.sub("", svg)
    svg = META_RE.sub("", svg)

    def round_float(m: re.Match[str]) -> str:
        x = float(m.group(0))
        s = f"{x:.{precision}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s

    svg = FLOAT_RE.sub(round_float, svg)
    svg = WS_RE.sub(" ", svg)
    svg = re.sub(r"\s*([=<>/])\s*", r"\1", svg)
    return svg.strip()


def is_valid_xml(svg: str) -> bool:
    try:
        etree.fromstring(svg.encode("utf-8"), parser=etree.XMLParser(recover=False))
        return True
    except Exception:
        return False


def try_render(svg: str) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=tempfile.NamedTemporaryFile(suffix=".png").name)
        return True
    except Exception:
        return False


def find_svg_in_record(record: dict) -> Optional[str]:
    # Prefer common column names.
    for key in ["svg", "SVG", "Svg", "code", "content", "text"]:
        val = record.get(key)
        if isinstance(val, str) and "<svg" in val.lower():
            return val
    # Robust fallback: scan all string values.
    for val in record.values():
        if isinstance(val, str) and "<svg" in val.lower():
            return val
    return None


def iter_svgs(dataset_names: list[str], streaming: bool = False, max_files: int = 0) -> Iterable[tuple[str, str]]:
    seen = 0
    for ds_name in dataset_names:
        ds = load_dataset(ds_name, split="train", streaming=streaming)
        for row in ds:
            svg = find_svg_in_record(row)
            if svg:
                yield ds_name, svg
                seen += 1
                if max_files and seen >= max_files:
                    return


def train_tokenizer(texts: list[str], vocab_size: int, out_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tokenizer.save(str(out_path))
    return tokenizer


def encode_svg(tokenizer: Tokenizer, svg: str) -> list[int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    return [bos] + tokenizer.encode(svg).ids + [eos]


def write_bin(path: Path, sequences: list[list[int]]) -> int:
    n = sum(len(s) for s in sequences)
    dtype = np.uint16
    arr = np.empty(n, dtype=dtype)
    pos = 0
    for seq in sequences:
        arr[pos : pos + len(seq)] = np.asarray(seq, dtype=dtype)
        pos += len(seq)
    arr.tofile(path)
    return int(n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["starvector/svg-icons-simple"])
    p.add_argument("--out", type=Path, default=Path("data/svg_bpe4096"))
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--max_token_len", type=int, default=1024)
    p.add_argument("--min_chars", type=int, default=50)
    p.add_argument("--precision", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_files", type=int, default=0, help="0 means no cap; useful for smoke tests")
    p.add_argument("--render_check", action="store_true", help="Slower: require CairoSVG render success")
    args = p.parse_args()

    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned: list[dict] = []
    counts = {"raw": 0, "too_short": 0, "xml_invalid": 0, "render_invalid": 0, "kept_after_xml": 0}
    print(f"Loading datasets: {args.datasets}")
    for source, raw_svg in tqdm(iter_svgs(args.datasets, max_files=args.max_files), desc="clean+xml"):
        counts["raw"] += 1
        svg = clean_svg(raw_svg, precision=args.precision)
        if len(svg) < args.min_chars:
            counts["too_short"] += 1
            continue
        if not is_valid_xml(svg):
            counts["xml_invalid"] += 1
            continue
        if args.render_check and not try_render(svg):
            counts["render_invalid"] += 1
            continue
        counts["kept_after_xml"] += 1
        cleaned.append({"source": source, "svg": svg, "chars": len(svg)})

    if len(cleaned) < 100:
        raise RuntimeError("Too few valid SVGs. Check dataset columns or filters.")

    with open(args.out / "cleaned_svgs.jsonl", "w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokenizer_path = args.out / "tokenizer.json"
    print(f"Training BPE tokenizer: vocab_size={args.vocab_size}")
    tokenizer = train_tokenizer([r["svg"] for r in cleaned], args.vocab_size, tokenizer_path)
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab > np.iinfo(np.uint16).max:
        raise RuntimeError("Vocab too large for uint16 storage. Use uint32 or smaller vocab.")

    filtered: list[dict] = []
    lengths = []
    for row in tqdm(cleaned, desc="tokenize+filter"):
        ids = encode_svg(tokenizer, row["svg"])
        row["tokens"] = len(ids)
        if len(ids) <= args.max_token_len:
            row["ids"] = ids
            filtered.append(row)
            lengths.append(len(ids))

    random.shuffle(filtered)
    n = len(filtered)
    n_train = int(0.98 * n)
    n_val = int(0.01 * n)
    splits = {
        "train": filtered[:n_train],
        "val": filtered[n_train : n_train + n_val],
        "test": filtered[n_train + n_val :],
    }

    split_stats = {}
    for split, rows in splits.items():
        toks = write_bin(args.out / f"{split}.bin", [r["ids"] for r in rows])
        split_stats[split] = {"files": len(rows), "tokens": toks}
        with open(args.out / f"{split}_svgs.jsonl", "w", encoding="utf-8") as f:
            for r in rows[:5000]:  # enough for qualitative examples, avoids huge sidecar files
                f.write(json.dumps({k: r[k] for k in ["source", "svg", "chars", "tokens"]}, ensure_ascii=False) + "\n")

    with open(args.out / "lengths.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_length"])
        for L in lengths:
            w.writerow([L])

    stats = {
        "datasets": args.datasets,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": actual_vocab,
        "max_token_len": args.max_token_len,
        "min_chars": args.min_chars,
        "precision": args.precision,
        "counts": counts,
        "after_token_filter_files": len(filtered),
        "length_stats": {
            "min": int(np.min(lengths)),
            "median": float(np.median(lengths)),
            "mean": float(np.mean(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(np.max(lengths)),
        },
        "splits": split_stats,
    }
    with open(args.out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    if split_stats["train"]["tokens"] < 100_000_000:
        print("WARNING: train token count is below 100M. Add svg-emoji-simple and/or subsample svg-fonts/simple-stack.")


if __name__ == "__main__":
    main()
