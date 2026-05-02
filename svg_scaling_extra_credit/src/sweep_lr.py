from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--out_root", type=Path, default=Path("runs/lr_sweep"))
    p.add_argument("--model_name", default="tiny")
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4, 6e-4, 1e-3, 2e-3, 3e-3])
    p.add_argument("--use_mup", action="store_true")
    p.add_argument("--base_shapes", type=Path, default=None)
    p.add_argument("--fallback", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=50, help="Use a smaller equal budget for LR sweep; final scaling uses 1 epoch.")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for lr in args.lrs:
        out = args.out_root / f"{args.model_name}_{'mup' if args.use_mup else 'sp'}_lr{lr:g}"
        cmd = [sys.executable, "-m", "src.train", "--data_dir", str(args.data_dir), "--model_name", args.model_name,
               "--out_dir", str(out), "--learning_rate", str(lr), "--max_steps", str(args.max_steps)]
        if args.use_mup:
            cmd.append("--use_mup")
            if args.base_shapes:
                cmd += ["--base_shapes", str(args.base_shapes)]
        if args.device:
            cmd += ["--device", args.device]
        if args.fallback:
            cmd.append("--fallback")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        with open(out / "summary.json") as f:
            summary = json.load(f)
        rows.append({"lr": lr, "val_loss": summary["best_val_loss"], "out_dir": str(out)})

    best = min(rows, key=lambda r: r["val_loss"])
    with open(args.out_root / "lr_sweep_summary.json", "w") as f:
        json.dump({"runs": rows, "best": best}, f, indent=2)
    print(json.dumps({"best": best, "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
