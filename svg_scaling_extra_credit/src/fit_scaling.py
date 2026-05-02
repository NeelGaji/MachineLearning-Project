from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def law(N, a, alpha, c):
    return a * np.power(N, -alpha) + c


def fit_one(df: pd.DataFrame, label: str, out_dir: Path):
    x = df["params"].to_numpy(dtype=float)
    y = df["val_loss"].to_numpy(dtype=float)
    c0 = max(0.0, y.min() - 0.2)
    p0 = [max(y.max() - y.min(), 1e-3) * x.min() ** 0.1, 0.1, c0]
    bounds = ([0.0, 0.0, 0.0], [1000.0, 5.0, y.min()])
    popt, pcov = curve_fit(law, x, y, p0=p0, bounds=bounds, maxfev=100000)
    yhat = law(x, *popt)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    # Bootstrap uncertainty for alpha and extrapolated 10x loss.
    rng = np.random.default_rng(42)
    alphas, preds = [], []
    target_N = 10 * x.max()
    for _ in range(1000):
        idx = rng.integers(0, len(x), len(x))
        try:
            boot, _ = curve_fit(law, x[idx], y[idx], p0=popt, bounds=bounds, maxfev=100000)
            alphas.append(boot[1])
            preds.append(law(target_N, *boot))
        except Exception:
            pass

    result = {
        "label": label,
        "a": float(popt[0]),
        "alpha": float(popt[1]),
        "c": float(popt[2]),
        "rmse": rmse,
        "target_params_10x_largest": float(target_N),
        "pred_loss_10x": float(law(target_N, *popt)),
        "alpha_ci95": [float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))] if alphas else None,
        "pred_loss_10x_ci95": [float(np.percentile(preds, 2.5)), float(np.percentile(preds, 97.5))] if preds else None,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True, help="summary.json files or directories containing summary.json")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/scaling"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in args.summaries:
        path = Path(item)
        if path.is_dir():
            path = path / "summary.json"
        with open(path) as f:
            s = json.load(f)
        rows.append({
            "model": s["model_name"],
            "param_type": "mup" if s.get("use_mup") else "sp",
            "params": s["params"],
            "val_loss": s["best_val_loss"],
            "elapsed_sec": s.get("elapsed_sec", 0),
            "tok_per_sec": s.get("tokens_seen", 0) / max(s.get("elapsed_sec", 1), 1e-9),
            "peak_mem_gb": s.get("peak_mem_gb", 0),
            "lr": s.get("learning_rate"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "scaling_results.csv", index=False)

    results = []
    plt.figure(figsize=(7, 5))
    for label, g in df.groupby("param_type"):
        g = g.sort_values("params")
        result = fit_one(g, label, args.out_dir)
        results.append(result)
        plt.scatter(g["params"], g["val_loss"], label=f"{label} observed")
        xs = np.geomspace(g["params"].min(), 10 * g["params"].max(), 200)
        plt.plot(xs, law(xs, result["a"], result["alpha"], result["c"]), label=f"{label} fit, alpha={result['alpha']:.3f}")
    plt.xscale("log")
    plt.xlabel("Parameters N (log scale)")
    plt.ylabel("Validation loss after one epoch")
    plt.title("SVG Transformer Scaling Laws")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "scaling_plot.png", dpi=200)

    with open(args.out_dir / "fit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("Wrote", args.out_dir)


if __name__ == "__main__":
    main()
