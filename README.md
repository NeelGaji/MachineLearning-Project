# SVG Transformer Scaling Laws Extra Credit Project

This repository is a complete starter implementation for the CS-GY 6923 optional SVG scaling-law project.
It covers preprocessing, ByteLevel BPE tokenization, decoder-only Transformer training, LR sweeps,
standard-parameterization scaling, µP experiments, scaling-law fitting, SVG generation, and SVG validity/render evaluation.

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a GPU if possible. For serious runs, an A100/L4/A10/T4-class GPU is much better than CPU.

## 1. Data preprocessing

Primary dataset only:

```bash
python -m src.preprocess \
  --datasets starvector/svg-icons-simple \
  --out data/svg_bpe4096 \
  --vocab_size 4096 \
  --max_token_len 1024 \
  --min_chars 50
```

If `stats.json` reports fewer than 100M training tokens, add supplementary data:

```bash
python -m src.preprocess \
  --datasets starvector/svg-icons-simple starvector/svg-emoji-simple starvector/svg-fonts-simple \
  --out data/svg_bpe4096 \
  --vocab_size 4096 \
  --max_token_len 1024 \
  --min_chars 50 \
  --max_files 300000
```

Optional render checking is slower but stronger:

```bash
python -m src.preprocess --datasets starvector/svg-icons-simple --out data/svg_bpe4096 --render_check
```

Render dataset examples for the report:

```bash
python -m src.render_examples --jsonl data/svg_bpe4096/train_svgs.jsonl --out_dir outputs/dataset_examples
```

## 2. Learning-rate sweep on Tiny model, standard parameterization

The project requires 5-7 LRs on a log scale. Use this sweep to choose the fixed LR for all SP model sizes.

```bash
python -m src.sweep_lr \
  --data_dir data/svg_bpe4096 \
  --out_root runs/sp_lr_sweep \
  --model_name tiny \
  --lrs 0.0001 0.0002 0.0003 0.0006 0.001 0.002 0.003 \
  --max_steps 1500
```

Open `runs/sp_lr_sweep/lr_sweep_summary.json` and pick `best.lr`.

## 3. One-epoch standard scaling runs

Replace `BEST_SP_LR` with the chosen LR.

```bash
for m in tiny small medium large xl; do
  python -m src.train --data_dir data/svg_bpe4096 --model_name $m \
    --out_dir runs/sp_$m --learning_rate BEST_SP_LR
 done
```

If memory/time fails, add `--fallback` and clearly justify the reduced scale in the report.

## 4. µP base shapes and LR sweep

Create µP base-shape metadata:

```bash
python scripts/make_mup_base_shapes.py --data_dir data/svg_bpe4096 --out mup_shapes.bsh
```

Sweep LR on Tiny µP:

```bash
python -m src.sweep_lr \
  --data_dir data/svg_bpe4096 \
  --out_root runs/mup_lr_sweep \
  --model_name tiny \
  --use_mup --base_shapes mup_shapes.bsh \
  --lrs 0.0001 0.0002 0.0003 0.0006 0.001 0.002 0.003 \
  --max_steps 1500
```

Then run the 5 µP model sizes with the best µP LR:

```bash
for m in tiny small medium large xl; do
  python -m src.train --data_dir data/svg_bpe4096 --model_name $m \
    --out_dir runs/mup_$m --learning_rate BEST_MUP_LR \
    --use_mup --base_shapes mup_shapes.bsh
 done
```

## 5. Scaling-law fitting and plots

```bash
python -m src.fit_scaling \
  --summaries runs/sp_tiny runs/sp_small runs/sp_medium runs/sp_large runs/sp_xl \
              runs/mup_tiny runs/mup_small runs/mup_medium runs/mup_large runs/mup_xl \
  --out_dir outputs/scaling
```

Outputs:

- `outputs/scaling/scaling_results.csv`
- `outputs/scaling/fit_results.json`
- `outputs/scaling/scaling_plot.png`

## 6. Best model generation

Use the best checkpoint, probably largest feasible µP/SP model.

Unconditional samples:

```bash
python -m src.generate --ckpt runs/mup_xl/best.pt --tokenizer data/svg_bpe4096/tokenizer.json \
  --out_dir outputs/samples_uncond_t08 --prefix_name unconditional --n 10 --temperature 0.8 --top_k 100
```

Prefix-conditioned samples:

```bash
for p in face open_path one_shape_group star_start; do
  python -m src.generate --ckpt runs/mup_xl/best.pt --tokenizer data/svg_bpe4096/tokenizer.json \
    --out_dir outputs/samples_prefix --prefix_name $p --n 2 --temperature 0.8 --top_k 100
 done
```

Evaluate generated SVG validity and render rate:

```bash
python -m src.evaluate_svg --sample_dir outputs/samples_uncond_t08 --render_dir outputs/rendered_uncond_t08
python -m src.evaluate_svg --sample_dir outputs/samples_prefix --render_dir outputs/rendered_prefix
```
