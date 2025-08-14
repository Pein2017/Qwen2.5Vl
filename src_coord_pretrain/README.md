# Coord Bootstrap Pre‑Training (Separate Module)

Purpose: teach `<|coord_*|>` usage (identity + small arithmetic) in a short text‑only SFT stage, then hand off the checkpoint to `src_new` for full grounding/detection training.

## Quick Start

- Generate data
```bash
source ~/.bashrc && conda activate ms
python src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output src_coord_pretrain/data/coord_bootstrap.jsonl \
  --num_identity 100000 --num_arithmetic 200000 --max_coord 1024 --seed 17
```

- Train
```bash
source ~/.bashrc && conda activate ms
python src_coord_pretrain/training/trainer.py \
  --config src_coord_pretrain/config/coord_bootstrap.yaml
```

## Notes
- Absolute paths required in config YAML (model_path, output_dir)
- Tokenizer/model must already include `<|coord_0|>` … `<|coord_1024|>`
- Assistant‑only labels via offset mapping; `<|im_end|>` included in span
