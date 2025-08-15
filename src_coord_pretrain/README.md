# Enhanced Coord Bootstrap Pre‑Training

Purpose: teach `<|coord_*|>` usage through coordinate token tuning with unlikelihood and optional reverse mapping, using a simple single‑phase pipeline (cosine scheduler). Vision tower is frozen; language tower and MLP aligner are tuned while masking gradients to keep base text embeddings fixed.

## 🚀 Quick Start

```bash
source ~/.bashrc && conda activate ms

# Generate training data
python src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output src_coord_pretrain/data/coord_bootstrap.jsonl \
  --num_identity 50000 --num_arithmetic 20000 --num_reverse 10000 \
  --max_coord 1024 --seed 42

# Run training
python src_coord_pretrain/training/trainer.py \
  --config src_coord_pretrain/config/coord_bootstrap.yaml
```

## 🎯 Features
- Single‑phase training with cosine LR scheduler
- Vision tower frozen
- Gradient mask on embeddings so only coordinate token embeddings update
- Unlikelihood training (digits and coord neighbor suppression)
- Optional reverse mapping samples supported by generator

## 📋 Key Config Options
```yaml
unlikelihood_enabled: true
unlikelihood_lambda_digits: 1.0
unlikelihood_lambda_coords: 1.0
unlikelihood_coord_window: 8
ul_topk_noncoord: 100
ul_topk_coord: 100
ul_neighbor_window: 8
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
llm_lr: 5.0e-6
mlp_lr: 2.0e-5
warmup_ratio: 0.1
lr_scheduler_type: "cosine"
max_epochs: 10
```

## 📈 Monitoring
- loss, llm_loss, unlikelihood_loss
- unlikelihood_digits, unlikelihood_coords (and top‑k variants when applicable)
- current_phase (fixed to "B"), llm_lr, mlp_lr, remaining_hrs

## 🔗 Integration with src_new
- Final checkpoint saved in `checkpoint-xxxx/` subfolder with model and tokenizer config.
- Use the final folder path directly in `src_new` configs.

## 🧪 Testing
- Dataset and collator tests validate assistant span labeling and coordinate token detection.
- Top‑K unlikelihood tests verify negative sampling and masks.

## Notes
- Paths in config YAML can be absolute or relative (resolved from CWD)
- Tokenizer/model must include `<|coord_0|>` … `<|coord_1024|>`
- Assistant‑only labels via offset mapping; `<|im_end|>` included in span
