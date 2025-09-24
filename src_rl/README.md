# src_rl – Reinforcement Learning Utilities for Qwen2.5-VL

This module hosts the reinforcement learning (GRPO) stack used to post-train the
Qwen2.5-VL dense captioning model. It mirrors the supervised (`src_new/`)
pipeline so that multimodal prompts, geometry wrappers, and tokenizer
conventions remain consistent between SFT, RL, and inference.

## What lives here
- `runner.py`: end-to-end launcher that loads the SFT checkpoint, materialises
  datasets, wires reward functions, and instantiates `src_rl.trainer.VisionGRPOTrainer`.
- `data/`: JSONL dataset loader that reconstructs generation-ready tensors via
  the shared `ConversationBuilder`.
- `prompting/`: helpers that resolve image paths, build prompts, and keep chat
  templates aligned with SFT.
- `rewards/`: reward components and registry used by both training and
  evaluation pipelines.
- `trainer.py`: vision-aware GRPO adapter used by the launcher to pass vision
  tensors (`pixel_values`, `image_grid_thw`) through generation and scoring.
- `eval.py`: offline evaluation entry point using the same loader stack.
- `tools/`: diagnostics (e.g., parity check for prompt rendering).

## Prerequisites
1. **Environment**: activate the same Conda env used for SFT, e.g.
   ```bash
   source ~/.bashrc
   conda activate ms
   ```
2. **Checkpoint**: supply an SFT checkpoint directory that contains the
   `pytorch_model.bin` shards, tokenizer, and processor artifacts. This is the
   `model_path` referenced in RL configs.
3. **Dataset**: RL expects the processed JSONL from the strict V2 data
   conversion pipeline (`data_conversion/`). Each line must include
   `images` (relative paths under `data_root`) plus metadata fields used by the
   reward functions.
4. **GPU**: training is targeted at GPU execution. The loader will fall back to
   CPU/float32 if CUDA or bf16 is unavailable, but GRPO will be extremely slow
   in that mode and is intended only for smoke checks.

## Quick start (training)
```bash
python -m src_rl.runner \
  --config /data3/Qwen2.5-VL-main/configs/rl/dense_grpo.yaml \
  --mode train
```

The launcher performs the following steps:
1. Applies the standard Qwen2.5-VL compatibility patches.
2. Loads tokenizer, processor, and `DetectionModel` with the same validation
   logic as `src_new` (pad=`eos`, left padding, eager attention, coordinate
   checks).
3. Builds RL datasets (`RLDenseJSONLDataset`) that deserialize JSONL samples and
   forward them through `ConversationBuilder` to obtain tensors.
4. Constructs reward callables via `src_rl.rewards.registry.REGISTRY` and injects
   their weights into `VisionGRPOTrainer`.
5. Launches GRPO training for the number of steps specified in the config.

The command is resume-aware: set `resume_from_checkpoint` in the config to an
output directory to continue training.

## Configuration reference (`configs/rl/*.yaml`)
Important keys consumed by the launcher:

| Key | Description |
| --- | --- |
| `model_path` | Absolute path to the base SFT checkpoint (required). |
| `train_data_path` / `val_data_path` | JSONL paths produced by data conversion (required). |
| `data_root` | Directory used to resolve relative image paths in JSONL (required). |
| `attn_implementation` | One of `eager`, `flash_attention_2`, `sdpa`. Loader always enforces eager during RL for stability. |
| `bf16` | Request bfloat16. Loader now probes runtime capability and falls back to float32 if bf16+CUDA is unsupported (logs a warning instead of raising). |
| `image.max_pixels` | Optional cap applied to the `Qwen2VLImageProcessor`. |
| `max_coord_value`, `coordinate_tokens_enabled` | Required by `ConversationBuilder` to keep geometry invariant logic explicit. |
| `sample_k`, `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` | Sampling parameters forwarded to GRPO. |
| `per_device_train_batch_size`, `update_steps`, `learning_rate`, `warmup_steps`, `max_steps` | Core GRPO hyperparameters. |
| `rewards` | Mapping of reward component names to weights. Components must exist in `src_rl.rewards.registry.REGISTRY`. |
| `output_dir` | Where checkpoints, logs, and trainer state are written. |

Any missing required key triggers a `ValueError` before training starts to avoid
silent defaults.

## Evaluation
`src_rl/eval.py` provides an offline scorer that reuses the loader stack:
```bash
python -m src_rl.eval \
  --config configs/rl/dense_grpo.yaml \
  --input_file /abs/path/to/val.jsonl \
  --data_root /abs/path/to/data_root \
  --output_file eval_report.json
```
It generates responses with the RL (or SFT) checkpoint, computes each reward
component, and emits aggregate metrics (parse success rate, wrapper/coords
compliance, ASCII separator usage, banned vocab rate, and weighted reward).

## Reward components
Implemented in `src_rl/rewards/format_rewards.py` and surfaced via
`src_rl/rewards/registry.py`:
- `parse`: detects the presence of any geometry block.
- `wrappers`: verifies that object and geometry wrapper pairs are balanced.
- `coords`: checks coordinate counts (4 for boxes, 8 for quads, even-length ≥4 for polylines).
- `separators`: encourages ASCII punctuation and proper comma spacing.
- `vocab`: penalises usage of banned domain terms.
- `coverage`: object count proximity to ground truth, tolerant within ±1.
- `geometry_sanity`: bounds and monotonicity checks, using `max_coord_value`/image size when available.
- `bbox_giou`: GIoU-based matching on AABBs; mapped from [-1,1] to [0,1] for stability.
- `quad_l1`: greedy L1 proximity on canonically ordered quad vertices (top-left start, clockwise).
- `line_l1`: greedy L1 proximity on line endpoints with canonical direction (leftmost-first).
- `ordering`: checks quad vertex ordering (top-left start, clockwise) and line direction ordering.

All functions are deterministic and side-effect free so they can be used both
online (during GRPO) and offline (evaluation, analysis scripts).

## Development tips
- Unit tests: run `pytest src_rl/rewards/test_format_rewards.py` and
  `pytest src_rl/rewards/test_detection_rewards.py` for quick coverage of reward helpers.
  Additional parity tests live under `src_rl/tools/`.
- Smoke test the loader with `python -m src_rl.runner --config <cfg> --mode load`
  to validate tokenizer/model parity without starting GRPO.
- The dataset loader reads JSONL eagerly into memory for random access (required
  by `trl`); keep datasets reasonably sized or shard them before training.
- BF16 capability is auto-detected. If you need to force float32 (e.g., CPU
  dev), set `bf16: false` in the config.

## Troubleshooting
- **`ValueError: Your setup doesn't support bf16/gpu.`** – Prior to the
  capability probe, this error was common. The new probe now downgrades to
  float32; ensure you pulled the latest changes and re-run the launcher.
- **Dataset path errors** – The loader raises actionable exceptions if JSONL or
  images are missing. Verify `train_data_path`, `val_data_path`, and `data_root`
  in the YAML.
- **Slow CPU runs** – Without CUDA the trainer operates in CPU-only mode. This
  is fine for functionality checks but impractical for production training.

For implementation details and invariants, refer to `src_new/UNIFIED_DOCUMENTATION.md`, as
`src_rl` intentionally mirrors the SFT preparation pipeline.

