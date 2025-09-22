## Focused module guide

### src_new/ — Supervised Fine-Tuning (SFT)
- **Purpose**: Main training pipeline for Qwen2.5‑VL SFT.
- **Key subpackages**: `config/`, `data/`, `processing/`, `models/`, `losses/`, `training/`, `utils/`.
- **Entrypoints**:
  - Direct: `conda activate ms` then
    ```bash
    python scripts/train_new.py --config configs/bbu_v2/base.yaml
    ```
  - Launcher: see `scripts/run_new_train.sh` (auto GPU selection, env setup, single/multi‑GPU).
- **Docs**: `src_new/UNIFIED_DOCUMENTATION.md` (pipeline, variants, losses, checkpoints).

### src_post/ — RL Post-Training (Group‑level GRPO)
- **Purpose**: Post‑train an SFT checkpoint for group‑level QC with Stage‑A summaries and a Stage‑B decision.
- **Entrypoint**:
  ```bash
  python -m src_post.runner --config configs/rl/group_qc_grpo.yaml
  ```
- **Docs**: `src_post/README.md` (rewards, prompts, GRPO flow, multi‑GPU usage).

### data_conversion/ — Corpus preparation
- **Purpose**: Convert raw annotations into the unified JSONL used by `src_new/`.
- **Quick run**:
  ```bash
  bash data_conversion/convert_dataset.sh
  ```
- **Outputs** (under `data/` by default): `train.jsonl`, `val.jsonl`, `teacher_pool.jsonl` (plus optional summary SFT files). See per‑file docs inside `data_conversion/` for details.

### scripts/run_new_train.sh — SFT launcher
- **What it does**: Sets up env (HF caches, NCCL, Triton when needed), picks GPUs, validates config, and launches `scripts/train_new.py` (or JSON/ref variants) in single‑GPU or PyTorch DDP.
- **Key knobs**:
  - `CONFIG_NAME` (e.g., `phase_3/standard`) → uses `configs/${CONFIG_NAME}.yaml`
  - `GPU_DEVICES` (e.g., `0,1,2,3`) or pass `1|2` to select presets
  - `ARCH` = `legacy` (src_new) | `json` (src_new_json) | `ref` (src_new_reference)
  - `MAX_STEPS` for short sanity runs
- **Examples**:
  ```bash
  # default (all visible GPUs)
  bash scripts/run_new_train.sh

  # preset GPU groups
  bash scripts/run_new_train.sh 1

  # override GPUs and log target
  GPU_DEVICES=0,1 LOG_NAME=run.log bash scripts/run_new_train.sh
  ```

### configs/ — Configuration files
- **SFT**: `configs/bbu_v2/*.yaml` and phase presets (e.g., `configs/phase_3/standard.yaml`). Consumed by `scripts/train_new.py` (and the launcher).
- **RL**: `configs/rl/group_qc_grpo.yaml` for `src_post.runner`.
- **Tip**: keep paths absolute or resolve via repo root; adjust model/data paths, attention impl (`attn_implementation`), batch sizes, and variant ratios as needed.

