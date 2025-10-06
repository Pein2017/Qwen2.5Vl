# Research (Phase 0)

## Decisions
- Logging tag map: Align with HF Trainer-style tags
  - Scalars: `train/loss`, `reward`, `reward_std`, `grad_norm`, `learning_rate`, `temperature`, `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`, `step`, `epoch`, `eta_minutes`.
  - Per-reward: `rewards/<name>/mean`, `rewards/<name>/std`.
- Debug run limits: ≤ 20 steps, 1 epoch, `sample_k=4`, `bf16=true`, `top_p=0.95`, `temperature=1.0`.
- Required YAML keys (strict): `model_path`, `train_data_path`, `val_data_path`, `data_root`, `output_dir`, `bf16`, `model.attn_implementation`, `model.image_max_pixels`, `loss.{caption,grounding,formatting}`, `layer_config`.

## Open items (tracked)
- Cross-rank K sampling: design remainder policy (`divisible|round_robin`), gather mode (rewards-only), seed policy.
- Reward curriculum defaults for stability; keep geometry rewards at 0.0 for smoke.

## Risks & Mitigations
- Metric drift across ranks → normalize and log world-aggregated statistics when enabled.
- OOM during log-prob → per-sample sequential path; clip generation length.
- Unclear YAML → fail-fast with actionable messages; include example debug YAML.
