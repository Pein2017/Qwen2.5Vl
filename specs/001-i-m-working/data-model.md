# Data Model (Phase 1)

## Entities
- RLLoaderConfig
  - Fields: `model_path: str`, `data_root: str`, `train_data_path: str`, `val_data_path: str`, `output_dir: str`, `bf16: bool`, `model.attn_implementation: str`, `model.image_max_pixels: int`
  - Invariants: explicit values required; fail-fast on missing.
- ManualTrainerConfig
  - Fields: `sample_k: int`, `max_new_tokens: int`, `min_new_tokens?: int`, `temperature: float`, `top_p: float`, `repetition_penalty: float`, `epsilon_low: float`, `epsilon_high: float`, `beta: float`, `standardize_rewards: bool`, `mask_truncated_completions: bool`, `gradient_accumulation_steps: int`, `per_device_train_batch_size: int`, `steps_per_generation: int`, `update_steps: int`, `logging_steps: int`, `save_steps: int`, `max_steps: int`, `bf16: bool`, `cross_rank_advantages: bool`.
  - Invariants: positive where applicable; mask/completion invariants checked.
- LoggingMetrics
  - Scalars: `train/loss`, `reward`, `reward_std`, `grad_norm`, `learning_rate`, `temperature`, `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`, `step`, `epoch`, `eta_minutes`.
  - Per-reward: `rewards/<name>/mean`, `rewards/<name>/std`.

## Relationships
- `RLLoaderConfig` feeds component loader; `ManualTrainerConfig` governs trainer loop; `LoggingMetrics` emitted by `TrainingStateManager` + trainer.
