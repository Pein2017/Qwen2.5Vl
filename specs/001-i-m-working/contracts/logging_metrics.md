# Logging Metrics Contract

Console and TensorBoard MUST record the following at `logging_steps` cadence:

## Scalars
- `train/loss`
- `reward`, `reward_std`
- `grad_norm`
- `learning_rate`
- `temperature`
- `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`
- `step`, `epoch`, `eta_minutes`

## Per-reward
- For each enabled reward name `R`:
  - `rewards/R/mean`
  - `rewards/R/std`

## Notes
- In distributed runs, log world-aggregated statistics for rewards/advantages when normalization is enabled.
- Console log lines SHOULD mirror TB scalar keys for easy grep.
