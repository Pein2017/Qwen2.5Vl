# Logging Contract: Dynamic Generation Length

Metrics (Trainer logs and TensorBoard):
- dynamic_length/mean_cap: float — mean per-sample cap used in this step
- dynamic_length/min_cap: float — min per-sample cap used in this step
- dynamic_length/max_cap: float — max per-sample cap used in this step
- completions/mean_length: float — mean effective completion length
- completions/min_length: float — min effective completion length
- completions/max_length: float — max effective completion length
- completions/terminated_ratio: float in [0,1] — fraction with <|im_end|>
- rewards/<name>/mean, rewards/<name>/std: reward component stats (includes length_vs_gt when present)
- clip_ratio/low_mean, clip_ratio/high_mean, clip_ratio/region_mean: PPO clipping diagnostics

Validation:
- dynamic_length/* present when grpo.dynamic_length.enabled=true
- If dynamic_length.enabled=false, dynamic_length/* MAY be omitted; fallback lengths still logged
