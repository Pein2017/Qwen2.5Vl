# Config Contract: Dynamic Generation Length

Required keys and types (YAML):

- grpo.dynamic_length (object)
  - enabled: boolean (required)
  - estimator: string, one of {tokenizer} (required)
  - alpha: number > 0 (required)
  - eos_margin: integer >= 0 (required)
  - min_cap: integer >= 0 (required)
  - max_cap: integer > 0 (required)
  - hard_cap: boolean (required)

- grpo.mask_truncated_completions: boolean (required; must be false for this feature to train on truncated outputs)
- rewards_config.length_vs_gt (object)
  - estimator: string, default tokenizer (optional)
  - lower: number in (0,1) (optional; default 0.7)
  - upper: number > lower (optional; default 1.2)
  - gamma: number > 0 (optional; default 3.0)
  - tail_numeric_weight: number in [0,1] (optional; default 0.4)
  - alpha: number > 0 (optional; default 1.1)

Validation rules:
- min_cap <= max_cap
- lower < upper
- If dynamic_length.enabled=false, per-sample cap MUST be bypassed and `grpo.max_new_tokens` used.
