"""Central metric key constants for RL logging.

This consolidates string keys used across trainer/buffer/loggers.
"""

# Core rewards
REWARD = "reward"
REWARD_STD = "reward_std"
RAW_REWARD = "raw_reward"
RAW_REWARD_STD = "raw_reward_std"

# Training
LEARNING_RATE = "learning_rate"
GRAD_NORM = "grad_norm"
EPOCH = "epoch"
ETA_MINUTES = "eta_minutes"
TEMPERATURE = "temperature"
BETA = "beta"
LOSS = "loss"

# Completion metrics
TERM_RATIO = "completions/terminated_ratio"
MEAN_LEN_TOK = "completions/mean_len_tok"
GT_MEAN_LEN_TOK = "gt/mean_len_tok"
# cap_hit_ratio removed with dynamic caps
RATIO_TO_GT = "completions/ratio_to_gt_mean"
OVER_UPPER = "completions/over_upper_ratio"
UNDER_LOWER = "completions/under_lower_ratio"
ZERO_LEN_RATIO = "completions/zero_len_ratio"

# Dynamic length metrics removed

# Clipping ratios
CLIP_LOW_MEAN = "clip_ratio/low_mean"
CLIP_HIGH_MEAN = "clip_ratio/high_mean"
CLIP_REGION_MEAN = "clip_ratio/region_mean"

# Buffer telemetry
BUFFER_STEPS_PER_GEN = "buffer/steps_per_generation"
BUFFER_REUSE_COUNT = "buffer/reuse_count"
BUFFER_EFFICIENCY = "buffer/generation_efficiency"
BUFFER_REUSE_ACTIVE = "buffer/reuse_active"

__all__ = [
    "REWARD",
    "REWARD_STD",
    "RAW_REWARD",
    "RAW_REWARD_STD",
    "LEARNING_RATE",
    "GRAD_NORM",
    "EPOCH",
    "ETA_MINUTES",
    "TEMPERATURE",
    "BETA",
    "LOSS",
    "TERM_RATIO",
    "MEAN_LEN_TOK",
    "GT_MEAN_LEN_TOK",
    # cap hit ratio removed
    "RATIO_TO_GT",
    "OVER_UPPER",
    "UNDER_LOWER",
    "ZERO_LEN_RATIO",
    # dynamic length keys removed
    "CLIP_LOW_MEAN",
    "CLIP_HIGH_MEAN",
    "CLIP_REGION_MEAN",
    "BUFFER_STEPS_PER_GEN",
    "BUFFER_REUSE_COUNT",
    "BUFFER_EFFICIENCY",
    "BUFFER_REUSE_ACTIVE",
]
