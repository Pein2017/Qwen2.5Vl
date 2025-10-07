## Dynamic Generation Length (GT‑Aware) for Dense Captioning RL

### Why
- Fixed `max_new_tokens` either truncates valid dense scenes or allows chaotic long tails.
- Dense captioning has ground‑truth (GT) objects per image; we can estimate a plausible assistant output length from GT and bound generation dynamically per sample.
- Combine a per‑sample hard cap with a reward that sharply penalizes overflow and malformed tails.

### Overview
- For each sample, estimate a GT‑based target length, then set a per‑sample generation cap:
  - cap = clamp(round(alpha * gt_len + eos_margin), min_cap, max_cap_global)
- Masking policy: allow truncated completions to contribute gradients; do not zero their masks. Optionally mask only tokens beyond the dynamic cap. Always log a truncation flag for diagnostics.
- Add a continuous reward `length_vs_gt` that is 1.0 inside a tolerance window of the GT length, decays for too short, and strongly penalizes overflow, especially when the overflow is numeric/noisy.

### Length estimator options
- Tokenizer length (recommended for cap):
  - Build canonical assistant text from `meta.objects` using `CoordinateTokenConverter.convert_objects_to_tokens(objects)`.
  - `gt_len = len(tokenizer(gt_text, add_special_tokens=False).input_ids)`.
- Proxy consistent with existing formatting rewards (used in `length_window`):
  - `gt_len ≈ count(numbers) + count(<|object_ref_start|>)`.

Choose one estimator for both cap and reward to avoid drift, or use tokenizer for cap and proxy for reward to match existing metrics; both are viable.

### Per‑sample hard cap (generation)
- Compute cap per sample before calling `model.generate`:
  - `cap = clamp(round(alpha * gt_len + eos_margin), min_cap, max_cap_global)`
- Use this `cap` as the sample’s `max_new_tokens` for all K iid completions of that sample.
- Set `mask_truncated_completions: false` so truncated completions contribute gradients; optionally enable `mask_overflow_only: true` to mask only tokens beyond the dynamic cap.

Suggested defaults:
- `alpha = 1.1`
- `eos_margin = 16`
- `min_cap = 64`
- `max_cap_global = 1200`
- Estimator: `tokenizer`

### Overflow/tail reward: `length_vs_gt`
- Let `r = gen_len / max(gt_len, 1)` (use the same estimator chosen above).
- Score in [0, 1]:
  - Inside window [lower, upper]: 1.0
  - Below lower: linear ramp `r/lower` (mild penalty for too short)
  - Above upper: exponential decay `exp(-gamma * (r - upper))` (strong overflow penalty)
- Optional tail penalty targeting noisy numeric tails:
  - Compute the portion of text beyond `ceil(alpha * gt_len)`; measure numeric fraction in tail (share of digits among characters or numbers among tokens) and subtract `tail_numeric_weight * numeric_fraction_in_tail`, clipped to [0,1].

Suggested defaults:
- `lower = 0.7`
- `upper = 1.2`
- `gamma = 3.0`
- `tail_numeric_weight = 0.4`

### Config knobs (proposed)
Add to YAML under GRPO and rewards.

```yaml
# ===== GRPO =====
grpo:
  # Enable dynamic per‑sample cap
  dynamic_length:
    enabled: true
    estimator: tokenizer           # or: numbers_wrappers
    alpha: 1.1
    eos_margin: 16
    min_cap: 64
    max_cap: 1200
    hard_cap: true                 # enforce via per‑sample max_new_tokens
  # Masking policy for completions
  mask_truncated_completions: false   # keep truncated completions in loss; optionally mask only beyond the cap

# ===== Rewards =====
rewards:
  # Observe only unless you assign a non‑zero weight
  length_vs_gt: 0.0

rewards_config:
  length_vs_gt:
    estimator: tokenizer           # match GRPO choice to avoid drift
    lower: 0.7
    upper: 1.2
    gamma: 3.0
    tail_numeric_weight: 0.4
```

### Integration points (no code changes yet)
- Dataset/meta: RL dataset already emits `meta.objects`; use it to build GT assistant text.
- Generation cap:
  - In `rl/buffer.py` before calling `generation.sample_k(...)`, compute `gt_len` from `meta.objects`, derive `cap`, and pass as `max_new_tokens` for that sample.
  - Ensure the same `cap` is used for all K completions of the same sample.
- Reward:
  - Add `length_vs_gt(text, meta)` to `rewards/format_rewards.py` (or a new small module), using the chosen estimator and parameters from `rewards_config.length_vs_gt`.
  - Register it in `rewards/registry.py` so it can be weighted or observed.
- Logging (console + TensorBoard):
  - Log `dynamic_length/{mean_cap,min_cap,max_cap}` each step along with existing completion length stats.
  - `RewardLogger` already logs `rewards/<name>/{mean,std}`; `length_vs_gt` will appear automatically when present.

### Pseudocode
```python
# inside generate_and_score (per sample)
objects = sample.get("meta", {}).get("objects", [])
if dynamic_length.enabled:
    gt_text = converter.convert_objects_to_tokens(objects)
    if estimator == "tokenizer":
        gt_len = len(tokenizer(gt_text, add_special_tokens=False).input_ids)
    else:
        gt_len = count_numbers(gt_text) + gt_text.count(OBJ_S)
    cap = clamp(round(alpha * gt_len + eos_margin), min_cap, max_cap)
else:
    cap = global_max_new_tokens

seqs = generation.sample_k(..., max_new_tokens=cap, ...)

# reward
def length_vs_gt(text, meta):
    gt_len = estimate_length_from(meta.objects)
    gen_len = estimate_length_from(text)
    r = gen_len / max(gt_len, 1)
    if r < lower:
        base = max(0.0, r / lower)
    elif r <= upper:
        base = 1.0
    else:
        base = math.exp(-gamma * (r - upper))
    tail_pen = 0.0
    if tail_numeric_weight > 0 and gen_len > alpha * gt_len:
        tail = text_portion_beyond(alpha * gt_len)
        frac_num = numeric_fraction(tail)
        tail_pen = tail_numeric_weight * frac_num
    # Keep truncated completions: reward still applies; no mask zeroing
    return max(0.0, min(1.0, base - tail_pen))
```

### Interactions and safeguards
- Set EOS id (`<|im_end|>`) and keep truncated completions in loss. Prefer masking only tokens beyond the dynamic cap if masking is needed; otherwise keep full completion masks so chaotic tails push gradients.
- Retain `no_repeat_ngram_size` at generation or a small repetition penalty; the overflow reward focuses on length, not n‑gram loops.
- Works with `sample_k`: each sample uses its own cap; all K completions share the same cap for comparability.
- Distributed: when using a shared sample across ranks, the cap computed from `meta.objects` is identical across ranks; no collective is needed to agree on it.

### Distributed stability (NCCL)
- Slow‑rank guard: the per‑sample cap prevents 1,500‑token tails on any single rank. Keep a resampling guard: if any rank’s generation time exceeds a threshold or yields zero completions, all ranks resample the same prompt before entering collectives.
- Synchronized buffer refresh: refresh/rebuild the generation buffer together across ranks so no rank enters gather while others are still consuming cached chunks.
- Pad/trim equality: pad or trim variable‑length tensors (e.g., `generation_logps`, reward vectors) to equal shapes before cross‑rank advantage computation.
- Diagnostics: log per‑rank completion lengths, generation timings, and the per‑sample dynamic cap; record resample events to correlate with timeouts.

### Expected impact
- Fewer runaway generations on small scenes; more budget for dense scenes.
- Strong penalty for “random chaos” tails (often numeric), improving reward signal quality.
- More stable memory/latency profiles; reduced risk of NCCL timeout due to extreme sequences.

### Open choices
- Estimator: `tokenizer` is more faithful; `numbers_wrappers` matches existing format rewards. Use one consistently for both cap and reward, or tokenizer for cap + proxy for reward if you prefer parity with `length_window`.
- Default hyperparameters (alpha/lower/upper/gamma/tail weight) should be validated on a small debug run.

### Minimal rollout plan
1. Add config knobs (GRPO.dynamic_length and rewards_config.length_vs_gt) without changing defaults for current runs.
2. Implement cap computation in `rl/buffer.py`; log `dynamic_length/*` stats.
3. Implement and register `length_vs_gt`; add it to `observe_rewards` first, then optionally give it a small weight.
4. Validate on `configs/dense_rl/debug.yaml` (100–200 steps) and inspect logs/plots.
