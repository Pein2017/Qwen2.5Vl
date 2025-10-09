# 🎯 ROOT CAUSE ANALYSIS: EOS Termination Issue

## Investigation Summary

### ✅ What's Working:
1. **EOS token correctly configured**: `<|im_end|>` = token ID 151645
2. **EOS passed to generation**: `generation.py` lines 103-104, 183-184 set `eos_token_id`
3. **Detection logic correct**: `buffer.py` lines 239-243 properly check for EOS in completions
4. **Term ratio calculation correct**: `grpo_trainer.py` line 410 correctly computes mean

### ❌ The Problem:
**Completions don't contain `<|im_end|>` token!**

All completions hitting `max_new_tokens` cap (512 in quick diagnostic).

## Why This Happens

### Most Likely: Dynamic Length Cap Too Restrictive

From `standard.yaml`:
```yaml
generation:
  max_new_tokens: 2048    # Hard cap
  min_new_tokens: 32
  dynamic_length:
    enabled: true
    estimator: "tokenizer"
    alpha: 1.1              # Cap = 1.1 × GT length
    eos_margin: 16
    min_cap: 64
    max_cap: 2048
```

**Problem**: If GT length is ~400 tokens:
- Cap = 1.1 × 400 = 440 tokens
- Model generates 440 tokens and stops (cap reached)
- Never gets chance to generate EOS naturally

### Secondary Issue: Dense Captioning is Long

Dense captioning with many objects can easily be 500+ tokens.
If model needs 600 tokens but cap is 440, it gets truncated mid-output.

## Evidence from Log

```
Step 1-5: term_ratio=0.000
```
Not a single completion out of 40 (5 steps × 8 completions) ended with EOS!

This strongly suggests **systematic truncation**, not random model behavior.

## The Fix

### Option 1: Increase Alpha (Quick Test)
```yaml
dynamic_length:
  alpha: 1.5   # Was 1.1, allow 50% longer
  # or
  alpha: 2.0   # Allow 100% longer
```

### Option 2: Disable Dynamic Length (Diagnostic)
```yaml
dynamic_length:
  enabled: false
```
Then `max_new_tokens: 2048` will be used directly.

### Option 3: Increase max_cap
```yaml
dynamic_length:
  max_cap: 3072  # Was 2048
```

## Expected Impact After Fix

**Before**:
```
term_ratio=0.000
reward variance: decreasing (truncation inconsistent)
```

**After**:
```
term_ratio=0.60-0.80  (60-80% end with EOS)
reward variance: stable (proper endings)
Learning: improved (consistent signal)
```

## Implementation

### Step 1: Test with alpha=1.5
Edit `configs/dense_rl/standard.yaml` line 46:
```yaml
alpha: 1.5  # Was 1.1
```

### Step 2: Run quick diagnostic again
```bash
bash run_8gpu_diagnostic.sh
```

### Step 3: Check term_ratio in logs
Should see `term_ratio > 0.5` within first few steps.

### Step 4: If still 0, disable dynamic_length
```yaml
dynamic_length:
  enabled: false
```

## Next Actions

1. **Apply fix** (edit config, rerun)
2. **Monitor term_ratio** (should jump to 60-80%)
3. **Check reward variance** (should stabilize)
4. **Run longer** (50-100 steps to see learning)

## Confidence Level

**95% confident** this is the root cause:
- 0/40 completions with EOS is too systematic to be model issue
- Dynamic caps are known to cause this
- Quick fix available

**If fix doesn't work**, then model genuinely doesn't generate EOS → SFT checkpoint issue.
