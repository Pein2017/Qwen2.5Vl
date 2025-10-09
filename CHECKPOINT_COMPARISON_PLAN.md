# Checkpoint EOS Comparison Plan

## Quick Approach: Reuse Existing GRPO Diagnostic

Instead of writing new inference code, use your working `test_eos_fix.sh`:

### Test Each Checkpoint

```bash
# 1. Phase 2 (current - already tested)
# Result: term_ratio=0.000

# 2. Phase 3 base
vim configs/dense_rl/diagnostic.yaml
# Change line 9:
#   model_path: "outputs/.../phase_3/.../best-800-eval_loss0.2135"

pkill -9 -f runner.py
bash test_eos_fix.sh > phase3_base_test.log 2>&1 &
sleep 90
grep "term_ratio" phase3_base_test.log | tail -5

# 3. Phase 3 with text_only
# Change model_path to: best-800-eval_loss0.2137
# Repeat test

# 4. Phase 3 resume
# Change model_path to: best-800-eval_loss0.1004  
# Repeat test
```

### What to Look For

```
term_ratio=0.000  → Checkpoint doesn't generate EOS
term_ratio=0.5-0.8 → Checkpoint DOES generate EOS ✅
```

### Alternative: Check with Longer max_new_tokens FIRST

Your Phase 2 might actually generate EOS, just needs >512 tokens!

```yaml
# diagnostic.yaml
generation:
  max_new_tokens: 2048  # Try full SFT length
  dynamic_length:
    enabled: false
```

Then test Phase 2 again. If term_ratio > 0, no need to test other checkpoints!

## Recommendation

**FIRST**: Test Phase 2 with max_new_tokens=2048
- If EOS appears → Problem solved, just config issue
- If still 0.000 → Then test Phase 3 checkpoints

This saves time since Phase 2 loads faster.
