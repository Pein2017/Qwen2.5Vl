# EOS Termination Investigation

## Critical Finding from Live Diagnostic
```
term_ratio=0.000 across ALL 5 steps
```
**0% of completions end with `<|im_end|>` token!**

## Why This Matters

### Impact on Training:
1. **Truncated outputs**: All completions hit max_new_tokens cap
2. **Inconsistent truncation**: Random cutoff points → variable rewards
3. **No "proper ending" signal**: Model never learns when to stop
4. **Explains oscillation**: Rewards vary based on where truncation happens

### Expected vs Actual:
- **Expected**: 80%+ should end with `<|im_end|>` naturally
- **Actual**: 0% → Something is broken

## Hypothesis Checklist

### H1: EOS token ID misconfigured ✅ CHECK THIS FIRST
```bash
# Verify EOS token is correct
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('outputs/7B-all_tokens/phase_2/9-21-phase_2-last_blocks_6-all_tokens/best-200-eval_loss0.7006', trust_remote_code=True)
print(f'EOS token: {tokenizer.eos_token}')
print(f'EOS token ID: {tokenizer.eos_token_id}')
print(f'<|im_end|> ID: {tokenizer.convert_tokens_to_ids(\"<|im_end|>\")}')
print(f'<|endoftext|> ID: {tokenizer.convert_tokens_to_ids(\"<|endoftext|>\")}')
"
```

### H2: EOS suppressed in generation config
Check if `suppress_tokens` or `forced_eos_token_id` is misconfigured in:
- `src_new/rl/buffer.py` generation call
- Generation config in model

### H3: Dynamic length cap too restrictive
Check if `dynamic_length` settings prevent reaching natural end:
```yaml
dynamic_length:
  enabled: true
  alpha: 1.1
  eos_margin: 16
  min_cap: 64
  max_cap: 2048
```

### H4: Model never learned to generate EOS
- SFT checkpoint issue
- Training didn't include EOS in labels
- EOS masked during SFT

### H5: EOS detection logic broken
Check `term_ratio` calculation in trainer - might be computing wrong token

## Investigation Plan

### Step 1: Verify EOS Token (30 sec)
Run tokenizer check above

### Step 2: Check Generation Call (2 min)
Inspect `buffer.py` line ~200 where `model.generate()` is called:
- Verify `eos_token_id` is passed
- Check if any suppression is active

### Step 3: Sample Output Inspection (1 min)
Look at actual generated text to see:
- Does it end abruptly?
- Is there content after where it should end?
- Are there malformed outputs?

### Step 4: Compare to SFT Behavior
Generate from same checkpoint in inference mode:
- Does it generate EOS there?
- If yes → GRPO generation config issue
- If no → SFT checkpoint issue

## Quick Fix Options

### If H1 (token ID wrong):
```python
# In buffer.py, explicitly set:
eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
```

### If H2 (suppressed):
```python
# Remove any suppress_tokens, ensure eos_token_id is set
gen_kwargs = {
    "eos_token_id": tokenizer.eos_token_id,
    # Remove suppress_tokens if present
}
```

### If H3 (cap too restrictive):
```yaml
# Increase max_cap or disable dynamic length temporarily
dynamic_length:
  enabled: false  # Test without it
```

### If H4 (model issue):
Switch to different checkpoint or retrain SFT with EOS

## Next Action

Run Step 1 verification NOW to identify root cause.
