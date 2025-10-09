# Temperature and EOS Generation Analysis

## Can Temperature Cause "Infinite" Generation?

### Short Answer
**No** - `max_new_tokens` prevents infinite generation.  
**But YES** - temperature can prevent EOS from being generated within the token limit!

## How Temperature Affects EOS

### Temperature = 0.0 (Greedy Decoding)
```python
# Always picks highest probability token
# Most likely to generate EOS if model learned it
# Deterministic output
```

### Temperature = 1.0 (Standard)
```python
# Samples from full probability distribution
# EOS has lower probability → less likely to be sampled
# More diverse but may miss EOS
```

### Temperature > 1.0 (e.g., 1.3, 1.5)
```python
# Flattens probability distribution
# Even LESS likely to sample EOS
# Very creative/random outputs
```

## Your Current Settings

From `standard.yaml` (lines 41-42):
```yaml
temperature: 1.0
top_p: 0.85
```

**With temperature=1.0 and do_sample=True**, the model:
1. Samples from probability distribution instead of taking argmax
2. EOS token might have lower probability than content tokens
3. Model keeps generating until hitting `max_new_tokens`

## Why This Matters for Your Issue

### Scenario 1: Model CAN generate EOS
- Temperature=0.0 → EOS appears naturally at ~800 tokens
- Temperature=1.0 → EOS might not be sampled until 1500+ tokens
- Temperature=1.5 → EOS almost never sampled

### Scenario 2: Model CANNOT generate EOS properly
- ANY temperature → No EOS
- This is what we're seeing: term_ratio=0.000 at ALL tested lengths

## Diagnostic: Does Temperature Matter?

### Test 1: Greedy Decoding (temperature=0.0)
```yaml
generation:
  max_new_tokens: 2048
  do_sample: false  # Forces greedy (ignores temperature)
```
If term_ratio > 0 → Temperature WAS the issue!

### Test 2: Lower Temperature
```yaml
generation:
  temperature: 0.1  # Very deterministic
  max_new_tokens: 2048
  do_sample: true
```
If term_ratio > 0 → High temperature was preventing EOS sampling

### Test 3: Your Current Settings
```yaml
generation:
  temperature: 1.0
  max_new_tokens: 2048
  do_sample: true
```
Already tested → term_ratio=0.000 (we'll confirm)

## GRPO Specific Concern

Your GRPO config uses:
```yaml
temperature: 1.0
sample_k: 8  # Generate 8 diverse completions
```

**Problem**: If temperature=1.0 prevents EOS from being sampled:
- All K completions hit max_new_tokens cap
- Rewards are inconsistent (truncation points vary)
- Training signal is weak

**Solution if temperature is the issue**:
1. Lower temperature to 0.3-0.5 during GRPO
2. Or increase max_new_tokens to 3072/4096
3. Or use scheduled temperature (start high, decay over training)

## Quick Test

Let me check if your recent test completed and what the term_ratio is...
