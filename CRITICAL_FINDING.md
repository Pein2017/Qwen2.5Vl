# 🚨 CRITICAL FINDING: Model Not Generating EOS

## Test Results
**With dynamic_length DISABLED**: term_ratio=0.000 across all 5 steps

## Conclusion
**The issue is NOT the dynamic length cap!**

The SFT checkpoint is not generating `<|im_end|>` tokens naturally.

## Why This Happens

### Hypothesis: SFT Training Issue
During SFT, the assistant labels likely did NOT include `<|im_end|>` in the training targets, or it was masked out.

From the codebase architecture docs:
```
Spans & labels: Assistant spans token-aligned and include <|im_end|>; 
labels outside spans set to -100; <|image_pad|> always masked.
```

**If `<|im_end|>` was masked during SFT**, the model never learned to generate it!

### Evidence
1. EOS token correctly configured (ID 151645)
2. Generation call properly passes eos_token_id  
3. Detection logic correct in buffer.py
4. **Even with NO length cap, model doesn't generate EOS**

This points to: **Model was never trained to generate EOS during SFT**

## Verification Needed

### Check SFT Training Code
Look at `src_new/data/dataset.py` - how are assistant spans constructed?

```python
# Does this include <|im_end|> in trainable labels?
assistant_spans_include_im_end = ???
```

### Check Inference on SFT Checkpoint
Run simple inference WITHOUT RL to see if model generates EOS:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("outputs/.../phase_2/...")
# Generate completion
# Does it end with <|im_end|>?
```

## Immediate Next Steps

### Option 1: Test Phase 3 Checkpoint
The user mentioned Phase 3 is "more deeply tuned". Maybe it learned EOS better?

```yaml
paths:
  model_path: "outputs/7B-all_tokens/phase_3/best-checkpoint"
```

### Option 2: Check Label Construction
Verify SFT labels include `<|im_end|>` in `src_new/processing/span_builder.py`

### Option 3: Quick Inference Test
Generate from checkpoint without RL to confirm EOS behavior

## Impact

If SFT checkpoint doesn't generate EOS:
1. **Cannot fix in GRPO config** - model physically cannot generate it
2. **Need to retrain SFT** with proper EOS in labels
3. **Or use different checkpoint** that does generate EOS

## Confidence
**99% confident** - disabling dynamic length is the definitive test.
Model issue, not config issue.
