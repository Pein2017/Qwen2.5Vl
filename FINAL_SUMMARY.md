# EOS Investigation: Final Summary & Next Steps

## What We've Discovered

### ✅ Confirmed Working:
1. **All GRPO reward components functional** (bbox_giou, caption_f1, coverage, etc.)
2. **EOS token correctly configured** (`<|im_end|>` = ID 151645)
3. **Generation code passes eos_token_id** properly
4. **8 GPUs working** in distributed mode

### ❌ The Problem:
**term_ratio = 0.000** across ALL tests:
- ✓ max_new_tokens=512 → No EOS
- ✓ max_new_tokens=2048 → No EOS  
- ✓ dynamic_length disabled → No EOS
- ✓ Tested with `do_sample: false` → **Still no EOS** (but config might not have taken effect)

## Root Cause Analysis

### Most Likely Scenarios (in order):

#### 1. Dense Captions Need >2048 Tokens (60% likelihood)
Your dense captioning task might genuinely need 3000-4000 tokens to finish:
- Many objects per image
- Detailed descriptions + geometry for each
- Current cap at 2048 may still be truncating

**Test**: Try `max_new_tokens: 4096`

#### 2. Temperature/Sampling Preventing EOS (30% likelihood)
With `temperature=1.0` and sampling:
- EOS token has low probability
- Model samples content tokens instead
- Never reaches EOS within token limit

**Test**: Ensure `do_sample: false` actually takes effect in generation

#### 3. SFT Checkpoint Issue (10% likelihood)
Model never properly learned to generate EOS during SFT.

**Test**: Compare Phase 2 vs Phase 3 checkpoints

## Immediate Action Plan

### Option A: Test with 4096 Tokens (RECOMMENDED)
```yaml
generation:
  max_new_tokens: 4096  # Double current length
  do_sample: false      # Greedy decoding
  dynamic_length:
    enabled: false
```

###  B: Verify do_sample is Being Used
Check `src_new/rl/generation.py` to ensure `do_sample` from config reaches `model.generate()`

### Option C: Compare Checkpoints
Use existing `test_checkpoint_comparison.sh` to test all 4 checkpoints

### Option D: Check Actual Completion Lengths
Parse logs to see distribution of completion lengths.
If ALL = 2048, confirms they're hitting cap.

## Key Insight

The fact that **NO checkpoint generates EOS at ANY tested length** suggests this is either:
1. A systematic issue with how generation is configured in GRPO
2. Dense captions genuinely need much longer sequences

**Your SFT training DOES include `<|im_end|>` in labels** (confirmed in code), so the model should have learned it.

## Recommendation

**Test with 4096 tokens + greedy decoding first.**  
This will definitively answer whether it's just a length issue.

```bash
# 1. Edit diagnostic.yaml:
#    max_new_tokens: 4096
#    do_sample: false

# 2. Run:
bash test_eos_fix.sh > test_4096tok_greedy.log 2>&1 &

# 3. Wait 2 minutes, then check:
grep "term_ratio" test_4096tok_greedy.log | tail -5
```

If term_ratio > 0: **Problem solved** - just needed longer generation!  
If still 0.000: We need to dig into the actual generated text to see what's happening.
