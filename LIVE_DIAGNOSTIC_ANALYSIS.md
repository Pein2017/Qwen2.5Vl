# Live Diagnostic Analysis

## Summary from 5-step run

### ✅ WORKING:
- 8 GPUs active (world_size=8)
- All reward components non-zero
- Advantages being computed (adv_std=1.0, adv_max~2.0)
- Rewards changing: 2.17 → 2.00 → 2.43

### 🚨 ISSUES FOUND:

#### 1. EOS Termination = 0% (CRITICAL)
```
term_ratio=0.000 across all steps
```
**Meaning**: No completions end with `<|im_end|>` - all hit max_tokens cap

**Possible causes**:
- EOS token not in vocabulary
- Model never learned to generate EOS
- EOS suppressed during generation
- Dynamic length cap too restrictive

**Impact**: 
- Completions may be truncated mid-sentence
- Reward for "proper ending" always fails
- Could explain oscillation if truncation point varies

#### 2. Reward Variance Collapsing
```
Step 1: ±0.36 (good diversity)
Step 3: ±0.15 (dropping)
Step 5: ±0.11 (low)
```

**Meaning**: Completions becoming more similar over time

**Possible causes**:
- Small dataset (only 10 samples) → overfitting fast
- Temperature=1.0 not high enough for exploration
- Policy collapsing to mode

#### 3. Reward Components Analysis
**High performers** (>0.9):
- geometry_sanity: 1.000 (perfect)
- wrappers: 1.000 (perfect)
- coords: 0.968-1.000 (near perfect)
- vocab: 0.938-1.000 (near perfect)
- quad_l1: 0.922-0.953 (very good)

**Low performers** (<0.5):
- line_giou: 0.055-0.318 (terrible)
- line_l1: 0.237-0.951 (inconsistent)
- quad_giou: 0.234-0.872 (inconsistent)
- caption_f1: 0.341-0.937 (inconsistent)

**Interpretation**:
- Model good at **formatting** (wrappers, coords, separators)
- Model good at **quad bounding boxes**
- Model **terrible at lines** (line_giou very low)
- **Caption quality varies wildly** (0.34 → 0.94)

This explains the oscillation:
- Formatting rewards are stable (~1.0)
- But detection/caption rewards vary a lot
- Overall reward oscillates between samples

### 🎯 Recommended Next Steps

#### Option A: Check EOS Issue (5 min)
Look at actual generated text to see if completions are proper or truncated:
```bash
# Check TensorBoard text logs
tensorboard --logdir tb_dense/quick_diagnostic_phase2
```

#### Option B: Run Longer (10 more steps)
See if variance continues to collapse or stabilizes:
- If variance → 0: Diversity problem confirmed
- If variance stays ~0.1: Might be normal for small dataset

#### Option C: Check Per-Sample Rewards
Export reward breakdown to see which samples get high vs low rewards

#### Option D: Test Different Temperature
Rerun with temp=1.3 to see if diversity improves

## Conclusion

**This is NOT a config error!** 

All rewards are active and working. The main issues are:

1. **EOS termination failing** (0% across all steps)
2. **Line detection very poor** (line_giou < 0.32)
3. **Reward variance collapsing** (but might be normal for 10 samples)

The flat curve in your long run might be because:
- Small reward variance on this tiny dataset
- EOS issue reducing signal
- Line detection consistently poor

**Recommend**: Run with more samples (50-100) and higher temperature to see real behavior.
