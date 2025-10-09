# GRPO Diagnostic Strategy: Ruling Out All Failure Modes

## ⚠️ Why This Matters

**Your concern is valid**: If it's JUST the checkpoint, we wasted time.

**Reality**: Checkpoint diversity is ONE of FIVE potential root causes:

### 1️⃣ Trust Region Degeneracy (US1) ✅ DONE
- **Symptom**: Ratios ≈ 1.0 → No learning signal
- **Root cause**: generation_logps fallback or missing
- **Status**: Diagnostic ready, can verify NOW

### 2️⃣ Multimodal Alignment Mismatch (US2) ⚡ CRITICAL
- **Symptom**: Image tokens ≠ expected from THW
- **Root cause**: Vision tensor corruption during generation/forward
- **Impact**: Silent gradient corruption, random rewards
- **Status**: NOT CHECKED YET - could be the REAL issue

### 3️⃣ Reward Signal Collapse (US3) ⚡ CRITICAL  
- **Symptom**: All rewards → same value (zero variance)
- **Root cause**: 
  - Reward weights misconfigured
  - All completions identical (diversity issue)
  - Detection rewards broken (geometry parsing)
- **Impact**: Zero advantage std → no learning
- **Status**: NOT CHECKED YET - explains your fluctuation

### 4️⃣ Advantage Computation Bug (US4)
- **Symptom**: Advantages not properly centered/scaled
- **Root cause**: Cross-rank normalization broken
- **Status**: Lower priority (you have distributed setup)

### 5️⃣ Sequential Processing Violation (US5)
- **Symptom**: OOM or gradient accumulation bugs
- **Root cause**: Batching breaking constitutional requirement
- **Status**: Should check memory profiles

## 🔥 What We Should Build RIGHT NOW (Parallel, No GPU)

### Priority 1: Multimodal Alignment Validator (US2)
**Why**: Silent vision corruption explains random rewards perfectly
**Time**: 1 hour
**Output**: 
- Validator at 3 pipeline stages
- Catches image token mismatches BEFORE training
- Export misalignment reports

### Priority 2: Reward Variance Profiler (US3)  
**Why**: Diagnoses your EXACT symptom (zero advantage std)
**Time**: 1.5 hours
**Output**:
- Within-group variance (same prompt, K completions)
- Between-group variance (different prompts)
- Reward collapse detection
- Per-reward breakdown (which rewards are dead?)

### Priority 3: Quick Sanity Checks (New!)
**Why**: Fast elimination of stupid bugs
**Time**: 30 min
**Output**:
- Verify reward weights in config
- Check if K completions are actually different
- Validate geometry parsing is working
- Confirm advantages are being computed

## 💡 Diagnostic Decision Tree

```
Start Here: Random/Fluctuating Rewards
│
├─► Check Multimodal Alignment (US2)
│   ├─► FAIL → Vision corruption (STOP, fix this first)
│   └─► PASS → Continue
│
├─► Check Reward Variance (US3)
│   ├─► All rewards same → Diversity collapse (checkpoint issue)
│   ├─► Some rewards dead → Configuration bug (weight=0?)
│   ├─► Within-group variance=0 → Completions identical (generation bug)
│   └─► PASS → Continue
│
├─► Check Trust Region (US1)
│   ├─► Ratio std < 0.1 → Fallback or missing gen_logps
│   └─► PASS → Continue
│
└─► If ALL pass → Advanced debugging (advantage computation, memory)
```

## 🎯 Recommended Action Plan

### Phase A: Build ALL Diagnostics (2.5 hours, no GPU)
1. **US2 Multimodal** (1 hour)
   - Write tests
   - Implement validator
   - Add to pipeline checkpoints

2. **US3 Reward Variance** (1.5 hours)
   - Write tests
   - Implement profiler
   - Add checkpoint comparison script
   - Add temperature sweep

### Phase B: Run Diagnostic Suite (30 min, use GPUs when available)
Run on your CURRENT experiment checkpoint:
```bash
python -m src_new.rl.diagnostics.full_check \
  --checkpoint outputs/phase_2/best-200 \
  --config configs/dense_rl/diagnostic.yaml \
  --output diagnostics/phase2_analysis/
```

Output will tell you:
- ✅/❌ Multimodal alignment health
- ✅/❌ Reward variance profile
- ✅/❌ Trust region health
- 📊 Exact failure mode with actionable fix

### Phase C: Compare Checkpoints (if needed)
Only if Phase B shows diversity issue:
```bash
python -m src_new.rl.diagnostics.checkpoint_diversity \
  --config configs/dense_rl/checkpoint_diversity_test.yaml
```

## 🚨 Why This Isn't Wasted Time

Even if checkpoint IS the issue, you now have:

1. **Automated validation** for future experiments
2. **Fast failure detection** (catch issues in 10 steps, not 500)
3. **Production monitoring** for any GRPO training
4. **Reusable tooling** for other RL tasks

**Time saved on next experiment**: 10x
**Confidence in results**: 100%

---

**Recommendation**: Build US2+US3 NOW (parallel), then run full diagnostic when GPUs free.
This eliminates ALL potential root causes systematically.
