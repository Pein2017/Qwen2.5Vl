# GRPO Post-Training Diagnostics: Findings & Resolution

**Feature**: 004-grpo-post-training  
**Date**: 2025-10-09  
**Status**: EOS Investigation Complete, Diagnostic Implementation In Progress  
**Constitution**: v4.1.1

---

## Executive Summary

Investigation into GRPO training instability initially suggested a critical EOS generation failure. After controlled single‑GPU tests reusing the exact GRPO preprocessing and `src_new/inference.py` with auto‑config, we discovered that both SFT Phase‑2 and Phase‑3 checkpoints DO generate `<|im_end|>` naturally under standard decoding (≤2048 tokens, temperature 0.7, sampling on). Therefore, the EOS failure observed in GRPO runs is NOT due to SFT capability but is emerging during RL/GRPO execution (generation loop, settings, or policy update effects).

**Updated Root Cause Hypothesis**: EOS termination ability exists in SFT checkpoints. The zero‑termination in GRPO arises from RL‑stage specifics (e.g., generation kwargs, prompt construction, masking/padding policy, or degradation of EOS likelihood during policy updates).

**Status**: 
- ✅ Diagnostic infrastructure built (US1: Trust Region, US2: Multimodal, US3: Rewards)
- ✅ EOS capability validated on SFT checkpoints (Phase‑2, Phase‑3)
- 🔍 Focusing diagnosis on RL/GRPO stage (buffer generation, ratio/trust‑region health, reward variance)

---

## Critical Finding: EOS Termination Behavior

### Updated Symptoms

```
GRPO runs: term_ratio ≈ 0.000, cap_hit_ratio ≈ 1.000
SFT (Phase‑2/3) single‑GPU inference: term_ratio = 1 for tested samples (EOS present)
```

Observed across multiple GRPO configurations (unchanged):
- ✗ `max_new_tokens=512` + `dynamic_length=true` → 0% EOS
- ✗ `max_new_tokens=2048` + `dynamic_length=false` → 0% EOS  
- ✗ `max_new_tokens=2048` + `do_sample=false` (greedy) → 0% EOS
- ⏳ `max_new_tokens=8192` + greedy (GRPO) → previously pending; SFT tests indicate EOS is possible without RL

### Hypothesis Evolution (Reconciled)

1. Dynamic length caps truncate before EOS
   - Disproven for GRPO alone; also SFT shows EOS without caps
2. 2048 tokens insufficient for dense captions
   - SFT shows EOS with ≤2048 generated tokens in tested samples
3. Current: EOS ability exists pre‑RL; GRPO stage prevents/erodes EOS
   - Candidates: generation kwargs mismatch, prompt differences, EOS masking in GRPO, reward shaping disincentivizing `<|im_end|>`, or trust‑region dynamics reducing EOS probability

### Verification Steps Taken (Added)

6. ✅ SFT Phase‑2 checkpoint test via `src_new/inference.py`: EOS observed
7. ✅ SFT Phase‑3 checkpoint test via `src_new/inference.py`: EOS observed
8. ✅ Inference bug fixes applied (method indentation and staticmethod call), parsing validated

### Implications for GRPO

**Why Training Breaks in RL stage**:
- Completions often hit `max_new_tokens` limits → truncated outputs
- Zero EOS → formatting rewards penalize; length distribution collapses → within‑group variance ≈ 0 → advantages ≈ 0
- Trust‑region may degenerate if generation log‑probs are mishandled

**Actionable Next Steps (prioritized)**:
1. Run inference on an actual RL checkpoint (post‑GRPO) to check EOS ability post‑updates
2. Cross‑check GRPO generation kwargs vs inference (temperature, sampling flags, repetition penalty, stop tokens)
3. Compare prompt construction (builder/template) parity between GRPO buffer and inference
4. Inspect GRPO masking (labels and generation) for accidental EOS suppression; ensure `<|im_end|>` not filtered
5. Verify trust‑region denominator uses stored generation‑policy log‑probs (avoid π_current/π_current)
6. Log EOS termination, cap‑hit, and completion length stats inside GRPO (already partially present); ensure TB keys populated

---

## Configuration Analysis: Critical Errors Found

### standard.yaml Issues (Fixed)

**Error 1**: `student_loss_weight: 0.0`
```yaml
# BEFORE (BROKEN)
loss:
  student_loss_weight: 0.0  # ❌ No gradient signal!
  
# AFTER (FIXED)
loss:
  student_loss_weight: 1.0  # ✅ Student learns
```

**Error 2**: Missing detection reward weights
```yaml
# BEFORE (BROKEN)
rewards:
  bbox_giou: 1.0
  quad_l1: 1.0
  # ...but no weights in rewards_config!

# AFTER (FIXED)  
rewards_config:
  bbox_giou_weight: 1.0
  quad_l1_weight: 1.0
  # ...explicit weights
```

**Error 3**: Insufficient generation length
```yaml
# BEFORE
generation:
  max_new_tokens: 2048  # Too short for dense captions
  temperature: 1.0      # Too high, adds noise
  
# AFTER
generation:
  max_new_tokens: 4096  # 2x increase
  temperature: 0.7      # Lower for stability
  dynamic_length:
    alpha: 2.0          # Allow 2x GT length
    max_cap: 4096       # Matches max_new_tokens
```

---

## Diagnostic Infrastructure Status

### Completed (User Story 1: Trust Region)

**Files Created**:
- `src_new/rl/diagnostics/trust_region.py`
- `src_new/rl/diagnostics/exporters.py`
- `tests/rl/diagnostics/test_trust_region.py`
- `tests/rl/diagnostics/test_exporters.py`

**Functionality**:
- ✅ `TrustRegionDiagnostic` dataclass
- ✅ Ratio statistics (mean, std, percentiles)
- ✅ Degeneracy detection (std < 0.1)
- ✅ TensorBoard logging
- ✅ Histogram export (PNG)
- ✅ All tests passing (18/18)

### Completed (User Story 2: Multimodal Alignment)

**Files Created**:
- `src_new/rl/diagnostics/multimodal.py`
- `tests/rl/diagnostics/test_multimodal.py`

**Functionality**:
- ✅ `MultimodalAlignmentCheck` dataclass
- ✅ Image token validation
- ✅ Pixel values row count validation
- ✅ THW shape validation
- ✅ Fail-fast error handling
- ✅ All tests passing (8/8)

### Completed (User Story 3: Reward Variance)

**Files Created**:
- `src_new/rl/diagnostics/rewards.py`
- `tests/rl/diagnostics/test_reward_variance.py`

**Functionality**:
- ✅ `RewardProfile` dataclass
- ✅ Within-group variance tracking
- ✅ Between-group variance tracking
- ✅ Diversity ratio computation
- ✅ Collapse detection
- ✅ All tests passing (10/10)

### Config Validator

**File**: `src_new/rl/diagnostics/config_validator.py`

**Checks Performed**:
- ✅ Reward weights validation
- ✅ Detection vs formatting reward balance
- ✅ Training parameter sanity
- ✅ File path existence
- ✅ Model settings compatibility

**Critical Finding**: Identified `student_loss_weight=0` and missing reward weights in user's config

### Checkpoint Diversity Tool (Not Yet Used)

**File**: `src_new/rl/diagnostics/checkpoint_diversity.py`

**Purpose**: Compare Phase 2 vs Phase 3 SFT checkpoints for generation diversity

**Status**: ⏳ Pending EOS resolution (diversity comparison meaningless if all completions truncated)

---

## Experiments Conducted

### Experiment 1: Dynamic Length Caps (FAILED)

**Config**: `diagnostic.yaml` with `dynamic_length: false`, `max_new_tokens: 2048`

**Results**:
```
term_ratio: 0.000
cap_hit_ratio: 1.000
reward/mean: -5.2 → -4.8 → -5.5 (unstable)
```

**Conclusion**: Dynamic length not the root cause

### Experiment 2: Greedy Decoding (FAILED)

**Config**: `do_sample: false` (temperature ignored), `max_new_tokens: 2048`

**Results**:
```
term_ratio: 0.000  (still!)
temperature: 1.0000  (config override not working)
```

**Conclusion**: 
- Greedy decoding didn't help
- Also discovered `do_sample` config might not be passed correctly to generation

### Experiment 3: Extreme Length Test (IN PROGRESS)

**Config**: 
```yaml
generation:
  max_new_tokens: 8192  # 4x original
  do_sample: false      # Greedy
  dynamic_length:
    enabled: false      # No caps
```

**Hypothesis**: Dense captions genuinely need >4000 tokens to complete

**Status**: ⏳ Running on 8 GPUs via `launch_diagnostic.sh`

**Expected Results**:
- **If `term_ratio > 0`**: Model CAN generate EOS, just needs more tokens → increase limits
- **If `term_ratio = 0`**: Model CANNOT generate EOS → fundamental capability issue

---

## Tools Created

### launch_diagnostic.sh

```bash
#!/bin/bash
# Quick diagnostic launcher for EOS testing

pkill -9 -f "runner.py" 2>/dev/null
sleep 2

/root/miniconda3/envs/ms/bin/accelerate launch --num_processes 8 \
  src_new/rl/runner.py \
  --config configs/dense_rl/diagnostic.yaml \
  --mode train
```

**Purpose**: One-command diagnostic runs with automatic cleanup

---

## Lessons Learned

### 1. SFT Training Must Enforce EOS

**Issue**: Model never learned to generate `<|im_end|>` for dense captions

**Why**: Dense captions are so long that SFT training rarely reaches EOS token in practice

**Fix Options**:
- Add explicit EOS reward in GRPO
- Retrain SFT with explicit termination supervision
- Accept truncation and adjust reward functions

### 2. Config Validation is Critical

**Issue**: User's config had `student_loss_weight=0`, breaking all training

**Why**: No validation at config load time

**Fix**: Implemented `GRPOConfigValidator` with comprehensive checks

### 3. Generation Parameters Need Verification

**Issue**: `do_sample: false` in config didn't disable sampling (logs showed `temperature=1.0`)

**Why**: Config might not be passed through to `model.generate()` correctly

**Fix**: Add instrumentation to verify generation kwargs at runtime

### 4. Diversity Collapse Diagnosis Requires Functional Model

**Issue**: Cannot measure diversity if all completions are truncated

**Why**: Checkpoint comparison is meaningless when 100% hit length cap

**Fix**: Resolve EOS issue first, then compare checkpoints

---

## Next Steps

### Immediate (Priority 1)

1. ✅ **Launch extreme length test** (`8192 max_tokens`)
2. ⏳ **Monitor `term_ratio`** in logs
3. 📋 **Decision Point**:
   - If `term_ratio > 0` → Increase default limits, proceed with GRPO
   - If `term_ratio = 0` → Add EOS reward or accept truncation

### Short-Term (Priority 2)

4. **Compare Phase 2 vs Phase 3 checkpoints** for diversity (after EOS resolved)
5. **Integrate diagnostic hooks** into `grpo_trainer.py` (US1, US2, US3)
6. **Run full diagnostic suite** on 50 steps with all logging enabled

### Long-Term (Priority 3)

7. **Implement remaining user stories** (US4: Gradients, US5: Sequential)
8. **Add EOS supervision** to GRPO via custom reward
9. **Retrain SFT** with explicit termination objectives

---

## Files to Preserve

### Spec Directory (KEEP ALL)
```
specs/004-grpo-post-training/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan
├── research.md          # Phase 0: GRPO diversity analysis
├── data-model.md        # Phase 1: Diagnostic entities
├── quickstart.md        # Usage guide
├── tasks.md             # Task breakdown
├── findings.md          # THIS FILE
└── contracts/           # Validation interfaces
```

### Source Code (KEEP)
```
src_new/rl/diagnostics/
├── __init__.py
├── trust_region.py
├── multimodal.py
├── rewards.py
├── checkpoint_diversity.py
├── config_validator.py
├── exporters.py
└── (contracts copied here)

tests/rl/diagnostics/
├── conftest.py
├── test_exporters.py
├── test_trust_region.py
├── test_multimodal.py
└── test_reward_variance.py
```

### Configs (KEEP)
```
configs/dense_rl/
├── diagnostic.yaml      # Minimal test config
└── standard.yaml        # Main GRPO config (FIXED)
```

### Scripts (KEEP ONLY)
```
launch_diagnostic.sh     # Diagnostic launcher
```

---

## Files to Remove

### Root Directory Cleanup

**Temporary Analysis Documents** (37 files):
- All `*.md` except `README.md`, `AGENTS.md`, `CLAUDE.md`
- All `*.log` files
- Temporary `*.sh` scripts except `launch_diagnostic.sh`

**Rationale**: All findings consolidated into `specs/004-grpo-post-training/findings.md`

---

## Key Metrics to Monitor

### EOS Health
```
generation/term_ratio      > 0.8   (80%+ completions end naturally)
generation/cap_hit_ratio   < 0.3   (30%- completions truncated)
generation/mean_length     ∈ [0.8×GT, 1.5×GT]
```

### Trust Region Health
```
trust_region/ratio_mean    ∈ [0.8, 1.5]
trust_region/ratio_std     > 0.1
trust_region/is_degenerate = 0.0
trust_region/fallback_count = 0
```

### Reward Diversity
```
rewards/*/within_group_std > 0.05  (for ≥60% of prompts)
rewards/*/diversity_ratio  > 0.1
rewards/*/is_collapsed     = 0.0
```

### Training Progress
```
reward/mean                Increasing over time
advantages/std             > 0.5
loss/grpo                  Decreasing over time
```

---

## References

- Constitution: `/data3/Qwen2.5-VL-main/.specify/memory/constitution.md` (v4.1.1)
- SFT Architecture: `/data3/Qwen2.5-VL-main/src_new/UNIFIED_DOCUMENTATION.md`
- GRPO Docs: `/data3/Qwen2.5-VL-main/src_new/rl/grpo_readme.md`
- Config System: `/data3/Qwen2.5-VL-main/configs/README.md`

---

**Last Updated**: 2025-10-09 (Extreme length test launched)  
**Next Review**: After 8192-token test results available
