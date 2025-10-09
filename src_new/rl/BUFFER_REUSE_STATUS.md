# Swift-Style Buffer Reuse Implementation Status

## Overview

This document tracks the implementation status of Swift-style buffer reuse for the BBU GRPO trainer. Buffer reuse is an optimization technique where generated completions are stored and reused across multiple optimizer steps, significantly improving training efficiency.

## Motivation

From the Swift framework analysis, buffer reuse provides:
- **Generation efficiency**: Generate once every S steps (S× fewer generations)
- **Sample efficiency**: Each completion used S times (S× better sample utilization)
- **Training stability**: Same data for S consecutive updates reduces variance
- **Memory efficiency**: Store on CPU, stream to GPU one-by-one

## Implementation Status: ✅ Core Infrastructure Ready (Experimental)

### ✅ Completed Components

1. **Configuration Support** (`src_new/config/rl_config_v2.py`)
   - Added `steps_per_generation` field to `GRPOConfig`
   - Validation: must be >= 1
   - Default: `1` (maintains current behavior)

2. **Buffer Storage Class** (`src_new/rl/generation_buffer.py`)
   - `GenerationBuffer` dataclass for CPU-side storage
   - Stores: prompt/completion IDs, old log-probs (trust region!), advantages, rewards, vision tensors
   - Methods: `get_completion_for_step()`, `is_exhausted()`, `increment_step()`
   - Diagnostic accumulation for clipping metrics

3. **Generation Phase** (`src_new/rl/grpo_trainer.py::_generate_buffer()`)
   - Generates P prompts × K completions sequentially
   - Stores all tensors on CPU immediately after generation
   - Preserves existing cross-rank advantage logic
   - Returns `GenerationBuffer` or `None` on failure

4. **Optimization Phase** (`src_new/rl/grpo_trainer.py::_compute_loss_for_single_completion()`)
   - Streams ONE completion from CPU → GPU at a time
   - Computes forward pass and GRPO loss
   - Uses stored `generation_logps` for trust region (critical!)
   - Returns loss tensor + diagnostics

5. **Logging** (`src_new/rl/grpo_trainer.py::_add_supplementary_metrics_to_logs()`)
   - Added `buffer/steps_per_generation`
   - Added `buffer/reuse_count`
   - Added `buffer/generation_efficiency`

6. **Configuration Files**
   - `configs/dense_rl/standard.yaml`: `steps_per_generation: 1`
   - `configs/dense_rl/debug.yaml`: `steps_per_generation: 1`

### ⚠️ Pending Integration

**Main Training Loop Refactor** (`src_new/rl/grpo_trainer.py::train()`)
- Current loop uses `_buffer_chunks` mechanism (lines ~1356-2000+)
- Need to integrate new `_generate_buffer()` and `_compute_loss_for_single_completion()`
- Replace existing generation/forward loop with buffer reuse cycle
- Maintain backward compatibility with `steps_per_generation=1`

**Why Not Completed:**
- The existing training loop is complex (~650 lines) with many edge cases
- `_buffer_chunks` mechanism already does some batching/splitting
- Requires careful integration to preserve:
  - Cross-rank synchronization
  - Slow generation detection
  - Dynamic sampling
  - Telemetry and logging
  - Evaluation and checkpointing

## Usage (Current)

### Default Behavior (Maintains Status Quo)
```yaml
grpo:
  steps_per_generation: 1  # Current behavior: generate every step
```

### Experimental Buffer Reuse (NOT RECOMMENDED YET)
```yaml
grpo:
  steps_per_generation: 4  # EXPERIMENTAL: Reuse each generation 4 times
```

**Warning:** Setting `steps_per_generation > 1` will trigger a warning at runtime:
```
steps_per_generation=4 > 1: Buffer reuse is EXPERIMENTAL.
Core infrastructure is ready but full integration pending. Use with caution.
```

## Next Steps

1. **Complete Training Loop Integration**
   - Replace generation loop (lines ~1376-1520) with `_generate_buffer()` call
   - Replace forward/backward loop (lines ~1740-1950) with streaming from `GenerationBuffer`
   - Preserve all existing synchronization and error handling

2. **Testing**
   - Unit tests for `GenerationBuffer` get/increment logic
   - Integration test: `steps_per_generation=1` matches current behavior exactly
   - Integration test: `steps_per_generation=4` produces valid gradients

3. **Validation**
   - Compare training curves: `steps_per_generation=1` vs `=4`
   - Verify trust region ratios are correct (not ~1.0)
   - Check memory usage doesn't spike

4. **Documentation**
   - Update main RL docs with buffer reuse explanation
   - Add performance benchmarks (expected 2-3× speedup for `steps_per_generation=4`)

## Design Notes

### Memory Constraints
- **User constraint**: Only 1 completion fits in GPU memory at a time
- **Solution**: Store entire buffer on CPU, stream to GPU one-by-one
- **Trade-off**: CPU↔GPU transfer overhead vs generation efficiency

### Trust Region (Critical!)
- Buffer stores `generation_logps` from the policy that generated completions
- Forward pass computes `current_logps` from current policy
- GRPO ratio: `π_current / π_generation` (NOT `π_current / π_current`)
- This is why buffer reuse works: stale ratios > 1.0 prevent degenerate updates

### Swift Alignment
- Swift default: `steps_per_generation == gradient_accumulation_steps`
- BBU default: `steps_per_generation == 1` (until full integration)
- Both are valid; Swift's choice optimizes for efficiency

## References
- Swift GRPO implementation: `ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py`
- Swift config: `ms-swift/swift/trainers/rlhf_arguments.py::GRPOConfig`
- Plan document: `/wire.plan.md`
- RL config docs: `configs/dense_rl/README.md`

## Contact
For questions or to contribute to completing the integration, see the main GRPO trainer implementation in `src_new/rl/grpo_trainer.py`.

