"""
Ready-to-integrate diagnostic code for GRPO Trainer

Feature: 004-grpo-post-training
Constitution: v4.1.1

INSERT THIS CODE into src_new/rl/grpo_trainer.py at marked locations.
"""

# ==============================================================================
# STEP 1: Add imports at top of file (after existing imports)
# ==============================================================================

from src_new.rl.diagnostics import (
    compute_multimodal_alignment,
    compute_reward_profile,
    compute_trust_region_diagnostic,
)

# ==============================================================================
# STEP 2: In train() method, after reward computation
# Location: After rewards_all tensor is computed
# ==============================================================================

# Diagnostic 1: Reward Variance Profile (every step)
if self._global_step % self.manual_cfg.logging_steps == 0:
    try:
        # Extract per-reward components if available
        reward_components_dict = {}
        if hasattr(self, '_last_reward_breakdown'):
            reward_components_dict = self._last_reward_breakdown
        
        reward_profile = compute_reward_profile(
            step=self._global_step,
            rewards=rewards_all,  # [batch_size * K] tensor
            per_reward_components=reward_components_dict,
            k_completions=self.manual_cfg.grpo_cfg.sample_k,
        )
        
        # Log warnings (will print to console if issues detected)
        reward_profile.log_warnings()
        
        # Log to TensorBoard
        self._tb_logger.log_scalars(
            reward_profile.to_tensorboard(),
            self._global_step
        )
    except Exception as e:
        logger.warning(f"[Diagnostic] Reward variance profiling failed: {e}")

# ==============================================================================
# STEP 3: In train() method, in the data loading loop
# Location: After batch is loaded from dataloader
# ==============================================================================

# Diagnostic 2: Multimodal Alignment Check (every 10 steps, first sample only)
if self._global_step % 10 == 0:
    try:
        mm_check = compute_multimodal_alignment(
            step=self._global_step,
            stage="dataset",
            sample_idx=0,
            input_ids=batch["input_ids"][0],  # First sample
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            tokenizer=self.tokenizer,
            merge_size=2,  # Qwen2.5-VL default
        )
        
        # Log warnings (critical errors will show immediately)
        mm_check.log_warnings()
        
        # Log to TensorBoard
        self._tb_logger.log_scalars(
            mm_check.to_tensorboard(),
            self._global_step
        )
        
        # Fail-fast: Stop training if multimodal alignment is broken
        if not mm_check.is_valid:
            logger.error(
                f"[Diagnostic] CRITICAL: Multimodal alignment check FAILED at step {self._global_step}. "
                f"Errors: {mm_check.errors}"
            )
            logger.error("[Diagnostic] Training aborted due to vision corruption.")
            raise RuntimeError("Multimodal alignment failure - vision tensors corrupted")
            
    except RuntimeError:
        raise  # Re-raise critical errors
    except Exception as e:
        logger.warning(f"[Diagnostic] Multimodal alignment check failed: {e}")

# ==============================================================================
# STEP 4: In loss computation loop, after computing policy ratios
# Location: After computing per-token ratios in loss_computer.compute_streaming_loss()
# ==============================================================================

# Diagnostic 3: Trust Region Validation (every step)
if self._global_step % self.manual_cfg.logging_steps == 0:
    try:
        # Compute ratios from current and generation logps
        # NOTE: This assumes you have access to these tensors - adapt variable names as needed
        trust_diag = compute_trust_region_diagnostic(
            step=self._global_step,
            ratios=ratios,  # exp(cur_logps - gen_logps) tensor
            generation_logps_present=(
                generation_result.get("generation_logps") is not None
            ),
            fallback_count=0,  # Track if fallback is used
        )
        
        # Log warnings
        trust_diag.log_warnings()
        
        # Log to TensorBoard
        self._tb_logger.log_scalars(
            trust_diag.to_tensorboard(),
            self._global_step
        )
        
        # Export histogram every 50 steps
        if self._global_step % 50 == 0 and self.accelerator.is_main_process:
            from pathlib import Path
            output_dir = Path(self.manual_cfg.output_dir)
            diagnostic_dir = output_dir / "diagnostics" / f"{self._global_step:06d}"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            
            trust_diag.export_histogram(
                ratios=ratios,
                output_path=diagnostic_dir / "trust_region_ratios.png"
            )
            
    except Exception as e:
        logger.warning(f"[Diagnostic] Trust region validation failed: {e}")

# ==============================================================================
# USAGE EXAMPLE (Full Integration)
# ==============================================================================

"""
After integration, your TensorBoard will show:

1. Multimodal Alignment:
   - multimodal/dataset/is_valid (should be 1.0)
   - multimodal/dataset/token_mismatch (should be 0.0)
   - multimodal/dataset/patches_match (should be 1.0)

2. Reward Variance:
   - reward_profile/within_group_std (should be > 0.01)
   - reward_profile/is_collapsed (should be 0.0)
   - reward_profile/num_dead_rewards (should be 0.0)
   - reward_components/*/mean (per-reward breakdown)

3. Trust Region:
   - trust_region/ratio_mean (should be ∈ [0.8, 1.5])
   - trust_region/ratio_std (should be > 0.1)
   - trust_region/is_degenerate (should be 0.0)

If ANY of these show problems, the diagnostic will:
- Print ERROR/WARNING to console
- Log to TensorBoard for historical tracking
- Abort training if critical (multimodal alignment)
"""
