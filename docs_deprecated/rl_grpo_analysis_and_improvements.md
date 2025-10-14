# GRPO Training Analysis & Improvement Recommendations

**Project:** Qwen2.5-VL Dense Caption Detection  
**Date:** October 4, 2025  
**Focus:** Training Accuracy & Stability Improvements  
**Revision:** 2.0 (corrected after TRL GRPOTrainer review)

---

## Executive Summary

This document analyzes the current `src_new/rl/` implementation (which inherits from TRL's `GRPOTrainer`) against the reference Qwen2-VL-Finetune implementation to identify opportunities for improving training accuracy and stability in dense captioning detection tasks.

> **Note (2025 update):** Since this review, the TRL-based `VisionGRPOTrainer` has been fully replaced by the manual `BBUGRPOTrainer`. The analysis below is preserved for historical context; implementation-specific observations should be mapped to the new trainer where applicable.

**Key Findings:**
- ✅ **CORRECTION**: VisionGRPOTrainer correctly inherits RepeatSampler, multi-iteration buffering, and detailed clipping metrics from TRL's GRPOTrainer
- ✅ **Strengths**: Superior multimodal handling, fail-fast validations, vision-aware buffering, correct GRPO algorithm implementation
- ⚠️ **Improvement Areas**: Temperature scheduling, reward standardization, advantage clipping, and enhanced monitoring
- 🎯 **Priority**: Focus on stability enhancements and quality-of-life improvements, not core algorithm fixes

---

## 1. Architecture Comparison

### 1.1 Inheritance Chain

```
VisionGRPOTrainer (src_new/rl/trainer.py)
    ↓ inherits
GRPOTrainer (TRL package)
    ↓ inherits
Trainer (HuggingFace Transformers)
```

**What VisionGRPOTrainer inherits from TRL GRPOTrainer:**
- ✅ RepeatSampler for grouped prompt distribution (lines 86-182 in TRL)
- ✅ Custom get_train_dataloader with steps_per_generation (lines 746-777 in TRL)
- ✅ _get_train_sampler with RepeatSampler (lines 779-813 in TRL)
- ✅ Multi-iteration buffering in _prepare_inputs (lines 969-999 in TRL)
- ✅ Detailed clipping metrics (low/high/region) (lines 1420-1436 in TRL)
- ✅ KL divergence computation
- ✅ Multiple loss types (grpo, bnpo, dr_grpo)

**What VisionGRPOTrainer overrides:**
- ✅ `_generate_and_score_completions`: Per-sample generation with vision tensor tracking
- ✅ `_prepare_inputs`: Vision-aware buffering (preserves sample order for vision slicing)
- ✅ `_compute_loss`: Passes vision tensors to log probability computation
- ✅ `_get_per_token_logps`: Custom batched processing with vision support
- ✅ Checkpoint saving: Uses custom CheckpointSaver

---

## 2. Critical Clarifications

### 2.1 RepeatSampler ✅ **ALREADY IMPLEMENTED**

**Status**: VisionGRPOTrainer does NOT override `_get_train_sampler` or `get_train_dataloader`, so it correctly inherits TRL's RepeatSampler implementation.

**How it works:**
```python
# From TRL GRPOTrainer (inherited)
def _get_train_sampler(self, dataset=None) -> Sampler:
    return RepeatSampler(
        data_source=dataset,
        mini_repeat_count=self.num_generations,
        batch_size=self.args.generation_batch_size // self.num_generations,
        repeat_count=self.num_iterations * self.args.steps_per_generation,
        shuffle=self.shuffle_dataset,
        seed=self.args.seed,
    )
```

**Impact**: ✅ Correct reward normalization across GPUs is already working.

---

### 2.2 Multi-Iteration Buffering ✅ **ALREADY IMPLEMENTED**

**Status**: VisionGRPOTrainer overrides `_prepare_inputs` with its own vision-aware buffering that preserves the multi-iteration logic.

**Implementation** (src_new/rl/trainer.py:1247-1382):
```python
def _prepare_inputs(self, generation_batch):
    mode = "train" if self.model.training else "eval"
    if mode == "train":
        generate_every = steps_per_generation * self.num_iterations
        
        if (self._step % generate_every == 0) or (self._buffered_inputs is None):
            # Generate once
            gen = self._generate_and_score_completions(generation_batch)
            # Split into chunks (vision-aware slicing)
            chunks = [...]
            self._buffered_inputs = chunks
        
        # Return buffered chunk for this step
        inputs = self._buffered_inputs[self._step % steps_per_generation]
        self._step += 1
        return inputs
```

**Key Difference from TRL**: 
- TRL uses `shuffle_tensor_dict` + `split_tensor_dict` 
- VisionGRPOTrainer preserves order and manually slices vision tensors by image/patch offsets
- This is **intentional** to correctly handle packed 2D vision tensors

**Impact**: ✅ Multi-iteration optimization is working correctly.

---

### 2.3 Detailed Clipping Metrics ✅ **ALREADY IMPLEMENTED**

**Status**: VisionGRPOTrainer does NOT override the clipping metric logging, so it inherits TRL's detailed metrics.

**From TRL GRPOTrainer** (lines 1420-1436):
```python
is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)

self._metrics[mode]["clip_ratio/low_mean"].append(...)
self._metrics[mode]["clip_ratio/high_mean"].append(...)
self._metrics[mode]["clip_ratio/region_mean"].append(...)
```

**Impact**: ✅ Comprehensive clipping monitoring is already available.

---

### 2.4 Per-Sample Generation 🎯 **INTENTIONAL DESIGN CHOICE**

**Status**: VisionGRPOTrainer generates completions one sample at a time instead of batched.

**Implementation** (src_new/rl/trainer.py:530-631):
```python
# Per-sample generation to keep forward batch size = 1
sequences_list: List[torch.Tensor] = []
for i in range(len(ids_list)):
    ids_i = ids_list[i].to(device).unsqueeze(0)
    mask_i = mask_list[i].to(device).unsqueeze(0)
    pv_i = pixel_values_list[i]  # Vision tensors
    grid_i = image_grid_thw_list[i]
    
    seq_i = unwrapped_model.generate(
        input_ids=ids_i,
        attention_mask=mask_i,
        pixel_values=pv_i,
        image_grid_thw=grid_i,
        generation_config=self.generation_config
    )
    sequences_list.append(seq_i)
```

**Rationale**:
- Vision models with high-resolution images can OOM with batch generation
- Ensures consistent memory usage per sample
- Allows tracking vision tensors per generation

**Trade-off**: Slower generation, but more stable and memory-efficient for large images.

---

## 3. What's Actually Missing or Could Be Improved

### Priority 1: STABILITY ENHANCEMENTS ⚠️

#### 3.1 Advantage Clipping ⭐ **NEW**

**Problem**: Extreme advantages can destabilize training, especially in detection tasks where rewards can vary widely.

**Solution**:
```python
# In _generate_and_score_completions (after computing advantages)
advantages = rewards - mean_grouped_rewards
if self.scale_rewards:
    advantages = advantages / (std_grouped_rewards + 1e-4)

# Add advantage clipping
if hasattr(self.args, "max_advantage_magnitude") and self.args.max_advantage_magnitude > 0:
    advantages = torch.clamp(
        advantages, 
        -self.args.max_advantage_magnitude, 
        self.args.max_advantage_magnitude
    )
```

**Config Addition**:
```yaml
grpo:
  max_advantage_magnitude: 5.0  # Clip to [-5, 5]
```

**Expected Impact:**
- ✅ More stable policy updates
- ✅ Prevents extreme gradient spikes
- ✅ Better convergence in early training

**Implementation Location**: `src_new/rl/trainer.py` in `_generate_and_score_completions`

---

#### 3.2 KL Divergence Adaptive Scheduling ⭐ **NEW**

**Problem**: Fixed β (KL coefficient) may be suboptimal across training phases.

**Solution**:
```python
# src_new/rl/trainer.py
class VisionGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_beta = self.beta
        self._target_kl = getattr(self.args, "target_kl", 0.01)
        self._beta_schedule = getattr(self.args, "beta_schedule", "constant")
        
    def _get_adaptive_beta(self):
        """Adjust beta based on recent KL divergence."""
        if self._beta_schedule == "constant":
            return self.beta
        
        elif self._beta_schedule == "adaptive":
            # Get recent KL values (last 10 steps)
            if "kl" in self._metrics["train"] and len(self._metrics["train"]["kl"]) >= 3:
                recent_kl = self._metrics["train"]["kl"][-10:]
                avg_kl = sum(recent_kl) / len(recent_kl)
                
                # Increase beta if KL too high, decrease if too low
                if avg_kl > self._target_kl * 1.5:
                    self.beta = min(self.beta * 1.05, self._initial_beta * 5.0)
                elif avg_kl < self._target_kl * 0.5:
                    self.beta = max(self.beta * 0.95, self._initial_beta * 0.1)
                
                self._metrics["train"]["beta_value"].append(self.beta)
        
        return self.beta
    
    # Call before computing loss
    def _compute_loss(self, model, inputs):
        if self._beta_schedule == "adaptive":
            self._get_adaptive_beta()
        return super()._compute_loss(model, inputs)
```

**Config Addition**:
```yaml
grpo:
  beta: 0.01
  beta_schedule: "adaptive"  # or "constant"
  target_kl: 0.01
```

**Expected Impact:**
- ✅ Automatic adjustment to prevent KL divergence drift
- ✅ Better balance between policy improvement and stability

---

### Priority 2: QUALITY IMPROVEMENTS 📊

#### 3.3 Temperature Scheduling ⭐ **NEW**

**Problem**: Fixed temperature doesn't adapt to training progress.

**Solution**:
```python
# src_new/rl/trainer.py
class VisionGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_temperature = self.temperature
        self._temperature_schedule = getattr(kwargs.get("config"), "temperature_schedule", "constant")
        
    def _get_current_temperature(self):
        """Compute temperature based on training progress."""
        if self._temperature_schedule == "constant":
            return self._initial_temperature
        
        progress = self.state.global_step / max(self.state.max_steps, 1)
        
        if self._temperature_schedule == "linear_decay":
            # Decay from initial to 0.7x over training
            return self._initial_temperature * (1.0 - 0.3 * progress)
        
        elif self._temperature_schedule == "cosine":
            import math
            # Cosine annealing from initial to 0.5x
            return self._initial_temperature * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        return self._initial_temperature
    
    def _generate_and_score_completions(self, inputs):
        # Override temperature
        original_temp = self.temperature
        self.temperature = self._get_current_temperature()
        
        # Update generation_config if exists
        if hasattr(self, "generation_config") and self.generation_config is not None:
            self.generation_config.temperature = self.temperature
        
        result = super()._generate_and_score_completions(inputs)
        
        # Restore (though it will change next call anyway)
        self.temperature = original_temp
        return result
```

**Config Addition**:
```yaml
grpo:
  temperature: 0.9
  temperature_schedule: "cosine"  # constant, linear_decay, or cosine
```

**Expected Impact:**
- ✅ Better exploration early (high temp) → exploitation late (low temp)
- ✅ More focused generation as training progresses

---

#### 3.4 Reward Standardization ⭐ **NEW**

**Problem**: Different reward components (formatting, geometry, vocab) have different scales, causing some to dominate.

**Solution**:
```python
# src_new/rl/rewards/standardizer.py (NEW FILE)
import torch
from typing import Dict

class RewardStandardizer:
    """Standardizes rewards to zero mean and unit variance using running statistics."""
    
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean: Dict[str, torch.Tensor] = {}
        self.running_var: Dict[str, torch.Tensor] = {}
        self.running_count: Dict[str, int] = {}
        
    def update_and_standardize(
        self, reward_name: str, rewards: torch.Tensor
    ) -> torch.Tensor:
        """Update running stats and return standardized rewards."""
        # Compute batch statistics
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False) + self.epsilon
        
        # Initialize or update running stats
        if reward_name not in self.running_mean:
            self.running_mean[reward_name] = batch_mean.detach()
            self.running_var[reward_name] = batch_var.detach()
            self.running_count[reward_name] = rewards.numel()
        else:
            self.running_mean[reward_name] = (
                self.momentum * self.running_mean[reward_name]
                + (1 - self.momentum) * batch_mean.detach()
            )
            self.running_var[reward_name] = (
                self.momentum * self.running_var[reward_name]
                + (1 - self.momentum) * batch_var.detach()
            )
            self.running_count[reward_name] += rewards.numel()
        
        # Standardize using running stats
        standardized = (rewards - self.running_mean[reward_name]) / torch.sqrt(
            self.running_var[reward_name]
        )
        
        return standardized
    
    def get_stats(self, reward_name: str) -> Dict[str, float]:
        """Get current running statistics for a reward."""
        if reward_name not in self.running_mean:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        return {
            "mean": self.running_mean[reward_name].item(),
            "std": torch.sqrt(self.running_var[reward_name]).item(),
            "count": self.running_count[reward_name],
        }
```

**Integration in Trainer**:
```python
# src_new/rl/trainer.py
from src_new.rl.rewards.standardizer import RewardStandardizer

class VisionGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        use_standardization = getattr(self.args, "standardize_rewards", False)
        self._reward_standardizer = RewardStandardizer() if use_standardization else None
    
    # In _calculate_rewards method (if overriding):
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        
        # Apply standardization if enabled
        if self._reward_standardizer is not None:
            for i, reward_name in enumerate(self.reward_func_names):
                rewards_per_func[:, i] = self._reward_standardizer.update_and_standardize(
                    reward_name, rewards_per_func[:, i]
                )
        
        return rewards_per_func
```

**Config Addition**:
```yaml
grpo:
  standardize_rewards: true
```

**Expected Impact:**
- ✅ Balanced contribution from all reward components
- ✅ More stable training dynamics
- ✅ Better reward composition

---

### Priority 3: MONITORING ENHANCEMENTS 📈

#### 3.5 Generation Quality Diagnostics ⭐ **NEW**

**Problem**: Limited visibility into what the model is actually generating.

**Solution**:
```python
# src_new/rl/trainer.py
from src_new.processing.parse_generated import parse_generated_objects

class VisionGRPOTrainer(GRPOTrainer):
    def _log_generation_diagnostics(
        self, completions_text: List[str], meta: List[Dict[str, Any]], mode: str
    ):
        """Log detailed generation quality metrics."""
        parsed_counts = []
        parse_successes = 0
        
        for text in completions_text:
            try:
                objects = parse_generated_objects(text)
                parsed_counts.append(len(objects))
                if len(objects) > 0:
                    parse_successes += 1
            except Exception:
                parsed_counts.append(0)
        
        # Average objects per generation
        avg_objects = sum(parsed_counts) / len(parsed_counts) if parsed_counts else 0
        self._metrics[mode]["generation/avg_objects_per_sample"].append(avg_objects)
        
        # Parse success rate
        parse_rate = parse_successes / len(completions_text) if completions_text else 0
        self._metrics[mode]["generation/parse_success_rate"].append(parse_rate)
        
        # Recall proxy (if ground truth available)
        if meta and "num_objects" in meta[0]:
            gt_counts = [m.get("num_objects", 0) for m in meta]
            recall_proxy = sum(
                min(p, g) for p, g in zip(parsed_counts, gt_counts)
            ) / max(sum(gt_counts), 1)
            self._metrics[mode]["generation/recall_proxy"].append(recall_proxy)
    
    # Call in _generate_and_score_completions after decoding
```

**Expected Impact:**
- ✅ Real-time tracking of generation quality
- ✅ Early detection of formatting issues
- ✅ Better understanding of training progress

---

#### 3.6 Gradient Norm Monitoring ⭐ **NEW**

**Problem**: No visibility into gradient magnitudes across different model components.

**Solution**:
```python
# src_new/rl/trainer.py
class VisionGRPOTrainer(GRPOTrainer):
    def training_step(self, model, inputs):
        """Override to add gradient norm logging."""
        loss = super().training_step(model, inputs)
        
        # Log gradient norms every logging_steps
        if self.state.global_step % self.args.logging_steps == 0:
            grad_norms = {"total": 0.0, "vision": 0.0, "merger": 0.0, "llm": 0.0}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm_sq = param.grad.data.norm(2).item() ** 2
                    grad_norms["total"] += param_norm_sq
                    
                    if "visual" in name:
                        grad_norms["vision"] += param_norm_sq
                    elif "merger" in name:
                        grad_norms["merger"] += param_norm_sq
                    elif "model.layers" in name:
                        grad_norms["llm"] += param_norm_sq
            
            # Take square root for final norms
            for key in grad_norms:
                grad_norms[key] = grad_norms[key] ** 0.5
                self._metrics["train"][f"gradient_norm/{key}"].append(grad_norms[key])
        
        return loss
```

**Expected Impact:**
- ✅ Detection of gradient explosion/vanishing
- ✅ Better understanding of layer-wise learning
- ✅ Informed learning rate adjustments

---

## 4. Updated Configuration Recommendations

### Recommended GRPO Config for Dense Caption Detection:

```yaml
# configs/rl/dense_caption_grpo_v2.yaml
extends:
  - configs/rl/base_grpo.yaml

model_path: /path/to/sft/checkpoint
train_data_path: /path/to/train.jsonl
val_data_path: /path/to/val.jsonl
data_root: /path/to/data

model:
  attn_implementation: "eager"
  torch_dtype: "bfloat16"
  image_max_pixels: 27525120

# GRPO algorithm parameters
grpo:
  sample_k: 4  # num_generations
  num_iterations: 2  # μ in GRPO paper (already working via TRL)
  max_new_tokens: 512
  
  # Sampling parameters
  temperature: 0.9
  temperature_schedule: "cosine"  # ⭐ NEW: anneal temperature
  top_p: 0.95
  repetition_penalty: 1.05
  
  # GRPO clipping (inherited from TRL)
  epsilon_low: 0.2
  epsilon_high: 0.2
  
  # KL divergence
  beta: 0.01
  beta_schedule: "adaptive"  # ⭐ NEW: adaptive KL coefficient
  target_kl: 0.01
  
  # Loss type (inherited from TRL)
  loss_type: "dr_grpo"
  
  # Reward & advantage
  scale_rewards: true
  max_advantage_magnitude: 5.0  # ⭐ NEW: clip advantages
  standardize_rewards: true  # ⭐ NEW: standardize reward components
  mask_truncated_completions: true

# Training parameters
training:
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  steps_per_generation: 4  # Already supported via TRL
  max_steps: 10000
  
  # Learning rate
  learning_rate: 5.0e-6
  lr_scheduler_type: "cosine"
  warmup_steps: 500
  
  # Optimization
  gradient_checkpointing: true
  max_grad_norm: 1.0
  
  # Precision
  bf16: true
  fp16: false

# Optimizer
optimizer:
  type: "adamw"
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.01

# Layer freezing
layer_config:
  vision_tower:
    freeze_patch_embed: true
    freeze_bottom_layers: true
    trainable_top_k_blocks: 2
  
  llm:
    freeze_bottom_layers: true
    trainable_top_k_blocks: 8
  
  merger:
    freeze: false

# Rewards
rewards:
  # Formatting rewards
  parse: 1.0
  wrappers: 0.5
  coords: 0.3
  separators: 0.2
  vocab: 0.5
  
  # Geometry rewards (higher weight)
  bbox_giou: 2.0
  quad_l1: 1.5
  line_l1: 1.5
  ordering: 0.5
  coverage: 1.0
  geometry_sanity: 1.0

# Logging
logging:
  logging_steps: 10
  report_to: ["tensorboard"]
  log_completions: true  # Inherited from TRL
  log_predictions: true
  log_rewards_breakdown: true
  num_completions_to_print: 3  # Inherited from TRL

# Checkpointing
checkpointing:
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  evaluation_strategy: "steps"
  eval_steps: 250
  load_best_model_at_end: true
  metric_for_best_model: "eval_reward"

# Runtime
runtime:
  seed: 17
  dataloader_num_workers: 4
  shuffle_dataset: true  # Used by RepeatSampler (inherited from TRL)

output:
  output_dir: "/data3/outputs/rl_dense_grpo_v3"
  run_name: "dense_grpo_enhanced"
```

---

## 5. Testing & Validation Plan

### Phase 1: Baseline Verification (1 day)
1. Run existing implementation for 500 steps
2. Verify RepeatSampler is working (check logs for prompt repetition)
3. Verify multi-iteration buffering (check generation frequency)
4. Establish baseline metrics

### Phase 2: Stability Enhancements (2-3 days)
1. Implement advantage clipping
2. Implement adaptive KL scheduling
3. Train for 1000 steps
4. Compare stability metrics:
   - Loss variance
   - Gradient norms
   - Clipping ratios
   - KL divergence trends

### Phase 3: Quality Improvements (3-5 days)
1. Implement temperature scheduling
2. Implement reward standardization
3. Add generation diagnostics
4. Full training run
5. Compare:
   - Detection metrics (IoU, F1)
   - Parse success rate
   - Generation quality
   - Final reward

### Phase 4: Ablation Study (1 week)
Test individual components:
- Advantage clipping vs. no clipping
- Adaptive KL vs. fixed beta
- Temperature scheduling vs. fixed
- Reward standardization vs. raw rewards

---

## 6. Expected Outcomes

### Quantitative Improvements:
- **Training Stability**: 20-30% reduction in loss variance (via advantage clipping & adaptive KL)
- **Convergence Speed**: Similar (core algorithm already correct)
- **Detection Quality**: 3-7% improvement in IoU (via better reward balancing)
- **Parse Success**: 5-10% improvement (via temperature scheduling)

### Qualitative Improvements:
- More consistent formatting across generations
- Better balance between caption quality and geometry accuracy
- Smoother training curves
- Fewer degenerate outputs in early training

---

## 7. Implementation Priority

### Week 1: Core Stability
- [ ] Implement advantage clipping
- [ ] Implement adaptive KL scheduling
- [ ] Test with small dataset (100 samples)
- [ ] Verify no regressions

### Week 2: Quality Enhancements
- [ ] Implement temperature scheduling
- [ ] Implement reward standardization
- [ ] Add generation diagnostics
- [ ] Full training run (1000 steps)

### Week 3: Monitoring & Polish
- [ ] Add gradient norm monitoring
- [ ] Enhanced logging
- [ ] Compare against baseline
- [ ] Hyperparameter tuning

### Week 4: Evaluation & Documentation
- [ ] Ablation studies
- [ ] Final evaluation
- [ ] Update documentation
- [ ] Export best practices

---

## 8. Conclusion

**IMPORTANT CORRECTION**: The current `VisionGRPOTrainer` correctly inherits the core GRPO algorithm from TRL's `GRPOTrainer`, including:
- ✅ RepeatSampler for proper prompt grouping
- ✅ Multi-iteration buffering for generation reuse
- ✅ Detailed clipping metrics for monitoring
- ✅ Correct advantage computation and loss calculation

The implementation is **fundamentally sound** and does not require major algorithmic fixes.

**Recommended Focus Areas:**
1. ⭐ Advantage clipping for stability
2. ⭐ Adaptive KL scheduling for balance
3. ⭐ Temperature scheduling for quality
4. ⭐ Reward standardization for component balance
5. 📊 Enhanced monitoring for debugging

These improvements focus on **stability, quality, and observability** rather than core algorithm correctness.

---

## 9. References

1. GRPO Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
2. TRL GRPOTrainer: https://github.com/huggingface/trl (v0.9.0+)
3. Official Qwen2-VL-Finetune: https://github.com/QwenLM/Qwen2-VL/tree/main/finetune
4. Current Implementation: `src_new/rl/` directory

---

**Document Version:** 2.0 (Revised)  
**Last Updated:** October 4, 2025  
**Revision Notes:** Corrected analysis after reviewing TRL GRPOTrainer source. Original document incorrectly stated that RepeatSampler and multi-iteration buffering were missing. The current implementation inherits these correctly from TRL.
