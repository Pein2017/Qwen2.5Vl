# Phase 1: Data Model - Diagnostic Entity Schemas

**Feature**: 004-grpo-post-training  
**Date**: 2025-10-09  
**Status**: Complete

## Overview

This document defines typed dataclasses for all diagnostic entities introduced by the GRPO post-training diagnostics feature. All schemas follow the Constitution's principle of explicit contracts and are designed for minimal overhead (<10% total) through lazy evaluation and rank-0-only persistence.

---

## Core Diagnostic Entities

###TrustRegionDiagnostic

Tracks GRPO ratio health at each training step to detect degeneracy (all ratios ≈ 1.0).

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrustRegionDiagnostic:
    """Per-step trust region statistics for GRPO ratio π_current / π_generation."""
    
    step: int
    # Ratio statistics (computed from exp(cur_logps - generation_logps))
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    ratio_p25: float  # 25th percentile
    ratio_p75: float  # 75th percentile
    
    # Health indicators
    is_degenerate: bool  # True if std < 0.1
    fallback_count: int  # Number of completions using current policy fallback
    clip_fraction: float  # Fraction of ratios clipped by [1-eps, 1+eps]
    
    # Generation log-probs verification
    generation_logps_present: bool
    generation_logps_shape: Optional[tuple[int, ...]]
    
    # Optional cross-rank statistics (if cross_rank_advantages enabled)
    global_ratio_mean: Optional[float] = None
    global_ratio_std: Optional[float] = None
    
    def to_tensorboard(self) -> dict[str, float]:
        """Export scalars for TensorBoard logging."""
        return {
            "trust_region/ratio_mean": self.ratio_mean,
            "trust_region/ratio_std": self.ratio_std,
            "trust_region/ratio_min": self.ratio_min,
            "trust_region/ratio_max": self.ratio_max,
            "trust_region/clip_fraction": self.clip_fraction,
            "trust_region/is_degenerate": float(self.is_degenerate),
            "trust_region/fallback_count": float(self.fallback_count),
        }
    
    def export_histogram(self, ratios: torch.Tensor, output_path: Path) -> None:
        """Export ratio distribution as PNG histogram (rank-0 only)."""
        import matplotlib.pyplot as plt
        plt.hist(ratios.cpu().numpy(), bins=50, range=(0.5, 1.5))
        plt.xlabel("Policy Ratio")
        plt.ylabel("Count")
        plt.title(f"Step {self.step}: Trust Region Ratio Distribution")
        plt.axvline(1.0, color='r', linestyle='--', label='Identity')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
```

**Usage**:
```python
# In completion_loss.py after ratio computation
diagnostic = TrustRegionDiagnostic(
    step=global_step,
    ratio_mean=ratios.mean().item(),
    ratio_std=ratios.std().item(),
    ratio_min=ratios.min().item(),
    ratio_max=ratios.max().item(),
    ratio_p25=ratios.quantile(0.25).item(),
    ratio_p75=ratios.quantile(0.75).item(),
    is_degenerate=ratios.std().item() < 0.1,
    fallback_count=fallback_counter,
    clip_fraction=(torch.abs(ratios - 1.0) > clip_eps).float().mean().item(),
    generation_logps_present=generation_logps is not None,
    generation_logps_shape=tuple(generation_logps.shape) if generation_logps is not None else None,
)
if accelerator.is_main_process:
    tb_logger.log_scalars(diagnostic.to_tensorboard(), step=global_step)
    if global_step % 10 == 0:
        diagnostic.export_histogram(ratios, output_dir / f"diagnostics/{global_step:06d}/ratios.png")
```

---

### MultimodalAlignmentCheck

Validates consistency of image tokens, `pixel_values`, and `image_grid_thw` at pipeline stages.

```python
@dataclass
class MultimodalAlignmentCheck:
    """Per-sample multimodal alignment validation."""
    
    stage: str  # "dataset" | "buffer_generation" | "loss_computation"
    sample_idx: int
    
    # Image token validation
    expected_image_tokens: int  # From (∑ t*h*w) // merge_size²
    actual_image_tokens: int    # Decoded <|image_pad|> count
    image_token_match: bool
    
    # Pixel values validation
    pixel_values_shape: tuple[int, ...]  # Should be [∑(t*h*w), channels, patch_size, patch_size]
    expected_pixel_rows: int             # From ∑(t*h*w)
    pixel_rows_match: bool
    
    # THW validation
    image_grid_thw_shape: tuple[int, ...]  # Should be [num_images, 3]
    num_images: int
    thw_shape_valid: bool
    
    # Overall health
    is_aligned: bool  # True if all checks pass
    
    def raise_on_mismatch(self) -> None:
        """Raise ImageTokenMismatchError if alignment fails (fail-fast principle)."""
        if not self.is_aligned:
            from src_new.processing.errors import ImageTokenMismatchError
            raise ImageTokenMismatchError(
                f"Multimodal misalignment at stage '{self.stage}' for sample {self.sample_idx}: "
                f"image_tokens expected={self.expected_image_tokens} actual={self.actual_image_tokens}, "
                f"pixel_rows expected={self.expected_pixel_rows} actual={self.pixel_values_shape[0]}"
            )
    
    def to_log_dict(self) -> dict[str, any]:
        """Export for console/file logging."""
        return {
            "stage": self.stage,
            "sample_idx": self.sample_idx,
            "image_token_match": self.image_token_match,
            "pixel_rows_match": self.pixel_rows_match,
            "thw_shape_valid": self.thw_shape_valid,
            "is_aligned": self.is_aligned,
        }
```

**Usage**:
```python
# In buffer.py after vision tensor preparation
check = MultimodalAlignmentCheck(
    stage="buffer_generation",
    sample_idx=idx,
    expected_image_tokens=sum(t*h*w for t,h,w in image_grid_thw) // (merge_size ** 2),
    actual_image_tokens=decoded_prompt.count("<|image_pad|>"),
    image_token_match=...,
    pixel_values_shape=tuple(pixel_values.shape),
    expected_pixel_rows=sum(t*h*w for t,h,w in image_grid_thw),
    pixel_rows_match=pixel_values.shape[0] == sum(t*h*w for t,h,w in image_grid_thw),
    image_grid_thw_shape=tuple(image_grid_thw.shape),
    num_images=len(image_grid_thw),
    thw_shape_valid=image_grid_thw.shape == (num_images, 3),
    is_aligned=all([...]),
)
check.raise_on_mismatch()  # Fail-fast validation
logger.debug(f"Multimodal alignment check: {check.to_log_dict()}")
```

---

### RewardProfile

Tracks reward function statistics with **within-group** and **between-group** variance separation.

```python
@dataclass
class RewardProfile:
    """Per-reward-function statistics for diversity diagnosis."""
    
    step: int
    reward_name: str
    
    # Raw statistics (before standardization)
    raw_mean: float
    raw_std: float
    raw_min: float
    raw_max: float
    nan_count: int
    
    # Within-group variance (across K completions per prompt)
    within_group_mean_std: float  # Mean of std(rewards[prompt_i, :]) across prompts
    within_group_median_std: float
    prompts_with_variance_above_threshold: int  # Count where std > 0.05
    total_prompts: int
    
    # Between-group variance (across prompts)
    between_group_std: float  # Std of mean(rewards[prompt_i, :])
    
    # Diversity ratio
    diversity_ratio: float  # within_std / between_std; <0.1 → over-fitting
    
    # Correlation with other rewards (computed externally)
    correlation_matrix: Optional[dict[str, float]] = None
    
    # Health indicators
    is_collapsed: bool  # True if within_group_mean_std < 0.01
    is_degenerate: bool  # True if diversity_ratio < 0.1
    
    def to_tensorboard(self) -> dict[str, float]:
        return {
            f"rewards/{self.reward_name}/raw_mean": self.raw_mean,
            f"rewards/{self.reward_name}/raw_std": self.raw_std,
            f"rewards/{self.reward_name}/within_group_std": self.within_group_mean_std,
            f"rewards/{self.reward_name}/between_group_std": self.between_group_std,
            f"rewards/{self.reward_name}/diversity_ratio": self.diversity_ratio,
            f"rewards/{self.reward_name}/is_collapsed": float(self.is_collapsed),
        }
```

**Usage**:
```python
# In grpo_trainer.py after reward computation
from src_new.rl.diagnostics.rewards import compute_reward_profile

reward_profiles = []
for reward_fn_name, reward_values in reward_dict.items():
    # reward_values shape: [prompt_batch_size, K]
    profile = compute_reward_profile(
        step=global_step,
        reward_name=reward_fn_name,
        reward_tensor=reward_values,  # [P, K]
        threshold=0.05,
    )
    reward_profiles.append(profile)
    tb_logger.log_scalars(profile.to_tensorboard(), step=global_step)
```

---

### CheckpointDiversityComparison

Stores diversity measurements across different SFT checkpoints (Phase 2 vs Phase 3).

```python
@dataclass
class CheckpointDiversityComparison:
    """Comparison of generation diversity between SFT checkpoints."""
    
    checkpoint_phase: str  # "phase_2" | "phase_3"
    checkpoint_path: str
    temperature: float
    num_prompts: int
    k_completions: int
    
    # Within-group diversity (per prompt)
    within_group_reward_stds: list[float]  # One std per prompt
    mean_within_std: float
    median_within_std: float
    prompts_above_threshold: int  # Where std > 0.05
    
    # Between-group diversity
    between_group_std: float
    
    # Advantage statistics
    advantage_mean: float
    advantage_std: float
    advantage_p25: float
    advantage_p75: float
    
    # Diversity ratio
    diversity_ratio: float  # mean_within_std / between_group_std
    
    # Viability assessment
    is_viable_for_grpo: bool  # True if advantage_std > 0.5 and diversity_ratio > 0.1
    
    def compare_to(self, other: 'CheckpointDiversityComparison') -> dict[str, float]:
        """Compare diversity metrics against another checkpoint."""
        return {
            "diversity_ratio_improvement": (self.diversity_ratio - other.diversity_ratio) / other.diversity_ratio,
            "within_std_ratio": self.mean_within_std / other.mean_within_std,
            "advantage_std_ratio": self.advantage_std / other.advantage_std,
            "viable_improvement": float(self.is_viable_for_grpo) - float(other.is_viable_for_grpo),
        }
```

**Usage**:
```python
# In scripts/run_checkpoint_diversity_test.sh workflow
phase2_result = CheckpointDiversityComparison(
    checkpoint_phase="phase_2",
    checkpoint_path="outputs/.../phase_2/.../best-200-eval_loss0.7006",
    temperature=0.9,
    num_prompts=50,
    k_completions=50,
    within_group_reward_stds=[...],  # Computed from reward variance
    mean_within_std=np.mean(within_stds),
    ...
    is_viable_for_grpo=advantage_std > 0.5 and diversity_ratio > 0.1,
)

phase3_result = CheckpointDiversityComparison(...)

comparison = phase2_result.compare_to(phase3_result)
logger.info(f"Checkpoint diversity comparison: {comparison}")
logger.info(f"Recommended checkpoint: {phase2_result.checkpoint_phase if phase2_result.is_viable_for_grpo else phase3_result.checkpoint_phase}")
```

---

## GradientFlowSnapshot

Tracks gradient norms per layer to detect vanishing/exploding gradients.

```python
@dataclass
class GradientFlowSnapshot:
    """Per-layer gradient statistics at a training step."""
    
    step: int
    layer_name: str
    
    # Gradient norms
    grad_norm_mean: float
    grad_norm_max: float
    grad_norm_min: float
    
    # Health indicators
    has_zero_gradients: bool  # True if norm < 1e-6
    has_exploding_gradients: bool  # True if norm > 100
    
    # Optional: gradient histogram bins for heatmap export
    histogram_bins: Optional[list[float]] = None
    
    @staticmethod
    def from_parameter(step: int, name: str, param: torch.nn.Parameter) -> 'GradientFlowSnapshot':
        """Factory method to create snapshot from parameter."""
        if param.grad is None:
            return GradientFlowSnapshot(
                step=step, layer_name=name, grad_norm_mean=0.0, grad_norm_max=0.0,
                grad_norm_min=0.0, has_zero_gradients=True, has_exploding_gradients=False,
            )
        grad_norm = param.grad.norm().item()
        return GradientFlowSnapshot(
            step=step, layer_name=name, grad_norm_mean=grad_norm, grad_norm_max=grad_norm,
            grad_norm_min=grad_norm, has_zero_gradients=grad_norm < 1e-6,
            has_exploding_gradients=grad_norm > 100,
        )
    
    def export_heatmap(self, all_snapshots: list['GradientFlowSnapshot'], output_path: Path) -> None:
        """Export gradient flow heatmap across layers (rank-0 only)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        layers = [s.layer_name for s in all_snapshots]
        norms = [s.grad_norm_mean for s in all_snapshots]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layers)), norms)
        plt.xticks(range(len(layers)), layers, rotation=90)
        plt.yscale('log')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title(f'Step {self.step}: Gradient Flow')
        plt.axhline(1e-6, color='r', linestyle='--', label='Vanishing threshold')
        plt.axhline(100, color='orange', linestyle='--', label='Exploding threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
```

---

### SequentialProcessingMonitor

Validates constitutional sequential processing requirement.

```python
@dataclass
class SequentialProcessingMonitor:
    """Per-step validation of batch_size=1 compliance and GPU memory."""
    
    step: int
    
    # Batch size assertions
    generation_batch_sizes: list[int]  # Should all be 1
    forward_batch_sizes: list[int]     # Should all be 1
    
    # GPU memory tracking (MB)
    peak_memory_allocated_mb: float
    peak_memory_reserved_mb: float
    
    # Compliance flags
    is_compliant: bool  # True if all batch sizes == 1
    has_oom_risk: bool  # True if peak_memory > 42GB
    
    def raise_on_violation(self) -> None:
        """Raise error if sequential processing violated (fail-fast)."""
        if not self.is_compliant:
            raise RuntimeError(
                f"Sequential processing violation at step {self.step}: "
                f"generation batch_sizes={self.generation_batch_sizes}, "
                f"forward batch_sizes={self.forward_batch_sizes}. "
                f"Constitution v4.1.1 mandates batch_size=1 for all operations."
            )
    
    def to_tensorboard(self) -> dict[str, float]:
        return {
            "sequential/is_compliant": float(self.is_compliant),
            "sequential/peak_memory_mb": self.peak_memory_allocated_mb,
            "sequential/has_oom_risk": float(self.has_oom_risk),
        }
```

---

## Validation Interfaces

See `contracts/` directory for detailed interface definitions.

---

## Storage and Serialization

All diagnostic entities support:
1. **TensorBoard export** via `to_tensorboard() -> dict[str, float]`
2. **File export** (rank-0 only) to `{output_dir}/diagnostics/{step:06d}/`
   - JSON for scalar metrics
   - PNG for histograms/heatmaps
3. **Console logging** via structured `to_log_dict() -> dict`

Example persistence helper:

```python
from pathlib import Path
import json

def save_diagnostic(entity, output_dir: Path, step: int):
    """Save diagnostic entity to JSON (rank-0 only)."""
    if not accelerator.is_main_process:
        return
    
    output_path = output_dir / "diagnostics" / f"{step:06d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / f"{entity.__class__.__name__.lower()}.json"
    with open(json_path, 'w') as f:
        json.dump(asdict(entity), f, indent=2)
    
    # Save visualizations if applicable
    if hasattr(entity, 'export_histogram'):
        entity.export_histogram(...)
    if hasattr(entity, 'export_heatmap'):
        entity.export_heatmap(...)
```

---

## Type Safety and Validation

All dataclasses use Python 3.10+ type hints and are validated with `pyright`:

```bash
pyright src_new/rl/diagnostics/
```

Optional runtime validation with `pydantic`:

```python
from pydantic.dataclasses import dataclass as pydantic_dataclass

@pydantic_dataclass
class TrustRegionDiagnostic:
    ...
    
    def __post_init__(self):
        assert 0 <= self.clip_fraction <= 1.0, "clip_fraction must be in [0, 1]"
        assert self.ratio_std >= 0, "ratio_std must be non-negative"
```

---

## Next Steps

- Implement dataclasses in `src_new/rl/diagnostics/*.py`
- Define validation interfaces in `contracts/`
- Create quickstart guide in `quickstart.md`
- Write unit tests in `tests/rl/diagnostics/`
