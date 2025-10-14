"""Typed dataclasses for RL helpers.

These dataclasses package non-tensor-logic payloads passed between
generation, trainer, and logging. They do not change any tensor
operations or model forward behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationResult:
    """Container for a single prompt-batch generation result.

    Fields mirror the historical dict-based payload used by the trainer.
    Use to_dict() to get the legacy mapping for consumers that still
    expect a dict.
    """

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    reward_names: List[str] = field(default_factory=list)

    # Optional reward variants
    raw_rewards: Optional[torch.Tensor] = None
    rewards_per_func: Optional[torch.Tensor] = None
    raw_rewards_per_func: Optional[torch.Tensor] = None

    # Generation-policy log-probs (trust region)
    generation_logps: Optional[torch.Tensor] = None

    # Packed vision tensors
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    images_per_sample: Optional[torch.Tensor] = None

    # Completion metadata
    completion_lengths: Optional[torch.Tensor] = None
    terminated_with_eos: Optional[torch.Tensor] = None
    truncated_flags: Optional[torch.Tensor] = None

    # Decoded text/context
    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)

    # Scalars
    temperature: float = 0.0
    beta: float = 0.0

    # Extra metrics (dynamic caps, ratios, sanitizer stats, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a legacy dict mapping compatible with existing consumers."""
        out: Dict[str, Any] = {
            "prompt_ids": self.prompt_ids,
            "prompt_mask": self.prompt_mask,
            "completion_ids": self.completion_ids,
            "completion_mask": self.completion_mask,
            "advantages": self.advantages,
            "rewards": self.rewards,
            "reward_names": list(self.reward_names),
            "prompts": list(self.prompts),
            "completions": list(self.completions),
            "meta": list(self.meta),
            "temperature": float(self.temperature),
            "beta": float(self.beta),
        }
        if self.raw_rewards is not None:
            out["raw_rewards"] = self.raw_rewards
        if self.rewards_per_func is not None:
            out["rewards_per_func"] = self.rewards_per_func
        if self.raw_rewards_per_func is not None:
            out["raw_rewards_per_func"] = self.raw_rewards_per_func
        if self.generation_logps is not None:
            out["generation_logps"] = self.generation_logps
        if self.pixel_values is not None:
            out["pixel_values"] = self.pixel_values
        if self.image_grid_thw is not None:
            out["image_grid_thw"] = self.image_grid_thw
        if self.images_per_sample is not None:
            out["images_per_sample"] = self.images_per_sample
        if self.completion_lengths is not None:
            out["completion_lengths"] = self.completion_lengths
        if self.terminated_with_eos is not None:
            out["terminated_with_eos"] = self.terminated_with_eos
        if self.truncated_flags is not None:
            out["truncated_flags"] = self.truncated_flags
        # Pass-through extras into the dict (flat copy)
        for k, v in self.extra.items():
            if k not in out:
                out[k] = v
        return out


@dataclass
class CompletionSlice:
    """A single completion slice for streamed loss computation."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    advantages: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    images_per_sample: Optional[torch.Tensor]
    temperature: float
    beta: float


@dataclass
class ClipDiagnostics:
    """Token-level clipping counters and aggregate ratios."""

    low: int = 0
    high: int = 0
    region: int = 0
    total: int = 0

    def ratios(self) -> Dict[str, float]:
        if self.total <= 0:
            return {
                "policy_clip_low_ratio": 0.0,
                "policy_clip_high_ratio": 0.0,
                "policy_clip_region_ratio": 0.0,
            }
        return {
            "policy_clip_low_ratio": float(self.low) / float(self.total),
            "policy_clip_high_ratio": float(self.high) / float(self.total),
            "policy_clip_region_ratio": float(self.region) / float(self.total),
        }


@dataclass
class PromptBatchMetrics:
    fill_ratio: float = 0.0
    reward_average: float = 0.0
    invalid_fraction: float = 0.0
    trajectories_collected: float = 0.0
    dropped_prompts: float = 0.0
    steps: float = 0.0


@dataclass
class BufferTelemetry:
    is_fresh: bool = False
    step_in_cycle: int = 0
    completions_per_cycle: int = 0
    completions_total: int = 0
    reuse_configured: int = 1
    reuse_active: int = 0


@dataclass
class TrainingLogs:
    """Flat scalars destined for TB/console."""

    loss: float = 0.0
    reward: float = 0.0
    reward_std: float = 0.0
    raw_reward: Optional[float] = None
    raw_reward_std: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    epoch: float = 0.0
    eta_minutes: float = 0.0
    temperature: float = 0.0
    beta: float = 0.0
    terminated_ratio: float = 0.0


__all__ = [
    "GenerationResult",
    "CompletionSlice",
    "ClipDiagnostics",
    "PromptBatchMetrics",
    "BufferTelemetry",
    "TrainingLogs",
]


