"""Generation buffer for Swift-style buffer reuse across multiple optimizer steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationBuffer:
    """Stores generation results on CPU for reuse across multiple optimizer steps.

    This enables the Swift-style "generate once, optimize S times" pattern where
    completions are generated less frequently and reused for multiple gradient updates.
    This improves:
    - Generation efficiency (S× fewer generations)
    - Sample efficiency (each completion used S times)
    - Training stability (same data for S consecutive updates)

    The buffer stores all tensors on CPU and streams them to GPU one completion at a time
    during the optimization phase to minimize GPU memory usage.
    """

    # Text IDs (CPU)
    prompt_ids: torch.Tensor  # [P, max_prompt_len]
    prompt_mask: torch.Tensor  # [P, max_prompt_len]
    completion_ids_list: List[torch.Tensor]  # P prompts, each [K, max_comp_len]
    completion_mask_list: List[torch.Tensor]  # P prompts, each [K, max_comp_len]

    # Generation-time log-probs (CPU, critical for trust region!)
    old_logps_list: List[torch.Tensor]  # P prompts, each [K, max_comp_len]

    # Rewards & advantages (CPU)
    advantages_list: List[torch.Tensor]  # P prompts, each [K]
    rewards_list: List[torch.Tensor]  # P prompts, each [K]

    # Per-function rewards for detailed logging (CPU)
    rewards_per_func_list: List[torch.Tensor]  # P prompts, each [K, num_rewards]
    raw_rewards_per_func_list: List[torch.Tensor]  # P prompts, each [K, num_rewards]

    # Termination and truncation flags (CPU) aligned per completion
    terminated_flags_list: List[torch.Tensor]  # P prompts, each [K]
    truncated_flags_list: List[torch.Tensor]  # P prompts, each [K]

    # Vision tensors (CPU)
    pixel_values_list: List[Optional[torch.Tensor]]  # P prompts
    image_grid_thw_list: List[Optional[torch.Tensor]]  # P prompts
    images_per_sample_list: List[Optional[torch.Tensor]]  # P prompts

    # Metadata for logging
    meta_list: List[Dict[str, Any]]  # P prompts
    temperature: float
    beta: float

    # Indexing state
    steps_per_generation: int
    current_step_in_cycle: int = 0  # 0 to (P × K - 1)

    # Diagnostics accumulation
    total_clip_low: int = field(default=0)
    total_clip_high: int = field(default=0)
    total_clip_region: int = field(default=0)
    total_clip_tokens: int = field(default=0)

    def is_exhausted(self) -> bool:
        """Check if all completions have been used."""
        total_completions = self.get_total_completions()
        return self.current_step_in_cycle >= total_completions

    def get_total_completions(self) -> int:
        """Total number of completions available in this buffer."""
        P = len(self.completion_ids_list)
        if P == 0:
            return 0
        K_per_prompt = self.completion_ids_list[0].size(0) if P > 0 else 0
        return P * K_per_prompt

    def get_completion_for_step(self, step_in_cycle: int) -> Dict[str, Any]:
        """Return one completion's data for sequential streaming.

        Args:
            step_in_cycle: Index from 0 to (P × K - 1)

        Returns:
            Dict with completion data (all on CPU, caller moves to GPU)

        Strategy: Cycle through completions within each prompt first, then move to next prompt.
        Example: S=4, P=2, K=2 → steps map to: P0C0, P0C1, P1C0, P1C1
        """
        P = len(self.completion_ids_list)
        K_per_prompt = self.completion_ids_list[0].size(0) if P > 0 else 0

        total_completions = P * K_per_prompt
        if step_in_cycle >= total_completions:
            raise ValueError(
                f"step_in_cycle {step_in_cycle} >= total {total_completions}"
            )

        # Map step_in_cycle to (prompt_idx, completion_idx)
        prompt_idx = step_in_cycle // K_per_prompt
        completion_idx = step_in_cycle % K_per_prompt

        return {
            "prompt_ids": self.prompt_ids[prompt_idx],  # [max_prompt_len]
            "prompt_mask": self.prompt_mask[prompt_idx],
            "completion_ids": self.completion_ids_list[prompt_idx][
                completion_idx
            ],  # [max_comp_len]
            "completion_mask": self.completion_mask_list[prompt_idx][completion_idx],
            "old_logps": self.old_logps_list[prompt_idx][
                completion_idx
            ],  # Critical for trust region!
            "advantages": self.advantages_list[prompt_idx][completion_idx],  # scalar
            "pixel_values": self.pixel_values_list[prompt_idx],
            "image_grid_thw": self.image_grid_thw_list[prompt_idx],
            "images_per_sample": self.images_per_sample_list[prompt_idx],
            "meta": self.meta_list[prompt_idx],
            "temperature": self.temperature,
            "beta": self.beta,
        }

    def increment_step(self):
        """Move to next completion in the buffer."""
        self.current_step_in_cycle += 1

    def accumulate_clip_diagnostics(
        self,
        clip_low_tokens: int,
        clip_high_tokens: int,
        clip_region_tokens: int,
        clip_total_tokens: int,
    ):
        """Accumulate clipping diagnostics across completions for logging."""
        self.total_clip_low += clip_low_tokens
        self.total_clip_high += clip_high_tokens
        self.total_clip_region += clip_region_tokens
        self.total_clip_tokens += clip_total_tokens

    def get_clip_diagnostics(self) -> Dict[str, Any]:
        """Get aggregated clipping diagnostics for logging."""
        if self.total_clip_tokens > 0:
            return {
                "clip_low_ratio": float(self.total_clip_low)
                / float(self.total_clip_tokens),
                "clip_high_ratio": float(self.total_clip_high)
                / float(self.total_clip_tokens),
                "clip_region_ratio": float(self.total_clip_region)
                / float(self.total_clip_tokens),
                "clip_total_tokens": self.total_clip_tokens,
            }
        else:
            return {
                "clip_low_ratio": 0.0,
                "clip_high_ratio": 0.0,
                "clip_region_ratio": 0.0,
                "clip_total_tokens": 0,
            }


__all__ = ["GenerationBuffer"]
