# Reward standardization utilities for RL
from __future__ import annotations

from typing import Dict

import torch


class RewardStandardizer:
    """
    Standardizes reward tensors to approximately zero mean and unit variance
    using momentum-based running statistics per reward name.

    Includes warmup period where batch statistics are used instead of running stats
    to avoid explosion from bad initial batches.
    """

    def __init__(
        self,
        *,
        mode: str = "std_only",
        momentum: float = 0.97,
        epsilon: float = 1e-8,
        warmup_steps: int = 10,
        min_std: float = 0.05,
        clip_abs: float = 3.0,
    ) -> None:
        self.mode: str = str(mode)
        self.momentum: float = float(momentum)
        self.epsilon: float = float(epsilon)
        self.warmup_steps: int = int(warmup_steps)
        self.min_std: float = float(min_std)
        self.clip_abs: float = float(clip_abs)
        self.running_mean: Dict[str, torch.Tensor] = {}
        self.running_var: Dict[str, torch.Tensor] = {}
        self.running_count: Dict[str, int] = {}
        self.update_count: Dict[str, int] = {}  # Track updates per reward for warmup

    def update_and_standardize(
        self, reward_name: str, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Update running stats and return standardized rewards on the same device/dtype.

        During warmup period, uses batch statistics only to avoid explosion from
        bad initial running stats.

        Args:
            reward_name: Logical key of the reward component
            rewards: Tensor of shape (N,) on any device
        Returns:
            Tensor of shape (N,) standardized per batch or running stats
        """
        if rewards.dim() != 1:
            rewards = rewards.view(-1)

        device = rewards.device
        dtype = rewards.dtype

        # Compute batch statistics
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False) + self.epsilon

        # Initialize tracking
        if reward_name not in self.running_mean:
            self.running_mean[reward_name] = batch_mean.detach().to(
                device=device, dtype=dtype
            )
            self.running_var[reward_name] = batch_var.detach().to(
                device=device, dtype=dtype
            )
            self.running_count[reward_name] = int(rewards.numel())
            self.update_count[reward_name] = 1
        else:
            # Momentum update (keep stats on the same device as incoming rewards)
            self.running_mean[reward_name] = self.momentum * self.running_mean[
                reward_name
            ] + (1.0 - self.momentum) * batch_mean.detach().to(
                device=device, dtype=dtype
            )
            self.running_var[reward_name] = self.momentum * self.running_var[
                reward_name
            ] + (1.0 - self.momentum) * batch_var.detach().to(
                device=device, dtype=dtype
            )
            self.running_count[reward_name] += int(rewards.numel())
            self.update_count[reward_name] += 1

        # Determine std and mean with warmup policy
        if self.update_count[reward_name] <= self.warmup_steps:
            std_used = torch.sqrt(batch_var)
            mean_used = batch_mean
        else:
            std_used = torch.sqrt(self.running_var[reward_name])
            mean_used = self.running_mean[reward_name]

        # Clamp std
        std_used = torch.clamp(std_used, min=self.min_std)

        # Mode selection
        if self.mode == "std_only":
            z = rewards / std_used
        else:
            z = (rewards - mean_used) / std_used

        # Optional clipping
        if self.clip_abs > 0:
            z = torch.clamp(z, -self.clip_abs, self.clip_abs)
        return z

    def get_stats(self, reward_name: str) -> Dict[str, float]:
        if reward_name not in self.running_mean:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        return {
            "mean": float(self.running_mean[reward_name].detach().cpu().item()),
            "std": float(
                torch.sqrt(self.running_var[reward_name]).detach().cpu().item()
            ),
            "count": int(self.running_count[reward_name]),
        }
