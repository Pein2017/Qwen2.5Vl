# Reward standardization utilities for RL
from __future__ import annotations

from typing import Dict

import torch


class RewardStandardizer:
    """
    Standardizes reward tensors to approximately zero mean and unit variance
    using momentum-based running statistics per reward name.
    """

    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-8) -> None:
        self.momentum: float = float(momentum)
        self.epsilon: float = float(epsilon)
        self.running_mean: Dict[str, torch.Tensor] = {}
        self.running_var: Dict[str, torch.Tensor] = {}
        self.running_count: Dict[str, int] = {}

    def update_and_standardize(
        self, reward_name: str, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Update running stats and return standardized rewards on the same device/dtype.
        Args:
            reward_name: Logical key of the reward component
            rewards: Tensor of shape (N,) on any device
        Returns:
            Tensor of shape (N,) standardized per running stats
        """
        if rewards.dim() != 1:
            rewards = rewards.view(-1)

        device = rewards.device
        dtype = rewards.dtype

        # Compute batch statistics
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False) + self.epsilon

        if reward_name not in self.running_mean:
            self.running_mean[reward_name] = batch_mean.detach().to(
                device=device, dtype=dtype
            )
            self.running_var[reward_name] = batch_var.detach().to(
                device=device, dtype=dtype
            )
            self.running_count[reward_name] = int(rewards.numel())
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

        standardized = (rewards - self.running_mean[reward_name]) / torch.sqrt(
            self.running_var[reward_name]
        )
        return standardized

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
