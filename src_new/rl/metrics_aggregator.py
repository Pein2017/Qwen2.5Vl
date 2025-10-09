"""Metric aggregation utilities for distributed RL training."""

from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator


class MetricsAggregator:
    """Handles cross-rank metric gathering and aggregation."""

    def __init__(self, accelerator: Accelerator, device: torch.device):
        self.accelerator = accelerator
        self.device = device
        self.world_size = getattr(accelerator.state, "num_processes", 1)

    def gather_tensor(
        self, tensor: Optional[torch.Tensor], to_cpu: bool = True
    ) -> torch.Tensor:
        """Gather tensor across ranks with optional CPU transfer."""
        if tensor is None:
            tensor = torch.zeros(1, device=self.device)

        tensor_det = tensor.detach()
        if self.world_size > 1:
            tensor_det = self.accelerator.gather(tensor_det)

        if to_cpu and tensor_det.device.type != "cpu":
            tensor_det = tensor_det.cpu()

        return tensor_det

    def gather_scalar_stats(self, tensor: Optional[torch.Tensor]) -> Dict[str, float]:
        """Gather tensor and return mean/std statistics."""
        gathered = self.gather_tensor(tensor)
        return {
            "mean": float(gathered.mean().item()),
            "std": float(gathered.std(unbiased=False).item()),
        }

    def gather_rewards_stats(
        self, generation_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Gather reward statistics from generation result."""
        stats = {}

        # Normalized rewards
        rewards = generation_result.get("rewards")
        if rewards is not None:
            reward_stats = self.gather_scalar_stats(rewards)
            stats["reward"] = reward_stats["mean"]
            stats["reward_std"] = reward_stats["std"]

        # Raw/unnormalized rewards
        raw_rewards = generation_result.get("raw_rewards")
        if raw_rewards is not None:
            raw_stats = self.gather_scalar_stats(raw_rewards)
            stats["raw_reward"] = raw_stats["mean"]
            stats["raw_reward_std"] = raw_stats["std"]

        return stats

    def gather_advantage_stats(
        self, generation_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Gather advantage statistics."""
        adv_tensor = generation_result.get("advantages")
        if adv_tensor is None:
            return {"std": 0.0, "max_abs": 0.0}

        gathered = self.gather_tensor(adv_tensor)
        return {
            "std": float(gathered.std(unbiased=False).item()),
            "max_abs": float(gathered.abs().max().item()),
        }

    def gather_per_reward_components(
        self,
        generation_result: Dict[str, Any],
        reward_names: List[str],
        key: str = "rewards_per_func",
    ) -> Dict[str, Dict[str, float]]:
        """Gather per-reward component statistics.

        Args:
            generation_result: Generation result dict
            reward_names: List of reward function names
            key: Key for reward components (rewards_per_func or raw_rewards_per_func)

        Returns:
            Dict mapping reward names to their mean/std statistics
        """
        rewards_per_func = generation_result.get(key)
        if rewards_per_func is None:
            return {}

        gathered = self.gather_tensor(rewards_per_func, to_cpu=False)

        components = {}
        for idx, name in enumerate(reward_names):
            if idx >= gathered.size(-1):
                continue
            col = gathered[:, idx]
            components[name] = {
                "mean": float(col.mean().item()),
                "std": float(col.std(unbiased=False).item()),
            }

        return components

    def gather_termination_ratio(self, generation_result: Dict[str, Any]) -> float:
        """Gather termination flags and compute ratio."""
        terminated_flags = generation_result.get("terminated_with_eos")
        if terminated_flags is None:
            return 0.0

        term = terminated_flags.detach().float()
        if self.world_size > 1:
            term = self.accelerator.gather(term)

        return float(term.mean().item())

    def gather_flag(self, local_flag: bool) -> bool:
        """Gather boolean flag across ranks (returns True if any rank is True)."""
        flag_tensor = torch.tensor(
            [1 if local_flag else 0], dtype=torch.int32, device=self.device
        )

        if self.world_size > 1:
            flags = self.accelerator.gather(flag_tensor)
            if flags.device.type != "cpu":
                flags = flags.cpu()
            flags = flags.view(-1)
            return bool(int(flags.max().item()))

        return local_flag

    def synchronize_index(self, idx: int, is_main: bool) -> int:
        """Broadcast index from main process to all ranks."""
        if self.world_size <= 1:
            return idx

        idx_val = int(idx) if is_main else 0
        idx_tensor = torch.tensor([idx_val], dtype=torch.long, device=self.device)
        gathered_idx = self.accelerator.gather(idx_tensor)

        return (
            int(gathered_idx[0].item())
            if gathered_idx.numel() > 0
            else int(idx_tensor.item())
        )


__all__ = ["MetricsAggregator"]
