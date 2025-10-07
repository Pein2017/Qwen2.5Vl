#!/usr/bin/env python3
"""Centralized reward logging utility for RL training.

Provides unified logging of reward metrics to console and TensorBoard without
hardcoded reward lists. All rewards are discovered dynamically from reward_names.
"""

from __future__ import annotations

from typing import Any, Dict, List


class RewardLogger:
    """Unified reward logging for console and TensorBoard.

    Automatically logs all rewards defined in reward_names without requiring
    hardcoded lists. Uses full registry names for all logging.

    Args:
        reward_names: List of reward keys (from registry) to log
    """

    def __init__(self, reward_names: List[str]):
        self.reward_names = list(reward_names)

    def format_console_summary(self, logs: Dict[str, Any]) -> str:
        """Build reward portion of console summary string.

        Returns space-separated reward metrics in format: name=value
        Only includes rewards present in logs.

        Args:
            logs: Dictionary containing reward metrics with keys like "rewards/{name}/mean"

        Returns:
            Formatted string like "parse=1.000 bbox_giou=0.632 caption_f1=0.418"
        """
        parts: List[str] = []

        for name in self.reward_names:
            mean_key = f"rewards/{name}/mean"
            if mean_key in logs:
                value = float(logs[mean_key])
                parts.append(f"{name}={value:.3f}")

        return " ".join(parts)

    def log_to_tensorboard(
        self,
        writer: Any,
        logs: Dict[str, Any],
        step: int,
    ) -> None:
        """Log all reward metrics to TensorBoard.

        Logs both mean and std for each reward under rewards/{name}/{stat}.

        Args:
            writer: TensorBoard SummaryWriter instance
            logs: Dictionary containing reward metrics
            step: Global training step for x-axis
        """
        if writer is None:
            return

        for name in self.reward_names:
            for stat in ["mean", "std"]:
                key = f"rewards/{name}/{stat}"
                if key in logs:
                    writer.add_scalar(key, logs[key], step)

    def get_reward_stats(self, logs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract reward statistics from logs as structured dict.

        Useful for saving to checkpoints or exporting metrics.

        Args:
            logs: Dictionary containing reward metrics

        Returns:
            Dict mapping reward name to {"mean": float, "std": float}
        """
        stats: Dict[str, Dict[str, float]] = {}

        for name in self.reward_names:
            mean_key = f"rewards/{name}/mean"
            std_key = f"rewards/{name}/std"

            if mean_key in logs or std_key in logs:
                stats[name] = {}
                if mean_key in logs:
                    stats[name]["mean"] = float(logs[mean_key])
                if std_key in logs:
                    stats[name]["std"] = float(logs[std_key])

        return stats


__all__ = ["RewardLogger"]
