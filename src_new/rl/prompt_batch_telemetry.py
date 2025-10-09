"""Telemetry tracker for prompt batch accumulation metrics.

Tracks per-cycle metrics for prompt batching GRPO training:
- Fill ratio: collected trajectories / expected trajectories
- Reward average: moving average over reward_average_window cycles
- Trajectories collected: total valid trajectories per cycle
- Dropped prompts: prompts discarded via drop_last
- Invalid fraction: invalid trajectories / total trajectories
"""

from collections import deque
from typing import Dict, Optional

import torch


class PromptBatchTelemetryTracker:
    """Tracks prompt batch accumulation telemetry across training cycles."""

    def __init__(
        self,
        expected_trajectories_per_cycle: int,
        reward_average_window: int = 5,
    ):
        """Initialize telemetry tracker.

        Args:
            expected_trajectories_per_cycle: Expected number of trajectories per cycle
            reward_average_window: Number of cycles to average rewards over
        """
        self.expected_trajectories = expected_trajectories_per_cycle
        self.reward_average_window = reward_average_window

        # Reward history for moving average
        self.reward_history: deque = deque(maxlen=reward_average_window)

        # Current cycle metrics
        self.cycle_step_count = 0
        self.reset_cycle()

    def reset_cycle(self) -> None:
        """Reset metrics for a new accumulation cycle."""
        self.trajectories_collected = 0
        self.invalid_count = 0
        self.dropped_prompts = 0
        self.cycle_rewards: list = []

    def register_trajectories(
        self, count: int, rewards: Optional[torch.Tensor] = None, is_valid: bool = True
    ) -> None:
        """Register trajectories for the current cycle.

        Args:
            count: Number of trajectories to register
            rewards: Optional tensor of rewards for these trajectories
            is_valid: Whether trajectories are valid
        """
        self.trajectories_collected += count
        if not is_valid:
            self.invalid_count += count
        elif rewards is not None:
            # Store rewards for averaging
            if rewards.numel() > 0:
                self.cycle_rewards.extend(rewards.flatten().tolist())

    def register_dropped_prompts(self, count: int) -> None:
        """Register dropped prompts due to drop_last.

        Args:
            count: Number of prompts dropped
        """
        self.dropped_prompts = count

    def compute_cycle_metrics(self) -> Dict[str, float]:
        """Compute metrics for the completed cycle.

        Returns:
            Dictionary of telemetry metrics
        """
        # Fill ratio
        fill_ratio = (
            self.trajectories_collected / self.expected_trajectories
            if self.expected_trajectories > 0
            else 0.0
        )

        # Invalid fraction
        invalid_fraction = (
            self.invalid_count / self.trajectories_collected
            if self.trajectories_collected > 0
            else 0.0
        )

        # Average reward for this cycle
        cycle_reward_mean = (
            sum(self.cycle_rewards) / len(self.cycle_rewards)
            if self.cycle_rewards
            else 0.0
        )

        # Update reward history and compute smoothed average
        if self.trajectories_collected > 0:
            self.reward_history.append(cycle_reward_mean)

        reward_average = (
            sum(self.reward_history) / len(self.reward_history)
            if self.reward_history
            else 0.0
        )

        # Increment cycle counter
        self.cycle_step_count += 1

        metrics = {
            "fill_ratio": fill_ratio,
            "reward_average": reward_average,
            "invalid_fraction": invalid_fraction,
            "trajectories_collected": float(self.trajectories_collected),
            "dropped_prompts": float(self.dropped_prompts),
            "steps": float(self.cycle_step_count),
        }

        # Reset for next cycle
        self.reset_cycle()

        return metrics

    def get_current_reward_window_size(self) -> int:
        """Get the current size of the reward history window.

        Returns:
            Number of rewards currently in the history
        """
        return len(self.reward_history)
