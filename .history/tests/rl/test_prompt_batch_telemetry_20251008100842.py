"""Unit tests for prompt batch telemetry metrics.

This test suite validates that telemetry correctly reports:
- Fill ratio (collected/expected trajectories)
- Smoothed reward average (moving window)
- Invalid fraction (bad trajectories/total)
- Trajectories collected per cycle
- Dropped prompts count
"""

from collections import deque
from typing import Any, Dict

import pytest


class RewardTrendTracker:
    """Mock tracker for smoothed reward metrics."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.reward_history: deque = deque(maxlen=window_size)

    def update(self, reward: float) -> None:
        """Add a new reward to the history."""
        self.reward_history.append(reward)

    def get_average(self) -> float:
        """Compute the smoothed reward average."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    def get_count(self) -> int:
        """Get the number of rewards in the window."""
        return len(self.reward_history)


class PromptBatchTelemetry:
    """Mock telemetry aggregator for prompt batch metrics."""

    def __init__(self, expected_trajectories: int, reward_window: int = 5):
        self.expected_trajectories = expected_trajectories
        self.reward_tracker = RewardTrendTracker(window_size=reward_window)
        self.current_cycle: Dict[str, Any] = self._reset_cycle()

    def _reset_cycle(self) -> Dict[str, Any]:
        """Reset current cycle metrics."""
        return {
            "trajectories_collected": 0,
            "invalid_count": 0,
            "dropped_prompts": 0,
            "total_reward": 0.0,
        }

    def register_trajectory(self, is_valid: bool, reward: float = 0.0) -> None:
        """Register a trajectory with validity and reward."""
        self.current_cycle["trajectories_collected"] += 1
        if not is_valid:
            self.current_cycle["invalid_count"] += 1
        else:
            self.current_cycle["total_reward"] += reward

    def register_dropped_prompts(self, count: int) -> None:
        """Register dropped prompts for the cycle."""
        self.current_cycle["dropped_prompts"] = count

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics for the cycle."""
        collected = self.current_cycle["trajectories_collected"]
        invalid = self.current_cycle["invalid_count"]

        # Fill ratio
        fill_ratio = (
            collected / self.expected_trajectories
            if self.expected_trajectories > 0
            else 0.0
        )

        # Invalid fraction
        invalid_fraction = invalid / collected if collected > 0 else 0.0

        # Average reward for valid trajectories
        valid_count = collected - invalid
        avg_reward = (
            self.current_cycle["total_reward"] / valid_count if valid_count > 0 else 0.0
        )

        # Update reward history
        if collected > 0:
            self.reward_tracker.update(avg_reward)

        # Smoothed reward
        reward_average = self.reward_tracker.get_average()

        metrics = {
            "fill_ratio": fill_ratio,
            "reward_average": reward_average,
            "invalid_fraction": invalid_fraction,
            "trajectories_collected": float(collected),
            "dropped_prompts": float(self.current_cycle["dropped_prompts"]),
        }

        # Reset for next cycle
        self.current_cycle = self._reset_cycle()

        return metrics


def test_fill_ratio_full_cycle():
    """Test fill ratio when all expected trajectories are collected."""
    telemetry = PromptBatchTelemetry(expected_trajectories=32)

    # Collect all 32 trajectories
    for _ in range(32):
        telemetry.register_trajectory(is_valid=True, reward=0.5)

    metrics = telemetry.compute_metrics()
    assert metrics["fill_ratio"] == 1.0
    assert metrics["trajectories_collected"] == 32.0


def test_fill_ratio_partial_cycle():
    """Test fill ratio when fewer trajectories are collected."""
    telemetry = PromptBatchTelemetry(expected_trajectories=32)

    # Only collect 24 trajectories
    for _ in range(24):
        telemetry.register_trajectory(is_valid=True, reward=0.5)

    metrics = telemetry.compute_metrics()
    assert metrics["fill_ratio"] == 0.75  # 24/32
    assert metrics["trajectories_collected"] == 24.0


def test_invalid_fraction_all_valid():
    """Test invalid fraction when all trajectories are valid."""
    telemetry = PromptBatchTelemetry(expected_trajectories=16)

    for _ in range(16):
        telemetry.register_trajectory(is_valid=True, reward=0.6)

    metrics = telemetry.compute_metrics()
    assert metrics["invalid_fraction"] == 0.0
    assert metrics["trajectories_collected"] == 16.0


def test_invalid_fraction_some_invalid():
    """Test invalid fraction when some trajectories are invalid."""
    telemetry = PromptBatchTelemetry(expected_trajectories=20)

    # 15 valid, 5 invalid
    for _ in range(15):
        telemetry.register_trajectory(is_valid=True, reward=0.7)
    for _ in range(5):
        telemetry.register_trajectory(is_valid=False, reward=0.0)

    metrics = telemetry.compute_metrics()
    assert metrics["invalid_fraction"] == 0.25  # 5/20
    assert metrics["trajectories_collected"] == 20.0


def test_reward_average_single_cycle():
    """Test reward average for a single cycle."""
    telemetry = PromptBatchTelemetry(expected_trajectories=10, reward_window=5)

    for _ in range(10):
        telemetry.register_trajectory(is_valid=True, reward=0.8)

    metrics = telemetry.compute_metrics()
    assert metrics["reward_average"] == pytest.approx(0.8)


def test_reward_average_moving_window():
    """Test reward average over multiple cycles with moving window."""
    telemetry = PromptBatchTelemetry(expected_trajectories=8, reward_window=3)

    # Cycle 1: reward = 0.5
    for _ in range(8):
        telemetry.register_trajectory(is_valid=True, reward=0.5)
    metrics1 = telemetry.compute_metrics()
    assert metrics1["reward_average"] == 0.5

    # Cycle 2: reward = 0.7
    for _ in range(8):
        telemetry.register_trajectory(is_valid=True, reward=0.7)
    metrics2 = telemetry.compute_metrics()
    assert metrics2["reward_average"] == pytest.approx((0.5 + 0.7) / 2)

    # Cycle 3: reward = 0.9
    for _ in range(8):
        telemetry.register_trajectory(is_valid=True, reward=0.9)
    metrics3 = telemetry.compute_metrics()
    assert metrics3["reward_average"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)

    # Cycle 4: reward = 0.6 (window=3, oldest 0.5 drops out)
    for _ in range(8):
        telemetry.register_trajectory(is_valid=True, reward=0.6)
    metrics4 = telemetry.compute_metrics()
    assert metrics4["reward_average"] == pytest.approx((0.7 + 0.9 + 0.6) / 3)


def test_dropped_prompts_tracking():
    """Test tracking of dropped prompts."""
    telemetry = PromptBatchTelemetry(expected_trajectories=32)

    # Collect partial trajectories
    for _ in range(16):
        telemetry.register_trajectory(is_valid=True, reward=0.5)

    # Register 2 dropped prompts
    telemetry.register_dropped_prompts(2)

    metrics = telemetry.compute_metrics()
    assert metrics["dropped_prompts"] == 2.0
    assert metrics["fill_ratio"] == 0.5  # 16/32


def test_zero_division_safety():
    """Test telemetry handles edge cases without division errors."""
    telemetry = PromptBatchTelemetry(expected_trajectories=0)

    metrics = telemetry.compute_metrics()
    assert metrics["fill_ratio"] == 0.0
    assert metrics["invalid_fraction"] == 0.0
    assert metrics["reward_average"] == 0.0
    assert metrics["trajectories_collected"] == 0.0


def test_all_invalid_trajectories():
    """Test telemetry when all trajectories are invalid."""
    telemetry = PromptBatchTelemetry(expected_trajectories=10)

    for _ in range(10):
        telemetry.register_trajectory(is_valid=False, reward=0.0)

    metrics = telemetry.compute_metrics()
    assert metrics["invalid_fraction"] == 1.0
    assert metrics["fill_ratio"] == 1.0
    assert metrics["reward_average"] == 0.0  # No valid trajectories


def test_cycle_reset():
    """Test that metrics reset properly between cycles."""
    telemetry = PromptBatchTelemetry(expected_trajectories=16)

    # Cycle 1
    for _ in range(16):
        telemetry.register_trajectory(is_valid=True, reward=0.6)
    telemetry.register_dropped_prompts(1)
    metrics1 = telemetry.compute_metrics()

    # Cycle 2 should start fresh
    for _ in range(8):
        telemetry.register_trajectory(is_valid=True, reward=0.9)
    metrics2 = telemetry.compute_metrics()

    assert metrics1["trajectories_collected"] == 16.0
    assert metrics1["dropped_prompts"] == 1.0
    assert metrics2["trajectories_collected"] == 8.0
    assert metrics2["dropped_prompts"] == 0.0  # Reset
    assert metrics2["fill_ratio"] == 0.5  # 8/16
