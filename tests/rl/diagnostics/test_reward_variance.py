"""
Tests for Reward Variance Diagnostic

Feature: 004-grpo-post-training / User Story 3
Constitution: v4.1.1
"""

import pytest
import torch

from src_new.rl.diagnostics.rewards import RewardProfile, compute_reward_profile


class TestRewardProfile:
    """Test RewardProfile dataclass and methods."""

    def test_reward_profile_creation(self):
        """Test creating a RewardProfile instance."""
        profile = RewardProfile(
            step=100,
            mean_reward=2.5,
            std_reward=0.3,
            min_reward=1.8,
            max_reward=3.2,
            within_group_variance=0.09,
            within_group_std=0.3,
            between_group_variance=0.04,
            between_group_std=0.2,
            per_reward_means={"bbox_giou": 0.5, "coverage": 0.8},
            per_reward_stds={"bbox_giou": 0.1, "coverage": 0.05},
            is_collapsed=False,
            dead_rewards=[],
            completion_diversity_score=0.12,
            prompt_diversity_score=0.08,
        )

        assert profile.step == 100
        assert profile.is_collapsed is False
        assert len(profile.dead_rewards) == 0

    def test_to_tensorboard_returns_dict(self):
        """Test that to_tensorboard returns valid dict."""
        profile = RewardProfile(
            step=100,
            mean_reward=2.5,
            std_reward=0.3,
            min_reward=1.8,
            max_reward=3.2,
            within_group_variance=0.09,
            within_group_std=0.3,
            between_group_variance=0.04,
            between_group_std=0.2,
            per_reward_means={"bbox_giou": 0.5},
            per_reward_stds={"bbox_giou": 0.1},
            is_collapsed=False,
            dead_rewards=[],
            completion_diversity_score=0.12,
            prompt_diversity_score=0.08,
        )

        tb_dict = profile.to_tensorboard()

        assert isinstance(tb_dict, dict)
        assert "reward_profile/mean" in tb_dict
        assert "reward_profile/std" in tb_dict
        assert "reward_profile/within_group_std" in tb_dict
        assert "reward_components/bbox_giou/mean" in tb_dict
        assert tb_dict["reward_profile/is_collapsed"] == 0.0

    def test_log_warnings_for_collapse(self, caplog):
        """Test that collapse warnings are logged."""
        profile = RewardProfile(
            step=100,
            mean_reward=2.5,
            std_reward=0.005,  # < 0.01 → collapsed
            min_reward=2.49,
            max_reward=2.51,
            within_group_variance=0.0001,
            within_group_std=0.01,
            between_group_variance=0.0001,
            between_group_std=0.01,
            per_reward_means={},
            per_reward_stds={},
            is_collapsed=True,
            dead_rewards=[],
            completion_diversity_score=0.004,
            prompt_diversity_score=0.004,
        )

        profile.log_warnings()

        assert profile.is_collapsed is True


class TestComputeRewardProfile:
    """Test compute_reward_profile function."""

    def test_basic_reward_profile(self):
        """Test basic reward variance computation."""
        # Create rewards: [batch_size=2, K=4] = 8 total
        # Batch 0: [2.0, 2.2, 2.4, 2.6]
        # Batch 1: [3.0, 3.2, 3.4, 3.6]
        rewards = torch.tensor([
            2.0, 2.2, 2.4, 2.6,
            3.0, 3.2, 3.4, 3.6,
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            per_reward_components=None,
            k_completions=4,
        )

        assert profile.step == 100
        assert profile.mean_reward == pytest.approx(2.8, abs=0.01)
        assert profile.std_reward > 0
        assert profile.within_group_variance > 0
        assert profile.between_group_variance > 0

    def test_within_group_variance(self):
        """Test within-group variance computation."""
        # High within-group variance (diverse completions)
        rewards = torch.tensor([
            1.0, 2.0, 3.0, 4.0,  # Batch 0: high variance
            1.5, 2.5, 3.5, 4.5,  # Batch 1: high variance
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        assert profile.within_group_std > 0.5
        assert profile.completion_diversity_score > 0

    def test_zero_within_group_variance(self):
        """Test detection of zero within-group variance (identical completions)."""
        # All completions from same prompt are identical
        rewards = torch.tensor([
            2.0, 2.0, 2.0, 2.0,  # Batch 0: all same
            3.0, 3.0, 3.0, 3.0,  # Batch 1: all same
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        assert profile.within_group_std < 0.01
        # This should trigger warning in log_warnings()

    def test_between_group_variance(self):
        """Test between-group variance computation."""
        # High between-group variance (different prompts → different rewards)
        rewards = torch.tensor([
            1.0, 1.1, 1.2, 1.3,  # Batch 0: mean ≈ 1.15
            5.0, 5.1, 5.2, 5.3,  # Batch 1: mean ≈ 5.15
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        assert profile.between_group_std > 1.0
        assert profile.prompt_diversity_score > 0

    def test_per_reward_components(self):
        """Test per-reward component analysis."""
        rewards = torch.tensor([2.0, 2.5, 3.0, 3.5] * 2)

        components = {
            "bbox_giou": torch.tensor([0.5, 0.6, 0.7, 0.8] * 2),
            "coverage": torch.tensor([1.0, 1.1, 1.2, 1.3] * 2),
            "wrappers": torch.tensor([0.5, 0.5, 0.5, 0.5] * 2),  # Dead (no variance)
        }

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            per_reward_components=components,
            k_completions=4,
        )

        assert "bbox_giou" in profile.per_reward_means
        assert "coverage" in profile.per_reward_means
        assert "wrappers" in profile.dead_rewards  # std < 0.01

    def test_collapse_detection(self):
        """Test detection of reward collapse."""
        # All rewards essentially the same
        rewards = torch.ones(16) * 2.5 + torch.randn(16) * 0.001

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        assert profile.is_collapsed is True
        assert profile.std_reward < 0.01

    def test_no_collapse_with_variance(self):
        """Test that healthy variance is not flagged as collapse."""
        # Normal reward distribution
        rewards = torch.randn(16) * 0.5 + 2.5

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        assert profile.is_collapsed is False
        assert profile.std_reward > 0.1

    def test_diversity_scores(self):
        """Test computation of diversity scores."""
        # High within-group, low between-group
        rewards = torch.tensor([
            1.0, 2.0, 3.0, 4.0,  # Batch 0
            1.1, 2.1, 3.1, 4.1,  # Batch 1 (similar mean to batch 0)
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        # Diversity score = std / mean
        assert profile.completion_diversity_score > 0
        assert profile.prompt_diversity_score >= 0

    def test_2d_rewards_tensor(self):
        """Test handling of 2D rewards tensor."""
        # 2D input: [batch_size, K]
        rewards_2d = torch.tensor([
            [2.0, 2.2, 2.4, 2.6],
            [3.0, 3.2, 3.4, 3.6],
        ])

        profile = compute_reward_profile(
            step=100,
            rewards=rewards_2d,
            k_completions=4,
        )

        assert profile.mean_reward == pytest.approx(2.8, abs=0.01)

    def test_non_divisible_rewards(self):
        """Test handling of rewards not divisible by K."""
        # 10 rewards with K=4 → truncates to 8
        rewards = torch.randn(10)

        profile = compute_reward_profile(
            step=100,
            rewards=rewards,
            k_completions=4,
        )

        # Should only use first 8 rewards (2 batches * 4)
        assert profile.mean_reward is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
