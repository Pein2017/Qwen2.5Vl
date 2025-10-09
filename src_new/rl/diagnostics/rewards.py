"""
Reward Variance Diagnostic

Feature: 004-grpo-post-training / User Story 3
Constitution: v4.1.1

Tracks reward variance to diagnose:
1. Reward collapse (all rewards same value)
2. Dead rewards (zero weight or zero variance)
3. Diversity issues (within-group variance = 0)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src_new.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RewardProfile:
    """Reward variance analysis results."""
    
    step: int
    
    # Aggregate statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    # Within-group variance (same prompt, K completions)
    within_group_variance: float
    within_group_std: float
    
    # Between-group variance (different prompts)
    between_group_variance: float
    between_group_std: float
    
    # Per-reward breakdown
    per_reward_means: Dict[str, float]
    per_reward_stds: Dict[str, float]
    
    # Collapse detection
    is_collapsed: bool  # True if std < 0.01
    dead_rewards: List[str]  # Rewards with std < 0.01
    
    # Diversity metrics
    completion_diversity_score: float  # within-group std / mean
    prompt_diversity_score: float  # between-group std / mean
    
    def to_tensorboard(self) -> Dict[str, float]:
        """Convert to TensorBoard scalar dict."""
        base_scalars = {
            "reward_profile/mean": self.mean_reward,
            "reward_profile/std": self.std_reward,
            "reward_profile/min": self.min_reward,
            "reward_profile/max": self.max_reward,
            "reward_profile/within_group_variance": self.within_group_variance,
            "reward_profile/within_group_std": self.within_group_std,
            "reward_profile/between_group_variance": self.between_group_variance,
            "reward_profile/between_group_std": self.between_group_std,
            "reward_profile/is_collapsed": float(self.is_collapsed),
            "reward_profile/num_dead_rewards": float(len(self.dead_rewards)),
            "reward_profile/completion_diversity": self.completion_diversity_score,
            "reward_profile/prompt_diversity": self.prompt_diversity_score,
        }
        
        # Add per-reward means and stds
        for name, value in self.per_reward_means.items():
            base_scalars[f"reward_components/{name}/mean"] = value
        for name, value in self.per_reward_stds.items():
            base_scalars[f"reward_components/{name}/std"] = value
        
        return base_scalars
    
    def log_warnings(self) -> None:
        """Log warnings for reward issues."""
        if self.is_collapsed:
            logger.error(
                f"[Reward Profile] Step {self.step}: COLLAPSED - std={self.std_reward:.6f} < 0.01"
            )
        
        if self.dead_rewards:
            logger.warning(
                f"[Reward Profile] Step {self.step}: {len(self.dead_rewards)} dead rewards "
                f"(std < 0.01): {', '.join(self.dead_rewards)}"
            )
        
        if self.within_group_std < 0.01:
            logger.error(
                f"[Reward Profile] Step {self.step}: Zero within-group variance - "
                f"completions are IDENTICAL (diversity issue)"
            )
        
        if self.completion_diversity_score < 0.05:
            logger.warning(
                f"[Reward Profile] Step {self.step}: Low completion diversity "
                f"(score={self.completion_diversity_score:.4f})"
            )


def compute_reward_profile(
    step: int,
    rewards: torch.Tensor,
    per_reward_components: Optional[Dict[str, torch.Tensor]] = None,
    k_completions: int = 8,
) -> RewardProfile:
    """
    Compute reward variance profile from total rewards and components.
    
    Args:
        step: Current training step
        rewards: Total rewards [batch_size * K] or [batch_size, K]
        per_reward_components: Dict of component rewards (same shape as rewards)
        k_completions: Number of completions per prompt (default 8)
        
    Returns:
        RewardProfile with variance analysis
    """
    # Flatten if needed
    if rewards.dim() == 2:
        batch_size = rewards.size(0)
        rewards_flat = rewards.view(-1)
    else:
        rewards_flat = rewards
        batch_size = rewards_flat.size(0) // k_completions
    
    # Reshape to [batch_size, K]
    if rewards_flat.size(0) % k_completions != 0:
        logger.warning(
            f"Rewards size {rewards_flat.size(0)} not divisible by k={k_completions}, "
            f"truncating to nearest multiple"
        )
        truncate_size = (rewards_flat.size(0) // k_completions) * k_completions
        rewards_flat = rewards_flat[:truncate_size]
        batch_size = truncate_size // k_completions
    
    rewards_reshaped = rewards_flat.view(batch_size, k_completions)
    
    # Aggregate statistics
    mean_reward = rewards_flat.mean().item()
    std_reward = rewards_flat.std().item()
    min_reward = rewards_flat.min().item()
    max_reward = rewards_flat.max().item()
    
    # Within-group variance (across completions for same prompt)
    within_group_vars = rewards_reshaped.var(dim=1, unbiased=False)  # [batch_size]
    within_group_variance = within_group_vars.mean().item()
    within_group_std = within_group_variance ** 0.5
    
    # Between-group variance (across prompts)
    prompt_means = rewards_reshaped.mean(dim=1)  # [batch_size]
    between_group_variance = prompt_means.var(unbiased=False).item()
    between_group_std = between_group_variance ** 0.5
    
    # Collapse detection
    is_collapsed = std_reward < 0.01
    
    # Per-reward component analysis
    per_reward_means = {}
    per_reward_stds = {}
    dead_rewards = []
    
    if per_reward_components is not None:
        for name, component_tensor in per_reward_components.items():
            # Flatten if needed
            if component_tensor.dim() == 2:
                component_flat = component_tensor.view(-1)
            else:
                component_flat = component_tensor
            
            # Truncate to match rewards
            if component_flat.size(0) > rewards_flat.size(0):
                component_flat = component_flat[: rewards_flat.size(0)]
            
            comp_mean = component_flat.mean().item()
            comp_std = component_flat.std().item()
            
            per_reward_means[name] = comp_mean
            per_reward_stds[name] = comp_std
            
            if comp_std < 0.01:
                dead_rewards.append(name)
    
    # Diversity scores
    completion_diversity_score = (
        within_group_std / abs(mean_reward) if abs(mean_reward) > 1e-9 else 0.0
    )
    prompt_diversity_score = (
        between_group_std / abs(mean_reward) if abs(mean_reward) > 1e-9 else 0.0
    )
    
    return RewardProfile(
        step=step,
        mean_reward=mean_reward,
        std_reward=std_reward,
        min_reward=min_reward,
        max_reward=max_reward,
        within_group_variance=within_group_variance,
        within_group_std=within_group_std,
        between_group_variance=between_group_variance,
        between_group_std=between_group_std,
        per_reward_means=per_reward_means,
        per_reward_stds=per_reward_stds,
        is_collapsed=is_collapsed,
        dead_rewards=dead_rewards,
        completion_diversity_score=completion_diversity_score,
        prompt_diversity_score=prompt_diversity_score,
    )
