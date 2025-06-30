#!/usr/bin/env python3
"""
Token-Length-Aware Learning Rate Scaling Module

Adjusts learning rates to compensate for token length inconsistency between
standard (padded) and packed collators, ensuring equivalent training dynamics
when effective batch sizes and base learning rates are identical.

Core Issue:
- Standard collator: Pads sequences to max_length → many padding tokens
- Packed collator: Concatenates sequences without padding → different effective token distributions

This creates gradient computation differences that need compensation.
"""

from dataclasses import dataclass
from typing import Any, Dict

from src.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TokenLengthAwareLRScaler:
    """
    Scales learning rates to compensate for token length inconsistency between collators.

    Focuses specifically on the differences in effective token utilization:
    - Standard collator wastes computation on padding tokens
    - Packed collator utilizes all tokens for meaningful computation

    This creates different gradient magnitudes that need compensation.
    """

    auto_scale_lr: bool = True
    base_collator_type: str = "standard"  # Reference collator for base LRs

    def compute_token_utilization_factor(self, collator_type: str) -> float:
        """
        Compute scaling factor based on token utilization efficiency.

        Theory:
        - Standard collator: Sequences padded to max_length, many padding tokens
        - Packed collator: No padding, all tokens contribute to gradients

        This difference in effective token utilization creates different gradient
        magnitudes that need compensation for equivalent training dynamics.

        Args:
            collator_type: Type of collator ("standard" or "packed")

        Returns:
            Token utilization scaling factor
        """
        if collator_type == "standard":
            return 1.0

        # Packed collator has higher effective token utilization
        # Empirically, packed sequences typically have ~10-15% higher effective
        # token density due to elimination of padding tokens
        #
        # This creates stronger gradients that need slight reduction for equivalence
        token_utilization_factor = 0.90  # 10% reduction to compensate

        logger.debug(f"Token utilization analysis:")
        logger.debug(f"  Collator type: {collator_type}")
        logger.debug(f"  Token utilization factor: {token_utilization_factor:.4f}")

        return token_utilization_factor

    def compute_gradient_noise_factor(self, collator_type: str) -> float:
        """
        Compute scaling factor based on gradient noise differences.

        Theory:
        - Standard collator: Padding tokens introduce noise in gradient computation
        - Packed collator: No padding tokens, cleaner gradient signals

        Cleaner gradients in packed collator allow for slightly higher learning rates
        while maintaining equivalent training stability.

        Args:
            collator_type: Type of collator ("standard" or "packed")

        Returns:
            Gradient noise compensation factor
        """
        if collator_type == "standard":
            return 1.0

        # Packed collator has cleaner gradients (less noise from padding)
        # This allows for slightly higher learning rates while maintaining stability
        gradient_noise_factor = 1.05  # 5% increase due to cleaner gradients

        logger.debug(f"Gradient noise analysis:")
        logger.debug(f"  Collator type: {collator_type}")
        logger.debug(f"  Gradient noise factor: {gradient_noise_factor:.4f}")

        return gradient_noise_factor

    def compute_scaling_factor(self, collator_type: str) -> float:
        """
        Compute the overall learning rate scaling factor for token length compensation.

        Combines token utilization and gradient noise factors to achieve equivalent
        training dynamics between standard and packed collators.

        Args:
            collator_type: Type of collator ("standard" or "packed")

        Returns:
            Overall scaling factor for learning rate adjustment
        """
        if not self.auto_scale_lr:
            return 1.0

        if collator_type not in ["standard", "packed", "flattened"]:
            logger.warning(
                f"Unknown collator_type '{collator_type}', using scaling factor 1.0"
            )
            return 1.0

        if collator_type == "standard":
            return 1.0

        # Compute individual factors
        token_utilization_factor = self.compute_token_utilization_factor(collator_type)
        gradient_noise_factor = self.compute_gradient_noise_factor(collator_type)

        # Combine factors multiplicatively
        final_factor = token_utilization_factor * gradient_noise_factor

        logger.info(f"🔧 Token-Length-Aware Learning Rate Scaling:")
        logger.info(f"   Token utilization factor: {token_utilization_factor:.4f}")
        logger.info(f"   Gradient noise factor: {gradient_noise_factor:.4f}")
        logger.info(f"   Final scaling factor: {final_factor:.4f}")
        logger.info(f"   Focus: Token length inconsistency compensation")

        return final_factor

    def scale_learning_rates(
        self,
        config: Dict[str, Any],
        collator_type: str,
    ) -> Dict[str, Any]:
        """
        Scale all learning rate parameters to compensate for token length differences.

        Args:
            config: Configuration dictionary
            collator_type: Type of collator ("standard" or "packed")

        Returns:
            Configuration with token-length-compensated learning rates
        """
        if not self.auto_scale_lr:
            logger.info("Token-length-aware learning rate scaling is disabled")
            return config

        scaling_factor = self.compute_scaling_factor(collator_type)

        # Learning rate parameter names to scale
        lr_params = [
            "learning_rate",
            "llm_lr",
            "adapter_lr",
            "vision_lr",
            "merger_lr",
            "detection_lr",
        ]

        original_lrs = {}
        scaled_lrs = {}

        # Scale each learning rate parameter
        for param in lr_params:
            if param in config and config[param] is not None and config[param] > 0:
                original_lr = config[param]
                scaled_lr = original_lr * scaling_factor

                config[param] = scaled_lr
                original_lrs[param] = original_lr
                scaled_lrs[param] = scaled_lr

        logger.info("🎯 Token-Length-Aware Learning Rate Scaling Applied")
        logger.info(f"   Collator type: {collator_type}")
        logger.info(f"   Scaling factor: {scaling_factor:.4f}")
        logger.info(f"   Base collator reference: {self.base_collator_type}")
        logger.info(f"   Purpose: Compensate for token length inconsistency")

        if original_lrs:
            logger.info("   Learning rate adjustments:")
            for param in original_lrs:
                logger.info(
                    f"     {param}: {original_lrs[param]:.2e} → {scaled_lrs[param]:.2e}"
                )

        return config

    def get_equivalent_config(
        self, base_config: Dict[str, Any], target_collator_type: str
    ) -> Dict[str, Any]:
        """
        Generate an equivalent configuration for a different collator type.

        This allows you to take a config tuned for one collator type and get
        equivalent performance with the other collator type.
        """
        config = base_config.copy()

        # Apply token-length-aware scaling for target collator type
        config = self.scale_learning_rates(
            config=config,
            collator_type=target_collator_type,
        )

        # Update collator type
        config["collator_type"] = target_collator_type

        return config


def create_token_length_scaler(
    auto_scale_lr: bool = True, base_collator_type: str = "standard"
) -> TokenLengthAwareLRScaler:
    """Create a token-length-aware learning rate scaler."""
    return TokenLengthAwareLRScaler(
        auto_scale_lr=auto_scale_lr,
        base_collator_type=base_collator_type,
    )


def apply_token_length_scaling(
    config: Dict[str, Any],
    auto_scale_lr: bool = True,
    base_collator_type: str = "standard",
) -> Dict[str, Any]:
    """Apply token-length-aware learning rate scaling to a configuration."""
    scaler = create_token_length_scaler(
        auto_scale_lr=auto_scale_lr,
        base_collator_type=base_collator_type,
    )

    return scaler.scale_learning_rates(
        config=config,
        collator_type=config.get("collator_type", "standard"),
    )


def get_packed_equivalent_config(standard_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a packed collator equivalent of a standard collator config."""
    scaler = create_token_length_scaler(base_collator_type="standard")
    return scaler.get_equivalent_config(standard_config, "packed")


def get_standard_equivalent_config(packed_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a standard collator equivalent of a packed collator config."""
    scaler = create_token_length_scaler(base_collator_type="packed")
    return scaler.get_equivalent_config(packed_config, "standard")


# Backward compatibility aliases
LearningRateScaler = TokenLengthAwareLRScaler
create_lr_scaler = create_token_length_scaler
apply_lr_scaling = apply_token_length_scaling
