"""
Loss Manager Module

This module handles loss computation, tracking, validation, and multi-task loss
combination for the Qwen2.5-VL model with coordinate token support.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.logger_utils import get_training_logger


class LossManager:
    """
    Handles loss management for the Qwen2.5-VL model with coordinate support.

    Responsibilities:
    - Loss component tracking and validation
    - Multi-task loss combination (LLM + coordinate)
    - Loss attachment to model outputs
    - Loss component reset and initialization
    """

    def __init__(
        self,
        coordinate_config: Any,
        coordinate_handler: Any,
        detection_integration: Any,
        logger: Optional[Any] = None,
    ):
        """Initialize loss manager."""
        self.coordinate_config = coordinate_config
        self.coordinate_handler = coordinate_handler
        self.detection_integration = detection_integration
        self.logger = logger or get_training_logger()

        # Initialize loss tracking components
        self._initialize_loss_tracking_components()

    def _initialize_loss_tracking_components(self):
        """Initialize loss tracking components."""
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0
        self._last_coordinate_losses = None

    def ensure_loss_tracking_initialized(self):
        """Ensure loss tracking components are initialized."""
        if not hasattr(self, "_last_llm_loss"):
            self._initialize_loss_tracking_components()

    def reset_loss_components_for_forward_pass(self):
        """Reset loss components at the start of a forward pass."""
        self.ensure_loss_tracking_initialized()

        # Reset individual loss components
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        # CRITICAL: Do NOT reset _last_coordinate_losses here!
        # The _last_coordinate_losses dict is only updated after coordinate computation

    def reset_loss_components_to_zero(self):
        """Reset all loss components to zero."""
        self.ensure_loss_tracking_initialized()

        # Reset all components to zero
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

    def compute_coordinate_aware_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, coord_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coordinate-aware loss combining LLM and coordinate losses.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]

        Returns:
            Combined loss tensor
        """
        device = logits.device

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_coord_mask = coord_mask[..., 1:].contiguous()

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)

        # Compute regular (non-coordinate) loss
        regular_mask = ~shift_coord_mask
        if regular_mask.any():
            regular_logits = shift_logits[regular_mask]
            regular_labels = shift_labels[regular_mask]

            if len(regular_logits) > 0:
                regular_loss = F.cross_entropy(
                    regular_logits, regular_labels, ignore_index=-100
                )
                weighted_regular_loss = (
                    self.coordinate_config.regular_loss_weight * regular_loss
                )
                total_loss += weighted_regular_loss
                self._last_llm_loss = regular_loss.item()

        # Compute coordinate loss if coordinate tokens are present
        if shift_coord_mask.any():
            coord_logits = shift_logits[
                shift_coord_mask
            ]  # [num_coord_tokens, vocab_size]
            coord_labels = shift_labels[shift_coord_mask]  # [num_coord_tokens]

            # Extract coordinate-specific logits (only coordinate token positions)
            coord_start = self.coordinate_handler.original_vocab_size
            coord_end = self.coordinate_handler.extended_vocab_size
            coord_specific_logits = coord_logits[:, coord_start:coord_end]

            # Convert labels to coordinate indices
            coord_indices = coord_labels - coord_start

            # Compute L1 loss directly using soft expectation
            l1_loss = self.detection_integration.compute_l1_loss(
                coord_specific_logits.unsqueeze(0),  # Add batch dim
                coord_indices.unsqueeze(0),  # Add batch dim
                torch.ones_like(coord_indices, dtype=torch.bool).unsqueeze(
                    0
                ),  # Add batch dim
            )

            self.logger.debug(f"      L1 loss: {l1_loss.item():.6f}")

            # Apply improved coordinate loss weight
            try:
                from src.models.improved_coordinate_init import (
                    get_improved_coordinate_loss_weight,
                )

                coordinate_loss_weight = get_improved_coordinate_loss_weight()
                self.logger.debug(
                    f"Using improved coordinate loss weight: {coordinate_loss_weight}"
                )
            except ImportError:
                coordinate_loss_weight = self.coordinate_config.coordinate_loss_weight
                self.logger.debug(
                    f"Using default coordinate loss weight: {coordinate_loss_weight}"
                )

            weighted_coord_loss = coordinate_loss_weight * l1_loss
            total_loss += weighted_coord_loss

            self.logger.debug(
                f"      Weighted coord loss: {weighted_coord_loss.item():.6f}"
            )

            # Update loss tracking - set coordinate loss for extraction by Training LossManager
            self._last_coordinate_l1_loss = l1_loss.item()

        return total_loss

    def update_loss_components_with_validation(self, loss_components: Dict[str, Any]):
        """Update loss components with validation."""
        required_loss_keys = [
            "llm_loss",
            "coordinate_l1_loss",
            "loss",
            "total_tokens",
            "coordinate_tokens",
            "regular_tokens",
        ]

        missing_keys = [key for key in required_loss_keys if key not in loss_components]
        if missing_keys:
            raise ValueError(f"Missing required loss component keys: {missing_keys}")

        # Update validated loss components
        self._last_llm_loss = self._validate_loss_value(
            loss_components["llm_loss"], "llm_loss"
        )
        self._last_coordinate_l1_loss = self._validate_loss_value(
            loss_components["coordinate_l1_loss"], "coordinate_l1_loss"
        )

        # Update token counts
        self._last_total_tokens = max(0, int(loss_components["total_tokens"]))
        self._last_coordinate_tokens = max(0, int(loss_components["coordinate_tokens"]))
        self._last_regular_tokens = max(0, int(loss_components["regular_tokens"]))

    def _validate_loss_value(self, value: Any, component_name: str) -> float:
        """Validate that a loss value is a valid float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{component_name} tensor must be scalar, got shape {value.shape}"
                )
            value = value.item()

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{component_name} must be numeric, got {type(value)}: {value}"
            )

        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{component_name} must be finite, got {value}")

        return float(value)

    def attach_coordinate_losses_to_outputs(self, outputs: Any):
        """Attach coordinate losses to model outputs."""
        self.ensure_loss_tracking_initialized()

        def _ensure_tensor(value):
            """Ensure value is a tensor."""
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, (int, float)):
                device = getattr(outputs, "logits", torch.tensor(0.0)).device
                return torch.tensor(float(value), device=device)
            else:
                raise ValueError(f"Cannot convert {type(value)} to tensor: {value}")

        # Prepare coordinate losses
        coordinate_losses = {
            "_llm_loss": _ensure_tensor(self._last_llm_loss),
            "_coordinate_l1_loss": _ensure_tensor(self._last_coordinate_l1_loss),
        }

        # Attach losses to outputs
        for key, value in coordinate_losses.items():
            setattr(outputs, key, value)

    def validate_loss_attachment(self, outputs: Any):
        """Validate that losses are properly attached to outputs."""
        required_loss_keys = [
            "_llm_loss",
            "_coordinate_l1_loss",
        ]

        missing_keys = []
        for key in required_loss_keys:
            if not hasattr(outputs, key):
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing coordinate losses in outputs: {missing_keys}")

        # Validate that attached losses are tensors
        for key in required_loss_keys:
            value = getattr(outputs, key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Output {key} must be tensor, got {type(value)}: {value}"
                )

    def get_last_coordinate_losses(self) -> Dict[str, torch.Tensor]:
        """Get last coordinate losses for external access."""
        # Ensure we have a device reference
        device = torch.device("cpu")
        if hasattr(self, "_device_ref"):
            device = self._device_ref

        if self._last_coordinate_losses is not None:
            return self._last_coordinate_losses.copy()
        else:
            # Return zero losses as fallback
            return {
                "_llm_loss": torch.tensor(0.0, device=device),
                "_coordinate_l1_loss": torch.tensor(0.0, device=device),
            }

    def update_loss_components(self, loss_components: Dict[str, float]):
        """Update loss components (simplified interface)."""
        self.update_loss_components_with_validation(loss_components)

    def get_loss_summary(self) -> Dict[str, float]:
        """Get summary of current loss components."""
        return {
            "llm_loss": self._last_llm_loss,
            "coordinate_l1_loss": self._last_coordinate_l1_loss,
            "total_tokens": self._last_total_tokens,
            "coordinate_tokens": self._last_coordinate_tokens,
            "regular_tokens": self._last_regular_tokens,
        }
