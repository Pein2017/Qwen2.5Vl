"""
Detection Integration Module

This module handles detection-specific logic, loss computations, and bounding box
operations for the Qwen2.5-VL model with detection capabilities.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.logger_utils import get_training_logger


class DetectionIntegration:
    """
    Handles detection-specific operations for the Qwen2.5-VL model.

    Responsibilities:
    - Detection-specific loss computations
    - Bounding box operations and normalization
    - Advanced detection losses (focal, GIoU)
    - Detection output formatting
    """

    def __init__(
        self,
        coordinate_config: Any,
        coordinate_handler: Any,
        logger: Optional[Any] = None,
    ):
        """Initialize detection integration."""
        self.coordinate_config = coordinate_config
        self.coordinate_handler = coordinate_handler
        self.logger = logger or get_training_logger()

    def compute_l1_loss(
        self,
        coord_logits: torch.Tensor,
        coord_indices: torch.Tensor,
        coord_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 loss for coordinate tokens using soft expectation.

        Args:
            coord_logits: Coordinate token logits [batch_size, seq_len, max_coord_value]
            coord_indices: Target coordinate indices [batch_size, seq_len]
            coord_mask: Mask for coordinate tokens [batch_size, seq_len]

        Returns:
            L1 loss tensor
        """
        if not coord_mask.any():
            return torch.tensor(0.0, device=coord_logits.device)

        # Extract coordinate logits and targets where mask is True
        coord_logits_masked = coord_logits[
            coord_mask
        ]  # [num_coord_tokens, max_coord_value]
        coord_indices_masked = coord_indices[coord_mask]  # [num_coord_tokens]

        # Extract expected coordinates using soft expectation
        expected_coords = self.coordinate_handler.extract_expected_coordinates(
            coord_logits_masked
        )

        # Compute L1 loss between expected and target coordinates
        l1_loss = F.l1_loss(expected_coords, coord_indices_masked.float())

        # Normalize by number of coordinate tokens
        num_coord_tokens = coord_mask.sum().item()
        if num_coord_tokens > 0:
            l1_loss = l1_loss / num_coord_tokens

        return l1_loss
