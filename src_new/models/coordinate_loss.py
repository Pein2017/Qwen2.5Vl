"""
Soft Expectation + L1 Loss for Coordinate Token Regression

This module implements advanced coordinate token loss computation using soft expectation
regression instead of standard cross-entropy loss. The approach computes expected
coordinate values from probability distributions over coordinate tokens and applies
L1 loss for better coordinate regression performance.

Mathematical Foundation:
    P(coord_value = v) = softmax(logits_v / temperature)
    expected_coord = Σ(v * P(coord_value = v))  # v ∈ [0, 2048]
    coordinate_loss = L1(expected_coord, ground_truth_coord)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# Configure logger
logger = logging.getLogger(__name__)


class SoftExpectationCoordinateLoss:
    """
    Implements soft expectation + L1 loss for coordinate token regression.

    This class provides a more sophisticated approach to coordinate token loss
    computation compared to standard cross-entropy loss. It computes probability
    distributions over coordinate values and uses soft expectation to derive
    continuous coordinate predictions.

    Key Benefits:
    - Smooth gradients for coordinate regression
    - Uncertainty modeling through probability distributions
    - Temperature-controlled prediction sharpness
    - Better convergence for coordinate prediction tasks
    """

    def __init__(
        self,
        coord_start_id: int = 151667,
        coord_end_id: int = 153716,
        temperature: float = 1.0,
        numerical_stability: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize soft expectation coordinate loss.

        Args:
            coord_start_id: First coordinate token ID (<|coord_0|>)
            coord_end_id: Last coordinate token ID + 1 (<|coord_2048|> + 1)
            temperature: Softmax temperature for sharpness control
            numerical_stability: Enable numerical stability improvements
            device: Device for tensor operations
        """
        self.coord_start_id = coord_start_id
        self.coord_end_id = coord_end_id
        self.temperature = temperature
        self.numerical_stability = numerical_stability
        self.device = device

        # Coordinate vocabulary size (2049 tokens: coord_0 to coord_2048)
        self.coord_vocab_size = coord_end_id - coord_start_id

        # Pre-compute coordinate value indices for efficiency
        self._coord_values = None

        logger.info(
            f"🎯 Initialized SoftExpectationCoordinateLoss: "
            f"range=[{coord_start_id}, {coord_end_id}), "
            f"vocab_size={self.coord_vocab_size}, "
            f"temperature={temperature}"
        )

    def _get_coord_values(self, device: torch.device) -> torch.Tensor:
        """
        Get coordinate value indices tensor [0, 1, 2, ..., 2048].

        Args:
            device: Target device for tensor

        Returns:
            Coordinate values tensor of shape [coord_vocab_size]
        """
        if self._coord_values is None or self._coord_values.device != device:
            self._coord_values = torch.arange(
                self.coord_vocab_size, device=device, dtype=torch.float32
            )
        return self._coord_values

    def compute_soft_expectation(
        self, coord_logits: torch.Tensor, temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute soft expectation values from coordinate token logits.

        Args:
            coord_logits: Coordinate token logits [num_tokens, coord_vocab_size]
            temperature: Optional temperature override

        Returns:
            Expected coordinate values [num_tokens]
        """
        if temperature is None:
            temperature = self.temperature

        # Apply temperature scaling
        scaled_logits = coord_logits / temperature

        # Numerical stability improvements
        if self.numerical_stability:
            # Clip extreme logits to prevent overflow/underflow
            scaled_logits = torch.clamp(scaled_logits, min=-50.0, max=50.0)

        # Compute probability distribution over coordinate values
        coord_probs = F.softmax(scaled_logits, dim=-1)

        # Additional numerical stability
        if self.numerical_stability:
            # Add small epsilon and renormalize
            coord_probs = coord_probs + 1e-8
            coord_probs = coord_probs / coord_probs.sum(dim=-1, keepdim=True)

        # Get coordinate value indices
        coord_values = self._get_coord_values(coord_logits.device)

        # Compute expected coordinate values via soft expectation
        # expected_coord = Σ(v * P(coord_value = v))
        expected_coords = torch.sum(coord_probs * coord_values, dim=-1)

        return expected_coords

    def compute_coordinate_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute soft expectation + L1 loss for coordinate tokens.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]
            temperature: Optional temperature override

        Returns:
            Tuple of (coordinate_loss, loss_info_dict)
        """
        # Check if we have any coordinate tokens
        num_coord_tokens = coord_mask.sum().item()

        if num_coord_tokens == 0:
            logger.debug("🔍 No coordinate tokens found in batch")
            return (
                torch.tensor(0.0, device=logits.device, requires_grad=True),
                {
                    "num_coord_tokens": 0,
                    "coordinate_l1_loss": 0.0,
                    "mean_expected_coord": 0.0,
                    "mean_target_coord": 0.0,
                },
            )

        # Step 1: Extract coordinate token logits from full vocabulary
        # Input logits shape: [batch_size, seq_len, vocab_size=153716]
        # Extract only coordinate token portion: [batch_size, seq_len, 2049]
        coord_logits_full = logits[:, :, self.coord_start_id : self.coord_end_id]

        # Step 2: Extract coordinate token positions
        coord_positions = torch.where(coord_mask)

        # Step 3: Apply coordinate mask to get logits only at coordinate positions
        # Shape: [num_coord_tokens, coord_vocab_size=2049]
        coord_logits = coord_logits_full[coord_positions]

        # Debug logging for coordinate logits extraction
        logger.debug(
            f"🔍 Coordinate logits extraction: "
            f"full_logits={logits.shape}, "
            f"coord_range=[{self.coord_start_id}:{self.coord_end_id}], "
            f"coord_logits_full={coord_logits_full.shape}, "
            f"coord_positions={len(coord_positions[0])}, "
            f"coord_logits={coord_logits.shape}"
        )

        # Validate coordinate logits shape
        if coord_logits.size(-1) != self.coord_vocab_size:
            raise ValueError(
                f"Coordinate logits size {coord_logits.size(-1)} != "
                f"expected coord_vocab_size {self.coord_vocab_size}"
            )

        # Compute soft expectation values
        expected_coords = self.compute_soft_expectation(coord_logits, temperature)

        # Extract ground truth coordinate values
        target_coord_ids = labels[coord_positions]
        target_coords = (
            target_coord_ids - self.coord_start_id
        )  # Convert to [0, 2048] range

        # Validate target coordinates are in valid range
        if torch.any(target_coords < 0) or torch.any(
            target_coords >= self.coord_vocab_size
        ):
            logger.warning(
                f"⚠️ Target coordinates out of range: "
                f"min={target_coords.min().item()}, max={target_coords.max().item()}, "
                f"expected_range=[0, {self.coord_vocab_size - 1}]"
            )
            # Clamp to valid range
            target_coords = torch.clamp(target_coords, 0, self.coord_vocab_size - 1)

        # Compute L1 loss between expected and target coordinates
        coordinate_loss = F.l1_loss(expected_coords, target_coords.float())

        # Prepare loss information for logging
        loss_info = {
            "num_coord_tokens": num_coord_tokens,
            "coordinate_l1_loss": coordinate_loss.item(),
            "mean_expected_coord": expected_coords.mean().item(),
            "mean_target_coord": target_coords.float().mean().item(),
        }

        logger.debug(
            f"🎯 Soft expectation coordinate loss: {coordinate_loss.item():.6f} "
            f"(tokens: {num_coord_tokens}, "
            f"expected: {loss_info['mean_expected_coord']:.2f}, "
            f"target: {loss_info['mean_target_coord']:.2f})"
        )

        return coordinate_loss, loss_info

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Callable interface for coordinate loss computation.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]
            temperature: Optional temperature override

        Returns:
            Tuple of (coordinate_loss, loss_info_dict)
        """
        return self.compute_coordinate_loss(logits, labels, coord_mask, temperature)


def create_coordinate_loss_from_token_processor(
    token_processor, tokenizer, temperature: float = 1.0, **kwargs
) -> SoftExpectationCoordinateLoss:
    """
    Factory function to create coordinate loss function using token processor.

    This ensures the coordinate token IDs match the actual tokenizer vocabulary.

    Args:
        token_processor: TokenProcessor instance with coordinate token mapping
        tokenizer: Extended tokenizer with coordinate tokens
        temperature: Softmax temperature
        **kwargs: Additional arguments for SoftExpectationCoordinateLoss

    Returns:
        Configured coordinate loss function with correct token IDs
    """
    # Get actual coordinate token range from token processor
    coord_start_id, coord_end_id = token_processor.get_coordinate_token_range(tokenizer)

    if coord_start_id == 0 and coord_end_id == 0:
        raise ValueError(
            "No coordinate tokens found in tokenizer. Ensure coordinate tokens are properly added to the tokenizer vocabulary."
        )

    # Adjust end_id to be exclusive (add 1)
    coord_end_id = coord_end_id + 1
    logger.info(f"Using coordinate token range: {coord_start_id} to {coord_end_id - 1}")

    return SoftExpectationCoordinateLoss(
        coord_start_id=coord_start_id,
        coord_end_id=coord_end_id,
        temperature=temperature,
        **kwargs,
    )
