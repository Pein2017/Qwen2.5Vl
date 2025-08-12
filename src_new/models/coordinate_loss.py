"""
Soft Expectation + L1 Loss for Coordinate Token Regression

This module implements advanced coordinate token loss computation using soft expectation
regression instead of standard cross-entropy loss. The approach computes expected
coordinate values from probability distributions over coordinate tokens and applies
L1 loss for better coordinate regression performance.

Mathematical Foundation:
    P(coord_value = v) = softmax(logits_v / temperature)
    expected_coord = Σ(v * P(coord_value = v))  # v ∈ [0, MAX_COORD]
    coordinate_loss = L1(expected_coord, ground_truth_coord)
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Configure rank-aware logger
from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


class GaussianKLRegularizer:
    """
    Numerically-stable Gaussian KL regularizer over coordinate bins.

    Computes KL(p_target || p_pred) where p_target is a Gaussian over bins centered at
a scalar ground-truth coordinate and p_pred comes from model logits via log_softmax.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def __call__(
        self,
        logits_valid: torch.Tensor,  # [N_valid, V]
        target_coords: torch.Tensor,  # [N_valid]
        coord_values: torch.Tensor,  # [V]
        temperature: float,
        sigma_bins: float,
    ) -> torch.Tensor:
        # Ensure float32 path for numerical stability
        logits_f32 = logits_valid.float()

        # Use log_softmax directly for stability
        log_p_pred = F.log_softmax(logits_f32 / float(temperature), dim=-1)

        # Build Gaussian targets over bins (in BIN units)
        # dist shape: [N_valid, V]
        dist = coord_values.unsqueeze(0) - target_coords.float().unsqueeze(1)
        sigma = float(max(sigma_bins, 1.0))  # σ sanity in bins
        denom = 2.0 * (sigma ** 2)
        p_tgt_unnorm = torch.exp(- (dist ** 2) / denom)

        # Normalize with epsilon guard
        p_tgt = p_tgt_unnorm / (p_tgt_unnorm.sum(dim=-1, keepdim=True) + self.eps)
        p_tgt = torch.clamp(p_tgt, min=self.eps)

        # KL expects log-probs for input and probs for target
        kl = F.kl_div(log_p_pred, p_tgt, reduction="batchmean")
        return kl


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

    # Class-level tracking for epoch-based warning logging
    _last_warning_epoch = -1
    _warning_logged_this_epoch = False

    # todo: change the coord_end_id
    def __init__(
        self,
        coord_start_id: int,
        coord_end_id: int,  # will be overridden by factory to exclusive end
        temperature: float,
        numerical_stability: bool = True,
        device: Optional[torch.device] = None,
        kl_weight: float = 0.0,
        label_sigma: Optional[float] = None,
    ):
        """
        Initialize soft expectation coordinate loss.

        Args:
            coord_start_id: First coordinate token ID (<|coord_0|>)
            coord_end_id: Last coordinate token ID + 1 (<|coord_MAX_COORD|> + 1)
            temperature: Softmax temperature for sharpness control
            numerical_stability: Enable numerical stability improvements
            device: Device for tensor operations
            kl_weight: Weight for Gaussian KL regularizer (0 disables KL)
            label_sigma: Gaussian sigma in BIN units; required if kl_weight>0
        """
        self.coord_start_id = coord_start_id
        self.coord_end_id = coord_end_id
        self.temperature = temperature
        self.numerical_stability = numerical_stability
        self.device = device
        self.kl_weight = float(kl_weight)
        self.label_sigma = None if label_sigma is None else float(label_sigma)

        # Coordinate vocabulary size (number of coordinate tokens)
        self.coord_vocab_size = coord_end_id - coord_start_id

        # Pre-compute coordinate value indices for efficiency
        self._coord_values = None

        # KL helper
        self._gaussian_kl = GaussianKLRegularizer(eps=1e-8)

        logger.info(
            f"🎯 Initialized SoftExpectationCoordinateLoss: "
            f"range=[{coord_start_id}, {coord_end_id}), "
            f"vocab_size={self.coord_vocab_size}, "
            f"temperature={temperature}, kl_weight={self.kl_weight}, label_sigma={self.label_sigma}"
        )

    @classmethod
    def update_epoch(cls, epoch: int):
        """
        Update the current epoch for warning logging control.

        Args:
            epoch: Current training epoch
        """
        if epoch != cls._last_warning_epoch:
            cls._last_warning_epoch = epoch
            cls._warning_logged_this_epoch = False

    def _get_coord_values(self, device: torch.device) -> torch.Tensor:
        """
        Get ACTUAL coordinate values corresponding to coordinate token IDs.

        Args:
            device: Target device for tensor

        Returns:
            Coordinate values tensor of shape [coord_vocab_size], where index i
            corresponds to token ID (coord_start_id + i) and value equals the
            actual coordinate value represented by that token.
        """
        if self._coord_values is None or self._coord_values.device != device:
            # Mapping is linear for Qwen2.5-VL: <|coord_k|> has id coord_start_id + k
            # Build via token-id arithmetic to make the dependency explicit.
            token_ids = torch.arange(
                self.coord_start_id,
                self.coord_end_id,
                dtype=torch.float32,
                device=device,
            )
            # Convert token IDs back to coordinate scalar values
            self._coord_values = token_ids - self.coord_start_id
        return self._coord_values

    def _extract_coord_value_from_token_id(self, token_id: int) -> int:
        """
        Extract the actual coordinate value represented by a coordinate token ID.
        For Qwen2.5-VL coordinate tokens: value = token_id - coord_start_id.
        Raises on out-of-range token IDs.
        """
        if token_id < self.coord_start_id or token_id >= self.coord_end_id:
            raise ValueError(
                f"Token ID {token_id} is out of coordinate range [{self.coord_start_id}, {self.coord_end_id})."
            )
        return int(token_id - self.coord_start_id)

    def _validate_coordinate_mapping(self) -> None:
        """Validate that coordinate tokens map to correct scalar values."""
        # Use representative samples; clip to available vocab size
        candidates = [0, 50, 123, 500, 1000]
        max_idx = self.coord_vocab_size - 1
        for expected in candidates:
            test_idx = min(max(expected, 0), max_idx)
            token_id = self.coord_start_id + test_idx
            actual = self._extract_coord_value_from_token_id(token_id)
            if actual != test_idx:
                raise ValueError(
                    "Coordinate mapping error: "
                    f"token {token_id} maps to {actual}, expected {test_idx}"
                )
        logger.info("✅ Coordinate token mapping validated")

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
                    "coordinate_kl": 0.0,
                },
            )

        # Step 1: Extract coordinate token logits from full vocabulary
        # Input logits shape: [batch_size, seq_len, vocab_size]
        # Extract only coordinate token portion: [batch_size, seq_len, coord_vocab_size]
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

        # Extract ground truth coordinate values
        target_coord_ids = labels[coord_positions]

        # Filter out positions where labels contain -100 (ignore tokens)
        valid_mask = target_coord_ids != -100

        if not valid_mask.any():
            # No valid coordinate targets found - return zero loss
            logger.debug("No valid coordinate targets found (all positions are -100)")
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {
                "num_coord_tokens": 0,
                "valid_coord_tokens": 0,
                "avg_expected_coord": 0.0,
                "avg_target_coord": 0.0,
                "coordinate_kl": 0.0,
            }

        # Filter to only valid positions
        valid_coord_logits = coord_logits[valid_mask]
        valid_target_coord_ids = target_coord_ids[valid_mask]

        # Convert target token IDs to coordinate values [0, MAX_COORD] range
        target_coords = valid_target_coord_ids - self.coord_start_id

        # Validate target coordinates are in valid range
        if torch.any(target_coords < 0) or torch.any(
            target_coords >= self.coord_vocab_size
        ):
            # Only log warning once per epoch to reduce noise
            if not self._warning_logged_this_epoch:
                logger.warning(
                    f"⚠️ Target coordinates out of range (epoch {self._last_warning_epoch}): "
                    f"min={target_coords.min().item()}, max={target_coords.max().item()}, "
                    f"expected_range=[0, {self.coord_vocab_size - 1}] "
                    f"(further warnings suppressed for this epoch)"
                )
                self._warning_logged_this_epoch = True
            # Clamp to valid range
            target_coords = torch.clamp(target_coords, 0, self.coord_vocab_size - 1)

        # Compute soft expectation values for valid positions only
        expected_coords = self.compute_soft_expectation(valid_coord_logits, temperature)

        # Compute L1 loss between expected and target coordinates
        coordinate_l1_loss = F.l1_loss(expected_coords, target_coords.float())
        coordinate_loss = coordinate_l1_loss

        # Optional Gaussian KL term (numerically stable)
        coordinate_kl = 0.0
        if self.kl_weight > 0.0:
            if self.label_sigma is None or self.label_sigma <= 0.0:
                raise ValueError(
                    "coordinate_kl_weight > 0 requires a positive coordinate_label_sigma in BIN units"
                )
            # Compute only when we have valid positions
            if valid_coord_logits.numel() > 0:
                # Get coordinate values for current device
                coord_values = self._get_coord_values(valid_coord_logits.device)
                # Compute KL in float32 along coord vocab
                coordinate_kl_tensor = self._gaussian_kl(
                    logits_valid=valid_coord_logits,
                    target_coords=target_coords,
                    coord_values=coord_values,
                    temperature=temperature if temperature is not None else self.temperature,
                    sigma_bins=float(self.label_sigma),
                )
                coordinate_loss = coordinate_loss + self.kl_weight * coordinate_kl_tensor
                coordinate_kl = float(coordinate_kl_tensor.detach().item())

        # Prepare loss information for logging
        loss_info = {
            "num_coord_tokens": len(coord_positions[0]),
            "valid_coord_tokens": valid_mask.sum().item(),
            "coordinate_l1_loss": float(coordinate_l1_loss.detach().item()),
            "mean_expected_coord": float(expected_coords.mean().item()),
            "mean_target_coord": float(target_coords.float().mean().item()),
            "coordinate_kl": coordinate_kl,
        }

        logger.debug(
            f"🎯 Soft expectation coordinate loss: {coordinate_loss.item():.6f} "
            f"(tokens: {loss_info['valid_coord_tokens']}, "
            f"L1: {loss_info['coordinate_l1_loss']:.6f}, KL: {loss_info['coordinate_kl']:.6f})"
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

    # OPTIMIZATION: Support lazy loading - allow coordinate tokens to be added later
    if coord_start_id == 0 and coord_end_id == 0:
        # Check if coordinate tokens are enabled in config
        if (
            hasattr(token_processor, "config")
            and token_processor.config.coordinate_tokens_enabled
        ):
            logger.warning(
                "Coordinate tokens not found in tokenizer yet. "
                "This is expected with lazy loading - tokens will be validated during training."
            )
            # Use placeholder range that will be updated when tokens are actually added
            coord_start_id = len(tokenizer.get_vocab())  # Start after current vocab
            coord_end_id = coord_start_id + token_processor.config.max_coord_value + 1
        else:
            raise ValueError(
                "No coordinate tokens found in tokenizer. Ensure coordinate tokens are properly added to the tokenizer vocabulary."
            )

    # Adjust end_id to be exclusive (ensure +1 semantics)
    coord_end_id = coord_end_id + 1

    # Strict validation: enforce Qwen2.5-VL range based on tokenizer and config
    # We infer expected end from start + max_coord_value, but keep the start fixed.
    HARD_START = 151667
    # Try to access max_coord_value if available
    expected_end_inclusive = None
    try:
        if hasattr(token_processor, "config") and hasattr(
            token_processor.config, "max_coord_value"
        ):
            expected_end_inclusive = HARD_START + int(
                token_processor.config.max_coord_value
            )
    except Exception:
        expected_end_inclusive = None

    if expected_end_inclusive is not None:
        expected_exclusive_end = expected_end_inclusive + 1
        if not (
            coord_start_id == HARD_START and coord_end_id == expected_exclusive_end
        ):
            raise ValueError(
                "❌ Coordinate token ID range mismatch (coordinate_loss factory). "
                f"Expected [{{HARD_START}}, {{expected_end_inclusive}}] inclusive, but got "
                f"[{{coord_start_id}}, {{coord_end_id - 1}}]. Ensure you are using a Qwen2.5-VL tokenizer and max_coord_value matches."
            )
    else:
        # Fallback strict check on start only
        if coord_start_id != HARD_START:
            raise ValueError(
                "❌ Coordinate token ID range mismatch (coordinate_loss factory). "
                f"Expected start {{HARD_START}}, but got start {{coord_start_id}}."
            )

    logger.info(
        f"Using coordinate token range: {coord_start_id} to {coord_end_id - 1} (validated)"
    )

    loss_fn = SoftExpectationCoordinateLoss(
        coord_start_id=coord_start_id,
        coord_end_id=coord_end_id,
        temperature=temperature,
        **kwargs,
    )
    # Validate coordinate token mapping explicitly at construction time
    try:
        loss_fn._validate_coordinate_mapping()
    except Exception:
        # Re-raise to fail-fast in training setup
        raise

    return loss_fn
