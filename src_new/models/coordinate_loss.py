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
    ):
        """
        Initialize soft expectation coordinate loss.

        Args:
            coord_start_id: First coordinate token ID (<|coord_0|>)
            coord_end_id: Last coordinate token ID + 1 (<|coord_MAX_COORD|> + 1)
            temperature: Softmax temperature for sharpness control
            numerical_stability: Enable numerical stability improvements
            device: Device for tensor operations
        """
        self.coord_start_id = coord_start_id
        self.coord_end_id = coord_end_id
        self.temperature = temperature
        self.numerical_stability = numerical_stability
        self.device = device

        # Coordinate vocabulary size (number of coordinate tokens)
        self.coord_vocab_size = coord_end_id - coord_start_id

        # Pre-compute coordinate value indices for efficiency
        self._coord_values = None

        logger.info(
            f"🎯 Initialized SoftExpectationCoordinateLoss: "
            f"range=[{coord_start_id}, {coord_end_id}), "
            f"vocab_size={self.coord_vocab_size}, "
            f"temperature={temperature}"
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

        # Prepare loss information for logging
        loss_info = {
            "num_coord_tokens": len(coord_positions[0]),
            "valid_coord_tokens": valid_mask.sum().item(),
            "coordinate_l1_loss": float(coordinate_l1_loss.detach().item()),
            "mean_expected_coord": float(expected_coords.mean().item()),
            "mean_target_coord": float(target_coords.float().mean().item()),
        }

        logger.debug(
            f"🎯 Soft expectation coordinate loss: {coordinate_loss.item():.6f} "
            f"(tokens: {loss_info['valid_coord_tokens']}, "
            f"L1: {loss_info['coordinate_l1_loss']:.6f})"
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
    )
    # Validate coordinate token mapping explicitly at construction time
    try:
        loss_fn._validate_coordinate_mapping()
    except Exception:
        # Re-raise to fail-fast in training setup
        raise

    return loss_fn


def build_kernel_indices_and_q(
    y: torch.Tensor, K: int, sigma: float, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-sample sparse kernel window around ground-truth bin indices.

    Args:
        y: Tensor of shape [N] with integer ground-truth in [0, K]
        K: Maximum bin value (inclusive). Total bins = K+1
        sigma: Kernel width in bins (Gaussian). Must be > 0
        window: Half-window size (radius). Output width = 2*window + 1

    Returns:
        idxs: Long tensor [N, W] with clamped indices in [0, K]
        q_vals: Float tensor [N, W] with unnormalized kernel values per index
    """
    if y is None or y.numel() == 0:
        width = 2 * int(window) + 1
        return (
            torch.empty(0, width, dtype=torch.long),
            torch.empty(0, width, dtype=torch.float32),
        )

    if not torch.is_tensor(y):
        raise TypeError("y must be a torch.Tensor")

    y_long = y.to(dtype=torch.long)
    N = y_long.shape[0]
    width = 2 * int(window) + 1

    # Offsets [-window, ..., +window]
    offsets = torch.arange(-window, window + 1, dtype=torch.long, device=y.device)
    # Broadcast to [N, W]
    idxs = y_long.unsqueeze(1) + offsets.unsqueeze(0)
    # Clamp to [0, K]
    idxs = torch.clamp(idxs, min=0, max=int(K))

    # Distances for kernel (in bins)
    d = (idxs - y_long.unsqueeze(1)).to(dtype=torch.float32)
    sigma_val = float(max(sigma, 1e-6))
    q_vals = torch.exp(-(d * d) / (2.0 * (sigma_val**2)))

    return idxs, q_vals


def kernelized_kl_sparse(
    coord_logits: torch.Tensor,  # [N, K+1]
    idxs: torch.Tensor,  # [N, W]
    q_vals: torch.Tensor,  # [N, W]
    tau: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute KL(q||p) where q is a sparse kernel distribution on a window and p
    is the model distribution over full coordinate bins.

    If inputs are empty (N==0), returns 0.0 on the correct device.
    """
    # Handle empty batch
    if (
        coord_logits is None
        or coord_logits.numel() == 0
        or idxs is None
        or idxs.numel() == 0
    ):
        device = (
            coord_logits.device
            if isinstance(coord_logits, torch.Tensor) and coord_logits.numel() > 0
            else None
        )
        return coord_logits.new_tensor(0.0) if device is not None else torch.tensor(0.0)

    # Ensure float32 and clamp for numerical stability
    logits = (coord_logits.float() / float(max(tau, eps))).clamp(-50.0, 50.0)
    p_full = torch.softmax(logits, dim=-1)

    # Gather probabilities on the sparse window
    if idxs.dtype != torch.long:
        idxs = idxs.to(dtype=torch.long)
    p_w = p_full.gather(dim=-1, index=idxs)

    # Normalize q over the window
    q_w = q_vals.to(dtype=torch.float32)
    q_w = q_w / (q_w.sum(dim=-1, keepdim=True) + eps)

    # KL(q||p) over window: sum q * (log q - log p)
    kl_vec = (q_w * (torch.log(q_w + eps) - torch.log(p_w + eps))).sum(dim=-1)

    out = torch.nan_to_num(kl_vec.mean(), nan=0.0, posinf=1e6, neginf=1e6)
    return out


def unlikelihood_topk_text(
    logits_all: torch.Tensor,  # [B, T, V]
    coord_mask: torch.Tensor,  # [B, T]
    noncoord_vocab_mask: torch.BoolTensor,  # [V]
    topk: int = 100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Unlikelihood loss on non-coordinate tokens at coordinate positions.

    Select top-k probabilities within the non-coordinate sub-vocab and penalize
    them via -log(1 - p). Returns 0.0 when coord_mask has no true positions.
    """
    if coord_mask is None:
        return logits_all.new_tensor(0.0)

    # Strict shape checks (fail-fast)
    if coord_mask.dim() != 2:
        raise ValueError(
            f"coord_mask must be 2D [B,T], got shape={tuple(coord_mask.shape)}"
        )
    if logits_all.dim() != 3:
        raise ValueError(
            f"logits_all must be 3D [B,T,V], got shape={tuple(logits_all.shape)}"
        )
    B, T, V = logits_all.shape
    if coord_mask.shape[0] != B or coord_mask.shape[1] != T:
        raise ValueError(
            f"Shape mismatch: logits_all[0:2]={B, T} vs coord_mask={tuple(coord_mask.shape)}"
        )
    if noncoord_vocab_mask is None or noncoord_vocab_mask.numel() != V:
        raise ValueError(
            f"noncoord_vocab_mask must have length V={V}, got {None if noncoord_vocab_mask is None else noncoord_vocab_mask.numel()}"
        )

    # Slice logits to non-coordinate vocab
    logits_text = logits_all[..., noncoord_vocab_mask].float().clamp(-50.0, 50.0)
    probs_text = torch.softmax(logits_text, dim=-1)

    k = int(min(int(topk), probs_text.size(-1)))
    if k <= 0:
        return logits_all.new_tensor(0.0)

    # Top-k over non-coordinate probabilities
    top_vals, _ = torch.topk(probs_text, k=k, dim=-1)
    loss = -torch.log(1.0 - top_vals + eps)  # [B, T, k]

    denom = coord_mask.sum() * k + eps
    out = (loss * coord_mask.unsqueeze(-1).float()).sum() / denom
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)


# Laplacian regularizer removed
