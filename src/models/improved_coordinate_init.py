"""
Improved Coordinate Token Initialization

This module provides enhanced initialization strategies for coordinate tokens
to improve training convergence based on analysis of the current implementation.

Key improvements:
1. Initialize coordinate tokens from integer token embeddings
2. Add positional encoding to distinguish coordinate values
3. Reduce coordinate loss weighting for better balance
4. Implement progressive training strategies
"""

from typing import Dict, Tuple

import numpy as np
import torch

from src.logger_utils import get_training_logger


logger = get_training_logger()


class ImprovedCoordinateInitializer:
    """
    Enhanced coordinate token initialization strategy.

    Addresses convergence issues by:
    - Copying embeddings from integer tokens (0-9)
    - Generating numerical patterns for higher values
    - Adding positional encoding for coordinate distinction
    - Maintaining proper embedding magnitudes
    """

    def __init__(self, tokenizer, model, max_coord_value: int = 2048):
        self.tokenizer = tokenizer
        self.model = model
        self.max_coord_value = max_coord_value

        # Find available integer tokens
        self.integer_tokens = self._find_integer_tokens()
        logger.info(f"🔢 Found integer tokens: {list(self.integer_tokens.keys())}")

    def _find_integer_tokens(self) -> Dict[int, Tuple[str, int]]:
        """Find integer tokens in vocabulary."""
        vocab = self.tokenizer.get_vocab()
        integer_tokens = {}

        for token, token_id in vocab.items():
            try:
                if (
                    token.strip().isdigit()
                    and len(token.strip()) > 0
                    and len(token) <= 4
                ):
                    integer_tokens[int(token)] = (token, token_id)
            except (ValueError, TypeError):
                continue

        return integer_tokens

    def apply_improved_initialization(self, coord_start_id: int, coord_end_id: int):
        """
        Apply improved initialization to coordinate token embeddings.

        Args:
            coord_start_id: Starting token ID for coordinate tokens
            coord_end_id: Ending token ID for coordinate tokens (exclusive)
        """
        logger.info(
            f"🔧 Applying improved coordinate initialization [{coord_start_id}:{coord_end_id}]"
        )

        embeddings = self.model.get_input_embeddings()

        with torch.no_grad():
            weights = embeddings.weight
            embedding_dim = weights.shape[1]

            # Get reference embeddings from integer tokens
            integer_embeddings = self._get_integer_embeddings(weights)

            # Initialize each coordinate token
            for coord_value in range(self.max_coord_value):
                coord_token_id = coord_start_id + coord_value

                if coord_token_id >= coord_end_id:
                    break

                # Generate improved embedding
                new_embedding = self._generate_coordinate_embedding(
                    coord_value, integer_embeddings, embedding_dim
                )

                # Set the embedding
                weights[coord_token_id] = new_embedding

            logger.info("✅ Improved coordinate initialization completed")

            # Validate the initialization
            self._validate_initialization(coord_start_id, coord_end_id, weights)

    def _get_integer_embeddings(self, weights: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract embeddings for integer tokens."""
        integer_embeddings = {}

        for i in range(10):  # 0-9
            if i in self.integer_tokens:
                token_id = self.integer_tokens[i][1]
                integer_embeddings[i] = weights[token_id].clone()

        logger.info(
            f"   📊 Using {len(integer_embeddings)} integer embeddings as reference"
        )
        return integer_embeddings

    def _generate_coordinate_embedding(
        self,
        coord_value: int,
        integer_embeddings: Dict[int, torch.Tensor],
        embedding_dim: int,
    ) -> torch.Tensor:
        """
        Generate embedding for a coordinate value.

        Strategy:
        1. For 0-9: Copy from integer tokens if available
        2. For 10+: Interpolate/extrapolate from integer patterns
        3. Add positional encoding for distinction
        4. Apply small perturbation to avoid identical embeddings
        """

        # Step 1: Get base embedding
        if coord_value < 10 and coord_value in integer_embeddings:
            # Direct copy for single digits
            base_embedding = integer_embeddings[coord_value].clone()
        elif len(integer_embeddings) >= 2:
            # Generate from numerical pattern
            base_embedding = self._interpolate_from_integers(
                coord_value, integer_embeddings
            )
        elif len(integer_embeddings) == 1:
            # Use the single available integer embedding
            base_embedding = list(integer_embeddings.values())[0].clone()
        else:
            # Fallback: small random initialization
            base_embedding = torch.randn(embedding_dim) * 0.02

        # Step 2: Add positional encoding
        positional_encoding = self._create_positional_encoding(
            coord_value, embedding_dim
        )

        # Step 3: Combine base + positional + perturbation
        final_embedding = base_embedding + positional_encoding

        # Step 4: Add small random perturbation
        perturbation = torch.randn_like(final_embedding) * 0.001
        final_embedding = final_embedding + perturbation

        # Step 5: Normalize to maintain proper magnitude
        target_norm = base_embedding.norm().item()
        if target_norm > 0:
            final_embedding = final_embedding / final_embedding.norm() * target_norm

        return final_embedding

    def _interpolate_from_integers(
        self, coord_value: int, integer_embeddings: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Interpolate/extrapolate embedding from integer patterns."""
        available_values = sorted(integer_embeddings.keys())

        if coord_value <= max(available_values):
            # Interpolation
            if coord_value in integer_embeddings:
                return integer_embeddings[coord_value].clone()

            # Find surrounding values
            lower_vals = [v for v in available_values if v < coord_value]
            upper_vals = [v for v in available_values if v > coord_value]

            if lower_vals and upper_vals:
                lower_val = max(lower_vals)
                upper_val = min(upper_vals)

                # Linear interpolation
                alpha = (coord_value - lower_val) / (upper_val - lower_val)
                return (1 - alpha) * integer_embeddings[
                    lower_val
                ] + alpha * integer_embeddings[upper_val]
            elif lower_vals:
                return integer_embeddings[max(lower_vals)].clone()
            else:
                return integer_embeddings[min(upper_vals)].clone()

        else:
            # Extrapolation
            if len(available_values) >= 2:
                # Use trend from last two values
                val1, val2 = available_values[-2], available_values[-1]
                direction = integer_embeddings[val2] - integer_embeddings[val1]
                steps = coord_value - val2
                return integer_embeddings[val2] + direction * steps
            else:
                # Single value - just use it
                return integer_embeddings[available_values[0]].clone()

    def _create_positional_encoding(
        self, position: int, embedding_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding for coordinate position."""
        pe = torch.zeros(embedding_dim)

        # Scale position to reasonable range
        scaled_pos = position / self.max_coord_value * 100

        for i in range(0, embedding_dim, 2):
            div_term = torch.exp(torch.tensor(i) * -(np.log(10000.0) / embedding_dim))
            pe[i] = torch.sin(scaled_pos * div_term)
            if i + 1 < embedding_dim:
                pe[i + 1] = torch.cos(scaled_pos * div_term)

        # Scale down to be additive, not dominant
        return pe * 0.05  # Reduced from 0.1 to be more subtle

    def _validate_initialization(
        self, coord_start_id: int, coord_end_id: int, weights: torch.Tensor
    ):
        """Validate coordinate embedding initialization."""
        coord_embeddings = weights[coord_start_id:coord_end_id]

        # Check for zero embeddings
        zero_count = (torch.norm(coord_embeddings, dim=1) < 1e-6).sum().item()
        if zero_count > 0:
            logger.warning(f"⚠️ Found {zero_count} zero coordinate embeddings")

        # Check embedding statistics
        norms = torch.norm(coord_embeddings, dim=1)
        logger.info(
            f"   📊 Coordinate embedding norms: mean={norms.mean():.6f}, std={norms.std():.6f}"
        )

        # Compare with integer embeddings if available
        if self.integer_tokens and 0 in self.integer_tokens:
            int_0_id = self.integer_tokens[0][1]
            coord_0_id = coord_start_id

            cos_sim = torch.cosine_similarity(
                weights[int_0_id], weights[coord_0_id], dim=0
            ).item()
            l2_dist = torch.norm(weights[int_0_id] - weights[coord_0_id]).item()

            logger.info(
                f"   🔍 coord_0 vs int_0: cosine_sim={cos_sim:.3f}, L2_dist={l2_dist:.3f}"
            )

            if cos_sim > 0.3:  # Lowered threshold since we add positional encoding
                logger.info(
                    "✅ Good initialization: coordinate tokens aligned with integer concepts"
                )
            else:
                logger.warning("⚠️ Coordinate tokens may need further alignment")


def apply_improved_coordinate_initialization(model_adapter):
    """
    Apply improved coordinate initialization to a model adapter.

    This function can be called after coordinate tokens are added but before training starts.

    Args:
        model_adapter: ModelAdapter instance with coordinate tokens already added
    """
    if (
        not hasattr(model_adapter, "coordinate_config")
        or not model_adapter.coordinate_config.enable_coordinate_tokens
    ):
        logger.info(
            "🔍 Coordinate tokens not enabled, skipping improved initialization"
        )
        return

    if (
        not hasattr(model_adapter, "extended_embeddings")
        or model_adapter.extended_embeddings is None
    ):
        logger.warning(
            "⚠️ Extended embeddings not found, cannot apply improved initialization"
        )
        return

    # Get coordinate token range
    coord_start_id = model_adapter.original_vocab_size
    coord_end_id = model_adapter.extended_vocab_size

    if coord_start_id >= coord_end_id:
        logger.info("🔍 No coordinate tokens to initialize")
        return

    # Create initializer and apply improvements
    # Note: We need to get tokenizer from somewhere - this would need to be passed in
    # For now, we'll skip the tokenizer-dependent parts
    logger.info("🔧 Applying improved coordinate initialization (simplified version)")

    with torch.no_grad():
        weights = model_adapter.extended_embeddings.weight
        embedding_dim = weights.shape[1]

        # Simple improved initialization: reduce variance and add small trends
        coord_embeddings = weights[coord_start_id:coord_end_id]

        # Reduce the variance (current embeddings may be too random)
        coord_embeddings *= 0.5

        # Add a small linear trend to distinguish coordinate positions
        num_coords = coord_end_id - coord_start_id
        for i in range(num_coords):
            # Add a small position-dependent bias
            position_bias = torch.randn(embedding_dim) * (i / num_coords) * 0.01
            coord_embeddings[i] += position_bias

        logger.info("✅ Applied simplified improved initialization")

        # Validate
        norms = torch.norm(coord_embeddings, dim=1)
        logger.info(
            f"   📊 New coordinate embedding norms: mean={norms.mean():.6f}, std={norms.std():.6f}"
        )


def get_improved_coordinate_loss_weight() -> float:
    """
    Get improved coordinate loss weight.

    Based on analysis, the current weight of 0.05 may be too high given that
    coordinate losses are 5-10x higher than LM losses.

    Returns:
        Reduced coordinate loss weight for better balance
    """
    return 0.01  # Reduced from 0.05 to 0.01


def get_adaptive_coordinate_loss_weight(
    training_step: int, max_steps: int = 10000
) -> float:
    """
    Get adaptive coordinate loss weight that changes during training.

    Args:
        training_step: Current training step
        max_steps: Maximum steps for weight schedule

    Returns:
        Adaptive coordinate loss weight
    """
    # Start with lower weight, gradually increase
    min_weight = 0.005
    max_weight = 0.02

    progress = min(1.0, training_step / max_steps)
    return min_weight + (max_weight - min_weight) * progress
