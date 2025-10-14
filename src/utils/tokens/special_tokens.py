"""
Unified Token Management for Qwen2.5-VL with Coordinate Support

Simple, automatic token management that:
1. Reuses existing tokens where possible
2. Automatically handles new token addition
3. Manages coordinate tokens [0, max_coord_value]
4. No manual token ID configuration needed
"""

from typing import Any, Dict, List, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from src.logger_utils import get_tokens_logger


logger = get_tokens_logger()

IGNORE_INDEX = -100


class UnifiedTokenManager:
    """
    Simplified, unified token management for Qwen2.5-VL.

    Automatically handles:
    - Existing token reuse (object_ref, box tokens)
    - New token addition (line, square tokens)
    - Coordinate token range [0, max_coord_value]
    - Token ID assignment without manual configuration
    """

    # Existing tokens we can reuse (automatically detected)
    EXISTING_TOKENS = {
        "object_ref_start": "<|object_ref_start|>",  # Usually ID: 151646
        "object_ref_end": "<|object_ref_end|>",  # Usually ID: 151647
        "box_start": "<|box_start|>",  # Usually ID: 151648
        "box_end": "<|box_end|>",  # Usually ID: 151649
    }

    # New tokens we need to add
    NEW_GEOMETRY_TOKENS = [
        "<|line_start|>",
        "<|line_end|>",
    ]

    # Vision tokens (existing)
    VISION_TOKENS = {
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
        "image_pad": "<|image_pad|>",
    }

    # Configuration class (for wrapper.py compatibility)
    class Config:
        """Configuration class compatible with wrapper.py"""

        def __init__(self, max_coord_value=2048):
            self.max_coord_value = max_coord_value
            self.coord_token_init_std = 0.02
            self.coordinate_loss_weight = 0.05
            self.regular_loss_weight = 1.0
            self.soft_expectation_temperature = 1.0
            self.enable_coordinate_tokens = True
            self.use_official_box_tokens = True
            self.enable_multi_geometry = True
            self.max_line_coordinates = 50

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        max_coord_value: int = 1024,
    ):
        """
        Initialize unified token manager.

        Args:
            tokenizer: The tokenizer to extend
            model: The model to resize embeddings for
            max_coord_value: Maximum coordinate value (creates tokens [0, max_coord_value-1])
        """
        self.tokenizer = tokenizer
        self.model = model
        self.max_coord_value = max_coord_value
        self.original_vocab_size = len(tokenizer.get_vocab())

        # Token ID mappings (automatically populated)
        self.token_ids: Dict[str, int] = {}

        # Coordinate token range tracking
        self.coord_start_id = None
        self.coord_end_id = None

        # Create config attribute for wrapper.py compatibility
        self.config = self.Config(max_coord_value=max_coord_value)

        # Initialize token system
        self._setup_tokens()

    def _setup_tokens(self):
        """Set up all tokens automatically."""
        logger.info("🔧 Setting up unified token management...")

        # 1. Detect existing tokens
        self._detect_existing_tokens()

        # 2. Add new geometry tokens
        new_tokens_added = self._add_new_geometry_tokens()

        # 3. Add coordinate tokens
        coord_tokens_added = self._add_coordinate_tokens()

        # 4. Resize model embeddings if needed
        total_new_tokens = new_tokens_added + coord_tokens_added
        if total_new_tokens > 0:
            self._resize_model_embeddings(total_new_tokens)

        logger.info(f"✅ Token setup complete. Added {total_new_tokens} new tokens.")

    def _detect_existing_tokens(self):
        """Detect and map existing tokens."""
        logger.info("🔍 Detecting existing tokens...")
        vocab = self.tokenizer.get_vocab()

        for name, token in self.EXISTING_TOKENS.items():
            if token in vocab:
                token_id = vocab[token]
                self.token_ids[name] = token_id
                logger.info(f"   ✅ Found {name}: {token} -> ID {token_id}")
            else:
                logger.warning(f"   ⚠️ Missing expected token: {token}")

        # Also detect vision tokens
        for name, token in self.VISION_TOKENS.items():
            if token in vocab:
                self.token_ids[name] = vocab[token]

    def _add_new_geometry_tokens(self) -> int:
        """Add new geometry tokens and return count added."""
        logger.info("➕ Adding new geometry tokens...")

        existing_tokens = self.tokenizer.get_vocab()
        tokens_to_add = []

        for token in self.NEW_GEOMETRY_TOKENS:
            if token not in existing_tokens:
                tokens_to_add.append(token)

        if tokens_to_add:
            # Add tokens to tokenizer
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens_to_add}
            )
            logger.info(f"   ✅ Added {num_added} geometry tokens: {tokens_to_add}")

            # Update our token ID mapping
            updated_vocab = self.tokenizer.get_vocab()
            for token in tokens_to_add:
                if token in updated_vocab:
                    # Map to our naming convention
                    if "line_start" in token:
                        self.token_ids["line_start"] = updated_vocab[token]
                    elif "line_end" in token:
                        self.token_ids["line_end"] = updated_vocab[token]
                    elif "square_start" in token:
                        self.token_ids["square_start"] = updated_vocab[token]
                    elif "square_end" in token:
                        self.token_ids["square_end"] = updated_vocab[token]

            return num_added
        else:
            logger.info("   ✅ All geometry tokens already exist")
            # Still need to map existing geometry tokens
            updated_vocab = self.tokenizer.get_vocab()
            for token in self.NEW_GEOMETRY_TOKENS:
                if token in updated_vocab:
                    if "line_start" in token:
                        self.token_ids["line_start"] = updated_vocab[token]
                    elif "line_end" in token:
                        self.token_ids["line_end"] = updated_vocab[token]
                    elif "square_start" in token:
                        self.token_ids["square_start"] = updated_vocab[token]
                    elif "square_end" in token:
                        self.token_ids["square_end"] = updated_vocab[token]
            return 0

    def _add_coordinate_tokens(self) -> int:
        """Add coordinate tokens [0, max_coord_value-1] and return count added."""
        if self.max_coord_value <= 0:
            logger.info("   ⚠️ Coordinate tokens disabled (max_coord_value <= 0)")
            return 0

        logger.info(f"➕ Adding coordinate tokens [0, {self.max_coord_value - 1}]...")

        # Generate coordinate token names (correct format: <|coord_X|>)
        coord_tokens = [f"<|coord_{i}|>" for i in range(self.max_coord_value)]

        # Check which ones need to be added
        existing_vocab = self.tokenizer.get_vocab()
        tokens_to_add = [token for token in coord_tokens if token not in existing_vocab]

        if tokens_to_add:
            # Add in batches to avoid memory issues
            batch_size = 1000
            total_added = 0

            for i in range(0, len(tokens_to_add), batch_size):
                batch = tokens_to_add[i : i + batch_size]
                num_added = self.tokenizer.add_tokens(batch)
                total_added += num_added
                logger.info(
                    f"   ✅ Added coordinate token batch {i // batch_size + 1}: {num_added} tokens"
                )

            logger.info(f"   ✅ Total coordinate tokens added: {total_added}")

            # Update coordinate token range tracking
            updated_vocab = self.tokenizer.get_vocab()
            coord_0_token = "<|coord_0|>"
            if coord_0_token in updated_vocab:
                self.coord_start_id = updated_vocab[coord_0_token]
                self.coord_end_id = self.coord_start_id + self.max_coord_value
                logger.info(
                    f"   🎯 Coordinate token range: [{self.coord_start_id}, {self.coord_end_id})"
                )

            return total_added
        else:
            logger.info("   ✅ All coordinate tokens already exist")
            # Still need to set coordinate token range
            updated_vocab = self.tokenizer.get_vocab()
            coord_0_token = "<|coord_0|>"
            if coord_0_token in updated_vocab:
                self.coord_start_id = updated_vocab[coord_0_token]
                self.coord_end_id = self.coord_start_id + self.max_coord_value
                logger.info(
                    f"   🎯 Coordinate token range: [{self.coord_start_id}, {self.coord_end_id})"
                )
            return 0

    def _resize_model_embeddings(self, num_new_tokens: int):
        """Resize model embeddings to accommodate new tokens."""
        logger.info(f"🔧 Resizing model embeddings for {num_new_tokens} new tokens...")

        old_size = self.model.get_input_embeddings().num_embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        new_size = self.model.get_input_embeddings().num_embeddings

        logger.info(f"   ✅ Embeddings resized: {old_size} -> {new_size}")

    def get_token_id(self, token_name: str) -> int:
        """Get token ID by name."""
        if token_name in self.token_ids:
            return self.token_ids[token_name]

        # Try direct lookup in tokenizer
        vocab = self.tokenizer.get_vocab()
        if token_name in vocab:
            return vocab[token_name]

        raise ValueError(f"Token '{token_name}' not found")

    def get_coordinate_token_id(self, coord_value: int) -> int:
        """Get token ID for a coordinate value."""
        if not (0 <= coord_value < self.max_coord_value):
            raise ValueError(
                f"Coordinate value {coord_value} out of range [0, {self.max_coord_value})"
            )

        token_name = f"<|coord_{coord_value}|>"
        vocab = self.tokenizer.get_vocab()

        if token_name not in vocab:
            raise ValueError(f"Coordinate token {token_name} not found in vocabulary")

        return vocab[token_name]

    def has_coordinate_tokens(self) -> bool:
        """Check if coordinate tokens are available."""
        return self.coord_start_id is not None and self.coord_end_id is not None

    def get_coordinate_token_range(self) -> Tuple[int, int]:
        """Get coordinate token range (start_id, end_id)."""
        if not self.has_coordinate_tokens():
            raise ValueError("Coordinate tokens not available")
        return self.coord_start_id, self.coord_end_id

    def validate_coordinate_token_consistency(self) -> None:
        """
        Validate coordinate token vocabulary-embedding consistency during model loading.

        This method performs comprehensive validation to prevent coordinate token
        vocabulary-embedding mismatches that can cause training instability.

        Raises:
            ValueError: If coordinate tokens are enabled but validation fails
            RuntimeError: If model embeddings are inconsistent with tokenizer vocabulary
        """
        logger.info("🔍 Validating coordinate token consistency...")

        # If coordinate tokens are disabled, skip validation
        if not self.config.enable_coordinate_tokens or self.max_coord_value <= 0:
            logger.info("✅ Coordinate tokens disabled - skipping validation")
            return

        # Check if coordinate tokens should be available
        if not self.has_coordinate_tokens():
            raise ValueError(
                "Coordinate tokens are enabled in configuration but not available in token manager. "
                "This indicates a setup failure during token initialization."
            )

        # Get coordinate token range
        coord_start_id, coord_end_id = self.get_coordinate_token_range()
        expected_coord_count = coord_end_id - coord_start_id

        # Validate coordinate token range
        if expected_coord_count != self.max_coord_value:
            raise ValueError(
                f"Coordinate token range mismatch: expected {self.max_coord_value} tokens, "
                f"but found {expected_coord_count} tokens in range [{coord_start_id}, {coord_end_id})"
            )

        # Validate tokenizer vocabulary contains coordinate tokens
        vocab = self.tokenizer.get_vocab()
        missing_tokens = []

        for i in range(self.max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in vocab:
                missing_tokens.append(coord_token)

        if missing_tokens:
            raise ValueError(
                f"Coordinate tokens missing from tokenizer vocabulary: "
                f"{len(missing_tokens)} tokens missing (first 5: {missing_tokens[:5]}). "
                f"This indicates incomplete token setup during model loading."
            )

        # Validate coordinate token IDs are sequential
        coord_token_ids = []
        for i in range(self.max_coord_value):
            coord_token = f"<|coord_{i}|>"
            token_id = vocab[coord_token]
            coord_token_ids.append(token_id)

        # Check if IDs are sequential starting from coord_start_id
        expected_ids = list(range(coord_start_id, coord_end_id))
        if coord_token_ids != expected_ids:
            raise ValueError(
                f"Coordinate token IDs are not sequential. "
                f"Expected range: {expected_ids[:5]}...{expected_ids[-5:]} "
                f"Found: {coord_token_ids[:5]}...{coord_token_ids[-5:]}. "
                f"This indicates token ID assignment corruption."
            )

        # Validate model embeddings cover coordinate token range
        model_vocab_size = self.model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = len(vocab)

        if model_vocab_size != tokenizer_vocab_size:
            logger.warning(
                f"⚠️ Model-tokenizer vocabulary size mismatch: "
                f"model embeddings={model_vocab_size}, tokenizer vocab={tokenizer_vocab_size}. "
                f"Difference: {tokenizer_vocab_size - model_vocab_size} tokens. "
                f"Attempting to fix by resizing model embeddings..."
            )
            # Try to fix the mismatch by resizing
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            new_model_vocab_size = self.model.get_input_embeddings().num_embeddings
            logger.info(
                f"✅ Model embeddings resized from {model_vocab_size} to {new_model_vocab_size}"
            )

            if new_model_vocab_size != tokenizer_vocab_size:
                raise RuntimeError(
                    f"Failed to fix vocabulary mismatch: "
                    f"model embeddings={new_model_vocab_size}, tokenizer vocab={tokenizer_vocab_size}. "
                    f"This indicates model embeddings were not properly resized during coordinate token setup."
                )
            # Update model_vocab_size after successful resize
            model_vocab_size = new_model_vocab_size

        # Validate coordinate tokens are within model embedding bounds
        if coord_end_id > model_vocab_size:
            raise RuntimeError(
                f"Coordinate token range [{coord_start_id}, {coord_end_id}) exceeds "
                f"model embedding size {model_vocab_size}. "
                f"This indicates model embeddings were not extended to accommodate coordinate tokens."
            )

        # Validate LM head consistency (if exists)
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            lm_head_size = self.model.lm_head.out_features
            if lm_head_size != model_vocab_size:
                raise RuntimeError(
                    f"LM head output size mismatch: "
                    f"lm_head={lm_head_size}, embeddings={model_vocab_size}. "
                    f"This indicates LM head was not properly resized for coordinate tokens."
                )

        # Success logging with detailed statistics
        logger.info("✅ Coordinate token validation passed:")
        logger.info(f"   📊 Coordinate tokens: {self.max_coord_value}")
        logger.info(f"   📊 Token range: [{coord_start_id}, {coord_end_id})")
        logger.info(f"   📊 Tokenizer vocab size: {tokenizer_vocab_size}")
        logger.info(f"   📊 Model embedding size: {model_vocab_size}")

        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            logger.info(f"   📊 LM head output size: {self.model.lm_head.out_features}")

        logger.info("🎯 Coordinate token consistency validation complete")

    @classmethod
    def create_tokenizer_with_coordinate_tokens(
        cls,
        tokenizer_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_coord_value: int = 1024,
    ) -> PreTrainedTokenizer:
        """
        Factory method to create a tokenizer with coordinate tokens.

        Args:
            tokenizer_path: Path to the base tokenizer
            max_coord_value: Maximum coordinate value

        Returns:
            Tokenizer with coordinate tokens added
        """
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Create token manager to add tokens
        cls(tokenizer, model, max_coord_value)

        return tokenizer

    @classmethod
    def has_coordinate_tokens_in_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> bool:
        """Check if tokenizer has coordinate tokens."""
        vocab = tokenizer.get_vocab()
        return "<|coord_0|>" in vocab

    @classmethod
    def get_coordinate_token_range_from_tokenizer(
        cls, tokenizer: PreTrainedTokenizer, max_coord_value: int = 1024
    ) -> Tuple[int, int]:
        """Get coordinate token range from tokenizer."""
        vocab = tokenizer.get_vocab()
        coord_0_token = "<|coord_0|>"
        if coord_0_token not in vocab:
            raise ValueError("Coordinate tokens not found in tokenizer")
        start_id = vocab[coord_0_token]
        return start_id, start_id + max_coord_value

    def wrap_coordinates(self, coords: List[float], geometry_type: str = "bbox") -> str:
        """
        Wrap coordinates with appropriate geometry tokens.

        Args:
            coords: List of coordinate values (integers or floats)
            geometry_type: "bbox", "line", or "quad"

        Returns:
            Token-wrapped coordinate string in format:
            "<|{geometry}_start|>[<|coord_x1|>,<|coord_x2|>,...]<|{geometry}_end|>"
        """
        # Get start/end tokens based on geometry type
        if geometry_type == "line":
            start_token = "<|line_start|>"
            end_token = "<|line_end|>"
        elif geometry_type == "quad":
            start_token = "<|quad_start|>"
            end_token = "<|quad_end|>"
        else:  # bbox (default)
            start_token = "<|box_start|>"
            end_token = "<|box_end|>"

        # Convert coordinates to coordinate tokens or integers based on mode
        if self.has_coordinate_tokens():
            # Coordinate mode: replace integers with coordinate tokens
            coord_tokens = []
            for coord in coords:
                # Ensure coordinate is an integer and within range
                coord_int = max(0, min(int(coord), self.max_coord_value - 1))
                coord_tokens.append(f"<|coord_{coord_int}|>")

            # Format: [<|coord_x1|>,<|coord_x2|>,...]
            coord_sequence = "[" + ",".join(coord_tokens) + "]"
        else:
            # Standard mode: keep coordinates as integers
            coord_ints = [
                max(0, min(int(coord), self.max_coord_value - 1)) for coord in coords
            ]
            coord_sequence = "[" + ",".join(map(str, coord_ints)) + "]"

        return f"{start_token}{coord_sequence}{end_token}"

    def wrap_description(self, description: str) -> str:
        """Wrap description with object reference tokens."""
        return f"<|object_ref_start|>{description}<|object_ref_end|>"

    def format_object(self, obj: Dict[str, Any]) -> str:
        """
        Format a complete object with coordinates and description.

        Args:
            obj: Object dict with geometry and description

        Returns:
            Formatted object string in format:
            "<|object_ref_start|>desc:xxxxx<|object_ref_end|>,<|{geometry}_start|>[coords]<|{geometry}_end|>"
        """
        # Determine geometry type and coordinates
        if "bbox_2d" in obj:
            coords = obj["bbox_2d"]
            geometry_type = "bbox"
        elif "line" in obj:
            coords = obj["line"]
            geometry_type = "line"
        elif "square" in obj:
            coords = obj["square"]
            geometry_type = "square"
        else:
            raise ValueError(f"Object missing geometry: {obj}")

        # Get description
        description = obj.get("desc", obj.get("description", obj.get("label", "")))

        # Format according to specification:
        # "<|object_ref_start|>desc:xxxxx<|object_ref_end|>,<|{geometry}_start|>[coords]<|{geometry}_end|>"
        wrapped_desc = f"<|object_ref_start|>desc:{description}<|object_ref_end|>"
        wrapped_coords = self.wrap_coordinates(coords, geometry_type)

        return f"{wrapped_desc},{wrapped_coords}"

    def compute_coordinate_losses(self, logits, labels, bbox_spans=None):
        """
        Compute coordinate losses for training.

        Args:
            logits: Model logits tensor
            labels: Label tensor
            bbox_spans: List of bounding box spans (optional, not used currently)

        Returns:
            Dictionary with coordinate loss information
        """
        import torch
        import torch.nn.functional as F

        logger.debug("🎯 Computing coordinate losses")

        # Get device from inputs
        device = logits.device

        # Verify coordinate tokens are available
        if not self.has_coordinate_tokens():
            logger.warning(
                "⚠️ Cannot compute coordinate losses: coordinate tokens not available"
            )
            return {
                "coordinate_loss": torch.tensor(0.0, device=device, requires_grad=True),
                "coordinate_l1_loss": torch.tensor(0.0, device=device),
                "coordinate_spans_found": 0,
                "total_coordinate_tokens": 0,
            }

        # Get coordinate token range
        coord_start_id, coord_end_id = self.get_coordinate_token_range()

        # Create coordinate token mask
        is_coord_token = torch.logical_and(
            labels >= coord_start_id, labels < coord_end_id
        )

        # Check if we have any coordinate tokens
        num_coord_tokens = is_coord_token.sum().item()

        if num_coord_tokens == 0:
            logger.debug("🔍 No coordinate tokens found in labels")
            return {
                "coordinate_loss": torch.tensor(0.0, device=device, requires_grad=True),
                "coordinate_l1_loss": torch.tensor(0.0, device=device),
                "coordinate_spans_found": 0,
                "total_coordinate_tokens": 0,
            }

        # Extract coordinate logits and labels
        # For coordinate tokens, we want to compute L1 loss on normalized probabilities

        # Get only the relevant positions where we have coordinate tokens
        # For each coordinate token position:
        # - Take logits for all token IDs at that position
        # - Apply softmax to get probabilities
        # - Calculate expected coordinate value

        # Extract valid positions (ignore padding/ignore index)
        valid_positions = torch.logical_and(is_coord_token, labels != -100)

        # If no valid positions, return zero losses
        if valid_positions.sum().item() == 0:
            logger.debug("🔍 No valid coordinate positions found")
            return {
                "coordinate_loss": torch.tensor(0.0, device=device, requires_grad=True),
                "coordinate_l1_loss": torch.tensor(0.0, device=device),
                "coordinate_spans_found": 0,
                "total_coordinate_tokens": 0,
            }

        # Calculate L1 loss for coordinate tokens
        batch_size = logits.shape[0]
        l1_losses = []

        for b in range(batch_size):
            # Get positions with coordinate tokens for this sample
            sample_positions = valid_positions[b]

            # Skip if no coordinate tokens in this sample
            if not sample_positions.any():
                continue

            # Get logits and labels at coordinate positions
            coord_logits = logits[
                b, sample_positions, :
            ]  # Shape: [num_coords, vocab_size]
            coord_labels = labels[b, sample_positions]  # Shape: [num_coords]

            # Get the true coordinate values (subtract offset)
            true_values = (
                coord_labels - coord_start_id
            )  # Convert token IDs to coordinate values

            # Convert to float and normalize by max_coord_value
            true_values = true_values.float() / self.max_coord_value

            # Compute soft expectation - weighted average of all coordinate values
            # For each position, we compute expected coordinate from the probability distribution

            # Extract coordinate logits and apply numerical stability improvements
            coord_logits_subset = coord_logits[:, coord_start_id:coord_end_id]

            # NUMERICAL STABILITY FIX 1: Clip extreme logits to prevent overflow/underflow
            coord_logits_subset = torch.clamp(coord_logits_subset, min=-50.0, max=50.0)

            # NUMERICAL STABILITY FIX 2: Use log-softmax for better numerical stability
            log_probs = F.log_softmax(coord_logits_subset, dim=1)
            coord_probs = torch.exp(log_probs)  # Shape: [num_coords, max_coord_value]

            # NUMERICAL STABILITY FIX 3: Add small epsilon to prevent division by zero
            coord_probs = coord_probs + 1e-8
            coord_probs = coord_probs / coord_probs.sum(
                dim=1, keepdim=True
            )  # Renormalize

            # Calculate expected coordinate value for each position
            coord_indices = torch.arange(
                self.max_coord_value, device=device, dtype=torch.float32
            )
            coord_indices = coord_indices / self.max_coord_value  # Normalize to [0, 1]

            # Compute expected coordinate value (weighted average)
            # Shape: [num_coords]
            expected_values = torch.sum(coord_probs * coord_indices.unsqueeze(0), dim=1)

            # NUMERICAL STABILITY FIX 4: Clamp expected values to valid range
            expected_values = torch.clamp(expected_values, min=0.0, max=1.0)

            # Compute L1 loss between expected and true values
            l1_loss = F.l1_loss(expected_values, true_values)

            # NUMERICAL STABILITY FIX 5: Check for NaN/Inf in loss and handle gracefully
            if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                logger.warning(
                    f"NaN/Inf detected in coordinate L1 loss computation. Using fallback value."
                )
                l1_loss = torch.tensor(
                    0.1, device=device, requires_grad=True
                )  # Small positive fallback
            l1_losses.append(l1_loss)

        # Combine losses from all samples
        if l1_losses:
            l1_loss = torch.stack(l1_losses).mean()
        else:
            l1_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Record total number of coordinate tokens for logging
        total_coord_tokens = valid_positions.sum().item()

        # Log detailed info at debug level
        logger.debug(f"🎯 Coordinate loss breakdown:")
        logger.debug(f"   Total coordinate tokens: {total_coord_tokens}")
        logger.debug(f"   L1 loss: {l1_loss.item():.6f}")

        # Return all loss components
        return {
            "coordinate_loss": l1_loss,  # Main loss used by wrapper
            "total_coordinate_loss": l1_loss,  # Total loss
            "coordinate_l1_loss": l1_loss,  # Individual L1 component
            "coordinate_spans_found": 1 if total_coord_tokens > 0 else 0,
            "total_coordinate_tokens": total_coord_tokens,
        }


def create_unified_token_manager(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_coord_value: int = 1024,
    coordinate_tokens_enabled: bool = True,
) -> UnifiedTokenManager:
    """
    Factory function to create unified token manager.

    Args:
        tokenizer: The tokenizer to extend
        model: The model to resize embeddings for
        max_coord_value: Maximum coordinate value (default: 2048)
        coordinate_tokens_enabled: Whether coordinate tokens are enabled globally

    Returns:
        Configured UnifiedTokenManager
    """
    manager = UnifiedTokenManager(tokenizer, model, max_coord_value)
    manager.config.enable_coordinate_tokens = coordinate_tokens_enabled
    return manager


# Backward compatibility class methods
class UnifiedTokenManagerCompat:
    """Backward compatibility methods for UnifiedTokenManager."""

    @staticmethod
    def create_tokenizer_with_coordinate_tokens(
        tokenizer_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_coord_value: int = 1024
    ) -> PreTrainedTokenizer:
        """Create tokenizer with coordinate tokens (backward compatibility)."""
        return UnifiedTokenManager.create_tokenizer_with_coordinate_tokens(
            tokenizer_path, max_coord_value
        )

    @staticmethod
    def has_coordinate_tokens(tokenizer: PreTrainedTokenizer) -> bool:
        """Check if tokenizer has coordinate tokens (backward compatibility)."""
        return UnifiedTokenManager.has_coordinate_tokens_in_tokenizer(tokenizer)

    @staticmethod
    def get_coordinate_token_range(
        tokenizer: PreTrainedTokenizer, max_coord_value: int = 1024
    ) -> Tuple[int, int]:
        """Get coordinate token range (backward compatibility)."""
        return UnifiedTokenManager.get_coordinate_token_range_from_tokenizer(
            tokenizer, max_coord_value
        )


# Backward compatibility will be added at the end of the file


class SpecialTokens:
    """Special tokens for Qwen2.5-VL chat formatting (backward compatibility)."""

    # Chat formatting tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    ENDOFTEXT = "<|endoftext|>"

    # Vision tokens
    IMAGE_PAD = "<|image_pad|>"

    def to_list(self) -> list:
        """Return special tokens as a list for tokenizer.add_special_tokens()."""
        return [
            self.IM_START,
            self.IM_END,
            self.ENDOFTEXT,
            self.IMAGE_PAD,
        ]

    def format_vision_tokens(self, num_tokens: int) -> str:
        """Format vision tokens with spaces to prevent tokenizer issues."""
        if num_tokens <= 0:
            return ""

        # Insert spaces between image_pad tokens to prevent tokenizer returning None IDs
        tokens = [self.IMAGE_PAD] * num_tokens
        return " ".join(tokens)


class SimpleCoordinateManager:
    """
    Simple coordinate manager that works without model embedding resizing.
    Used during ChatProcessor initialization when model is not available.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_coord_value: int = 1024):
        """
        Initialize simple coordinate manager.

        Args:
            tokenizer: The tokenizer to work with
            max_coord_value: Maximum coordinate value
        """
        self.tokenizer = tokenizer
        self.max_coord_value = max_coord_value

        logger.debug(
            f"🔧 SimpleCoordinateManager.__init__ called with max_coord_value={max_coord_value}"
        )
        logger.debug(f"🔧 Tokenizer vocab size: {len(tokenizer.get_vocab())}")

        # Configuration for compatibility
        self.config = type(
            "Config",
            (),
            {
                "enable_coordinate_tokens": False,  # Will be updated based on vocabulary
                "max_coord_value": max_coord_value,
                "box_start_id": None,
                "box_end_id": None,
            },
        )()

        # Token range placeholders - will be updated when coordinate tokens are found
        self.coord_start_id = None
        self.coord_end_id = None

        # Initialize token ranges
        logger.debug(f"🔧 Calling _update_token_ranges()...")
        self._update_token_ranges()
        logger.debug(
            f"🔧 After _update_token_ranges(): enable_coordinate_tokens={self.config.enable_coordinate_tokens}"
        )

    def has_coordinate_tokens(self) -> bool:
        """Check if coordinate tokens are available."""
        return self.coord_start_id is not None and self.coord_end_id is not None

    def _update_token_ranges(self):
        """Update coordinate token ranges from current tokenizer vocabulary."""
        vocab = self.tokenizer.get_vocab()

        # Look for box tokens
        box_start_token = "<|box_start|>"
        box_end_token = "<|box_end|>"

        if box_start_token in vocab:
            self.config.box_start_id = vocab[box_start_token]
        if box_end_token in vocab:
            self.config.box_end_id = vocab[box_end_token]

        # Look for coordinate tokens (correct format: <|coord_0|>)
        coord_0_token = "<|coord_0|>"
        logger.debug(f"🔍 Looking for coordinate token: {coord_0_token}")
        logger.debug(f"🔍 Vocabulary size: {len(vocab)}")
        logger.debug(
            f"🔍 Sample vocab keys: {list(vocab.keys())[-10:]}"
        )  # Show last 10 tokens

        if coord_0_token in vocab:
            self.coord_start_id = vocab[coord_0_token]
            self.coord_end_id = self.coord_start_id + self.max_coord_value
            self.config.enable_coordinate_tokens = True  # Enable coordinate mode
            logger.info(
                f"🎯 Found coordinate tokens: range [{self.coord_start_id}, {self.coord_end_id})"
            )
        else:
            # Check if coordinate tokens were expected to be present
            # If this is being used in a context where coordinate tokens should exist, raise an error
            self.coord_start_id = None
            self.coord_end_id = None
            self.config.enable_coordinate_tokens = False  # Use standard mode
            logger.error(
                "❌ CRITICAL: Coordinate tokens not found in vocabulary but SimpleCoordinateManager was initialized. "
                "This suggests coordinate tokens should be present but are missing from the tokenizer vocabulary."
            )
            raise ValueError(
                "Coordinate tokens not found in vocabulary. If coordinate processing is required, "
                "ensure the tokenizer vocabulary has been extended with coordinate tokens before initialization."
            )
            logger.debug(
                f"🔍 Coordinate token '{coord_0_token}' not found in vocabulary"
            )
            # Debug: Check if any coordinate tokens exist
            coord_tokens_found = [
                token for token in vocab.keys() if token.startswith("<|coord_")
            ]
            logger.debug(
                f"🔍 Found {len(coord_tokens_found)} coordinate tokens in vocab"
            )
            if coord_tokens_found:
                logger.debug(
                    f"🔍 First few coordinate tokens: {coord_tokens_found[:5]}"
                )
                logger.debug(
                    f"🔍 Last few coordinate tokens: {coord_tokens_found[-5:]}"
                )
                # Try to find coord_0 with different patterns
                for token in coord_tokens_found[:10]:
                    logger.debug(f"🔍 Coordinate token: {token} -> ID {vocab[token]}")

    def wrap_coordinates(self, coords: list, geometry_type: str = "bbox") -> str:
        """
        Wrap coordinates with appropriate geometry tokens with enhanced validation.

        Args:
            coords: List of coordinate values (integers)
            geometry_type: "bbox", "line", or "square"

        Returns:
            Token-wrapped coordinate string

        Raises:
            ValueError: If coordinate validation fails
        """
        # Validate inputs
        if not isinstance(coords, list):
            raise ValueError(f"Coordinates must be a list, got {type(coords)}")

        if not coords:
            raise ValueError("Coordinate list cannot be empty")

        if geometry_type not in ["bbox", "line", "square"]:
            raise ValueError(
                f"Invalid geometry_type '{geometry_type}', must be one of: bbox, line, square"
            )

        # Get start/end tokens based on geometry type
        if geometry_type == "line":
            start_token = "<|line_start|>"
            end_token = "<|line_end|>"
        elif geometry_type == "quad":
            start_token = "<|quad_start|>"
            end_token = "<|quad_end|>"
        else:  # bbox (default)
            start_token = "<|box_start|>"
            end_token = "<|box_end|>"

        # Convert coordinates based on mode with enhanced validation
        if self.has_coordinate_tokens():
            # Coordinate mode: replace integers with coordinate tokens
            coord_tokens = []
            for i, coord in enumerate(coords):
                try:
                    # Enhanced integer conversion with validation
                    if isinstance(coord, str):
                        if (
                            not coord.strip()
                            .replace("-", "")
                            .replace(".", "")
                            .isdigit()
                        ):
                            raise ValueError(
                                f"Non-numeric coordinate string: '{coord}'"
                            )
                        coord = float(coord.strip())

                    # Convert to integer with bounds checking
                    coord_int = int(coord)

                    # Validate coordinate range
                    if coord_int < 0:
                        logger.warning(
                            f"Negative coordinate {coord_int} at index {i}, clamping to 0"
                        )
                        coord_int = 0
                    elif coord_int >= self.max_coord_value:
                        logger.warning(
                            f"Coordinate {coord_int} at index {i} exceeds max_coord_value {self.max_coord_value}, "
                            f"clamping to {self.max_coord_value - 1}"
                        )
                        coord_int = self.max_coord_value - 1

                    # Validate coordinate token range availability
                    if not (
                        self.coord_start_id is not None
                        and self.coord_end_id is not None
                    ):
                        raise RuntimeError(
                            f"Coordinate token ranges not initialized: "
                            f"start_id={self.coord_start_id}, end_id={self.coord_end_id}"
                        )

                    # Calculate expected token ID
                    expected_token_id = self.coord_start_id + coord_int
                    if expected_token_id >= self.coord_end_id:
                        raise ValueError(
                            f"Coordinate {coord_int} maps to token ID {expected_token_id} "
                            f"which exceeds coordinate token range [{self.coord_start_id}, {self.coord_end_id})"
                        )

                    # Generate coordinate token
                    coord_token = f"<|coord_{coord_int}|>"
                    coord_tokens.append(coord_token)

                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid coordinate at index {i}: '{coord}' - {e}"
                    ) from e

            coord_sequence = "[" + ",".join(coord_tokens) + "]"

            # Final validation: ensure coordinate tokens were generated
            if not any("<|coord_" in token for token in coord_tokens):
                raise RuntimeError(
                    f"Coordinate token conversion failed - no coordinate tokens generated from {coords}"
                )

        else:
            # Standard mode: keep coordinates as integers with validation
            coord_ints = []
            for i, coord in enumerate(coords):
                try:
                    # Enhanced integer conversion
                    if isinstance(coord, str):
                        if (
                            not coord.strip()
                            .replace("-", "")
                            .replace(".", "")
                            .isdigit()
                        ):
                            raise ValueError(
                                f"Non-numeric coordinate string: '{coord}'"
                            )
                        coord = float(coord.strip())

                    coord_int = int(coord)
                    coord_int = max(0, min(coord_int, self.max_coord_value - 1))
                    coord_ints.append(coord_int)

                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid coordinate at index {i}: '{coord}' - {e}"
                    ) from e

            coord_sequence = "[" + ",".join(map(str, coord_ints)) + "]"

        result = f"{start_token}{coord_sequence}{end_token}"

        # Log successful conversion for debugging
        logger.debug(
            f"✅ Coordinate conversion: {coords} → {result} "
            f"(coordinate_tokens={'enabled' if self.has_coordinate_tokens() else 'disabled'})"
        )

        return result

    def format_object(self, obj: dict) -> str:
        """
        Format a complete object with coordinates and description.

        Args:
            obj: Object dict with geometry and description

        Returns:
            Formatted object string in format:
            "<|object_ref_start|>desc:xxxxx<|object_ref_end|>,<|{geometry}_start|>[coords]<|{geometry}_end|>"
        """
        # Determine geometry type and coordinates
        if "bbox_2d" in obj:
            coords = obj["bbox_2d"]
            geometry_type = "bbox"
        elif "line" in obj:
            coords = obj["line"]
            geometry_type = "line"
        elif "square" in obj:
            coords = obj["square"]
            geometry_type = "square"
        else:
            raise ValueError(f"Object missing geometry: {obj}")

        # Get description
        description = obj.get("desc", obj.get("description", obj.get("label", "")))

        # Format according to specification
        wrapped_desc = f"<|object_ref_start|>desc:{description}<|object_ref_end|>"
        wrapped_coords = self.wrap_coordinates(coords, geometry_type)

        return f"{wrapped_desc},{wrapped_coords}"

    def convert_json_to_coordinate_format(self, json_string: str) -> str:
        """
        Convert JSON format to coordinate token format with enhanced reliability.

        Args:
            json_string: JSON string to convert

        Returns:
            Coordinate token formatted string

        Raises:
            RuntimeError: If conversion fails and coordinate tokens are expected
        """
        # Early validation of coordinate token availability
        if not self.has_coordinate_tokens():
            logger.debug("🎯 Coordinate tokens not available, returning JSON format")
            return json_string

        # Validate that coordinate token ranges are properly initialized
        if self.coord_start_id is None or self.coord_end_id is None:
            logger.error(
                f"❌ Coordinate token ranges not initialized: "
                f"start_id={self.coord_start_id}, end_id={self.coord_end_id}"
            )
            raise RuntimeError("Coordinate token ranges not properly initialized")

        try:
            import json

            # Parse and validate JSON structure with enhanced error reporting
            try:
                objects = json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON input for coordinate conversion: {e}")
                logger.error(f"   📄 JSON content: {json_string[:200]}...")
                raise RuntimeError(
                    f"Invalid JSON input for coordinate conversion: {e}"
                ) from e

            # Handle non-list inputs gracefully
            if not isinstance(objects, list):
                logger.debug(
                    f"🎯 JSON input is not a list (type: {type(objects)}), returning as-is"
                )
                return json_string

            # Handle empty lists
            if not objects:
                logger.debug("🎯 Empty object list, returning JSON format")
                return json_string

            logger.debug(f"🎯 Converting {len(objects)} objects to coordinate format")

            # Convert each object to coordinate token format with enhanced error tracking
            coordinate_formatted_objects = []
            conversion_errors = []
            successful_conversions = 0

            for i, obj in enumerate(objects):
                # Validate object structure
                if not isinstance(obj, dict):
                    error_msg = f"Object {i} is not a dictionary: {type(obj)} = {obj}"
                    conversion_errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
                    continue

                try:
                    # Enhanced object formatting with detailed error context
                    logger.debug(f"   🔄 Converting object {i}: {obj}")
                    formatted_obj = self._format_object_with_validation(obj, i)
                    coordinate_formatted_objects.append(formatted_obj)
                    successful_conversions += 1

                    logger.debug(f"   ✅ Object {i} converted: {formatted_obj}")

                    # Validate that coordinate tokens were actually generated
                    if self.has_coordinate_tokens() and "<|coord_" not in formatted_obj:
                        error_msg = f"Object {i} conversion succeeded but no coordinate tokens found in output"
                        conversion_errors.append(error_msg)
                        logger.error(f"❌ {error_msg}: {formatted_obj}")

                except (ValueError, KeyError, RuntimeError) as e:
                    error_msg = f"Object {i} formatting failed: {e}"
                    conversion_errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
                    logger.error(f"   📄 Object data: {obj}")

            # Report conversion statistics
            logger.debug(
                f"🎯 Conversion complete: {successful_conversions}/{len(objects)} objects converted"
            )

            # Check if we had any conversion errors - FAIL-FAST approach
            if conversion_errors:
                error_summary = "; ".join(conversion_errors)
                logger.error(
                    f"❌ Coordinate token conversion failed for {len(conversion_errors)} objects"
                )
                for error in conversion_errors:
                    logger.error(f"   • {error}")

                raise RuntimeError(
                    f"Coordinate token conversion failed for {len(conversion_errors)}/{len(objects)} objects: {error_summary}"
                )

            # Ensure we have at least one successful conversion
            if not coordinate_formatted_objects:
                raise RuntimeError(
                    "No objects were successfully converted to coordinate format"
                )

            # Join results with space separation
            result = " ".join(coordinate_formatted_objects)

            # Final validation: ensure coordinate tokens are present in output
            if self.has_coordinate_tokens():
                coord_token_count = result.count("<|coord_")
                if coord_token_count == 0:
                    raise RuntimeError(
                        f"Coordinate conversion completed but no coordinate tokens found in output. "
                        f"Result: {result[:200]}..."
                    )

                logger.debug(
                    f"✅ Final result contains {coord_token_count} coordinate tokens"
                )

            logger.debug(
                f"✅ Successfully converted {len(coordinate_formatted_objects)} objects to coordinate format"
            )
            logger.debug(f"   📄 Result preview: {result[:150]}...")

            return result

        except Exception as e:
            # Enhanced error logging with comprehensive context
            logger.error(f"❌ Coordinate token conversion failed: {e}")
            logger.error(f"   📄 Input JSON: {json_string}")
            logger.error(
                f"   🎯 Coordinate tokens available: {self.has_coordinate_tokens()}"
            )
            logger.error(
                f"   🔢 Token range: [{self.coord_start_id}, {self.coord_end_id})"
            )
            logger.error(f"   📊 Max coordinate value: {self.max_coord_value}")

            # Re-raise with context for debugging
            raise RuntimeError(
                f"Failed to convert JSON to coordinate format: {e}. "
                f"Input: {json_string[:100]}{'...' if len(json_string) > 100 else ''}"
            ) from e

    def _format_object_with_validation(self, obj: dict, obj_index: int) -> str:
        """
        Format object with enhanced validation and error reporting.

        Args:
            obj: Object dictionary to format
            obj_index: Index of object for error reporting

        Returns:
            Formatted coordinate token string

        Raises:
            ValueError: If object validation fails
        """
        # Enhanced geometry validation with detailed error reporting
        geometry_types = ["bbox_2d", "line", "square"]
        available_geometry = [gt for gt in geometry_types if gt in obj]

        if not available_geometry:
            raise ValueError(
                f"Object {obj_index} missing geometry data. "
                f"Expected one of: {geometry_types}, found keys: {list(obj.keys())}"
            )

        if len(available_geometry) > 1:
            logger.warning(
                f"⚠️ Object {obj_index} has multiple geometry types: {available_geometry}. "
                f"Using first available: {available_geometry[0]}"
            )

        # Get the primary geometry type and coordinates
        geometry_type_key = available_geometry[0]
        coords = obj[geometry_type_key]

        # Enhanced coordinate validation
        if not isinstance(coords, list):
            raise ValueError(
                f"Object {obj_index} {geometry_type_key} must be a list, got {type(coords)}"
            )

        if not coords:
            raise ValueError(f"Object {obj_index} {geometry_type_key} is empty")

        # Validate coordinate count for each geometry type
        if geometry_type_key == "bbox_2d":
            if len(coords) != 4:
                raise ValueError(
                    f"Object {obj_index} bbox_2d expected 4 coordinates [x1, y1, x2, y2], "
                    f"got {len(coords)}: {coords}"
                )
        elif geometry_type_key == "square":
            if len(coords) != 8:
                raise ValueError(
                    f"Object {obj_index} square expected 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4], "
                    f"got {len(coords)}: {coords}"
                )
        elif geometry_type_key == "line":
            if len(coords) < 4 or len(coords) % 2 != 0:
                raise ValueError(
                    f"Object {obj_index} line expected ≥4 coordinates with even count (x,y pairs), "
                    f"got {len(coords)}: {coords}"
                )

        # Enhanced coordinate value validation with better error messages
        valid_coords = []
        for i, coord in enumerate(coords):
            try:
                # Enhanced type conversion with better error handling
                if isinstance(coord, str):
                    coord_str = coord.strip()
                    if not coord_str:
                        raise ValueError("Empty coordinate string")

                    # Handle negative numbers and decimals
                    if not (
                        coord_str.replace("-", "").replace(".", "").isdigit()
                        or (
                            coord_str.startswith("-")
                            and coord_str[1:].replace(".", "").isdigit()
                        )
                    ):
                        raise ValueError(f"Non-numeric coordinate string: '{coord}'")

                    coord = float(coord_str)

                coord_int = int(coord)

                # Enhanced range validation with specific error messages
                if coord_int < 0:
                    raise ValueError(
                        f"Object {obj_index} {geometry_type_key}[{i}] = {coord_int} is negative. "
                        f"Coordinates must be non-negative integers."
                    )

                if coord_int >= self.max_coord_value:
                    raise ValueError(
                        f"Object {obj_index} {geometry_type_key}[{i}] = {coord_int} "
                        f"exceeds max_coord_value {self.max_coord_value}. "
                        f"Valid range: [0, {self.max_coord_value - 1}]"
                    )

                # Additional validation for coordinate token mode
                if self.has_coordinate_tokens():
                    expected_token_id = int(self.coord_start_id) + coord_int
                    if expected_token_id >= self.coord_end_id:
                        raise ValueError(
                            f"Object {obj_index} {geometry_type_key}[{i}] = {coord_int} "
                            f"maps to token ID {expected_token_id} which exceeds coordinate token range "
                            f"[{self.coord_start_id}, {self.coord_end_id})"
                        )

                valid_coords.append(coord_int)

            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Object {obj_index} {geometry_type_key}[{i}] has invalid coordinate '{coord}': {e}"
                ) from e

        # Enhanced description validation
        description_keys = ["desc", "description", "label"]
        description = None
        found_key = None

        for key in description_keys:
            if key in obj:
                description = obj[key]
                found_key = key
                break

        if description is None:
            raise ValueError(
                f"Object {obj_index} missing description. "
                f"Expected one of: {description_keys}, found keys: {list(obj.keys())}"
            )

        if not isinstance(description, str):
            raise ValueError(
                f"Object {obj_index} {found_key} must be string, got {type(description)}: {description}"
            )

        description = description.strip()
        if not description:
            raise ValueError(
                f"Object {obj_index} {found_key} is empty or whitespace-only"
            )

        # Additional description validation
        if len(description) > 200:  # Reasonable limit for descriptions
            logger.warning(
                f"⚠️ Object {obj_index} description is very long ({len(description)} chars): "
                f"{description[:50]}..."
            )

        # Map geometry type keys to internal geometry types
        geometry_type_map = {"bbox_2d": "bbox", "line": "line", "square": "square"}

        geometry_type_map.get(geometry_type_key, "bbox")

        # Create a clean object for the original format_object method
        clean_obj = {geometry_type_key: valid_coords, "desc": description}

        # Log formatting attempt for debugging
        logger.debug(
            f"🔄 Formatting object {obj_index}: {geometry_type_key}={valid_coords}, desc='{description[:30]}...'"
        )

        # Use the original format_object method with enhanced error context
        try:
            result = self.format_object(clean_obj)

            # Post-formatting validation
            if not result or not result.strip():
                raise ValueError("format_object returned empty result")

            # Validate coordinate tokens were generated (if enabled)
            if self.has_coordinate_tokens():
                if "<|coord_" not in result:
                    raise ValueError(
                        f"Coordinate tokens enabled but none found in formatted result: {result}"
                    )

                # Count coordinate tokens and validate against input coordinates
                coord_token_count = result.count("<|coord_")
                if coord_token_count != len(valid_coords):
                    logger.warning(
                        f"⚠️ Object {obj_index}: Expected {len(valid_coords)} coordinate tokens, "
                        f"found {coord_token_count} in result: {result}"
                    )

            logger.debug(f"✅ Object {obj_index} formatted successfully: {result}")
            return result

        except Exception as e:
            raise ValueError(
                f"Object {obj_index} internal formatting failed with clean_obj={clean_obj}: {e}"
            ) from e


class TokenFormatter:
    """Token formatting utilities (backward compatibility)."""

    def __init__(self):
        self.special_tokens = SpecialTokens()


# Add backward compatibility static methods to UnifiedTokenManager
# This must be done after the class is fully defined


# Create wrapper functions that properly handle the static method calls
def has_coordinate_tokens_static(tokenizer: PreTrainedTokenizer) -> bool:
    """Static method wrapper for backward compatibility."""
    return UnifiedTokenManager.has_coordinate_tokens_in_tokenizer(tokenizer)


def get_coordinate_token_range_static(
    tokenizer: PreTrainedTokenizer, max_coord_value: int = 1024
) -> Tuple[int, int]:
    """Static method wrapper for backward compatibility."""
    return UnifiedTokenManager.get_coordinate_token_range_from_tokenizer(
        tokenizer, max_coord_value
    )


# Add the static methods to the class using setattr to avoid conflicts
# Use different names to avoid overriding instance methods
setattr(
    UnifiedTokenManager,
    "has_coordinate_tokens_static",
    staticmethod(has_coordinate_tokens_static),
)
setattr(
    UnifiedTokenManager,
    "get_coordinate_token_range_static",
    staticmethod(get_coordinate_token_range_static),
)
