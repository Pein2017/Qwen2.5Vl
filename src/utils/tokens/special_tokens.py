"""
Unified Token Management for Qwen2.5-VL with Coordinate Support

Simple, automatic token management that:
1. Reuses existing tokens where possible
2. Automatically handles new token addition
3. Manages coordinate tokens [0, max_coord_value]
4. No manual token ID configuration needed
"""

from typing import Any, Dict, List

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
        "<|square_start|>",
        "<|square_end|>",
    ]

    # Vision tokens (existing)
    VISION_TOKENS = {
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
        "image_pad": "<|image_pad|>",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        max_coord_value: int = 2048,
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
            return 0

    def _add_coordinate_tokens(self) -> int:
        """Add coordinate tokens [0, max_coord_value-1] and return count added."""
        logger.info(f"➕ Adding coordinate tokens [0, {self.max_coord_value - 1}]...")

        # Generate coordinate token names
        coord_tokens = [f"<coord_{i}>" for i in range(self.max_coord_value)]

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
            return total_added
        else:
            logger.info("   ✅ All coordinate tokens already exist")
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

        token_name = f"<coord_{coord_value}>"
        vocab = self.tokenizer.get_vocab()

        if token_name not in vocab:
            raise ValueError(f"Coordinate token {token_name} not found in vocabulary")

        return vocab[token_name]

    def wrap_coordinates(self, coords: List[float], geometry_type: str = "bbox") -> str:
        """
        Wrap coordinates with appropriate geometry tokens.

        Args:
            coords: List of coordinate values
            geometry_type: "bbox", "line", or "square"

        Returns:
            Token-wrapped coordinate string
        """
        # Get start/end tokens based on geometry type
        if geometry_type == "line":
            start_token = "<|line_start|>"
            end_token = "<|line_end|>"
        elif geometry_type == "square":
            start_token = "<|square_start|>"
            end_token = "<|square_end|>"
        else:  # bbox (default)
            start_token = "<|box_start|>"
            end_token = "<|box_end|>"

        # Convert coordinates to coordinate tokens
        coord_tokens = []
        for coord in coords:
            # Scale normalized coordinates [0,1] to coordinate token range [0, max_coord_value)
            # Assume input coordinates are normalized to [0,1] range
            if isinstance(coord, (int, float)) and 0 <= coord <= 1:
                # Scale to coordinate token range
                coord_scaled = coord * (self.max_coord_value - 1)
                coord_int = max(0, min(int(coord_scaled), self.max_coord_value - 1))
            else:
                # Handle absolute coordinates or out-of-range values
                coord_int = max(0, min(int(coord), self.max_coord_value - 1))
            
            coord_tokens.append(f"<coord_{coord_int}>")

        # Join without spaces (coordinate tokens are contiguous)
        coord_sequence = "".join(coord_tokens)

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
            Formatted object string with tokens
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
        description = obj.get("desc", obj.get("description", ""))

        # Format both parts
        wrapped_coords = self.wrap_coordinates(coords, geometry_type)
        wrapped_desc = self.wrap_description(description)

        return f"{wrapped_coords}{wrapped_desc}"

    def compute_coordinate_losses(self, logits, labels, bbox_spans=None):
        """
        Compute coordinate losses for training.

        Args:
            logits: Model logits tensor
            labels: Label tensor
            bbox_spans: List of bounding box spans (optional)

        Returns:
            Dictionary with coordinate loss information
        """
        import torch

        # For now, return empty losses since coordinate token training is complex
        # This is a placeholder implementation to allow training to proceed
        logger.debug("🎯 Computing coordinate losses (placeholder implementation)")

        return {
            "coordinate_loss": torch.tensor(
                0.0, device=logits.device, requires_grad=True
            ),  # Expected by wrapper
            "total_coordinate_loss": torch.tensor(
                0.0, device=logits.device, requires_grad=True
            ),
            "coordinate_l1_loss": torch.tensor(0.0, device=logits.device),
            "coordinate_spans_found": 0,
            "total_coordinate_tokens": 0,
        }


def create_unified_token_manager(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, max_coord_value: int = 2048
) -> UnifiedTokenManager:
    """
    Factory function to create unified token manager.

    Args:
        tokenizer: The tokenizer to extend
        model: The model to resize embeddings for
        max_coord_value: Maximum coordinate value (default: 2048)

    Returns:
        Configured UnifiedTokenManager
    """
    return UnifiedTokenManager(tokenizer, model, max_coord_value)


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

    def __init__(self, tokenizer: PreTrainedTokenizer, max_coord_value: int = 2048):
        """
        Initialize simple coordinate manager.

        Args:
            tokenizer: The tokenizer to work with
            max_coord_value: Maximum coordinate value
        """
        self.tokenizer = tokenizer
        self.max_coord_value = max_coord_value

        # Configuration for compatibility
        self.config = type(
            "Config",
            (),
            {
                "enable_coordinate_tokens": True,
                "max_coord_value": max_coord_value,
                "box_start_id": None,
                "box_end_id": None,
            },
        )()

        # Token range placeholders - will be updated when coordinate tokens are found
        self.coord_start_id = None
        self.coord_end_id = None

        # Initialize token ranges
        self._update_token_ranges()

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

        # Look for coordinate tokens
        coord_0_token = "<coord_0>"
        if coord_0_token in vocab:
            self.coord_start_id = vocab[coord_0_token]
            self.coord_end_id = self.coord_start_id + self.max_coord_value
            logger.info(
                f"🎯 Found coordinate tokens: range [{self.coord_start_id}, {self.coord_end_id})"
            )
        else:
            logger.warning(
                "⚠️ Coordinate tokens not found in vocabulary - coordinate features disabled"
            )

    def convert_json_to_coordinate_format(self, json_string: str) -> str:
        """
        Convert JSON format to coordinate token format.
        Returns JSON string as fallback if coordinate tokens are not available.
        """
        if self.coord_start_id is None:
            logger.debug("🎯 Coordinate tokens not available, returning JSON format")
            return json_string

        try:
            import json
            # Parse the JSON string to extract objects
            objects = json.loads(json_string)
            
            if not isinstance(objects, list):
                logger.debug("🎯 JSON is not a list, returning original format")
                return json_string
            
            # Convert each object to coordinate token format
            coordinate_formatted_objects = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                    
                try:
                    formatted_obj = self.format_object(obj)
                    coordinate_formatted_objects.append(formatted_obj)
                except (ValueError, KeyError) as e:
                    logger.debug(f"🎯 Failed to format object {obj}: {e}, keeping as JSON")
                    # If formatting fails, keep the object in JSON format
                    coordinate_formatted_objects.append(json.dumps(obj))
            
            # Join all formatted objects with spaces
            result = " ".join(coordinate_formatted_objects)
            logger.debug(f"🎯 Successfully converted JSON to coordinate format: {len(coordinate_formatted_objects)} objects")
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"🎯 Failed to parse JSON or convert to coordinate format: {e}, returning original")
            return json_string


class TokenFormatter:
    """Token formatting utilities (backward compatibility)."""

    def __init__(self):
        self.special_tokens = SpecialTokens()
