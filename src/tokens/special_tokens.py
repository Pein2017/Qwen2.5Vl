"""
Special Tokens Manager for Qwen2.5-VL

Manages vision-related special tokens used in Qwen2.5-VL model.
Object detection now uses pure JSON format instead of special tokens.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpecialTokens:
    """Centralized special tokens management for Qwen2.5-VL vision processing."""

    # Chat tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    # Vision tokens (still needed for image processing)
    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"
    VISION_PAD = "<|vision_pad|>"
    IMAGE_PAD = "<|image_pad|>"
    VIDEO_PAD = "<|video_pad|>"

    # Standard tokens
    ENDOFTEXT = "<|endoftext|>"

    # Placeholder tokens (replaced during processing)
    IMAGE_PLACEHOLDER = "<image>"
    VIDEO_PLACEHOLDER = "<video>"

    # Object detection tokens (deprecated - now using JSON format)
    # These are kept for reference but not used in processing
    OBJECT_REF_START = "<|object_ref_start|>"  # DEPRECATED
    OBJECT_REF_END = "<|object_ref_end|>"  # DEPRECATED
    BOX_START = "<|box_start|>"  # DEPRECATED
    BOX_END = "<|box_end|>"  # DEPRECATED
    QUAD_START = "<|quad_start|>"  # DEPRECATED
    QUAD_END = "<|quad_end|>"  # DEPRECATED

    # Token ID mapping (from tokenizer_config.json)
    TOKEN_IDS = {
        IM_START: 151644,
        IM_END: 151645,
        VISION_START: 151652,
        VISION_END: 151653,
        VISION_PAD: 151654,
        IMAGE_PAD: 151655,
        VIDEO_PAD: 151656,
        ENDOFTEXT: 151643,
        # Deprecated object detection tokens (kept for reference)
        OBJECT_REF_START: 151646,
        OBJECT_REF_END: 151647,
        BOX_START: 151648,
        BOX_END: 151649,
        QUAD_START: 151650,
        QUAD_END: 151651,
    }

    @classmethod
    def get_token_id(cls, token: str) -> Optional[int]:
        """Get token ID for a special token."""
        return cls.TOKEN_IDS.get(token)

    @classmethod
    def format_vision_tokens(cls, num_image_tokens: int) -> str:
        """Format vision token sequence for an image."""
        image_pads = cls.IMAGE_PAD * num_image_tokens
        return f"{cls.VISION_START}{image_pads}{cls.VISION_END}"

    @classmethod
    def format_chat_message(cls, role: str, content: str) -> str:
        """Format a chat message with proper tokens."""
        return f"{cls.IM_START}{role}\n{content}{cls.IM_END}"

    @classmethod
    def validate_tokenizer_config(cls, tokenizer_config_path: str) -> bool:
        """Validate that tokenizer config contains expected special tokens."""
        try:
            with open(tokenizer_config_path, "r") as f:
                config = json.load(f)

            additional_special_tokens = config.get("additional_special_tokens", [])

            # Check that vision tokens are present (object detection tokens no longer required)
            expected_tokens = [
                cls.IM_START,
                cls.IM_END,
                cls.VISION_START,
                cls.VISION_END,
                cls.VISION_PAD,
                cls.IMAGE_PAD,
                cls.VIDEO_PAD,
            ]

            missing_tokens = []
            for token in expected_tokens:
                if token not in additional_special_tokens:
                    missing_tokens.append(token)

            if missing_tokens:
                logger.warning(
                    f"Missing vision tokens in tokenizer config: {missing_tokens}"
                )
                return False

            logger.info("✅ All required vision tokens validated in tokenizer config")
            return True

        except Exception as e:
            logger.error(f"Failed to validate tokenizer config: {e}")
            return False


class TokenFormatter:
    """Helper class for formatting content with special tokens (vision only)."""

    def __init__(self):
        self.tokens = SpecialTokens()

    def format_conversation_turn(self, role: str, content: str) -> str:
        """Format a single conversation turn."""
        return self.tokens.format_chat_message(role, content)

    def format_complete_conversation(self, messages: list) -> str:
        """Format a complete conversation."""
        formatted_turns = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_turn = self.format_conversation_turn(role, content)
            formatted_turns.append(formatted_turn)

        # Add final assistant start for generation
        formatted_turns.append(f"{self.tokens.IM_START}assistant\n")

        return "\n".join(formatted_turns)

    def format_objects_as_json(self, objects: list) -> str:
        """
        Format objects as JSON array (new Qwen2.5-VL compatible format).

        Args:
            objects: List of objects with 'box' and 'desc' keys

        Returns:
            JSON string representation of the objects
        """
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            box = obj.get("box", [0, 0, 0, 0])
            desc = obj.get("desc", "unknown")

            json_obj = {"bbox": box, "description": desc}
            json_objects.append(json_obj)

        return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))
