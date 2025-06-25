"""
Special Tokens Manager for Qwen2.5-VL

Manages vision-related special tokens used in Qwen2.5-VL model.
Object detection now uses pure JSON format instead of special tokens.
"""

import json
from typing import Any, Dict, List, Optional

from src.logger_utils import get_tokens_logger

logger = get_tokens_logger()

IGNORE_INDEX = -100


class SpecialTokens:
    """Centralized special tokens management for Qwen2.5-VL vision processing."""

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"
    IMAGE_PAD = "<|image_pad|>"

    # Deprecated / not currently used but kept for vocabulary completeness
    VISION_PAD = "<|vision_pad|>"
    VIDEO_PAD = "<|video_pad|>"
    ENDOFTEXT = "<|endoftext|>"
    IMAGE_PLACEHOLDER = "<image>"
    VIDEO_PLACEHOLDER = "<video>"

    # For label masking
    IGNORE_INDEX = IGNORE_INDEX

    TOKEN_IDS = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        # Deprecated / unused
        "<|vision_pad|>": 151654,
        "<|video_pad|>": 151656,
        "<|endoftext|>": 151643,
    }

    def to_list(self) -> List[str]:
        """Return a list of all special token strings."""
        return [
            self.IM_START,
            self.IM_END,
            self.VISION_START,
            self.VISION_END,
            self.IMAGE_PAD,
            self.VISION_PAD,
            self.VIDEO_PAD,
            self.ENDOFTEXT,
            self.IMAGE_PLACEHOLDER,
            self.VIDEO_PLACEHOLDER,
        ]

    @classmethod
    def get_token_id(cls, token: str) -> Optional[int]:
        """Get token ID for a special token."""
        return cls.TOKEN_IDS.get(token)

    @classmethod
    def format_vision_tokens(cls, num_image_tokens: int) -> str:
        """Format vision token sequence for an image.

        The tokenizer sometimes fails to recognise *contiguously* concatenated
        special tokens (e.g. ``"<|image_pad|><|image_pad|>"``) and returns
        ``None`` IDs.  We insert a single whitespace delimiter between each
        `<|image_pad|>` marker so that every occurrence is isolated and can be
        mapped to a valid token id.  Whitespace is an ordinary token which is
        ignored by the vision encoder, so this change is loss-less for the
        model while guaranteeing robust tokenisation.
        """

        if num_image_tokens <= 0:
            raise ValueError(f"num_image_tokens must be > 0, got {num_image_tokens}")

        # Insert spaces between successive <|image_pad|> tokens
        image_pads = " ".join([cls.IMAGE_PAD] * num_image_tokens)
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

    def __init__(self) -> None:
        self.tokens = SpecialTokens()

    def format_conversation_turn(self, role: str, content: str) -> str:
        """Format a single conversation turn."""
        return self.tokens.format_chat_message(role, content)

    def format_complete_conversation(self, messages: List[Dict[str, str]]) -> str:
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

    def format_objects_as_json(self, objects: List[Dict[str, Any]]) -> str:
        """
        Format objects as JSON array (new Qwen2.5-VL compatible format).

        Args:
            objects: List of objects with 'bbox_2d' and 'desc' keys

        Returns:
            JSON string representation of the objects
        """
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            bbox_2d = obj.get("bbox_2d", [0, 0, 0, 0])
            desc = obj.get("desc", "unknown")

            json_obj = {"bbox_2d": bbox_2d, "desc": desc}
            json_objects.append(json_obj)

        return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))
