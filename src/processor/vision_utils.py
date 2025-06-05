"""
Vision Token Expansion Utilities

Handles the expansion of <image> tokens to the correct number of
<|image_pad|> tokens based on image grid_thw calculations.
"""

import logging
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class VisionTokenExpander:
    """Handles vision token expansion for Qwen2.5-VL."""

    def __init__(self, image_processor):
        """Initialize with image processor."""
        self.image_processor = image_processor
        self.merge_size = getattr(image_processor, "merge_size", 2)

    def calculate_token_count(self, image_path: str) -> int:
        """Calculate number of tokens needed for an image."""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Process with image processor to get grid_thw
            processed = self.image_processor.preprocess(image, return_tensors="pt")
            grid_thw = processed["image_grid_thw"][0]  # [t, h, w]

            # Calculate tokens: grid_thw.prod() // merge_size^2
            # Following official processor logic
            token_count = grid_thw.prod().item() // (self.merge_size**2)

            logger.debug(
                f"Image {image_path}: grid_thw={grid_thw.tolist()}, tokens={token_count}"
            )
            return token_count

        except Exception as e:
            logger.error(f"Failed to calculate tokens for {image_path}: {e}")
            # Fallback to default
            return 64

    def expand_image_tokens(self, text: str, image_paths: List[str]) -> str:
        """
        Expand <image> tokens in text to correct number of <|image_pad|> tokens.

        Args:
            text: Text containing <image> tokens
            image_paths: List of image paths corresponding to <image> tokens

        Returns:
            Text with <image> replaced by <|image_pad|> * N tokens
        """
        if "<image>" not in text:
            return text

        # Count image tokens
        image_token_count = text.count("<image>")
        if image_token_count != len(image_paths):
            raise ValueError(
                f"Mismatch: {image_token_count} <image> tokens but {len(image_paths)} images"
            )

        # Calculate token counts for each image
        token_counts = [self.calculate_token_count(path) for path in image_paths]

        # Replace tokens one by one
        result = text
        for i, token_count in enumerate(token_counts):
            if "<image>" not in result:
                break

            # Replace with expanded tokens
            expanded_tokens = "<|image_pad|>" * token_count
            result = result.replace("<image>", expanded_tokens, 1)

            logger.debug(f"Replaced image {i + 1} with {token_count} tokens")

        return result

    def process_conversation_images(
        self, conversation_text: str, image_paths: List[str]
    ) -> Tuple[str, List[int]]:
        """
        Process conversation text and return expanded text with token counts.

        Returns:
            Tuple of (expanded_text, token_counts_per_image)
        """
        # Calculate token counts
        token_counts = [self.calculate_token_count(path) for path in image_paths]

        # Expand tokens
        expanded_text = self.expand_image_tokens(conversation_text, image_paths)

        return expanded_text, token_counts


def create_vision_token_expander(image_processor) -> VisionTokenExpander:
    """Factory function to create vision token expander."""
    return VisionTokenExpander(image_processor)
