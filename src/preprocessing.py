#!/usr/bin/env python3
"""
Qwen2.5-VL Preprocessing with Clean Architecture

This module implements a clean separation between:
1. Raw semantic data (no special tokens)
2. Chat templates (Jinja2-based conversation formatting)
3. Model processor (token expansion and model-specific formatting)

Architecture:
Raw Data → Chat Templates → Model Processor → Training Data
"""

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import Qwen2VLProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Chat Templates
# ============================================================================


class ChatTemplates:
    """Jinja2-style chat templates for conversation formatting."""

    @staticmethod
    def format_single_round(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format single-round conversation.

        Input: {"images": [...], "objects": [...]}
        Output: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        # User message with image placeholder
        user_content = (
            "<image>\nPlease describe the objects in this image with their locations."
        )

        # Assistant response with object descriptions
        objects = data.get("objects", [])
        if not objects:
            assistant_content = "I don't see any objects in this image."
        else:
            descriptions = []
            for obj in objects:
                box = obj["box"]
                desc = obj["desc"]
                # Use clean object reference format
                descriptions.append(
                    f'<object_ref_start>{{"box": {box}, "desc": "{desc}"}}<object_ref_end>'
                )

            assistant_content = " ".join(descriptions)

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    @staticmethod
    def format_multi_round(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format multi-round conversation with examples.

        Input: {"images": [...], "objects": [...], "examples": [...]}
        Output: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        conversation = []

        # Add examples first
        examples = data.get("examples", [])
        for example in examples:
            example_turns = ChatTemplates.format_single_round(example)
            conversation.extend(example_turns)

        # Add main query
        main_turns = ChatTemplates.format_single_round(data)
        conversation.extend(main_turns)

        return conversation

    @staticmethod
    def format_conversation(
        data: Dict[str, Any], multi_round: bool = False
    ) -> List[Dict[str, str]]:
        """Format conversation based on mode."""
        if multi_round:
            return ChatTemplates.format_multi_round(data)
        else:
            return ChatTemplates.format_single_round(data)


# ============================================================================
# Vision Token Processor
# ============================================================================


class VisionTokenProcessor:
    """Handles vision token expansion and image processing."""

    def __init__(self, processor: Qwen2VLProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Special tokens
        self.image_token = "<image>"
        self.image_pad_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

    def calculate_image_tokens(self, image_path: str) -> int:
        """Calculate number of tokens needed for an image."""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Get grid dimensions
            inputs = self.processor(images=[image], text="dummy", return_tensors="pt")
            grid_thw = inputs.get("image_grid_thw")

            if grid_thw is not None and len(grid_thw) > 0:
                # Calculate tokens: grid_thw.prod() // spatial_merge_unit
                spatial_merge_unit = getattr(self.processor, "spatial_merge_unit", 4)
                num_tokens = grid_thw[0].prod().item() // spatial_merge_unit
                return num_tokens
            else:
                # Fallback to default
                return 256

        except Exception as e:
            logger.warning(f"Failed to calculate tokens for {image_path}: {e}")
            return 256

    def expand_image_tokens(self, text: str, image_paths: List[str]) -> str:
        """
        Expand <image> tokens to proper format.

        <image> → <|vision_start|><|image_pad|> * N <|vision_end|>
        """
        if not image_paths:
            return text

        # Calculate tokens for each image
        image_token_counts = []
        for image_path in image_paths:
            num_tokens = self.calculate_image_tokens(image_path)
            image_token_counts.append(num_tokens)

        # Replace <image> tokens
        result = text
        image_idx = 0

        while self.image_token in result and image_idx < len(image_token_counts):
            num_tokens = image_token_counts[image_idx]

            # Create expanded token sequence
            pad_tokens = self.image_pad_token * num_tokens
            expanded = f"{self.vision_start_token}{pad_tokens}{self.vision_end_token}"

            # Replace first occurrence
            result = result.replace(self.image_token, expanded, 1)
            image_idx += 1

        return result

    def process_conversation(
        self, conversation: List[Dict[str, str]], image_paths: List[str]
    ) -> List[Dict[str, str]]:
        """Process conversation with proper token expansion."""
        processed_conversation = []

        for turn in conversation:
            content = turn["content"]

            # Expand image tokens in content
            if self.image_token in content:
                content = self.expand_image_tokens(content, image_paths)

            processed_conversation.append({"role": turn["role"], "content": content})

        return processed_conversation


# ============================================================================
# Model Processor
# ============================================================================


class QwenModelProcessor:
    """Main processor combining templates and vision processing."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.vision_processor = VisionTokenProcessor(self.processor)

        # Add EOS token to final assistant responses
        self.eos_token = self.tokenizer.eos_token or "<|endoftext|>"

    def process_sample(
        self, data: Dict[str, Any], multi_round: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single sample from clean semantic data to model inputs.

        Input: Clean semantic data
        Output: Model-ready training sample
        """
        # 1. Format conversation using templates
        conversation = ChatTemplates.format_conversation(data, multi_round=multi_round)

        # 2. Process with vision token expansion
        image_paths = data.get("images", [])
        processed_conversation = self.vision_processor.process_conversation(
            conversation, image_paths
        )

        # 3. Add EOS token to final assistant response
        if processed_conversation and processed_conversation[-1]["role"] == "assistant":
            processed_conversation[-1]["content"] += self.eos_token

        # 4. Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            processed_conversation, tokenize=False, add_generation_prompt=False
        )

        # 5. Load images
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")

        # 6. Create model inputs
        if images:
            inputs = self.processor(
                text=[formatted_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=[formatted_text], return_tensors="pt", padding=True
            )

        # 7. Create labels for training
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask user tokens (only train on assistant responses)
        labels = self._mask_user_tokens(labels, processed_conversation)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "conversation": processed_conversation,
            "formatted_text": formatted_text,
        }

        # Add vision inputs if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"][0]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"][0]

        return result

    def _mask_user_tokens(
        self, labels: torch.Tensor, conversation: List[Dict[str, str]]
    ) -> torch.Tensor:
        """Mask user tokens in labels (only train on assistant responses)."""
        # For now, return labels as-is
        # TODO: Implement proper masking based on conversation structure
        return labels

    def process_dataset(
        self,
        input_file: str,
        output_file: str,
        multi_round: bool = False,
        max_samples: Optional[int] = None,
    ):
        """Process entire dataset."""
        # Load clean semantic data
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Processing {len(samples)} samples...")

        # Process each sample
        processed_samples = []
        for i, sample in enumerate(samples):
            try:
                processed = self.process_sample(sample, multi_round=multi_round)

                # Convert tensors to lists for JSON serialization
                serializable = {}
                for key, value in processed.items():
                    if isinstance(value, torch.Tensor):
                        serializable[key] = value.tolist()
                    else:
                        serializable[key] = value

                processed_samples.append(serializable)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")

        # Save processed data
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(
            f"Saved {len(processed_samples)} processed samples to {output_file}"
        )


# ============================================================================
# Main Functions
# ============================================================================


def main():
    """Main preprocessing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-VL Preprocessing")
    parser.add_argument(
        "--input_file", required=True, help="Input clean semantic data file"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output processed data file"
    )
    parser.add_argument(
        "--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--multi_round", action="store_true", help="Enable multi-round conversations"
    )
    parser.add_argument("--max_samples", type=int, help="Maximum samples to process")

    args = parser.parse_args()

    # Create processor
    processor = QwenModelProcessor(args.model_name)

    # Process dataset
    processor.process_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        multi_round=args.multi_round,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
