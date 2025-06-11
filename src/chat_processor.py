#!/usr/bin/env python3
"""
Unified Chat Processor for Qwen2.5-VL BBU Dataset

This module provides a clean, single-purpose processor for the simplified JSONL format:
- Loads JSONL with 'examples' and 'target' structure
- Creates proper few-shot chat templates
- Uses pure JSON format for object detection (compatible with Qwen2.5-VL)
- Manages vision token expansion
- Prepares data for training with proper masking

Pipeline: Simplified JSONL → ChatProcessor → Training-ready samples
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.config import config
from src.logger_utils import get_chat_logger
from src.prompt import (
    CHINESE_BASE_PROMPT,
    CHINESE_CANDIDATES_SECTION,
    CHINESE_FEW_SHOT_SECTION,
    ENGLISH_BASE_PROMPT,
    ENGLISH_CANDIDATES_SECTION,
    ENGLISH_FEW_SHOT_SECTION,
)
from src.tokens import SpecialTokens

logger = get_chat_logger()


class ChatProcessor:
    """
    Single-purpose processor for BBU dataset chat template creation.

    Handles the complete pipeline from simplified JSONL to training-ready format.
    Uses pure JSON format for object detection output (Qwen2.5-VL compatible).
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
    ):
        """
        Initialize the chat processor using global configuration.

        Args:
            tokenizer: Qwen2.5-VL tokenizer
            image_processor: Qwen2.5-VL image processor
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # Store data root from global config
        self.data_root = Path(config.data_root)

        # Initialize special tokens (only for vision tokens, not for object detection)
        self.tokens = SpecialTokens()

        # Get language from global config
        self.language = config.language

        # Build system prompt using global config
        self.system_prompt = self._build_system_prompt()

        # Log configuration
        logger.info(f"✅ ChatProcessor initialized:")
        logger.info(f"   Language: {self.language}")
        logger.info(f"   Data root: {config.data_root}")
        logger.info(f"   Model max length: {config.max_total_length}")
        logger.info(f"   Use candidates: {config.use_candidates}")
        logger.info(f"   Output format: Pure JSON (Qwen2.5-VL compatible)")

        # Log a sample of the system prompt
        logger.info(f"📄 System prompt sample (first 500 chars):")
        logger.info(f"   {repr(self.system_prompt[:500])}")
        logger.info(f"📄 System prompt sample (last 200 chars):")
        logger.info(f"   {repr(self.system_prompt[-200:])}")

    def _build_system_prompt(self) -> str:
        """Build system prompt with pure JSON format for object detection."""

        if self.language == "chinese":
            base_prompt = CHINESE_BASE_PROMPT
            candidates_template = CHINESE_CANDIDATES_SECTION
            few_shot_section = CHINESE_FEW_SHOT_SECTION
        else:  # English
            base_prompt = ENGLISH_BASE_PROMPT
            candidates_template = ENGLISH_CANDIDATES_SECTION
            few_shot_section = ENGLISH_FEW_SHOT_SECTION

        # Add candidate phrases if provided
        if config.use_candidates and config.candidates_file:
            candidates_path = Path(config.candidates_file)
            if candidates_path.exists():
                with open(candidates_path, "r", encoding="utf-8") as f:
                    candidates_data = json.load(f)

                phrase_list = candidates_data.get("phrase_list", [])

                # Format as numbered list for better readability
                formatted_phrases = "\n".join(
                    [f"{i + 1}) {phrase}" for i, phrase in enumerate(phrase_list)]
                )

                # Format the template with the phrases
                candidates_section = candidates_template.format(
                    formatted_phrases=formatted_phrases
                )
                base_prompt += candidates_section
            else:
                logger.warning(f"Candidates file not found: {config.candidates_file}")

        return base_prompt + few_shot_section

    def process_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single sample from simplified JSONL format.

        Args:
            raw_sample: Sample with 'examples' and 'target' structure

        Returns:
            Dict containing input_ids, labels, attention_mask, pixel_values, image_grid_thw, and ground_truth_objects
        """
        # 1. Create conversation messages
        conversation_messages = self._create_conversation_messages(raw_sample)

        # 2. Process images, expand vision tokens, and get image dimensions
        (
            processed_conversation,
            images,
            image_dims,
        ) = self._process_images_and_tokens(
            conversation_messages, self._extract_all_image_paths(raw_sample)
        )

        # 3. Apply chat template and tokenize
        input_ids, labels = self._tokenize_conversation(processed_conversation)

        # 4. Process images for model input
        pixel_values, image_grid_thw = self._process_images_for_model(images)

        # 5. Extract and normalize ground truth objects for the target image
        ground_truth_objects = self._extract_and_normalize_ground_truth(
            raw_sample, image_dims
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ground_truth_objects": ground_truth_objects,
        }

    def _create_conversation_messages(
        self, sample: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create conversation messages from sample data."""
        messages = []

        # Add system message
        messages.append({"role": "system", "content": self.system_prompt})

        # Process examples if present
        if "examples" in sample:
            for example in sample["examples"]:
                # User message with image
                user_content = "<image>"
                messages.append({"role": "user", "content": user_content})

                # Assistant response with JSON format
                objects = example.get("objects", [])
                sorted_objects = self._sort_objects_by_position(objects)
                assistant_response = self._format_objects_response(sorted_objects)
                messages.append({"role": "assistant", "content": assistant_response})

        # Add target
        target = sample.get("target", sample)  # Fallback to sample itself if no target
        user_content = "<image>"
        messages.append({"role": "user", "content": user_content})

        # Target response
        target_objects = target.get("objects", [])
        sorted_target_objects = self._sort_objects_by_position(target_objects)
        target_response = self._format_objects_response(sorted_target_objects)
        messages.append({"role": "assistant", "content": target_response})

        return messages

    def _extract_all_image_paths(self, sample: Dict[str, Any]) -> List[str]:
        """Extract all image paths from sample."""
        image_paths = []

        # Extract from examples
        if "examples" in sample:
            for example in sample["examples"]:
                image_paths.extend(example.get("images", []))

        # Extract from target
        target = sample.get("target", sample)
        image_paths.extend(target.get("images", []))

        return image_paths

    def _sort_objects_by_position(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort objects by position (top-to-bottom, left-to-right)."""

        def sort_key(obj):
            box = obj.get("box", [0, 0, 0, 0])
            return (box[1], box[0])  # Sort by y first, then x

        return sorted(objects, key=sort_key)

    def _format_objects_response(self, objects: List[Dict[str, Any]]) -> str:
        """Format objects into JSON array format."""
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            box = obj.get("box", [0, 0, 0, 0])
            desc = obj.get("desc", "unknown")

            # Create JSON object
            json_obj = {"bbox_2d": box, "label": desc}
            json_objects.append(json_obj)

        # Return formatted JSON array
        return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))

    def _process_images_and_tokens(
        self, conversation: List[Dict[str, str]], image_paths: List[str]
    ) -> Tuple[List[Dict[str, str]], List[Image.Image], List[Tuple[int, int]]]:
        """
        Process images and expand vision tokens in conversation.

        Replaces <image> placeholders with proper vision token sequences.
        """
        processed_conversation = []
        images = []
        image_dims = []
        image_index = 0

        for message in conversation:
            content = message["content"]

            # Process image placeholders
            while "<image>" in content and image_index < len(image_paths):
                # Load image
                image_path = self.data_root / image_paths[image_index]
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    image_dims.append(image.size)  # (width, height)
                except Exception as e:
                    logger.error(f"Failed to load image: {image_path}, error: {e}")
                    # Remove the placeholder and skip to the next image
                    content = content.replace("<image>", "", 1)
                    image_index += 1
                    continue

                # Calculate number of vision tokens required for this image
                num_vision_tokens = self._calculate_image_tokens(image)

                # Create vision token sequence
                vision_token_sequence = (
                    self.tokens.VISION_START
                    + self.tokens.IMAGE_PAD * num_vision_tokens
                    + self.tokens.VISION_END
                )

                # Replace placeholder with vision tokens
                content = content.replace("<image>", vision_token_sequence, 1)
                image_index += 1

            processed_conversation.append({"role": message["role"], "content": content})

        # Final validation
        if image_index != len(image_paths):
            logger.warning(
                f"Mismatch between image placeholders and image paths. "
                f"Found {image_index} placeholders, but {len(image_paths)} images."
            )

        return processed_conversation, images, image_dims

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Calculate the number of vision tokens for a given image."""
        # Use the configured image processor to get the grid size
        try:
            grid_thw = self.image_processor.get_grid_thw(
                {"pixel_values": torch.randn(1, 3, image.height, image.width)}
            )
        except Exception:
            # Fallback for older image processors
            patch_size = self.image_processor.patch_size
            grid_h = image.height // patch_size
            grid_w = image.width // patch_size
            grid_thw = torch.tensor([1, grid_h, grid_w])

        # Default merge size for Qwen2.5-VL is 2
        merge_size = 2
        merge_length = merge_size**2
        num_tokens = grid_thw.prod().item() // merge_length

        return int(num_tokens)

    def _extract_and_normalize_ground_truth(
        self, sample: Dict[str, Any], image_dims: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts GT objects from the target and normalizes their bounding boxes.
        """
        target = sample.get("target", sample)
        target_objects = target.get("objects", [])

        # The last image in the list corresponds to the target.
        if not image_dims or not target_objects:
            return []

        target_image_dims = image_dims[-1]
        width, height = target_image_dims

        normalized_objects = []
        for obj in target_objects:
            box = obj.get("box")
            desc = obj.get("desc")

            if box is None or desc is None:
                continue

            # Validate box format and coordinates
            if not (isinstance(box, list) and len(box) == 4) or not (
                0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            ):
                logger.warning(
                    f"Invalid or out-of-bounds box: {box} for image size {width}x{height}. Skipping."
                )
                continue

            normalized_box = [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ]
            normalized_objects.append({"box": normalized_box, "desc": desc})

        return normalized_objects

    def _tokenize_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize conversation and create labels with proper masking.
        """
        # Use tokenizer's chat template
        formatted_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Debug: Log the formatted text before adding endoftext
        logger.debug(
            f"📄 Formatted text before endoftext: {repr(formatted_text[-100:])}"
        )

        # Check if endoftext is already present
        if not formatted_text.endswith(self.tokens.ENDOFTEXT):
            # Add end of text token only if not already present
            formatted_text += self.tokens.ENDOFTEXT
            logger.debug(f"✅ Added ENDOFTEXT token")
        else:
            logger.debug(f"✅ ENDOFTEXT token already present")

        # Debug: Log the formatted text after adding endoftext
        logger.debug(
            f"📄 Formatted text after endoftext: {repr(formatted_text[-100:])}"
        )
        logger.debug(f"🔍 ENDOFTEXT token: {repr(self.tokens.ENDOFTEXT)}")

        # Tokenize
        tokens = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_ids = tokens["input_ids"].squeeze()

        # Log sequence length for debugging
        logger.debug(f"📏 SEQUENCE LENGTH INFO:")
        logger.debug(f"   Formatted text length: {len(formatted_text)} chars")
        logger.debug(f"   Tokenized sequence length: {len(input_ids)} tokens")
        logger.debug(f"   Model max length: {config.max_total_length}")
        if len(input_ids) >= config.max_total_length:
            logger.warning(
                f"⚠️ Sequence was truncated from {len(formatted_text)} chars to {len(input_ids)} tokens"
            )

        # Debug: Check if endoftext token is in the tokenized sequence
        endoftext_id = self.tokens.get_token_id(self.tokens.ENDOFTEXT)
        if endoftext_id:
            endoftext_count = (input_ids == endoftext_id).sum().item()
            logger.debug(
                f"🔍 ENDOFTEXT token ID {endoftext_id} found {endoftext_count} times in input_ids"
            )

            # Also check what the tokenizer thinks the endoftext token should be
            # EXPLICIT: Get eos_token_id from tokenizer - no defaults
        if hasattr(self.tokenizer, "eos_token_id"):
            tokenizer_endoftext_id = self.tokenizer.eos_token_id
        else:
            tokenizer_endoftext_id = None
            logger.debug(f"🔍 Tokenizer eos_token_id: {tokenizer_endoftext_id}")

            # Check if the tokenizer has the special token
            if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                actual_endoftext_id = self.tokenizer.convert_tokens_to_ids(
                    self.tokens.ENDOFTEXT
                )
                logger.debug(
                    f"🔍 Tokenizer convert_tokens_to_ids for '{self.tokens.ENDOFTEXT}': {actual_endoftext_id}"
                )

        # Also check by decoding the last few tokens
        last_tokens = input_ids[-5:] if len(input_ids) >= 5 else input_ids
        decoded_last = self.tokenizer.decode(last_tokens, skip_special_tokens=False)
        logger.debug(f"🔍 Last 5 tokens decoded: {repr(decoded_last)}")

        # Check the full token sequence for debugging
        all_token_ids = input_ids.tolist()
        logger.debug(f"🔍 All token IDs (last 10): {all_token_ids[-10:]}")

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Mask non-assistant tokens
        labels = self._mask_non_assistant_tokens(labels, conversation, formatted_text)

        return input_ids, labels

    def _mask_non_assistant_tokens(
        self,
        labels: torch.Tensor,
        conversation: List[Dict[str, str]],
        formatted_text: str,
    ) -> torch.Tensor:
        """
        Mask tokens that should not contribute to loss.
        Only the final assistant response should be trained on.
        """
        # Find the last assistant message (target response)
        last_assistant_content = None
        for message in reversed(conversation):
            if message["role"] == "assistant":
                last_assistant_content = message["content"]
                break

        if last_assistant_content is None:
            return labels

        # Find where the last assistant response starts in the formatted text
        last_assistant_start = formatted_text.rfind(last_assistant_content)
        if last_assistant_start == -1:
            return labels

        # Tokenize up to the last assistant response
        prefix_text = formatted_text[:last_assistant_start]
        prefix_tokens = self.tokenizer(prefix_text, return_tensors="pt", padding=False)
        prefix_length = len(prefix_tokens["input_ids"].squeeze())

        # Mask everything before the last assistant response
        labels[:prefix_length] = -100

        return labels

    def _process_images_for_model(
        self, images: List[Image.Image]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process images for model input using official approach with data_conversion pixel settings."""
        if not images:
            return None, None

        # Log original image sizes for debugging
        logger.debug(f"🖼️ PROCESSING {len(images)} IMAGES:")
        for i, img in enumerate(images):
            logger.debug(f"   Image {i}: {img.size} (W*H), mode={img.mode}")

        # Use processor directly like official QwenVL implementation
        # The processor is already configured with data_conversion/vision_process.py values
        processed = self.image_processor.preprocess(images, return_tensors="pt")

        pixel_values = processed["pixel_values"]
        image_grid_thw = processed.get("image_grid_thw")

        # CRITICAL: Log the vision token analysis
        logger.debug(f"🚨 VISION TOKEN ANALYSIS:")
        logger.debug(f"   Input images: {len(images)} images")
        logger.debug(f"   Original sizes: {[img.size for img in images]}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(f"   Vision tokens generated (pre-merge): {pixel_values.shape[0]}")

        if image_grid_thw is not None:
            logger.debug(f"   image_grid_thw shape: {image_grid_thw.shape}")
            logger.debug(f"   image_grid_thw values: {image_grid_thw.tolist()}")

            # Calculate both pre-merge and post-merge token counts for clarity
            # EXPLICIT: Get merge_size from image processor - no defaults
            if hasattr(self.image_processor, "merge_size"):
                merge_size = self.image_processor.merge_size
            else:
                # Use Qwen2.5-VL default
                merge_size = 2
            merge_length = merge_size**2

            total_pre_merge = 0
            total_post_merge = 0

            for i, grid in enumerate(image_grid_thw):
                t, h, w = grid.tolist()
                pre_merge_tokens = t * h * w
                post_merge_tokens = pre_merge_tokens // merge_length
                total_pre_merge += pre_merge_tokens
                total_post_merge += post_merge_tokens

                logger.debug(
                    f"   Image {i}: grid=({t},{h},{w}) → {pre_merge_tokens} pre-merge → {post_merge_tokens} final tokens"
                )

            logger.debug(
                f"   TOTAL: {total_pre_merge} pre-merge → {total_post_merge} final tokens (merge_size={merge_size}²={merge_length})"
            )
            logger.debug(
                f"   Vision token efficiency: {total_post_merge}/{total_pre_merge} = {total_post_merge / total_pre_merge:.1%}"
            )
        else:
            logger.debug(f"   No image_grid_thw available")

        # CRITICAL: Ensure bf16 precision for pixel_values
        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)
            logger.debug(
                f"🔧 Converted pixel_values from {processed['pixel_values'].dtype} to bf16: {pixel_values.dtype}"
            )
        else:
            logger.debug(f"✅ pixel_values already in bf16: {pixel_values.dtype}")

        # Debug: Log the shapes to understand the processing
        logger.debug(f"🖼️ OFFICIAL IMAGE PROCESSING:")
        logger.debug(f"   Number of images: {len(images)}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(
            f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        return pixel_values, image_grid_thw

    def prepare_inputs_for_inference(
        self, images: List[Image.Image], text: str, is_first_step: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for inference following the official Qwen2.5-VL pattern.

        Args:
            images: List of PIL images
            text: Input text prompt
            is_first_step: Whether this is the first generation step (prefill)

        Returns:
            Dict containing properly formatted inputs for model.generate()
        """
        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=False)

        # Prepare base inputs
        model_inputs = {
            "input_ids": text_inputs["input_ids"],
        }

        # Add attention_mask only if it exists
        if (
            "attention_mask" in text_inputs
            and text_inputs["attention_mask"] is not None
        ):
            model_inputs["attention_mask"] = text_inputs["attention_mask"]

        # Handle vision inputs based on generation step
        if is_first_step and images:
            # First step: process images normally
            pixel_values, image_grid_thw = self._process_images_for_model(images)

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values

            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

            logger.debug(f"🔥 FIRST STEP: Added vision inputs")
            logger.debug(
                f"   pixel_values shape: {pixel_values.shape if pixel_values is not None else None}"
            )
            logger.debug(
                f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
            )
        else:
            # Subsequent steps: don't add vision inputs at all
            # The model's prepare_inputs_for_generation will handle this
            logger.debug(f"🔥 SUBSEQUENT STEP: No vision inputs added")

        # Filter to only include valid generation parameters
        valid_generation_params = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }

        filtered_inputs = {
            key: value
            for key, value in model_inputs.items()
            if key in valid_generation_params and value is not None
        }

        logger.debug(f"🔧 FILTERED INFERENCE INPUTS: {list(filtered_inputs.keys())}")

        return filtered_inputs


def create_chat_processor(
    tokenizer,
    image_processor,
    data_root: str = "./",
    model_max_length: int = 8192,
    use_candidates: bool = False,
    candidates_file: Optional[str] = None,
) -> ChatProcessor:
    """
    Factory function to create ChatProcessor.

    Args:
        tokenizer: Qwen2.5-VL tokenizer
        image_processor: Qwen2.5-VL image processor
        data_root: Root directory for image paths
        model_max_length: Maximum sequence length
        use_candidates: Whether to use candidate phrases
        candidates_file: Path to candidate phrases JSON

    Returns:
        Configured ChatProcessor instance
    """
    return ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        model_max_length=model_max_length,
        use_candidates=use_candidates,
        candidates_file=candidates_file,
    )
