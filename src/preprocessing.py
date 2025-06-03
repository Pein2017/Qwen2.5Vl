"""
Unified preprocessing pipeline for both training and evaluation.

This module provides a single, optimized preprocessing pipeline that handles:
- Multi-image and single-image inputs
- Vision token replacement
- Tensor formatting
- Position ID calculation
- Chat template application

Used by both training (BBUDataset) and evaluation (inference scripts).

Key Improvements:
- Better error handling and validation
- Performance optimizations
- Cleaner code structure
- Enhanced logging and debugging
"""

import copy
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from .rope2d import get_rope_index_25
from .utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class ImageProcessor:
    """Optimized image processing with better error handling."""

    def __init__(self, image_processor, data_root: str = "./"):
        self.image_processor = image_processor
        self.data_root = Path(data_root)

    def process_images(
        self, image_paths: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Process images with enhanced error handling."""
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if not image_paths:
            raise ValidationError("No image paths provided")

        pixel_values_list = []
        image_grid_thw_list = []
        grid_thw_merged_list = []

        for i, image_path in enumerate(image_paths):
            try:
                pixel_values, grid_thw, grid_thw_merged = self._process_single_image(
                    image_path
                )
                pixel_values_list.append(pixel_values)
                image_grid_thw_list.append(grid_thw)
                grid_thw_merged_list.append(grid_thw_merged)
            except Exception as e:
                raise ValidationError(
                    f"Failed to process image {i} ({image_path}): {e}"
                )

        # Concatenate all results
        try:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.stack(image_grid_thw_list, dim=0)
        except Exception as e:
            raise ValidationError(f"Failed to concatenate image tensors: {e}")

        return pixel_values, image_grid_thw, grid_thw_merged_list

    def _process_single_image(
        self, image_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Process a single image with validation."""
        # Resolve path
        if not os.path.isabs(image_path):
            full_path = self.data_root / image_path
        else:
            full_path = Path(image_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Load and validate image
        try:
            image = Image.open(full_path).convert("RGB")
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Image has zero dimensions")
        except Exception as e:
            raise ValueError(f"Failed to load image {full_path}: {e}")

        # Process with image processor
        try:
            processor = copy.deepcopy(self.image_processor)
            visual_processed = processor.preprocess(image, return_tensors="pt")

            pixel_values = visual_processed["pixel_values"]
            if isinstance(pixel_values, list):
                pixel_values = pixel_values[0]

            grid_thw = visual_processed["image_grid_thw"][0]

            # Calculate grid_thw_merged exactly like official implementation
            grid_thw_merged = grid_thw.prod() // self.image_processor.merge_size**2

            return pixel_values, grid_thw, grid_thw_merged.item()

        except Exception as e:
            raise ValueError(f"Failed to process image {full_path}: {e}")


class VisionTokenReplacer:
    """Handles vision token replacement with validation."""

    @staticmethod
    def replace_vision_tokens(content: str, grid_thw_merged_list: List[int]) -> str:
        """Replace <image> tokens with vision tokens using training logic."""
        if DEFAULT_IMAGE_TOKEN not in content:
            return content

        # Validate token count alignment
        image_token_count = content.count(DEFAULT_IMAGE_TOKEN)
        if image_token_count != len(grid_thw_merged_list):
            raise ValidationError(
                f"Mismatch: Found {image_token_count} <image> tokens but {len(grid_thw_merged_list)} images. "
                f"Each <image> token must correspond to exactly one image."
            )

        # Replace tokens one by one to maintain order
        result = content
        for i, grid_thw_merged in enumerate(grid_thw_merged_list):
            if DEFAULT_IMAGE_TOKEN not in result:
                break

            # Create vision token sequence for this specific image
            vision_tokens = (
                "<|vision_start|>"
                + "<|image_pad|>" * grid_thw_merged
                + "<|vision_end|>"
            )

            # Replace one token at a time to maintain order
            result = result.replace(DEFAULT_IMAGE_TOKEN, vision_tokens, 1)

        # Verify all tokens were replaced
        if DEFAULT_IMAGE_TOKEN in result:
            remaining_count = result.count(DEFAULT_IMAGE_TOKEN)
            raise ValidationError(
                f"Failed to replace all <image> tokens. {remaining_count} tokens remaining."
            )

        return result


class ConversationFormatter:
    """Handles conversation formatting and validation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_response: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[Dict[str, str]]:
        """Create conversation in the exact format used by training."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if assistant_response is not None:
            conversation.append({"role": "assistant", "content": assistant_response})

        return conversation

    def apply_chat_template(
        self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> torch.Tensor:
        """Apply chat template with error handling."""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
            return input_ids
        except Exception as e:
            raise ValidationError(f"Failed to apply chat template: {e}")


def preprocess_qwen_2_visual(
    sources,
    tokenizer,
    grid_thw_image: List = [],
) -> Dict:
    """
    Enhanced preprocessing function for Qwen2.5-VL with improved error handling.

    Key enhancement: Only the LAST assistant response in each conversation is trained on.
    All previous assistant responses (examples) are masked with IGNORE_INDEX.
    """
    if not sources:
        raise ValidationError("No sources provided for preprocessing")

    roles = {"human": "user", "gpt": "assistant"}

    # Create a copy of tokenizer to avoid modifying the original
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for source_idx, source in enumerate(sources):
        try:
            # Normalize source format
            if (
                source
                and roles.get(source[0].get("from", ""), source[0].get("from", ""))
                != "user"
            ):
                source = source[1:]

            input_id, target = [], []

            # Find all assistant responses to determine which is the last one
            assistant_indices = []
            for idx, conv in enumerate(source):
                role = conv.get("role", conv.get("from", ""))
                role = roles.get(role, role)
                if role == "assistant":
                    assistant_indices.append(idx)

            # Validate telecom inspection format
            _validate_telecom_format(source, assistant_indices)

            # Process each conversation turn
            for conv_idx, conv in enumerate(source):
                role = conv.get("role", conv.get("from", ""))
                content = conv.get("content", conv.get("value", ""))
                role = roles.get(role, role)

                # Handle image tokens in user messages
                if role == "user" and "<image>" in content:
                    content = _process_image_tokens(
                        content, grid_thw_image, visual_replicate_index_image
                    )
                    visual_replicate_index_image += content.count("<|vision_start|>")

                # Tokenize the conversation turn
                try:
                    conv_formatted = [{"role": role, "content": content}]
                    encode_id = tokenizer.apply_chat_template(conv_formatted)
                    input_id.extend(encode_id)
                except Exception as e:
                    raise ValidationError(
                        f"Failed to tokenize conversation turn {conv_idx}: {e}"
                    )

                # Determine whether to train on this turn
                if role in ["user", "system"]:
                    # Never train on user or system messages
                    target.extend([IGNORE_INDEX] * len(encode_id))
                elif role == "assistant":
                    # Only train on the LAST assistant response
                    is_last_assistant = (
                        (conv_idx == assistant_indices[-1])
                        if assistant_indices
                        else True
                    )

                    if is_last_assistant:
                        # Train on the last assistant response (skip the first 3 tokens)
                        target_mask = encode_id.copy()
                        target_mask[:3] = [IGNORE_INDEX] * 3
                        target.extend(target_mask)
                    else:
                        # Don't train on example assistant responses
                        target.extend([IGNORE_INDEX] * len(encode_id))

            # Validate sequence lengths
            if len(input_id) != len(target):
                raise ValidationError(
                    f"Length mismatch: input_ids={len(input_id)}, targets={len(target)}"
                )

            input_ids.append(input_id)
            targets.append(target)

        except Exception as e:
            raise ValidationError(f"Failed to process source {source_idx}: {e}")

    try:
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
    except Exception as e:
        raise ValidationError(f"Failed to convert to tensors: {e}")

    return dict(input_ids=input_ids, labels=targets)


def _process_image_tokens(content: str, grid_thw_image: List, start_index: int) -> str:
    """Process image tokens in content with validation."""
    if "<image>" not in content:
        return content

    parts = content.split("<image>")
    new_parts = []

    for i in range(len(parts) - 1):
        new_parts.append(parts[i])

        image_index = start_index + i
        if image_index >= len(grid_thw_image):
            raise ValidationError(
                f"Image index {image_index} out of range (max: {len(grid_thw_image) - 1})"
            )

        replacement = (
            "<|vision_start|>"
            + "<|image_pad|>" * grid_thw_image[image_index]
            + "<|vision_end|>"
        )
        new_parts.append(replacement)

    new_parts.append(parts[-1])
    return "".join(new_parts)


def _validate_telecom_format(source: List[Dict], assistant_indices: List[int]) -> None:
    """Validate conversation format with improved error handling."""
    # Allow disabling validation via environment variable
    skip_validation = os.getenv("BBU_SKIP_VALIDATION", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    early_training_mode = os.getenv("BBU_EARLY_TRAINING", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    if skip_validation:
        return

    # Check that we have at least one assistant response
    if not assistant_indices:
        if early_training_mode:
            print(
                "⚠️ Warning: No assistant responses found in conversation - continuing anyway"
            )
            return
        else:
            raise ValidationError("No assistant responses found in conversation")

    malformed_responses = 0
    total_responses = len(assistant_indices)

    # Validate assistant responses
    for idx in assistant_indices:
        conv = source[idx]
        content = conv.get("content", conv.get("value", ""))

        try:
            valid_format = _validate_unquoted_format(content)
            if not valid_format:
                malformed_responses += 1
                if early_training_mode:
                    print(
                        "⚠️ Warning: Assistant response doesn't match expected format - continuing"
                    )
                    print(f"   Content preview: {repr(content[:200])}")
        except Exception as e:
            malformed_responses += 1
            if early_training_mode:
                print(
                    f"⚠️ Warning: Error validating assistant response: {e} - continuing"
                )

    # Only fail if ALL responses are malformed
    if malformed_responses == total_responses and total_responses > 0:
        if early_training_mode:
            print(
                f"❌ Critical: All {total_responses} assistant responses are malformed"
            )
            print("💡 Suggestion: Check your data conversion and format")
            print("🔄 Continuing anyway for training robustness...")
        else:
            raise ValidationError(
                f"All {total_responses} assistant responses are malformed"
            )


def _validate_unquoted_format(content: str) -> bool:
    """Validate that content matches the expected unquoted format."""
    if not content or not content.strip():
        return False

    content_clean = content.strip()

    # Check for array structure
    if not (content_clean.startswith("[") and content_clean.endswith("]")):
        return False

    # Pattern for unquoted format
    unquoted_pattern = (
        r'\{\s*bbox\s*:\s*\[([^\]]+)\]\s*,\s*desc\s*:\s*[\'"]([^\'"]*)[\'"]?\s*\}'
    )

    # Check if we can find at least one valid object
    matches = re.findall(unquoted_pattern, content_clean, re.IGNORECASE | re.DOTALL)

    if matches:
        # Validate that at least one match has valid coordinates
        for bbox_str, desc_str in matches:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4 and desc_str.strip():
                    return True
            except (ValueError, IndexError):
                continue

    # Try alternative patterns for edge cases
    alt_patterns = [
        r"\{\s*bbox\s*:\s*\[([^\]]+)\]\s*,\s*desc\s*:\s*([^,\}]+)\s*\}",
        r'\{\s*"?bbox(?:_2d)?"?\s*:\s*\[([^\]]+)\]\s*,\s*"?(?:desc|description)"?\s*:\s*[\'"]([^\'"]*)[\'"]?\s*\}',
    ]

    for pattern in alt_patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE)
        if matches:
            for bbox_str, desc_str in matches:
                try:
                    coords = [float(x.strip()) for x in bbox_str.split(",")]
                    if len(coords) == 4 and desc_str.strip():
                        return True
                except (ValueError, IndexError):
                    continue

    return False


class Preprocessor:
    """
    Unified preprocessor for both training and evaluation with enhanced features.

    Key improvements:
    - Better error handling and validation
    - Performance optimizations
    - Enhanced logging
    - Cleaner code structure
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        data_root: str = "./",
        model_max_length: int = 8192,
    ):
        """Initialize preprocessor with validation."""
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")
        if image_processor is None:
            raise ValueError("Image processor cannot be None")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_root = data_root
        self.model_max_length = model_max_length

        # Initialize components
        self.img_processor = ImageProcessor(image_processor, data_root)
        self.token_replacer = VisionTokenReplacer()
        self.conv_formatter = ConversationFormatter(tokenizer)

        # Configure tokenizer and image processor
        self._configure_components()

    def _configure_components(self):
        """Configure tokenizer and image processor with exact settings."""
        # Set exact chat template from official implementation
        self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        # Configure image processor exactly like official
        self.image_processor.max_pixels = getattr(
            self.image_processor, "max_pixels", 1003520
        )
        self.image_processor.min_pixels = getattr(
            self.image_processor, "min_pixels", 784
        )
        self.image_processor.size = {
            "longest_edge": self.image_processor.max_pixels,
            "shortest_edge": self.image_processor.min_pixels,
        }

    def process_images(
        self, image_paths: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Process images with enhanced error handling."""
        return self.img_processor.process_images(image_paths)

    def replace_vision_tokens(
        self, content: str, grid_thw_merged_list: List[int]
    ) -> str:
        """Replace vision tokens with validation."""
        return self.token_replacer.replace_vision_tokens(content, grid_thw_merged_list)

    def create_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_response: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[Dict[str, str]]:
        """Create conversation with validation."""
        return self.conv_formatter.create_conversation(
            system_prompt, user_prompt, assistant_response, add_generation_prompt
        )

    def apply_chat_template(
        self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> torch.Tensor:
        """Apply chat template with error handling."""
        return self.conv_formatter.apply_chat_template(
            conversation, add_generation_prompt
        )

    def calculate_position_ids(
        self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Calculate position IDs with error handling."""
        try:
            position_ids, _ = get_rope_index_25(
                spatial_merge_size=self.image_processor.merge_size,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
            )
            return position_ids
        except Exception as e:
            raise ValidationError(f"Failed to calculate position IDs: {e}")

    def process_sample_for_training(
        self, sample: Dict, sample_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Process a sample for training with comprehensive error handling."""
        try:
            # Extract and validate image paths
            if "images" in sample:
                image_paths = sample["images"]
            elif "image" in sample:
                image_paths = [sample["image"]]
            else:
                raise ValidationError("No image paths found in sample")

            if not image_paths:
                raise ValidationError("Empty image paths list")

            # Process images
            pixel_values, image_grid_thw, grid_thw_merged_list = self.process_images(
                image_paths
            )

            # Extract and validate conversations
            conversations = sample.get("conversations")
            if not conversations:
                raise ValidationError("No conversations found in sample")

            # Process conversations for training
            processed_data = self._process_conversations_for_training(
                conversations, grid_thw_merged_list, len(image_paths)
            )

            # Calculate position IDs
            position_ids = self.calculate_position_ids(
                processed_data["input_ids"].unsqueeze(0), image_grid_thw
            )

            return {
                "input_ids": processed_data["input_ids"],
                "labels": processed_data["labels"],
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "position_ids": position_ids,
                "attention_mask": [len(processed_data["input_ids"])],
                "sample_idx": sample_idx,
            }

        except Exception as e:
            raise ValidationError(
                f"Failed to process training sample {sample_idx}: {e}"
            )

    def _process_conversations_for_training(
        self,
        conversations: List[Dict],
        grid_thw_merged_list: List[int],
        num_images: int,
    ) -> Dict[str, torch.Tensor]:
        """Process conversations for training with vision token validation."""
        input_ids_list = []
        labels_list = []

        # Start with system message
        system_conv = [{"role": "system", "content": "You are a helpful assistant."}]
        system_ids = self.tokenizer.apply_chat_template(
            system_conv, add_generation_prompt=False
        )
        input_ids_list.extend(system_ids)
        labels_list.extend([IGNORE_INDEX] * len(system_ids))

        # Track vision token processing
        vision_tokens_processed = 0

        for conv in conversations:
            role = conv.get("role", conv.get("from", ""))
            content = conv.get("content", conv.get("value", ""))

            # Normalize role names
            role = {"human": "user", "gpt": "assistant"}.get(role, role)

            # Replace vision tokens in user messages with validation
            if role == "user" and DEFAULT_IMAGE_TOKEN in content:
                image_tokens_in_turn = content.count(DEFAULT_IMAGE_TOKEN)

                # Validate we have enough images for this turn
                if vision_tokens_processed + image_tokens_in_turn > num_images:
                    raise ValidationError(
                        f"Too many <image> tokens: found {image_tokens_in_turn} in turn, "
                        f"but only {num_images - vision_tokens_processed} images remaining"
                    )

                # Get the grid_thw_merged values for this turn's images
                turn_grid_thw = grid_thw_merged_list[
                    vision_tokens_processed : vision_tokens_processed
                    + image_tokens_in_turn
                ]

                # Replace vision tokens for this turn
                content = self.replace_vision_tokens(content, turn_grid_thw)
                vision_tokens_processed += image_tokens_in_turn

            # Tokenize this conversation turn
            conv_formatted = [{"role": role, "content": content}]
            conv_ids = self.tokenizer.apply_chat_template(
                conv_formatted, add_generation_prompt=False
            )

            input_ids_list.extend(conv_ids)

            # Create labels - ignore input tokens, train on assistant response
            if role in ["user", "system"]:
                labels_list.extend([IGNORE_INDEX] * len(conv_ids))
            else:
                # Train on assistant response, but ignore first few tokens
                target_mask = conv_ids.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3  # Ignore first 3 tokens
                labels_list.extend(target_mask)

        # Convert to tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}

    def process_sample_for_inference(
        self, image_paths: Union[str, List[str]], system_prompt: str, user_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Process a sample for inference with comprehensive error handling."""
        try:
            # Process images
            pixel_values, image_grid_thw, grid_thw_merged_list = self.process_images(
                image_paths
            )

            # Replace vision tokens in user prompt
            user_prompt_processed = self.replace_vision_tokens(
                user_prompt, grid_thw_merged_list
            )

            # Create conversation
            conversation = self.create_conversation(
                system_prompt=system_prompt,
                user_prompt=user_prompt_processed,
                add_generation_prompt=True,
            )

            # Apply chat template
            input_ids = self.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            # Create attention mask
            attention_mask = torch.ones_like(input_ids)

            # Calculate position IDs
            position_ids = self.calculate_position_ids(input_ids, image_grid_thw)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "position_ids": position_ids,
            }

        except Exception as e:
            raise ValidationError(f"Failed to process inference sample: {e}")


def create_preprocessor(
    tokenizer, image_processor, data_root: str = "./", model_max_length: int = 8192
) -> Preprocessor:
    """Factory function to create a unified preprocessor with validation."""
    try:
        return Preprocessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_root=data_root,
            model_max_length=model_max_length,
        )
    except Exception as e:
        raise ValidationError(f"Failed to create preprocessor: {e}")
