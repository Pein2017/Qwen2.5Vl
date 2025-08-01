"""
Optimized data handling for Qwen2.5-VL training with unified preprocessing.

This module provides:
- Dataset: Dataset using unified preprocessing for clean data processing
- Utilities for ground truth extraction from conversation format

Key Features:
- Unified preprocessing pipeline (no file dependencies)
- Multi-image conversation support
- Compatible with flash attention and DeepSpeed
- Optimized for large sequences and multi-GPU training
"""

import json

# Create logger specifically for data module
import logging
import random
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


if TYPE_CHECKING:
    from src_new.config.config import Config
    from src_new.data.teacher_pool import TeacherPoolManager


def get_data_logger() -> logging.Logger:
    """Get logger for data module."""
    from src_new.config.config import _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL

    logger = logging.getLogger("data")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_GLOBAL_LOG_LEVEL)
        _CONFIGURED_LOGGERS.add("data")
    return logger


logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries.

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class Dataset(TorchDataset):
    """Unified dataset with end-to-end data processing.

    Combines conversation building, tokenization, and dataset management
    in a single unified processor, eliminating intermediate schemas.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Optional[Any],
        teacher_pool_manager: Optional["TeacherPoolManager"],
        config: "Config",
    ):
        """
        Initialize unified dataset with integrated data processing.

        Args:
            data_path: Path to JSONL file (preprocessed format)
            tokenizer: Tokenizer for text processing
            image_processor: Image processor for vision inputs
            teacher_pool_manager: Manager for teacher examples
            config: Configuration object with all settings

        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If the dataset contains invalid samples
        """
        # FAIL-FAST: Validate required parameters
        if not data_path:
            raise ValueError("data_path cannot be empty")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if config is None:
            raise ValueError("config cannot be None")

        # Store configuration
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.teacher_pool_manager = teacher_pool_manager
        self.config = config

        # Initialize processing components
        self._initialize_processing_components()

        # Load and validate data
        self.raw_data = self._load_data()
        self.samples = self._validate_and_filter_samples(self.raw_data)

        # Initialize teacher assignment tracking
        self.teacher_assignments = {}
        self.teacher_assignment_counts = {}

        logger.info(f"✅ Dataset initialized with {len(self.samples)} samples")

    def _initialize_processing_components(self):
        """Initialize processing components based on configuration."""
        # Data processing settings
        self.max_total_length = getattr(self.config, "max_total_length", 4096)
        self.data_root = getattr(self.config, "data_root", "")
        self.language = getattr(self.config, "language", "chinese")
        self.teacher_ratio = getattr(self.config, "teacher_ratio", 0.5)
        self.num_teacher_samples = getattr(self.config, "num_teacher_samples", 1)

        # System prompt (if needed)
        self.system_prompt = self._build_system_prompt()

        # Initialize coordinate token processing
        self.coordinate_tokens_enabled = getattr(
            self.config, "coordinate_tokens_enabled", False
        )
        if self.coordinate_tokens_enabled:
            from src_new.processing.token_processor import TokenConfig, TokenProcessor

            token_config = TokenConfig(
                coordinate_tokens_enabled=True,
                max_coord_value=self.config.max_coord_value,
            )
            self.token_processor = TokenProcessor(token_config)
            logger.info(
                "🎯 Dataset initialized with coordinate token processing enabled"
            )
        else:
            self.token_processor = None

    def _build_system_prompt(self) -> str:
        """Build system prompt based on configuration."""
        if self.language == "chinese":
            return "你是一个有用的AI助手，你叫Qwen。"
        else:
            return "You are a helpful AI assistant named Qwen."

    def _validate_and_filter_samples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter samples from raw data.

        Args:
            raw_data: Raw data loaded from JSONL file

        Returns:
            List of valid samples

        Raises:
            ValueError: If no valid samples remain after filtering
        """
        valid_samples = []
        invalid_count = 0
        empty_count = 0

        for idx, sample in enumerate(raw_data):
            # Skip samples with no content
            if not sample:
                empty_count += 1
                logger.debug(f"Sample {idx}: empty sample")
                continue

            # Validate required fields
            if "image" not in sample and "images" not in sample:
                invalid_count += 1
                logger.debug(
                    f"Sample {idx}: missing image/images field. Keys: {list(sample.keys())}"
                )
                continue

            # Additional validation - check if images field is valid
            if "images" in sample:
                if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
                    invalid_count += 1
                    logger.debug(
                        f"Sample {idx}: invalid images field: {sample['images']}"
                    )
                    continue

            # Add sample index for tracking
            sample["idx"] = idx
            valid_samples.append(sample)
            logger.debug(f"Sample {idx}: valid sample with keys: {list(sample.keys())}")

        # Log validation results
        if invalid_count > 0:
            logger.warning(f"⚠️ Skipped {invalid_count} invalid samples")
        if empty_count > 0:
            logger.warning(f"⚠️ Skipped {empty_count} empty samples")

        # FAIL-FAST: Ensure we have valid samples
        if not valid_samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        return valid_samples

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed sample by index."""
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Process and return a single sample.

        This is the main entry point for sample processing, handling both
        student and teacher samples based on configuration.

        Args:
            idx: Sample index

        Returns:
            Processed sample with tensors ready for model input
        """
        # Get raw sample
        raw_sample = self.samples[idx]

        # Create structured sample with teacher assignments
        structured_sample = self._create_structured_sample(raw_sample, idx)

        # Process the sample through the unified pipeline
        processed_sample = self._process_sample_unified(structured_sample, idx)

        return processed_sample

    def _process_sample_unified(
        self, structured_sample: Dict[str, Any], idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sample through the unified pipeline.

        This is the core processing function that handles:
        1. Loading and processing images
        2. Building conversation format
        3. Tokenizing and preparing model inputs

        Args:
            structured_sample: Structured sample data with potential teacher assignments
            idx: Sample index for logging

        Returns:
            Processed sample with tensors ready for model input
        """
        # Extract teacher samples if present
        teacher_samples = structured_sample.get("teacher_samples", [])
        has_teachers = len(teacher_samples) > 0

        # Process main sample based on teacher availability
        if has_teachers:
            # Create teacher-student conversation with multiple assistant messages
            conversation, images = self._create_conversation_with_teacher(
                structured_sample
            )
            logger.debug(f"✅ Created teacher-student conversation for sample {idx}")
        else:
            # Use student sample for processing (student-only)
            conversation, images = self._create_simple_conversation(structured_sample)
            logger.debug(f"✅ Created student-only conversation for sample {idx}")

        # Process images if present (including both teacher and student images)
        pixel_values, image_grid_thw = self._process_images(structured_sample)

        # Tokenize conversation
        tokenized_inputs = self._tokenize_simple_conversation(
            conversation, structured_sample, images
        )

        # Add image data to inputs
        if pixel_values is not None:
            tokenized_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            tokenized_inputs["image_grid_thw"] = image_grid_thw

        return tokenized_inputs

    def _create_simple_conversation(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[Any]]:
        """
        Create a simple conversation format from a sample.

        Args:
            sample: Sample data

        Returns:
            Tuple of (conversation_text, list_of_images)
        """
        # Load images
        image_paths = []
        if "image" in sample:
            image_paths = [sample["image"]]
        elif "images" in sample:
            image_paths = sample["images"]

        images = self._load_images_from_paths(image_paths)

        # Build conversation components
        conversation_parts = []

        # Add system prompt if needed
        if self.system_prompt:
            conversation_parts.append(
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
            )

        # Add user message with image placeholders
        user_message = sample.get("prompt", "")
        if not user_message:
            user_message = "Describe this image in detail."

        # Create user message with image tokens
        user_message_with_images = self._expand_vision_tokens(user_message, images)
        conversation_parts.append(
            f"<|im_start|>user\n{user_message_with_images}<|im_end|>"
        )

        # Add assistant response
        if "objects" in sample:
            # Format object detection response
            objects = sample["objects"]
            assistant_response = self._format_objects_response(objects)
        else:
            # Use caption as response
            assistant_response = sample.get("caption", "")
            if not assistant_response:
                assistant_response = sample.get("response", "")

        conversation_parts.append(
            f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
        )

        # Combine conversation parts
        conversation = "\n".join(conversation_parts)

        return conversation, images

    def _format_objects_response(self, objects: List[Dict[str, Any]]) -> str:
        """
        Format object detection response in the required format.

        Args:
            objects: List of detected objects with coordinates

        Returns:
            Formatted response string with object coordinates
        """
        if not objects:
            return "I don't see any objects in this image."

        response_parts = []
        for obj in objects:
            obj_name = obj.get("category", "object")

            # Handle different geometry types
            if "bbox_2d" in obj:
                coordinates = obj["bbox_2d"]
                geometry_type = "bbox_2d"
            elif "x1" in obj and "y1" in obj and "x2" in obj and "y2" in obj:
                # Legacy format - convert to bbox_2d
                coordinates = [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
                geometry_type = "bbox_2d"
            elif "quad" in obj:
                coordinates = obj["quad"]
                geometry_type = "quad"
            elif "line" in obj:
                coordinates = obj["line"]
                geometry_type = "line"
            else:
                # Raise error for unsupported geometry types
                available_keys = [
                    k for k in obj.keys() if k not in ["desc", "category"]
                ]
                raise ValueError(
                    f"Object contains unsupported geometry type. "
                    f"Expected one of: bbox_2d, quad, line. "
                    f"Found geometry keys: {available_keys}. "
                    f"Full object: {obj}"
                )

            # Format coordinates based on coordinate token mode
            if self.coordinate_tokens_enabled and self.token_processor:
                # Use coordinate tokens
                coord_tokens = self.token_processor.coordinates_to_tokens(coordinates)
                coord_str = " ".join(coord_tokens)

                # Use appropriate geometry wrapper tokens
                if geometry_type == "bbox_2d":
                    response_parts.append(
                        f"<|box_start|>{obj_name} {coord_str}<|box_end|>"
                    )
                elif geometry_type == "square":
                    response_parts.append(
                        f"<|square_start|>{obj_name} {coord_str}<|square_end|>"
                    )
                elif geometry_type == "line":
                    response_parts.append(
                        f"<|line_start|>{obj_name} {coord_str}<|line_end|>"
                    )
            else:
                # Standard mode - use integer coordinates
                coord_str = " ".join(map(str, coordinates))
                response_parts.append(
                    f"<object>{obj_name}</object> <coord>{coord_str}</coord>"
                )

        return "\n".join(response_parts)

    def _load_images_from_paths(self, image_paths: List[str]) -> List[Any]:
        """
        Load images from paths.

        Args:
            image_paths: List of image paths

        Returns:
            List of loaded image objects
        """
        images = []
        for path in image_paths:
            # Handle relative paths with data_root
            if self.data_root and not Path(path).is_absolute():
                full_path = Path(self.data_root) / path
            else:
                full_path = Path(path)

            # Load image using PIL with fail-fast validation (matching src/ implementation)
            from PIL import Image

            # Fail-fast: raise explicit error if the image cannot be loaded
            if not full_path.exists():
                raise FileNotFoundError(
                    f"❌ CRITICAL: Image file not found: {full_path}. "
                    f"All training images must be accessible for vision-language training."
                )

            # No try/catch - let errors propagate immediately (fail-fast)
            image = Image.open(str(full_path)).convert("RGB")
            images.append(image)

        return images

    def _create_conversation_with_teacher(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[Any]]:
        """
        Create a teacher-student conversation format from a sample with teachers.

        Args:
            sample: Sample data with teacher_samples

        Returns:
            Tuple of (conversation_text, images)
        """
        # Get teacher samples
        teacher_samples = sample.get("teacher_samples", [])
        if not teacher_samples:
            # Fallback to simple conversation if no teachers
            return self._create_simple_conversation(sample)

        # Start with system message
        conversation_parts = [
            "<|im_start|>system",
            "你是一个有用的AI助手，你叫Qwen。<|im_end|>",
        ]

        all_images = []

        # Add teacher demonstrations first
        for i, teacher_sample in enumerate(teacher_samples):
            # Load teacher images
            teacher_image_paths = teacher_sample.get("images", [])
            teacher_images = []
            if teacher_image_paths:
                teacher_images = self._load_images_from_paths(teacher_image_paths)
                all_images.extend(teacher_images)

            # Add teacher user message with image tokens
            if teacher_images:
                # Calculate image tokens for teacher images
                teacher_image_tokens = []
                for img in teacher_images:
                    num_tokens = self._calculate_image_tokens(img)
                    tokens = self._format_vision_tokens(num_tokens)
                    teacher_image_tokens.append(tokens)

                # Create user message with image tokens
                image_tokens_str = "\n".join(teacher_image_tokens)
                user_message = f"{image_tokens_str}\nDescribe this image in detail."
            else:
                user_message = "Describe this image in detail."

            conversation_parts.extend(
                [
                    "<|im_start|>user",
                    user_message + "<|im_end|>",
                ]
            )

            # Add teacher assistant response
            teacher_response = self._format_objects_response(
                teacher_sample.get("objects", [])
            )
            conversation_parts.extend(
                [
                    "<|im_start|>assistant",
                    teacher_response + "<|im_end|>",
                ]
            )

        # Load student images
        student_image_paths = sample.get("images", [])
        student_images = []
        if student_image_paths:
            student_images = self._load_images_from_paths(student_image_paths)
            all_images.extend(student_images)

        # Add student task with image tokens
        if student_images:
            # Calculate image tokens for student images
            student_image_tokens = []
            for img in student_images:
                num_tokens = self._calculate_image_tokens(img)
                tokens = self._format_vision_tokens(num_tokens)
                student_image_tokens.append(tokens)

            # Create user message with image tokens
            image_tokens_str = "\n".join(student_image_tokens)
            user_message = f"{image_tokens_str}\nDescribe this image in detail."
        else:
            user_message = "Describe this image in detail."

        conversation_parts.extend(
            [
                "<|im_start|>user",
                user_message + "<|im_end|>",
            ]
        )

        # Add student assistant response
        student_response = self._format_objects_response(sample.get("objects", []))
        conversation_parts.extend(
            [
                "<|im_start|>assistant",
                student_response + "<|im_end|>",
            ]
        )

        # Join conversation
        conversation = "\n".join(conversation_parts)

        return conversation, all_images

    def _expand_vision_tokens(self, conversation: str, images: List[Any]) -> str:
        """
        Expand image placeholders with vision tokens.

        Args:
            conversation: Conversation text with image placeholders
            images: List of loaded images

        Returns:
            Conversation with expanded vision tokens
        """
        # Replace image placeholders with vision tokens
        if not images:
            return conversation

        # For single image case (most common)
        if len(images) == 1:
            image_tokens = self._format_vision_tokens(
                self._calculate_image_tokens(images[0])
            )
            return f"{image_tokens}\n{conversation}"

        # For multi-image case
        image_tokens_list = []
        for image in images:
            tokens = self._format_vision_tokens(self._calculate_image_tokens(image))
            image_tokens_list.append(tokens)

        # Join all image tokens and conversation
        all_tokens = "\n".join(image_tokens_list)
        return f"{all_tokens}\n{conversation}"

    def _calculate_image_tokens(self, image: Any) -> int:
        """
        Calculate number of vision tokens for an image.

        CRITICAL FIX: Use the same logic as the original chat processor to ensure
        the number of image tokens matches the number of image features.

        Args:
            image: Image object (PIL Image)

        Returns:
            Number of vision tokens (post-merge for Qwen2.5-VL)
        """
        # FAIL-FAST: Validate image processor has required attributes
        if self.image_processor is None:
            raise ValueError("Image processor is required for token calculation")

        if not hasattr(self.image_processor, "preprocess"):
            raise AttributeError("Image processor must have 'preprocess' method")

        try:
            # Run the *actual* preprocessing pipeline for a single image
            # This matches the original src/chat_processor.py implementation exactly
            processed = self.image_processor.preprocess([image], return_tensors="pt")

            # FAIL-FAST: Validate processed output contains required fields
            if "image_grid_thw" not in processed:
                raise ValueError(
                    "Image processor did not return required 'image_grid_thw' field"
                )

            grid_thw = processed["image_grid_thw"][0]  # (t, h, w)

            # FAIL-FAST: Require merge_size to be explicitly defined
            if not hasattr(self.image_processor, "merge_size"):
                raise ValueError(
                    "Image processor must have 'merge_size' attribute defined"
                )

            merge_size = self.image_processor.merge_size
            tokens_per_merge = merge_size**2

            # Number of flattened patch tokens after the spatial-merge step
            # This matches the original implementation exactly
            num_tokens: int = int(grid_thw.prod().item() // tokens_per_merge)

            # Use print for debugging since logger.debug might not show
            logger.debug(
                f"🔍 Image token calculation: grid_thw={grid_thw.tolist()}, "
                f"merge_size={merge_size}, tokens_per_merge={tokens_per_merge}, "
                f"num_tokens={num_tokens}"
            )

            return num_tokens

        except Exception as e:
            logger.error(f"Failed to calculate image tokens: {e}")
            raise RuntimeError(f"Image token calculation failed: {e}") from e

    def _format_vision_tokens(self, num_tokens: int) -> str:
        """
        Format vision tokens with spaces to prevent tokenizer issues.

        Matches the original src/ implementation exactly.

        Args:
            num_tokens: Number of vision tokens

        Returns:
            Formatted vision tokens string with <|image_pad|> tokens
        """
        if num_tokens <= 0:
            return ""

        # Insert spaces between image_pad tokens to prevent tokenizer returning None IDs
        # This matches the original src/utils/tokens/special_tokens.py implementation
        IMAGE_PAD = "<|image_pad|>"
        tokens = [IMAGE_PAD] * num_tokens
        return " ".join(tokens)

    def _tokenize_simple_conversation(
        self, conversation: str, raw_sample: Dict[str, Any], images: List[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize conversation with simple format.

        Args:
            conversation: Formatted conversation string
            raw_sample: Raw sample data
            images: List of images (optional)

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Tokenize inputs
        tokenized = self.tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_total_length,
        )

        # Get input IDs and attention mask
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]

        # Create labels by masking non-assistant tokens and extract spans
        labels = input_ids.clone()

        # Check if this sample has teacher examples (multiple assistant messages)
        num_assistant_messages = conversation.count("<|im_start|>assistant")
        has_teachers = num_assistant_messages > 1

        if has_teachers:
            # Use span-aware masking for teacher-student samples
            labels, teacher_spans, student_spans = (
                self._mask_non_assistant_tokens_with_spans(labels, conversation)
            )
        else:
            # Use simple masking for student-only samples
            labels = self._mask_non_assistant_tokens_simple(labels, conversation)
            teacher_spans = []
            student_spans = []

        # Return tokenized inputs with spans
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add spans if they exist
        if teacher_spans or student_spans:
            result["teacher_assistant_spans"] = teacher_spans
            result["student_assistant_spans"] = student_spans
            logger.debug(
                f"Generated spans - Teachers: {len(teacher_spans)}, Students: {len(student_spans)}"
            )

        return result

    def _mask_non_assistant_tokens_simple(
        self, labels: torch.Tensor, conversation: str
    ) -> torch.Tensor:
        """
        Mask non-assistant tokens in labels.

        Args:
            labels: Labels tensor
            conversation: Conversation string

        Returns:
            Masked labels tensor
        """
        # Find assistant start position
        assistant_start = conversation.find("<|im_start|>assistant")
        if assistant_start == -1:
            # No assistant message, mask everything
            return torch.full_like(labels, -100)

        # Find position in tokenized sequence
        assistant_start_tokens = self.tokenizer.encode(
            conversation[:assistant_start], add_special_tokens=False
        )
        assistant_start_pos = len(assistant_start_tokens)

        # Mask everything before assistant
        labels[:assistant_start_pos] = -100

        return labels

    def _mask_non_assistant_tokens_with_spans(
        self, labels: torch.Tensor, conversation: str
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Mask non-assistant tokens and extract teacher/student spans.

        Args:
            labels: Labels tensor
            conversation: Conversation string

        Returns:
            Tuple of (masked_labels, teacher_spans, student_spans)
        """
        # Preserve original labels for restoring assistant spans
        original_labels = labels.clone()

        # Mask everything initially
        labels.fill_(-100)

        # Find all assistant message positions
        assistant_positions = []
        search_start = 0

        while True:
            assistant_start = conversation.find("<|im_start|>assistant", search_start)
            if assistant_start == -1:
                break

            # Find the end of this assistant message
            assistant_end = conversation.find("<|im_end|>", assistant_start)
            if assistant_end == -1:
                # Malformed conversation, break
                break

            assistant_positions.append(
                (assistant_start, assistant_end + len("<|im_end|>"))
            )
            search_start = assistant_end + 1

        if not assistant_positions:
            # No assistant messages found
            return labels, [], []

        # Convert text positions to token positions and unmask assistant content
        teacher_spans = []
        student_spans = []

        for i, (text_start, text_end) in enumerate(assistant_positions):
            # Find token positions for this assistant message
            prefix_text = conversation[:text_start]
            assistant_text = conversation[text_start:text_end]

            # Tokenize prefix to find start position
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            assistant_tokens = self.tokenizer.encode(
                assistant_text, add_special_tokens=False
            )

            token_start = len(prefix_tokens)
            token_end = token_start + len(assistant_tokens)

            # Ensure we don't go out of bounds
            token_start = max(0, min(token_start, len(labels)))
            token_end = max(token_start, min(token_end, len(labels)))

            if token_start < token_end:
                # Unmask this assistant message
                labels[token_start:token_end] = original_labels[token_start:token_end]

                # Determine if this is teacher or student
                # All assistant messages except the last are teachers
                if i < len(assistant_positions) - 1:
                    # This is a teacher
                    teacher_spans.append((token_start, token_end))
                else:
                    # This is the student (last assistant message)
                    student_spans.append((token_start, token_end))

        return labels, teacher_spans, student_spans

    def _process_images(
        self, structured_sample: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process images from structured sample, including both teacher and student images.

        CRITICAL FIX: This method now processes ALL images that appear in the conversation,
        including teacher images, to ensure the number of image features matches the number
        of image tokens in the conversation.

        Args:
            structured_sample: Structured sample data with potential teacher assignments

        Returns:
            Tuple of (pixel_values, image_grid_thw)
        """
        # Skip if no image processor
        if self.image_processor is None:
            return None, None

        # Collect ALL image paths in the order they appear in the conversation
        all_image_paths = []

        # First, add teacher images (they appear first in the conversation)
        teacher_samples = structured_sample.get("teacher_samples", [])
        for teacher_sample in teacher_samples:
            teacher_image_paths = teacher_sample.get("images", [])
            all_image_paths.extend(teacher_image_paths)

        # Then, add student images (they appear last in the conversation)
        student_image_paths = []
        if "image" in structured_sample:
            student_image_paths = [structured_sample["image"]]
        elif "images" in structured_sample:
            student_image_paths = structured_sample["images"]

        all_image_paths.extend(student_image_paths)

        # If no images found, return None
        if not all_image_paths:
            return None, None

        # Load and process all images
        images = []
        for path in all_image_paths:
            # Handle relative paths with data_root
            if self.data_root and not Path(path).is_absolute():
                full_path = Path(self.data_root) / path
            else:
                full_path = Path(path)

            # Load image using PIL with fail-fast validation (matching src/ implementation)
            from PIL import Image

            # Fail-fast: raise explicit error if the image cannot be loaded
            if not full_path.exists():
                raise FileNotFoundError(
                    f"❌ CRITICAL: Image file not found: {full_path}. "
                    f"All training images must be accessible for vision-language training."
                )

            # No try/catch - let errors propagate immediately (fail-fast)
            image = Image.open(str(full_path)).convert("RGB")
            images.append(image)

        # Process loaded images
        if not images:
            return None, None

        logger.debug(
            f"Processing {len(images)} images: "
            f"{len(teacher_samples)} teacher samples with {len(all_image_paths) - len(student_image_paths)} teacher images, "
            f"{len(student_image_paths)} student images"
        )

        return self._process_images_from_list(images)

    def _process_images_from_list(
        self, images: List[Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process a list of loaded images.

        Args:
            images: List of loaded image objects (no None values - fail-fast validation)

        Returns:
            Tuple of (pixel_values, image_grid_thw)
        """
        # Process images with image processor
        if self.image_processor is None:
            return None, None

        # No None filtering - all images should be valid (fail-fast approach)
        if not images:
            return None, None

        # No try/catch - let errors propagate immediately (fail-fast)
        result = self.image_processor(images, return_tensors="pt")
        pixel_values = result.get("pixel_values", None)
        image_grid_thw = result.get("image_grid_thw", None)

        logger.debug(
            f"Image processing result: pixel_values={pixel_values.shape if pixel_values is not None else None}, "
            f"image_grid_thw={image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        return pixel_values, image_grid_thw

    def _create_structured_sample(
        self, flat_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """
        Create structured sample with teacher assignments.

        Args:
            flat_sample: Flat sample data
            idx: Sample index

        Returns:
            Structured sample with teacher assignments
        """
        # Start with base sample
        structured_sample = flat_sample.copy()

        # Determine if we should use teacher samples
        use_teacher = (
            self.teacher_pool_manager is not None
            and random.random() < self.teacher_ratio
        )

        # Add teacher samples if needed
        if use_teacher:
            teacher_samples = self._sample_teachers_for_student(structured_sample, idx)
            structured_sample["teacher_samples"] = teacher_samples

            # Track teacher assignments
            if teacher_samples:
                teacher_ids = [t.get("teacher_id", "unknown") for t in teacher_samples]
                self.teacher_assignments[idx] = teacher_ids

                # Update assignment counts
                for teacher_id in teacher_ids:
                    self.teacher_assignment_counts[teacher_id] = (
                        self.teacher_assignment_counts.get(teacher_id, 0) + 1
                    )

        return structured_sample

    def _sample_teachers_for_student(
        self, student_sample: Dict[str, Any], idx: int
    ) -> List[Dict[str, Any]]:
        """
        Sample teacher examples for a student sample.

        Args:
            student_sample: Student sample data
            idx: Sample index

        Returns:
            List of teacher samples
        """
        # Skip if no teacher pool manager
        if self.teacher_pool_manager is None:
            return []

        # Get image paths
        image_paths = []
        if "image" in student_sample:
            image_paths = [student_sample["image"]]
        elif "images" in student_sample:
            image_paths = student_sample["images"]

        # Skip if no images
        if not image_paths:
            return []

        # Use first image for teacher matching
        primary_image = image_paths[0]

        # Try to get matching teachers first (image-based matching)
        try:
            teacher_samples = self.teacher_pool_manager.get_teachers_for_image(
                primary_image, self.num_teacher_samples
            )
            if teacher_samples:
                logger.debug(
                    f"✅ Found {len(teacher_samples)} image-matched teachers for {primary_image}"
                )
                return teacher_samples
        except Exception as e:
            logger.debug(
                f"⚠️ Image-based teacher matching failed for {primary_image}: {e}"
            )

        # Fallback: Random teacher assignment (for cases where images don't match)
        try:
            teacher_samples = self.teacher_pool_manager.get_random_teachers(
                self.num_teacher_samples
            )
            if teacher_samples:
                logger.debug(
                    f"✅ Assigned {len(teacher_samples)} random teachers for sample {idx}"
                )
                return teacher_samples
        except Exception as e:
            logger.warning(f"⚠️ Failed to get random teachers for sample {idx}: {e}")

        return []

    def get_teacher_assignment_summary(self) -> Dict[str, Any]:
        """
        Get summary of teacher assignments.

        Returns:
            Dictionary with teacher assignment statistics
        """
        total_assignments = sum(self.teacher_assignment_counts.values())
        total_samples = len(self.samples)
        teacher_ratio = total_assignments / total_samples if total_samples > 0 else 0

        # Get top teachers
        top_teachers = sorted(
            self.teacher_assignment_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_samples": total_samples,
            "samples_with_teachers": len(self.teacher_assignments),
            "teacher_ratio": teacher_ratio,
            "top_teachers": dict(top_teachers),
        }

    def _load_data(self) -> List[Dict]:
        """
        Load data from file.

        Returns:
            List of raw data samples

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data file is empty or invalid
        """
        # FAIL-FAST: Validate data path
        if not self.data_path:
            raise ValueError("data_path cannot be empty")

        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load data with error handling
        try:
            raw_data = read_jsonl(self.data_path)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file {self.data_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read data file {self.data_path}: {e}")

        # FAIL-FAST: Validate data loaded correctly
        if not raw_data:
            raise ValueError(f"Data file is empty: {self.data_path}")

        # Handle dataset size limiting with clear logging
        original_size = len(raw_data)
        max_dataset_size = getattr(self.config, "max_dataset_size", -1)

        # Handle None case (fallback to -1 for full dataset)
        if max_dataset_size is None:
            max_dataset_size = -1

        if max_dataset_size > 0 and max_dataset_size < original_size:
            # Limit dataset size for debugging/testing
            raw_data = raw_data[:max_dataset_size]
            logger.info(
                f"🔬 [DATASET LIMITING] Using limited dataset: {len(raw_data)}/{original_size} samples"
            )
            logger.info(
                f"📊 Dataset size limited to {max_dataset_size} samples for debugging/testing"
            )
        else:
            # Use full dataset
            logger.info(
                f"📊 [FULL DATASET] Using complete dataset: {len(raw_data)} samples"
            )
            if max_dataset_size > 0:
                logger.info(
                    f"💡 max_dataset_size={max_dataset_size} >= dataset size, using all available samples"
                )

        logger.info(f"✅ Loaded {len(raw_data)} samples from {self.data_path}")
        return raw_data


def extract_ground_truth_from_sample(
    sample_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract ground truth objects from sample data.

    Args:
        sample_data: Sample data

    Returns:
        List of ground truth objects
    """
    # Return objects directly if present
    if "objects" in sample_data:
        return sample_data["objects"]

    # Try to extract from response
    if "response" in sample_data:
        # TODO: Implement extraction from response string
        pass

    return []


class TrainerCompatibleDataset(TorchDataset):
    """
    Wrapper for dataset to ensure compatibility with HuggingFace Trainer.

    This wrapper ensures that:
    1. All tensors have consistent shapes
    2. All required fields are present
    3. Any unexpected errors are gracefully handled
    """

    def __init__(self, base_dataset: TorchDataset):
        """
        Initialize trainer-compatible dataset.

        Args:
            base_dataset: Base dataset to wrap
        """
        self.base_dataset = base_dataset
        self.logger = get_data_logger()

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get processed sample by index with error handling.

        Args:
            idx: Sample index

        Returns:
            Processed sample with tensors ready for model input
        """
        try:
            # Get item from base dataset
            result = self.base_dataset[idx]

            # Validate and format tensors
            return self._validate_and_format_tensors(result, idx)
        except Exception as e:
            # Log error and return empty sample
            self.logger.error(f"⚠️ Error processing sample {idx}: {e}")

            # Return minimal valid sample
            return {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long),
                "labels": torch.full((1,), -100, dtype=torch.long),
            }

    def _validate_and_format_tensors(
        self, result: Dict[str, Any], idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validate and format tensors for trainer compatibility.

        Args:
            result: Result from base dataset
            idx: Sample index

        Returns:
            Validated and formatted tensors
        """
        # Ensure required fields are present
        required_fields = ["input_ids", "attention_mask", "labels"]
        for field in required_fields:
            if field not in result:
                self.logger.warning(f"⚠️ Missing required field {field} in sample {idx}")
                result[field] = torch.zeros(1, dtype=torch.long)

        # Ensure all tensors are PyTorch tensors
        for key, value in result.items():
            if not isinstance(value, torch.Tensor):
                try:
                    result[key] = torch.tensor(value)
                except Exception:
                    # Remove non-tensor fields
                    del result[key]

        return result


# Export functions and classes
__all__ = [
    "Dataset",
    "TrainerCompatibleDataset",
    "read_jsonl",
    "extract_ground_truth_from_sample",
]
