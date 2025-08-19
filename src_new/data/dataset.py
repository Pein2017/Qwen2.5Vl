"""
HuggingFace-first dataset for Qwen2.5-VL training.

This is a clean refactored version that replaces the complex custom processing
with official HuggingFace components + focused coordinate token logic.

Key Features:
- Uses ConversationProcessor with official HuggingFace processor
- Eliminates 800+ lines of custom conversation processing
- Fail-fast validation with explicit errors
- Clean separation of concerns
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new.config.config import Config
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.processing.conversation import ConversationBuilder
from src_new.processing.special_tokens import (
    ASSISTANT_SPAN_PATTERN,
    GEOMETRY_TOKENS,
    IM_END,
    IMAGE_PAD,
)
from src_new.types.arrays import (
    jaxtyped_beartype,
)
from src_new.utils.path_manager import create_path_manager


def get_data_logger() -> logging.Logger:
    """Get rank-aware logger for data module."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("data")
    except ImportError:
        # Fallback to standard logging
        logger = logging.getLogger("data")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries.

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class Dataset(TorchDataset):
    """HuggingFace-first dataset with clean conversation processing."""

    # Class attribute type annotations (non-trivial state)
    data_path: str
    tokenizer: PreTrainedTokenizerBase
    image_processor: Optional[Qwen2VLImageProcessor]
    teacher_pool_manager: Optional[TeacherPoolManager]
    config: Config

    data_root: str
    teacher_ratio: float
    num_teacher_samples: int

    hf_processor: Optional[Qwen2VLProcessor]
    conversation_processor: Optional[ConversationBuilder]

    raw_data: List[Dict[str, Any]]
    samples: List[Dict[str, Any]]

    teacher_assignments: Dict[str, Any]
    teacher_assignment_counts: Dict[str, int]

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Optional[Qwen2VLImageProcessor],
        teacher_pool_manager: Optional[TeacherPoolManager],
        config: Config,
    ):
        """
        Initialize HuggingFace-first dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer (not used in HF-first approach)
            image_processor: Image processor (not used in HF-first approach)
            teacher_pool_manager: Manager for teacher examples
            config: Configuration object

        Raises:
            ValueError: If required parameters are invalid
        """
        if not data_path:
            raise ValueError("data_path cannot be empty")

        if config is None:
            raise ValueError("config cannot be None")

        # Store configuration
        self.data_path = data_path
        self.tokenizer = tokenizer  # Kept for compatibility, not used
        self.image_processor = image_processor  # Kept for compatibility, not used
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

        logger.info(
            f"✅ HuggingFace-first dataset initialized with {len(self.samples)} samples"
        )

    def _initialize_processing_components(self):
        """Initialize HuggingFace-first processing components."""
        # Data processing settings
        self.data_root = self.config.data_root
        self.teacher_ratio = self.config.teacher_ratio
        self.num_teacher_samples = self.config.num_teacher_samples

        # HuggingFace processor will be set by trainer
        self.hf_processor = None
        self.conversation_processor = None

        logger.info("🎯 HuggingFace-first processing components initialized")

    def set_processor(self, hf_processor: Qwen2VLProcessor) -> None:
        """
        Set the HuggingFace processor and initialize conversation processor.

        This method should be called by the trainer after the processor is available.

        Args:
            hf_processor: Official HuggingFace Qwen2VLProcessor

        Raises:
            ValueError: If hf_processor is None
        """
        if hf_processor is None:
            raise ValueError("hf_processor cannot be None")

        self.hf_processor = hf_processor

        from src_new.processing.conversation import ConversationBuilder

        # Fail-fast: max_coord_value must come from YAML (no defaults allowed)
        if not hasattr(self.config, "max_coord_value"):
            raise ValueError(
                "max_coord_value is required in configuration (YAML) but was not found on Config."
            )
        max_coord_value = self.config.max_coord_value
        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be a positive integer, got {max_coord_value!r}"
            )

        if not hasattr(self.config, "coordinate_tokens_enabled"):
            raise ValueError(
                "coordinate_tokens_enabled must be explicitly set in configuration (True/False)"
            )
        if not isinstance(self.config.coordinate_tokens_enabled, bool):
            raise ValueError(
                f"coordinate_tokens_enabled must be a bool, got {type(self.config.coordinate_tokens_enabled)}: {self.config.coordinate_tokens_enabled!r}"
            )

        self.conversation_processor = ConversationBuilder(
            processor=hf_processor,
            max_coord_value=max_coord_value,
            coordinate_tokens_enabled=self.config.coordinate_tokens_enabled,
        )

        logger.info("✅ HuggingFace processor and conversation processor initialized")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        try:
            return read_jsonl(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {e}")

    def _validate_and_filter_samples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and filter samples."""
        if not raw_data:
            raise ValueError("No data found in JSONL file")

        valid_samples = []
        for i, sample in enumerate(raw_data):
            if self._is_valid_sample(sample):
                valid_samples.append(sample)
            else:
                logger.warning(f"Skipping invalid sample {i}")

        if not valid_samples:
            raise ValueError("No valid samples found after filtering")

        # Apply max_dataset_size limit if specified in config
        # -1 means use all samples, None or 0 means no limit, positive values limit the dataset
        max_dataset_size = self.config.max_dataset_size
        if max_dataset_size is not None and max_dataset_size > 0:
            logger.debug(
                f"🔧 DEBUG MODE: Limiting dataset from {len(valid_samples)} to {max_dataset_size} samples"
            )
            valid_samples = valid_samples[:max_dataset_size]
        elif max_dataset_size == -1:
            logger.info(
                f"📊 Using all {len(valid_samples)} samples (max_dataset_size=-1)"
            )

        logger.info(
            f"Filtered {len(valid_samples)} valid samples from {len(raw_data)} total"
        )
        return valid_samples

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if sample is valid."""
        # Must have objects
        if "objects" not in sample or not sample["objects"]:
            return False

        # Must have images
        if "images" not in sample or not sample["images"]:
            return False

        # Each object must have valid geometry and description
        for obj in sample["objects"]:
            if "desc" not in obj:
                return False

            # Must have at least one geometry type
            geometry_types = list(GEOMETRY_TOKENS.keys())
            if not any(geom_type in obj for geom_type in geometry_types):
                return False

        return True

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)

    @jaxtyped_beartype
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get processed sample by index.

        Args:
            idx: Sample index

        Returns:
            Processed sample with tensors ready for model input

        Raises:
            ValueError: If conversation processor is not initialized
        """
        if self.conversation_processor is None:
            raise ValueError(
                "Conversation processor not initialized. Call set_processor() first."
            )

        # Get raw sample
        raw_sample = self.samples[idx]

        # Create structured sample with teacher assignments
        structured_sample = self._create_structured_sample(raw_sample, idx)

        # Process the sample through HuggingFace-first pipeline
        return self._process_sample_unified(structured_sample, idx)

    def _create_structured_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Create structured sample with teacher assignments."""
        structured_sample = raw_sample.copy()

        # Add teacher examples if conditions are met
        if (
            self.teacher_pool_manager
            and random.random() < self.teacher_ratio
            and len(self.teacher_pool_manager.teacher_pool) > 0
        ):
            # Select random teachers
            num_teachers = min(
                random.randint(1, self.num_teacher_samples),
                len(self.teacher_pool_manager.teacher_pool),
            )

            selected_teachers = random.sample(
                self.teacher_pool_manager.teacher_pool, num_teachers
            )

            structured_sample["teacher_samples"] = selected_teachers
            logger.debug(f"Assigned {num_teachers} teachers to sample {idx}")

        return structured_sample

    @jaxtyped_beartype
    def _process_sample_unified(
        self, structured_sample: Dict[str, Any], idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process sample through HuggingFace-first pipeline.

        Args:
            structured_sample: Structured sample data with potential teacher assignments
            idx: Sample index for logging

        Returns:
            Processed sample with tensors ready for model input

        Raises:
            ValueError: If processing fails
        """
        try:
            # Extract teacher samples if present
            teacher_samples = (
                structured_sample["teacher_samples"]
                if "teacher_samples" in structured_sample
                else []
            )
            has_teachers = len(teacher_samples) > 0

            # Load images
            if has_teachers:
                # Load teacher images
                teacher_images_list = []
                for teacher_sample in teacher_samples:
                    if "images" not in teacher_sample:
                        raise ValueError(
                            "Teacher sample missing required 'images' list"
                        )
                    teacher_image_paths = teacher_sample["images"]
                    teacher_images = self._load_images_from_paths(teacher_image_paths)
                    teacher_images_list.append(teacher_images)

                # Load student images
                if "images" not in structured_sample:
                    raise ValueError("Sample missing required 'images' list")
                student_image_paths = structured_sample["images"]
                student_images = self._load_images_from_paths(student_image_paths)

                # Use HuggingFace-first conversation processor
                inputs = (
                    self.conversation_processor.create_teacher_student_conversation(
                        student_sample=structured_sample,
                        teacher_samples=teacher_samples,
                        student_images=student_images,
                        teacher_images_list=teacher_images_list,
                    )
                )

                logger.debug(
                    f"✅ Created teacher-student conversation for sample {idx}"
                )
            else:
                # Load student images only
                if "images" not in structured_sample:
                    raise ValueError("Sample missing required 'images' list")
                image_paths = structured_sample["images"]
                images = self._load_images_from_paths(image_paths)

                # Use HuggingFace-first conversation processor
                inputs = self.conversation_processor.create_simple_conversation(
                    sample=structured_sample, images=images
                )
                logger.debug(f"✅ Created student-only conversation for sample {idx}")

            # Create labels with proper masking for training and extract spans
            labels, teacher_spans, student_spans = (
                self._create_masked_labels_with_spans(
                    inputs["input_ids"],
                    self.tokenizer,
                    has_teachers=(len(teacher_samples) > 0),
                )
            )
            inputs["labels"] = labels
            inputs["teacher_assistant_spans"] = teacher_spans
            inputs["student_assistant_spans"] = student_spans
            # Do not populate unified assistant_spans in legacy mode to avoid overriding teacher/student logic
            inputs.pop("assistant_spans", None)

            return inputs

        except Exception as e:
            # Fail-fast: re-raise with context without swallowing the root cause
            logger.error(f"Error processing sample {idx}: {type(e).__name__}: {e}")
            raise

    def _create_masked_labels_with_spans(
        self, input_ids: torch.Tensor, tokenizer, has_teachers: bool = False
    ) -> tuple[torch.Tensor, List[tuple[int, int]], List[tuple[int, int]]]:
        """
        Create properly masked labels using OFFSET MAPPING for accurate span detection.

        This method uses tokenizer's offset_mapping to get precise token-to-character
        alignment, ensuring spans correspond to actual token positions.

        Args:
            input_ids: Token sequence [1, seq_len]
            tokenizer: Tokenizer with offset_mapping support
            has_teachers: Whether this conversation has teacher examples

        Returns:
            Tuple of (masked_labels, teacher_spans, student_spans)
        """
        # Remove batch dimension for processing
        input_ids_1d = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
        labels = input_ids_1d.clone()
        teacher_spans = []
        student_spans = []

        try:
            # STEP 1: Get the full conversation text
            full_text = tokenizer.decode(input_ids_1d, skip_special_tokens=False)

            # STEP 2: Re-tokenize with offset mapping for accurate alignment
            tokenized_with_offsets = tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors="pt",
            )

            offset_mapping = tokenized_with_offsets["offset_mapping"][
                0
            ]  # Remove batch dim

            # STEP 3: Find assistant content spans using accurate text-to-token mapping
            assistant_spans = self._find_assistant_spans_with_offsets(
                full_text, offset_mapping, has_teachers
            )

            # STEP 4: Mask all tokens initially
            labels.fill_(-100)

            # STEP 5: Unmask assistant content spans
            for span_info in assistant_spans:
                start_token, end_token, is_teacher = span_info

                # Validate span bounds
                if 0 <= start_token < end_token <= len(labels):
                    # Try to include the immediate <|im_end|> token (if present) in the span
                    try:
                        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
                    except Exception:
                        im_end_id = None

                    final_end = end_token
                    if im_end_id is not None and im_end_id != -1:
                        # Prefer the token right at end_token, but allow a tiny lookahead window
                        # to be robust to tokenization boundary nuances.
                        for pos in range(end_token, min(len(labels), end_token + 5)):
                            if int(input_ids_1d[pos].item()) == int(im_end_id):
                                final_end = pos + 1  # include <|im_end|> in the span
                                logger.debug(
                                    f"✅ Extended span to include <|im_end|> at position {pos} for original span ({start_token}, {end_token})"
                                )
                                break

                    # Unmask this span (now possibly extended to include <|im_end|>)
                    labels[start_token:final_end] = input_ids_1d[start_token:final_end]

                    # Track spans for loss computation (use the extended end)
                    span = (start_token, final_end)
                    if is_teacher:
                        teacher_spans.append(span)
                        logger.debug(f"✅ Teacher span: {span}")
                    else:
                        student_spans.append(span)
                        logger.debug(f"✅ Student span: {span}")
                else:
                    logger.error(
                        f"Invalid span bounds: {start_token}:{end_token} for sequence length {len(labels)}"
                    )

            # STEP 6: Mask image pad tokens
            image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
            if image_pad_id is not None:
                image_pad_mask = input_ids_1d == image_pad_id
                labels[image_pad_mask] = -100

        except Exception as e:
            # Fail-fast: do not silently continue with invalid spans
            logger.error(
                f"Error in offset-based span detection: {type(e).__name__}: {e}"
            )
            raise

        # Log final span statistics
        logger.debug(
            f"📊 Final spans - Teachers: {len(teacher_spans)}, Students: {len(student_spans)}"
        )

        # Restore batch dimension if needed
        if input_ids.dim() > 1:
            labels = labels.unsqueeze(0)

        return labels, teacher_spans, student_spans

    def _find_assistant_spans_with_offsets(
        self, full_text: str, offset_mapping: torch.Tensor, has_teachers: bool
    ) -> List[tuple[int, int, bool]]:
        """
        Find assistant content spans using offset mapping for accurate token alignment.

        Args:
            full_text: Full conversation text
            offset_mapping: Token-to-character mapping from tokenizer
            has_teachers: Whether conversation has teacher examples

        Returns:
            List of (start_token, end_token, is_teacher) tuples
        """

        assistant_spans = []
        assistant_count = 0

        # Find all assistant segments using centralized regex
        for match in re.finditer(ASSISTANT_SPAN_PATTERN, full_text, re.DOTALL):
            # Get character positions
            content_start_char = match.start(1)  # Start of assistant content (group 1)
            content_end_char = match.end(1)  # End of assistant content (group 1)

            # Convert character positions to token positions using offset mapping
            start_token = self._char_to_token_position(
                content_start_char, offset_mapping
            )
            end_token = self._char_to_token_position(content_end_char, offset_mapping)

            if start_token is not None and end_token is not None:
                # Determine if this is a teacher or student
                if has_teachers and assistant_count == 0:
                    # First assistant in teacher-student mode is teacher
                    is_teacher = True
                else:
                    # All others are students
                    is_teacher = False

                assistant_spans.append((start_token, end_token, is_teacher))
                assistant_count += 1

                logger.debug(
                    f"Found {'teacher' if is_teacher else 'student'} span: "
                    f"chars {content_start_char}:{content_end_char} -> tokens {start_token}:{end_token}"
                )

        return assistant_spans

    def _char_to_token_position(
        self, char_pos: int, offset_mapping: torch.Tensor
    ) -> Optional[int]:
        """
        Convert character position to token position using offset mapping.

        Args:
            char_pos: Character position in text
            offset_mapping: Token offset mapping [(start_char, end_char), ...]

        Returns:
            Token position or None if not found
        """
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char <= char_pos < end_char:
                return token_idx
            elif char_pos == end_char and token_idx < len(offset_mapping) - 1:
                # Handle boundary case - character position at token boundary
                return token_idx + 1

        # If exact match not found, find closest token
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char >= char_pos:
                return token_idx

        return len(offset_mapping)  # End of sequence

    def _load_images_from_paths(self, image_paths: List[str]) -> List[Image.Image]:
        """Load images from paths."""
        path_manager = create_path_manager(self.data_root)
        images = []
        for img_path in image_paths:
            try:
                resolved = path_manager.resolve_path(img_path)
                resolved_str = str(resolved)
                image = Image.open(resolved_str).convert("RGB")
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                raise

        return images


# Export classes
__all__ = [
    "Dataset",
    "read_jsonl",
]
