"""
Optimized data handling for Qwen2.5-VL training with unified preprocessing.

This module provides:
- BBUDataset: Dataset using UnifiedPreprocessor for clean data processing
- FlattenedDataCollator: Default collator using packed sequences (like official repo)
- StandardDataCollator: Traditional padding-based collator for compatibility
- Utilities for ground truth extraction from conversation format

Key Features:
- Unified preprocessing pipeline (no file dependencies)
- Multi-image conversation support
- Compatible with flash attention and DeepSpeed
- Optimized for large sequences and multi-GPU training
"""

import json
import random
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.chat_processor import ChatProcessor
from src.config import get_config

# Get the debug logger from losses.py
from src.logger_utils import get_data_logger
from src.teacher_pool import TeacherPoolManager, create_teacher_pool_manager
from src.utils.schema import ChatProcessorOutput, assert_collated_batch
from src.utils.tokens import SpecialTokens
from src.utils.utils import IGNORE_INDEX


logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries.

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class BBUDataset(Dataset):
    """Dataset using UnifiedPreprocessor for clean data processing."""

    def __init__(
        self,
        data_path: str,
        chat_processor: ChatProcessor,
        teacher_pool_manager: Optional[TeacherPoolManager],
        teacher_ratio: float,
        is_training: bool,
        config=None,
    ):
        """
        Initialize BBU dataset with flat sample format and dynamic teacher pairing.

        Args:
            data_path: Path to all_samples.jsonl file (flat format)
            chat_processor: Chat processor instance
            teacher_pool_manager: Manager for teacher examples
            teacher_ratio: Ratio of samples to use teacher examples (0.0 = no teachers)
            is_training: Whether this is a training dataset (affects prompt selection)
            config: Configuration object (explicit config or global config)

        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If the dataset contains invalid samples
        """
        # Get config for this instance
        if config is None:
            config = get_config()

        self.data_path = data_path
        self.chat_processor = chat_processor
        self.teacher_pool_manager = teacher_pool_manager
        self.teacher_ratio = teacher_ratio
        self.is_training = is_training

        # Set context for chat processor if available
        context = "training" if is_training else "evaluation"
        if hasattr(self.chat_processor, "set_context"):
            self.chat_processor.set_context(context)  # type: ignore[attr-defined]
        else:
            logger.debug(
                "ChatProcessor has no `set_context`; proceeding without context flag."
            )

        # Load flat samples from all_samples.jsonl
        self.data = self._load_data()

        logger.info(f"Loaded {len(self.data)} flat samples from {data_path}")
        logger.info(f"Dataset mode: {'training' if is_training else 'evaluation'}")
        logger.info(f"Teacher ratio: {teacher_ratio}")
        if teacher_pool_manager:
            logger.info(f"Teacher pool size: {len(teacher_pool_manager)}")

        # Initialize special tokens
        self.tokens = SpecialTokens()

        # Initialize teacher assignment tracking
        self._teacher_assignment_stats = {
            "total_samples": 0,
            "samples_with_teacher": 0,
            "samples_without_teacher": 0,
        }

        # Get number of teachers from config
        if not hasattr(config, "num_teacher_samples"):
            raise AttributeError(
                "'num_teacher_samples' must be specified in YAML configuration"
            )

        self._num_teachers = int(config.num_teacher_samples)

        # Use consistent teacher ratio for both train and validation
        self.teacher_ratio = config.teacher_ratio

        # Disable teacher sampling if requested
        if self._num_teachers == 0:
            logger.info(
                f"Dataset {self.data_path}: teacher sampling disabled (num_teacher_samples=0)"
            )
            self.teacher_ratio = 0.0
        else:
            logger.info(
                f"Dataset {self.data_path}: teacher ratio set to {self.teacher_ratio}, num_teachers={self._num_teachers}"
            )

        # Instantiate teacher pool manager if needed
        if self._num_teachers > 0 and teacher_pool_manager is None:
            self.teacher_pool_manager = create_teacher_pool_manager(config)
        else:
            self.teacher_pool_manager = teacher_pool_manager

        if self._num_teachers > 0 and not self.teacher_pool_manager:
            raise ValueError(
                "Teacher sampling enabled but teacher pool manager is not available"
            )

    @property
    def data_root(self) -> str:
        """Get data root from global config."""
        return get_config().data_root

    @property
    def model_max_length(self) -> int:
        """Get model max length from global config."""
        return get_config().max_total_length

    def _validate_and_filter_samples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate samples with strict requirements.

        Raises:
            ValueError: If any sample is invalid
        """
        valid_samples = []
        is_training = "train" in self.data_path.lower()
        min_images = 1  # At least one image must be present

        logger.debug(
            f"🔍 Validating samples with minimum {min_images} images ({'training' if is_training else 'validation'} mode)"
        )

        for idx, sample in enumerate(raw_data):
            # Strict validation for sample format
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} is not a dictionary")

            # Handle teacher pool format (with 'teachers' and 'student' fields)
            if "teachers" in sample and "student" in sample:
                # For teacher pool files, we validate the student part
                student_sample = sample["student"]
                if not isinstance(student_sample, dict):
                    raise ValueError(
                        f"Sample {idx} has invalid 'student' field (not a dictionary)"
                    )

                if "images" not in student_sample or "objects" not in student_sample:
                    raise ValueError(
                        f"Student in sample {idx} missing required fields 'images' or 'objects'. Found keys: {list(student_sample.keys())}"
                    )

                if (
                    not isinstance(student_sample["images"], list)
                    or len(student_sample["images"]) == 0
                ):
                    raise ValueError(
                        f"Student in sample {idx} has empty 'images' field"
                    )

                if not isinstance(student_sample["objects"], list):
                    raise ValueError(
                        f"Student in sample {idx} has invalid 'objects' field (must be a list)"
                    )

                # Add the validated teacher-student sample
                valid_samples.append(sample)
                continue

            # Validate flat sample format (images + objects)
            if "images" not in sample or "objects" not in sample:
                raise ValueError(
                    f"Sample {idx} missing required fields 'images' or 'objects'. Found keys: {list(sample.keys())}"
                )

            if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
                raise ValueError(f"Sample {idx} has empty 'images' field")

            if not isinstance(sample["objects"], list):
                raise ValueError(
                    f"Sample {idx} has invalid 'objects' field (must be a list)"
                )

            # Add the validated flat sample
            valid_samples.append(sample)

        if not valid_samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        logger.debug(f"✅ Validated {len(valid_samples)} samples")
        return valid_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with strict validation - no fallbacks."""
        logger.debug(f"🔍 DATASET: Loading flat sample {idx}")
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Internal getter to handle flat sample processing and teacher pairing."""
        # Get flat sample from data
        flat_sample = self.data[idx]

        # DEBUG: Detailed logging for flat sample
        logger.debug(f"🔍 FLAT SAMPLE {idx}:")
        logger.debug(f"   Keys: {list(flat_sample.keys())}")
        logger.debug(
            f"   Objects count: {len(flat_sample['objects']) if 'objects' in flat_sample else 0}"
        )
        logger.debug(f"   Images: {flat_sample['images']}")

        # Create teacher-student structured sample
        structured_sample = self._create_structured_sample(flat_sample, idx)

        # DEBUG: Log structured sample
        logger.debug(f"🔍 STRUCTURED SAMPLE {idx}:")
        if "student" in structured_sample and "objects" in structured_sample["student"]:
            student_objects = structured_sample["student"]["objects"]
            logger.debug(
                f"   Student objects count: {len(student_objects) if student_objects else 0}"
            )
        if "teachers" in structured_sample:
            teachers = structured_sample["teachers"]
            logger.debug(f"   Teachers count: {len(teachers) if teachers else 0}")

        # Process through chat processor
        processed_data = self.chat_processor.process_sample(structured_sample)

        # Convert ChatProcessorOutput to Dict[str, torch.Tensor] if needed
        if not isinstance(processed_data, dict):
            from dataclasses import asdict

            processed_data = asdict(processed_data)

        return processed_data

    def _create_structured_sample(
        self, flat_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """
        Create structured teacher-student sample from flat sample.

        Args:
            flat_sample: Flat sample from all_samples.jsonl
            idx: Sample index for reproducible teacher sampling

        Returns:
            Dict with "teachers" (List[Sample]) and "student" (Sample) keys
        """
        # For samples without teachers (determined by teacher_ratio)
        if (
            self.teacher_ratio == 0.0
            or self.teacher_pool_manager is None
            or self._num_teachers == 0
        ):
            return {"teachers": [], "student": flat_sample}

        # Sample teachers based on teacher_ratio probability
        teachers = self._sample_teachers_for_student(flat_sample, idx)

        # Build teacher-student structure
        return {"teachers": teachers, "student": flat_sample}

    def _sample_teachers_for_student(
        self, student_sample: Dict[str, Any], idx: int
    ) -> List[Dict[str, Any]]:
        """
        Sample teacher examples for a student sample based on teacher_ratio.

        Args:
            student_sample: The student sample to create teachers for
            idx: Sample index for reproducible sampling

        Returns:
            List of teacher samples (empty if teachers are not used for this sample)
        """
        # Track statistics
        self._teacher_assignment_stats["total_samples"] += 1

        # Decide whether to use teachers based on teacher_ratio
        use_teachers = random.random() < self.teacher_ratio

        # Update statistics
        if use_teachers:
            self._teacher_assignment_stats["samples_with_teacher"] += 1
        else:
            self._teacher_assignment_stats["samples_without_teacher"] += 1
            return []  # No teachers for this sample

        # Log statistics periodically (every 100 samples)
        if self._teacher_assignment_stats["total_samples"] % 100 == 0:
            total = self._teacher_assignment_stats["total_samples"]
            with_teacher = self._teacher_assignment_stats["samples_with_teacher"]
            actual_ratio = with_teacher / total if total > 0 else 0.0
            logger.debug(
                f"📊 Teacher Assignment Stats (sample {idx}): "
                f"{with_teacher}/{total} samples with teacher "
                f"(actual ratio: {actual_ratio:.3f}, configured: {self.teacher_ratio:.3f})"
            )

        # Create reproducible seed for this sample
        epoch_seed = hash((idx, random.getstate()[1][0])) % (2**32)

        # Get teachers from pool using the reproducible seed
        if self.teacher_pool_manager is None:
            raise ValueError(
                f"Teacher pool manager is None but teacher sampling is enabled"
            )

        # Get multiple teachers from pool
        teachers = self.teacher_pool_manager.get_multiple_teachers(
            num_teachers=self._num_teachers,
            seed=epoch_seed,
        )

        logger.debug(
            f"Sample {idx}: Sampled {len(teachers)} teachers (seed={epoch_seed})"
        )
        return teachers

    def get_teacher_assignment_summary(self) -> Dict[str, Any]:
        """Get summary of teacher assignment statistics."""
        stats = self._teacher_assignment_stats
        total = stats["total_samples"]
        with_teacher = stats["samples_with_teacher"]
        without_teacher = stats["samples_without_teacher"]

        return {
            "total_samples_processed": total,
            "samples_with_teacher": with_teacher,
            "samples_without_teacher": without_teacher,
            "actual_teacher_ratio": with_teacher / total if total > 0 else 0.0,
            "configured_teacher_ratio": self.teacher_ratio,
            "ratio_accuracy": abs(
                (with_teacher / total if total > 0 else 0.0) - self.teacher_ratio
            ),
        }

    def _load_data(self) -> List[Dict]:
        """Load flat samples from JSONL file.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains no valid samples
        """
        # Load raw data
        raw_data = read_jsonl(self.data_path)
        logger.debug(f"📊 Loaded {len(raw_data)} raw samples from {self.data_path}")

        if not raw_data:
            raise ValueError(f"No samples found in {self.data_path}")

        # Basic validation and filtering for flat format
        validated_data = self._validate_and_filter_samples(raw_data)
        logger.debug(f"📊 After validation: {len(validated_data)} valid samples")

        return validated_data


def extract_ground_truth_from_sample(
    sample_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: This function is no longer needed as the ChatProcessor
    now handles ground truth extraction and normalization directly.
    Kept for historical reference but should not be used.
    """
    raise NotImplementedError(
        "extract_ground_truth_from_sample is deprecated and should not be called. "
        "Use the ChatProcessor's process_sample method instead."
    )


@dataclass
class StandardDataCollator:
    """
    Standard data collator optimized for Flash Attention and memory efficiency.

    Uses LEFT padding to batch max length for full Flash Attention compatibility.

    CRITICAL: This collator now uses LEFT padding (padding_side='left') which is
    required for Qwen2.5-VL Flash Attention. Make sure tokenizer.padding_side='left'
    is set when creating the tokenizer.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self, instances: Sequence[Any]
    ) -> Mapping[
        str,
        Union[torch.Tensor, List[int], List[List[int]], List[List[Tuple[int, int]]]],
    ]:
        """Collate a batch of :class:`ChatProcessorOutput` or raw dicts.

        Raises:
            ValueError: If instances contain incompatible or missing data
            AssertionError: If attention mask validation fails
        """

        # ------------------------------------------------------------------
        # Normalise instance format: if caller passed dataclasses convert them
        # to plain dicts so the rest of the logic remains unchanged.
        # ------------------------------------------------------------------
        if instances and isinstance(instances[0], ChatProcessorOutput):
            instances = [asdict(ins) for ins in instances]  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Extract teacher-student spans from instances before dict conversion
        # ------------------------------------------------------------------
        teacher_spans_batch: list[list[tuple[int, int]]] = []
        student_spans_batch: list[list[tuple[int, int]]] = []

        for instance in instances:
            # Extract spans from each instance - defaults to empty list for samples without teachers
            teacher_spans = (
                instance["teacher_assistant_spans"]
                if "teacher_assistant_spans" in instance
                else []
            )
            student_spans = (
                instance["student_assistant_spans"]
                if "student_assistant_spans" in instance
                else []
            )

            teacher_spans_batch.append(teacher_spans)
            student_spans_batch.append(student_spans)

        # FAIL-FAST: Validate required fields in all instances
        for i, instance in enumerate(instances):
            if "input_ids" not in instance:
                raise ValueError(f"Instance {i} missing required field 'input_ids'")
            if "labels" not in instance:
                raise ValueError(f"Instance {i} missing required field 'labels'")

            if "pixel_values" in instance and instance["pixel_values"] is not None:
                # Use default merge_size for vision token calculation
                merge_size = 2  # Default Qwen2.5-VL merge_size

                # Log vision token information for debugging
                if (
                    "image_grid_thw" in instance
                    and instance["image_grid_thw"] is not None
                ):
                    grid_thw = instance["image_grid_thw"]
                    merge_length = merge_size**2

                    total_final_tokens = 0
                    for grid in grid_thw:
                        total_final_tokens += grid.prod().item() // merge_length

                    pre_merge_tokens = instance["pixel_values"].shape[0]
                    logger.debug(
                        f"Sample {i}: {pre_merge_tokens} pre-merge → {total_final_tokens} final tokens"
                    )
                else:
                    raise ValueError(
                        f"Sample {i} has pixel_values but missing image_grid_thw"
                    )

        # 1. Extract sequences
        input_ids_list: List[torch.Tensor] = [
            instance["input_ids"].squeeze() for instance in instances
        ]
        labels_list: List[torch.Tensor] = [
            instance["labels"].squeeze() for instance in instances
        ]
        position_ids_list: List[Optional[torch.Tensor]] = [
            instance["position_ids"] if "position_ids" in instance else None
            for instance in instances
        ]

        # 2. Calculate batch dimensions
        sequence_lengths = [seq.shape[-1] for seq in input_ids_list]
        batch_max_length: int = max(sequence_lengths)
        batch_size = len(instances)

        # 3. Analyze vision token information for debugging
        vision_info = []
        for i, instance in enumerate(instances):
            has_images = (
                "pixel_values" in instance and instance["pixel_values"] is not None
            )
            image_count = instance["pixel_values"].shape[0] if has_images else 0
            vision_info.append(
                {
                    "has_images": has_images,
                    "image_count": image_count,
                    "seq_len": sequence_lengths[i],
                }
            )

        # Log comprehensive sequence information for debugging
        logger.debug(f"📊 BATCH SEQUENCE INFO:")
        logger.debug(f"   Batch size: {batch_size}")
        logger.debug(f"   Individual lengths: {sequence_lengths}")
        logger.debug(f"   Max length in batch: {batch_max_length}")
        logger.debug(f"   Min length in batch: {min(sequence_lengths)}")
        logger.debug(f"   Total tokens (sum): {sum(sequence_lengths)}")

        # Log vision token information
        total_images = sum(info["image_count"] for info in vision_info)
        logger.debug(f"🖼️ VISION TOKEN INFO:")
        logger.debug(f"   Total images in batch: {total_images}")
        logger.debug(
            f"   Samples with images: {sum(1 for info in vision_info if info['has_images'])}"
        )
        for i, info in enumerate(vision_info):
            if info["has_images"]:
                logger.debug(
                    f"   Sample {i}: {info['image_count']} images, seq_len={info['seq_len']}"
                )

        # 4. Create padded tensors efficiently (single allocation)
        # Make sure we have a valid pad token ID
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        padded_input_ids = torch.full(
            size=(batch_size, batch_max_length),
            fill_value=int(pad_token_id),
            dtype=input_ids_list[0].dtype,
        )

        # Ensure IGNORE_INDEX is a numeric value
        padded_labels = torch.full(
            size=(batch_size, batch_max_length),
            fill_value=-100,  # Use -100 directly instead of IGNORE_INDEX
            dtype=labels_list[0].dtype,
        )

        # Create attention mask for flash attention (boolean mask)
        # This mask represents the original text sequence lengths BEFORE vision token expansion
        attention_mask = torch.zeros(
            (batch_size, batch_max_length),
            dtype=torch.bool,
        )

        # 5. Fill padded tensors (LEFT padding for Flash Attention compatibility)
        for i, (input_seq, label_seq) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = input_seq.shape[-1]
            # LEFT padding: place actual data at the END of the padded sequence
            start_idx = batch_max_length - seq_len
            padded_input_ids[i, start_idx:] = input_seq
            padded_labels[i, start_idx:] = label_seq
            attention_mask[i, start_idx:] = True

        # ------------------------------------------------------------------
        # Sanity-check attention_mask: each row must contain exactly `seq_len`
        # True values corresponding to the non-padded tokens for that sample.
        # ------------------------------------------------------------------
        for i, seq_len in enumerate(sequence_lengths):
            actual_len = attention_mask[i].sum().item()
            if actual_len != seq_len:
                raise AssertionError(
                    f"StandardDataCollator: attention_mask row {i} has {actual_len} true values, expected {seq_len}"
                )

        # 6. Build batch dict
        batch: Dict[str, torch.Tensor] = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }

        # Log final attention mask info for flash attention debugging
        logger.debug(f"🎯 ATTENTION MASK INFO:")
        logger.debug(f"   Attention mask shape: {attention_mask.shape}")
        logger.debug(f"   Attention mask dtype: {attention_mask.dtype}")
        mask_lengths = attention_mask.sum(dim=-1).tolist()
        logger.debug(f"   Attention mask lengths: {mask_lengths}")
        logger.debug(f"   Uniform attention masks: {len(set(mask_lengths)) == 1}")

        # 7. Handle position_ids if provided (LEFT padding for Flash Attention)
        if any(pos_ids is not None for pos_ids in position_ids_list):
            padded_position_ids_list: List[torch.Tensor] = []
            for i, pos_ids in enumerate(position_ids_list):
                if pos_ids is not None:
                    seq_len = pos_ids.shape[-1]
                    padded_pos = torch.zeros(
                        (3, 1, batch_max_length), dtype=pos_ids.dtype
                    )
                    # LEFT padding: place actual data at the END
                    start_idx = batch_max_length - seq_len
                    padded_pos[:, :, start_idx:] = pos_ids
                else:
                    padded_pos = torch.zeros((3, 1, batch_max_length), dtype=torch.long)
                padded_position_ids_list.append(padded_pos)

            batch["position_ids"] = torch.cat(padded_position_ids_list, dim=1)

        # 8. Handle images - FAIL-FAST approach
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
            and instance["pixel_values"] is not None
            and instance["pixel_values"].shape[0] > 0
        ]

        if images:
            # Track image counts per sample for proper extraction during generation
            image_counts_per_sample = []
            for instance in instances:
                if (
                    "pixel_values" in instance
                    and instance["pixel_values"] is not None
                    and instance["pixel_values"].shape[0] > 0
                ):
                    image_counts_per_sample.append(instance["pixel_values"].shape[0])
                else:
                    image_counts_per_sample.append(0)

            # Store image counts in batch for later use
            batch["image_counts_per_sample"] = image_counts_per_sample

            # Concatenate valid images
            batch["pixel_values"] = torch.cat(images, dim=0)

            # Ensure bf16 precision for pixel_values
            if batch["pixel_values"].dtype != torch.bfloat16:
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
                logger.debug(f"🔧 Converted pixel_values to bf16")

            # Handle image grid info - REQUIRED if pixel_values exist
            grid_thw_list = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
                and instance["image_grid_thw"] is not None
                and instance["image_grid_thw"].shape[0] > 0
            ]

            if not grid_thw_list:
                raise ValueError(
                    "pixel_values present but no valid image_grid_thw found. "
                    "Both pixel_values and image_grid_thw must be consistent."
                )

            batch["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)
            logger.debug(f"🖼️ Image grid info: {batch['image_grid_thw'].shape}")
            logger.debug(f"🖼️ Image counts per sample: {image_counts_per_sample}")
        else:
            # No images in batch
            batch["image_counts_per_sample"] = [0] * batch_size

        # 9. Extract ground truth objects for detection loss
        ground_truth_objects = []
        for instance in instances:
            if "ground_truth_objects" in instance:
                ground_truth_objects.append(instance["ground_truth_objects"])
            else:
                ground_truth_objects.append([])

        batch["ground_truth_objects"] = ground_truth_objects

        # Add teacher-student spans to batch
        batch["teacher_assistant_spans"] = teacher_spans_batch
        batch["student_assistant_spans"] = student_spans_batch

        # Fail-fast shape validation (raises AssertionError on mismatch)
        assert_collated_batch(batch)

        return batch


@dataclass
class PackedDataCollator:
    """Memory-efficient collator that *packs* all samples into a single row.

    This completely removes padding.  Each sample's true length is encoded in
    a prefix-sum vector (cu_seqlens) stored in the *attention_mask* field – the
    exact format expected by `flash_attn_varlen_func` used in Qwen2-VL.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self, instances: Sequence[Any]
    ) -> Mapping[
        str,
        Union[torch.Tensor, List[int], List[List[int]], List[List[Tuple[int, int]]]],
    ]:
        # Convert dataclass inputs to dicts (if needed) early.
        if instances and isinstance(instances[0], ChatProcessorOutput):
            instances = [asdict(ins) for ins in instances]  # type: ignore[assignment]

        # Fail-fast validation of required fields
        for i, instance in enumerate(instances):
            if "input_ids" not in instance:
                raise ValueError(f"Instance {i} missing required field 'input_ids'")
            if "labels" not in instance:
                raise ValueError(f"Instance {i} missing required field 'labels'")

        # ------------------------------------------------------------------
        # Extract teacher-student spans and adjust for packed sequences
        # ------------------------------------------------------------------
        teacher_spans_batch: list[list[tuple[int, int]]] = []
        student_spans_batch: list[list[tuple[int, int]]] = []

        for instance in instances:
            # Extract spans from each instance - defaults to empty list for samples without teachers
            teacher_spans = (
                instance["teacher_assistant_spans"]
                if "teacher_assistant_spans" in instance
                else []
            )
            student_spans = (
                instance["student_assistant_spans"]
                if "student_assistant_spans" in instance
                else []
            )

            teacher_spans_batch.append(teacher_spans)
            student_spans_batch.append(student_spans)

        # ------------------------------------------------------------------
        # 1. Gather required per-sample tensors
        # ------------------------------------------------------------------
        input_ids_list = [ins["input_ids"] for ins in instances]
        labels_list = [ins["labels"] for ins in instances]
        # Note: we deliberately ignore any caller-provided `position_ids` when
        #       packing because they are likely already **shifted** for
        #       individual sequences and therefore incompatible once all
        #       samples are concatenated.  We regenerate a fresh, flat
        #       1-D vector that restarts from 0 at every sample boundary.

        # ------------------------------------------------------------------
        # 2. Compute per-sample lengths and cumulative sequence lens vector
        #    Flash-Attention var-len kernel expects **inclusive prefix-sum** of
        #    sequence lengths with a leading zero (cu_seqlens).
        # ------------------------------------------------------------------
        seq_lens: list[int] = [ids.shape[1] for ids in input_ids_list]
        cu_seqlens = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(0)

        # ------------------------------------------------------------------
        # Adjust teacher-student spans for packed sequences
        # After packing, spans need to be offset by sample start positions
        # ------------------------------------------------------------------
        adjusted_teacher_spans: list[list[tuple[int, int]]] = []
        adjusted_student_spans: list[list[tuple[int, int]]] = []

        for i, (teacher_spans, student_spans) in enumerate(
            zip(teacher_spans_batch, student_spans_batch)
        ):
            offset = cu_seqlens[
                i
            ].item()  # Start position of this sample in packed sequence

            # Adjust teacher spans
            adjusted_teacher = [
                (start + offset, end + offset) for start, end in teacher_spans
            ]
            adjusted_teacher_spans.append(adjusted_teacher)

            # Adjust student spans
            adjusted_student = [
                (start + offset, end + offset) for start, end in student_spans
            ]
            adjusted_student_spans.append(adjusted_student)

        # ------------------------------------------------------------------
        # 3. Concatenate along sequence dimension (dim=1) – no padding.
        # ------------------------------------------------------------------
        input_ids = torch.cat(input_ids_list, dim=1)
        labels = torch.cat(labels_list, dim=1)

        # NEW: Mask cross-sample prediction targets --------------------------------------------------
        # After packing, the first token of each *subsequent* sample would otherwise be trained with
        # context from the *previous* sample.  To avoid this erroneous supervision we set the label
        # of every sample-boundary token to IGNORE_INDEX so it is excluded from the LM loss.
        if cu_seqlens.numel() > 2:  # more than one sample in the packed batch
            boundary_indices = cu_seqlens[1:-1].to(
                torch.long
            )  # start positions of samples 2, 3, ...
            labels[..., boundary_indices] = IGNORE_INDEX
        # -------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 4. Position-ids handling – **single** row (temporal axis only).
        #    Shape expected by `prepare_fa2_from_position_ids` is (B, T).  We
        #    treat the packed batch as B = 1.
        # ------------------------------------------------------------------
        pos_vectors: list[torch.Tensor] = [
            torch.arange(l, dtype=torch.long) for l in seq_lens
        ]
        position_ids = torch.cat(pos_vectors, dim=0).unsqueeze(0)  # (1, total_len)

        # ------------------------------------------------------------------
        # 5. Assemble batch dict – attention_mask holds `cu_seqlens` vector.
        # ------------------------------------------------------------------
        # FlashAttention2 in recent transformers (>=4.40) handles packed sequences via the **position_ids** path.
        # Unfortunately the version bundled in our environment still *requires* a 2-D boolean mask to avoid the
        # scalar-padding bug shown in `run.log` (see _get_unpad_data → F.pad).  We therefore:
        #   • keep the prefix-sum vector under an auxiliary key so future upgrades can switch back easily, and
        #   • provide a dummy (all-True) mask of shape (1, T) that satisfies the older code path.

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            # Intentionally omit / set None so flash-attn var-len path is used
            "attention_mask": None,
            "position_ids": position_ids,
            # Debug/optional: provide cu_seqlens to downstream code (trainer will strip before model)
            "cu_seqlens": cu_seqlens,
        }

        # ------------------------------------------------------------------
        # 6. Vision tensors (images / grids) – unchanged relative to the
        #    previous implementation.
        # ------------------------------------------------------------------
        pixel_values_list = [
            ins["pixel_values"]
            for ins in instances
            if "pixel_values" in ins and ins["pixel_values"] is not None
        ]

        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

            # Ensure image_grid_thw is present for each pixel_values
            grid_thw_list = [
                ins["image_grid_thw"]
                for ins in instances
                if "image_grid_thw" in ins and ins["image_grid_thw"] is not None
            ]

            if not grid_thw_list or len(grid_thw_list) != len(pixel_values_list):
                raise ValueError(
                    "pixel_values present but missing or inconsistent image_grid_thw. "
                    "Both must be provided together."
                )

            batch["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)

            # Track image counts per sample for compatibility with utilities
            batch["image_counts_per_sample"] = [
                ins["pixel_values"].shape[0]
                if "pixel_values" in ins and ins["pixel_values"] is not None
                else 0
                for ins in instances
            ]
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
            batch["image_counts_per_sample"] = [0] * len(instances)

        # ------------------------------------------------------------------
        # 7. Keep ground-truth objects (list per sample).
        # ------------------------------------------------------------------
        batch["ground_truth_objects"] = [
            ins["ground_truth_objects"] if "ground_truth_objects" in ins else []
            for ins in instances
        ]

        # Add adjusted teacher-student spans to batch
        batch["teacher_assistant_spans"] = adjusted_teacher_spans
        batch["student_assistant_spans"] = adjusted_student_spans

        # Extra safety: every new sequence must start with 0 in position_ids
        start_indices = cu_seqlens[:-1]
        if not torch.all(position_ids[0, start_indices] == 0):
            raise AssertionError(
                "PackedDataCollator: position_ids do not reset to 0 at sequence starts"
            )

        return batch


def create_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    collator_type: str,
) -> Any:
    """
    Create a data collator based on the specified type.

    Args:
        tokenizer: The tokenizer to use
        collator_type: Type of collator ("standard" or "packed")

    Returns:
        Data collator instance

    Raises:
        ValueError: If an unknown collator type is specified
    """
    if collator_type == "standard":
        return StandardDataCollator(tokenizer=tokenizer)
    elif collator_type in {"packed", "flattened"}:
        return PackedDataCollator(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown collator_type: {collator_type}")


# ---------------------------------------------------------------------------
# Alias for clarity – official docs often call this strategy *flattened*.
# Keeping both names avoids breaking existing configs.
# ---------------------------------------------------------------------------

FlattenedDataCollator = PackedDataCollator  # backward-compatible alias
