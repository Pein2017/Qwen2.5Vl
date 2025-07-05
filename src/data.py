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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.chat_processor import ChatProcessor
from src.config import config

# Get the debug logger from losses.py
from src.logger_utils import get_data_logger
from src.utils.schema import ChatProcessorOutput, assert_collated_batch
from src.teacher_pool import TeacherPoolManager, create_teacher_pool_manager
from src.utils.tokens import SpecialTokens
from src.utils.utils import IGNORE_INDEX

logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
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
    ):
        """
        Initialize BBU dataset.

        Args:
            data_path: Path to JSONL data file
            chat_processor: Chat processor instance
            teacher_pool_manager: Manager for teacher examples
            teacher_ratio: Ratio of samples to use teacher examples (0.0 = no teachers)
            is_training: Whether this is a training dataset (affects prompt selection)
        """
        self.data_path = data_path
        self.chat_processor = chat_processor
        self.teacher_pool_manager = teacher_pool_manager
        self.teacher_ratio = teacher_ratio
        self.is_training = is_training

        # --------------------------------------------------------------
        # Optional ChatProcessor context switch.
        # Older versions exposed ``set_context`` (training/eval) but the
        # streamlined implementation used in this project no longer needs
        # it.  To remain backward-compatible we *only* invoke the method when
        # it actually exists, avoiding AttributeError in worker processes.
        # --------------------------------------------------------------
        context = "training" if is_training else "evaluation"
        if hasattr(self.chat_processor, "set_context"):
            self.chat_processor.set_context(context)  # type: ignore[attr-defined]
        else:
            logger.debug(
                "ChatProcessor has no `set_context`; proceeding without context flag."
            )

        # Load data
        self.data = self._load_data()

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Dataset mode: {'training' if is_training else 'evaluation'}")
        logger.info(f"Teacher ratio: {teacher_ratio}")
        if teacher_pool_manager:
            logger.info(f"Teacher pool size: {len(teacher_pool_manager)}")

        # Load candidates if enabled
        self.candidates = None
        if config.use_candidates and config.candidates_file:
            self.candidates = self._load_candidates()

        # Initialize special tokens first (needed for validation)
        self.tokens = SpecialTokens()

        # Calculate sequence lengths for optimization
        self._sequence_lengths = None
        # Note: calculate_lengths feature removed for simplicity
        
        # Initialize teacher assignment tracking
        self._teacher_assignment_stats = {
            'total_samples': 0,
            'samples_with_teacher': 0,
            'samples_without_teacher': 0
        }

        # ---------------- Dynamic teacher sampling ----------------
        # Dynamic teacher sampling: require explicit configuration.
        if not hasattr(config, "num_teacher_samples"):
            raise AttributeError(
                "'num_teacher_samples' must be specified in YAML configuration"
            )

        self._num_teachers = int(config.num_teacher_samples)
        # Disable dynamic teacher injection for validation data (zero-shot eval)
        if "val" in self.data_path.lower():
            logger.info(
                "Validation dataset detected, disabling teacher sampling for zero-shot evaluation."
            )
            self._num_teachers = 0

        # For training: mix teacher-student samples with single-shot samples
        # This helps the model learn both multi-chat and single-shot patterns
        if "train" in self.data_path.lower():
            self.teacher_ratio = getattr(config, "teacher_ratio", 0.7)
            logger.info(f"Training dataset: teacher ratio set to {self.teacher_ratio}")
        else:
            self.teacher_ratio = 0.0

        # Initialize teacher pool manager only if not provided
        if self._num_teachers > 0 and teacher_pool_manager is None:
            teacher_pool_manager = create_teacher_pool_manager()

        # Assign (may be None if no teachers / eval mode)
        self.teacher_pool_manager = teacher_pool_manager

        if self._num_teachers > 0 and self.teacher_pool_manager is not None:
            logger.info(
                f"Teacher pool manager active with {len(self.teacher_pool_manager)} teachers"
            )
        elif self._num_teachers > 0 and self.teacher_pool_manager is None:
            logger.warning(
                "Teacher pool manager unavailable, disabling teacher sampling"
            )
            self._num_teachers = 0

    @property
    def data_root(self) -> str:
        """Get data root from global config."""
        return config.data_root

    @property
    def model_max_length(self) -> int:
        """Get model max length from global config."""
        return config.max_total_length

    @property
    def use_candidates(self) -> bool:
        """Get use candidates flag from global config."""
        return config.use_candidates

    @property
    def candidates_file(self) -> str:
        """Get candidates file path from global config."""
        return config.candidates_file

    def _validate_and_filter_samples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and filter samples with strict requirements."""
        valid_samples = []
        is_training = "train" in self.data_path.lower()
        min_images = 1  # At least one image must be present

        logger.debug(
            f"🔍 Validating samples with minimum {min_images} images ({'training' if is_training else 'validation'} mode)"
        )

        for idx, sample in enumerate(raw_data):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} is not a dictionary")

            # Count total images and validate structure
            total_images = 0
            objects_to_validate = []

            # Handle different sample formats
            if "teachers" in sample and "student" in sample:
                # New teacher/student format
                teachers = sample["teachers"]
                student = sample["student"]

                if not isinstance(teachers, list):
                    raise ValueError(f"Sample {idx} 'teachers' must be a list")
                if not isinstance(student, dict):
                    raise ValueError(f"Sample {idx} 'student' must be a dict")

                # Validate teachers
                for t_idx, teacher in enumerate(teachers):
                    if (
                        not isinstance(teacher, dict)
                        or "images" not in teacher
                        or "objects" not in teacher
                    ):
                        raise ValueError(
                            f"Sample {idx} teacher[{t_idx}] missing 'images' or 'objects'"
                        )
                    if (
                        not isinstance(teacher["images"], list)
                        or len(teacher["images"]) == 0
                    ):
                        raise ValueError(
                            f"Sample {idx} teacher[{t_idx}] has empty images"
                        )
                    total_images += len(teacher["images"])
                    objects_to_validate.extend(teacher["objects"])

                # Validate student
                if "images" not in student or "objects" not in student:
                    raise ValueError(
                        f"Sample {idx} student missing 'images' or 'objects'"
                    )
                if (
                    not isinstance(student["images"], list)
                    or len(student["images"]) == 0
                ):
                    raise ValueError(f"Sample {idx} student has empty images")
                total_images += len(student["images"])
                objects_to_validate.extend(student["objects"])

            elif "examples" in sample and "target" in sample:
                # Legacy examples/target format
                examples = sample["examples"]
                target = sample["target"]

                if not isinstance(examples, list):
                    raise ValueError(f"Sample {idx} 'examples' must be a list")
                if not isinstance(target, dict):
                    raise ValueError(f"Sample {idx} 'target' must be a dict")

                # Validate examples
                for e_idx, example in enumerate(examples):
                    if (
                        not isinstance(example, dict)
                        or "images" not in example
                        or "objects" not in example
                    ):
                        raise ValueError(
                            f"Sample {idx} example[{e_idx}] missing 'images' or 'objects'"
                        )
                    if (
                        not isinstance(example["images"], list)
                        or len(example["images"]) == 0
                    ):
                        raise ValueError(
                            f"Sample {idx} example[{e_idx}] has empty images"
                        )
                    total_images += len(example["images"])
                    objects_to_validate.extend(example["objects"])

                # Validate target
                if "images" not in target or "objects" not in target:
                    raise ValueError(
                        f"Sample {idx} target missing 'images' or 'objects'"
                    )
                if not isinstance(target["images"], list) or len(target["images"]) == 0:
                    raise ValueError(f"Sample {idx} target has empty images")
                total_images += len(target["images"])
                objects_to_validate.extend(target["objects"])

            elif "images" in sample and "objects" in sample:
                # Simple format (will have teachers added later)
                if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
                    raise ValueError(f"Sample {idx} has empty images")
                if not isinstance(sample["objects"], list):
                    raise ValueError(f"Sample {idx} has invalid objects")
                total_images = len(sample["images"])
                objects_to_validate.extend(sample["objects"])

            else:
                raise ValueError(
                    f"Sample {idx} has invalid format. Expected 'teachers'+'student', "
                    f"'examples'+'target', or 'images'+'objects' keys. "
                    f"Found keys: {list(sample.keys())}"
                )

            # Check minimum image requirement
            if total_images < min_images:
                raise ValueError(
                    f"Sample {idx} has only {total_images} images, "
                    f"minimum {min_images} required"
                )

            # Validate object structure
            for obj_idx, obj in enumerate(objects_to_validate):
                if not (isinstance(obj, dict) and "bbox_2d" in obj and "desc" in obj):
                    raise ValueError(
                        f"Sample {idx} object[{obj_idx}] missing 'bbox_2d' or 'desc' keys"
                    )

            valid_samples.append(sample)

        if len(valid_samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.data_path}")

        validation_ratio = len(valid_samples) / len(raw_data)
        if validation_ratio < 0.8:
            logger.debug(
                f"⚠️ WARNING: Only {validation_ratio:.1%} of samples passed validation"
            )

        return valid_samples

    def _calculate_sequence_lengths(self):
        """Pre-calculate sequence lengths for optimization."""
        logger.debug("📏 Pre-calculating sequence lengths...")
        lengths = []

        for i in range(min(100, len(self.data))):  # Sample first 100 for estimation
            sample = self._get_item(i)
            if "input_ids" in sample:
                lengths.append(sample["input_ids"].shape[-1])
            else:
                lengths.append(8192)  # Default estimate

        self._sequence_lengths = lengths
        avg_length = sum(lengths) / len(lengths) if lengths else 8192
        logger.debug(f"📏 Average sequence length: {avg_length:.0f} tokens")

    @property
    def sequence_lengths(self) -> List[int]:
        """Get sequence lengths for optimization."""
        return self._sequence_lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with strict validation - no fallbacks."""
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Internal getter to handle data processing and GT extraction."""
        raw_sample = self.data[idx]

        # Create structured sample with teacher/student format
        structured_sample = self._create_structured_sample(raw_sample, idx)

        # Process through chat processor
        processed_data = self.chat_processor.process_sample(structured_sample)

        return processed_data

    def _create_structured_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """
        Create structured sample in consistent teacher/student format.

        Args:
            raw_sample: Raw sample from JSONL
            idx: Sample index for reproducible teacher sampling

        Returns:
            Dict with "teachers" (List[Sample]) and "student" (Sample) keys
        """
        # Evaluation mode: return single student sample (no teachers)
        if not self.is_training:
            if "teachers" in raw_sample and "student" in raw_sample:
                return {"teachers": [], "student": raw_sample["student"]}
            elif "examples" in raw_sample and "target" in raw_sample:
                return {"teachers": [], "student": raw_sample["target"]}
            else:
                return {"teachers": [], "student": raw_sample}

        # Training mode: handle teacher sampling
        teachers = []
        student = raw_sample

        # Case 1: Sample already has teacher/student structure
        if "teachers" in raw_sample and "student" in raw_sample:
            teachers = raw_sample["teachers"]
            student = raw_sample["student"]

        # Case 2: Sample has legacy examples/target structure
        elif "examples" in raw_sample and "target" in raw_sample:
            teachers = raw_sample["examples"]
            student = raw_sample["target"]

        # Case 3: Simple sample - add teachers from pool if available
        else:
            teachers = self._sample_teachers_for_student(raw_sample, idx)
            student = raw_sample

        return {"teachers": teachers, "student": student}

    def _sample_teachers_for_student(
        self, student_sample: Dict[str, Any], idx: int
    ) -> List[Dict[str, Any]]:
        """
        Sample teacher examples for a student sample.

        Args:
            student_sample: The student sample to create teachers for
            idx: Sample index for reproducible sampling

        Returns:
            List of teacher samples
        """
        # No teachers in evaluation mode or if teacher pool unavailable
        if (
            not self.is_training
            or self._num_teachers == 0
            or self.teacher_pool_manager is None
        ):
            return []

        # Decide whether to use teachers based on teacher ratio
        use_teachers = random.random() < self.teacher_ratio
        
        # Track teacher assignment statistics
        self._teacher_assignment_stats['total_samples'] += 1
        if use_teachers:
            self._teacher_assignment_stats['samples_with_teacher'] += 1
        else:
            self._teacher_assignment_stats['samples_without_teacher'] += 1
        
        # Log statistics periodically (every 100 samples)
        if self._teacher_assignment_stats['total_samples'] % 100 == 0:
            total = self._teacher_assignment_stats['total_samples']
            with_teacher = self._teacher_assignment_stats['samples_with_teacher']
            actual_ratio = with_teacher / total if total > 0 else 0.0
            logger.info(
                f"📊 Teacher Assignment Stats (sample {idx}): "
                f"{with_teacher}/{total} samples with teacher "
                f"(actual ratio: {actual_ratio:.3f}, configured: {self.teacher_ratio:.3f})"
            )
        
        if not use_teachers:
            return []

        # Create reproducible seed for this sample
        epoch_seed = hash((idx, random.getstate()[1][0])) % (2**32)

        # Sample teachers from pool
        multi_chat_sample = self.teacher_pool_manager.create_multi_chat_sample(
            student_sample=student_sample,
            num_teachers=self._num_teachers,
            seed=epoch_seed,
        )

        # Extract teachers from the multi-chat sample
        if "teachers" in multi_chat_sample:
            teachers = multi_chat_sample["teachers"]
        elif "examples" in multi_chat_sample:
            teachers = multi_chat_sample["examples"]
        else:
            teachers = []

        logger.debug(
            f"Sample {idx}: Sampled {len(teachers)} teachers (seed={epoch_seed})"
        )
        return teachers
    
    def get_teacher_assignment_summary(self) -> Dict[str, Any]:
        """Get summary of teacher assignment statistics."""
        stats = self._teacher_assignment_stats
        total = stats['total_samples']
        with_teacher = stats['samples_with_teacher']
        without_teacher = stats['samples_without_teacher']
        
        return {
            'total_samples_processed': total,
            'samples_with_teacher': with_teacher,
            'samples_without_teacher': without_teacher,
            'actual_teacher_ratio': with_teacher / total if total > 0 else 0.0,
            'configured_teacher_ratio': self.teacher_ratio,
            'ratio_accuracy': abs((with_teacher / total) - self.teacher_ratio) if total > 0 else 0.0
        }

    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        # Load raw data
        raw_data = read_jsonl(self.data_path)
        logger.debug(f"📊 Loaded {len(raw_data)} raw samples from {self.data_path}")

        # Basic validation and filtering
        validated_data = self._validate_and_filter_samples(raw_data)
        logger.debug(f"📊 After validation: {len(validated_data)} valid samples")

        return validated_data

    def _load_candidates(self) -> Optional[Dict]:
        """Load candidate phrases if available."""
        if not self.candidates_file:
            return None

        import json

        if not Path(self.candidates_file).exists():
            raise FileNotFoundError(
                f"Candidates file not found: {self.candidates_file}. Failing fast as per project policy."
            )

        try:
            with open(self.candidates_file, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Candidates file {self.candidates_file} is not valid JSON: {e}"
            ) from e

        logger.debug(
            f"📊 Loaded {len(candidates)} candidate phrases from {self.candidates_file}"
        )
        return candidates


def extract_ground_truth_from_sample(
    sample_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: This function is no longer needed as the ChatProcessor
    now handles ground truth extraction and normalization directly.
    Kept for historical reference but should not be used.
    """
    raise DeprecationWarning(
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
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """Collate a batch of :class:`ChatProcessorOutput` or raw dicts."""

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
            # Extract spans from each instance
            teacher_spans = instance.get("teacher_assistant_spans", [])
            student_spans = instance.get("student_assistant_spans", [])

            teacher_spans_batch.append(teacher_spans)
            student_spans_batch.append(student_spans)

        # FAIL-FAST: Process all instances without filtering or fallbacks
        for i, instance in enumerate(instances):
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                # Retrieve merge_size **once** for all samples in this batch
                from src.config import get_config

                merge_size = get_config().merge_size

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
                    vision_tokens = instance["pixel_values"].shape[0]
                    logger.debug(
                        f"Sample {i}: {vision_tokens} vision tokens (no grid_thw)"
                    )

        # 1. Extract sequences
        input_ids_list: List[torch.Tensor] = [
            instance["input_ids"].squeeze() for instance in instances
        ]
        labels_list: List[torch.Tensor] = [
            instance["labels"].squeeze() for instance in instances
        ]
        position_ids_list: List[Optional[torch.Tensor]] = [
            instance.get("position_ids") for instance in instances
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
        padded_input_ids = torch.full(
            (batch_size, batch_max_length),
            self.tokenizer.pad_token_id,
            dtype=input_ids_list[0].dtype,
        )
        padded_labels = torch.full(
            (batch_size, batch_max_length),
            IGNORE_INDEX,
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
            if "pixel_values" in instance and instance["pixel_values"].shape[0] > 0
        ]

        if images:
            # Track image counts per sample for proper extraction during generation
            image_counts_per_sample = []
            for instance in instances:
                if "pixel_values" in instance and instance["pixel_values"].shape[0] > 0:
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

        # Ensure optional keys are always present for schema validation
        batch.setdefault("pixel_values", None)
        batch.setdefault("image_grid_thw", None)

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

    def __call__(self, instances: Sequence[Any]) -> Dict[str, Any]:
        # Convert dataclass inputs to dicts (if needed) early.
        if instances and isinstance(instances[0], ChatProcessorOutput):
            instances = [asdict(ins) for ins in instances]  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Extract teacher-student spans and adjust for packed sequences
        # ------------------------------------------------------------------
        teacher_spans_batch: list[list[tuple[int, int]]] = []
        student_spans_batch: list[list[tuple[int, int]]] = []

        for instance in instances:
            # Extract spans from each instance
            teacher_spans = instance.get("teacher_assistant_spans", [])
            student_spans = instance.get("student_assistant_spans", [])

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
        if any(
            "pixel_values" in ins and ins["pixel_values"] is not None
            for ins in instances
        ):
            batch["pixel_values"] = torch.cat(
                [
                    ins["pixel_values"]
                    for ins in instances
                    if ins.get("pixel_values") is not None
                ],
                dim=0,
            )
            batch["image_grid_thw"] = torch.cat(
                [
                    ins["image_grid_thw"]
                    for ins in instances
                    if ins.get("image_grid_thw") is not None
                ],
                dim=0,
            )

            # Track image counts per sample for compatibility with utilities
            batch["image_counts_per_sample"] = [
                ins["pixel_values"].shape[0]
                if ins.get("pixel_values") is not None
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
            ins.get("ground_truth_objects", []) for ins in instances
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
