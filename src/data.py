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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.chat_processor import ChatProcessor
from src.config import config

# Get the debug logger from losses.py
from src.logger_utils import get_data_logger
from src.schema import assert_collated_batch  # Runtime batch validation
from src.tokens import SpecialTokens
from src.utils import IGNORE_INDEX

logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class BBUDataset(Dataset):
    """Dataset using UnifiedPreprocessor for clean data processing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Qwen2VLImageProcessor,
        data_path: str,
    ) -> None:
        """
        Initialize BBU Dataset using global configuration.
        All configuration values are accessed from the global config singleton.
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_path = data_path

        # Load data
        self.data = self._load_data()

        # Load candidates if enabled
        self.candidates = None
        if config.use_candidates and config.candidates_file:
            self.candidates = self._load_candidates()

        # Initialize chat processor - no config parameters needed
        self.chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
        )

        # Initialize special tokens first (needed for validation)
        self.tokens = SpecialTokens()

        # Calculate sequence lengths for optimization
        self._sequence_lengths = None
        # Note: calculate_lengths feature removed for simplicity

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

        # Determine minimum image requirement based on data type
        is_training = "train" in self.data_path.lower()

        # Dynamic teacher sampling ensures that additional teacher images
        # will be injected *after* this validation step. Therefore, for
        # target-only samples during training we cannot enforce the same
        # strict ≥2 images rule at this stage – those extra images are not
        # present yet.  We relax the check to require only that each raw
        # sample contains at least one target image.  The composite sample
        # constructed in __getitem__ will always exceed the original 2-image
        # threshold once the teachers have been added.

        min_images = 1  # Always at least one image must be present

        logger.debug(
            f"🔍 Validating simplified JSONL samples with minimum {min_images} images ({'training' if is_training else 'validation'} mode)"
        )

        for idx, sample in enumerate(raw_data):
            # Basic structure validation
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} is not a dictionary")

            # ------------------------------------------------------
            # Two acceptable formats:
            #   A) Few-shot  → keys [examples, target]
            #   B) Target-only → keys [images, objects] (dynamic pairing)
            # ------------------------------------------------------

            has_examples = "examples" in sample and "target" in sample
            has_target_only = (
                "images" in sample and "objects" in sample and "target" not in sample
            )

            if not (has_examples or has_target_only):
                raise ValueError(
                    f"Sample {idx} must contain either ('examples'+'target') or ('images'+'objects') keys"
                )

            total_images = 0

            if has_examples:
                examples = sample["examples"]
                target = sample["target"]

                # Validate examples structure
                if not isinstance(examples, list) or len(examples) == 0:
                    raise ValueError(f"Sample {idx} has empty or invalid examples")

                # Validate target structure
                if not isinstance(target, dict):
                    raise ValueError(f"Sample {idx} has invalid target structure")

                # --- count & validate images in examples ---
                for ex_idx, example in enumerate(examples):
                    if not isinstance(example, dict):
                        raise ValueError(
                            f"Sample {idx}, example {ex_idx} is not a dictionary"
                        )
                    if "images" not in example or "objects" not in example:
                        raise ValueError(
                            f"Sample {idx}, example {ex_idx} missing 'images' or 'objects' key"
                        )

                    example_images = example["images"]
                    if not isinstance(example_images, list) or len(example_images) == 0:
                        raise ValueError(
                            f"Sample {idx}, example {ex_idx} has empty or invalid images"
                        )
                    total_images += len(example_images)

                # --- validate target ---
                if "images" not in target or "objects" not in target:
                    raise ValueError(
                        f"Sample {idx} target missing 'images' or 'objects' key"
                    )

                target_images = target["images"]
                if not isinstance(target_images, list) or len(target_images) == 0:
                    raise ValueError(f"Sample {idx} target has empty or invalid images")

                total_images += len(target_images)

            else:  # target-only format
                if "images" not in sample or "objects" not in sample:
                    raise ValueError(
                        f"Sample {idx} must contain 'images' and 'objects'"
                    )

                images = sample["images"]
                if not isinstance(images, list) or len(images) == 0:
                    raise ValueError(f"Sample {idx} has empty or invalid images list")

                total_images = len(images)

            # --- minimum image requirement ---
            if total_images < min_images:
                raise ValueError(
                    f"Sample {idx} has only {total_images} total images, "
                    f"but minimum {min_images} required for {'training' if is_training else 'validation'}"
                )

            # --- basic box/desc validation ---
            objs_to_check = []
            if has_examples:
                for example in sample["examples"]:
                    objs_to_check.extend(example["objects"])
                objs_to_check.extend(sample["target"]["objects"])
            else:
                objs_to_check.extend(sample["objects"])

            for obj in objs_to_check:
                if not (isinstance(obj, dict) and "box" in obj and "desc" in obj):
                    raise ValueError(
                        f"Sample {idx} contains malformed object entry {obj}"
                    )

            valid_samples.append(sample)

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"❌ CRITICAL ERROR: No valid samples found in {self.data_path}!\n"
                f"   All samples failed validation. Check your data format:\n"
                f"   - Must have 'examples' and 'target' keys\n"
                f"   - Examples and target must have 'images' and 'objects' lists\n"
                f"   - Minimum {min_images} total images per sample"
            )

        validation_ratio = len(valid_samples) / len(raw_data)
        if validation_ratio < 0.8:  # Less than 80% valid samples
            logger.debug(
                f"⚠️ WARNING: Only {validation_ratio:.1%} of samples passed validation!"
            )
            logger.debug("   Consider reviewing your data quality and format.")

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

        # ----------------------------------------------------------
        # Dynamic teacher injection for *target-only* samples.
        # If the sample already contains teachers → use as-is.
        # Otherwise sample `self._num_teachers` other items uniformly
        # at random **every call** (different epoch ⇒ new pairing).
        # ----------------------------------------------------------

        if "examples" not in raw_sample and self._num_teachers > 0:
            # Build list of unique teacher indices (exclude self).
            pool = list(range(len(self.data)))
            pool.remove(idx)
            if len(pool) < self._num_teachers:
                raise ValueError(
                    "Not enough distinct samples to draw dynamic teachers from."
                )

            teacher_indices = random.sample(pool, self._num_teachers)

            examples = []
            for t_idx in teacher_indices:
                t_sample = self.data[t_idx]
                # Support both formats when picking teacher
                if "images" in t_sample and "objects" in t_sample:
                    t_images = t_sample["images"]
                    t_objects = t_sample["objects"]
                else:
                    # Few-shot style – take the *target* part
                    t_images = t_sample["target"]["images"]
                    t_objects = t_sample["target"]["objects"]

                examples.append({"images": t_images, "objects": t_objects})

            composite_sample = {"examples": examples, "target": raw_sample}
            sample_for_processor = composite_sample
        else:
            # Already in multi-round style or teacher count == 0
            sample_for_processor = raw_sample

        processed_data = self.chat_processor.process_sample(sample_for_processor)

        return processed_data

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
    Standard data collator optimized for flash attention and memory efficiency.

    Uses padding to batch max length but with optimized memory management
    and flash attention compatibility.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self, instances: Sequence[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """
        Collate batch with optimized memory management for flash attention.

        Args:
            instances: Sequence of dicts containing input_ids, labels, etc.

        Returns:
            Batch dict optimized for flash attention
        """
        # FAIL-FAST: Process all instances without filtering or fallbacks
        for i, instance in enumerate(instances):
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                # Log vision token information for debugging
                if (
                    "image_grid_thw" in instance
                    and instance["image_grid_thw"] is not None
                ):
                    grid_thw = instance["image_grid_thw"]
                    merge_size = 2  # Default merge size for Qwen2.5-VL
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

        # Process all instances - no filtering, no fallbacks

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

        # 5. Fill padded tensors
        for i, (input_seq, label_seq) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = input_seq.shape[-1]
            padded_input_ids[i, :seq_len] = input_seq
            padded_labels[i, :seq_len] = label_seq
            attention_mask[i, :seq_len] = True

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

        # 7. Handle position_ids if provided
        if any(pos_ids is not None for pos_ids in position_ids_list):
            padded_position_ids_list: List[torch.Tensor] = []
            for pos_ids in position_ids_list:
                if pos_ids is not None:
                    seq_len = pos_ids.shape[-1]
                    padded_pos = torch.zeros(
                        (3, 1, batch_max_length),
                        dtype=pos_ids.dtype,
                    )
                    padded_pos[:, :, :seq_len] = pos_ids
                else:
                    padded_pos = torch.zeros(
                        (3, 1, batch_max_length),
                        dtype=torch.long,
                    )
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

        # Fail-fast shape validation (raises AssertionError on mismatch)
        assert_collated_batch(batch)

        return batch


def create_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_total_length: Optional[int] = None,
    collator_type: str = "flattened",
    **kwargs: Any,
) -> Any:
    """
    Create a data collator based on the specified type.

    Args:
        tokenizer: The tokenizer to use
        max_total_length: Maximum total sequence length for packed sequences (flattened only)
        collator_type: Type of collator ("flattened" | "standard")
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Data collator instance
    """
    if collator_type == "standard":
        return StandardDataCollator(
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown collator_type: {collator_type}")
