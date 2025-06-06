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
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from src.chat_processor import ChatProcessor
from src.config.base import Config
from src.tokens import SpecialTokens
from src.utils import IGNORE_INDEX


# Get the debug logger from losses.py
def get_debug_logger():
    """Get the debug logger for data processing."""
    return logging.getLogger("qwen_debug")


data_logger = get_debug_logger()

# Global sample logger for debugging
_sample_logger = None


def get_sample_logger():
    """Get or create sample logger for debugging."""
    global _sample_logger
    if _sample_logger is None:
        import logging

        _sample_logger = logging.getLogger("sample_debug")
    return _sample_logger


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class BBUDataset(Dataset):
    """Dataset using UnifiedPreprocessor for clean data processing."""

    def __init__(
        self,
        config: Config,
        tokenizer,
        image_processor,
        data_path: str,
    ):
        """
        Initialize BBU Dataset with explicit configuration.
        No defaults allowed - all parameters must be specified in config.
        """
        # EXPLICIT: No defaults - config must specify all required fields
        required_fields = [
            "data_root",
            "model_max_length",
            "use_candidates",
            "candidates_file",
        ]
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Config must specify '{field}' - no defaults allowed")

        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_path = data_path

        # Use explicit config values - no getattr defaults
        self.data_root = config.data_root
        self.model_max_length = config.model_max_length
        self.use_candidates = config.use_candidates
        self.candidates_file = config.candidates_file

        # Load data
        self.data = self._load_data()

        # Load candidates if enabled
        self.candidates = None
        if self.use_candidates and self.candidates_file:
            self.candidates = self._load_candidates()

        # Initialize chat processor
        self.chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            data_root=self.data_root,
            model_max_length=self.model_max_length,
            use_candidates=self.use_candidates,
            candidates_file=self.candidates_file,
        )

        # Initialize special tokens first (needed for validation)
        self.tokens = SpecialTokens()

        # Calculate sequence lengths for optimization
        self._sequence_lengths = None
        if hasattr(config, "calculate_lengths") and config.calculate_lengths:
            self._calculate_sequence_lengths()

    def _validate_and_filter_samples(self, raw_data: List[Dict]) -> List[Dict]:
        """Validate and filter samples with strict requirements."""
        valid_samples = []

        # Determine minimum image requirement based on data type
        is_training = "train" in self.data_path.lower()
        min_images = 2 if is_training else 1

        print(
            f"🔍 Validating simplified JSONL samples with minimum {min_images} images ({'training' if is_training else 'validation'} mode)"
        )

        for idx, sample in enumerate(raw_data):
            # Basic structure validation
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} is not a dictionary")

            if "examples" not in sample:
                raise ValueError(f"Sample {idx} missing 'examples' key")

            if "target" not in sample:
                raise ValueError(f"Sample {idx} missing 'target' key")

            examples = sample["examples"]
            target = sample["target"]

            # Validate examples structure
            if not isinstance(examples, list) or len(examples) == 0:
                raise ValueError(f"Sample {idx} has empty or invalid examples")

            # Validate target structure
            if not isinstance(target, dict):
                raise ValueError(f"Sample {idx} has invalid target structure")

            # Count total images
            total_images = 0

            # Validate examples
            for ex_idx, example in enumerate(examples):
                if not isinstance(example, dict):
                    raise ValueError(
                        f"Sample {idx}, example {ex_idx} is not a dictionary"
                    )

                if "images" not in example:
                    raise ValueError(
                        f"Sample {idx}, example {ex_idx} missing 'images' key"
                    )

                if "objects" not in example:
                    raise ValueError(
                        f"Sample {idx}, example {ex_idx} missing 'objects' key"
                    )

                example_images = example["images"]
                if not isinstance(example_images, list) or len(example_images) == 0:
                    raise ValueError(
                        f"Sample {idx}, example {ex_idx} has empty or invalid images"
                    )

                total_images += len(example_images)

            # Validate target
            if "images" not in target:
                raise ValueError(f"Sample {idx} target missing 'images' key")

            if "objects" not in target:
                raise ValueError(f"Sample {idx} target missing 'objects' key")

            target_images = target["images"]
            if not isinstance(target_images, list) or len(target_images) == 0:
                raise ValueError(f"Sample {idx} target has empty or invalid images")

            total_images += len(target_images)

            # STRICT REQUIREMENT: Minimum image count
            if total_images < min_images:
                raise ValueError(
                    f"Sample {idx} has only {total_images} total images, "
                    f"but minimum {min_images} required for {'training' if is_training else 'validation'}"
                )

            # Validate objects structure in examples
            for ex_idx, example in enumerate(examples):
                objects = example["objects"]
                if not isinstance(objects, list):
                    raise ValueError(
                        f"Sample {idx}, example {ex_idx} objects is not a list"
                    )

                for obj_idx, obj in enumerate(objects):
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"Sample {idx}, example {ex_idx}, object {obj_idx} is not a dictionary"
                        )

                    if "box" not in obj or "desc" not in obj:
                        raise ValueError(
                            f"Sample {idx}, example {ex_idx}, object {obj_idx} missing required keys"
                        )

            # Validate objects structure in target
            target_objects = target["objects"]
            if not isinstance(target_objects, list):
                raise ValueError(f"Sample {idx} target objects is not a list")

            for obj_idx, obj in enumerate(target_objects):
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Sample {idx}, target object {obj_idx} is not a dictionary"
                    )

                if "box" not in obj or "desc" not in obj:
                    raise ValueError(
                        f"Sample {idx}, target object {obj_idx} missing required keys"
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
            print(
                f"⚠️ WARNING: Only {validation_ratio:.1%} of samples passed validation!"
            )
            print("   Consider reviewing your data quality and format.")

        return valid_samples

    def _calculate_sequence_lengths(self):
        """Pre-calculate sequence lengths for optimization."""
        print("📏 Pre-calculating sequence lengths...")
        lengths = []

        for i in range(min(100, len(self.data))):  # Sample first 100 for estimation
            sample = self._get_item(i)
            if "input_ids" in sample:
                lengths.append(sample["input_ids"].shape[-1])
            else:
                lengths.append(8192)  # Default estimate

        self._sequence_lengths = lengths
        avg_length = sum(lengths) / len(lengths) if lengths else 8192
        print(f"📏 Average sequence length: {avg_length:.0f} tokens")

    @property
    def sequence_lengths(self):
        """Get sequence lengths for optimization."""
        return self._sequence_lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with strict validation - no fallbacks."""
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process a single sample using ChatProcessor."""
        sample = self.data[idx]

        # Process using the chat processor
        result = self.chat_processor.process_sample(sample)
        data_logger.debug(f"✅ Successfully processed sample {idx}")

        # Validate result
        if "input_ids" not in result:
            raise ValueError("Missing input_ids in preprocessed result")

        data_logger.debug(
            f"✅ Sample {idx} processed (automatic memory management enabled)"
        )

        return result

    def _load_data(self) -> List[Dict]:
        """Load and validate data from JSONL file."""
        # Load raw data
        raw_data = read_jsonl(self.data_path)
        print(f"📊 Loaded {len(raw_data)} raw samples from {self.data_path}")

        # Basic validation and filtering
        validated_data = self._validate_and_filter_samples(raw_data)
        print(f"📊 After validation: {len(validated_data)} valid samples")

        return validated_data

    def _load_candidates(self) -> Optional[Dict]:
        """Load candidate phrases if available."""
        if not self.candidates_file:
            return None

        try:
            import json

            with open(self.candidates_file, "r") as f:
                candidates = json.load(f)
            print(
                f"📊 Loaded {len(candidates)} candidate phrases from {self.candidates_file}"
            )
            return candidates
        except Exception as e:
            print(f"⚠️ Failed to load candidates from {self.candidates_file}: {e}")
            return None


def extract_ground_truth_objects_from_conversation(
    conversations: List[Dict[str, str]],
) -> Optional[List[Dict]]:
    """
    Extract ground truth objects from conversation format.

    Prioritizes Qwen2.5-VL special token format, falls back to legacy JSON format.
    """
    for conv in conversations:
        role = conv.get("role", conv.get("from", ""))
        content = conv.get("content", conv.get("value", ""))

        if role == "assistant" and content:
            # First, try special token format (preferred)
            objects = _extract_objects_with_special_tokens(content)
            if objects:
                return objects

            # Fallback to legacy JSON format for backward compatibility
            try:
                # Try to parse as JSON array
                objects = json.loads(content)
                if isinstance(objects, list):
                    validated_objects = []
                    for obj in objects:
                        if isinstance(obj, dict) and "bbox" in obj and "desc" in obj:
                            # Validate bbox format
                            bbox = obj["bbox"]
                            if isinstance(bbox, list) and len(bbox) == 4:
                                validated_objects.append(
                                    {"bbox": bbox, "description": obj["desc"]}
                                )

                    if validated_objects:
                        return validated_objects
            except (json.JSONDecodeError, KeyError, TypeError):
                # If JSON parsing fails, try regex extraction
                return _extract_objects_with_regex(content)

    return None


def _extract_objects_with_special_tokens(content: str) -> Optional[List[Dict]]:
    """
    Extract objects from Qwen2.5-VL special token format.

    Format: <|object_ref_start|>description<|object_ref_end|><|box_start|>(x1, y1), (x2, y2)<|box_end|>
    """
    objects = []

    # Pattern to match special token format
    pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\(([^)]+)\),\s*\(([^)]+)\)<\|box_end\|>"

    matches = re.findall(pattern, content, re.DOTALL)

    for desc, coords1, coords2 in matches:
        try:
            # Parse coordinates: (x1, y1), (x2, y2)
            x1, y1 = map(float, coords1.split(", "))
            x2, y2 = map(float, coords2.split(", "))
            bbox = [x1, y1, x2, y2]
            objects.append({"bbox": bbox, "description": desc.strip()})
        except (ValueError, IndexError):
            continue

    return objects if objects else None


def _extract_objects_with_regex(content: str) -> Optional[List[Dict]]:
    """Extract objects using regex when JSON parsing fails (legacy fallback)."""
    objects = []

    # Pattern for format: {bbox:[x1,y1,x2,y2],desc:'...'}
    pattern = r'\{\s*bbox\s*:\s*\[([^\]]+)\]\s*,\s*desc\s*:\s*[\'"]([^\'\"]+)[\'"]\s*\}'

    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

    for bbox_str, desc in matches:
        try:
            # Parse coordinates
            coords = [float(x.strip()) for x in bbox_str.split(",")]
            if len(coords) == 4:
                objects.append({"bbox": coords, "description": desc.strip()})
        except ValueError:
            continue

    return objects if objects else None


@dataclass
class StandardDataCollator:
    """
    Standard data collator optimized for flash attention and memory efficiency.

    Uses padding to batch max length but with optimized memory management
    and flash attention compatibility.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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
                    data_logger.debug(
                        f"Sample {i}: {pre_merge_tokens} pre-merge → {total_final_tokens} final tokens"
                    )
                else:
                    vision_tokens = instance["pixel_values"].shape[0]
                    data_logger.debug(
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
        data_logger.info(f"📊 BATCH SEQUENCE INFO:")
        data_logger.info(f"   Batch size: {batch_size}")
        data_logger.info(f"   Individual lengths: {sequence_lengths}")
        data_logger.info(f"   Max length in batch: {batch_max_length}")
        data_logger.info(f"   Min length in batch: {min(sequence_lengths)}")
        data_logger.info(f"   Total tokens (sum): {sum(sequence_lengths)}")

        # Log vision token information
        total_images = sum(info["image_count"] for info in vision_info)
        data_logger.debug(f"🖼️ VISION TOKEN INFO:")
        data_logger.debug(f"   Total images in batch: {total_images}")
        data_logger.debug(
            f"   Samples with images: {sum(1 for info in vision_info if info['has_images'])}"
        )
        for i, info in enumerate(vision_info):
            if info["has_images"]:
                data_logger.debug(
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
        data_logger.debug(f"🎯 ATTENTION MASK INFO:")
        data_logger.debug(f"   Attention mask shape: {attention_mask.shape}")
        data_logger.debug(f"   Attention mask dtype: {attention_mask.dtype}")
        mask_lengths = attention_mask.sum(dim=-1).tolist()
        data_logger.debug(f"   Attention mask lengths: {mask_lengths}")
        data_logger.debug(f"   Uniform attention masks: {len(set(mask_lengths)) == 1}")

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

        # 8. Handle images efficiently
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance and instance["pixel_values"].shape[0] > 0
        ]

        if images:
            # Concatenate valid images
            batch["pixel_values"] = torch.cat(images, dim=0)

            # Ensure bf16 precision for pixel_values
            if batch["pixel_values"].dtype != torch.bfloat16:
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
                data_logger.debug(f"🔧 Converted pixel_values to bf16")

            # Handle image grid info
            grid_thw_list = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
                and instance["image_grid_thw"].shape[0] > 0
            ]

            if grid_thw_list:
                batch["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)
                data_logger.debug(f"🖼️ Image grid info: {batch['image_grid_thw'].shape}")
            else:
                # Remove pixel_values if no valid grid_thw
                del batch["pixel_values"]
                data_logger.warning(
                    "⚠️ Removed pixel_values due to invalid image_grid_thw"
                )

        return batch


def create_data_collator(
    tokenizer,
    max_total_length: Optional[int] = None,
    collator_type: str = "flattened",
    **kwargs,
):
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
