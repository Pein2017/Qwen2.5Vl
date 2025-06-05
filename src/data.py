"""
Optimized data handling for Qwen2.5-VL training with flattened sequences.

This module provides:
- BBUDataset: Dataset for multi-round conversation format with <image> tokens
- FlattenedDataCollator: Default collator using packed sequences (like official repo)
- StandardDataCollator: Traditional padding-based collator for compatibility
- Utilities for ground truth extraction from conversation format

Key Features:
- Flattened sequence collation (no padding waste)
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

from src.preprocessing import create_preprocessor
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
    """Dataset for multi-round conversation format with image support."""

    def __init__(self, config, tokenizer, image_processor, data_path: str):
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_path = data_path

        # Load data
        raw_data = read_jsonl(data_path)
        print(f"📊 Loaded {len(raw_data)} raw samples from {data_path}")

        # STRICT VALIDATION: Filter and validate samples
        self.data = self._validate_and_filter_samples(raw_data)
        print(f"📊 After validation: {len(self.data)} valid samples")

        # Create preprocessor
        self.preprocessor = create_preprocessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_root=getattr(config, "data_root", "./"),
            model_max_length=getattr(config, "model_max_length", 8192),
        )

        # Calculate sequence lengths for optimization
        self._sequence_lengths = None
        if hasattr(config, "calculate_lengths") and config.calculate_lengths:
            self._calculate_sequence_lengths()

    def _validate_and_filter_samples(self, raw_data: List[Dict]) -> List[Dict]:
        """Strictly validate and filter samples to ensure data quality."""
        valid_samples = []

        # Determine minimum image requirements based on data split
        is_training = "train" in self.data_path.lower()
        min_images = 2 if is_training else 1

        print(
            f"🔍 Validating samples with minimum {min_images} images ({'training' if is_training else 'validation'} mode)"
        )

        for idx, sample in enumerate(raw_data):
            try:
                # Basic structure validation
                if not isinstance(sample, dict):
                    raise ValueError(f"Sample {idx} is not a dictionary")

                if "conversations" not in sample:
                    raise ValueError(f"Sample {idx} missing 'conversations' key")

                if "images" not in sample:
                    raise ValueError(f"Sample {idx} missing 'images' key")

                conversations = sample["conversations"]
                image_paths = sample["images"]

                # Validate conversations structure
                if not isinstance(conversations, list) or len(conversations) == 0:
                    raise ValueError(f"Sample {idx} has empty or invalid conversations")

                # Validate images structure
                if not isinstance(image_paths, list) or len(image_paths) == 0:
                    raise ValueError(f"Sample {idx} has empty or invalid images list")

                # STRICT REQUIREMENT: Minimum image count
                if len(image_paths) < min_images:
                    raise ValueError(
                        f"Sample {idx} has only {len(image_paths)} images, "
                        f"but minimum {min_images} required for {'training' if is_training else 'validation'}"
                    )

                # Validate conversation format
                if conversations[0].get("role") != "system":
                    raise ValueError(f"Sample {idx} must start with system message")

                # Count and validate <image> tokens
                image_token_count = 0
                for conv_idx, conv in enumerate(conversations):
                    if not isinstance(conv, dict):
                        raise ValueError(
                            f"Sample {idx}, conversation {conv_idx} is not a dictionary"
                        )

                    content = conv.get("content", "")
                    if not isinstance(content, str):
                        raise ValueError(
                            f"Sample {idx}, conversation {conv_idx} has non-string content"
                        )

                    image_token_count += content.count("<image>")

                # STRICT REQUIREMENT: Image token alignment
                if image_token_count != len(image_paths):
                    raise ValueError(
                        f"Sample {idx}: Found {image_token_count} <image> tokens but {len(image_paths)} image paths. "
                        f"Every image must have exactly one <image> token in the conversations."
                    )

                # STRICT REQUIREMENT: Minimum image tokens
                if image_token_count < min_images:
                    raise ValueError(
                        f"Sample {idx}: Found only {image_token_count} <image> tokens, "
                        f"but minimum {min_images} required"
                    )

                # Validate that we have at least one assistant response
                has_assistant = any(
                    conv.get("role") == "assistant" for conv in conversations
                )
                if not has_assistant:
                    raise ValueError(f"Sample {idx} has no assistant responses")

                # Additional validation: Check for proper multi-round structure
                roles = [conv.get("role") for conv in conversations]
                if (
                    len(set(roles)) < 2
                ):  # Should have at least system/user and assistant
                    raise ValueError(
                        f"Sample {idx} has insufficient role diversity: {roles}"
                    )

                valid_samples.append(sample)

            except ValueError as e:
                print(f"⚠️ Skipping invalid sample {idx}: {e}")
                continue
            except Exception as e:
                print(f"❌ Unexpected error validating sample {idx}: {e}")
                continue

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"❌ CRITICAL ERROR: No valid samples found in {self.data_path}!\n"
                f"   All samples failed validation. Check your data format and requirements:\n"
                f"   - Minimum {min_images} images per sample\n"
                f"   - Proper conversation structure with system/user/assistant roles\n"
                f"   - Exact alignment between <image> tokens and image paths"
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
            try:
                sample = self._get_item(i)
                if "input_ids" in sample:
                    lengths.append(sample["input_ids"].shape[-1])
            except Exception:
                lengths.append(2048)  # Default estimate

        self._sequence_lengths = lengths
        avg_length = sum(lengths) / len(lengths) if lengths else 2048
        print(f"📏 Average sequence length: {avg_length:.0f} tokens")

    @property
    def sequence_lengths(self):
        """Get sequence lengths for optimization."""
        return self._sequence_lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with strict validation - no fallbacks."""
        try:
            return self._get_item(idx)
        except Exception as e:
            logger = get_sample_logger()
            logger.error(f"❌ CRITICAL: Failed to process sample {idx}: {e}")
            # DO NOT provide fallback - let the error bubble up to expose data issues
            raise RuntimeError(
                f"❌ SAMPLE PROCESSING FAILED: Sample {idx} could not be processed!\n"
                f"   Error: {e}\n"
                f"   This indicates a critical issue in the data pipeline.\n"
                f"   Fix the data or preprocessing logic instead of masking the error."
            ) from e

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process a single sample from the multi-round conversation format."""
        data_logger.info(f"📊 PROCESSING SAMPLE {idx}")
        sample = self.data[idx]

        # Validate sample format
        if "conversations" not in sample:
            raise ValueError(f"Sample {idx} missing 'conversations' key")
        if "images" not in sample:
            raise ValueError(f"Sample {idx} missing 'images' key")

        conversations = sample["conversations"]
        image_paths = sample["images"]

        data_logger.info(f"   Number of conversations: {len(conversations)}")
        data_logger.info(f"   Number of images: {len(image_paths)}")
        data_logger.info(f"   Image paths: {image_paths}")

        # Log conversation content
        for i, conv in enumerate(conversations):
            role = conv.get("role", "unknown")
            content = conv.get("content", "")
            data_logger.info(f"   Conv {i}: {role} - {len(content)} chars")
            data_logger.info(f"     Content preview: {repr(content[:200])}")
            if "<image>" in content:
                image_count_in_conv = content.count("<image>")
                data_logger.info(f"     Contains {image_count_in_conv} <image> tokens")

        # Validate conversation format
        if not conversations or conversations[0].get("role") != "system":
            raise ValueError(f"Sample {idx} must start with system message")

        # Count <image> tokens in conversations
        image_token_count = 0
        for conv in conversations:
            content = conv.get("content", "")
            image_token_count += content.count("<image>")

        data_logger.info(f"   Total <image> tokens found: {image_token_count}")

        # Validate image alignment
        if image_token_count != len(image_paths):
            data_logger.error(
                f"   ❌ IMAGE MISMATCH: {image_token_count} <image> tokens vs {len(image_paths)} paths"
            )
            raise ValueError(
                f"Sample {idx}: Found {image_token_count} <image> tokens but {len(image_paths)} image paths"
            )

        # Process using the unified preprocessor
        try:
            data_logger.info("🔄 CALLING PREPROCESSOR")
            result = self.preprocessor.process_sample_for_training(sample, idx)

            data_logger.info("✅ PREPROCESSOR COMPLETED")
            data_logger.info(f"   Result keys: {list(result.keys())}")

            # Validate and fix result components
            if "input_ids" not in result:
                raise ValueError("Missing input_ids in preprocessed result")

            input_ids = result["input_ids"]
            if input_ids.numel() == 0:
                raise ValueError("Empty input_ids tensor")

            data_logger.info(f"   Input IDs shape: {input_ids.shape}")
            data_logger.info(
                f"   Input IDs sample (first 50): {input_ids.flatten()[:50].tolist()}"
            )
            data_logger.info(
                f"   Input IDs sample (last 50): {input_ids.flatten()[-50:].tolist()}"
            )

            # Log special tokens from preprocessor
            from .losses import log_special_tokens

            log_special_tokens(
                self.tokenizer, input_ids, f"SAMPLE {idx} POST-PREPROCESSING"
            )

            # Ensure position_ids has correct shape
            if "position_ids" in result:
                pos_ids = result["position_ids"]

                # Handle case where position_ids is None (using official model calculation)
                if pos_ids is None:
                    # Remove position_ids from result - let the model handle it
                    del result["position_ids"]
                else:
                    expected_seq_len = input_ids.shape[-1]

                    # Validate position_ids dimensions
                    if pos_ids.dim() == 3 and pos_ids.shape[-1] != expected_seq_len:
                        logger = get_sample_logger()
                        logger.warning(
                            f"Position IDs length mismatch: expected {expected_seq_len}, got {pos_ids.shape[-1]}"
                        )
                        # Recreate position_ids with correct length
                        result["position_ids"] = (
                            torch.arange(expected_seq_len).view(1, -1).expand(3, -1)
                        )
                    elif pos_ids.dim() != 3:
                        # Fix incorrect dimensions
                        result["position_ids"] = (
                            torch.arange(expected_seq_len).view(1, -1).expand(3, -1)
                        )

            # Log visual data
            if "pixel_values" in result:
                data_logger.info(
                    f"   Pixel values shape: {result['pixel_values'].shape}"
                )
            if "image_grid_thw" in result:
                data_logger.info(
                    f"   Image grid THW shape: {result['image_grid_thw'].shape}"
                )
                data_logger.info(
                    f"   Image grid THW values: {result['image_grid_thw']}"
                )

            # Add attention_mask for collators
            if "input_ids" in result:
                seq_len = result["input_ids"].shape[-1]
                result["attention_mask"] = torch.ones(seq_len, dtype=torch.long)

            data_logger.info(f"📊 SAMPLE {idx} PROCESSING COMPLETE")
            data_logger.info("=" * 100)

            return result

        except Exception as e:
            raise ValueError(f"Sample {idx} preprocessing failed: {e}")


@dataclass
class FlattenedDataCollator:
    """
    Flattened data collator for packed sequence training.

    This is the default and recommended collator, matching the official repo approach.
    It concatenates samples into packed sequences without padding, providing:
    - Maximum memory efficiency (no padding waste)
    - Compatibility with flash attention
    - Optimal performance for variable-length sequences
    - Support for multi-GPU training with DeepSpeed

    NOTE: Currently disabled due to attention mask compatibility issues.
    Use StandardDataCollator for stable training.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_total_length: Optional[int] = None

    def __post_init__(self):
        self._batch_count = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch using packed sequences (official repo approach)."""
        self._batch_count += 1

        # Extract components
        input_ids = [instance["input_ids"].squeeze() for instance in instances]
        labels = [instance["labels"].squeeze() for instance in instances]
        position_ids = [instance.get("position_ids") for instance in instances]

        # Extract attention masks (sequence lengths)
        attention_mask = []
        for instance in instances:
            if "attention_mask" in instance:
                mask = instance["attention_mask"]
                if mask.dim() > 1:
                    mask = mask.squeeze()
                # Convert to sequence length
                seq_len = mask.sum().item() if mask.dtype == torch.bool else len(mask)
                attention_mask.append(seq_len)
            else:
                # Fallback: use input_ids length
                attention_mask.append(len(instance["input_ids"].squeeze()))

        # Validate total length
        total_length = sum(attention_mask)
        if self.max_total_length and total_length > self.max_total_length:
            raise ValueError(
                f"❌ TOTAL LENGTH EXCEEDED!\n"
                f"   Total sequence length: {total_length}\n"
                f"   Maximum allowed: {self.max_total_length}\n"
                f"   Samples in batch: {len(instances)}\n"
                f"   Individual lengths: {attention_mask}\n"
                f"\n"
                f"🔧 SOLUTIONS:\n"
                f"   1. Reduce per_device_train_batch_size\n"
                f"   2. Increase max_total_length\n"
                f"   3. Use shorter sequences\n"
            )

        # Create proper boolean attention mask for flash attention compatibility
        # Instead of cumulative lengths, create a proper boolean mask
        batch_size = len(instances)
        max_seq_len = max(attention_mask)

        # Create 2D attention mask [batch_size, max_seq_len]
        attention_mask_2d = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(attention_mask):
            attention_mask_2d[i, :seq_len] = True

        # Pad sequences to max length for proper batching
        padded_input_ids = []
        padded_labels = []
        padded_position_ids = []

        for i, (input_seq, label_seq, pos_ids) in enumerate(
            zip(input_ids, labels, position_ids)
        ):
            seq_len = len(input_seq)

            # Pad input_ids
            padded_input = torch.full(
                (max_seq_len,), self.tokenizer.pad_token_id, dtype=input_seq.dtype
            )
            padded_input[:seq_len] = input_seq
            padded_input_ids.append(padded_input)

            # Pad labels
            padded_label = torch.full(
                (max_seq_len,), IGNORE_INDEX, dtype=label_seq.dtype
            )
            padded_label[:seq_len] = label_seq
            padded_labels.append(padded_label)

            # Handle position_ids (can be None if using official model calculation)
            if pos_ids is not None:
                padded_pos = torch.zeros(3, 1, max_seq_len, dtype=pos_ids.dtype)
                padded_pos[:, :, :seq_len] = pos_ids
                padded_position_ids.append(padded_pos)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": attention_mask_2d,  # Proper boolean mask for flash attention
        }

        # Only add position_ids if they were provided
        if padded_position_ids:
            batch["position_ids"] = torch.cat(padded_position_ids, dim=1)

        # Handle images (concatenate across all samples)
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        ]

        if images:
            concat_images = torch.cat(images, dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)

            batch["pixel_values"] = concat_images
            batch["image_grid_thw"] = grid_thw

        return batch


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
    Standard data collator that pads each batch 到 batch 中最长序列的长度，
    完全动态计算，不做任何截断，保证每个样本 loss-free。
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate 一个 batch，动态 padding 到 batch 内最长长度，不做 truncation。

        Args:
            instances: 一个 sequence，每个元素为 dict，至少包含:
                - "input_ids": torch.Tensor，形状 [seq_len]
                - "labels":    torch.Tensor，形状 [seq_len]
                - 可选 "position_ids": torch.Tensor，形状 [3, 1, seq_len]
                - 可选 "pixel_values": torch.Tensor（图像张量）
                - 可选 "image_grid_thw": torch.Tensor

        Returns:
            一个 dict，包含:
                - "input_ids":       torch.Tensor，形状 [batch_size, batch_max_length]
                - "labels":          torch.Tensor，形状 [batch_size, batch_max_length]
                - "attention_mask":  torch.Tensor，形状 [batch_size, batch_max_length]
                - 可选 "position_ids": torch.Tensor，形状 [3, batch_size, batch_max_length]
                - 可选 "pixel_values": torch.Tensor，把所有样本的 pixel_values concat
                - 可选 "image_grid_thw": torch.Tensor，把所有样本的 image_grid_thw concat
        """
        # 1. 提取每个实例的 input_ids、labels、position_ids 列表
        input_ids_list: List[torch.Tensor] = [
            instance["input_ids"].squeeze() for instance in instances
        ]
        labels_list: List[torch.Tensor] = [
            instance["labels"].squeeze() for instance in instances
        ]
        position_ids_list: List[Optional[torch.Tensor]] = [
            instance.get("position_ids") for instance in instances
        ]

        # 2. 动态计算本次 batch 中最长序列长度
        batch_max_length: int = max(seq.shape[-1] for seq in input_ids_list)

        # 3. 准备 pad 后张量的容器
        batch_size = len(instances)
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
        attention_mask = torch.zeros(
            (batch_size, batch_max_length),
            dtype=torch.bool,
        )

        # 4. 逐样本拷贝原始序列，不做截断
        for i, (input_seq, label_seq) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = input_seq.shape[-1]  # 一定 <= batch_max_length
            padded_input_ids[i, :seq_len] = input_seq
            padded_labels[i, :seq_len] = label_seq
            attention_mask[i, :seq_len] = True

        # 5. 构造输出 dict
        batch: Dict[str, torch.Tensor] = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }

        # 6. 如果提供了 position_ids，就一并 pad/pad，并按维度 cat
        if any(pos_ids is not None for pos_ids in position_ids_list):
            # 每个 position_ids 的 shape 为 [3, 1, seq_len]
            padded_position_ids_list: List[torch.Tensor] = []
            for pos_ids in position_ids_list:
                if pos_ids is not None:
                    # 在最后一个维度 pad 到 batch_max_length
                    seq_len = pos_ids.shape[-1]
                    padded_pos = torch.zeros(
                        (3, 1, batch_max_length),
                        dtype=pos_ids.dtype,
                    )
                    padded_pos[:, :, :seq_len] = pos_ids
                else:
                    # 如果当前样本没有传 position_ids，创建全零占位
                    padded_pos = torch.zeros(
                        (3, 1, batch_max_length),
                        dtype=torch.long,
                    )
                padded_position_ids_list.append(padded_pos)

            # cat 到一起，得到形状 [3, batch_size, batch_max_length]
            batch["position_ids"] = torch.cat(padded_position_ids_list, dim=1)

        # 7. 如果有图像数据，则把所有样本的 pixel_values、image_grid_thw 串联
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        ]
        if images:
            # 假设每个 pixel_values 的 shape 为 [N_images, C, H, W]，都能在第 0 维 cat
            batch["pixel_values"] = torch.cat(images, dim=0)

            grid_thw_list = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            batch["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)

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
    if collator_type == "flattened":
        return FlattenedDataCollator(
            tokenizer=tokenizer,
            max_total_length=max_total_length,
        )
    elif collator_type == "standard":
        return StandardDataCollator(
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown collator_type: {collator_type}")


def test_data_collator():
    """Test function to verify FlattenedDataCollator works correctly."""
    print("🧪 Testing StandardDataCollator...")

    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.model_max_length = 8192

    tokenizer = MockTokenizer()
    collator = FlattenedDataCollator(tokenizer=tokenizer, max_total_length=16384)

    # Create mock instances
    instances = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.zeros(3, 1, 4),
            "attention_mask": torch.ones(4),
            "pixel_values": torch.randn(49, 1152),  # 7x7 image tokens
            "image_grid_thw": torch.tensor([[1, 7, 7]]),
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
            "labels": torch.tensor([1, 2, 3, 4, 5, 6]),
            "position_ids": torch.zeros(3, 1, 6),
            "attention_mask": torch.ones(6),
            "pixel_values": torch.randn(49, 1152),  # 7x7 image tokens
            "image_grid_thw": torch.tensor([[1, 7, 7]]),
        },
    ]

    try:
        batch = collator(instances)
        print("✅ Batch created successfully!")
        print(f"   - input_ids shape: {batch['input_ids'].shape}")
        print(f"   - labels shape: {batch['labels'].shape}")
        print(f"   - attention_mask: {batch['attention_mask']}")
        print(f"   - position_ids shape: {batch['position_ids'].shape}")
        print(f"   - pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   - image_grid_thw shape: {batch['image_grid_thw'].shape}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_data_collator()
