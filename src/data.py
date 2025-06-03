"""
Data loading and processing for Qwen2.5VL BBU training.
Clean implementation with minimal data collators.
"""

import itertools
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from .logging import get_data_logger  # Use global logger
from .preprocessing import create_preprocessor, preprocess_qwen_2_visual

IGNORE_INDEX = -100

# Global logger for raw samples to avoid recreating
_sample_logger = None


def get_sample_logger():
    """Get the sample logger using the global logging system."""
    return get_data_logger()


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file line by line."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class BBUDataset(Dataset):
    """Dataset for BBU quality inspection."""

    def __init__(self, config, tokenizer, image_processor, data_path: str):
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # Load data
        self.data = read_jsonl(data_path)
        print(f"📊 Loaded {len(self.data)} samples from {data_path}")

        # Create preprocessor
        self.preprocessor = create_preprocessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_root=config.data_root,
            model_max_length=config.model_max_length,
        )

        # Pre-calculate sequence lengths for adaptive batching
        self._sequence_lengths = None
        if hasattr(config, "adaptive_batching") and config.adaptive_batching:
            self._calculate_sequence_lengths()

    def _calculate_sequence_lengths(self):
        """Pre-calculate sequence lengths for adaptive batching."""
        print("📏 Pre-calculating sequence lengths for adaptive batching...")
        self._sequence_lengths = []

        for idx in range(len(self.data)):
            try:
                # Quick length estimation based on conversation content
                conversations = self.data[idx]["conversations"]
                est_length = sum(
                    len(conv.get("value", conv.get("content", "")).split())
                    for conv in conversations
                )
                # Add vision token estimate (rough)
                est_length += 256  # Rough estimate for vision tokens
                self._sequence_lengths.append(est_length)
            except Exception as e:
                print(f"Warning: Could not estimate length for sample {idx}: {e}")
                self._sequence_lengths.append(500)  # Default estimate

        print(f"✅ Calculated {len(self._sequence_lengths)} sequence length estimates")

    @property
    def sequence_lengths(self):
        """Get pre-calculated sequence lengths."""
        return self._sequence_lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a processed sample for training."""
        # Retry mechanism for robustness
        num_retries = 3

        for attempt in range(num_retries):
            try:
                return self._get_item(idx)
            except Exception as e:
                if attempt == num_retries - 1:
                    raise e
                print(f"Retry {attempt + 1} for sample {idx}: {e}")

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item using official preprocessing logic.

        Args:
            idx: Sample index

        Returns:
            Processed sample ready for training
        """
        sample = self.data[idx]

        # Import the official preprocessing function

        # Handle image paths exactly like official implementation
        if "images" in sample:
            image_paths = sample["images"]
        elif "image" in sample:
            image_paths = [sample["image"]]
        else:
            image_paths = []

        # Process images exactly like official implementation
        if image_paths:
            pixel_values_list = []
            image_grid_thw_list = []
            grid_thw_merged_list = []

            for image_path in image_paths:
                # Handle relative paths
                if not os.path.isabs(image_path):
                    full_path = os.path.join(self.config.data_root, image_path)
                else:
                    full_path = image_path

                # Load and process image exactly like official
                image = Image.open(full_path).convert("RGB")

                # Process image without deepcopy to save memory
                visual_processed = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )

                pixel_values = visual_processed["pixel_values"]
                if isinstance(pixel_values, list):
                    pixel_values = pixel_values[0]

                grid_thw = visual_processed["image_grid_thw"][0]

                # CRITICAL: Calculate grid_thw_merged exactly like official
                grid_thw_merged = grid_thw.prod() // self.image_processor.merge_size**2

                pixel_values_list.append(pixel_values)
                image_grid_thw_list.append(grid_thw)
                grid_thw_merged_list.append(grid_thw_merged.item())

                # Clean up image to save memory
                del image, visual_processed

            # Concatenate like official implementation
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.stack(image_grid_thw_list, dim=0)
        else:
            pixel_values = None
            image_grid_thw = None
            grid_thw_merged_list = []

        # Process conversations using official preprocessing
        chat_sources = [sample["conversations"]]
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged_list if grid_thw_merged_list else [],
        )

        # Calculate position IDs exactly like official implementation
        if image_grid_thw is not None:
            from .rope2d import get_rope_index_25

            position_ids, _ = get_rope_index_25(
                spatial_merge_size=self.image_processor.merge_size,
                input_ids=data_dict["input_ids"],
                image_grid_thw=image_grid_thw,
            )
        else:
            # Text-only case
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        # Add position IDs and attention mask
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        # Add image data if present
        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        return data_dict


def pad_and_cat(tensor_list):
    """Official pad_and_cat function from qwen-vl-finetune."""
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


class DataCollator:
    """
    Main data collator that pads sequences to the same length in a batch.

    This is the primary collator used for training. It handles:
    - Padding sequences to the longest sequence in the batch
    - Simple dynamic length (just uses max sequence length)
    - Position IDs for mRoPE (3D tensors)
    - Image data concatenation
    - Comprehensive length validation
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: Optional[int] = None,
        use_dynamic_length: bool = True,
        adaptive_batching: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_dynamic_length = use_dynamic_length
        self.adaptive_batching = adaptive_batching
        self._batch_count = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances with simple padding to longest sequence."""
        self._batch_count += 1

        # Extract components
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        # 🔧 DISTRIBUTED TRAINING FIX: Determine target device from input tensors
        # In distributed training, we need to ensure all tensors are on the same device
        target_device = input_ids[0].device if input_ids else torch.device("cpu")

        # ✅ STRICT LENGTH VALIDATION: Raise error if any sequence exceeds max_length
        max_seq_len = max(len(seq) for seq in input_ids)
        configured_max_length = self.max_seq_length or self.tokenizer.model_max_length

        if max_seq_len > configured_max_length:
            # Find which sequences are too long for detailed error reporting
            long_sequences = [
                (i, len(seq))
                for i, seq in enumerate(input_ids)
                if len(seq) > configured_max_length
            ]

            error_msg = (
                f"❌ INPUT LENGTH VALIDATION FAILED!\n"
                f"   Maximum sequence length in batch: {max_seq_len}\n"
                f"   Configured maximum length: {configured_max_length}\n"
                f"   Sequences exceeding limit: {len(long_sequences)}\n"
                f"   Details: {long_sequences[:5]}{'...' if len(long_sequences) > 5 else ''}\n"
                f"\n"
                f"🔧 SOLUTIONS:\n"
                f"   1. Increase model_max_length in config (recommended)\n"
                f"   2. Reduce number of examples in multi-image training\n"
                f"   3. Use shorter prompts or responses\n"
                f"\n"
                f"💡 CURRENT CONFIG:\n"
                f"   - model_max_length: {self.tokenizer.model_max_length}\n"
                f"   - max_seq_length: {self.max_seq_length}\n"
                f"   - use_dynamic_length: {self.use_dynamic_length}\n"
                f"\n"
                f"⚠️  NO DATA WILL BE TRUNCATED - TRAINING STOPPED TO PREVENT DATA LOSS"
            )

            raise ValueError(error_msg)

        # Calculate max length (simple padding to longest sequence)
        if self.use_dynamic_length:
            max_length = self._calculate_max_length(input_ids)
        else:
            max_length = configured_max_length

        # ✅ SAFETY CHECK: Ensure we never truncate
        if max_length < max_seq_len:
            max_length = max_seq_len
            print(
                f"⚠️  Adjusted max_length from {max_length} to {max_seq_len} to prevent truncation"
            )

        # 🔧 DEVICE FIX: Ensure all input tensors are on the same device before padding
        input_ids = [ids.to(target_device) for ids in input_ids]
        labels = [lbls.to(target_device) for lbls in labels]

        # Pad sequences using PyTorch's built-in padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # 🔧 DEVICE FIX: Ensure padded tensors are on correct device
        input_ids = input_ids.to(target_device)
        labels = labels.to(target_device)

        # Create attention mask - ensure it's on the correct device
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(target_device)

        # Handle position_ids for mRoPE (3D tensors)
        if position_ids[0] is not None:
            # Ensure position_ids are on correct device before padding
            position_ids = [pos_ids.to(target_device) for pos_ids in position_ids]
            position_ids = pad_and_cat(position_ids)
            position_ids = position_ids.to(target_device)
        else:
            position_ids = None

        # Handle images - ensure they're on correct device
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        ]
        if images:
            # Ensure all images are on the same device before concatenation
            images = [img.to(target_device) for img in images]
            concat_images = torch.cat(images, dim=0)

            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            if grid_thw:
                grid_thw = [thw.to(target_device) for thw in grid_thw]
                grid_thw = torch.cat(grid_thw, dim=0)
            else:
                grid_thw = None
        else:
            concat_images = None
            grid_thw = None

        # Log batch statistics periodically
        if self._batch_count % 100 == 0:
            self._log_batch_stats_with_validation(
                input_ids, max_length, max_seq_len, configured_max_length
            )

        # 🔧 FINAL DEVICE CHECK: Ensure all tensors are on the same device
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": concat_images,
            "image_grid_thw": grid_thw,
        }

        # Validate all tensors are on the same device
        devices = set()
        for key, tensor in batch.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                devices.add(tensor.device)

        if len(devices) > 1:
            print(f"⚠️ Warning: Tensors on multiple devices in batch: {devices}")
            print(f"   Target device: {target_device}")
            for key, tensor in batch.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    print(f"   {key}: {tensor.device}")

        return batch

    def _calculate_max_length(self, input_ids: List[torch.Tensor]) -> int:
        """Calculate max length - simply return the longest sequence length."""
        seq_lengths = [len(seq) for seq in input_ids]
        return max(seq_lengths)

    def _log_batch_stats_with_validation(
        self, input_ids, max_length, max_seq_len, configured_max_length
    ):
        seq_lengths = [len(ids) for ids in input_ids]
        avg_length = sum(seq_lengths) / len(seq_lengths)
        padding_waste = self._calculate_padding_waste(seq_lengths, max_length)
        print(
            f"📊 Batch {self._batch_count}: max_len={max_length}, "
            f"avg_len={avg_length:.1f}, padding_waste={padding_waste:.1f}%, "
            f"max_seq_len={max_seq_len}, configured_max_length={configured_max_length}"
        )

    def _calculate_padding_waste(
        self, seq_lengths: List[int], max_length: int
    ) -> float:
        """Calculate percentage of padding waste."""
        total_tokens = len(seq_lengths) * max_length
        actual_tokens = sum(seq_lengths)
        padding_tokens = total_tokens - actual_tokens
        return (padding_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0


@dataclass
class FlattenedDataCollator:
    """
    Flattened data collator for reference (causes attention mask issues in practice).

    This collator concatenates samples into a single flattened sequence without padding.
    While it saves memory, it requires complex attention mask handling and is not
    recommended for general use. Kept for reference and future experimentation.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_total_length: Optional[int] = None

    def __post_init__(self):
        self._batch_count = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch using data flattening - experimental, not recommended."""
        self._batch_count += 1

        # Extract components exactly like official implementation
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )

        # Flatten attention_mask to sequence lengths (official format)
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )

        # Length validation
        total_length = sum(attention_mask)

        if self.max_total_length and total_length > self.max_total_length:
            raise ValueError(
                f"❌ TOTAL LENGTH VALIDATION FAILED (FlattenedDataCollator)!\n"
                f"   Total sequence length: {total_length}\n"
                f"   Maximum allowed: {self.max_total_length}\n"
                f"   This collator is experimental and not recommended for production use."
            )

        # Create cumulative sequence lengths for flash attention
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        # Concatenate sequences
        input_ids = torch.cat([ids.unsqueeze(0) for ids in input_ids], dim=1)
        labels = torch.cat([lbls.unsqueeze(0) for lbls in labels], dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

        # Handle images
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)

            batch["pixel_values"] = concat_images
            batch["image_grid_thw"] = grid_thw

        # Log efficiency info
        if self._batch_count % 50 == 1:
            print(
                f"🚀 Flattened batching - Batch {self._batch_count}: "
                f"samples={len(instances)}, total_length={input_ids.size(1)}, "
                f"efficiency=100% (no padding waste) - EXPERIMENTAL"
            )

        return batch


def extract_ground_truth_objects_from_conversation(
    conversations: List[Dict[str, str]],
) -> Optional[List[Dict]]:
    """
    Extract ground truth objects from telecom conversation format.

    Args:
        conversations: List of conversation turns

    Returns:
        List of ground truth objects with bbox and description, or None
    """
    # Find the assistant response
    for conv in conversations:
        role = conv.get("role", conv.get("from", ""))
        content = conv.get("content", conv.get("value", ""))

        if role == "assistant" and content:
            try:
                # Try to parse as JSON array
                objects = json.loads(content)
                if isinstance(objects, list):
                    validated_objects = []
                    for obj in objects:
                        if (
                            isinstance(obj, dict)
                            and "bbox" in obj
                            and "description" in obj
                        ):
                            # Validate bbox format
                            bbox = obj["bbox"]
                            if isinstance(bbox, list) and len(bbox) == 4:
                                validated_objects.append(
                                    {"bbox": bbox, "description": obj["description"]}
                                )

                    if validated_objects:
                        return validated_objects
            except (json.JSONDecodeError, KeyError, TypeError):
                # If JSON parsing fails, try regex extraction
                return _extract_objects_with_regex(content)

    return None


def _extract_objects_with_regex(content: str) -> Optional[List[Dict]]:
    """
    Extract objects using regex when JSON parsing fails.

    Args:
        content: Assistant response content

    Returns:
        List of extracted objects or None
    """
    objects = []

    # Pattern for telecom format: {"bbox": [x1,y1,x2,y2], "description": "..."}
    pattern = r'\{\s*"bbox"\s*:\s*\[([^\]]+)\]\s*,\s*"description"\s*:\s*"([^"]+)"\s*\}'

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


# Factory function for easy collator creation
def create_data_collator(
    tokenizer,
    strategy: str = "padding",
    max_seq_length: Optional[int] = None,
    use_dynamic_length: bool = True,
    **kwargs,
):
    """
    Factory function to create the appropriate data collator.

    Args:
        tokenizer: The tokenizer to use
        strategy: "padding" (default) or "flattened" (experimental)
        max_seq_length: Maximum sequence length for padding strategy
        use_dynamic_length: Whether to use dynamic length optimization
        **kwargs: Additional arguments

    Returns:
        DataCollator instance
    """
    if strategy == "padding":
        return DataCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            use_dynamic_length=use_dynamic_length,
            **kwargs,
        )
    elif strategy == "flattened":
        print("⚠️  Warning: FlattenedDataCollator is experimental and may cause issues")
        return FlattenedDataCollator(tokenizer=tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'padding' or 'flattened'")


def test_data_collator():
    """Simple test function to verify DataCollator works correctly."""
    print("🧪 Testing DataCollator...")

    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.model_max_length = 2048

    tokenizer = MockTokenizer()
    collator = DataCollator(tokenizer, use_dynamic_length=True)

    # Create mock instances
    instances = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.zeros(3, 1, 4),
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 7, 7]]),
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
            "labels": torch.tensor([1, 2, 3, 4, 5, 6]),
            "position_ids": torch.zeros(3, 1, 6),
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 7, 7]]),
        },
    ]

    try:
        batch = collator(instances)
        print("✅ Batch created successfully!")
        print(f"   - input_ids shape: {batch['input_ids'].shape}")
        print(f"   - labels shape: {batch['labels'].shape}")
        print(f"   - attention_mask shape: {batch['attention_mask'].shape}")
        print(f"   - position_ids shape: {batch['position_ids'].shape}")
        print(f"   - pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   - image_grid_thw shape: {batch['image_grid_thw'].shape}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_data_collator()
