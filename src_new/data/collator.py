"""
Data collation for Qwen2.5-VL training.

This module provides:
- StandardDataCollator: Standard data collator for Qwen2.5-VL
- PackedDataCollator: Packed data collator for efficient training
- create_data_collator: Factory function for creating data collators
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


if TYPE_CHECKING:
    from src_new.config.config import Config


def get_collator_logger() -> logging.Logger:
    """Get rank-aware logger for collator module."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("collator")
    except ImportError:
        # Fallback to standard logging
        logger = logging.getLogger("collator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_collator_logger()


@dataclass
class StandardDataCollator:
    """
    Standard data collator for Qwen2.5-VL.

    Handles padding, attention masks, and labels for standard batch processing.
    """

    tokenizer: PreTrainedTokenizerBase
    config: Optional["Config"] = None
    pad_token_id: Optional[int] = None
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        """Initialize collator with defaults from config or tokenizer."""
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

        if self.max_length is None and self.config is not None:
            self.max_length = getattr(self.config, "max_total_length", 4096)

        logger.info(
            f"StandardDataCollator initialized with max_length={self.max_length}"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batch.

        Args:
            features: List of features from dataset

        Returns:
            Batch with padded tensors
        """
        # Extract tensors from features
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Get image features if available
        pixel_values = None
        image_grid_thw = None
        if "pixel_values" in features[0]:
            # CRITICAL FIX: Handle Qwen2.5-VL patch-based pixel_values format
            # Qwen2.5-VL uses flattened patches [num_patches, patch_features], not standard image tensors
            pixel_values_list = []
            for f in features:
                pv = f["pixel_values"]
                if pv.dim() == 2:  # [num_patches, patch_features] - Qwen2.5-VL format
                    # This is the expected format for Qwen2.5-VL flattened patches
                    pixel_values_list.append(pv)
                elif (
                    pv.dim() == 4
                ):  # [num_images, channels, height, width] - Standard format
                    # Multi-image case: flatten to individual images
                    for i in range(pv.shape[0]):
                        pixel_values_list.append(
                            pv[i]
                        )  # Each is [channels, height, width]
                elif pv.dim() == 3:  # [channels, height, width] - Standard format
                    # Single image case
                    pixel_values_list.append(pv)
                else:
                    raise ValueError(
                        f"❌ CRITICAL: Invalid pixel_values dimensions {pv.dim()}. "
                        f"Expected 2D [patches, features] for Qwen2.5-VL, 3D [C,H,W] or 4D [N,C,H,W] for standard format, "
                        f"got shape {pv.shape}."
                    )

            # For Qwen2.5-VL, concatenate patches along the first dimension
            if pixel_values_list and pixel_values_list[0].dim() == 2:
                # Qwen2.5-VL patch format: concatenate patches
                pixel_values = torch.cat(
                    pixel_values_list, dim=0
                )  # [total_patches, patch_features]
            else:
                # Standard image format: stack images
                pixel_values = torch.stack(
                    pixel_values_list
                )  # [total_images, channels, height, width]
        if "image_grid_thw" in features[0]:
            # CRITICAL FIX: Handle image_grid_thw correctly for teacher-student training
            # In teacher-student setup, each sample may contain multiple images
            image_grid_thw_list = []
            for f in features:
                grid_thw = f["image_grid_thw"]

                # Handle different tensor shapes based on teacher-student structure
                if grid_thw.dim() == 2:
                    if grid_thw.shape[0] == 1 and grid_thw.shape[1] == 3:
                        # Single image case: [1, 3] -> [3]
                        grid_thw = grid_thw.squeeze(0)
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2 and grid_thw.shape[1] == 3:
                        # Teacher-student case: [2, 3] -> flatten to 2 separate [3] tensors
                        # This represents 2 images (teacher + student) in one sample
                        for i in range(grid_thw.shape[0]):
                            image_grid_thw_list.append(grid_thw[i])  # Each is [3]
                    else:
                        raise ValueError(
                            f"❌ CRITICAL: Unsupported image_grid_thw shape {grid_thw.shape}. "
                            f"Expected [1, 3] for single image or [2, 3] for teacher-student pair."
                        )
                elif grid_thw.dim() == 1:
                    if grid_thw.shape[0] == 3:
                        # Already correct shape: [3]
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2:
                        # Missing time dimension: [h, w] -> [1, h, w]
                        h, w = grid_thw
                        grid_thw = torch.tensor(
                            [1, h, w], dtype=grid_thw.dtype, device=grid_thw.device
                        )
                        image_grid_thw_list.append(grid_thw)
                    else:
                        raise ValueError(
                            f"❌ CRITICAL: Invalid 1D image_grid_thw shape {grid_thw.shape}. "
                            f"Expected [3] for (t,h,w) or [2] for (h,w)."
                        )
                else:
                    raise ValueError(
                        f"❌ CRITICAL: Invalid image_grid_thw dimensions {grid_thw.dim()}. "
                        f"Expected 1D [3] or 2D [1,3] or [2,3], got shape {grid_thw.shape}."
                    )

            # Stack all individual image grid_thw tensors to get [total_images, 3]
            # where total_images = batch_size * images_per_sample
            image_grid_thw = torch.stack(image_grid_thw_list)

        # Extract teacher-student spans if available
        teacher_assistant_spans = None
        student_assistant_spans = None
        if "teacher_assistant_spans" in features[0]:
            teacher_assistant_spans = [f["teacher_assistant_spans"] for f in features]
        if "student_assistant_spans" in features[0]:
            student_assistant_spans = [f["student_assistant_spans"] for f in features]

        # Pad sequences
        padded_input_ids = self._pad_sequence(input_ids, self.pad_token_id)
        padded_attention_mask = self._pad_sequence(attention_mask, 0)
        padded_labels = self._pad_sequence(labels, self.label_pad_token_id)

        # Create batch
        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

        # Convert attention_mask to boolean and make contiguous (FlashAttention 2 friendly)
        if isinstance(batch["attention_mask"], torch.Tensor):
            batch["attention_mask"] = (
                batch["attention_mask"].to(torch.bool).contiguous()
            )

        # Add image features if available
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
            logger.debug(
                f"Added image_grid_thw to batch: {image_grid_thw.shape} = {image_grid_thw}"
            )

        # Add teacher-student spans if available
        if teacher_assistant_spans is not None:
            batch["teacher_assistant_spans"] = teacher_assistant_spans
            logger.debug(
                f"Added teacher_assistant_spans to batch: {len(teacher_assistant_spans)} samples"
            )
        if student_assistant_spans is not None:
            batch["student_assistant_spans"] = student_assistant_spans
            logger.debug(
                f"Added student_assistant_spans to batch: {len(student_assistant_spans)} samples"
            )

        # PRE-BATCH VALIDATION: Ensure multimodal tensors are consistent to avoid CUDA OOB later
        try:
            if "pixel_values" in batch and "image_grid_thw" in batch:
                pv = batch["pixel_values"]
                grid = batch["image_grid_thw"]
                # Expect flattened patches for Qwen2.5-VL
                if pv.dim() not in (2,):
                    raise ValueError(
                        f"Standard collator: pixel_values must be flattened [num_patches, patch_features], got {pv.shape}"
                    )
                if grid.dim() != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"Standard collator: image_grid_thw must be [num_images, 3], got {grid.shape}"
                    )
                expected = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
                actual = int(pv.shape[0])
                if expected != actual:
                    raise ValueError(
                        f"Standard collator: pixel_values rows ({actual}) != sum(t*h*w) ({expected}) from image_grid_thw"
                    )
        except Exception as e:
            logger.error(f"❌ Multimodal validation failure (standard): {e}")
            raise

        return batch

    def _pad_sequence(
        self, sequences: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of sequences
            pad_value: Value to use for padding

        Returns:
            Padded tensor
        """
        # Get sequence lengths
        lengths = [seq.size(0) for seq in sequences]
        max_len = max(lengths)

        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            padding_length = max_len - seq.size(0)
            if padding_length > 0:
                padding = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack padded sequences
        return torch.stack(padded_sequences)


@dataclass
class PackedDataCollator:
    """
    Packed data collator for Qwen2.5-VL.

    Handles packing multiple sequences into a single batch for efficient training.
    """

    tokenizer: PreTrainedTokenizerBase
    config: Optional["Config"] = None
    pad_token_id: Optional[int] = None
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        """Initialize collator with defaults from config or tokenizer."""
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

        if self.max_length is None and self.config is not None:
            self.max_length = getattr(self.config, "max_total_length", 4096)

        logger.info(f"PackedDataCollator initialized with max_length={self.max_length}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into packed batch.

        Args:
            features: List of features from dataset

        Returns:
            Batch with packed tensors
        """
        # Check for image features
        has_images = "pixel_values" in features[0]

        if has_images:
            # Handle image batches with standard padding
            return self._collate_with_images(features)
        else:
            # Use efficient packing for text-only batches
            return self._collate_packed(features)

    def _collate_with_images(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate features with images.

        Args:
            features: List of features from dataset

        Returns:
            Batch with padded tensors
        """
        # Extract tensors from features
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        # CRITICAL FIX: Handle Qwen2.5-VL patch-based pixel_values format
        # Qwen2.5-VL uses flattened patches [num_patches, patch_features], not standard image tensors
        pixel_values_list = []
        for f in features:
            pv = f["pixel_values"]
            if pv.dim() == 2:  # [num_patches, patch_features] - Qwen2.5-VL format
                # This is the expected format for Qwen2.5-VL flattened patches
                pixel_values_list.append(pv)
            elif (
                pv.dim() == 4
            ):  # [num_images, channels, height, width] - Standard format
                # Multi-image case: flatten to individual images
                for i in range(pv.shape[0]):
                    pixel_values_list.append(pv[i])  # Each is [channels, height, width]
            elif pv.dim() == 3:  # [channels, height, width] - Standard format
                # Single image case
                pixel_values_list.append(pv)
            else:
                raise ValueError(
                    f"❌ CRITICAL: Invalid pixel_values dimensions {pv.dim()}. "
                    f"Expected 2D [patches, features] for Qwen2.5-VL, 3D [C,H,W] or 4D [N,C,H,W] for standard format, "
                    f"got shape {pv.shape}."
                )

        # For Qwen2.5-VL, we need to handle batching correctly
        # The model expects pixel_values to be properly batched
        if pixel_values_list and pixel_values_list[0].dim() == 2:
            # Qwen2.5-VL patch format: concatenate patches from all samples
            # This creates a single tensor with all patches from all images in the batch
            pixel_values = torch.cat(
                pixel_values_list, dim=0
            )  # [total_patches_in_batch, patch_features]
        else:
            # Standard image format: stack images
            pixel_values = torch.stack(
                pixel_values_list
            )  # [total_images, channels, height, width]

        # Extract teacher-student spans if available
        teacher_assistant_spans = None
        student_assistant_spans = None
        if "teacher_assistant_spans" in features[0]:
            teacher_assistant_spans = [f["teacher_assistant_spans"] for f in features]
        if "student_assistant_spans" in features[0]:
            student_assistant_spans = [f["student_assistant_spans"] for f in features]

        # Pad sequences
        padded_input_ids = self._pad_sequence(input_ids, self.pad_token_id)
        padded_attention_mask = self._pad_sequence(attention_mask, 0)
        padded_labels = self._pad_sequence(labels, self.label_pad_token_id)

        # Create batch
        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
            "pixel_values": pixel_values,
        }

        # Convert attention_mask to boolean and make contiguous (FlashAttention 2 friendly)
        if isinstance(batch["attention_mask"], torch.Tensor):
            batch["attention_mask"] = (
                batch["attention_mask"].to(torch.bool).contiguous()
            )

        # CRITICAL FIX: Handle image_grid_thw correctly
        # The issue: image_grid_thw from processor has shape [1, 3]
        # When we stack multiple samples, we want [batch_size, 3], not [batch_size, 1, 3]
        if "image_grid_thw" in features[0]:
            # CRITICAL FIX: Handle image_grid_thw correctly for teacher-student training
            # In teacher-student setup, each sample may contain multiple images
            image_grid_thw_list = []
            for f in features:
                grid_thw = f["image_grid_thw"]

                # Handle different tensor shapes based on teacher-student structure
                if grid_thw.dim() == 2:
                    if grid_thw.shape[0] == 1 and grid_thw.shape[1] == 3:
                        # Single image case: [1, 3] -> [3]
                        grid_thw = grid_thw.squeeze(0)
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2 and grid_thw.shape[1] == 3:
                        # Teacher-student case: [2, 3] -> flatten to 2 separate [3] tensors
                        # This represents 2 images (teacher + student) in one sample
                        for i in range(grid_thw.shape[0]):
                            image_grid_thw_list.append(grid_thw[i])  # Each is [3]
                    else:
                        raise ValueError(
                            f"❌ CRITICAL: Unsupported image_grid_thw shape {grid_thw.shape}. "
                            f"Expected [1, 3] for single image or [2, 3] for teacher-student pair."
                        )
                elif grid_thw.dim() == 1:
                    if grid_thw.shape[0] == 3:
                        # Already correct shape: [3]
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2:
                        # Missing time dimension: [h, w] -> [1, h, w]
                        h, w = grid_thw
                        grid_thw = torch.tensor(
                            [1, h, w], dtype=grid_thw.dtype, device=grid_thw.device
                        )
                        image_grid_thw_list.append(grid_thw)
                    else:
                        raise ValueError(
                            f"❌ CRITICAL: Invalid 1D image_grid_thw shape {grid_thw.shape}. "
                            f"Expected [3] for (t,h,w) or [2] for (h,w)."
                        )
                else:
                    raise ValueError(
                        f"❌ CRITICAL: Invalid image_grid_thw dimensions {grid_thw.dim()}. "
                        f"Expected 1D [3] or 2D [1,3] or [2,3], got shape {grid_thw.shape}."
                    )

            # Stack all individual image grid_thw tensors to get [total_images, 3]
            # where total_images = batch_size * images_per_sample
            image_grid_thw = torch.stack(image_grid_thw_list)
            batch["image_grid_thw"] = image_grid_thw
            logger.debug(
                f"Added image_grid_thw to batch: {image_grid_thw.shape} = {image_grid_thw}"
            )

        # Add teacher-student spans if available
        if teacher_assistant_spans is not None:
            batch["teacher_assistant_spans"] = teacher_assistant_spans
            logger.debug(
                f"Added teacher_assistant_spans to batch: {len(teacher_assistant_spans)} samples"
            )
        if student_assistant_spans is not None:
            batch["student_assistant_spans"] = student_assistant_spans
            logger.debug(
                f"Added student_assistant_spans to batch: {len(student_assistant_spans)} samples"
            )

        # PRE-BATCH VALIDATION: Ensure multimodal tensors are consistent to avoid CUDA OOB later
        try:
            if "pixel_values" in batch and "image_grid_thw" in batch:
                pv = batch["pixel_values"]
                grid = batch["image_grid_thw"]
                # Expect flattened patches for Qwen2.5-VL here as well
                if pv.dim() not in (2,):
                    raise ValueError(
                        f"Packed collator: pixel_values must be flattened [total_patches, patch_features], got {pv.shape}"
                    )
                if grid.dim() != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"Packed collator: image_grid_thw must be [num_images, 3], got {grid.shape}"
                    )
                expected = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
                actual = int(pv.shape[0])
                if expected != actual:
                    raise ValueError(
                        f"Packed collator: pixel_values rows ({actual}) != sum(t*h*w) ({expected}) from image_grid_thw"
                    )
        except Exception as e:
            logger.error(f"❌ Multimodal validation failure (packed): {e}")
            raise

        return batch

    def _collate_packed(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate features into packed batch.

        Args:
            features: List of features from dataset

        Returns:
            Batch with packed tensors
        """
        # Extract tensors from features
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        # Concatenate all sequences with separator
        for feature in features:
            all_input_ids.append(feature["input_ids"])
            all_attention_mask.append(feature["attention_mask"])
            all_labels.append(feature["labels"])

        # Pack sequences
        packed_input_ids, packed_attention_mask, packed_labels = self._pack_sequences(
            all_input_ids, all_attention_mask, all_labels
        )

        # Create batch
        batch = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
        }

        # Convert attention_mask to boolean and make contiguous (FlashAttention 2 friendly)
        if isinstance(batch["attention_mask"], torch.Tensor):
            batch["attention_mask"] = (
                batch["attention_mask"].to(torch.bool).contiguous()
            )

        return batch

    def _pack_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pack sequences into a single batch.

        Args:
            input_ids: List of input ID tensors
            attention_mask: List of attention mask tensors
            labels: List of label tensors

        Returns:
            Tuple of packed tensors
        """
        # Calculate total length
        total_length = sum(ids.size(0) for ids in input_ids)

        # Create packed tensors
        packed_input_ids = torch.zeros(1, total_length, dtype=torch.long)
        packed_attention_mask = torch.zeros(1, total_length, dtype=torch.long)
        packed_labels = torch.full(
            (1, total_length), self.label_pad_token_id, dtype=torch.long
        )

        # Pack sequences
        offset = 0
        for ids, mask, lbl in zip(input_ids, attention_mask, labels):
            length = ids.size(0)
            packed_input_ids[0, offset : offset + length] = ids
            packed_attention_mask[0, offset : offset + length] = mask
            packed_labels[0, offset : offset + length] = lbl
            offset += length

        return packed_input_ids, packed_attention_mask, packed_labels

    def _pad_sequence(
        self, sequences: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of sequences (can be 1D or 2D with batch dimension)
            pad_value: Value to use for padding

        Returns:
            Padded tensor
        """
        if not sequences:
            raise ValueError("Cannot pad empty sequence list")

        # Handle 2D tensors with batch dimension [1, seq_len] -> [seq_len]
        if sequences[0].dim() == 2 and sequences[0].size(0) == 1:
            sequences = [seq.squeeze(0) for seq in sequences]

        # Get sequence lengths
        lengths = [seq.size(0) for seq in sequences]
        max_len = max(lengths)

        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            padding_length = max_len - seq.size(0)
            if padding_length > 0:
                padding = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack padded sequences
        return torch.stack(padded_sequences)


@dataclass
class TrainerCompatibleDataCollator:
    """
    Wrapper for data collators to ensure compatibility with HuggingFace Trainer.

    This wrapper ensures that:
    1. All tensors have consistent shapes
    2. All required fields are present
    3. Any unexpected errors are gracefully handled
    """

    base_collator: Union[StandardDataCollator, PackedDataCollator]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features with error handling.

        Args:
            features: List of features from dataset

        Returns:
            Batch with tensors ready for model input
        """
        # Use base collator - FAIL FAST, no try-except
        batch = self.base_collator(features)

        # Validate batch
        return self._validate_batch(batch)

    def _validate_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Validate batch for trainer compatibility.

        Args:
            batch: Batch from base collator

        Returns:
            Validated batch
        """
        # Ensure required fields are present
        required_fields = ["input_ids", "attention_mask", "labels"]
        for field in required_fields:
            if field not in batch:
                logger.warning(f"⚠️ Missing required field {field} in batch")
                batch[field] = torch.zeros(1, 1, dtype=torch.long)

        # Ensure all tensors have at least 2 dimensions
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                batch[key] = value.unsqueeze(0)

        return batch


def create_data_collator(
    collator_type: str,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional["Config"] = None,
) -> TrainerCompatibleDataCollator:
    """
    Create data collator based on type.

    Args:
        collator_type: Type of collator ("standard" or "packed")
        tokenizer: Tokenizer for text processing
        config: Configuration object with collator settings

    Returns:
        TrainerCompatibleDataCollator wrapping the requested collator type

    Raises:
        ValueError: If collator_type is not supported
    """
    # Create base collator based on type
    if collator_type == "standard":
        base_collator = StandardDataCollator(tokenizer=tokenizer, config=config)
    elif collator_type == "packed":
        base_collator = PackedDataCollator(tokenizer=tokenizer, config=config)
    else:
        raise ValueError(f"Unsupported collator_type: {collator_type}")

    # Wrap with trainer-compatible collator
    return TrainerCompatibleDataCollator(base_collator=base_collator)
