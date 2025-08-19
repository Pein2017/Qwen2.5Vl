from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new.types.arrays import (
    jaxtyped_beartype,
)
from src_new.types.batch import MultimodalBatch
from src_new.types.shapes import (
    IMAGE_GRID_THW_SHAPE_DESC,
    PIXEL_VALUES_PACKED_SHAPE_DESC,
)

from .collator_utils import get_collator_logger


if TYPE_CHECKING:
    from src_new.config.config import Config

logger = get_collator_logger()


@dataclass
class PackedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    config: Optional["Config"] = None
    pad_token_id: Optional[int] = None
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id
        if self.max_length is None and self.config is not None:
            self.max_length = self.config.max_total_length
        logger.info(f"PackedDataCollator initialized with max_length={self.max_length}")

    @jaxtyped_beartype
    def __call__(self, features: List[Dict[str, Any]]) -> MultimodalBatch:
        has_images = "pixel_values" in features[0]
        if has_images:
            return self._collate_with_images(features)
        else:
            return self._collate_packed(features)

    @jaxtyped_beartype
    def _collate_with_images(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values_list = []
        for f in features:
            pv = f["pixel_values"]
            if pv.dim() == 2:
                pixel_values_list.append(pv)
            elif pv.dim() == 4:
                for i in range(pv.shape[0]):
                    pixel_values_list.append(pv[i])
            elif pv.dim() == 3:
                pixel_values_list.append(pv)
            else:
                raise ValueError(
                    f"Invalid pixel_values dims={pv.dim()} shape={pv.shape} (expected {PIXEL_VALUES_PACKED_SHAPE_DESC})"
                )
        if pixel_values_list and pixel_values_list[0].dim() == 2:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.stack(pixel_values_list)

        teacher_assistant_spans = None
        student_assistant_spans = None
        assistant_spans = None
        if "teacher_assistant_spans" in features[0]:
            teacher_assistant_spans = [f["teacher_assistant_spans"] for f in features]
        if "student_assistant_spans" in features[0]:
            student_assistant_spans = [f["student_assistant_spans"] for f in features]
        if "assistant_spans" in features[0]:
            assistant_spans = [f["assistant_spans"] for f in features]

        padded_input_ids = self._pad_sequence(input_ids, self.pad_token_id)
        padded_attention_mask = self._pad_sequence(attention_mask, 0)
        padded_labels = self._pad_sequence(labels, self.label_pad_token_id)

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
            "pixel_values": pixel_values,
        }
        if teacher_assistant_spans is not None:
            batch["teacher_assistant_spans"] = teacher_assistant_spans
        if student_assistant_spans is not None:
            batch["student_assistant_spans"] = student_assistant_spans
        if assistant_spans is not None:
            batch["assistant_spans"] = assistant_spans

        if "image_grid_thw" in features[0]:
            image_grid_thw_list = []
            for f in features:
                grid_thw = f["image_grid_thw"]
                if grid_thw.dim() == 2:
                    if grid_thw.shape[0] == 1 and grid_thw.shape[1] == 3:
                        grid_thw = grid_thw.squeeze(0)
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2 and grid_thw.shape[1] == 3:
                        for i in range(grid_thw.shape[0]):
                            image_grid_thw_list.append(grid_thw[i])
                    else:
                        raise ValueError(
                            f"Unsupported image_grid_thw shape {grid_thw.shape} (expected {IMAGE_GRID_THW_SHAPE_DESC})"
                        )
                elif grid_thw.dim() == 1:
                    if grid_thw.shape[0] == 3:
                        image_grid_thw_list.append(grid_thw)
                    elif grid_thw.shape[0] == 2:
                        h, w = grid_thw
                        grid_thw = torch.tensor(
                            [1, h, w], dtype=grid_thw.dtype, device=grid_thw.device
                        )
                        image_grid_thw_list.append(grid_thw)
                    else:
                        raise ValueError(
                            f"Invalid 1D image_grid_thw shape {grid_thw.shape} (expected 3 elements for THW)"
                        )
                else:
                    raise ValueError(
                        f"Invalid image_grid_thw dims={grid_thw.dim()} shape={grid_thw.shape} (expected {IMAGE_GRID_THW_SHAPE_DESC})"
                    )
            image_grid_thw = torch.stack(image_grid_thw_list)
            batch["image_grid_thw"] = image_grid_thw

        try:
            if "pixel_values" in batch and "image_grid_thw" in batch:
                pv = batch["pixel_values"]
                grid = batch["image_grid_thw"]
                if pv.dim() not in (2,):
                    raise ValueError(
                        f"Packed collator: pixel_values must be {PIXEL_VALUES_PACKED_SHAPE_DESC}, got {pv.shape}"
                    )
                if grid.dim() != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"Packed collator: image_grid_thw must be {IMAGE_GRID_THW_SHAPE_DESC}, got {grid.shape}"
                    )
                expected = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
                actual = int(pv.shape[0])
                if expected != actual:
                    raise ValueError(
                        f"Packed collator: pixel_values rows ({actual}) != sum(t*h*w) ({expected}) from image_grid_thw"
                    )
        except Exception as e:
            logger.error(f"Multimodal validation failure (packed): {e}")
            raise

        return batch

    @jaxtyped_beartype
    def _collate_packed(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for feature in features:
            all_input_ids.append(feature["input_ids"])
            all_attention_mask.append(feature["attention_mask"])
            all_labels.append(feature["labels"])
        packed_input_ids, packed_attention_mask, packed_labels = self._pack_sequences(
            all_input_ids, all_attention_mask, all_labels
        )
        batch = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
        }
        return batch

    @jaxtyped_beartype
    def _pack_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_length = sum(ids.size(0) for ids in input_ids)
        packed_input_ids = torch.zeros(1, total_length, dtype=torch.long)
        packed_attention_mask = torch.zeros(1, total_length, dtype=torch.long)
        packed_labels = torch.full(
            (1, total_length), self.label_pad_token_id, dtype=torch.long
        )
        offset = 0
        for ids, mask, lbl in zip(input_ids, attention_mask, labels):
            length = ids.size(0)
            packed_input_ids[0, offset : offset + length] = ids
            packed_attention_mask[0, offset : offset + length] = mask
            packed_labels[0, offset : offset + length] = lbl
            offset += length
        return packed_input_ids, packed_attention_mask, packed_labels

    @jaxtyped_beartype
    def _pad_sequence(
        self, sequences: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        flattened: List[torch.Tensor] = []
        for seq in sequences:
            # Accept [seq] or [1, seq]; squeeze the extra leading dim when present
            if seq.dim() == 1:
                flat = seq
            elif seq.dim() == 2 and seq.shape[0] == 1:
                flat = seq.squeeze(0)
            else:
                raise ValueError(
                    f"Expected 1D or [1, L] sequence tensor, got dims={seq.dim()} shape={tuple(seq.shape)}"
                )
            flattened.append(flat)
        lengths = [seq.size(0) for seq in flattened]
        max_len = max(lengths)
        padded_sequences = []
        for seq in flattened:
            padding_length = max_len - seq.size(0)
            if padding_length > 0:
                padding = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)


__all__ = ["PackedDataCollator"]
