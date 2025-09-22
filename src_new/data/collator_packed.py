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
        # Ignore configured max_total_length to avoid truncation; perform true packing
        self.max_length = None
        logger.info("PackedDataCollator initialized with max_length=None (no truncation; true packing)")

    @jaxtyped_beartype
    def __call__(self, features: List[Dict[str, Any]]) -> MultimodalBatch:
        has_images = any(("pixel_values" in f) and (f["pixel_values"] is not None) for f in features)
        if has_images:
            return self._collate_with_images(features)
        else:
            return self._collate_packed(features)

    @jaxtyped_beartype
    def _collate_with_images(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        # Flatten to 1D sequences if coming as [1, L]
        def _to_1d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 2 and t.shape[0] == 1:
                return t.squeeze(0)
            return t

        input_ids_1d = [_to_1d(f["input_ids"]) for f in features]
        attention_mask_1d = [_to_1d(f["attention_mask"]) for f in features]
        labels_1d = [_to_1d(f["labels"]) for f in features]
        pixel_values_list: List[torch.Tensor] = []
        for f in features:
            if "pixel_values" not in f or f["pixel_values"] is None:
                continue
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
        if pixel_values_list:
            if pixel_values_list[0].dim() == 2:
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = torch.stack(pixel_values_list)
        else:
            pixel_values = None

        # True text packing: concatenate sequences into a single row [1, sum(Li)]
        total_length = sum(int(t.size(0)) for t in input_ids_1d)
        packed_input_ids = torch.zeros(1, total_length, dtype=torch.long)
        packed_attention_mask = torch.zeros(1, total_length, dtype=torch.long)
        packed_labels = torch.full(
            (1, total_length), self.label_pad_token_id, dtype=torch.long
        )

        offsets: List[int] = []
        offset = 0
        for ids, am, lbl in zip(input_ids_1d, attention_mask_1d, labels_1d):
            length = int(ids.size(0))
            packed_input_ids[0, offset : offset + length] = ids
            packed_attention_mask[0, offset : offset + length] = am
            packed_labels[0, offset : offset + length] = lbl
            offsets.append(offset)
            offset += length
        # Only emit segment_lengths if isolation is enabled in config
        segment_lengths = None
        if self.config and getattr(self.config, 'packed_segment_isolation', False):
            segment_lengths = torch.tensor([int(t.size(0)) for t in input_ids_1d], dtype=torch.long)

        # Mask cross-sample boundaries to avoid learning transitions across samples
        # For next-token CE (shifted), mask the first token of each subsequent sample
        for b_off in offsets[1:]:
            packed_labels[0, b_off] = self.label_pad_token_id

        # Merge and offset spans into a single row when present
        def _offset_merge_spans(spans_list: List[List[tuple]], offs: List[int]) -> List[tuple]:
            merged: List[tuple] = []
            for spans, o in zip(spans_list, offs):
                if spans:
                    for s, e in spans:
                        merged.append((int(s) + o, int(e) + o))
            return merged

        batch = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
            "segment_lengths": segment_lengths,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if "conversation_variant" in features[0]:
            batch["conversation_variant"] = features[0]["conversation_variant"]

        if "teacher_assistant_spans" in features[0]:
            t_spans_lists = [f["teacher_assistant_spans"] for f in features]
            merged_t = _offset_merge_spans(t_spans_lists, offsets)
            batch["teacher_assistant_spans"] = [merged_t]
        if "student_assistant_spans" in features[0]:
            s_spans_lists = [f["student_assistant_spans"] for f in features]
            merged_s = _offset_merge_spans(s_spans_lists, offsets)
            batch["student_assistant_spans"] = [merged_s]
        if "assistant_spans" in features[0]:
            a_spans_lists = [f["assistant_spans"] for f in features]
            merged_a = _offset_merge_spans(a_spans_lists, offsets)
            batch["assistant_spans"] = [merged_a]


        # Provide the original item count for diagnostics (ignored by model forward)
        try:
            batch["num_items_in_batch"] = torch.tensor([len(features)], dtype=torch.long)
        except Exception:
            pass

        image_grid_present = any(
            ("image_grid_thw" in f) and (f["image_grid_thw"] is not None) for f in features
        )
        if image_grid_present:
            image_grid_thw_list = []
            for f in features:
                grid_thw = f.get("image_grid_thw")
                if grid_thw is None:
                    continue
                if grid_thw.dim() == 2:
                    if grid_thw.shape[0] == 1 and grid_thw.shape[1] == 3:
                        image_grid_thw_list.append(grid_thw.squeeze(0))
                    elif grid_thw.shape[0] >= 1 and grid_thw.shape[1] == 3:
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
                        image_grid_thw_list.append(
                            torch.tensor([1, h, w], dtype=grid_thw.dtype, device=grid_thw.device)
                        )
                    else:
                        raise ValueError(
                            f"Invalid 1D image_grid_thw shape {grid_thw.shape} (expected 3 elements for THW)"
                        )
                else:
                    raise ValueError(
                        f"Invalid image_grid_thw dims={grid_thw.dim()} shape={grid_thw.shape} (expected {IMAGE_GRID_THW_SHAPE_DESC})"
                    )
            if image_grid_thw_list:
                image_grid_thw = torch.stack(image_grid_thw_list)
                batch["image_grid_thw"] = image_grid_thw
            else:
                image_grid_thw = None
        else:
            image_grid_thw = None

        if pixel_values is not None and image_grid_thw is None:
            raise ValueError(
                "Packed collator: pixel_values present but image_grid_thw missing; cannot build multimodal batch."
            )

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
