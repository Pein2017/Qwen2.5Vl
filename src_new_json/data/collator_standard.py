from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new_json.types.arrays import (
    jaxtyped_beartype,
)
from src_new_json.types.batch import MultimodalBatch
from src_new_json.types.shapes import (
    IMAGE_GRID_THW_SHAPE_DESC,
    PIXEL_VALUES_STANDARD_SHAPE_DESC,
)

from .collator_utils import get_collator_logger


if TYPE_CHECKING:
    from src_new_json.config.config import Config

logger = get_collator_logger()


@dataclass
class StandardDataCollator:
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
        logger.info(
            f"StandardDataCollator initialized with max_length={self.max_length}"
        )

    @jaxtyped_beartype
    def __call__(self, features: List[Dict[str, Any]]) -> MultimodalBatch:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        pixel_values = None
        image_grid_thw = None
        if "pixel_values" in features[0]:
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
                        f"Invalid pixel_values dims={pv.dim()} shape={pv.shape} (expected {PIXEL_VALUES_STANDARD_SHAPE_DESC})"
                    )
            if pixel_values_list and pixel_values_list[0].dim() == 2:
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = torch.stack(pixel_values_list)
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

        padded_input_ids = self._pad_sequence(input_ids, self.pad_token_id)
        padded_attention_mask = self._pad_sequence(attention_mask, 0)
        padded_labels = self._pad_sequence(labels, self.label_pad_token_id)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
            logger.debug(
                f"Added image_grid_thw to batch: {image_grid_thw.shape} = {image_grid_thw}"
            )


        # Propagate assistant spans so grouped losses (caption/grounding/formatting) work identically to packed
        try:
            if "teacher_assistant_spans" in features[0]:
                batch["teacher_assistant_spans"] = [f.get("teacher_assistant_spans", []) for f in features]
            if "student_assistant_spans" in features[0]:
                batch["student_assistant_spans"] = [f.get("student_assistant_spans", []) for f in features]
            if "assistant_spans" in features[0]:
                # Unified spans fallback (treated as student spans in the wrapper/loss manager)
                batch["assistant_spans"] = [f.get("assistant_spans", []) for f in features]
            if "conversation_variant" in features[0]:
                batch["conversation_variant"] = features[0]["conversation_variant"]
            # Provide original batch size for diagnostics (optional)
            try:
                batch["num_items_in_batch"] = torch.tensor([len(features)], dtype=torch.long)
            except Exception:
                pass
            # Fail-fast: ensure at least student or unified spans present per item
            spans_all = batch.get("student_assistant_spans") or batch.get("assistant_spans")
            if not spans_all or not any(len(s) > 0 for s in spans_all):
                raise ValueError("Collator: missing assistant spans in batch; cannot compute grouped losses.")
        except Exception as e:
            logger.error(f"Failed to attach assistant spans in standard collator: {e}")
            raise

        try:
            if "pixel_values" in batch and "image_grid_thw" in batch:
                pv = batch["pixel_values"]
                grid = batch["image_grid_thw"]
                if pv.dim() not in (2,):
                    raise ValueError(
                        f"Standard collator: pixel_values must be {PIXEL_VALUES_STANDARD_SHAPE_DESC}, got {pv.shape}"
                    )
                if grid.dim() != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"Standard collator: image_grid_thw must be {IMAGE_GRID_THW_SHAPE_DESC}, got {grid.shape}"
                    )
                expected = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
                actual = int(pv.shape[0])
                if expected != actual:
                    raise ValueError(
                        f"Standard collator: pixel_values rows ({actual}) != sum(t*h*w) ({expected}) from image_grid_thw"
                    )
        except Exception as e:
            logger.error(f"Multimodal validation failure (standard): {e}")
            raise

        return batch

    @jaxtyped_beartype
    def _pad_sequence(
        self, sequences: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        lengths = [seq.size(0) for seq in sequences]
        max_len = max(lengths)
        padded_sequences = []
        for seq in sequences:
            padding_length = max_len - seq.size(0)
            if padding_length > 0:
                padding = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)


__all__ = ["StandardDataCollator"]
