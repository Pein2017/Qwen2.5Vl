from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new.data.collator_shared import (
    extract_image_grid_thw,
    extract_pixel_values,
    pad_sequence,
    validate_multimodal_batch,
)
from src_new.types.arrays import jaxtyped_beartype
from src_new.types.batch import MultimodalBatch

from .collator_utils import get_collator_logger


if TYPE_CHECKING:
    from src_new.config.config import Config

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
        # Ignore configured max_total_length to avoid truncation; pad to longest in batch
        self.max_length = None
        logger.info(
            "StandardDataCollator initialized with max_length=None (no truncation; pad to longest in batch)"
        )

    @jaxtyped_beartype
    def __call__(self, features: List[Dict[str, Any]]) -> MultimodalBatch:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        pixel_values = extract_pixel_values(features)
        image_grid_thw = extract_image_grid_thw(features)

        padded_input_ids = pad_sequence(input_ids, self.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, 0)
        padded_labels = pad_sequence(labels, self.label_pad_token_id)

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
        validate_multimodal_batch(pixel_values, image_grid_thw)

        # Propagate assistant spans so grouped losses (caption/grounding/formatting) work identically to packed
        try:
            if "teacher_assistant_spans" in features[0]:
                batch["teacher_assistant_spans"] = [
                    f.get("teacher_assistant_spans", []) for f in features
                ]
            if "student_assistant_spans" in features[0]:
                batch["student_assistant_spans"] = [
                    f.get("student_assistant_spans", []) for f in features
                ]
            if "assistant_spans" in features[0]:
                # Unified spans fallback (treated as student spans in the wrapper/loss manager)
                batch["assistant_spans"] = [
                    f.get("assistant_spans", []) for f in features
                ]
            if "conversation_variant" in features[0]:
                batch["conversation_variant"] = features[0]["conversation_variant"]
            # Provide original batch size for diagnostics (optional)
            try:
                batch["num_items_in_batch"] = torch.tensor(
                    [len(features)], dtype=torch.long
                )
            except Exception:
                pass
            # Fail-fast: ensure at least student or unified spans present per item
            spans_all = batch.get("student_assistant_spans") or batch.get(
                "assistant_spans"
            )
            if not spans_all or not any(len(s) > 0 for s in spans_all):
                raise ValueError(
                    "Collator: missing assistant spans in batch; cannot compute grouped losses."
                )
        except Exception as e:
            logger.error(f"Failed to attach assistant spans in standard collator: {e}")
            raise

        return batch


__all__ = ["StandardDataCollator"]
