from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL
)


def _char_to_token_position(char_pos: int, offsets: torch.Tensor) -> Optional[int]:
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if int(start_char) <= char_pos < int(end_char):
            return token_idx
        elif char_pos == int(end_char) and token_idx < len(offsets) - 1:
            return token_idx + 1
    for token_idx, (start_char, _end_char) in enumerate(offsets):
        if int(start_char) >= char_pos:
            return token_idx
    return len(offsets)


@dataclass(frozen=True)
class CollatorConfig:
    include_im_end_in_span: bool = True
    strict_single_coord_token: bool = True


class DataCollatorCoordBootstrap:
    """
    Pads batch and builds assistant-only labels using offset mapping to locate spans.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[CollatorConfig] = None,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        self.tokenizer = tokenizer
        self.config = config or CollatorConfig()
        self._im_end_id: Optional[int] = None
        try:
            convert_fn = cast(
                Callable[[str], int], self.tokenizer.convert_tokens_to_ids
            )
            self._im_end_id = cast(Optional[int], convert_fn("<|im_end|>"))
        except Exception:
            self._im_end_id = None

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [f["attention_mask"] for f in features]

        labels_list: List[torch.Tensor] = []
        for input_ids in input_ids_list:
            labels = self._build_labels_for_sample(input_ids)
            labels_list.append(labels)

        # Pad inputs using tokenizer.pad by constructing a dict format it expects
        tmp_batch: List[Dict[str, Any]] = [
            {"input_ids": ids, "attention_mask": am}
            for ids, am in zip(input_ids_list, attn_list)
        ]
        padded_any: Any = self.tokenizer.pad(
            tmp_batch, padding=True, return_tensors="pt"
        )
        input_ids_padded = cast(torch.Tensor, padded_any["input_ids"])  # pyright: ignore[reportIndexIssue]
        attention_mask_padded = cast(torch.Tensor, padded_any["attention_mask"])  # pyright: ignore[reportIndexIssue]

        # Pad labels to the same sequence length
        max_len = int(input_ids_padded.shape[1])
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        if labels_padded.shape[1] < max_len:
            pad_cols = max_len - labels_padded.shape[1]
            labels_padded = torch.nn.functional.pad(
                labels_padded, (0, pad_cols), value=-100
            )
        elif labels_padded.shape[1] > max_len:
            # This should not happen; guard for safety
            labels_padded = labels_padded[:, :max_len]

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
        return out

    def _build_labels_for_sample(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Remove batch dim if present
        ids_1d = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
        labels = ids_1d.clone()

        # Decode full text
        full_text = self.tokenizer.decode(ids_1d, skip_special_tokens=False)

        # Retokenize with offsets for alignment
        toks_any: Any = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        offsets = cast(torch.Tensor, toks_any["offset_mapping"])[0]  # pyright: ignore[reportIndexIssue]

        # Find assistant content span
        m = _ASSISTANT_BLOCK_RE.search(full_text)
        if not m:
            raise ValueError("Assistant content block not found in ChatML text")
        content_start_char = m.start(1)
        content_end_char = m.end(1)

        start_tok = _char_to_token_position(content_start_char, offsets)
        end_tok = _char_to_token_position(content_end_char, offsets)
        if (
            start_tok is None
            or end_tok is None
            or not (0 <= start_tok < end_tok <= len(labels))
        ):
            raise ValueError(
                f"Invalid assistant span token bounds: {start_tok}:{end_tok} for length {len(labels)}"
            )

        final_end = end_tok
        if (
            self.config.include_im_end_in_span
            and self._im_end_id is not None
            and self._im_end_id != -1
        ):
            for pos in range(end_tok, min(len(labels), end_tok + 5)):
                if int(ids_1d[pos].item()) == int(self._im_end_id):
                    final_end = pos + 1
                    break

        # Mask all first
        labels.fill_(-100)
        # Unmask assistant span
        labels[start_tok:final_end] = ids_1d[start_tok:final_end]

        # Optional strictness: ensure exactly one coord token textual marker in assistant content
        if self.config.strict_single_coord_token:
            assistant_text = m.group(1).strip()
            if assistant_text.count("<|coord_") != 1:
                raise ValueError(
                    "Assistant content must contain exactly one '<|coord_*|>' token"
                )
            if assistant_text != assistant_text.strip():
                # basic whitespace sanity (should be exact token)
                pass

        return labels
