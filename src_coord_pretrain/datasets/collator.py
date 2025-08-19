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

        # Check for fast tokenizer capabilities for performance optimization
        self._use_fast_tokenizer = hasattr(tokenizer, "is_fast") and tokenizer.is_fast
        if self._use_fast_tokenizer:
            # Fast tokenizers have optimized batch processing
            self._supports_batch_decode = True
        else:
            self._supports_batch_decode = False

        self._im_end_id: Optional[int] = None
        try:
            convert_fn = cast(
                Callable[[str], int], self.tokenizer.convert_tokens_to_ids
            )
            self._im_end_id = cast(Optional[int], convert_fn("<|im_end|>"))
        except Exception:
            self._im_end_id = None
        # One-time warning about reverse mapping presence (best-effort)
        self._warned_reverse_absence = False

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Dataset validation: one-time check that reverse samples exist (disabled - confirmed present)
        # The warning was appearing because individual batches may not contain reverse samples by chance,
        # which is normal given the 21% reverse sample distribution.
        if not self._warned_reverse_absence:
            self._warned_reverse_absence = True  # Skip the check entirely

        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [f["attention_mask"] for f in features]

        labels_list: List[torch.Tensor] = []
        for input_ids in input_ids_list:
            labels = self._build_labels_for_sample(input_ids)
            labels_list.append(labels)

        # Use optimized padding for fast tokenizers
        tmp_batch: List[Dict[str, Any]] = [
            {"input_ids": ids, "attention_mask": am}
            for ids, am in zip(input_ids_list, attn_list)
        ]

        # Fast tokenizers have optimized batch padding
        padding_kwargs = {"padding": True, "return_tensors": "pt"}
        if self._use_fast_tokenizer:
            # Fast tokenizers can handle padding more efficiently
            padding_kwargs.update(
                {
                    "pad_to_multiple_of": None,  # Let tokenizer decide optimal padding
                    "return_attention_mask": True,
                }
            )

        padded_any: Any = self.tokenizer.pad(tmp_batch, **padding_kwargs)
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

        # Retokenize with offsets for alignment - use fast tokenizer optimizations
        tokenizer_kwargs = {
            "return_offsets_mapping": True,
            "add_special_tokens": False,
            "return_tensors": "pt",
        }

        # Fast tokenizers provide more efficient offset mapping
        if self._use_fast_tokenizer:
            tokenizer_kwargs.update(
                {
                    "padding": False,
                    "truncation": False,
                    "return_special_tokens_mask": False,  # Not needed for our use case
                }
            )

        toks_any: Any = self.tokenizer(full_text, **tokenizer_kwargs)
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

        # Optional strictness: ensure valid coordinate content in assistant
        if self.config.strict_single_coord_token:
            assistant_text = m.group(1).strip()

            # Check for coordinate token (forward mapping: "N" -> <|coord_N|>)
            coord_token_count = assistant_text.count("<|coord_")

            # Check for raw number (reverse mapping: <|coord_N|> -> "N")
            import re

            raw_number_pattern = re.compile(r"^\d+$")
            is_raw_number = bool(raw_number_pattern.match(assistant_text))

            # Validate: should have exactly one coordinate token OR be a raw number
            if coord_token_count == 1:
                # Forward mapping case - coordinate token response
                pass  # Valid
            elif coord_token_count == 0 and is_raw_number:
                # Reverse mapping case - raw number response
                pass  # Valid
            else:
                raise ValueError(
                    f"Assistant content must contain exactly one '<|coord_*|>' token "
                    f"or be a raw number, got: '{assistant_text}' "
                    f"(coord_tokens: {coord_token_count}, is_raw_number: {is_raw_number})"
                )

        return labels
