from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")


def _is_absolute(path: str) -> bool:
    p = Path(path)
    return p.is_absolute()


def _read_jsonl_strict(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if not p.is_file():
        raise ValueError(f"Expected a file at data_path, got non-file: {path}")
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e.msg}"
                ) from e
            records.append(obj)
    return records


@dataclass(frozen=True)
class DatasetConfig:
    data_path: str
    max_coord_value: int
    use_apply_chat_template: bool

    def validate(self) -> None:
        if not _is_absolute(self.data_path):
            raise ValueError(
                f"data_path must be an absolute path, got: {self.data_path!r}"
            )
        if not isinstance(self.max_coord_value, int) or self.max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be a positive int, got: {self.max_coord_value!r}"
            )
        if not isinstance(self.use_apply_chat_template, bool):
            raise ValueError(
                f"use_apply_chat_template must be a bool, got {type(self.use_apply_chat_template)}"
            )


class CoordBootstrapDataset(TorchDataset):
    """
    Text-only ChatML dataset for coordinate token bootstrapping.

    Each record is a dict with fields:
    - messages: List[{"role": "user"|"assistant", "content": str}] with exactly two turns
    - meta: optional metadata used for checks

    Assistant content must be exactly one coordinate token: <|coord_N|> with 0 <= N <= max_coord_value.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DatasetConfig,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        self.tokenizer = tokenizer
        if config is None:
            raise ValueError("config cannot be None")
        config.validate()
        self.config = config

        self._raw = _read_jsonl_strict(config.data_path)
        self._records = self._validate_and_filter(self._raw)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self._records[idx]
        messages = rec["messages"]
        if self.config.use_apply_chat_template:
            try:
                encoded_any: Any = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_tensors="pt",
                )
                encoded = cast(Mapping[str, torch.Tensor], encoded_any)
            except Exception as e:
                raise ValueError(
                    f"apply_chat_template failed. Ensure the tokenizer supports ChatML. Error: {type(e).__name__}: {e}"
                )
            input_ids = cast(torch.Tensor, encoded["input_ids"])[0]
            attention_mask = cast(torch.Tensor, encoded["attention_mask"])[0]
        else:
            # Manual ChatML serialization (explicit opt-in)
            parts: List[str] = []
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    raise ValueError(
                        "Each message must have string 'role' and 'content'"
                    )
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            joined = "".join(parts)
            tok_any: Any = self.tokenizer(
                joined,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tok = cast(Mapping[str, torch.Tensor], tok_any)
            input_ids = cast(torch.Tensor, tok["input_ids"])[0]
            attention_mask = cast(torch.Tensor, tok["attention_mask"])[0]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _validate_and_filter(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not raw:
            raise ValueError("No data found in JSONL file")
        kept: List[Dict[str, Any]] = []
        for i, rec in enumerate(raw):
            try:
                self._validate_record(rec)
            except Exception as e:
                raise ValueError(
                    f"Invalid record at index {i}: {type(e).__name__}: {e}"
                )
            kept.append(rec)
        return kept

    def _validate_record(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            raise ValueError("Record must be a dict")
        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(
                "Record must contain 'messages' list with exactly 2 turns (user, assistant)"
            )
        roles = [m.get("role") for m in messages]
        if roles != ["user", "assistant"]:
            raise ValueError(f"Roles must be ['user','assistant'], got {roles}")
        assistant_content = messages[1].get("content")
        if not isinstance(assistant_content, str):
            raise ValueError("Assistant content must be a string")
        m = _COORD_TOKEN_RE.match(assistant_content)
        if not m:
            raise ValueError(
                "Assistant content must be exactly one coord token like '<|coord_123|>'"
            )
        val = int(m.group(1))
        if not (0 <= val <= int(self.config.max_coord_value)):
            raise ValueError(
                f"Coordinate value out of range: {val}; expected 0..{self.config.max_coord_value}"
            )
