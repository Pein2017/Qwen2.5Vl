#!/usr/bin/env python3
"""Callback to dump full conversation texts (prompt + completion) during TRL training.

Non-invasive: can be enabled via runner hook or user code by adding the callback to the trainer.
Writes JSONL files under a target directory, one file per evaluation pass.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


class ConversationDumpCallback(TrainerCallback):
    """Dumps prompt/completion pairs as full conversation text to JSONL.

    Each line contains: {
        "step": int,
        "rank": int,
        "prompt_text": str,       # includes system/user and <|image_pad|>
        "completion_text": str,   # assistant completion (may include <|im_end|>)
        "conversation_text": str  # prompt + completion
    }
    """

    def __init__(
        self,
        output_dir: str,
        *,
        max_samples_per_step: int = 16,
        file_prefix: str = "step",
        overwrite_step_files: bool = True,
    ) -> None:
        self.output_dir = str(output_dir)
        self.max_samples_per_step = int(max_samples_per_step)
        self.file_prefix = str(file_prefix)
        self.overwrite_step_files = bool(overwrite_step_files)
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_trainer(self) -> Any | None:
        # Transformers sets `callback.trainer = trainer` when adding callbacks
        return getattr(self, "trainer", None)

    def _get_rank(self, trainer: Any) -> int:
        try:
            acc = getattr(trainer, "accelerator", None)
            if acc is None:
                return 0
            return int(getattr(acc.state, "process_index", 0))
        except Exception:
            return 0

    def _is_main(self, trainer: Any) -> bool:
        try:
            acc = getattr(trainer, "accelerator", None)
            if acc is None:
                return True
            return bool(acc.is_main_process)
        except Exception:
            return True

    def _extract_texts(self, trainer: Any) -> tuple[List[str], List[str]]:
        logs = getattr(trainer, "_textual_logs", None)
        if not isinstance(logs, dict):
            raise RuntimeError(
                "ConversationDumpCallback: trainer._textual_logs is missing or not a dict"
            )
        prompts = list(logs.get("prompt", []))
        completions = list(logs.get("completion", []))
        return prompts, completions

    # No-op for training steps (eval-only hook)
    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        return

    # Dump after each evaluation phase
    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        trainer = self._get_trainer()
        if trainer is None:
            raise RuntimeError("ConversationDumpCallback: trainer reference not set")
        if not self._is_main(trainer):
            return

        prompts, completions = self._extract_texts(trainer)
        if not prompts or not completions:
            raise RuntimeError(
                "ConversationDumpCallback: no prompt/completion texts captured for this eval pass"
            )

        n = min(self.max_samples_per_step, len(prompts), len(completions))
        if n <= 0:
            raise RuntimeError(
                "ConversationDumpCallback: computed zero samples to dump at evaluation"
            )

        try:
            step = int(getattr(state, "global_step", 0) or 0)
        except Exception:
            step = 0

        filename = f"{self.file_prefix}-{step}.jsonl"
        path = os.path.join(self.output_dir, filename)
        mode = "w" if self.overwrite_step_files else "a"

        rank = self._get_rank(trainer)

        with open(path, mode, encoding="utf-8") as f:
            for i in range(-n, 0):
                prompt_text = prompts[i]
                completion_text = completions[i]
                row = {
                    "step": step,
                    "rank": rank,
                    "prompt_text": prompt_text,
                    "completion_text": completion_text,
                    "conversation_text": f"{prompt_text}{completion_text}",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["ConversationDumpCallback"]
