"""Unified training state for GRPO trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src_new.rl.generation_buffer import GenerationBuffer


@dataclass
class GRPOTrainingState:
    """Unified state for GRPO training cycle.

    This consolidates scattered state variables into a single manager,
    making the training loop cleaner and buffer reuse explicit.
    """

    global_step: int = 0
    sampler_pos: int = 0
    generation_buffer: Optional[GenerationBuffer] = None
    buffer_step_idx: int = 0  # Current position in buffer reuse cycle

    def is_buffer_exhausted(self) -> bool:
        """Check if current buffer needs refresh.

        Returns True if:
        - No buffer exists yet
        - All completions in current buffer have been used
        """
        if self.generation_buffer is None:
            return True
        return self.generation_buffer.is_exhausted()

    def reset_buffer(self) -> None:
        """Clear buffer state for new generation.

        Called when:
        - Non-finite loss detected
        - Slow generation detected
        - Any other error requiring resample
        """
        self.generation_buffer = None
        self.buffer_step_idx = 0


__all__ = ["GRPOTrainingState"]
