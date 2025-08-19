import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from src_new.types.batch import MultimodalBatch


def get_collator_logger() -> logging.Logger:
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("collator")
    except ImportError:
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
class TrainerCompatibleDataCollator:
    base_collator: Any

    def __call__(self, features: List[Dict[str, Any]]) -> MultimodalBatch:
        batch = self.base_collator(features)
        return self._validate_batch(batch)  # type: ignore[return-value]

    def _validate_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Ensure required fields are present
        required_fields = ["input_ids", "attention_mask", "labels"]
        for field in required_fields:
            if field not in batch:
                raise ValueError(
                    f"Missing required field '{field}' in batch (fail-fast). "
                    f"Present keys: {sorted(list(batch.keys()))}"
                )
        # Ensure all tensors are at least 2D
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                batch[key] = value.unsqueeze(0)
        return batch


__all__ = [
    "get_collator_logger",
    "TrainerCompatibleDataCollator",
]
