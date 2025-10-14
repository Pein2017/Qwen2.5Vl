#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO training loop orchestrator for src_post."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src_post.pipeline import MetricsAggregator


@dataclass
class TrainerComponents:
    model: nn.Module
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]


class GRPOTrainer:
    """Thin wrapper around the GRPO optimization loop.

    This class centralizes gradient accumulation coordination, metric
    aggregation, and optimizer stepping while leaving sampling,
    reward computation, and loss construction to the caller.
    """

    def __init__(
        self,
        components: TrainerComponents,
        grad_accum_steps: int,
        metrics_window: int = 50,
    ) -> None:
        self.components = components
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.step_counter = 0
        self.accum_counter = 0
        self.metrics = MetricsAggregator(window_size=metrics_window)

    def zero_grad(self) -> None:
        self.components.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.components.optimizer.step()
        if self.components.scheduler is not None:
            try:
                self.components.scheduler.step()
            except Exception:
                pass
        self.zero_grad()
        self.step_counter += 1

    def should_sync(self) -> bool:
        return (self.accum_counter + 1) % self.grad_accum_steps == 0

    def advance_accum(self) -> None:
        self.accum_counter += 1

    def update_metrics(self, scalars: Dict[str, float]) -> Dict[str, float]:
        return self.metrics.update(scalars)

    @property
    def global_step(self) -> int:
        return self.step_counter
