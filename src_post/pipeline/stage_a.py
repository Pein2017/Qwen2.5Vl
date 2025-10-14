#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A sampling utilities."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from PIL import Image
from transformers.generation import LogitsProcessorList

from src_post.generation.generation import build_stage_a_context_lines
from src_post.logging.diagnostics import (
    build_item_level_diagnostics,
    compute_phase_a_diagnostics,
)
from src_post.pipeline.dto import StageADiagnostics, StageAResult


@dataclass
class StageASamplerConfig:
    sanitize: bool
    mission: Optional[str]
    enable_diagnostics: bool


class StageASampler:
    """Encapsulates Stage-A context generation + diagnostics."""

    def __init__(
        self,
        policy,
        processor,
        conv_builder,
        generation_config,
        base_logits_processors: Optional[Sequence] = None,
        stopping=None,
        config: Optional[StageASamplerConfig] = None,
    ) -> None:
        self._policy = policy
        self._processor = processor
        self._conv_builder = conv_builder
        self._generation_config = generation_config
        self._base_logits_processors = list(base_logits_processors or [])
        self._stopping = stopping
        self._config = config or StageASamplerConfig(
            sanitize=True, mission=None, enable_diagnostics=False
        )

    def _clone_logits_processors(self) -> LogitsProcessorList:
        return LogitsProcessorList(list(self._base_logits_processors))

    def generate(
        self, images: List[Image.Image], enable_diagnostics: Optional[bool] = None
    ) -> StageAResult:
        """Run Stage-A greedy decoding and optional diagnostics."""

        diag_enabled = (
            self._config.enable_diagnostics
            if enable_diagnostics is None
            else bool(enable_diagnostics)
        )
        logits_processors = self._clone_logits_processors()
        t_start = time.time()
        summary_lines = build_stage_a_context_lines(
            policy=self._policy,
            processor=self._processor,
            images=images,
            gen_cfg=self._generation_config,
            logits_processors=logits_processors,
            stopping=self._stopping,
            sanitize=bool(self._config.sanitize),
            mission=self._config.mission,
            conv_builder=self._conv_builder,
        )
        latency = time.time() - t_start
        checklist = self._conv_builder.rules_for_mission(self._config.mission)

        diagnostics = None
        if diag_enabled:
            try:
                diag_values = compute_phase_a_diagnostics(
                    summary_lines, checklist, mission=self._config.mission
                )
                item_diag = build_item_level_diagnostics(
                    summary_lines=summary_lines, mission=self._config.mission
                )
                diagnostics = StageADiagnostics(
                    values=dict(diag_values or {}), item_level=item_diag
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Stage-A diagnostics failed: {exc}") from exc

        return StageAResult(
            summary_lines=list(summary_lines),
            checklist_lines=list(checklist or []),
            latency_sec=float(latency),
            diagnostics=diagnostics,
        )
