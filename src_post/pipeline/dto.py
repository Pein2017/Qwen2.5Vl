#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured data transfer objects used across the refactored pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class StageADiagnostics:
    """Lightweight container for Stage-A diagnostic signals."""

    values: Dict[str, float] = field(default_factory=dict)
    item_level: Optional[Dict[str, Any]] = None

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))


@dataclass(frozen=True)
class StageAResult:
    """Stage-A summaries and optional diagnostics."""

    summary_lines: List[str]
    checklist_lines: List[str]
    latency_sec: float
    diagnostics: Optional[StageADiagnostics] = None

    def validate(self, expected_image_count: Optional[int]) -> None:
        """Fail-fast validation to ensure Stage-A results align with expectations."""
        if expected_image_count is not None and expected_image_count >= 0:
            if len(self.summary_lines) != int(expected_image_count):
                raise ValueError(
                    f"Stage-A summary count ({len(self.summary_lines)}) "
                    f"does not match expected image count ({expected_image_count})."
                )
        for idx, line in enumerate(self.summary_lines):
            if not isinstance(line, str):
                raise TypeError(f"Stage-A line at index {idx} is not a string: {line!r}")
            if not line.strip():
                raise ValueError(f"Stage-A line at index {idx} is empty.")

    @property
    def preview(self) -> str:
        return ", ".join(self.summary_lines)


@dataclass(frozen=True)
class StageBCandidate:
    """Single Stage-B sample with reward metadata."""

    text: str
    token_ids: Sequence[int]
    reward: float
    label: Optional[str]
    reason: Optional[str]
    log_prob: Optional[float] = None
    generation_logps: Optional[Sequence[float]] = None

    def trimmed_reason(self, max_chars: int = 80) -> str:
        if not self.reason:
            return ""
        trimmed = self.reason.strip()
        if len(trimmed) <= max_chars:
            return trimmed
        return trimmed[: max(0, max_chars - 1)].rstrip() + "…"


@dataclass(frozen=True)
class StageBResult:
    """Collection of Stage-B candidates and derived statistics."""

    candidates: List[StageBCandidate]
    tf_p_pass: float
    tf_p_fail: float
    latency_sampling_sec: float

    best_index: Optional[int] = None
    baseline_margin: Optional[float] = None
    decision_ce: Optional[float] = None
    diagnostics: Optional["StageBSamplingDiagnostics"] = None

    def validate(self) -> None:
        if not self.candidates:
            return
        for idx, cand in enumerate(self.candidates):
            if not isinstance(cand.text, str):
                raise TypeError(f"Candidate #{idx} text is not a string: {cand.text!r}")
            if cand.token_ids is None:
                raise ValueError(f"Candidate #{idx} missing token ids.")

    @property
    def rewards(self) -> List[float]:
        return [float(c.reward) for c in self.candidates]

    @property
    def labels(self) -> List[Optional[str]]:
        return [c.label for c in self.candidates]

    @property
    def reasons(self) -> List[Optional[str]]:
        return [c.reason for c in self.candidates]

    def best_candidate(self) -> Optional[StageBCandidate]:
        if self.best_index is None:
            if not self.candidates:
                return None
            return max(self.candidates, key=lambda c: c.reward)
        if 0 <= self.best_index < len(self.candidates):
            return self.candidates[self.best_index]
        return None


@dataclass(frozen=True)
class GroupUpdate:
    """Aggregated signals for a processed group iteration."""

    stage_a: StageAResult
    stage_b: StageBResult
    rewards: Dict[str, float]
    metrics: Dict[str, float]

    def validate(self, expected_image_count: Optional[int]) -> None:
        self.stage_a.validate(expected_image_count)
        self.stage_b.validate()


@dataclass(frozen=True)
class StageBSamplingDiagnostics:
    """Reward and sampling diagnostics for Stage-B candidates."""

    reward_mean: float
    reward_std: float
    duplicate_count: int
    raw_texts: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def all_identical(self) -> bool:
        return len(self.raw_texts) > 0 and self.duplicate_count >= (len(self.raw_texts) - 1)
