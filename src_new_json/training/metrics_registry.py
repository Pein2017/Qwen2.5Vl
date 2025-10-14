#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central registry and utilities for training/eval metrics.

- Defines canonical role and group names
- Provides helpers to compute group-total metrics from flattened logs
- Keeps logic in one place so Trainer and adapters can reuse
"""
from __future__ import annotations

from typing import Dict


ROLE_NAMES: tuple[str, str] = ("teacher", "student")
GROUP_NAMES: tuple[str, str, str] = ("caption", "grounding", "formatting")


def _to_float(value) -> float | None:
	try:
		return float(value)
	except Exception:
		return None


def compute_group_totals_from_flat(
	logs: Dict[str, float], *, prefix: str | None = None
) -> Dict[str, float]:
	"""Compute group-total losses from flattened per-role logs.

	Expected flattened keys present in logs:
	- "teacher_caption_loss", "student_caption_loss", etc.

	Emits (with optional prefix):
	- "{prefix}/group_caption_loss_total", etc.
	"""
	out: Dict[str, float] = {}
	for group in GROUP_NAMES:
		key_t = f"teacher_{group}_loss"
		key_s = f"student_{group}_loss"
		vt = _to_float(logs.get(key_t))
		vs = _to_float(logs.get(key_s))
		if vt is None and vs is None:
			continue
		total = (vt or 0.0) + (vs or 0.0)
		name = f"group_{group}_loss_total"
		if prefix:
			name = f"{prefix}/{name}"
		out[name] = total
	return out
