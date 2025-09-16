#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundational utilities for grouped LLM loss (caption/grounding/formatting).

This module intentionally avoids any tokenizer- or ID-specific logic.
It only provides the residual-assignment helper to ensure full coverage
of assistant spans by assigning any uncovered tokens to the formatting group.
"""

from __future__ import annotations

from typing import Tuple

import torch


def assign_residual_to_formatting(
	m_cap: torch.Tensor,
	m_grd: torch.Tensor,
	m_fmt: torch.Tensor,
	assist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Ensure full coverage by assigning residual assistant tokens to formatting.

	All tensors are expected to be boolean masks of identical shape.
	"""
	union = m_cap | m_grd | m_fmt
	residual = assist & (~union)
	if residual.any():
		m_fmt = m_fmt | residual
	return m_cap, m_grd, m_fmt
