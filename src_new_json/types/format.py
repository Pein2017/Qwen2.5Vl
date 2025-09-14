#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum


class FormatMode(str, Enum):
	SPECIAL_TOKENS = "special_tokens"
	COORD_TOKENS = "coord_tokens"


class ConversationVariant(str, Enum):
	DENSE_CAPTION = "dense_caption"
	COORDS_TO_DESC = "coords_to_desc"
	DESC_TO_COORDS = "desc_to_coords"
	SUMMARY = "summary"


__all__ = ["FormatMode", "ConversationVariant"]
