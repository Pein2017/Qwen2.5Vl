#!/usr/bin/env python3
"""
RL conversation utilities wrapping src_new ConversationBuilder for GRPO.

- Provides helpers to load images via PathManager and build generation-ready tensors
- Mirrors SFT data preparation to preserve chat/image invariants
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from src_new.processing.conversation.builder import ConversationBuilder
from src_new.utils.path_manager import create_path_manager


@dataclass(frozen=True)
class RLConversationContext:
    """Holds objects needed to build conversations in RL runs."""

    builder: ConversationBuilder
    data_root: str
    max_coord_value: Optional[int] = None


def _load_images_abs(paths: List[str], *, data_root: str) -> List[Image.Image]:
    pm = create_path_manager(data_root)
    out: List[Image.Image] = []
    for p in paths:
        resolved = str(pm.resolve_path(p))
        img = Image.open(resolved).convert("RGB")
        out.append(img)
    return out


def build_simple_generation_inputs(
    sample: Dict[str, Any], ctx: RLConversationContext
) -> Dict[str, Any]:
    """
    Build single-turn generation tensors using the same logic as SFT.

    Expects sample to contain 'images': List[str]. Other fields (objects, width/height)
    are not required for building the prompt.
    """
    if not isinstance(sample, dict):
        raise ValueError("sample must be a dict")
    images_field = sample.get("images")
    if not isinstance(images_field, list) or len(images_field) == 0:
        raise ValueError("sample['images'] must be a non-empty list of paths")

    images = _load_images_abs(images_field, data_root=ctx.data_root)
    return ctx.builder.create_simple_conversation_for_generation(sample=sample, images=images)
