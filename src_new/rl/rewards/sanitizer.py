"""Utilities to sanitize truncated/non‑EOS completions for reward evaluation.

Default policy: when a completion did not terminate with EOS or was truncated,
remove only the last complete geometry block (box/quad/line) from the text.
This avoids scoring an ambiguous tail without affecting length-wise rewards.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _find_rightmost_geometry_span(text: str) -> Optional[Tuple[int, int]]:
    """Find the [start, end) span of the rightmost complete geometry block using string search.

    This avoids regex backtracking on long sequences. We search for the last
    occurrence of any end token, then locate the corresponding start token by
    searching backwards.
    """
    pairs = (
        ("<|box_start|>", "<|box_end|>"),
        ("<|quad_start|>", "<|quad_end|>"),
        ("<|line_start|>", "<|line_end|>"),
    )
    # Find the rightmost end token among all types
    best_end_pos = -1
    best_pair: Optional[Tuple[str, str]] = None
    for start_tok, end_tok in pairs:
        pos = text.rfind(end_tok)
        if pos > best_end_pos:
            best_end_pos = pos
            best_pair = (start_tok, end_tok)
    if best_end_pos < 0 or best_pair is None:
        return None
    start_tok, end_tok = best_pair
    # Find the last start token strictly before the chosen end token
    start_pos = text.rfind(start_tok, 0, best_end_pos)
    if start_pos < 0:
        return None
    end_pos = best_end_pos + len(end_tok)
    return (start_pos, end_pos)


def sanitize_tail_geometry_block(text: str) -> str:
    """Remove the rightmost complete geometry block from text, if any.

    If no complete block is present, returns the original text unchanged.
    """
    if not text:
        return text
    span = _find_rightmost_geometry_span(text)
    if span is None:
        return text
    s, e = span
    # Drop exactly the matched geometry block; keep everything else intact.
    return f"{text[:s]}{text[e:]}"


__all__ = ["sanitize_tail_geometry_block"]
