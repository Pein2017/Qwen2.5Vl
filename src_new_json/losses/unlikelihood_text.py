# DEPRECATED: Use implementations in src_new_json/losses/coord_aux.py instead.
# This module remains only for backward compatibility of digit-specific helpers.

from typing import List, Optional

import torch


def _gather_logit_rows(
    logits_all: torch.Tensor, mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Select rows from [B, T, V] logits where mask [B, T] is True.
    Returns [N, V] or None if mask has no True.
    """
    if logits_all.dim() != 3:
        raise ValueError(f"logits_all must be [B,T,V], got {tuple(logits_all.shape)}")
    if mask.shape[:2] != logits_all.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} incompatible with logits {tuple(logits_all.shape)}"
        )
    pos = mask.nonzero(as_tuple=False)
    if pos.numel() == 0:
        return None
    return logits_all[pos[:, 0], pos[:, 1], :]


def build_noncoord_vocab_mask(
    vocab_size: int, coord_start: int, coord_end_exclusive: int
) -> torch.Tensor:
    """
    DEPRECATED: import from src_new_json.losses.coord_aux instead.
    Build boolean mask [V] selecting all NON-coordinate tokens given an exclusive coord range.
    """
    if not (0 <= coord_start <= coord_end_exclusive <= vocab_size):
        raise ValueError(
            f"Invalid coord range ({coord_start},{coord_end_exclusive}) for vocab_size={vocab_size}"
        )
    noncoord = torch.ones(vocab_size, dtype=torch.bool)
    noncoord[coord_start:coord_end_exclusive] = False
    return noncoord


def unlikelihood_topk_text_generic(
    logits_all: torch.Tensor,
    text_mask: torch.Tensor,
    noncoord_vocab_mask: torch.Tensor,
    *,
    topk: int = 50,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Generic non-coordinate unlikelihood over top-k non-coordinate tokens.
    - logits_all: [B, T, V]
    - text_mask: [B, T] positions to apply the loss
    - noncoord_vocab_mask: [V] True for tokens to penalize (e.g., non-coord)
    Returns scalar tensor.
    """
    rows = _gather_logit_rows(logits_all, text_mask)
    if rows is None:
        return logits_all.new_tensor(0.0)

    # Select only non-coordinate columns
    if noncoord_vocab_mask.dtype != torch.bool or noncoord_vocab_mask.dim() != 1:
        raise ValueError("noncoord_vocab_mask must be [V] bool tensor")
    rows = rows[:, noncoord_vocab_mask]

    # Log-softmax for stability
    log_probs = torch.log_softmax(rows, dim=-1)
    k = min(topk, log_probs.shape[-1])
    topk_vals, _ = torch.topk(log_probs, k=k, dim=-1)

    # Unlikelihood loss: -log(1 - p) ~= -log(max(eps, 1 - exp(lp)))
    probs = torch.exp(topk_vals)
    penalty = -torch.log(torch.clamp(1.0 - probs, min=eps))
    return penalty.mean()


def get_digit_token_ids(tokenizer) -> List[int]:
    """Best-effort digit token ids for ASCII '0'..'9'."""
    ids: List[int] = []
    for ch in "0123456789":
        tid = tokenizer.convert_tokens_to_ids(ch)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.append(int(tid))
    return sorted(set(ids))


def unlikelihood_topk_digits(
    logits_all: torch.Tensor,
    text_mask: torch.Tensor,
    tokenizer,
    *,
    topk: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize top-k digit tokens at selected positions. Safe no-op if digits unknown.
    """
    rows = _gather_logit_rows(logits_all, text_mask)
    if rows is None:
        return logits_all.new_tensor(0.0)

    digit_ids = get_digit_token_ids(tokenizer)
    if not digit_ids:
        return logits_all.new_tensor(0.0)

    V = rows.shape[-1]
    mask = torch.zeros(V, dtype=torch.bool, device=rows.device)
    for i in digit_ids:
        if 0 <= i < V:
            mask[i] = True
    if not mask.any():
        return logits_all.new_tensor(0.0)

    log_probs = torch.log_softmax(rows[:, mask], dim=-1)
    k = min(topk, log_probs.shape[-1])
    topk_vals, _ = torch.topk(log_probs, k=k, dim=-1)
    probs = torch.exp(topk_vals)
    penalty = -torch.log(torch.clamp(1.0 - probs, min=eps))
    return penalty.mean()


__all__ = [
    "build_noncoord_vocab_mask",  # deprecated
    "unlikelihood_topk_text_generic",
    "get_digit_token_ids",
    "unlikelihood_topk_digits",
]
