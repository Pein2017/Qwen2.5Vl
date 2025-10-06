"""Minimal distributed helpers for the manual RL trainer."""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.distributed as dist


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if _is_dist():
        dist.barrier()


def broadcast_indices(indices: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast index tensor from ``src`` across ranks."""

    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    if _is_dist():
        indices = indices.contiguous()
        dist.broadcast(indices, src=src)
    return indices


def broadcast_gen_cfg(gen_cfg: dict[str, Any], src: int = 0) -> dict[str, Any]:
    """Broadcast a small config mapping via CPU tensor serialization."""

    if not _is_dist():
        return gen_cfg

    if dist.get_rank() == src:
        payload_bytes = json.dumps(gen_cfg).encode("utf-8")
        payload = torch.tensor(
            list(payload_bytes), dtype=torch.uint8, device="cpu"
        )
    else:
        payload = torch.empty(0, dtype=torch.uint8, device="cpu")
    lengths = torch.tensor([payload.numel()], dtype=torch.long, device="cpu")
    dist.broadcast(lengths, src=src)
    if payload.numel() != int(lengths.item()):
        payload = torch.empty(int(lengths.item()), dtype=torch.uint8, device="cpu")
    dist.broadcast(payload, src=src)
    if dist.get_rank() == src:
        return gen_cfg
    data = bytes(payload.cpu().tolist())
    return json.loads(data.decode("utf-8"))


def all_gather_rewards(rewards: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return rewards
    gathered = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, rewards)
    return torch.cat(gathered, dim=0)


def compute_global_advantages(local_advantages: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return local_advantages
    gathered = all_gather_rewards(local_advantages)
    mean = gathered.mean()
    std = gathered.std(unbiased=False)
    std = torch.clamp(std, min=1e-4)
    normed = (local_advantages - mean) / std
    return normed


def seed_for_rank(base_seed: int) -> int:
    if not _is_dist():
        return base_seed
    return base_seed + dist.get_rank()


__all__ = [
    "barrier",
    "broadcast_indices",
    "broadcast_gen_cfg",
    "all_gather_rewards",
    "compute_global_advantages",
    "seed_for_rank",
]
