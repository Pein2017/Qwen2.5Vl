from __future__ import annotations

import os
import random


def seed_everything(
    seed: int, deterministic: bool = False, set_hf_seed: bool = True
) -> None:
    """Seed all known randomness sources for reproducibility.

    Args:
            seed: Non-negative integer seed value.
            deterministic: If True, enable deterministic behaviors in PyTorch backends when possible (default False).
            set_hf_seed: If True, also call transformers.set_seed to align HF internals.

    Raises:
            ValueError: If seed is invalid.
    """
    if seed is None:
        raise ValueError("seed cannot be None")
    try:
        seed_value: int = int(seed)
    except Exception as exc:  # pragma: no cover (defensive)
        raise ValueError(f"seed must be an int; got {seed!r}") from exc
    if seed_value < 0:
        raise ValueError(f"seed must be non-negative; got {seed_value}")

    # Python hashing and PRNG
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)

    # NumPy
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed_value)
    except Exception:
        pass

    # PyTorch
    try:
        import torch  # type: ignore

        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                # Enforce algorithmic determinism where supported
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
                # Required by cuBLAS for determinism in some ops
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            except Exception:
                pass
    except Exception:
        pass

    # HuggingFace transformers seeding (covers datasets/dataloaders used by HF)
    if set_hf_seed:
        try:
            from transformers import set_seed as hf_set_seed  # type: ignore

            hf_set_seed(seed_value)
        except Exception:
            pass
