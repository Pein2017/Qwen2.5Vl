from typing import List

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - torch should exist in ms env
    torch = None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_gpu: mark test as needing at least one CUDA device")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    if torch is None or not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="Requires GPU but no CUDA device available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip)
