"""
Pytest Configuration for BBU Training Pipeline Tests

This file provides pytest configuration, fixtures, and test utilities
for the comprehensive BBU training pipeline test suite.
"""

import os
import sys
from pathlib import Path

import pytest


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from src.logger_utils import configure_global_logging


def pytest_configure(config):
    """Configure pytest environment."""
    # Configure logging for tests
    configure_global_logging(rank=0, world_size=1)

    # Set environment variables for testing
    os.environ["BBU_TEST_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings

    print("🧪 BBU Training Pipeline Test Suite")
    print("=" * 50)

    # Configure custom pytest markers
    config.addinivalue_line("markers", "slow: marks tests as slow (integration tests)")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "data: marks tests for data pipeline")
    config.addinivalue_line("markers", "model: marks tests for model loading")
    config.addinivalue_line("markers", "training: marks tests for training components")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add slow marker to integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add gpu marker to tests that require GPU
        if any(
            keyword in item.name.lower()
            for keyword in ["memory", "forward", "training"]
        ):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def test_environment():
    """Provide test environment information."""
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "project_root": project_root,
        "test_mode": True,
    }


@pytest.fixture(scope="session")
def cleanup_on_exit():
    """Ensure cleanup happens even if tests fail."""
    yield

    # Cleanup any remaining temporary files
    import shutil
    import tempfile

    temp_dir = Path(tempfile.gettempdir())

    for temp_path in temp_dir.glob("bbu_test_data_*"):
        try:
            if temp_path.is_dir():
                shutil.rmtree(temp_path)
                print(f"🧹 Cleaned up: {temp_path}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup {temp_path}: {e}")
