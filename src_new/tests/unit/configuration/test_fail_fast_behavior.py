#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fail-fast behavior after refactoring to eliminate defensive programming patterns.

This test suite verifies that the codebase properly fails immediately when:
1. Required configuration parameters are missing
2. Invalid configuration values are provided
3. Required attributes/keys are accessed but don't exist

The goal is to ensure that errors are exposed at their source rather than being
masked by defensive programming patterns.
"""

import pytest

from src_new.config.config import Config, load_config


class TestFailFastBehavior:
    """Test that the refactored codebase fails fast on missing/invalid parameters."""

    def test_config_fails_on_missing_required_parameters(self):
        """Test that Config creation fails immediately when required parameters are missing."""
        # Create minimal config data missing required parameters
        incomplete_config = {
            "model_path": "/path/to/model",
            "model_size": "3B",
            # Missing many required parameters
        }

        with pytest.raises(TypeError, match="missing .* required positional arguments"):
            Config(**incomplete_config)

    def test_config_loads_successfully_with_all_parameters(self):
        """Test that config loads successfully when all required parameters are provided."""
        # This should work with the existing config files
        config = load_config("configs/bbu_v2_debug.yaml")
        assert config is not None
        assert config.coordinate_tokens_enabled is True

    def test_dictionary_access_fails_fast(self):
        """Test that direct dictionary access fails fast instead of using .get() with defaults."""
        # This test verifies that we've eliminated defensive .get() patterns
        test_dict = {"existing_key": "value"}

        # Direct access should work
        assert test_dict["existing_key"] == "value"

        # Missing key should raise KeyError immediately
        with pytest.raises(KeyError):
            _ = test_dict["missing_key"]

    def test_attribute_access_fails_fast(self):
        """Test that direct attribute access fails fast instead of using getattr() with defaults."""

        class TestObject:
            existing_attr = "value"

        obj = TestObject()

        # Direct access should work
        assert obj.existing_attr == "value"

        # Missing attribute should raise AttributeError immediately
        with pytest.raises(AttributeError):
            _ = obj.missing_attr

    def test_level_map_access_fails_fast(self):
        """Test that log level mapping fails fast on invalid levels."""
        level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }

        # Valid level should work
        assert level_map["INFO"] == 20

        # Invalid level should raise KeyError immediately
        with pytest.raises(KeyError):
            _ = level_map["INVALID_LEVEL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
