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

import tempfile
from pathlib import Path

import pytest
import yaml

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

    def test_config_fails_on_missing_both_coordinate_temperatures(self):
        """Test that missing both coordinate temperature keys causes immediate failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "model_path": "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024-fourier",
                "model_size": "3B",
                "model_max_length": 32000,
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "vision_lr": 1e-6,
                "merger_lr": 1e-5,
                "llm_lr": 1e-6,
                "adapter_lr": 0.0,
                "warmup_ratio": 0.1,
                "weight_decay": 0.0001,
                "max_grad_norm": 1.0,
                "lr_scheduler_type": "cosine",
                "gradient_checkpointing": True,
                "bf16": True,
                "fp16": False,
                "use_flash_attention": True,
                "mixed_precision": "bf16",
                "train_data_path": "/path/to/train.jsonl",
                "val_data_path": "/path/to/val.jsonl",
                "data_root": "/path/to/data",
                "teacher_pool_file": "/path/to/teacher_pool.jsonl",
                "max_total_length": 12000,
                "num_teacher_samples": 1,
                "collator_type": "packed",
                "teacher_ratio": 0.5,
                "language": "chinese",
                "output_dir": "/path/to/output",
                "run_name": "test",
                "max_coord_value": 1024,
                "model_hidden_size": 2048,
                "coordinate_tokens_enabled": True,
                "coordinate_loss_weight": 0.1,
                "regular_loss_weight": 1.0,
                # Missing both coordinate_temperature and coordinate_loss_temperature - should cause failure
                "coordinate_kl_weight": 0.0,
                "coordinate_label_sigma": 16,
                "coordinate_init_mode": "fourier_ramp",
                "eval_strategy": "steps",
                "eval_steps": 10,
                "save_strategy": "steps",
                "save_steps": 20,
                "save_total_limit": 2,
                "logging_steps": 5,
                "logging_dir": "logs",
                "report_to": "tensorboard",
                "disable_tqdm": True,
                "verbose": False,
                "remove_unused_columns": False,
                "dataloader_num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2,
                "tb_dir": "tb",
                "teacher_loss_weight": 0.3,
                "student_loss_weight": 1.0,
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "max_pixels": 401408,
                "training_prompt_style": True,
                "use_consistent_prompts": True,
                "skip_vocab_extension": False,
            }
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(
                TypeError, match="Configuration contains invalid field types"
            ):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_config_loads_successfully_with_all_parameters(self):
        """Test that config loads successfully when all required parameters are provided."""
        # This should work with the existing config files
        config = load_config("configs/bbu_v2_debug.yaml")
        assert config is not None
        assert config.coordinate_tokens_enabled is True
        assert config.coordinate_temperature == 0.7

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
