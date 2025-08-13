#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Configuration Validation Tests

This module tests the unified configuration system's validation capabilities,
ensuring all required fields are properly validated and defaults are applied.

Key Features:
- Tests configuration loading from YAML files
- Tests field validation and type checking
- Tests default value application
- Tests error handling for invalid configurations
- Tests configuration completeness
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src_new.config.config import Config, load_config


class TestConfigValidation:
    """Test suite for configuration validation."""

    def _setup_temp_paths(self, config_dict):
        """Helper method to set up temporary paths for configuration testing."""
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()

        # Update paths to use temporary directories
        config_dict["model_path"] = str(model_dir)
        config_dict["data_root"] = str(data_dir)
        config_dict["train_data_path"] = str(data_dir / "train.jsonl")
        config_dict["val_data_path"] = str(data_dir / "val.jsonl")
        config_dict["teacher_pool_file"] = str(data_dir / "teacher_pool.jsonl")
        config_dict["output_dir"] = str(Path(temp_dir) / "output")

        return temp_dir

    @pytest.fixture
    def valid_config_dict(self):
        """Create a valid configuration dictionary with all required fields."""
        return {
            # === REQUIRED FIELDS ===
            # Model settings
            "model_path": "/fake/path/to/model",
            "model_size": "3B",
            "model_max_length": 32000,
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            # Training settings
            "num_train_epochs": 20,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-6,
            "vision_lr": 5e-7,
            "merger_lr": 1e-5,
            "llm_lr": 5e-6,
            "adapter_lr": 0.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0001,
            "max_grad_norm": 0.5,
            "lr_scheduler_type": "cosine",
            "gradient_checkpointing": True,
            "bf16": True,
            "fp16": False,
            "use_flash_attention": True,
            "mixed_precision": "bf16",
            # Data settings - only data_root needed, resolver will handle the rest
            "data_root": "data",
            "max_total_length": 12000,
            "num_teacher_samples": 1,
            "collator_type": "packed",
            "teacher_ratio": 0.5,
            "language": "chinese",
            # Output settings
            "output_dir": "output",
            "run_name": "test_run",
            "max_coord_value": 1024,
            "model_hidden_size": 2048,
            # Coordinate token configuration (required)
            "coordinate_tokens_enabled": True,
            "coordinate_loss_weight": 0,  # Match bbu_v2_use_coord.yaml (int, not float)
            "regular_loss_weight": 1.0,
            "coordinate_temperature": 0.7,
            "coordinate_label_sigma": 16,  # Match YAML (int, not float)
            "coordinate_init_mode": "fourier_ramp",
            # Coordinate auxiliary losses (required)
            "coord_aux_enabled": False,
            "coord_aux_tau": 1.2,
            "coord_aux_sigma_bins": 8.0,
            "coord_aux_window_bins": 32,
            "coord_aux_topk": 100,
            "coord_aux_lambda_kce": 0.5,
            "coord_aux_lambda_unlike": 0.05,
            "coord_aux_lambda_lap1": 1e-4,
            "coord_aux_lambda_lap2": 1e-5,
            # Evaluation settings (required)
            "eval_strategy": "steps",
            "eval_steps": 500,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 2,
            # Logging settings (required)
            "logging_steps": 10,
            "logging_dir": "logs",
            "report_to": "tensorboard",
            "disable_tqdm": False,
            "verbose": False,
            # Essential settings (required)
            "remove_unused_columns": False,
            # Dataloader performance settings (required)
            "dataloader_num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            # Output settings (required)
            "tb_dir": "tb",
            # Teacher-student loss weights (required)
            "teacher_loss_weight": 0.3,
            "student_loss_weight": 1.0,
            # Vision processing parameters (required)
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "max_pixels": 401408,
            # Training control flags (required)
            "training_prompt_style": True,
            "use_consistent_prompts": True,
            # Model loading control flags (required)
            "skip_vocab_extension": False,
        }

    @pytest.fixture
    def temp_config_file(self, valid_config_dict):
        """Create a temporary configuration file."""
        # Create temporary directories for paths that need to exist
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()

        # Update paths to use temporary directories
        valid_config_dict["model_path"] = str(model_dir)
        valid_config_dict["data_root"] = str(data_dir)
        valid_config_dict["train_data_path"] = str(data_dir / "train.jsonl")
        valid_config_dict["val_data_path"] = str(data_dir / "val.jsonl")
        valid_config_dict["teacher_pool_file"] = str(data_dir / "teacher_pool.jsonl")
        valid_config_dict["output_dir"] = str(Path(temp_dir) / "output")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            return f.name

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        config = load_config(temp_config_file)

        assert isinstance(config, Config)
        assert config.model_size == "3B"
        assert config.coordinate_tokens_enabled is True
        assert config.max_coord_value == 1024

        # Clean up
        Path(temp_config_file).unlink()

    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_missing_required_fields(self, valid_config_dict):
        """Test validation with missing required fields."""
        # Remove required field
        del valid_config_dict["model_path"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
                load_config(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_invalid_data_types(self, valid_config_dict):
        """Test validation with invalid data types."""
        # Set invalid type for numeric field
        valid_config_dict["learning_rate"] = "not_a_number"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            with pytest.raises((ValueError, TypeError)):
                load_config(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_invalid_ranges(self, valid_config_dict):
        """Test validation with values outside valid ranges."""
        # Create temporary directories for paths that need to exist
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()

        # Update paths to use temporary directories
        valid_config_dict["model_path"] = str(model_dir)
        valid_config_dict["data_root"] = str(data_dir)
        valid_config_dict["train_data_path"] = str(data_dir / "train.jsonl")
        valid_config_dict["val_data_path"] = str(data_dir / "val.jsonl")
        valid_config_dict["teacher_pool_file"] = str(data_dir / "teacher_pool.jsonl")
        valid_config_dict["output_dir"] = str(Path(temp_dir) / "output")

        # Set invalid range for ratio field
        valid_config_dict["teacher_ratio"] = 1.5  # Should be between 0 and 1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            # This should raise an error for invalid teacher_ratio
            with pytest.raises(
                ValueError, match="teacher_ratio must be between 0 and 1"
            ):
                load_config(temp_file)
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_coordinate_token_settings_consistency(self, valid_config_dict):
        """Test consistency between coordinate token settings."""
        # Set up temporary paths
        temp_dir = self._setup_temp_paths(valid_config_dict)

        # Test with coordinate tokens disabled
        valid_config_dict["coordinate_tokens_enabled"] = False
        valid_config_dict["coordinate_loss_weight"] = 0.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert config.coordinate_tokens_enabled is False
            # Coordinate loss weight should be 0 when disabled
            assert config.coordinate_loss_weight == 0.0
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_teacher_student_settings_consistency(self, valid_config_dict):
        """Test consistency between teacher-student settings."""
        # Set up temporary paths
        temp_dir = self._setup_temp_paths(valid_config_dict)

        # Test with teacher ratio 0 (no teachers)
        valid_config_dict["teacher_ratio"] = 0.0
        valid_config_dict["num_teacher_samples"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert config.teacher_ratio == 0.0
            assert config.num_teacher_samples == 0
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_data_path_validation(self, valid_config_dict):
        """Test data path validation."""
        # Set up temporary paths first
        temp_dir = self._setup_temp_paths(valid_config_dict)

        # Test with non-existent data paths (should not fail loading but might warn)
        valid_config_dict["train_data_path"] = "nonexistent/train.jsonl"
        valid_config_dict["val_data_path"] = "nonexistent/val.jsonl"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            # Should load successfully (path validation might be deferred)
            config = load_config(temp_file)
            assert config.train_data_path == "nonexistent/train.jsonl"
            assert config.val_data_path == "nonexistent/val.jsonl"
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_output_directory_creation(self, valid_config_dict):
        """Test output directory handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up temporary paths first
            temp_paths_dir = self._setup_temp_paths(valid_config_dict)

            output_dir = Path(temp_dir) / "test_output"
            valid_config_dict["output_dir"] = str(output_dir)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.safe_dump(valid_config_dict, f)
                temp_file = f.name

            try:
                config = load_config(temp_file)
                assert config.output_dir == str(output_dir)
            finally:
                Path(temp_file).unlink()
                import shutil

                shutil.rmtree(temp_paths_dir)

    def test_learning_rate_settings(self, valid_config_dict):
        """Test learning rate settings validation."""
        # Set up temporary paths
        temp_dir = self._setup_temp_paths(valid_config_dict)

        # Test with different learning rates for different components
        valid_config_dict["learning_rate"] = 5e-6
        valid_config_dict["vision_lr"] = 1e-7
        valid_config_dict["merger_lr"] = 1e-5
        valid_config_dict["llm_lr"] = 5e-6

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert config.learning_rate == 5e-6
            assert config.vision_lr == 1e-7
            assert config.merger_lr == 1e-5
            assert config.llm_lr == 5e-6
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_batch_size_and_accumulation(self, valid_config_dict):
        """Test batch size and gradient accumulation settings."""
        # Set up temporary paths
        temp_dir = self._setup_temp_paths(valid_config_dict)

        valid_config_dict["per_device_train_batch_size"] = 2
        valid_config_dict["gradient_accumulation_steps"] = 4

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(valid_config_dict, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert config.per_device_train_batch_size == 2
            assert config.gradient_accumulation_steps == 4
            # Effective batch size would be 2 * 4 = 8
        finally:
            Path(temp_file).unlink()
            import shutil

            shutil.rmtree(temp_dir)

    def test_config_attribute_access(self, temp_config_file):
        """Test that all configuration attributes are accessible."""
        config = load_config(temp_config_file)

        # Test that common attributes exist and are accessible
        required_attrs = [
            "model_path",
            "model_size",
            "coordinate_tokens_enabled",
            "max_coord_value",
            "teacher_ratio",
            "learning_rate",
            "output_dir",
            "train_data_path",
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing attribute: {attr}"
            # Should be able to access the attribute
            value = getattr(config, attr)
            assert value is not None or attr in [
                "val_data_path"
            ]  # Some attrs can be None

        # Clean up
        Path(temp_config_file).unlink()
