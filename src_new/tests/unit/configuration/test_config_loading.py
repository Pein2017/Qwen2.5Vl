"""
Tests for configuration loading and validation.

This module tests the unified configuration system's ability to:
- Load configuration from YAML files
- Validate required fields and types
- Handle default values appropriately
- Maintain backward compatibility with existing configs
"""

from pathlib import Path

import pytest
import yaml

from src_new.tests.fixtures import (
    assert_config_validity,
    create_sample_config,
    create_temp_files,
)


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_valid_config(self, temp_dir):
        """Test loading a valid configuration file."""
        # Create test config
        config_data = create_sample_config()
        config_files = {"test_config.yaml": config_data}
        test_dir = create_temp_files(config_files, temp_dir)
        config_path = test_dir / "test_config.yaml"

        # Mock the config loading (will implement when src_new/config exists)
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        # Validate loaded config
        assert_config_validity(loaded_config)
        assert loaded_config["model_path"] == config_data["model_path"]
        assert loaded_config["learning_rate"] == config_data["learning_rate"]
        assert (
            loaded_config["coordinate_tokens_enabled"]
            == config_data["coordinate_tokens_enabled"]
        )

    def test_bbu_v2_yaml_compatibility(self):
        """Test compatibility with existing bbu_v2.yaml config."""
        # Load the standard use_coord config
        bbu_config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")

        if not bbu_config_path.exists():
            pytest.skip("bbu_v2.yaml not found - skipping compatibility test")

        with open(bbu_config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Validate that all required fields are present
        required_fields = [
            "model_path",
            "model_size",
            "model_max_length",
            "attn_implementation",
            "torch_dtype",
            "num_train_epochs",
            "per_device_train_batch_size",
            "learning_rate",
            # Accept either explicit paths or derived via data_root; the use_coord config uses data_root
            "data_root",
            "teacher_ratio",
            "coordinate_tokens_enabled",
            "max_coord_value",
            "coordinate_loss_weight",
            "output_dir",
        ]

        for field in required_fields:
            assert field in bbu_config, (
                f"Missing required field in bbu_v2_use_coord.yaml: {field}"
            )

        # Validate field types and values
        assert isinstance(bbu_config["coordinate_tokens_enabled"], bool)
        assert isinstance(bbu_config["max_coord_value"], int)
        assert bbu_config["max_coord_value"] > 0
        assert isinstance(bbu_config["coordinate_loss_weight"], (int, float))
        assert bbu_config["coordinate_loss_weight"] >= 0

    def test_config_field_validation(self, temp_dir):
        """Test validation of configuration fields."""
        base_config = create_sample_config()

        # Test missing required field
        invalid_config = base_config.copy()
        del invalid_config["model_path"]

        config_files = {"invalid_config.yaml": invalid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "invalid_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # This should raise an assertion error due to missing model_path
        with pytest.raises(
            AssertionError, match="Missing required config field: model_path"
        ):
            assert_config_validity(loaded_config)

    def test_config_type_validation(self, temp_dir):
        """Test validation of configuration field types."""
        # Test invalid learning rate type
        invalid_config = create_sample_config(learning_rate="invalid")

        config_files = {"invalid_type_config.yaml": invalid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "invalid_type_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # This should raise an assertion error due to invalid type
        with pytest.raises(AssertionError, match="learning_rate must be numeric"):
            assert_config_validity(loaded_config)

    def test_config_value_validation(self, temp_dir):
        """Test validation of configuration field values."""
        # Test negative learning rate
        invalid_config = create_sample_config(learning_rate=-1e-5)

        config_files = {"invalid_value_config.yaml": invalid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "invalid_value_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # This should raise an assertion error due to negative value
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            assert_config_validity(loaded_config)

    def test_coordinate_token_config_validation(self, temp_dir):
        """Test validation of coordinate token specific configuration."""
        # Test valid coordinate token config
        valid_config = create_sample_config(
            coordinate_tokens_enabled=True,
            max_coord_value=1024,
            coordinate_loss_weight=0.05,
        )

        config_files = {"coord_config.yaml": valid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "coord_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # Should not raise any errors
        assert_config_validity(loaded_config)

        # Validate specific coordinate token fields
        assert loaded_config["coordinate_tokens_enabled"] is True
        assert loaded_config["max_coord_value"] == 1024
        assert loaded_config["coordinate_loss_weight"] == 0.05

    def test_teacher_student_config_validation(self, temp_dir):
        """Test validation of teacher-student learning configuration."""
        valid_config = create_sample_config(
            teacher_ratio=0.7,
            teacher_loss_weight=0.3,
            student_loss_weight=1.0,
            num_teacher_samples=1,
        )

        config_files = {"teacher_config.yaml": valid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "teacher_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # Should not raise any errors
        assert_config_validity(loaded_config)

        # Validate teacher-student specific fields
        assert 0 <= loaded_config["teacher_ratio"] <= 1
        assert loaded_config["teacher_loss_weight"] >= 0
        assert loaded_config["student_loss_weight"] >= 0

    def test_training_config_validation(self, temp_dir):
        """Test validation of training-specific configuration."""
        valid_config = create_sample_config(
            num_train_epochs=10,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
        )

        config_files = {"training_config.yaml": valid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "training_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # Should not raise any errors
        assert_config_validity(loaded_config)

        # Validate training specific fields
        assert isinstance(loaded_config["num_train_epochs"], int)
        assert loaded_config["num_train_epochs"] > 0
        assert isinstance(loaded_config["per_device_train_batch_size"], int)
        assert loaded_config["per_device_train_batch_size"] > 0

    def test_data_config_validation(self, temp_dir):
        """Test validation of data-specific configuration."""
        valid_config = create_sample_config(
            train_data_path="data/train.jsonl",
            val_data_path="data/val.jsonl",
            teacher_pool_file="data/teacher_pool.jsonl",
            collator_type="packed",
            language="chinese",
            max_total_length=12000,
        )

        config_files = {"data_config.yaml": valid_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "data_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # Should not raise any errors
        assert_config_validity(loaded_config)

        # Validate data specific fields
        assert loaded_config["collator_type"] in ["standard", "packed"]
        assert loaded_config["language"] == "chinese"
        assert isinstance(loaded_config["max_total_length"], int)
        assert loaded_config["max_total_length"] > 0


class TestConfigDefaults:
    """Test configuration default value handling."""

    def test_default_values(self, temp_dir):
        """Test that default values are properly applied."""
        # Create minimal config with only required fields
        minimal_config = {
            "model_path": "/test/model/path",
            "model_size": "3B",
            "learning_rate": 5e-6,
            "train_data_path": "data/train.jsonl",
            "val_data_path": "data/val.jsonl",
        }

        config_files = {"minimal_config.yaml": minimal_config}
        test_dir = create_temp_files(config_files, temp_dir)

        with open(test_dir / "minimal_config.yaml", "r") as f:
            loaded_config = yaml.safe_load(f)

        # Test that required fields are present
        assert "model_path" in loaded_config
        assert "learning_rate" in loaded_config

        # Mock applying defaults (will implement when config module exists)
        default_config = create_sample_config()
        for key, value in default_config.items():
            if key not in loaded_config:
                loaded_config[key] = value

        # Validate that defaults were applied correctly
        assert loaded_config["coordinate_tokens_enabled"] is True  # Default value
        assert loaded_config["teacher_ratio"] == 0.5  # Default value
        assert loaded_config["collator_type"] == "packed"  # Default value


class TestConfigCompatibility:
    """Test configuration backward compatibility."""

    def test_field_name_compatibility(self):
        """Test that all field names match existing config structure."""
        expected_fields = {
            # Model settings
            "model_path",
            "model_size",
            "model_max_length",
            "attn_implementation",
            "torch_dtype",
            "use_cache",
            "model_hidden_size",
            "model_num_layers",
            "model_vocab_size",
            # Training settings
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "vision_lr",
            "merger_lr",
            "llm_lr",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "lr_scheduler_type",
            "gradient_checkpointing",
            "bf16",
            "fp16",
            # Data settings - only data_root needed after migration
            "data_root",
            "teacher_ratio",
            "collator_type",
            "language",
            "max_total_length",
            "num_teacher_samples",
            # Coordinate token settings
            "coordinate_tokens_enabled",
            "max_coord_value",
            "coordinate_loss_weight",
            "regular_loss_weight",
            # Teacher-student settings
            "teacher_loss_weight",
            "student_loss_weight",
            # Output settings
            "output_dir",
            "logging_steps",
            "logging_dir",
            # Training control
            "training_prompt_style",
            "use_consistent_prompts",
        }

        sample_config = create_sample_config()
        config_fields = set(sample_config.keys())

        # Check that all expected fields are present
        missing_fields = expected_fields - config_fields
        assert not missing_fields, f"Missing expected fields: {missing_fields}"

        # Check for unexpected fields (informational)
        extra_fields = config_fields - expected_fields
        if extra_fields:
            print(f"Extra fields in config (may be intentional): {extra_fields}")

    def test_config_value_formats(self, temp_dir):
        """Test that configuration values maintain expected formats."""
        config = create_sample_config()

        # Test specific format requirements
        assert isinstance(config["coordinate_tokens_enabled"], bool)
        assert isinstance(config["max_coord_value"], int)
        assert isinstance(config["teacher_ratio"], (int, float))
        assert 0 <= config["teacher_ratio"] <= 1

        # Test data root format (should be string) - paths are auto-resolved
        assert isinstance(config["data_root"], str)

        # Test learning rate formats
        assert isinstance(config["learning_rate"], (int, float))
        assert isinstance(config["vision_lr"], (int, float))
        assert isinstance(config["merger_lr"], (int, float))
        assert isinstance(config["llm_lr"], (int, float))


@pytest.mark.compatibility
class TestExistingConfigCompatibility:
    """Test compatibility with existing configuration files."""

    def test_load_existing_bbu_v2_config(self):
        """Test loading the actual bbu_v2.yaml configuration file."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2_use_coord.yaml not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # This should work without any modifications to the existing file
        assert_config_validity(config, check_converted=True)

        # Verify key compatibility fields
        assert "coordinate_tokens_enabled" in config
        assert "teacher_ratio" in config
        assert "collator_type" in config

        # Verify the config contains all necessary components for our system
        essential_components = [
            "model_path",
            # Paths may be derived via data_root; use that as the required key
            "data_root",
            "coordinate_tokens_enabled",
            # teacher_pool_file may also be derived via data_root in runtime; keep it optional here
        ]

        for component in essential_components:
            assert component in config, f"Missing essential component: {component}"

    def test_no_config_changes_required(self):
        """Verify that existing config requires no changes."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2_use_coord.yaml not found")

        # Load original config
        with open(config_path, "r") as f:
            original_config = yaml.safe_load(f)

        # Create equivalent new config
        new_config = create_sample_config(
            model_path=original_config.get("model_path"),
            train_data_path=original_config.get("train_data_path"),
            val_data_path=original_config.get("val_data_path"),
            coordinate_tokens_enabled=original_config.get("coordinate_tokens_enabled"),
            teacher_ratio=original_config.get("teacher_ratio"),
        )

        # Key fields should match exactly
        key_fields = [
            "coordinate_tokens_enabled",
            "teacher_ratio",
            "max_coord_value",
            "coordinate_loss_weight",
            "teacher_loss_weight",
            "student_loss_weight",
        ]

        for field in key_fields:
            if field in original_config:
                assert field in new_config
                # Values should be compatible (allowing for type flexibility)
                assert type(original_config[field]) == type(new_config[field])
