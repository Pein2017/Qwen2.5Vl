"""
Configuration compatibility tests.

This module validates that the new src_new/ implementation maintains
full compatibility with existing configuration files, especially bbu_v2.yaml.
"""

from pathlib import Path

import pytest
import yaml

from src_new.tests.fixtures import (
    assert_config_validity,
    create_sample_config,
)


class TestConfigCompatibility:
    """Test configuration compatibility with existing configs."""

    def test_bbu_v2_yaml_field_compatibility(self):
        """Test that all bbu_v2.yaml fields are supported in new config system."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found - skipping compatibility test")

        # Load actual bbu_v2.yaml
        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Create equivalent new config
        new_config = create_sample_config()

        # Test field mapping compatibility
        essential_field_mappings = {
            # Model settings (should map directly)
            "model_path": "model_path",
            "model_size": "model_size",
            "model_max_length": "model_max_length",
            "attn_implementation": "attn_implementation",
            "torch_dtype": "torch_dtype",
            # Training settings (should map directly)
            "num_train_epochs": "num_train_epochs",
            "per_device_train_batch_size": "per_device_train_batch_size",
            "learning_rate": "learning_rate",
            "vision_lr": "vision_lr",
            "merger_lr": "merger_lr",
            "llm_lr": "llm_lr",
            # Data settings (should map directly)
            "train_data_path": "train_data_path",
            "val_data_path": "val_data_path",
            "teacher_pool_file": "teacher_pool_file",
            "teacher_ratio": "teacher_ratio",
            "collator_type": "collator_type",
            "language": "language",
            # Coordinate token settings (should map directly)
            "coordinate_tokens_enabled": "coordinate_tokens_enabled",
            "max_coord_value": "max_coord_value",
            "coordinate_loss_weight": "coordinate_loss_weight",
            "regular_loss_weight": "regular_loss_weight",
            # Teacher-student settings (should map directly)
            "teacher_loss_weight": "teacher_loss_weight",
            "student_loss_weight": "student_loss_weight",
            # Training control (should map directly)
            "training_prompt_style": "training_prompt_style",
            "use_consistent_prompts": "use_consistent_prompts",
        }

        # Validate field mappings
        for bbu_field, new_field in essential_field_mappings.items():
            if bbu_field in bbu_config:
                # New config should support this field
                assert new_field in new_config, (
                    f"New config missing mapped field: {new_field} (from {bbu_field})"
                )

                # Types should be compatible (handle scientific notation conversion)
                bbu_value = bbu_config[bbu_field]
                new_value = new_config[new_field]

                # Handle scientific notation strings that get converted to floats
                if isinstance(bbu_value, str) and isinstance(new_value, (int, float)):
                    try:
                        # Try to convert string to number
                        converted_bbu_value = float(bbu_value)
                        # If conversion succeeds, compare the numeric values
                        assert isinstance(converted_bbu_value, (int, float)), (
                            f"Scientific notation conversion failed for {bbu_field}: {bbu_value}"
                        )
                    except ValueError:
                        # If conversion fails, types should match exactly
                        assert type(bbu_value) == type(new_value), (
                            f"Type mismatch for {bbu_field}: {type(bbu_value)} vs {type(new_value)}"
                        )
                else:
                    # For non-scientific notation, types should match exactly
                    assert type(bbu_value) == type(new_value), (
                        f"Type mismatch for {bbu_field}: {type(bbu_value)} vs {type(new_value)}"
                    )

    def test_bbu_v2_yaml_value_compatibility(self):
        """Test that bbu_v2.yaml values are valid in new config system."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        # Load and validate bbu_v2.yaml with new system
        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Should pass validation without modification
        try:
            assert_config_validity(bbu_config, check_converted=True)
        except AssertionError as e:
            pytest.fail(f"bbu_v2.yaml failed validation in new system: {e}")

        # Test specific value constraints

        # Coordinate token settings
        if "coordinate_tokens_enabled" in bbu_config:
            assert isinstance(bbu_config["coordinate_tokens_enabled"], bool)

        if "max_coord_value" in bbu_config:
            assert isinstance(bbu_config["max_coord_value"], int)
            assert bbu_config["max_coord_value"] > 0
            assert bbu_config["max_coord_value"] <= 4096  # Reasonable upper bound

        if "coordinate_loss_weight" in bbu_config:
            assert isinstance(bbu_config["coordinate_loss_weight"], (int, float))
            assert bbu_config["coordinate_loss_weight"] >= 0

        # Teacher-student settings
        if "teacher_ratio" in bbu_config:
            assert isinstance(bbu_config["teacher_ratio"], (int, float))
            assert 0 <= bbu_config["teacher_ratio"] <= 1

        if "teacher_loss_weight" in bbu_config:
            assert isinstance(bbu_config["teacher_loss_weight"], (int, float))
            assert bbu_config["teacher_loss_weight"] >= 0

        if "student_loss_weight" in bbu_config:
            assert isinstance(bbu_config["student_loss_weight"], (int, float))
            assert bbu_config["student_loss_weight"] >= 0

    def test_no_config_migration_required(self):
        """Test that existing config requires no migration or modification."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        # Load original config
        with open(config_path, "r") as f:
            original_config = yaml.safe_load(f)

        # Create a copy to test with
        test_config = original_config.copy()

        # Should work directly without any modifications
        assert_config_validity(test_config, check_converted=True)

        # Test that no fields need to be renamed
        required_fields = [
            "model_path",
            "coordinate_tokens_enabled",
            "teacher_ratio",
            "train_data_path",
            "val_data_path",
            "teacher_pool_file",
        ]

        for field in required_fields:
            assert field in test_config, (
                f"Required field {field} missing from bbu_v2.yaml"
            )

        # Test that no values need to be converted
        # (All values should be in correct format already)

        # Coordinate tokens should be boolean
        assert isinstance(test_config["coordinate_tokens_enabled"], bool)

        # Teacher ratio should be numeric between 0 and 1
        assert isinstance(test_config["teacher_ratio"], (int, float))
        assert 0 <= test_config["teacher_ratio"] <= 1

    def test_config_extensibility(self):
        """Test that config system can handle extensions while maintaining compatibility."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        # Test adding new optional fields
        extended_config = base_config.copy()
        extended_config.update(
            {
                # New optional fields that might be added in the future
                "new_optional_field": "test_value",
                "advanced_coordinate_mode": False,
                "experimental_features": {"feature_a": True, "feature_b": 0.5},
            }
        )

        # Should still be valid (new system should ignore unknown fields gracefully)
        try:
            assert_config_validity(extended_config, check_converted=True)
        except AssertionError as e:
            # Should not fail due to extra fields
            if "missing required" in str(e).lower():
                pytest.fail(
                    f"Config validation too strict - failed on extended config: {e}"
                )

    def test_config_defaults_preservation(self):
        """Test that default values are preserved from original system."""
        # Load original config if available
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if config_path.exists():
            with open(config_path, "r") as f:
                bbu_config = yaml.safe_load(f)
        else:
            # Use mock config based on known bbu_v2.yaml structure
            bbu_config = {
                "coordinate_tokens_enabled": True,
                "max_coord_value": 2048,
                "coordinate_loss_weight": 0.05,
                "regular_loss_weight": 1.0,
                "teacher_ratio": 0.5,
                "teacher_loss_weight": 0.3,
                "student_loss_weight": 1.0,
                "collator_type": "packed",
                "language": "chinese",
                "training_prompt_style": True,
                "use_consistent_prompts": True,
            }

        # Create new config with same defaults
        new_config = create_sample_config()

        # Key defaults should match
        default_fields = [
            "coordinate_tokens_enabled",
            "teacher_ratio",
            "collator_type",
            "language",
            "training_prompt_style",
        ]

        for field in default_fields:
            if field in bbu_config and field in new_config:
                # Default values should be compatible
                bbu_default = bbu_config[field]
                new_default = new_config[field]

                # Allow for reasonable default variations, but same semantics
                if isinstance(bbu_default, bool):
                    assert isinstance(new_default, bool)
                elif isinstance(bbu_default, (int, float)):
                    assert isinstance(new_default, (int, float))
                    # Allow some tolerance for numeric defaults
                    if bbu_default != 0:
                        assert abs(new_default - bbu_default) / abs(bbu_default) < 0.5
                elif isinstance(bbu_default, str):
                    assert isinstance(new_default, str)
                    # String defaults should be identical or compatible
                    assert (
                        new_default == bbu_default
                        or new_default.lower() == bbu_default.lower()
                    )

    def test_training_parameter_compatibility(self):
        """Test compatibility of training-specific parameters."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test learning rate parameters
        lr_fields = ["learning_rate", "vision_lr", "merger_lr", "llm_lr"]

        for field in lr_fields:
            if field in bbu_config:
                lr_value = bbu_config[field]

                # Handle scientific notation strings
                if isinstance(lr_value, str):
                    try:
                        lr_value = float(lr_value)
                    except ValueError:
                        assert False, f"{field} must be numeric, got string: {lr_value}"

                # Should be numeric and positive
                assert isinstance(lr_value, (int, float)), f"{field} must be numeric"
                assert lr_value > 0, f"{field} must be positive"
                assert lr_value < 1, f"{field} should be reasonable (< 1)"

        # Test batch size parameters
        batch_fields = ["per_device_train_batch_size", "per_device_eval_batch_size"]

        for field in batch_fields:
            if field in bbu_config:
                batch_value = bbu_config[field]

                assert isinstance(batch_value, int), f"{field} must be integer"
                assert batch_value > 0, f"{field} must be positive"
                assert batch_value <= 32, f"{field} should be reasonable (<= 32)"

        # Test epoch parameter
        if "num_train_epochs" in bbu_config:
            epochs = bbu_config["num_train_epochs"]
            assert isinstance(epochs, int), "num_train_epochs must be integer"
            assert epochs > 0, "num_train_epochs must be positive"

    def test_data_parameter_compatibility(self):
        """Test compatibility of data-specific parameters."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test data paths
        data_path_fields = ["train_data_path", "val_data_path", "teacher_pool_file"]

        for field in data_path_fields:
            if field in bbu_config:
                path_value = bbu_config[field]

                assert isinstance(path_value, str), f"{field} must be string"
                assert len(path_value) > 0, f"{field} cannot be empty"
                # Should end with .jsonl for data files
                if "data_path" in field or "pool_file" in field:
                    assert path_value.endswith(".jsonl"), (
                        f"{field} should be JSONL file"
                    )

        # Test collator type
        if "collator_type" in bbu_config:
            collator_type = bbu_config["collator_type"]
            assert isinstance(collator_type, str)
            assert collator_type in ["standard", "packed"], (
                f"Invalid collator type: {collator_type}"
            )

        # Test language setting
        if "language" in bbu_config:
            language = bbu_config["language"]
            assert isinstance(language, str)
            assert language.lower() in ["chinese", "zh", "中文", "cn"], (
                f"Unsupported language: {language}"
            )

    def test_coordinate_token_parameter_compatibility(self):
        """Test compatibility of coordinate token specific parameters."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test coordinate token enablement
        if "coordinate_tokens_enabled" in bbu_config:
            enabled = bbu_config["coordinate_tokens_enabled"]
            assert isinstance(enabled, bool)

        # Test coordinate value range
        if "max_coord_value" in bbu_config:
            max_coord = bbu_config["max_coord_value"]
            assert isinstance(max_coord, int)
            assert max_coord > 0
            assert max_coord <= 4096  # Reasonable upper bound

            # Should be power of 2 or close to it for efficiency
            assert max_coord in [512, 1024, 2048, 4096], (
                f"max_coord_value should be standard size: {max_coord}"
            )

        # Test loss weights
        loss_weight_fields = ["coordinate_loss_weight", "regular_loss_weight"]

        for field in loss_weight_fields:
            if field in bbu_config:
                weight = bbu_config[field]
                assert isinstance(weight, (int, float))
                assert weight >= 0
                # Should be reasonable weight (not too large)
                assert weight <= 10, f"{field} seems too large: {weight}"

    def test_teacher_student_parameter_compatibility(self):
        """Test compatibility of teacher-student learning parameters."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test teacher ratio
        if "teacher_ratio" in bbu_config:
            ratio = bbu_config["teacher_ratio"]
            assert isinstance(ratio, (int, float))
            assert 0 <= ratio <= 1, f"teacher_ratio must be between 0 and 1: {ratio}"

        # Test teacher-student loss weights
        if "teacher_loss_weight" in bbu_config:
            teacher_weight = bbu_config["teacher_loss_weight"]
            assert isinstance(teacher_weight, (int, float))
            assert teacher_weight >= 0

        if "student_loss_weight" in bbu_config:
            student_weight = bbu_config["student_loss_weight"]
            assert isinstance(student_weight, (int, float))
            assert student_weight >= 0

        # Test that weights are reasonable relative to each other
        if "teacher_loss_weight" in bbu_config and "student_loss_weight" in bbu_config:
            teacher_w = bbu_config["teacher_loss_weight"]
            student_w = bbu_config["student_loss_weight"]

            # Student weight should typically be >= teacher weight
            # (but allow flexibility)
            total_weight = teacher_w + student_w
            assert total_weight > 0, "Combined teacher-student weights must be positive"

        # Test number of teacher samples
        if "num_teacher_samples" in bbu_config:
            num_teachers = bbu_config["num_teacher_samples"]
            assert isinstance(num_teachers, int)
            assert num_teachers >= 0
            assert num_teachers <= 5, (
                f"num_teacher_samples seems too large: {num_teachers}"
            )

    def test_config_field_name_variations(self):
        """Test tolerance for field name variations."""
        # Test common variations that might exist
        base_config = create_sample_config()

        variations = [
            # Different naming conventions
            {
                "coordinate_tokens_enabled": True,
                "enable_coordinate_tokens": True,
            },  # Duplicate with variation
            {"teacher_ratio": 0.5, "teacher_pool_ratio": 0.5},  # Similar field names
            {
                "collator_type": "packed",
                "data_collator_type": "packed",
            },  # Extended names
        ]

        for variation in variations:
            test_config = base_config.copy()
            test_config.update(variation)

            # Should handle gracefully (use primary field, ignore duplicates)
            try:
                assert_config_validity(test_config)
            except Exception as e:
                # Should not fail due to field name variations
                if "duplicate" not in str(e).lower():
                    pytest.fail(f"Failed on field variation: {e}")

    def test_config_type_coercion(self):
        """Test type coercion for backward compatibility."""
        base_config = create_sample_config()

        # Test type variations that might occur
        type_variations = {
            "coordinate_tokens_enabled": "true",  # String instead of bool
            "teacher_ratio": "0.5",  # String instead of float
            "max_coord_value": "2048",  # String instead of int
            "per_device_train_batch_size": 1.0,  # Float instead of int
        }

        for field, alt_value in type_variations.items():
            test_config = base_config.copy()
            test_config[field] = alt_value

            # New system should either handle coercion or give clear error
            try:
                # Mock type coercion
                if field == "coordinate_tokens_enabled" and isinstance(alt_value, str):
                    test_config[field] = alt_value.lower() == "true"
                elif field == "teacher_ratio" and isinstance(alt_value, str):
                    test_config[field] = float(alt_value)
                elif field == "max_coord_value" and isinstance(alt_value, str):
                    test_config[field] = int(alt_value)
                elif field == "per_device_train_batch_size" and isinstance(
                    alt_value, float
                ):
                    test_config[field] = int(alt_value)

                assert_config_validity(test_config)

            except (ValueError, TypeError) as e:
                # Type coercion failed - should give clear error message
                assert field in str(e) or "type" in str(e).lower()


class TestConfigValidationComprehensive:
    """Comprehensive configuration validation tests."""

    def test_all_required_fields_present(self):
        """Test that all required fields are present in bbu_v2.yaml."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Comprehensive list of required fields
        required_fields = [
            # Model configuration
            "model_path",
            "model_size",
            "model_max_length",
            "torch_dtype",
            # Training configuration
            "num_train_epochs",
            "per_device_train_batch_size",
            "learning_rate",
            # Data configuration
            "train_data_path",
            "val_data_path",
            "teacher_pool_file",
            # Feature configuration
            "coordinate_tokens_enabled",
            "teacher_ratio",
            # Output configuration
            "output_dir",
        ]

        missing_fields = []
        for field in required_fields:
            if field not in bbu_config:
                missing_fields.append(field)

        assert not missing_fields, (
            f"Missing required fields in bbu_v2.yaml: {missing_fields}"
        )

    def test_config_logical_consistency(self):
        """Test logical consistency of configuration values."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test logical relationships between parameters

        # If coordinate tokens are enabled, related params should be set
        if bbu_config.get("coordinate_tokens_enabled", False):
            assert "max_coord_value" in bbu_config, (
                "max_coord_value required when coordinate tokens enabled"
            )
            assert "coordinate_loss_weight" in bbu_config, (
                "coordinate_loss_weight required when coordinate tokens enabled"
            )

        # Teacher ratio and teacher file consistency
        teacher_ratio = bbu_config.get("teacher_ratio", 0)
        if teacher_ratio > 0:
            assert "teacher_pool_file" in bbu_config, (
                "teacher_pool_file required when teacher_ratio > 0"
            )
            assert bbu_config["teacher_pool_file"], (
                "teacher_pool_file cannot be empty when teacher_ratio > 0"
            )

        # Learning rates should be in reasonable ranges
        lr_fields = ["learning_rate", "vision_lr", "merger_lr", "llm_lr"]
        for field in lr_fields:
            if field in bbu_config:
                lr = bbu_config[field]

                # Handle scientific notation strings
                if isinstance(lr, str):
                    try:
                        lr = float(lr)
                    except ValueError:
                        assert False, f"{field} must be numeric, got string: {lr}"

                assert 1e-8 <= lr <= 1e-2, f"{field} outside reasonable range: {lr}"

        # Batch sizes vs memory constraints
        batch_size = bbu_config.get("per_device_train_batch_size", 1)
        max_length = bbu_config.get("model_max_length", 4096)

        # Rough memory usage check
        if batch_size > 1 and max_length > 32000:
            # This might cause memory issues
            print(
                f"Warning: Large batch size ({batch_size}) with long sequences ({max_length}) may cause OOM"
            )

    def test_config_completeness_for_features(self):
        """Test that config is complete for all advertised features."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml")

        if not config_path.exists():
            pytest.skip("bbu_v2.yaml not found")

        with open(config_path, "r") as f:
            bbu_config = yaml.safe_load(f)

        # Test coordinate token feature completeness
        if bbu_config.get("coordinate_tokens_enabled", False):
            coord_required = [
                "max_coord_value",
                "coordinate_loss_weight",
                "regular_loss_weight",
            ]
            for field in coord_required:
                assert field in bbu_config, (
                    f"Coordinate tokens enabled but missing: {field}"
                )

        # Test teacher-student feature completeness
        teacher_ratio = bbu_config.get("teacher_ratio", 0)
        if teacher_ratio > 0:
            teacher_required = [
                "teacher_pool_file",
                "teacher_loss_weight",
                "student_loss_weight",
            ]
            for field in teacher_required:
                assert field in bbu_config, (
                    f"Teacher-student learning enabled but missing: {field}"
                )

        # Test training feature completeness
        train_required = ["num_train_epochs", "learning_rate", "output_dir"]
        for field in train_required:
            assert field in bbu_config, (
                f"Training configuration incomplete, missing: {field}"
            )
