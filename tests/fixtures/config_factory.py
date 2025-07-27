"""
Test Configuration Factory

Professional configuration management for BBU training pipeline tests.
Creates different configuration variants for comprehensive testing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config import BBUConfig, load_config
from src.logger_utils import get_logger


logger = get_logger("test_config")


class ConfigFactory:
    """Factory for creating test configurations with different settings."""

    BASE_CONFIG_PATH = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"

    # Test-specific overrides for fast execution with memory efficiency
    TEST_OVERRIDES = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,  # Keep at 2 for collator testing
        "per_device_eval_batch_size": 2,  # Keep at 2 for collator testing
        "max_total_length": 1024,  # Reduced from 2048 to save memory
        "save_strategy": "no",
        "dataloader_num_workers": 0,
        "logging_steps": 1,
        "eval_steps": 1,
        "eval_strategy": "steps",
        "disable_tqdm": True,
    }

    def __init__(self, test_configs_dir: Optional[str] = None):
        """
        Initialize test configuration factory.

        Args:
            test_configs_dir: Directory to store test configuration files
        """
        if test_configs_dir is None:
            test_configs_dir = "/data3/Qwen2.5-VL-main/tests/configs"

        self.configs_dir = Path(test_configs_dir)
        self.configs_dir.mkdir(exist_ok=True)

        logger.info(f"📄 TestConfigFactory initialized: {self.configs_dir}")

    def create_coordinate_enabled_config(
        self, data_root: str, collator_type: str = "standard"
    ) -> str:
        """
        Create configuration with coordinate tokens enabled.

        Args:
            data_root: Path to test data directory
            collator_type: Type of data collator ("standard" or "packed")

        Returns:
            Path to created configuration file
        """
        config_data = self._load_base_config()

        # Coordinate token specific overrides
        coordinate_overrides = {
            "coordinate_tokens_enabled": True,
            "max_coord_value": 2048,
            "coordinate_loss_weight": 0.05,
            "regular_loss_weight": 1.0,
            "coordinate_lr": 5e-6,  # Enable coordinate learning rate
        }

        # Apply all overrides
        config_data.update(self.TEST_OVERRIDES)
        config_data.update(coordinate_overrides)
        config_data.update(
            {
                "data_root": data_root,
                "train_data_path": f"{data_root}/train.jsonl",
                "val_data_path": f"{data_root}/val.jsonl",
                "teacher_pool_file": f"{data_root}/teacher.jsonl",
                "collator_type": collator_type,
                "output_dir": f"{data_root}/output_coordinate_{collator_type}",
                "run_name": f"test_coordinate_{collator_type}",
            }
        )

        config_path = self.configs_dir / f"coordinate_enabled_{collator_type}.yaml"
        self._save_config(config_data, config_path)

        logger.info(f"✅ Created coordinate enabled config: {config_path}")
        return str(config_path)

    def create_coordinate_disabled_config(
        self, data_root: str, collator_type: str = "standard"
    ) -> str:
        """
        Create configuration with coordinate tokens disabled.

        Args:
            data_root: Path to test data directory
            collator_type: Type of data collator ("standard" or "packed")

        Returns:
            Path to created configuration file
        """
        config_data = self._load_base_config()

        # Coordinate disabled specific overrides
        coordinate_overrides = {
            "coordinate_tokens_enabled": False,
            "max_coord_value": 2048,  # Standard value as per bbu_v2.yaml reference
            "coordinate_loss_weight": 0.0,
            "regular_loss_weight": 1.0,
            "coordinate_lr": 0.0,  # Disable coordinate learning rate
        }

        # Apply all overrides
        config_data.update(self.TEST_OVERRIDES)
        config_data.update(coordinate_overrides)
        config_data.update(
            {
                "data_root": data_root,
                "train_data_path": f"{data_root}/train.jsonl",
                "val_data_path": f"{data_root}/val.jsonl",
                "teacher_pool_file": f"{data_root}/teacher.jsonl",
                "collator_type": collator_type,
                "output_dir": f"{data_root}/output_standard_{collator_type}",
                "run_name": f"test_standard_{collator_type}",
            }
        )

        config_path = self.configs_dir / f"coordinate_disabled_{collator_type}.yaml"
        self._save_config(config_data, config_path)

        logger.info(f"✅ Created coordinate disabled config: {config_path}")
        return str(config_path)

    def create_packed_collator_config(self, data_root: str) -> str:
        """
        Create configuration specifically for testing packed collator.

        Args:
            data_root: Path to test data directory

        Returns:
            Path to created configuration file
        """
        return self.create_coordinate_enabled_config(data_root, "packed")

    def create_standard_collator_config(self, data_root: str) -> str:
        """
        Create configuration specifically for testing standard collator.

        Args:
            data_root: Path to test data directory

        Returns:
            Path to created configuration file
        """
        return self.create_coordinate_enabled_config(data_root, "standard")

    def create_minimal_config(
        self,
        data_root: str,
        coordinate_enabled: bool = True,
        collator_type: str = "standard",
    ) -> str:
        """
        Create minimal configuration for fast testing.

        Args:
            data_root: Path to test data directory
            coordinate_enabled: Whether to enable coordinate tokens
            collator_type: Type of data collator

        Returns:
            Path to created configuration file
        """
        config_data = self._load_base_config()

        # Minimal overrides for fastest testing with memory efficiency
        minimal_overrides = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,  # Keep at 2 for collator testing
            "per_device_eval_batch_size": 2,  # Keep at 2 for collator testing
            "max_total_length": 512,  # Reduced for minimal tests
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "eval_steps": 1,
            "eval_strategy": "steps",
            "save_strategy": "no",
            "dataloader_num_workers": 0,
            "disable_tqdm": True,
            "coordinate_tokens_enabled": coordinate_enabled,
            "max_coord_value": 2048,  # Always 2048 as per bbu_v2.yaml reference
            "coordinate_loss_weight": 0.05 if coordinate_enabled else 0.0,
            "coordinate_lr": 5e-6 if coordinate_enabled else 0.0,
            "remove_unused_columns": False,  # Critical for coordinate mode compatibility
        }

        config_data.update(minimal_overrides)
        config_data.update(
            {
                "data_root": data_root,
                "train_data_path": f"{data_root}/train.jsonl",
                "val_data_path": f"{data_root}/val.jsonl",
                "teacher_pool_file": f"{data_root}/teacher.jsonl",
                "collator_type": collator_type,
                "output_dir": f"{data_root}/output_minimal",
                "run_name": "test_minimal",
            }
        )

        suffix = "coord" if coordinate_enabled else "standard"
        config_path = self.configs_dir / f"minimal_{suffix}_{collator_type}.yaml"
        self._save_config(config_data, config_path)

        logger.info(f"✅ Created minimal config: {config_path}")
        return str(config_path)

    def load_test_config(self, config_path: str) -> BBUConfig:
        """
        Load test configuration and return BBUConfig instance.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded and validated BBUConfig instance
        """
        return load_config(config_path)

    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration as a dictionary."""
        with open(self.BASE_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _save_config(self, config_data: Dict[str, Any], config_path: Path) -> None:
        """Save configuration data to YAML file."""

        # Convert Path objects to strings to avoid YAML serialization issues
        def convert_paths_to_strings(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            else:
                return obj

        clean_config_data = convert_paths_to_strings(config_data)

        with open(config_path, "w") as f:
            yaml.dump(clean_config_data, f, default_flow_style=False, sort_keys=False)

    def cleanup_test_configs(self) -> None:
        """Remove all generated test configuration files."""

        if self.configs_dir.exists():
            for config_file in self.configs_dir.glob("*.yaml"):
                config_file.unlink()
            logger.info(f"🧹 Cleaned up test configs: {self.configs_dir}")

    def get_available_configs(self) -> List[str]:
        """Get list of available test configuration files."""
        return [str(p) for p in self.configs_dir.glob("*.yaml")]
