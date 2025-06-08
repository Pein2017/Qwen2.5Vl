"""
Simplified Configuration Manager
Uses Pydantic schema for validation and type safety.
Clean separation of concerns with minimal complexity.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from .schema import TrainingConfiguration


class ConfigurationManager:
    """
    Simplified configuration manager using Pydantic schema.
    Handles hierarchical composition and validation.
    """

    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            profiles_dir: Directory containing profile YAML files
        """
        if profiles_dir is None:
            # Default to configs directory
            project_root = Path(__file__).parent.parent.parent
            profiles_dir = project_root / "configs" / "profiles"

        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def load_configuration(
        self, profile: str, overrides: Optional[List[str]] = None
    ) -> TrainingConfiguration:
        """
        Load and validate configuration from profile.

        Args:
            profile: Profile name (e.g., 'base', 'debug', 'qwen7b')
            overrides: List of override strings (e.g., ['training.num_train_epochs=5'])

        Returns:
            TrainingConfiguration: Validated configuration

        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValidationError: If configuration is invalid
        """
        # Load profile configuration
        config_dict = self._load_profile(profile)

        # Apply overrides
        if overrides:
            config_dict = self._apply_overrides(config_dict, overrides)

        # Auto-generate dynamic values
        config_dict = self._resolve_dynamic_values(config_dict)

        # Validate and create configuration
        try:
            config = TrainingConfiguration(**config_dict)
        except ValidationError as e:
            raise ValidationError(
                f"Configuration validation failed for profile '{profile}': {e}"
            )

        return config

    def _load_profile(self, profile: str) -> Dict[str, Any]:
        """Load profile configuration with inheritance support."""
        profile_path = self.profiles_dir / f"{profile}.yaml"

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle inheritance
        if "inherits" in config_dict:
            base_profile = config_dict.pop("inherits")
            base_config = self._load_profile(base_profile)
            config_dict = self._merge_configs(base_config, config_dict)

        return config_dict

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_overrides(
        self, config_dict: Dict[str, Any], overrides: List[str]
    ) -> Dict[str, Any]:
        """Apply command-line overrides to configuration."""
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid override format: {override}. Expected 'key=value'"
                )

            key, value = override.split("=", 1)

            # Parse value
            parsed_value = self._parse_override_value(value)

            # Apply nested key
            self._set_nested_key(config_dict, key, parsed_value)

        return config_dict

    def _parse_override_value(self, value: str) -> Any:
        """Parse override value to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Numeric values
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String values
        return value

    def _set_nested_key(self, config_dict: Dict[str, Any], key: str, value: Any):
        """Set nested key in configuration dictionary."""
        keys = key.split(".")
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _resolve_dynamic_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dynamic values in configuration."""
        # Auto-generate run_name if not provided
        if "output" in config_dict and config_dict["output"].get("run_name") is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = config_dict.get("model", {}).get("model_size", "unknown")
            num_epochs = config_dict.get("training", {}).get("num_train_epochs", 0)
            run_name = f"qwen_{model_size}_{num_epochs}ep_{timestamp}"
            config_dict["output"]["run_name"] = run_name

        # Auto-generate logging_dir if not provided
        if (
            "logging" in config_dict
            and config_dict["logging"].get("logging_dir") is None
        ):
            tb_dir = config_dict.get("output", {}).get("tb_dir", "tb_logs")
            run_name = config_dict.get("output", {}).get("run_name", "default")
            config_dict["logging"]["logging_dir"] = f"{tb_dir}/{run_name}"

        return config_dict

    def save_configuration(self, config: TrainingConfiguration, output_path: str):
        """Save configuration to YAML file."""
        config_dict = config.dict()

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def list_profiles(self) -> List[str]:
        """List available configuration profiles."""
        profiles = []
        for yaml_file in self.profiles_dir.glob("*.yaml"):
            profiles.append(yaml_file.stem)
        return sorted(profiles)

    def print_configuration(self, config: TrainingConfiguration):
        """Print configuration in a readable format."""
        print("=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)

        sections = [
            ("MODEL", config.model),
            ("DATA", config.data),
            ("LEARNING RATES", config.learning_rates),
            ("TRAINING", config.training),
            ("EVALUATION", config.evaluation),
            ("LOGGING", config.logging),
            ("MONITORING", config.monitoring),
            ("LOSS", config.loss),
            ("PERFORMANCE", config.performance),
            ("OUTPUT", config.output),
            ("STABILITY", config.stability),
            ("DEBUG", config.debug),
        ]

        for section_name, section_config in sections:
            print(f"\n{section_name}:")
            print("-" * len(section_name))
            for field_name, field_value in section_config.dict().items():
                print(f"  {field_name}: {field_value}")

        print("=" * 80)


# Convenience functions
def load_config(
    profile: str, overrides: Optional[List[str]] = None
) -> TrainingConfiguration:
    """Load configuration using default manager."""
    manager = ConfigurationManager()
    return manager.load_configuration(profile, overrides)


def list_available_profiles() -> List[str]:
    """List available configuration profiles."""
    manager = ConfigurationManager()
    return manager.list_profiles()
