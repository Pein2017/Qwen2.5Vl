"""
Configuration management for Qwen2.5-VL BBU training using OmegaConf.
Simplified for flattened YAML structure with explicit values only.
Environment variables are handled by the launcher script.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """
    Simplified configuration manager for BBU training using OmegaConf.

    Handles flattened YAML configuration loading and validation.
    No hierarchical composition or defaults - all values must be explicit.
    Environment variables are handled by the launcher script.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to configs directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)
        self.resolvers_registered = False

        # Register custom resolvers
        self._register_resolvers()

    def _register_resolvers(self):
        """Register custom OmegaConf resolvers for dynamic values."""
        if self.resolvers_registered:
            return

        # Current timestamp resolver
        try:
            OmegaConf.register_resolver(
                "now",
                lambda format_str="%Y%m%d_%H%M%S": datetime.now().strftime(format_str),
            )
        except AssertionError:
            # Resolver already registered, skip
            pass

        # Random port resolver for distributed training
        try:
            OmegaConf.register_resolver(
                "random_port", lambda start=20001, end=29999: random.randint(start, end)
            )
        except AssertionError:
            # Resolver already registered, skip
            pass

        self.resolvers_registered = True

    def load_config(
        self, config_name: str, overrides: Optional[list] = None
    ) -> DictConfig:
        """
        Load configuration from YAML file with optional overrides.

        Args:
            config_name: Name of the configuration (e.g., "debug", "qwen25vl_3b")
            overrides: List of override strings (e.g., ["num_train_epochs=5"])

        Returns:
            DictConfig: Loaded and resolved configuration
        """
        # Construct config path
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config_path = self.config_dir / config_name

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(config_path, "r") as f:
            cfg = OmegaConf.load(f)

        # Handle defaults (hierarchical composition)
        if "defaults" in cfg:
            cfg = self._compose_config(cfg)

        # Apply overrides
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)

        # Resolve dynamic values
        cfg = self._resolve_config(cfg)

        # Validate configuration
        self._validate_config(cfg)

        return cfg

    def _compose_config(self, cfg: DictConfig) -> DictConfig:
        """
        Compose configuration using defaults (Hydra-style).

        Args:
            cfg: Configuration with defaults section

        Returns:
            DictConfig: Composed configuration
        """
        if "defaults" not in cfg:
            return cfg

        # Start with empty config
        composed_cfg = OmegaConf.create({})

        # Process defaults in order
        for default in cfg.defaults:
            if default == "_self_":
                # Merge current config (excluding defaults)
                current_cfg = OmegaConf.create(cfg)
                if "defaults" in current_cfg:
                    current_cfg = OmegaConf.create(
                        {k: v for k, v in current_cfg.items() if k != "defaults"}
                    )
                composed_cfg = OmegaConf.merge(composed_cfg, current_cfg)
            else:
                # Load and merge default config
                default_path = self.config_dir / f"{default}.yaml"
                if default_path.exists():
                    with open(default_path, "r") as f:
                        default_cfg = OmegaConf.load(f)

                    # Recursively compose if default has defaults
                    if "defaults" in default_cfg:
                        default_cfg = self._compose_config(default_cfg)

                    composed_cfg = OmegaConf.merge(composed_cfg, default_cfg)

        # If _self_ was not in defaults, merge current config at the end
        if "_self_" not in cfg.defaults:
            current_cfg = OmegaConf.create(cfg)
            if "defaults" in current_cfg:
                current_cfg = OmegaConf.create(
                    {k: v for k, v in current_cfg.items() if k != "defaults"}
                )
            composed_cfg = OmegaConf.merge(composed_cfg, current_cfg)

        return composed_cfg

    def _resolve_config(self, cfg: DictConfig) -> DictConfig:
        """
        Resolve dynamic values in configuration.

        Args:
            cfg: Configuration to resolve

        Returns:
            DictConfig: Resolved configuration
        """
        # Auto-generate run name if not set
        if cfg.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"qwen_{cfg.model_size}_{cfg.num_train_epochs}ep_{timestamp}"
            cfg.run_name = run_name

        # Auto-generate logging directory
        if cfg.logging_dir is None:
            cfg.logging_dir = f"{cfg.tb_dir}/{cfg.run_name}"

        resolved_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(resolved_dict)

        return cfg

    def _validate_config(self, cfg: DictConfig):
        """
        Validate configuration for common issues.
        Environment and GPU settings are handled by launcher script.

        Args:
            cfg: Configuration to validate
        """
        # Check required paths
        if not Path(cfg.model_path).exists():
            raise ValueError(f"Model path does not exist: {cfg.model_path}")

        # Check data paths
        data_root = Path(cfg.data_root)
        train_path = data_root / cfg.train_data_path
        val_path = data_root / cfg.val_data_path

        if not train_path.exists():
            raise ValueError(f"Training data not found: {train_path}")
        if not val_path.exists():
            raise ValueError(f"Validation data not found: {val_path}")

        # Check learning rates
        lr_values = [cfg.vision_lr, cfg.mlp_lr, cfg.llm_lr]
        if all(lr == 0 for lr in lr_values):
            raise ValueError(
                "All learning rates are zero - at least one module must be enabled"
            )

        # Check batch size configuration
        if cfg.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")

        if cfg.total_batch_size <= 0:
            raise ValueError("total_batch_size must be positive")

        print("ℹ️  GPU and environment configuration handled by launcher script")

    def save_config(self, cfg: DictConfig, output_path: str):
        """
        Save configuration to file.

        Args:
            cfg: Configuration to save
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            OmegaConf.save(cfg, f)

    def print_config(self, cfg: DictConfig, resolve: bool = True):
        """
        Print configuration in a readable format.

        Args:
            cfg: Configuration to print
            resolve: Whether to resolve interpolations before printing
        """
        if resolve:
            cfg = OmegaConf.create(cfg)  # Create copy to avoid modifying original
            OmegaConf.resolve(cfg)

        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    def to_legacy_config(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Convert OmegaConf configuration to legacy Config class format.
        Environment and GPU settings will be added by launcher script.

        Args:
            cfg: OmegaConf configuration

        Returns:
            Dict: Legacy configuration format (flattened)
        """
        # Convert to dictionary and return directly since YAML is already flattened
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Add some derived values for compatibility
        config_dict["output_dir"] = f"{cfg.output_base_dir}/{cfg.run_name}"
        config_dict["batch_size"] = cfg.per_device_train_batch_size
        config_dict["num_epochs"] = cfg.num_train_epochs

        # Set deepspeed path if enabled
        if config_dict["deepspeed_enabled"]:
            config_dict["deepspeed"] = config_dict["deepspeed_config_file"]
        else:
            config_dict["deepspeed"] = None

        # Set log_dir for compatibility
        config_dict["log_dir"] = "logs"

        return config_dict


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_name: str, overrides: Optional[list] = None) -> DictConfig:
    """
    Convenience function to load configuration.

    Args:
        config_name: Name of the configuration
        overrides: List of override strings

    Returns:
        DictConfig: Loaded configuration
    """
    return config_manager.load_config(config_name, overrides)


def list_available_configs() -> list:
    """
    List all available configuration files.

    Returns:
        List of available configuration names
    """
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_files = list(config_dir.glob("*.yaml"))
    return [f.stem for f in config_files]
