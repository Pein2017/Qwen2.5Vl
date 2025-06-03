"""
Unified Configuration System for Qwen2.5-VL BBU Training.

This module provides:
1. Clean dataclass with automatic YAML loading and validation
2. Sensible defaults for optional parameters
3. Clear separation: YAML (hyperparams) vs Launcher (environment) vs Dataclass (schema)
4. Minimal redundancy and easy maintenance

Key improvements:
- No more try/except blocks for each field
- Automatic type validation and coercion
- Environment variables handled by launcher script only
- Clear error messages for missing/invalid fields
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from transformers import TrainingArguments


@dataclass
class UnifiedConfig:
    """
    Unified Configuration with automatic validation and sensible defaults.

    Core Philosophy:
    - YAML files: Only hyperparameters and model-specific settings
    - Launcher script: Environment variables and GPU/distributed settings
    - This dataclass: Schema, validation, and defaults
    """

    # ========================================================================
    # CORE REQUIRED PARAMETERS (must be in YAML)
    # ========================================================================
    model_path: str
    train_data_path: str
    val_data_path: str

    # ========================================================================
    # MODEL CONFIGURATION (with sensible defaults)
    # ========================================================================
    model_size: str = "3B"  # "3B" | "7B"
    model_max_length: int = 10000
    cache_dir: str = "/data4/swift/model_cache"
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"

    # ========================================================================
    # DATA CONFIGURATION (with sensible defaults)
    # ========================================================================
    data_root: str = "./"
    max_pixels: int = 1003520
    min_pixels: int = 784

    # ========================================================================
    # LEARNING RATES (auto-determines training strategy)
    # ========================================================================
    learning_rate: float = 5e-7
    vision_lr: float = 5e-8  # >0 = train vision encoder
    mlp_lr: float = 4e-7  # >0 = train MLP connector
    llm_lr: float = 1e-8  # >0 = train LLM

    # ========================================================================
    # TRAINING HYPERPARAMETERS (with sensible defaults)
    # ========================================================================
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    total_batch_size: int = 16
    gradient_accumulation_steps: int = 8

    # ========================================================================
    # OPTIMIZATION (with sensible defaults)
    # ========================================================================
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True

    # ========================================================================
    # PRECISION (with sensible defaults)
    # ========================================================================
    bf16: bool = True
    fp16: bool = False

    # ========================================================================
    # EVALUATION AND SAVING (with sensible defaults)
    # ========================================================================
    eval_strategy: str = "steps"
    eval_steps: int = 20
    save_strategy: str = "steps"
    save_steps: int = 20
    save_total_limit: int = 2

    # ========================================================================
    # LOGGING (with sensible defaults)
    # ========================================================================
    logging_steps: int = 10
    log_level: str = "INFO"
    report_to: str = "tensorboard"
    verbose: bool = True
    logging_dir: Optional[str] = None  # Auto-generated

    # ========================================================================
    # LOSS CONFIGURATION (with sensible defaults)
    # ========================================================================
    loss_type: str = "object_detection"
    lm_weight: float = 1.0
    bbox_weight: float = 0.6
    giou_weight: float = 0.4
    class_weight: float = 0.3
    hungarian_matching: bool = True
    detection_mode: str = "inference"
    inference_frequency: int = 5
    max_generation_length: int = 1024
    use_semantic_similarity: bool = True

    # ========================================================================
    # PERFORMANCE (with sensible defaults)
    # ========================================================================
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    data_flatten: bool = False
    batching_strategy: str = "simple"
    remove_unused_columns: bool = False

    # ========================================================================
    # DEEPSPEED (with sensible defaults)
    # ========================================================================
    deepspeed_enabled: bool = True
    deepspeed_config_file: str = "scripts/zero2.json"

    # ========================================================================
    # OUTPUT (auto-generated, not in YAML)
    # ========================================================================
    output_base_dir: str = "output_detection"
    run_name: Optional[str] = None  # Auto-generated
    tb_dir: str = "tb_detection"

    # ========================================================================
    # STABILITY AND MONITORING (with sensible defaults)
    # ========================================================================
    max_consecutive_nan: int = 5
    max_consecutive_zero: int = 5
    max_nan_ratio: float = 0.3
    nan_monitoring_window: int = 100
    allow_occasional_nan: bool = True
    nan_recovery_enabled: bool = True
    learning_rate_reduction_factor: float = 0.5
    gradient_clip_reduction_factor: float = 0.5

    # ========================================================================
    # DEBUG MODE (with sensible defaults)
    # ========================================================================
    test_samples: int = 2
    test_forward_pass: bool = False
    debug_mode: bool = False

    # ========================================================================
    # ENVIRONMENT SETTINGS (set by launcher, not YAML)
    # ========================================================================
    # These are set by the launcher script and should NOT be in YAML
    output_dir: Optional[str] = None  # Auto-generated from output_base_dir + run_name
    nproc_per_node: Optional[int] = None  # Set by launcher
    cuda_visible_devices: Optional[str] = None  # Set by launcher
    master_addr: Optional[str] = None  # Set by launcher
    master_port: Optional[int] = None  # Set by launcher

    def __post_init__(self):
        """Post-initialization validation and auto-generation of derived fields."""
        # Validate required fields
        self._validate_required_fields()

        # Auto-generate derived fields
        self._generate_derived_fields()

        # Validate configuration consistency
        self._validate_configuration()

    def _validate_required_fields(self):
        """Validate that all required fields are provided."""
        required_fields = ["model_path", "train_data_path", "val_data_path"]
        missing_fields = []

        for field_name in required_fields:
            value = getattr(self, field_name)
            if not value or (isinstance(value, str) and value.strip() == ""):
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"❌ Missing required configuration fields: {missing_fields}\n"
                f"💡 Please provide these fields in your YAML configuration file."
            )

    def _generate_derived_fields(self):
        """Generate derived fields from core configuration."""
        # Auto-generate run_name if not provided
        if not self.run_name:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = (
                f"{self.model_size.lower()}_lr{self.learning_rate}_{timestamp}"
            )

        # Auto-generate output_dir
        if not self.output_dir:
            self.output_dir = f"{self.output_base_dir}/{self.run_name}"

        # Auto-generate logging_dir if not provided
        if not self.logging_dir:
            self.logging_dir = f"logs/{self.run_name}"

        # Auto-calculate gradient accumulation steps if needed
        if hasattr(self, "nproc_per_node") and self.nproc_per_node:
            expected_grad_accum = self.total_batch_size // (
                self.nproc_per_node * self.per_device_train_batch_size
            )
            if expected_grad_accum > 0:
                self.gradient_accumulation_steps = expected_grad_accum

    def _validate_configuration(self):
        """Validate configuration for consistency and correctness."""
        # Check that at least one learning rate is non-zero
        if self.vision_lr == 0 and self.mlp_lr == 0 and self.llm_lr == 0:
            raise ValueError(
                "❌ At least one learning rate (vision_lr, mlp_lr, llm_lr) must be > 0.\n"
                f"💡 Current values: vision_lr={self.vision_lr}, mlp_lr={self.mlp_lr}, llm_lr={self.llm_lr}"
            )

        # Validate paths exist
        self._validate_paths()

        # Validate DeepSpeed config if enabled
        if self.deepspeed_enabled and not Path(self.deepspeed_config_file).exists():
            raise ValueError(
                f"❌ DeepSpeed config file not found: {self.deepspeed_config_file}"
            )

    def _validate_paths(self):
        """Validate that required paths exist."""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ValueError(f"❌ Model path does not exist: {self.model_path}")

        data_root = Path(self.data_root)
        train_path = data_root / self.train_data_path
        val_path = data_root / self.val_data_path

        if not train_path.exists():
            raise ValueError(f"❌ Training data not found: {train_path}")
        if not val_path.exists():
            raise ValueError(f"❌ Validation data not found: {val_path}")

    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained."""
        return self.vision_lr > 0

    @property
    def tune_mlp(self) -> bool:
        """Auto-determine if MLP connector should be trained."""
        return self.mlp_lr > 0

    @property
    def tune_llm(self) -> bool:
        """Auto-determine if LLM should be trained."""
        return self.llm_lr > 0

    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.mlp_lr, self.llm_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1

    @property
    def deepspeed(self) -> Optional[str]:
        """Get DeepSpeed config path if enabled."""
        return self.deepspeed_config_file if self.deepspeed_enabled else None

    def update_environment_config(
        self,
        nproc_per_node: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        """
        Update environment configuration from launcher script.

        This method is called by the launcher to set distributed training parameters.
        These should NEVER be in YAML files - only set by the launcher script.
        """
        if nproc_per_node is not None:
            self.nproc_per_node = nproc_per_node
        if cuda_visible_devices is not None:
            self.cuda_visible_devices = cuda_visible_devices
        if master_addr is not None:
            self.master_addr = master_addr
        if master_port is not None:
            self.master_port = master_port

        # Regenerate derived fields that depend on environment config
        self._generate_derived_fields()


class ConfigManager:
    """
    Unified Configuration Manager.

    Handles loading YAML files and creating UnifiedConfig instances with
    clear error messages and validation.
    """

    @staticmethod
    def load_config(
        config_name: str, overrides: Optional[List[str]] = None
    ) -> UnifiedConfig:
        """
        Load configuration from YAML file with optional overrides.

        Args:
            config_name: Name of config file (without .yaml extension)
            overrides: List of key=value override strings

        Returns:
            UnifiedConfig: Validated configuration instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        # Load YAML file
        config_path = Path(f"configs/{config_name}.yaml")
        if not config_path.exists():
            available_configs = [f.stem for f in Path("configs").glob("*.yaml")]
            raise FileNotFoundError(
                f"❌ Configuration file not found: {config_path}\n"
                f"💡 Available configurations: {available_configs}"
            )

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Apply overrides if provided
        if overrides:
            for override in overrides:
                if "=" not in override:
                    raise ValueError(
                        f"❌ Invalid override format: {override}. Use key=value format."
                    )

                key, value = override.split("=", 1)

                # Try to convert value to appropriate type
                try:
                    # Try int first
                    if value.isdigit() or (
                        value.startswith("-") and value[1:].isdigit()
                    ):
                        value = int(value)
                    # Try float (including scientific notation like 1e-5)
                    elif "." in value or "e" in value.lower():
                        value = float(value)
                    # Try bool
                    elif value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    # Keep as string
                except ValueError:
                    pass  # Keep as string

                config_dict[key] = value

        # Post-process config_dict to ensure proper type conversion
        config_dict = ConfigManager._post_process_config_dict(config_dict)

        # Create and return UnifiedConfig instance
        try:
            return UnifiedConfig(**config_dict)
        except TypeError as e:
            # Extract field information from error
            error_msg = str(e)
            if "unexpected keyword argument" in error_msg:
                field_name = error_msg.split("'")[1]
                raise ValueError(
                    f"❌ Unknown configuration field: {field_name}\n"
                    f"💡 This field is not supported in the configuration schema.\n"
                    f"💡 Please check your YAML file or remove this field."
                )
            raise ValueError(f"❌ Configuration error: {e}")

    @staticmethod
    def _post_process_config_dict(config_dict: Dict) -> Dict:
        """
        Post-process configuration dictionary to ensure proper type conversion.

        This handles cases where YAML loading doesn't properly convert scientific notation
        and other edge cases.
        """
        # Fields that should be converted to float
        float_fields = [
            "learning_rate",
            "vision_lr",
            "mlp_lr",
            "llm_lr",
            "warmup_ratio",
            "max_grad_norm",
            "weight_decay",
            "lm_weight",
            "bbox_weight",
            "giou_weight",
            "class_weight",
            "max_nan_ratio",
            "learning_rate_reduction_factor",
            "gradient_clip_reduction_factor",
        ]

        # Fields that should be converted to int
        int_fields = [
            "model_max_length",
            "max_pixels",
            "min_pixels",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "total_batch_size",
            "gradient_accumulation_steps",
            "eval_steps",
            "save_steps",
            "save_total_limit",
            "logging_steps",
            "dataloader_num_workers",
            "prefetch_factor",
            "inference_frequency",
            "max_generation_length",
            "max_consecutive_nan",
            "max_consecutive_zero",
            "nan_monitoring_window",
            "test_samples",
        ]

        # Fields that should be converted to bool
        bool_fields = [
            "gradient_checkpointing",
            "bf16",
            "fp16",
            "pin_memory",
            "data_flatten",
            "remove_unused_columns",
            "deepspeed_enabled",
            "hungarian_matching",
            "use_semantic_similarity",
            "allow_occasional_nan",
            "nan_recovery_enabled",
            "test_forward_pass",
            "debug_mode",
            "verbose",
        ]

        processed_dict = config_dict.copy()

        # Convert float fields
        for field in float_fields:
            if field in processed_dict:
                try:
                    processed_dict[field] = float(processed_dict[field])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails

        # Convert int fields
        for field in int_fields:
            if field in processed_dict:
                try:
                    processed_dict[field] = int(processed_dict[field])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails

        # Convert bool fields
        for field in bool_fields:
            if field in processed_dict:
                value = processed_dict[field]
                if isinstance(value, str):
                    processed_dict[field] = value.lower() in ("true", "1", "yes", "on")
                else:
                    try:
                        processed_dict[field] = bool(value)
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

        return processed_dict

    @staticmethod
    def save_config(config: UnifiedConfig, output_path: str):
        """Save configuration to YAML file for reproducibility."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict, excluding environment variables
        config_dict = {}
        for field_name, field_obj in config.__dataclass_fields__.items():
            value = getattr(config, field_name)

            # Skip environment variables (these should not be saved)
            if field_name in [
                "nproc_per_node",
                "cuda_visible_devices",
                "master_addr",
                "master_port",
            ]:
                continue

            # Skip None values
            if value is not None:
                config_dict[field_name] = value

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    @staticmethod
    def print_config(config: UnifiedConfig):
        """Pretty print configuration for debugging."""
        print("🔧 Configuration Summary:")
        print("=" * 50)

        # Group fields by category
        categories = {
            "Model": [
                "model_path",
                "model_size",
                "model_max_length",
                "cache_dir",
                "attn_implementation",
                "torch_dtype",
            ],
            "Data": [
                "train_data_path",
                "val_data_path",
                "data_root",
                "max_pixels",
                "min_pixels",
            ],
            "Learning Rates": ["learning_rate", "vision_lr", "mlp_lr", "llm_lr"],
            "Training": [
                "num_train_epochs",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "total_batch_size",
                "gradient_accumulation_steps",
            ],
            "Optimization": [
                "warmup_ratio",
                "lr_scheduler_type",
                "max_grad_norm",
                "weight_decay",
                "gradient_checkpointing",
            ],
            "Loss": [
                "loss_type",
                "lm_weight",
                "bbox_weight",
                "giou_weight",
                "class_weight",
            ],
            "Output": ["output_dir", "run_name", "logging_dir"],
            "Environment": [
                "nproc_per_node",
                "cuda_visible_devices",
                "master_addr",
                "master_port",
            ],
        }

        for category, field_names in categories.items():
            print(f"\n📋 {category}:")
            for field_name in field_names:
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    print(f"   {field_name}: {value}")

    @staticmethod
    def to_legacy_config(config: UnifiedConfig) -> Dict:
        """
        Convert UnifiedConfig to legacy config dict format.

        This is a temporary method to support the existing codebase
        while we transition to the new configuration system.
        """
        legacy_dict = {}

        # Map all fields from UnifiedConfig to legacy format
        for field_name, field_obj in config.__dataclass_fields__.items():
            value = getattr(config, field_name)
            if value is not None:
                legacy_dict[field_name] = value

        return legacy_dict


def create_training_arguments_from_config(config: UnifiedConfig) -> TrainingArguments:
    """
    Create TrainingArguments from UnifiedConfig.

    This replaces the old create_training_arguments_with_deepspeed function
    with a cleaner approach that uses the unified configuration system.
    """
    from transformers import TrainingArguments

    # Build TrainingArguments from config
    args_dict = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "weight_decay": config.weight_decay,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "evaluation_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "logging_dir": config.logging_dir,
        "log_level": config.log_level.lower(),
        "report_to": config.report_to,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.pin_memory,
        "dataloader_prefetch_factor": config.prefetch_factor,
        "remove_unused_columns": config.remove_unused_columns,
        "run_name": config.run_name,
    }

    # Add DeepSpeed configuration if enabled
    if config.deepspeed_enabled and config.deepspeed:
        if Path(config.deepspeed).exists():
            args_dict["deepspeed"] = config.deepspeed
        else:
            raise ValueError(f"❌ DeepSpeed config file not found: {config.deepspeed}")

    return TrainingArguments(**args_dict)


# Convenience function for easy config loading
def load_config(
    config_name: str, overrides: Optional[List[str]] = None
) -> UnifiedConfig:
    """Convenience function to load configuration."""
    return ConfigManager.load_config(config_name, overrides)
