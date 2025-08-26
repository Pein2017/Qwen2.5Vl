"""
Unified configuration management for Qwen2.5-VL training.

This module provides a single configuration class that directly maps to the
existing bbu_v2.yaml file without requiring any modifications to the YAML.

Key Features:
- Direct field mapping from YAML to dataclass fields
- Comprehensive type annotations and validation
- Fail-fast error handling with detailed error messages
- Strict validation requiring all fields to be explicitly defined
"""

import logging
from dataclasses import MISSING, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin

import yaml

from src_new.config.augmentation_config import (
    AngleRotateConfig as AngleRotateConfigType,
)
from src_new.config.augmentation_config import (
    AugmentationConfig as AugmentationConfigType,
)
from src_new.config.augmentation_config import (
    ColorJitterConfig as ColorJitterConfigType,
)
from src_new.config.augmentation_config import (
    validate_angle_rotate_config as validate_angle_rotate_config_fn,
)
from src_new.config.augmentation_config import (
    validate_augmentation_config as validate_augmentation_config_fn,
)
from src_new.config.augmentation_config import (
    validate_color_jitter_config as validate_color_jitter_config_fn,
)
from src_new.utils.data_resolver import DataResolver
from src_new.utils.validation import (
    PathValidationError,
    PathValidator,
    normalize_path_input,
)


logger = logging.getLogger(__name__)

# Global logging configuration (compatibility with legacy callers)
# These values mirror the state managed by src_new.utils.rank_aware_logging.
_GLOBAL_LOG_LEVEL = logging.INFO
_CONFIGURED_LOGGERS = set()


def set_global_log_level(level: str) -> None:
    """Set global log level for all loggers in the system.

    This delegates to the centralized rank-aware logging utilities to ensure
    consistent behavior across ranks and modules. The local variables remain
    only for backward compatibility with legacy callers.
    """
    global _GLOBAL_LOG_LEVEL

    # Normalize to string for the rank-aware API, which accepts str or int
    normalized_level = level
    try:
        # Defer to rank-aware system for propagation across all loggers
        from ..utils.rank_aware_logging import set_global_log_level as _set_global

        _set_global(normalized_level)
    except Exception:
        # Fallback: update root logger if rank-aware utilities are unavailable
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            resolved = level_map[level.upper()]
        else:
            resolved = int(level)
        logging.getLogger().setLevel(resolved)
        for logger_name in _CONFIGURED_LOGGERS:
            logging.getLogger(logger_name).setLevel(resolved)
    finally:
        # Keep local mirror updated for compatibility (e.g., script fallback)
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            _GLOBAL_LOG_LEVEL = level_map[level.upper()]
        else:
            _GLOBAL_LOG_LEVEL = int(level)


from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("config")


def _convert_scientific_notation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numeric-like strings to numbers based on Config type annotations.

    Only keys whose annotation is float are converted; other strings are left intact.
    """
    converted_data: Dict[str, Any] = dict(data)
    annotations = getattr(Config, "__annotations__", {})

    for key, value in list(data.items()):
        if isinstance(value, str) and key in annotations:
            ann = annotations[key]
            origin = get_origin(ann)
            # Only convert for float annotations (allow Optional[float])
            is_float_ann = ann is float or (
                origin is __import__("typing").Union and float in get_args(ann)
            )
            if is_float_ann:
                try:
                    converted_data[key] = float(value)
                except ValueError:
                    # Keep original string if not convertible
                    converted_data[key] = value
        # Leave other keys as-is
    return converted_data


def _is_value_of_type(value: Any, annotation: Any) -> bool:
    """Lightweight runtime type check for common typing annotations.

    Supports Optional[T], List[T], and basic primitives (int, float, bool, str).
    """
    if annotation is Any:
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] is represented as Union[T, NoneType]
    if origin is Optional or (origin is None and annotation is Optional):
        # Fallback guard; but typing.Optional always yields Union
        return value is None or True

    if origin is None:
        # Bare types like int, float, bool, str
        expected = annotation
        try:
            if expected is bool:
                return isinstance(value, bool)
            if expected is int:
                return isinstance(value, int) and not isinstance(value, bool)
            if expected is float:
                # Accept int for float
                return (
                    isinstance(value, float) or isinstance(value, int)
                ) and not isinstance(value, bool)
            if expected is str:
                return isinstance(value, str)
            # Unknown annotations: accept
            return True
        except Exception:
            return True

    # Handle Union[..., NoneType] as Optional
    if origin is list:
        # List[T]
        if not isinstance(value, list):
            return False
        if not args:
            return True
        elem_ann = args[0]
        return all(_is_value_of_type(elem, elem_ann) for elem in value)

    if origin is dict:
        # Not used heavily; accept
        return isinstance(value, dict)

    if origin is tuple:
        return isinstance(value, tuple)

    if origin is type(Optional[int]):  # pragma: no cover (defensive)
        return value is None or _is_value_of_type(value, args[0])

    # Union: any arg matches
    if origin is __import__("typing").Union:
        return any(_is_value_of_type(value, a) for a in args)

    # Default: do not block
    return True


def _collect_schema_issues(data: Dict[str, Any]) -> List[str]:
    """Collect unknown fields, missing required fields, and type mismatches.

    Returns list of human-readable issue strings.
    """
    issues: List[str] = []

    cfg_fields = {f.name: f for f in dataclass_fields(Config)}
    known_names = set(cfg_fields.keys())
    data_names = set(data.keys())

    unknown = sorted(list(data_names - known_names))
    if unknown:
        issues.append("Unknown fields: " + ", ".join(unknown))

    # Missing required fields (those without defaults)
    missing = [
        f.name
        for f in cfg_fields.values()
        if f.default is MISSING
        and f.default_factory is MISSING
        and f.name not in data_names
    ]
    if missing:
        issues.append("Missing required fields: " + ", ".join(sorted(missing)))

    # Type mismatches for present known fields
    type_mismatches: List[str] = []
    annotations = Config.__annotations__
    for name in sorted(known_names & data_names):
        ann = annotations.get(name)
        if ann is None:
            continue
        val = data.get(name)
        # Allow None for optional types handled in downstream validation
        try:
            origin = get_origin(ann)
            args = get_args(ann)
            is_optional = origin is __import__("typing").Union and type(None) in args
        except Exception:
            is_optional = False

        if val is None and is_optional:
            continue
        if not _is_value_of_type(val, ann):
            type_mismatches.append(f"{name} (expected {ann}, got {type(val).__name__})")

    if type_mismatches:
        issues.append("Type mismatches: " + ", ".join(type_mismatches))

    return issues


@dataclass(frozen=True)
class Config:
    """
    Unified configuration class with direct mapping to bbu_v2.yaml.

    All field names and types match exactly with the existing YAML structure
    to ensure zero-modification compatibility. Every parameter is explicitly
    defined with proper type hints and validation.

    Note: All required fields (no defaults) must come first in dataclass definition.
    """

    # === REQUIRED FIELDS (no defaults) ===
    # Model settings
    model_path: str
    attn_implementation: str
    torch_dtype: str
    use_cache: bool

    # Training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool

    # Data settings
    train_data_path: str
    val_data_path: str
    data_root: str
    teacher_pool_file: str
    max_total_length: int
    num_teacher_samples: int
    max_dataset_size: int

    # Teacher-student loss (aggregated)
    teacher_loss_weight: float
    student_loss_weight: float

    # Features
    coordinate_tokens_enabled: bool
    coordinate_init_mode: Optional[str]

    # Vision processing parameters
    merge_size: int
    max_pixels: int  # Qwen2VL image processor's max_pixels

    # Output settings
    output_dir: str
    run_name: str  # tensorboard event name
    tb_dir: str

    # Coordinate/token limits
    max_coord_value: int

    # Loss settings
    coordinate_loss_weight: float
    regular_loss_weight: float
    teacher_ratio: float

    # Collator
    collator_type: str

    # HF Trainer: evaluation/checkpoint settings
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool

    # Logging settings
    logging_steps: int
    report_to: str
    disable_tqdm: bool

    # Dataloader performance
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    remove_unused_columns: bool

    # Coordinate auxiliary losses (enabled via YAML)
    coord_aux_enabled: bool

    # Progressive unfreeze (enabled via YAML)
    prog_unfreeze_enabled: bool

    # === OPTIONAL FIELDS WITH DEFAULTS (truly optional) ===
    seed: int = 17
    new_geometry_tokens: Optional[List[str]] = None
    augmentation: Optional[AugmentationConfigType] = None

    # Coordinate aux knobs (only when coord_aux_enabled)
    coord_aux_tau: Optional[float] = None
    coord_aux_sigma_bins: Optional[int] = None
    coord_aux_window_bins: Optional[int] = None
    coord_aux_topk: Optional[int] = None
    coord_aux_lambda_kce: Optional[float] = None
    coord_aux_lambda_unlike: Optional[float] = None

    # Progressive unfreeze knobs
    prog_unfreeze_epoch_stage0_end: Optional[int] = None
    prog_unfreeze_epoch_stage1_end: Optional[int] = None
    prog_unfreeze_top_k_layers: Optional[int] = None
    prog_unfreeze_coord_slice_only: bool = True

    # Optional learning rates for specific parameter groups (progressive unfreeze / fine-grained control)
    lr_merger: Optional[float] = None
    lr_coord_slice: Optional[float] = None
    lr_top_layers: Optional[float] = None
    lr_full_model: Optional[float] = None

    # === COMPUTED PROPERTIES ===
    @property
    def run_output_dir(self) -> str:
        """Computed output directory path."""
        return str(Path(self.output_dir) / self.run_name)

    @property
    def tensorboard_dir(self) -> str:
        """Computed TensorBoard directory path."""
        return str(Path(self.tb_dir) / self.run_name)

    @property
    def log_file_dir(self) -> str:
        """Computed log file directory path."""
        return str(Path(self.run_output_dir) / "logs")

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Performs fail-fast validation to catch configuration errors early
        with detailed error messages to help debugging.

        Raises:
        ValueError: If any configuration values are invalid
        FileNotFoundError: If required paths don't exist
        """
        self._validate_model_settings()
        self._validate_training_settings()
        self._validate_data_settings()
        self._validate_coordinate_settings()
        self._validate_progressive_unfreeze_settings()
        self._validate_augmentation_settings()

        logger.info("✅ Configuration validation passed")

    def _validate_model_settings(self) -> None:
        """Validate model-related settings."""
        if not self.model_path:
            raise ValueError("model_path cannot be empty")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        if self.attn_implementation not in ["flash_attention_2", "eager", "sdpa"]:
            raise ValueError(f"Invalid attn_implementation: {self.attn_implementation}")

        if self.torch_dtype not in ["float16", "bfloat16", "float32", "bf16", "fp16"]:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}")

        if self.max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {self.max_pixels}")

    def _validate_training_settings(self) -> None:
        """Validate training-related settings."""
        if self.num_train_epochs <= 0:
            raise ValueError(
                f"num_train_epochs must be positive, got {self.num_train_epochs}"
            )

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, got {self.per_device_train_batch_size}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.vision_lr < 0:
            raise ValueError(f"vision_lr cannot be negative, got {self.vision_lr}")

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError(
                f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}"
            )

        # Seed must be non-negative
        if getattr(self, "seed", 17) < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    def _validate_data_settings(self) -> None:
        """Validate data-related settings."""
        if not self.train_data_path:
            raise ValueError("train_data_path cannot be empty")

        if not self.val_data_path:
            raise ValueError("val_data_path cannot be empty")

        if not self.teacher_pool_file:
            raise ValueError("teacher_pool_file cannot be empty")

        # Validate existence using centralized validator (accept relative or aliases)
        try:
            PathValidator.validate_file_exists(self.train_data_path)
            PathValidator.validate_file_exists(self.val_data_path)
            PathValidator.validate_file_exists(self.teacher_pool_file)
            PathValidator.validate_directory_exists(self.data_root)
        except (ValueError, PathValidationError) as e:
            raise ValueError(f"Invalid data paths: {e}")

        if self.teacher_ratio < 0 or self.teacher_ratio > 1:
            raise ValueError(
                f"teacher_ratio must be between 0 and 1, got {self.teacher_ratio}"
            )

        if self.collator_type not in ["packed", "standard"]:
            raise ValueError(f"Invalid collator_type: {self.collator_type}")

        # Required: validate coordinate init mode
        if self.coordinate_tokens_enabled:
            if self.coordinate_init_mode is None:
                raise ValueError(
                    "coordinate_init_mode is required when coordinate_tokens_enabled=True"
                )
            allowed = {"ms_mean", "fourier_ramp"}
            if self.coordinate_init_mode not in allowed:
                raise ValueError(
                    f"coordinate_init_mode must be one of {sorted(allowed)}, got {self.coordinate_init_mode!r}"
                )

        # Output/log paths: accept relative; no existence check required here

        # Initialize new_geometry_tokens if not provided
        if self.coordinate_tokens_enabled and self.new_geometry_tokens is None:
            # Only add line tokens - quad tokens already exist in Qwen2.5-VL
            object.__setattr__(
                self,
                "new_geometry_tokens",
                [
                    "<|line_start|>",
                    "<|line_end|>",
                ],
            )

    def _validate_coordinate_settings(self) -> None:
        """Validate coordinate token settings."""
        if self.max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be positive, got {self.max_coord_value}"
            )

        if self.coordinate_loss_weight < 0:
            raise ValueError(
                f"coordinate_loss_weight cannot be negative, got {self.coordinate_loss_weight}"
            )

        if self.regular_loss_weight <= 0:
            raise ValueError(
                f"regular_loss_weight must be positive, got {self.regular_loss_weight}"
            )

        # Coordinate auxiliary losses validation
        if self.coord_aux_enabled:
            missing: list[str] = []
            if self.coord_aux_tau is None:
                missing.append("coord_aux_tau")
            if self.coord_aux_sigma_bins is None:
                missing.append("coord_aux_sigma_bins")
            if self.coord_aux_window_bins is None:
                missing.append("coord_aux_window_bins")
            if self.coord_aux_topk is None:
                missing.append("coord_aux_topk")
            if self.coord_aux_lambda_kce is None:
                missing.append("coord_aux_lambda_kce")
            if self.coord_aux_lambda_unlike is None:
                missing.append("coord_aux_lambda_unlike")
            if missing:
                raise ValueError(
                    "coord_aux_enabled=True but missing required fields: "
                    + ", ".join(missing)
                )
            if self.coord_aux_tau is not None and self.coord_aux_tau <= 0:
                raise ValueError(f"coord_aux_tau must be > 0, got {self.coord_aux_tau}")
            if self.coord_aux_sigma_bins is not None and self.coord_aux_sigma_bins <= 0:
                raise ValueError(
                    f"coord_aux_sigma_bins must be > 0, got {self.coord_aux_sigma_bins}"
                )
            if (
                self.coord_aux_window_bins is not None
                and self.coord_aux_window_bins < 1
            ):
                raise ValueError(
                    f"coord_aux_window_bins must be >= 1, got {self.coord_aux_window_bins}"
                )
            if self.coord_aux_topk is not None and self.coord_aux_topk < 1:
                raise ValueError(
                    f"coord_aux_topk must be >= 1, got {self.coord_aux_topk}"
                )
            if (
                self.coord_aux_lambda_kce is not None
                and self.coord_aux_lambda_unlike is not None
                and (self.coord_aux_lambda_kce < 0 or self.coord_aux_lambda_unlike < 0)
            ):
                raise ValueError("coord_aux_lambda_kce/unlike must be non-negative")
        # Required: validate coordinate init mode
        if self.coordinate_tokens_enabled:
            if self.coordinate_init_mode is None:
                raise ValueError(
                    "coordinate_init_mode is required when coordinate_tokens_enabled=True"
                )
            allowed = {"ms_mean", "fourier_ramp"}
            if self.coordinate_init_mode not in allowed:
                raise ValueError(
                    f"coordinate_init_mode must be one of {sorted(allowed)}, got {self.coordinate_init_mode!r}"
                )

        # Output/log paths: accept relative; no existence check required here

        # Initialize new_geometry_tokens if not provided
        if self.coordinate_tokens_enabled and self.new_geometry_tokens is None:
            # Only add line tokens - quad tokens already exist in Qwen2.5-VL
            object.__setattr__(
                self,
                "new_geometry_tokens",
                [
                    "<|line_start|>",
                    "<|line_end|>",
                ],
            )

    def _validate_progressive_unfreeze_settings(self) -> None:
        """Validate progressive unfreeze related settings when enabled."""
        if not self.prog_unfreeze_enabled:
            return

        # Epoch boundaries
        if self.prog_unfreeze_epoch_stage0_end is not None:
            if self.prog_unfreeze_epoch_stage0_end < 1:
                raise ValueError(
                    f"prog_unfreeze_epoch_stage0_end must be >= 1 when provided, got {self.prog_unfreeze_epoch_stage0_end}"
                )
        if (
            self.prog_unfreeze_epoch_stage1_end is not None
            and self.prog_unfreeze_epoch_stage0_end is not None
        ):
            if (
                self.prog_unfreeze_epoch_stage1_end
                <= self.prog_unfreeze_epoch_stage0_end
            ):
                raise ValueError(
                    "prog_unfreeze_epoch_stage1_end must be > prog_unfreeze_epoch_stage0_end"
                )

        # Top-K layers
        if (
            self.prog_unfreeze_top_k_layers is not None
            and self.prog_unfreeze_top_k_layers < 1
        ):
            raise ValueError(
                f"prog_unfreeze_top_k_layers must be >= 1 when provided, got {self.prog_unfreeze_top_k_layers}"
            )

        # Learning rates (if provided) must be positive
        for lr_name in (
            "lr_merger",
            "lr_coord_slice",
            "lr_top_layers",
            "lr_full_model",
        ):
            if hasattr(self, lr_name):
                lr_value = getattr(self, lr_name)
                if lr_value is not None and lr_value <= 0:
                    raise ValueError(
                        f"{lr_name} must be positive when provided, got {lr_value}"
                    )

    def _validate_augmentation_settings(self) -> None:
        """Validate augmentation settings when provided."""
        aug = getattr(self, "augmentation", None)
        if aug is None:
            return
        try:
            validate_augmentation_config_fn(aug)
            validate_angle_rotate_config_fn(aug.op)
        except Exception as e:
            raise ValueError(f"Invalid augmentation configuration: {e}")


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file with comprehensive validation.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML file is malformed
        ValueError: If configuration contains invalid values
        TypeError: If configuration values have wrong types
    """
    # FAIL-FAST: Validate config path
    if not config_path:
        raise ValueError("config_path cannot be empty")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_file.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    # Load YAML with error handling
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read config file {config_path}: {e}")

    # FAIL-FAST: Validate YAML loaded correctly
    if data is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(data)}")

    # Convert numeric-like strings to floats based on annotations
    data = _convert_scientific_notation(data)

    # === Unified dataset path defaults ===
    # Auto-derive data paths from data_root using centralized data resolver
    # Expected structure under data_root:
    #   - images/  (image files referenced in JSONL as ./images/xxx.jpeg)
    #   - train.jsonl
    #   - val.jsonl
    #   - teacher_pool.jsonl
    try:
        if "data_root" in data and data["data_root"]:
            data_root_value = data["data_root"]
            # Use DataResolver for automatic path discovery and validation
            # Only derive paths if they're not explicitly provided (backward compatibility)
            missing_paths = [
                key
                for key in ["train_data_path", "val_data_path", "teacher_pool_file"]
                if key not in data or not data[key]
            ]
            if missing_paths:
                try:
                    dataset_paths = DataResolver.resolve_dataset_paths(data_root_value)

                    # Only set paths that weren't explicitly provided
                    if "train_data_path" not in data or not data["train_data_path"]:
                        data["train_data_path"] = str(dataset_paths.train_data_path)
                    if "val_data_path" not in data or not data["val_data_path"]:
                        data["val_data_path"] = str(dataset_paths.val_data_path)
                    if "teacher_pool_file" not in data or not data["teacher_pool_file"]:
                        data["teacher_pool_file"] = str(dataset_paths.teacher_pool_file)

                    logger.debug(
                        f"✅ Auto-resolved dataset paths from data_root: {data_root_value}"
                    )
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(
                        f"⚠️ Could not auto-resolve dataset paths from data_root '{data_root_value}': {e}"
                    )
                    # Fall back to manual derivation for backward compatibility
                    data_root_path = Path(data_root_value)
                    if "train_data_path" not in data or not data["train_data_path"]:
                        data["train_data_path"] = str(data_root_path / "train.jsonl")
                    if "val_data_path" not in data or not data["val_data_path"]:
                        data["val_data_path"] = str(data_root_path / "val.jsonl")
                    if "teacher_pool_file" not in data or not data["teacher_pool_file"]:
                        data["teacher_pool_file"] = str(
                            data_root_path / "teacher_pool.jsonl"
                        )
    except Exception as e:
        # Do not block config loading if derivation fails; validation will catch later
        logger.debug(f"Data path derivation failed: {e}")
        pass

    # Normalize path-like fields; preserve relativity (no forced absolute)
    # Normalize data_root first
    if "data_root" in data and data["data_root"]:
        try:
            data["data_root"] = str(normalize_path_input(data["data_root"]))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for 'data_root': {data.get('data_root')}; {e}"
            )

    # Determine base for dataset files: prefer current working directory (PWD)
    # Relative paths will be interpreted from the process working directory

    # Normalize model_path relative to config_dir
    if "model_path" in data and data["model_path"]:
        try:
            data["model_path"] = str(normalize_path_input(data["model_path"]))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for 'model_path': {data.get('model_path')}; {e}"
            )

    # Normalize dataset file paths relative to dataset_base
    for key in ("train_data_path", "val_data_path", "teacher_pool_file"):
        if key in data and data[key]:
            try:
                data[key] = str(normalize_path_input(data[key]))
            except Exception as e:
                raise ValueError(
                    f"Failed to normalize path for '{key}': {data.get(key)}; {e}"
                )

    # Normalize output/log directories relative to config_dir
    for key in ("output_dir", "tb_dir", "logging_dir"):
        if key in data and data[key]:
            try:
                data[key] = str(normalize_path_input(data[key]))
            except Exception as e:
                raise ValueError(
                    f"Failed to normalize path for '{key}': {data.get(key)}; {e}"
                )

    # Build augmentation dataclasses from nested dicts (if provided)
    try:
        if "augmentation" in data and data["augmentation"] is not None:
            aug_dict = data["augmentation"]
            if not isinstance(aug_dict, dict):
                raise ValueError(
                    f"augmentation must be a mapping when provided, got {type(aug_dict)}"
                )
            op_dict = aug_dict.get("op")
            if not isinstance(op_dict, dict):
                raise ValueError("augmentation.op must be a mapping when provided")
            # Coerce fill_color to tuple if present
            fill_color_val = op_dict.get("fill_color")
            if fill_color_val is not None and isinstance(fill_color_val, list):
                op_dict = dict(op_dict)
                op_dict["fill_color"] = tuple(fill_color_val)
            angle_op_cfg = AngleRotateConfigType(**op_dict)
            # Optional color_jitter
            cj_cfg = None
            if "color_jitter" in aug_dict and aug_dict["color_jitter"] is not None:
                cj_dict = aug_dict["color_jitter"]
                if not isinstance(cj_dict, dict):
                    raise ValueError(
                        "augmentation.color_jitter must be a mapping when provided"
                    )

                def _to_tuple(name):
                    v = cj_dict.get(name)
                    return tuple(v) if isinstance(v, list) else v

                # Enforce explicit values; no implicit defaults here
                if "enabled" not in cj_dict or "apply_prob" not in cj_dict:
                    raise ValueError(
                        "color_jitter requires explicit 'enabled' and 'apply_prob' in config"
                    )
                cj_cfg = ColorJitterConfigType(
                    enabled=bool(cj_dict["enabled"]),
                    apply_prob=float(cj_dict["apply_prob"]),
                    brightness=_to_tuple("brightness"),
                    contrast=_to_tuple("contrast"),
                    saturation=_to_tuple("saturation"),
                    sharpness=_to_tuple("sharpness"),
                    order=cj_dict.get("order"),
                )
                validate_color_jitter_config_fn(cj_cfg)

            # Optional albumentations random aug
            alb_rand_cfg = None
            if (
                "albumentations_rand" in aug_dict
                and aug_dict["albumentations_rand"] is not None
            ):
                ar_dict = aug_dict["albumentations_rand"]
                if not isinstance(ar_dict, dict):
                    raise ValueError(
                        "augmentation.albumentations_rand must be a mapping when provided"
                    )
                from src_new.config.augmentation_config import (
                    AlbumentationsRandAugConfig as _AlbRand,
                )
                from src_new.config.augmentation_config import (
                    validate_albumentations_rand_config as _validate_alb,
                )

                # Enforce explicit values; no implicit defaults here
                required_ar = [
                    "enabled",
                    "apply_prob",
                    "num_ops",
                    "magnitude",
                    "safe_ops_only",
                ]
                missing_ar = [k for k in required_ar if k not in ar_dict]
                if missing_ar:
                    raise ValueError(
                        "albumentations_rand requires explicit fields: "
                        + ", ".join(missing_ar)
                    )
                alb_rand_cfg = _AlbRand(
                    enabled=bool(ar_dict["enabled"]),
                    apply_prob=float(ar_dict["apply_prob"]),
                    num_ops=int(ar_dict["num_ops"]),
                    magnitude=float(ar_dict["magnitude"]),
                    safe_ops_only=bool(ar_dict["safe_ops_only"]),
                )
                _validate_alb(alb_rand_cfg)

            # Optional object copy-paste
            ocp_cfg = None
            if (
                "object_copy_paste" in aug_dict
                and aug_dict["object_copy_paste"] is not None
            ):
                ocp_dict = aug_dict["object_copy_paste"]
                if not isinstance(ocp_dict, dict):
                    raise ValueError(
                        "augmentation.object_copy_paste must be a mapping when provided"
                    )
                from src_new.config.augmentation_config import (
                    ObjectCopyPasteConfig as _OCP,
                )
                from src_new.config.augmentation_config import (
                    validate_object_copy_paste_config as _validate_ocp,
                )

                ocp_cfg = _OCP(
                    enabled=bool(ocp_dict["enabled"]),
                    per_object_prob=float(ocp_dict["per_object_prob"]),
                    num_copies_per_object=int(ocp_dict["num_copies_per_object"]),
                    translate_px=int(ocp_dict["translate_px"]),
                    rotation_jitter_deg=float(ocp_dict["rotation_jitter_deg"]),
                    scale_jitter_min=float(ocp_dict["scale_jitter_min"]),
                    scale_jitter_max=float(ocp_dict["scale_jitter_max"]),
                    occ_grid_downscale=int(ocp_dict["occ_grid_downscale"]),
                    occ_margin_px=int(ocp_dict["occ_margin_px"]),
                    max_occ_fraction=float(ocp_dict["max_occ_fraction"]),
                    max_iou_with_existing=float(ocp_dict["max_iou_with_existing"]),
                    attempts=int(ocp_dict["attempts"]),
                    allowed_types=ocp_dict.get("allowed_types"),
                )
                _validate_ocp(ocp_cfg)

            # Optional object blur
            obj_blur_cfg = None
            if "object_blur" in aug_dict and aug_dict["object_blur"] is not None:
                ob_dict = aug_dict["object_blur"]
                if not isinstance(ob_dict, dict):
                    raise ValueError(
                        "augmentation.object_blur must be a mapping when provided"
                    )
                from src_new.config.augmentation_config import (
                    ObjectBlurConfig as _OB,
                )
                from src_new.config.augmentation_config import (
                    validate_object_blur_config as _validate_ob,
                )

                obj_blur_cfg = _OB(
                    enabled=bool(ob_dict["enabled"]),
                    per_object_prob=float(ob_dict["per_object_prob"]),
                    blur_type=str(ob_dict["blur_type"]),
                    radius_min=float(ob_dict["radius_min"]),
                    radius_max=float(ob_dict["radius_max"]),
                )
                _validate_ob(obj_blur_cfg)

            # Optional rand pool
            rand_pool_cfg = None
            if "rand_pool" in aug_dict and aug_dict["rand_pool"] is not None:
                rp_dict = aug_dict["rand_pool"]
                if not isinstance(rp_dict, dict):
                    raise ValueError(
                        "augmentation.rand_pool must be a mapping when provided"
                    )
                from src_new.config.augmentation_config import (
                    RandAugPoolConfig as _RP,
                )

                rand_pool_cfg = _RP(
                    enabled=bool(rp_dict["enabled"]),
                    apply_prob=float(rp_dict["apply_prob"]),
                    num_ops=int(rp_dict["num_ops"]),
                    include_object_affine=bool(
                        rp_dict.get("include_object_affine", False)
                    ),
                    include_object_copy_paste=bool(
                        rp_dict.get("include_object_copy_paste", False)
                    ),
                    include_object_blur=bool(rp_dict.get("include_object_blur", False)),
                )

            # Optional criteria
            criteria_cfg = None
            if "criteria" in aug_dict and aug_dict["criteria"] is not None:
                cr_dict = aug_dict["criteria"]
                if not isinstance(cr_dict, dict):
                    raise ValueError(
                        "augmentation.criteria must be a mapping when provided"
                    )
                from src_new.config.augmentation_config import CriteriaConfig as _CR
                from src_new.config.augmentation_config import (
                    OcclusionCriterionConfig as _OC,
                )

                occ_cfg = None
                occ_dict = cr_dict.get("occlusion")
                if occ_dict is not None:
                    if not isinstance(occ_dict, dict):
                        raise ValueError(
                            "augmentation.criteria.occlusion must be a mapping"
                        )
                    occ_cfg = _OC(
                        enabled=bool(occ_dict["enabled"]),
                        min_overlap_fraction_bbox=float(
                            occ_dict["min_overlap_fraction_bbox"]
                        ),
                        min_overlap_fraction_line=float(
                            occ_dict["min_overlap_fraction_line"]
                        ),
                        mask_downscale=int(occ_dict["mask_downscale"]),
                        line_width_px=int(occ_dict["line_width_px"]),
                    )
                criteria_cfg = _CR(occlusion=occ_cfg)

            aug_cfg = AugmentationConfigType(
                enabled=aug_dict["enabled"],
                rng_seed=aug_dict["rng_seed"],
                apply_to_teachers=aug_dict["apply_to_teachers"],
                lines_policy=aug_dict["lines_policy"],
                debug_visualization=aug_dict["debug_visualization"],
                debug_output_dir=aug_dict.get("debug_output_dir"),
                op=angle_op_cfg,
                color_jitter=cj_cfg,
                albumentations_rand=alb_rand_cfg,
                object_copy_paste=ocp_cfg,
                object_blur=obj_blur_cfg,
                rand_pool=rand_pool_cfg,
                criteria=criteria_cfg,
            )
            # Validate early (fail-fast)
            validate_angle_rotate_config_fn(angle_op_cfg)
            validate_augmentation_config_fn(aug_cfg)
            data["augmentation"] = aug_cfg
    except KeyError as e:
        raise ValueError(f"Missing required augmentation field: {e}")
    except TypeError as e:
        raise TypeError(f"Invalid augmentation field types: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse augmentation configuration: {e}")

    # Aggregate schema issues before constructing the dataclass
    schema_issues = _collect_schema_issues(data)
    if schema_issues:
        details = "\n - " + "\n - ".join(schema_issues)
        raise ValueError(
            f"Configuration schema validation failed with {len(schema_issues)} issue(s):{details}"
        )

    # All parameters must be explicitly provided in YAML configuration
    # No defaults are provided here to ensure fail-fast behavior

    # Create config with comprehensive error handling
    try:
        config = Config(**data)
    except TypeError as e:
        raise TypeError(f"Configuration contains invalid field types: {e}")
    except Exception as e:
        raise ValueError(f"Failed to create configuration from {config_path}: {e}")

    logger.info(f"✅ Configuration loaded successfully from {config_path}")
    logger.info(f"📋 Model path: {config.model_path}")
    logger.info(f"📋 Data: train={config.train_data_path}, val={config.val_data_path}")
    logger.info(
        f"📋 Coordinate tokens: {'enabled' if config.coordinate_tokens_enabled else 'disabled'}"
    )
    logger.info(f"📋 Teacher ratio: {config.teacher_ratio}")

    return config


def save_config(config: "Config", output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        output_path: Path where to save the YAML file

    Raises:
        ValueError: If config is None or output_path is empty
        OSError: If unable to write to the output path
    """
    if config is None:
        raise ValueError("config cannot be None")

    if not output_path:
        raise ValueError("output_path cannot be empty")

    # Convert dataclass to dictionary
    config_dict = {
        field_name: getattr(config, field_name)
        for field_name in config.__dataclass_fields__
    }

    # Write to YAML file
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=True)

    except Exception as e:
        raise OSError(f"Failed to save config to {output_path}: {e}")

    logger.info(f"✅ Configuration saved to {output_path}")
