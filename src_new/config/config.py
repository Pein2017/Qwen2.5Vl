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
    AugmentationConfig as AugmentationConfigType,
)
from src_new.config.augmentation_config import (
    validate_augmentation_config as validate_augmentation_config_fn,
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


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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

    # Group loss weights for caption/grounding/formatting (required)
    caption_loss_weight: float
    grounding_loss_weight: float
    formatting_loss_weight: float

    # === OPTIONAL FIELDS WITH DEFAULTS (truly optional) ===
    seed: int = 17
    new_geometry_tokens: Optional[List[str]] = None
    augmentation: Optional[AugmentationConfigType] = None
    # Optional: dedicated teacher augmentation (simple photometric-only). If provided,
    # it overrides `augmentation.apply_to_teachers` and is applied to teacher samples only.
    teacher_augmentation: Optional[AugmentationConfigType] = None
    use_aug: bool = False
    augmentation_schedule: Optional[List[Dict[str, Any]]] = None
    # Phase-freeze control for separate runs: one of {off, phase_1, phase_2, phase_3}
    phase_name: str = "off"

    # Coordinate aux knobs (only when coord_aux_enabled)
    coord_aux_tau: Optional[float] = None
    coord_aux_sigma_bins: Optional[int] = None
    coord_aux_window_bins: Optional[int] = None
    coord_aux_topk: Optional[int] = None
    coord_aux_lambda_kce: Optional[float] = None
    coord_aux_lambda_unlike: Optional[float] = None

    # Optional learning rates for specific parameter groups
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
        self._validate_augmentation_settings()
        self._validate_phase_name()
        self._validate_learning_rate_groups()
        self._validate_group_loss_settings()

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

    def _validate_learning_rate_groups(self) -> None:
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
            # Still validate schedule if present
            sched = getattr(self, "augmentation_schedule", None)
            if sched is not None:
                self._validate_augmentation_schedule(sched)
            return
        try:
            validate_augmentation_config_fn(aug)
        except Exception as e:
            raise ValueError(f"Invalid augmentation configuration: {e}")

        # Validate schedule alongside an explicit augmentation config if provided
        sched = getattr(self, "augmentation_schedule", None)
        if sched is not None:
            self._validate_augmentation_schedule(sched)

        # Optional teacher-specific augmentation
        taug = getattr(self, "teacher_augmentation", None)
        if taug is not None:
            try:
                validate_augmentation_config_fn(taug)
            except Exception as e:
                raise ValueError(f"Invalid teacher_augmentation configuration: {e}")

    def _validate_augmentation_schedule(self, schedule: List[Dict[str, Any]]) -> None:
        """
        Validate the augmentation schedule.

        Expected format (per entry): {"start_epoch": int, "preset": str}
        where preset ∈ {off, conservative, moderate, aggressive}.
        """
        if not isinstance(schedule, list):
            raise ValueError("augmentation_schedule must be a list of dictionaries.")

        valid_presets = {"off", "conservative", "moderate", "aggressive"}
        for step in schedule:
            if not isinstance(step, dict):
                raise ValueError(
                    f"Each step in augmentation_schedule must be a dictionary, got {type(step)}"
                )

            if "start_epoch" not in step or "preset" not in step:
                raise ValueError(
                    "Each augmentation_schedule step must have 'start_epoch' and 'preset' fields."
                )

            start_epoch = step["start_epoch"]
            preset = step["preset"]

            if not isinstance(start_epoch, int) or start_epoch < 0:
                raise ValueError(
                    f"augmentation_schedule.start_epoch must be a non-negative int, got {start_epoch!r}"
                )
            if not isinstance(preset, str) or preset not in valid_presets:
                raise ValueError(
                    f"augmentation_schedule.preset must be one of {sorted(valid_presets)}, got {preset!r}"
                )

    def _validate_phase_name(self) -> None:
        # Accept off or explicit phase markers
        allowed = {"off", "phase_1", "phase_2", "phase_3"}
        pn = str(getattr(self, "phase_name", "off") or "off").lower()
        if pn not in allowed:
            raise ValueError(f"phase_name must be one of {sorted(allowed)}, got {pn!r}")

    def _validate_group_loss_settings(self) -> None:
        """Validate group loss weights (strict, fail-fast)."""
        for name in (
            "caption_loss_weight",
            "grounding_loss_weight",
            "formatting_loss_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        total = float(
            self.caption_loss_weight
            + self.grounding_loss_weight
            + self.formatting_loss_weight
        )
        if total <= 0:
            raise ValueError(
                "All group weights are zero; at least one of caption/grounding/formatting must be > 0"
            )


def load_config(override_config_path: str) -> Config:
    """
    Load configuration from a base YAML and an override YAML file.

    This function implements a hierarchical configuration system. It first loads a
    base configuration (`base.yaml`) from a conventional location and then merges
    an experiment-specific override file on top of it.

    Args:
        override_config_path: Path to the experiment-specific override YAML file.

    Returns:
        Config: Validated configuration object after merging.

    Raises:
        FileNotFoundError: If either the base or override config file doesn't exist.
        yaml.YAMLError: If a YAML file is malformed.
        ValueError: If the resulting configuration is invalid.
    """
    # FAIL-FAST: Validate override config path
    if not override_config_path:
        raise ValueError("override_config_path cannot be empty")

    override_file = Path(override_config_path)
    if not override_file.exists():
        raise FileNotFoundError(
            f"Override configuration file not found: {override_config_path}"
        )
    if not override_file.is_file():
        raise ValueError(
            f"Override configuration path is not a file: {override_config_path}"
        )

    # Determine and load base config from its conventional location
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        base_config_path = project_root / "configs" / "phase_1" / "base.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(
                f"Base configuration file not found: {base_config_path}"
            )

        with open(base_config_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        raise RuntimeError(
            f"Failed to load or parse base config at {base_config_path}: {e}"
        )

    # Load override YAML
    try:
        with open(override_file, "r", encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        raise RuntimeError(
            f"Failed to load or parse override config at {override_config_path}: {e}"
        )

    # Deep merge the configurations
    data = _deep_merge_dicts(base_data, override_data)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(data)}")

    # Convert numeric-like strings to floats based on annotations
    data = _convert_scientific_notation(data)

    # === Dataset path resolution (auto-derive from data_root when missing) ===
    required_paths = ["train_data_path", "val_data_path", "teacher_pool_file"]
    missing_paths = [k for k in required_paths if k not in data or not data[k]]
    if missing_paths:
        if "data_root" not in data or not data["data_root"]:
            raise ValueError(
                "Missing required dataset paths: "
                + ", ".join(missing_paths)
                + " and data_root is not provided to derive them"
            )
        try:
            ds_paths = DataResolver.resolve_dataset_paths(data["data_root"])
        except Exception as e:
            raise ValueError(
                f"Failed to derive dataset paths from data_root={data['data_root']}: {e}"
            )
        data["train_data_path"] = str(ds_paths.train_data_path)
        data["val_data_path"] = str(ds_paths.val_data_path)
        data["teacher_pool_file"] = str(ds_paths.teacher_pool_file)

    # Normalize path-like fields; preserve relativity (no forced absolute)
    # Normalize data_root first
    if "data_root" not in data or not data["data_root"]:
        raise ValueError("data_root is required and must be non-empty")
    try:
        data["data_root"] = str(normalize_path_input(data["data_root"]))
    except Exception as e:
        raise ValueError(
            f"Failed to normalize path for 'data_root': {data.get('data_root')}; {e}"
        )

    # Normalize model_path
    if "model_path" not in data or not data["model_path"]:
        raise ValueError("model_path is required and must be non-empty")
    try:
        data["model_path"] = str(normalize_path_input(data["model_path"]))
    except Exception as e:
        raise ValueError(
            f"Failed to normalize path for 'model_path': {data.get('model_path')}; {e}"
        )

    # Normalize dataset file paths relative to dataset_base
    for key in ("train_data_path", "val_data_path", "teacher_pool_file"):
        try:
            data[key] = str(normalize_path_input(data[key]))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for '{key}': {data.get(key)}; {e}"
            )

    # Normalize output/log directories relative to config_dir
    for key in ("output_dir", "tb_dir"):
        if key not in data or not data[key]:
            raise ValueError(f"Missing required output/log path: {key}")
        try:
            data[key] = str(normalize_path_input(data[key]))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for '{key}': {data.get(key)}; {e}"
            )

    # Gate augmentation by use_aug flag (no external files)
    if "use_aug" not in data:
        raise ValueError("use_aug must be explicitly set to true or false in the YAML")
    if not isinstance(data["use_aug"], bool):
        raise ValueError("use_aug must be a boolean (true/false)")
    use_aug_flag = data["use_aug"]
    if not use_aug_flag:
        # Explicitly ignore any inline augmentation block when disabled
        if "augmentation" in data:
            data.pop("augmentation", None)
    else:
        # Require either an inline augmentation block or an augmentation_schedule
        has_aug_block = "augmentation" in data and data["augmentation"] is not None
        has_schedule = (
            "augmentation_schedule" in data
            and isinstance(data["augmentation_schedule"], list)
            and len(data["augmentation_schedule"]) > 0
        )
        if not (has_aug_block or has_schedule):
            raise ValueError(
                "use_aug=True but neither 'augmentation' block nor 'augmentation_schedule' is provided in the YAML. Provide a preset-based block or a schedule."
            )

    # Build augmentation dataclasses from nested dicts (if provided)
    try:
        if "augmentation" in data and data["augmentation"] is not None:
            aug_dict = data["augmentation"]
            if not isinstance(aug_dict, dict):
                raise ValueError(
                    f"augmentation must be a mapping when provided, got {type(aug_dict)}"
                )

            # New: allow preset-based shorthand to drastically reduce hyperparameters
            if "preset" in aug_dict:
                from src_new.augmentation.presets import (
                    PresetOptions as _PresetOptions,
                )
                from src_new.augmentation.presets import (
                    build_augmentation_config_from_preset as _build_from_preset,
                )

                preset_value = (
                    str(aug_dict["preset"]) if aug_dict["preset"] is not None else None
                )
                if preset_value is None:
                    raise ValueError(
                        "augmentation.preset cannot be null; choose off|conservative|moderate|aggressive"
                    )
                # Minimal options allowed alongside preset
                rng_seed = (
                    int(aug_dict["rng_seed"]) if "rng_seed" in aug_dict else 12345
                )
                apply_to_teachers = (
                    bool(aug_dict["apply_to_teachers"])
                    if "apply_to_teachers" in aug_dict
                    else False
                )
                lines_policy = (
                    str(aug_dict["lines_policy"])
                    if "lines_policy" in aug_dict
                    else "transform"
                )
                debug_visualization = (
                    bool(aug_dict["debug_visualization"])
                    if "debug_visualization" in aug_dict
                    else False
                )
                debug_output_dir = (
                    aug_dict["debug_output_dir"]
                    if "debug_output_dir" in aug_dict
                    else None
                )

                opts = _PresetOptions(
                    preset=preset_value,  # type: ignore[arg-type]
                    rng_seed=rng_seed,
                    apply_to_teachers=apply_to_teachers,
                    lines_policy=lines_policy,  # validated downstream
                    debug_visualization=debug_visualization,
                    debug_output_dir=debug_output_dir,
                )
                aug_cfg = _build_from_preset(opts)
                validate_augmentation_config_fn(aug_cfg)
                data["augmentation"] = aug_cfg
            else:
                # Explicit object-aware config path (no legacy 'op' support)
                from src_new.config.augmentation_config import (
                    AugmentationConfig as _Aug,
                )
                from src_new.config.augmentation_config import (
                    CriteriaConfig as _CR,
                )
                from src_new.config.augmentation_config import (
                    ImageGeomConfig as _IG,
                )
                from src_new.config.augmentation_config import (
                    LineAugConfig as _LN,
                )
                from src_new.config.augmentation_config import (
                    OcclusionCriterionConfig as _OC,
                )
                from src_new.config.augmentation_config import (
                    PhotometricConfig as _PH,
                )
                from src_new.config.augmentation_config import (
                    TypePolicyConfig as _TP,
                )

                ig = None
                if "image_geom" in aug_dict and aug_dict["image_geom"] is not None:
                    ig_dict = aug_dict["image_geom"]
                    if not isinstance(ig_dict, dict):
                        raise ValueError(
                            "augmentation.image_geom must be a mapping when provided"
                        )
                    ig = _IG(
                        rotate_deg_range=tuple(ig_dict["rotate_deg_range"]),
                        translate_pct=float(ig_dict["translate_pct"]),
                        scale_range=tuple(ig_dict["scale_range"]),
                        perspective_pct=float(ig_dict["perspective_pct"]),
                        crop_pct=float(ig_dict["crop_pct"]),
                        multiscale_short_edges=ig_dict.get("multiscale_short_edges"),
                    )

                ph = None
                if "photometric" in aug_dict and aug_dict["photometric"] is not None:
                    ph_dict = aug_dict["photometric"]
                    if not isinstance(ph_dict, dict):
                        raise ValueError(
                            "augmentation.photometric must be a mapping when provided"
                        )
                    ph = _PH(
                        enabled=bool(ph_dict["enabled"]),
                        apply_prob=float(ph_dict["apply_prob"]),
                        num_ops=int(ph_dict["num_ops"]),
                        magnitude=float(ph_dict["magnitude"]),
                        ocr_safe_pool=bool(ph_dict["ocr_safe_pool"]),
                    )

                ln = None
                if "lines" in aug_dict and aug_dict["lines"] is not None:
                    ln_dict = aug_dict["lines"]
                    if not isinstance(ln_dict, dict):
                        raise ValueError(
                            "augmentation.lines must be a mapping when provided"
                        )
                    ln = _LN(
                        enabled=bool(ln_dict["enabled"]),
                        jitter_px_minmax=tuple(ln_dict["jitter_px_minmax"]),
                        resample_points=int(ln_dict["resample_points"]),
                        min_length_px=int(ln_dict["min_length_px"]),
                    )

                # Optional type_policies
                tp = None
                if (
                    "type_policies" in aug_dict
                    and aug_dict["type_policies"] is not None
                ):
                    tp = {}
                    if not isinstance(aug_dict["type_policies"], dict):
                        raise ValueError(
                            "augmentation.type_policies must be a mapping when provided"
                        )
                    for k, v in aug_dict["type_policies"].items():
                        if not isinstance(v, dict):
                            raise ValueError(f"type_policies.{k} must be a mapping")
                        tp[k] = _TP(
                            allow_move=bool(v["allow_move"]),
                            allow_copy_paste=bool(v["allow_copy_paste"]),
                            allow_blur=bool(v["allow_blur"]),
                            occluder_prob=float(v["occluder_prob"]),
                            inpaint_source=bool(v["inpaint_source"]),
                            max_iou_with_existing=v.get("max_iou_with_existing"),
                            max_occ_fraction=v.get("max_occ_fraction"),
                            occ_grid_downscale=v.get("occ_grid_downscale"),
                            occ_margin_px=v.get("occ_margin_px"),
                            same_plane_constraint=bool(
                                v.get("same_plane_constraint", True)
                            ),
                            copy_paste_attempts=v.get("copy_paste_attempts"),
                            alpha_feather_px=v.get("alpha_feather_px"),
                            allowed_copy_types=v.get("allowed_copy_types"),
                        )

                # Optional criteria
                cr = None
                if "criteria" in aug_dict and aug_dict["criteria"] is not None:
                    cr_dict = aug_dict["criteria"]
                    if not isinstance(cr_dict, dict):
                        raise ValueError(
                            "augmentation.criteria must be a mapping when provided"
                        )
                    occ_cfg = None
                    if "occlusion" in cr_dict and cr_dict["occlusion"] is not None:
                        oc = cr_dict["occlusion"]
                        if not isinstance(oc, dict):
                            raise ValueError(
                                "augmentation.criteria.occlusion must be a mapping"
                            )
                        occ_cfg = _OC(
                            enabled=bool(oc["enabled"]),
                            min_overlap_fraction_bbox=float(
                                oc["min_overlap_fraction_bbox"]
                            ),
                            min_overlap_fraction_line=float(
                                oc["min_overlap_fraction_line"]
                            ),
                            mask_downscale=int(oc["mask_downscale"]),
                            line_width_px=int(oc["line_width_px"]),
                        )
                    cr = _CR(occlusion=occ_cfg)

                aug_cfg = _Aug(
                    enabled=bool(aug_dict["enabled"]),
                    rng_seed=int(aug_dict["rng_seed"]),
                    apply_to_teachers=bool(aug_dict["apply_to_teachers"]),
                    lines_policy=str(aug_dict["lines_policy"]),
                    debug_visualization=bool(aug_dict["debug_visualization"]),
                    debug_output_dir=aug_dict.get("debug_output_dir"),
                    criteria=cr,
                    image_geom=ig,
                    photometric=ph,
                    lines=ln,
                    type_policies=tp,
                    ocr=None,
                )
                validate_augmentation_config_fn(aug_cfg)
                data["augmentation"] = aug_cfg

        # Optional: teacher_augmentation (photometric-only; simple mapping)
        if "teacher_augmentation" in data and data["teacher_augmentation"] is not None:
            taug_dict = data["teacher_augmentation"]
            if not isinstance(taug_dict, dict):
                raise ValueError(
                    f"teacher_augmentation must be a mapping when provided, got {type(taug_dict)}"
                )
            from src_new.config.augmentation_config import (
                AugmentationConfig as _TAug,
            )
            from src_new.config.augmentation_config import (
                PhotometricConfig as _TPH,
            )

            # rng_seed optional; default 12345
            t_rng_seed = int(taug_dict.get("rng_seed", 12345))
            # photometric block required
            if "photometric" not in taug_dict or taug_dict["photometric"] is None:
                raise ValueError(
                    "teacher_augmentation.photometric must be provided (photometric-only policy)"
                )
            tph_dict = taug_dict["photometric"]
            if not isinstance(tph_dict, dict):
                raise ValueError(
                    "teacher_augmentation.photometric must be a mapping when provided"
                )
            tph = _TPH(
                enabled=bool(tph_dict["enabled"]),
                apply_prob=float(tph_dict["apply_prob"]),
                num_ops=int(tph_dict["num_ops"]),
                magnitude=float(tph_dict["magnitude"]),
                ocr_safe_pool=bool(tph_dict["ocr_safe_pool"]),
            )
            taug_cfg = _TAug(
                enabled=True,
                rng_seed=t_rng_seed,
                apply_to_teachers=True,  # explicit teacher pipeline
                lines_policy="transform",
                debug_visualization=False,
                debug_output_dir=None,
                criteria=None,
                image_geom=None,  # enforce photometric-only
                photometric=tph,
                lines=None,
                type_policies=None,
                ocr=None,
            )
            validate_augmentation_config_fn(taug_cfg)
            data["teacher_augmentation"] = taug_cfg
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
        raise ValueError(
            f"Failed to create configuration from {override_config_path}: {e}"
        )

    logger.info(f"✅ Configuration loaded successfully from {override_config_path}")
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
