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

from src_new_json.config.augmentation_config import (
    AugmentationConfig as AugmentationConfigType,
)
from src_new_json.config.augmentation_config import (
    validate_augmentation_config as validate_augmentation_config_fn,
)
from src_new_json.utils.data_resolver import DataResolver
from src_new_json.utils.validation import (
    PathValidationError,
    PathValidator,
    normalize_path_input,
)


logger = logging.getLogger(__name__)

# Global logging configuration (compatibility with legacy callers)
# These values mirror the state managed by src_new_json.utils.rank_aware_logging.
_GLOBAL_LOG_LEVEL = logging.INFO
_CONFIGURED_LOGGERS = set()

# Prefer rank-aware logger for this module
try:
    from ..utils.rank_aware_logging import get_rank_aware_logger as _get_logger
    logger = _get_logger("config")
except Exception:
    pass


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


def _normalize_smart_resize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    required = {"enabled", "factor", "min_pixels", "max_pixels", "max_ratio"}
    missing = required.difference(mapping.keys())
    if missing:
        raise ValueError(
            f"smart_resize mapping missing required keys: {sorted(missing)}"
        )
    normalized = {
        "enabled": bool(mapping["enabled"]),
        "factor": int(mapping["factor"]),
        "min_pixels": int(mapping["min_pixels"]),
        "max_pixels": int(mapping["max_pixels"]),
        "max_ratio": float(mapping["max_ratio"]),
    }
    if normalized["factor"] <= 0:
        raise ValueError("smart_resize.factor must be positive")
    if normalized["min_pixels"] <= 0 or normalized["max_pixels"] <= 0:
        raise ValueError("smart_resize min_pixels/max_pixels must be positive")
    if normalized["min_pixels"] > normalized["max_pixels"]:
        raise ValueError("smart_resize.min_pixels must be <= max_pixels")
    if normalized["max_ratio"] <= 0:
        raise ValueError("smart_resize.max_ratio must be positive")
    return normalized


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
    # Group loss weights (compatibility)
    caption_loss_weight: float
    grounding_loss_weight: float
    formatting_loss_weight: float

    # Features (coordinate tokens removed in JSON mode)

    # Vision processing parameters
    merge_size: int
    max_pixels: int  # Qwen2VL image processor's max_pixels

    # Output settings
    output_dir: str
    run_name: str  # tensorboard event name
    tb_dir: str

    # Coordinate/token limits (removed in JSON mode)

    # Loss settings
    teacher_ratio: float

    # Collator
    collator_type: str

    # HF Trainer: evaluation/checkpoint settings (compatibility)
    eval_strategy: str
    eval_steps: int
    save_strategy: str

    # Checkpointing
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool

    # Logging settings (compatibility)
    logging_steps: int
    report_to: str
    disable_tqdm: bool

    # Dataloader performance (compatibility)
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    remove_unused_columns: bool

    # Optimizer/learning rate groups (optional overrides)
    lr_merger: Optional[float] = None
    lr_top_layers: Optional[float] = None
    lr_full_model: Optional[float] = None

    # Optional: conversation variant ratios for training sampling
    # keys in {"dense_caption", "coords_to_desc", "desc_to_coords"}, values are non-negative weights
    conversation_variant_ratios: Optional[Dict[str, float]] = None

    # === OPTIONAL FIELDS WITH DEFAULTS (compatibility) ===
    seed: int = 17

    # Augmentation controls
    augmentation: Optional[AugmentationConfigType] = None
    teacher_augmentation: Optional[AugmentationConfigType] = None
    use_aug: bool = False
    augmentation_schedule: Optional[List[Dict[str, Any]]] = None
    augmentation_smart_resize_defaults: Optional[Dict[str, Any]] = None
    phase_name: str = "off"

    # Optional phase-freeze overrides (phase_3 selective unfreeze; unified keys)
    llm_top_k_block: Optional[int] = None
    vision_top_k_block: Optional[int] = None
    freeze_patch_embed: Optional[bool] = None
    trainable_token_strings: Optional[List[str]] = None

    # Dynamic contrastive pairing (JSON mode—parity with src_new)
    dynamic_pairing_enabled: bool = True
    dynamic_pair_target_assignment: str = "current"  # {random,current,opposite}
    dynamic_pair_temperature: float = 1.2
    dynamic_pair_cross_bucket_explore_prob: float = 0.0
    # Large-pool controls for sampling (explicit; used by BucketedSamplingEngine)
    pool_fraction: float = 0.2
    pool_max: int = 1024

    # Overlap-pool and hardness controls (JSON dynamic pairing)
    hardness_alpha: float = 0.1
    hardness_warmup_epochs: int = 1


    # Span extraction options
    span_include_im_end_in_labels: bool = True
    # Debug: enable strict alignment assertions (decode↔re-tokenize, span/mask invariants)
    debug_alignment: bool = False

    # Best-checkpoint interval control (optional)
    # If best_checkpoint_min_interval_steps is provided, it takes precedence.
    # Otherwise the interval is computed as eval_steps * best_checkpoint_interval_multiplier.
    best_checkpoint_min_interval_steps: Optional[int] = None
    best_checkpoint_interval_multiplier: int = 10
    
    # Removed: packed segment isolation no longer supported


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
        self._validate_augmentation_settings()
        self._validate_phase_name()
        self._validate_learning_rate_groups()
        self._validate_group_loss_settings()
        self._validate_span_settings()

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
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    def _validate_data_settings(self) -> None:
        """Validate data-related settings."""
        if not self.train_data_path:
            raise ValueError("train_data_path cannot be empty")

        if not self.val_data_path:
            raise ValueError("val_data_path cannot be empty")

        # Teacher pool file validation based on dynamic pairing mode
        if not self.teacher_pool_file:
            if self.dynamic_pairing_enabled:
                logger.info("teacher_pool_file not provided; using dynamic pairing from train set")
            else:
                logger.warning("teacher_pool_file not provided and dynamic_pairing_enabled=False; teacher-student training may be limited")
        elif self.dynamic_pairing_enabled:
            logger.info("teacher_pool_file provided but dynamic_pairing_enabled=True; dynamic pairing will be used instead")

        # Validate existence using centralized validator (accept relative or aliases)
        try:
            PathValidator.validate_file_exists(self.train_data_path)
            PathValidator.validate_file_exists(self.val_data_path)
            # Only validate teacher_pool_file if provided and dynamic pairing is disabled
            if self.teacher_pool_file and not self.dynamic_pairing_enabled:
                PathValidator.validate_file_exists(self.teacher_pool_file)
            elif self.teacher_pool_file and self.dynamic_pairing_enabled:
                # If file exists, validate it, but don't require it
                try:
                    PathValidator.validate_file_exists(self.teacher_pool_file)
                except (ValueError, PathValidationError):
                    logger.info("teacher_pool_file specified but not found; will use dynamic pairing instead")
                    # Clear the teacher_pool_file since it's not needed for dynamic pairing
                    object.__setattr__(self, 'teacher_pool_file', "")
            PathValidator.validate_directory_exists(self.data_root)
        except (ValueError, PathValidationError) as e:
            raise ValueError(f"Invalid data paths: {e}")

        if self.teacher_ratio < 0 or self.teacher_ratio > 1:
            raise ValueError(
                f"teacher_ratio must be between 0 and 1, got {self.teacher_ratio}"
            )


        if self.collator_type not in ["standard"]:
            raise ValueError(f"Invalid collator_type: {self.collator_type}")

        # Dynamic pairing strict validation (JSON mode)
        if not isinstance(self.dynamic_pairing_enabled, bool):
            raise ValueError("dynamic_pairing_enabled must be a boolean")
        if self.dynamic_pair_target_assignment not in {"random", "current", "opposite"}:
            raise ValueError(
                "dynamic_pair_target_assignment must be one of {'random','current','opposite'}"
            )
        if self.dynamic_pair_temperature <= 0:
            raise ValueError(
                f"dynamic_pair_temperature must be > 0, got {self.dynamic_pair_temperature}"
            )
        if not (0.0 <= float(self.dynamic_pair_cross_bucket_explore_prob) <= 1.0):
            raise ValueError(
                f"dynamic_pair_cross_bucket_explore_prob must be in [0,1], got {self.dynamic_pair_cross_bucket_explore_prob}"
            )

        # Sampling large-pool controls
        if not (0.0 <= float(self.pool_fraction) <= 1.0):
            raise ValueError(
                f"pool_fraction must be in [0,1], got {self.pool_fraction}"
            )
        if not isinstance(self.pool_max, int) or self.pool_max < 1:
            raise ValueError(
                f"pool_max must be a positive int, got pool_max={self.pool_max}"
            )

        # Overlap-pool and hardness validation (strict)
        if not (0.0 < float(self.hardness_alpha) <= 1.0):
            raise ValueError(f"hardness_alpha must be in (0,1], got {self.hardness_alpha}")
        if not (isinstance(self.hardness_warmup_epochs, int) and self.hardness_warmup_epochs >= 0):
            raise ValueError(
                f"hardness_warmup_epochs must be a non-negative int, got {self.hardness_warmup_epochs}"
            )

        # Conversation variant ratios (optional)
        sampling = self.conversation_variant_ratios
        if sampling is not None:
            if not isinstance(sampling, dict):
                raise ValueError("conversation_variant_ratios must be a dict if provided")
            # Validate keys strictly against canonical set
            allowed = {"dense_caption", "coords_to_desc", "desc_to_coords", "summary"}
            total = 0.0
            for k, v in sampling.items():
                if k not in allowed:
                    raise ValueError(
                        f"conversation_variant_ratios contains unsupported key: {k}. Allowed: {sorted(allowed)}"
                    )
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(
                        f"conversation_variant_ratios[{k}] must be a non-negative number"
                    )
                total += float(v)
            if total <= 0:
                raise ValueError("Sum of conversation_variant_ratios must be > 0")

        # Output/log paths: accept relative; no existence check required here

    def _validate_coordinate_settings(self) -> None:
        """Coordinate-token validation removed in JSON mode."""
        return

    def _validate_learning_rate_groups(self) -> None:
        # Learning rates (if provided) must be positive
        for lr_name in (
            "lr_merger",
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
        aug = self.augmentation
        if aug is None:
            # Still validate schedule if present
            sched = self.augmentation_schedule
            if sched is not None:
                self._validate_augmentation_schedule(sched)
            return
        try:
            validate_augmentation_config_fn(aug)
        except Exception as e:
            raise ValueError(f"Invalid augmentation configuration: {e}")

        # Validate schedule alongside an explicit augmentation config if provided
        sched = self.augmentation_schedule
        if sched is not None:
            self._validate_augmentation_schedule(sched)

        # Optional teacher-specific augmentation
        taug = self.teacher_augmentation
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
        default_sr = getattr(self, "augmentation_smart_resize_defaults", None)
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

            sr_dict = step.get("smart_resize")
            if sr_dict is not None:
                if not isinstance(sr_dict, dict):
                    raise ValueError(
                        f"augmentation_schedule.smart_resize must be a mapping when provided (preset '{preset}')"
                    )
                step["smart_resize"] = _normalize_smart_resize_mapping(sr_dict)
            elif preset != "off":
                if default_sr is None:
                    raise ValueError(
                        f"augmentation_schedule entry for preset '{preset}' is missing smart_resize and no augmentation.smart_resize defaults are defined."
                    )

            sr_dict = step.get("smart_resize")
            if preset != "off":
                if sr_dict is None:
                    raise ValueError(
                        f"augmentation_schedule entry for preset '{preset}' must include a smart_resize block"
                    )
            if sr_dict is not None:
                if not isinstance(sr_dict, dict):
                    raise ValueError(
                        f"augmentation_schedule.smart_resize must be a mapping when provided (preset '{preset}')"
                    )
                required_sr = {
                    "enabled",
                    "factor",
                    "min_pixels",
                    "max_pixels",
                    "max_ratio",
                }
                missing_sr = required_sr.difference(sr_dict.keys())
                if missing_sr:
                    raise ValueError(
                        f"augmentation_schedule.smart_resize for preset '{preset}' missing keys: {sorted(missing_sr)}"
                    )

    def _validate_phase_name(self) -> None:
        # Accept off or explicit phase markers
        allowed = {"off", "phase_1", "phase_2", "phase_3"}
        pn = str(self.phase_name or "off").lower()
        if pn not in allowed:
            raise ValueError(f"phase_name must be one of {sorted(allowed)}, got {pn!r}")
        # Light validation for selective unfreeze overrides (allow -1 to mean "unfreeze all")
        for k in ("llm_top_k_block", "vision_top_k_block"):
            v = getattr(self, k, None)
            if v is not None and (not isinstance(v, int) or v < -1):
                raise ValueError(f"{k} must be an int >= -1 when provided, got {v!r}")
        fpe = self.freeze_patch_embed
        if fpe is not None and not isinstance(fpe, bool):
            raise ValueError("freeze_patch_embed must be a boolean when provided")
        tts = self.trainable_token_strings
        if tts is not None:
            if not isinstance(tts, list) or not all(isinstance(x, str) for x in tts):
                raise ValueError("trainable_token_strings must be a list of strings")

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

    def _validate_span_settings(self) -> None:
        """Validate span extraction options."""
        if not isinstance(self.span_include_im_end_in_labels, bool):
            raise ValueError(
                "span_include_im_end_in_labels must be a boolean (true/false)"
            )
        if not isinstance(self.debug_alignment, bool):
            raise ValueError("debug_alignment must be a boolean (true/false)")
        # Validate best-checkpoint interval settings
        bmin = getattr(self, "best_checkpoint_min_interval_steps", None)
        if bmin is not None and (not isinstance(bmin, int) or bmin < 0):
            raise ValueError(
                f"best_checkpoint_min_interval_steps must be a non-negative int when provided, got {bmin!r}"
            )
        mult = getattr(self, "best_checkpoint_interval_multiplier", 10)
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(
                f"best_checkpoint_interval_multiplier must be a positive int, got {mult!r}"
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

    # NOTE: Do not require fields in override alone; validate after merge so base or override can satisfy
    # required settings like 'max_pixels'. This ensures base defaults work and experiments can override.

    # Deep merge the configurations
    data = _deep_merge_dicts(base_data, override_data)

    # Validate required fields in the MERGED configuration (fail-fast)
    if (
        ("max_pixels" not in data)
        or (data["max_pixels"] is None)
        or (not isinstance(data["max_pixels"], (int, float)))
        or int(data["max_pixels"]) <= 0
    ):
        raise ValueError(
            "Configuration must include a positive 'max_pixels' value in either base or experiment YAML."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(data)}")

    # Convert numeric-like strings to floats based on annotations
    data = _convert_scientific_notation(data)

    if "augmentation_smart_resize_defaults" not in data:
        data["augmentation_smart_resize_defaults"] = None

    # Remove backward-compatible key normalization; enforce canonical keys upstream
    if "conversation_variant_ratios" in data and isinstance(data["conversation_variant_ratios"], dict):
        pass

    # === Dataset path resolution (auto-derive from data_root when missing) ===
    # Check if dynamic pairing is enabled to determine if teacher_pool_file is required
    if "dynamic_pairing_enabled" not in data:
        data["dynamic_pairing_enabled"] = False # Default to False if not set
    dynamic_pairing_enabled = bool(data["dynamic_pairing_enabled"])
    
    # Base required paths (always needed)
    required_paths = ["train_data_path", "val_data_path"]
    # Only require teacher_pool_file if dynamic pairing is disabled
    if not dynamic_pairing_enabled:
        required_paths.append("teacher_pool_file")
    
    missing_paths = [k for k in required_paths if k not in data or not data[k]]
    if missing_paths:
        if "data_root" not in data or not data["data_root"]:
            raise ValueError(
                "Missing required dataset paths: "
                + ", ".join(missing_paths)
                + " and data_root is not provided to derive them"
            )
        try:
            # Don't require teacher pool when dynamic pairing is enabled
            ds_paths = DataResolver.resolve_dataset_paths(
                data["data_root"], 
                require_teacher_pool=not dynamic_pairing_enabled
            )
        except Exception as e:
            raise ValueError(
                f"Failed to derive dataset paths from data_root={data['data_root']}: {e}"
            )
        data["train_data_path"] = str(ds_paths.train_data_path)
        data["val_data_path"] = str(ds_paths.val_data_path)
        # Set teacher_pool_file regardless (it will be ignored if dynamic pairing is enabled)
        data["teacher_pool_file"] = str(ds_paths.teacher_pool_file)

    # Normalize path-like fields; preserve relativity (no forced absolute)
    # Normalize data_root first
    if "data_root" not in data or not data["data_root"]:
        raise ValueError("data_root is required and must be non-empty")
    try:
        data_root_val = data["data_root"]
        data["data_root"] = str(normalize_path_input(data_root_val))
    except Exception as e:
        raise ValueError(
            f"Failed to normalize path for 'data_root': {data_root_val}; {e}"
        )

    # Normalize model_path
    if "model_path" not in data or not data["model_path"]:
        raise ValueError("model_path is required and must be non-empty")
    try:
        model_path_val = data["model_path"]
        data["model_path"] = str(normalize_path_input(model_path_val))
    except Exception as e:
        raise ValueError(
            f"Failed to normalize path for 'model_path': {model_path_val}; {e}"
        )

    # Normalize dataset file paths relative to dataset_base
    for key in ("train_data_path", "val_data_path", "teacher_pool_file"):
        try:
            val = data[key]
            data[key] = str(normalize_path_input(val))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for '{key}': {val}; {e}"
            )

    # Normalize output/log directories relative to config_dir
    for key in ("output_dir", "tb_dir"):
        # default tb_dir to output_dir if not provided
        if key not in data or not data[key]:
            if key == "tb_dir" and "output_dir" in data and data["output_dir"]:
                data["tb_dir"] = data["output_dir"]
            else:
                raise ValueError(f"Missing required output/log path: {key}")
        try:
            val = data[key]
            data[key] = str(normalize_path_input(val))
        except Exception as e:
            raise ValueError(
                f"Failed to normalize path for '{key}': {val}; {e}"
            )

    # Gate augmentation by use_aug flag (no external files)
    if "use_aug" not in data:
        data["use_aug"] = False
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

            smart_resize_defaults = None
            sr_dict = aug_dict.get("smart_resize")
            if sr_dict is not None:
                if not isinstance(sr_dict, dict):
                    raise ValueError(
                        "augmentation.smart_resize must be a mapping when provided"
                    )
                smart_resize_defaults = _normalize_smart_resize_mapping(sr_dict)

            # New: allow preset-based shorthand to drastically reduce hyperparameters
            if "preset" in aug_dict:
                from src_new_json.augmentation.presets import (
                    PresetOptions as _PresetOptions,
                )
                from src_new_json.augmentation.presets import (
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
                if "rng_seed" not in aug_dict:
                    raise ValueError("augmentation.rng_seed must be explicitly provided when using 'preset'")
                rng_seed = int(aug_dict["rng_seed"]) 
                if "apply_to_teachers" not in aug_dict:
                    raise ValueError("augmentation.apply_to_teachers must be explicitly provided (true/false)")
                apply_to_teachers = bool(aug_dict["apply_to_teachers"]) 
                # Default to identity (do not move lines at object-level) if omitted
                lines_policy = str(aug_dict.get("lines_policy", "identity")) 
                debug_visualization = bool(aug_dict.get("debug_visualization", False)) 
                debug_output_dir = (
                    aug_dict["debug_output_dir"]
                    if "debug_output_dir" in aug_dict
                    else None
                )
                if preset_value != "off" and smart_resize_defaults is None:
                    raise ValueError(
                        f"augmentation.smart_resize must be provided when using preset '{preset_value}'."
                    )

                sr_enabled = (
                    smart_resize_defaults["enabled"]
                    if smart_resize_defaults is not None
                    else None
                )
                sr_factor = (
                    smart_resize_defaults["factor"]
                    if smart_resize_defaults is not None
                    else None
                )
                sr_min = (
                    smart_resize_defaults["min_pixels"]
                    if smart_resize_defaults is not None
                    else None
                )
                sr_max = (
                    smart_resize_defaults["max_pixels"]
                    if smart_resize_defaults is not None
                    else None
                )
                sr_ratio = (
                    smart_resize_defaults["max_ratio"]
                    if smart_resize_defaults is not None
                    else None
                )

                opts = _PresetOptions(
                    preset=preset_value,  # type: ignore[arg-type]
                    rng_seed=rng_seed,
                    apply_to_teachers=apply_to_teachers,
                    lines_policy=lines_policy,  # validated downstream
                    debug_visualization=debug_visualization,
                    debug_output_dir=debug_output_dir,
                    smart_resize_enabled=sr_enabled,
                    smart_resize_factor=sr_factor,
                    smart_resize_min_pixels=sr_min,
                    smart_resize_max_pixels=sr_max,
                    smart_resize_max_ratio=sr_ratio,
                )
                aug_cfg = _build_from_preset(opts)
                validate_augmentation_config_fn(aug_cfg)
                data["augmentation"] = aug_cfg
                data["augmentation_smart_resize_defaults"] = smart_resize_defaults
            else:
                # Explicit object-aware config path (no legacy 'op' support)
                from src_new_json.config.augmentation_config import (
                    AugmentationConfig as _Aug,
                )
                from src_new_json.config.augmentation_config import (
                    CriteriaConfig as _CR,
                )
                from src_new_json.config.augmentation_config import (
                    ImageGeomConfig as _IG,
                )
                from src_new_json.config.augmentation_config import (
                    LineAugConfig as _LN,
                )
                from src_new_json.config.augmentation_config import (
                    OcclusionCriterionConfig as _OC,
                )
                from src_new_json.config.augmentation_config import (
                    PhotometricConfig as _PH,
                )
                from src_new_json.config.augmentation_config import (
                    SmartResizeConfig as _SR,
                )
                from src_new_json.config.augmentation_config import (
                    TypePolicyConfig as _TP,
                )

                sr = None
                if smart_resize_defaults is not None:
                    sr = _SR(
                        enabled=smart_resize_defaults["enabled"],
                        factor=smart_resize_defaults["factor"],
                        min_pixels=smart_resize_defaults["min_pixels"],
                        max_pixels=smart_resize_defaults["max_pixels"],
                        max_ratio=smart_resize_defaults["max_ratio"],
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
                                v["same_plane_constraint"]
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
                    lines_policy=str(aug_dict.get("lines_policy", "identity")),
                    debug_visualization=bool(aug_dict.get("debug_visualization", False)),
                    debug_output_dir=aug_dict.get("debug_output_dir"),
                    criteria=cr,
                    smart_resize=sr,
                    image_geom=ig,
                    photometric=ph,
                    lines=ln,
                    type_policies=tp,
                    ocr=None,
                )
                validate_augmentation_config_fn(aug_cfg)
                data["augmentation"] = aug_cfg
                if smart_resize_defaults is not None:
                    data["augmentation_smart_resize_defaults"] = smart_resize_defaults
                elif aug_cfg.smart_resize is not None:
                    data["augmentation_smart_resize_defaults"] = {
                        "enabled": aug_cfg.smart_resize.enabled,
                        "factor": aug_cfg.smart_resize.factor,
                        "min_pixels": aug_cfg.smart_resize.min_pixels,
                        "max_pixels": aug_cfg.smart_resize.max_pixels,
                        "max_ratio": aug_cfg.smart_resize.max_ratio,
                    }
                else:
                    data["augmentation_smart_resize_defaults"] = None

        # Optional: teacher_augmentation (photometric-only; simple mapping)
        if "teacher_augmentation" in data and data["teacher_augmentation"] is not None:
            taug_dict = data["teacher_augmentation"]
            if not isinstance(taug_dict, dict):
                raise ValueError(
                    f"teacher_augmentation must be a mapping when provided, got {type(taug_dict)}"
                )
            from src_new_json.config.augmentation_config import (
                AugmentationConfig as _TAug,
            )
            from src_new_json.config.augmentation_config import (
                PhotometricConfig as _TPH,
            )

            # rng_seed must be explicit
            if "rng_seed" not in taug_dict:
                raise ValueError("teacher_augmentation.rng_seed must be explicitly provided")
            t_rng_seed = int(taug_dict["rng_seed"])
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

    # Drop legacy fields from older configs that are irrelevant in JSON mode
    LEGACY_IGNORED_FIELDS = {
        "coordinate_init_mode",
        "coordinate_loss_weight",
        "coordinate_tokens_enabled",
        "max_coord_value",
        "packed_segment_isolation",
    }
    ignored_present = sorted([k for k in list(data.keys()) if k in LEGACY_IGNORED_FIELDS])
    for k in ignored_present:
        data.pop(k, None)
    if ignored_present:
        logger.info("Ignoring legacy config fields (JSON mode): " + ", ".join(ignored_present))

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
