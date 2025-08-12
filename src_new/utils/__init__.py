"""Utility modules for the Qwen2.5-VL project.

This package contains utility classes and functions for:
- Path management and resolution
- Debug logging and monitoring
- Performance monitoring
- Checkpoint validation
- Rank-aware logging for distributed training
- Centralized validation utilities
- Error message formatting
- Tensor validation
- Common imports and utilities
"""

# Legacy imports (maintained for backward compatibility)
from .checkpoint_validator import CheckpointValidator, validate_checkpoint
from .common_imports import (
    TYPE_CHECKING,
    Any,
    ConfigDict,
    Dict,
    F,
    Image,
    List,
    Optional,
    Path,
    # Type aliases
    PathLike,
    SampleDict,
    Tensor,
    TensorLike,
    Tuple,
    Union,
    check_numpy_available,
    check_pil_available,
    # Availability checks
    check_torch_available,
    check_transformers_available,
    check_yaml_available,
    dataclass,
    ensure_list,
    # Utility functions
    ensure_path,
    get_available_device,
    # Standard library
    json,
    logging,
    nn,
    np,
    os,
    require_numpy,
    require_pil,
    require_torch,
    require_transformers,
    require_yaml,
    safe_import,
    setup_logging,
    sys,
    # Third-party (conditional)
    torch,
    warnings,
    yaml,
)
from .debug_logging import DebugLogger, debug_logger
from .error_formatting import (
    ErrorMessageBuilder,
    format_file_not_found,
    format_tensor_error,
    format_validation_error,
)

# New centralized utilities
from .logger_factory import (
    get_debug_logger,
    get_inference_logger,
    get_module_logger,
    get_processing_logger,
    get_training_logger,
    reconfigure_logger,
)
from .logger_factory import (
    set_global_log_level as set_factory_log_level,
)
from .path_manager import (
    PathManager,
    create_path_manager,
    resolve_image_paths,
    safe_resolve_image_paths,
)
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .rank_aware_logging import (
    all_ranks,
    configure_rank_aware_logging,
    get_logger,
    get_rank_aware_logger,
    rank0_only,
    set_global_log_level,
)
from .tensor_validation import (
    TensorValidationError,
    TensorValidator,
    validate_batch_size,
    validate_multimodal_tensors,
    validate_tensor_shape,
)
from .validation import (
    DirectoryValidationError,
    PathValidationError,
    PathValidator,
    ValidationError,
    validate_dataset_structure,
    validate_directory,
    validate_file,
)


__all__ = [
    # Path management
    "PathManager",
    "create_path_manager",
    "resolve_image_paths",
    "safe_resolve_image_paths",
    # Legacy logging (backward compatibility)
    "get_logger",
    "get_rank_aware_logger",
    "configure_rank_aware_logging",
    "set_global_log_level",
    "rank0_only",
    "all_ranks",
    "DebugLogger",
    "debug_logger",
    # New centralized logging
    "get_module_logger",
    "get_debug_logger",
    "get_training_logger",
    "get_inference_logger",
    "get_processing_logger",
    "reconfigure_logger",
    "set_factory_log_level",
    # Validation utilities
    "ValidationError",
    "PathValidationError",
    "DirectoryValidationError",
    "PathValidator",
    "validate_file",
    "validate_directory",
    "validate_dataset_structure",
    # Error formatting
    "ErrorMessageBuilder",
    "format_file_not_found",
    "format_validation_error",
    "format_tensor_error",
    # Tensor validation
    "TensorValidationError",
    "TensorValidator",
    "validate_tensor_shape",
    "validate_batch_size",
    "validate_multimodal_tensors",
    # Performance monitoring
    "get_performance_monitor",
    "PerformanceMonitor",
    # Checkpoint validation
    "validate_checkpoint",
    "CheckpointValidator",
    # Common imports (selected frequently used ones)
    "Path",
    "PathLike",
    "ConfigDict",
    "SampleDict",
    "TensorLike",
    "json",
    "logging",
    "dataclass",
    "torch",
    "nn",
    "F",
    "Tensor",
    "Image",
    "np",
    "yaml",
    "check_torch_available",
    "check_pil_available",
    "require_torch",
    "require_pil",
    "ensure_path",
    "ensure_list",
    "get_available_device",
    "setup_logging",
]
