"""Modern BBU Configuration System - YAML is the single source of truth."""

from .config import (
    BBUConfig,
    CoordinateConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
    VisionConfig,
    get_config,
    init_config,
    load_config,
)
