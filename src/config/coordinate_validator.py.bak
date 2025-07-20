"""
Comprehensive Coordinate Token Configuration Validator

This module provides validation for coordinate token configurations to ensure
complete consistency across all components of the Qwen2.5-VL training system.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from src.logger_utils import get_logger


@dataclass
class CoordinateConfigValidationResult:
    """Result of coordinate configuration validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_params: List[str]
    inconsistent_params: List[Tuple[str, str]]


class CoordinateConfigValidator:
    """
    Comprehensive validator for coordinate token configurations.
    
    Validates configuration consistency across:
    - YAML configuration files
    - Model configuration
    - Runtime parameters
    - Component integration
    """
    
    # Required coordinate configuration parameters
    REQUIRED_PARAMS = {
        "coordinate_config_enable_coordinate_tokens",
        "coordinate_config_max_coord_value",
        "coordinate_config_coordinate_loss_weight",
        "coordinate_config_regular_loss_weight",
        "coordinate_config_soft_expectation_temperature",
        "coordinate_config_focal_loss_alpha",
        "coordinate_config_focal_loss_gamma",
        "coordinate_config_use_official_box_tokens",
    }
    
    # Optional parameters with defaults
    OPTIONAL_PARAMS = {
        "coordinate_config_box_start_id": 151648,
        "coordinate_config_box_end_id": 151649,
        "coordinate_config_enable_validation": True,
        "coordinate_config_enable_caching": True,
        "coordinate_config_batch_processing": True,
        "coordinate_config_coord_token_init_std": 0.01,
    }
    
    # Parameter type validation
    PARAM_TYPES = {
        "coordinate_config_enable_coordinate_tokens": bool,
        "coordinate_config_max_coord_value": int,
        "coordinate_config_coordinate_loss_weight": float,
        "coordinate_config_regular_loss_weight": float,
        "coordinate_config_soft_expectation_temperature": float,
        "coordinate_config_focal_loss_alpha": float,
        "coordinate_config_focal_loss_gamma": float,
        "coordinate_config_box_start_id": int,
        "coordinate_config_box_end_id": int,
        "coordinate_config_enable_validation": bool,
        "coordinate_config_enable_caching": bool,
        "coordinate_config_batch_processing": bool,
        "coordinate_config_use_official_box_tokens": bool,
        "coordinate_config_coord_token_init_std": float,
    }
    
    # Parameter value constraints
    PARAM_CONSTRAINTS = {
        "coordinate_config_max_coord_value": lambda x: x > 0 and x <= 4096,
        "coordinate_config_coordinate_loss_weight": lambda x: x >= 0.0,
        "coordinate_config_regular_loss_weight": lambda x: x >= 0.0,
        "coordinate_config_soft_expectation_temperature": lambda x: x > 0.0,
        "coordinate_config_focal_loss_alpha": lambda x: 0.0 <= x <= 1.0,
        "coordinate_config_focal_loss_gamma": lambda x: x >= 0.0,
        "coordinate_config_box_start_id": lambda x: x >= 0,
        "coordinate_config_box_end_id": lambda x: x >= 0,
        "coordinate_config_coord_token_init_std": lambda x: x > 0.0,
    }
    
    def __init__(self):
        self.logger = get_logger("coordinate_config_validator")
    
    def validate_coordinate_config(
        self,
        config: Dict[str, Union[bool, int, float, str]],
        config_source: str = "unknown"
    ) -> CoordinateConfigValidationResult:
        """
        Validate coordinate token configuration.
        
        Args:
            config: Configuration dictionary to validate
            config_source: Source of configuration (for error reporting)
            
        Returns:
            CoordinateConfigValidationResult with validation results
        """
        errors = []
        warnings = []
        missing_params = []
        inconsistent_params = []
        
        # Check required parameters
        for param in self.REQUIRED_PARAMS:
            if param not in config:
                missing_params.append(param)
                errors.append(f"Missing required parameter: {param}")
        
        # Check parameter types
        for param, value in config.items():
            if param in self.PARAM_TYPES:
                expected_type = self.PARAM_TYPES[param]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Parameter '{param}' has type {type(value).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
        
        # Check parameter constraints
        for param, value in config.items():
            if param in self.PARAM_CONSTRAINTS:
                constraint = self.PARAM_CONSTRAINTS[param]
                if not constraint(value):
                    errors.append(f"Parameter '{param}' value {value} violates constraint")
        
        # Check coordinate token system consistency
        if config.get("enable_coordinate_tokens", False):
            coord_errors, coord_warnings = self._validate_coordinate_system(config)
            errors.extend(coord_errors)
            warnings.extend(coord_warnings)
        
        # Check for deprecated parameters
        deprecated_warnings = self._check_deprecated_params(config)
        warnings.extend(deprecated_warnings)
        
        is_valid = len(errors) == 0 and len(missing_params) == 0
        
        return CoordinateConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_params=missing_params,
            inconsistent_params=inconsistent_params
        )
    
    def _validate_coordinate_system(
        self, config: Dict[str, Union[bool, int, float, str]]
    ) -> Tuple[List[str], List[str]]:
        """Validate coordinate token system configuration."""
        errors = []
        warnings = []
        
        # Check max_coord_value consistency
        max_coord = config.get("max_coord_value", 2048)
        if max_coord not in [2048, 4096]:
            warnings.append(
                f"max_coord_value={max_coord} is unusual, typically 2048 or 4096"
            )
        
        # Check box token IDs
        box_start_id = config.get("box_start_id", 151648)
        box_end_id = config.get("box_end_id", 151649)
        
        if box_start_id >= box_end_id:
            errors.append(
                f"box_start_id ({box_start_id}) must be less than box_end_id ({box_end_id})"
            )
        
        # Check loss weights
        coord_weight = config.get("coordinate_loss_weight", 1.0)
        regular_weight = config.get("regular_loss_weight", 1.0)
        
        if coord_weight == 0.0 and regular_weight == 0.0:
            errors.append("Both coordinate_loss_weight and regular_loss_weight are zero")
        
        # Check temperature parameter
        temperature = config.get("soft_expectation_temperature", 1.0)
        if temperature <= 0.0:
            errors.append("soft_expectation_temperature must be positive")
        
        # Check focal loss parameters
        alpha = config.get("focal_loss_alpha", 0.25)
        gamma = config.get("focal_loss_gamma", 2.0)
        
        if alpha < 0.0 or alpha > 1.0:
            errors.append("focal_loss_alpha must be in range [0, 1]")
        
        if gamma < 0.0:
            errors.append("focal_loss_gamma must be non-negative")
        
        return errors, warnings
    
    def _check_deprecated_params(
        self, config: Dict[str, Union[bool, int, float, str]]
    ) -> List[str]:
        """Check for deprecated configuration parameters."""
        warnings = []
        
        # Legacy coordinate parameters
        legacy_params = [
            "use_legacy_coordinates",
            "use_legacy_bbox_generation",
            "use_legacy_bbox_tokens",
            "use_legacy_bbox_pattern",
            "use_legacy_bbox_format",
            "use_legacy_bbox_regex",
            "use_legacy_bbox_processing",
            "use_legacy_bbox_scale",
        ]
        
        for param in legacy_params:
            if param in config:
                warnings.append(f"Deprecated parameter found: {param}")
        
        return warnings
    
    def validate_config_consistency(
        self,
        configs: Dict[str, Dict[str, Union[bool, int, float, str]]],
    ) -> CoordinateConfigValidationResult:
        """
        Validate consistency across multiple configuration sources.
        
        Args:
            configs: Dictionary mapping source names to configuration dictionaries
            
        Returns:
            CoordinateConfigValidationResult with consistency validation results
        """
        errors = []
        warnings = []
        missing_params = []
        inconsistent_params = []
        
        # Validate each individual configuration
        for source, config in configs.items():
            result = self.validate_coordinate_config(config, source)
            if not result.is_valid:
                errors.extend([f"[{source}] {error}" for error in result.errors])
                warnings.extend([f"[{source}] {warning}" for warning in result.warnings])
                missing_params.extend([f"[{source}] {param}" for param in result.missing_params])
        
        # Check consistency across configurations
        if len(configs) > 1:
            consistency_errors = self._check_cross_config_consistency(configs)
            errors.extend(consistency_errors)
        
        is_valid = len(errors) == 0 and len(missing_params) == 0
        
        return CoordinateConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_params=missing_params,
            inconsistent_params=inconsistent_params
        )
    
    def _check_cross_config_consistency(
        self, configs: Dict[str, Dict[str, Union[bool, int, float, str]]]
    ) -> List[str]:
        """Check consistency across multiple configuration sources."""
        errors = []
        
        # Parameters that must be consistent across all configs
        consistency_params = [
            "enable_coordinate_tokens",
            "max_coord_value",
            "box_start_id",
            "box_end_id",
        ]
        
        for param in consistency_params:
            values = {}
            for source, config in configs.items():
                if param in config:
                    values[source] = config[param]
            
            if len(set(values.values())) > 1:
                value_str = ", ".join([f"{source}={value}" for source, value in values.items()])
                errors.append(f"Inconsistent '{param}' values: {value_str}")
        
        return errors
    
    def generate_validation_report(
        self, result: CoordinateConfigValidationResult
    ) -> str:
        """Generate human-readable validation report."""
        lines = []
        
        if result.is_valid:
            lines.append("✅ Coordinate configuration validation PASSED")
        else:
            lines.append("❌ Coordinate configuration validation FAILED")
        
        if result.errors:
            lines.append("\n❌ ERRORS:")
            for error in result.errors:
                lines.append(f"  - {error}")
        
        if result.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        
        if result.missing_params:
            lines.append("\n🔍 MISSING PARAMETERS:")
            for param in result.missing_params:
                lines.append(f"  - {param}")
        
        if result.inconsistent_params:
            lines.append("\n⚡ INCONSISTENT PARAMETERS:")
            for param, sources in result.inconsistent_params:
                lines.append(f"  - {param}: {sources}")
        
        return "\n".join(lines)


def validate_coordinate_config(
    config: Dict[str, Union[bool, int, float, str]],
    config_source: str = "unknown"
) -> CoordinateConfigValidationResult:
    """
    Convenience function to validate coordinate configuration.
    
    Args:
        config: Configuration dictionary to validate
        config_source: Source of configuration (for error reporting)
        
    Returns:
        CoordinateConfigValidationResult with validation results
    """
    validator = CoordinateConfigValidator()
    return validator.validate_coordinate_config(config, config_source)


def validate_config_consistency(
    configs: Dict[str, Dict[str, Union[bool, int, float, str]]]
) -> CoordinateConfigValidationResult:
    """
    Convenience function to validate consistency across multiple configurations.
    
    Args:
        configs: Dictionary mapping source names to configuration dictionaries
        
    Returns:
        CoordinateConfigValidationResult with consistency validation results
    """
    validator = CoordinateConfigValidator()
    return validator.validate_config_consistency(configs)