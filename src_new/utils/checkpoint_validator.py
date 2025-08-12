"""
Checkpoint validation utilities for inference-ready checkpoints.

This module provides tools to validate that saved checkpoints have the correct
structure and components for inference compatibility.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from .rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


class CheckpointValidator:
    """Validator for inference-ready checkpoint structure and content."""

    # Required files for inference-ready checkpoints
    REQUIRED_FILES = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]

    # Required model weight files (at least one must exist)
    MODEL_WEIGHT_FILES = [
        "model.safetensors",  # Single file format
        "pytorch_model.bin",  # Legacy format
    ]

    # Optional but recommended files
    OPTIONAL_FILES = [
        "generation_config.json",
        "coordinate_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
    ]

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint validator.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.validation_results = {}

    def validate_checkpoint(self) -> Tuple[bool, Dict[str, any]]:
        """
        Validate checkpoint structure and content.

        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            "checkpoint_path": str(self.checkpoint_path),
            "exists": self.checkpoint_path.exists(),
            "required_files": {},
            "model_weights": {},
            "optional_files": {},
            "coordinate_tokens": {},
            "config_validation": {},
            "errors": [],
            "warnings": [],
        }

        if not results["exists"]:
            results["errors"].append(
                f"Checkpoint directory does not exist: {self.checkpoint_path}"
            )
            return False, results

        # Validate required files
        self._validate_required_files(results)

        # Validate model weights
        self._validate_model_weights(results)

        # Validate optional files
        self._validate_optional_files(results)

        # Validate coordinate token configuration
        self._validate_coordinate_tokens(results)

        # Validate configuration files
        self._validate_configs(results)

        # Determine overall validity
        is_valid = (
            len(results["errors"]) == 0
            and results["model_weights"]["has_weights"]
            and all(results["required_files"].values())
        )

        results["is_valid"] = is_valid
        return is_valid, results

    def _validate_required_files(self, results: Dict):
        """Validate presence of required files."""
        for file_name in self.REQUIRED_FILES:
            file_path = self.checkpoint_path / file_name
            exists = file_path.exists()
            results["required_files"][file_name] = exists

            if not exists:
                results["errors"].append(f"Missing required file: {file_name}")

    def _validate_model_weights(self, results: Dict):
        """Validate model weight files."""
        weight_files_found = []

        # Check for single model files
        for weight_file in self.MODEL_WEIGHT_FILES:
            file_path = self.checkpoint_path / weight_file
            if file_path.exists():
                weight_files_found.append(weight_file)
                results["model_weights"][weight_file] = {
                    "exists": True,
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                }

        # Check for sharded model files
        sharded_files = list(self.checkpoint_path.glob("model-*.safetensors"))
        if sharded_files:
            results["model_weights"]["sharded_files"] = len(sharded_files)
            total_size = sum(f.stat().st_size for f in sharded_files)
            results["model_weights"]["total_sharded_size_mb"] = total_size / (
                1024 * 1024
            )

            # Check for index file
            index_file = self.checkpoint_path / "model.safetensors.index.json"
            if index_file.exists():
                results["model_weights"]["has_index"] = True
            else:
                results["warnings"].append("Sharded model found but missing index file")

        # Determine if we have valid model weights
        has_weights = len(weight_files_found) > 0 or len(sharded_files) > 0
        results["model_weights"]["has_weights"] = has_weights

        if not has_weights:
            results["errors"].append("No model weight files found")

    def _validate_optional_files(self, results: Dict):
        """Validate optional files."""
        for file_name in self.OPTIONAL_FILES:
            file_path = self.checkpoint_path / file_name
            results["optional_files"][file_name] = file_path.exists()

    def _validate_coordinate_tokens(self, results: Dict):
        """Validate coordinate token configuration."""
        coord_config_path = self.checkpoint_path / "coordinate_config.json"

        if coord_config_path.exists():
            try:
                with open(coord_config_path, "r") as f:
                    coord_config = json.load(f)

                results["coordinate_tokens"]["config_exists"] = True
                results["coordinate_tokens"]["enabled"] = coord_config.get(
                    "coordinate_tokens_enabled", False
                )
                results["coordinate_tokens"]["max_coord_value"] = coord_config.get(
                    "max_coord_value"
                )
                results["coordinate_tokens"]["vocab_size_extended"] = coord_config.get(
                    "vocab_size_extended"
                )

                # Validate tokenizer has extended vocabulary
                tokenizer_config_path = self.checkpoint_path / "tokenizer_config.json"
                if tokenizer_config_path.exists():
                    with open(tokenizer_config_path, "r") as f:
                        tokenizer_config = json.load(f)

                    vocab_size = tokenizer_config.get("vocab_size")
                    if vocab_size and vocab_size > 151665:
                        results["coordinate_tokens"]["tokenizer_extended"] = True
                    else:
                        results["warnings"].append(
                            "Coordinate tokens enabled but tokenizer not extended"
                        )

            except Exception as e:
                results["errors"].append(f"Failed to parse coordinate_config.json: {e}")
        else:
            results["coordinate_tokens"]["config_exists"] = False

    def _validate_configs(self, results: Dict):
        """Validate configuration files."""
        # Validate model config
        config_path = self.checkpoint_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    model_config = json.load(f)

                results["config_validation"]["model_config"] = {
                    "valid": True,
                    "vocab_size": model_config.get("vocab_size"),
                    "model_type": model_config.get("model_type"),
                }
            except Exception as e:
                results["errors"].append(f"Failed to parse config.json: {e}")

        # Validate preprocessor config
        preprocessor_path = self.checkpoint_path / "preprocessor_config.json"
        if preprocessor_path.exists():
            try:
                with open(preprocessor_path, "r") as f:
                    preprocessor_config = json.load(f)

                results["config_validation"]["preprocessor_config"] = {
                    "valid": True,
                    "max_pixels": preprocessor_config.get("max_pixels"),
                }
            except Exception as e:
                results["errors"].append(
                    f"Failed to parse preprocessor_config.json: {e}"
                )

    def print_validation_report(self, results: Dict):
        """Print a formatted validation report."""
        logger.info(f"\n🔍 Checkpoint Validation Report")
        logger.info(f"📁 Path: {results['checkpoint_path']}")
        is_valid = bool(results["is_valid"]) if "is_valid" in results else False
        logger.info(f"✅ Valid: {is_valid}")

        if results["errors"]:
            logger.error(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results["errors"]:
                logger.error(f"   • {error}")

        if results["warnings"]:
            logger.warning(f"\n⚠️ Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                logger.warning(f"   • {warning}")

        logger.info(f"\n📋 Required Files:")
        for file_name, exists in results["required_files"].items():
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {file_name}")

        logger.info(f"\n💾 Model Weights:")
        if results["model_weights"]["has_weights"]:
            logger.info(f"   ✅ Model weights found")
            for key, value in results["model_weights"].items():
                if key not in ["has_weights"] and isinstance(value, (int, float)):
                    logger.info(f"   📊 {key}: {value}")
        else:
            logger.error(f"   ❌ No model weights found")

        if (
            "config_exists" in results["coordinate_tokens"]
            and results["coordinate_tokens"]["config_exists"]
        ):
            logger.info(f"\n🎯 Coordinate Tokens:")
            coord_info = results["coordinate_tokens"]
            logger.info(f"   ✅ Configuration found")
            logger.info(f"   📊 Enabled: {coord_info.get('enabled')}")
            logger.info(f"   📊 Max coord value: {coord_info.get('max_coord_value')}")
            logger.info(
                f"   📊 Extended vocab size: {coord_info.get('vocab_size_extended')}"
            )


def validate_checkpoint(
    checkpoint_path: str, print_report: bool = True
) -> Tuple[bool, Dict]:
    """
    Validate a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory
        print_report: Whether to print validation report

    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = CheckpointValidator(checkpoint_path)
    is_valid, results = validator.validate_checkpoint()

    if print_report:
        validator.print_validation_report(results)

    return is_valid, results


def validate_all_checkpoints(output_dir: str) -> Dict[str, Tuple[bool, Dict]]:
    """
    Validate all checkpoints in an output directory.

    Args:
        output_dir: Directory containing checkpoints

    Returns:
        Dictionary mapping checkpoint names to validation results
    """
    output_path = Path(output_dir)
    checkpoint_dirs = [
        d
        for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    results = {}
    for checkpoint_dir in sorted(checkpoint_dirs):
        checkpoint_name = checkpoint_dir.name
        is_valid, validation_results = validate_checkpoint(
            str(checkpoint_dir), print_report=False
        )
        results[checkpoint_name] = (is_valid, validation_results)

    return results
