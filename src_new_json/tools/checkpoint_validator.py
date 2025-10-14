"""
Checkpoint validation utilities for inference-ready checkpoints.

This module provides tools to validate that saved checkpoints have the correct
structure and components for inference compatibility.
"""

from pathlib import Path
from typing import Dict, Tuple

from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("checkpoint_validator")


class CheckpointValidator:
    """Validator for inference-ready checkpoint structure and content."""

    # Required files for inference-ready checkpoints
    REQUIRED_FILES = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "generation_config.json",
    ]

    # Required model weight files (at least one must exist)
    MODEL_WEIGHT_FILES = [
        "model.safetensors",  # Single file format
        "pytorch_model.bin",  # Legacy format
    ]

    # Optional but recommended files
    OPTIONAL_FILES = [
        "vocab.json",
        "merges.txt",
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
                }

        # Check for sharded model files
        sharded_files = list(self.checkpoint_path.glob("model-*.safetensors"))
        if sharded_files:
            results["model_weights"]["sharded_files"] = len(sharded_files)
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

    def _validate_configs(self, results: Dict):
        """Placeholder for config validation (JSON mode)."""
        return

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
