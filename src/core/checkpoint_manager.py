"""
Checkpoint Manager for BBU Training System

Centralized model saving, loading, and checkpoint management.
Extracted from trainer utilities for better separation of concerns.

Key Features:
- Safe model saving with HuggingFace compatibility
- Checkpoint validation and metadata
- Model configuration preservation
- Error handling with fail-fast approach
- Support for both training and inference checkpoints
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers.models.auto.processing_auto import AutoProcessor

# Removed explicit config import - config will be passed as parameter
from src.logger_utils import get_training_logger


class CheckpointManager:
    """Manager for model checkpoints and saving/loading operations."""

    def __init__(self, config=None):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Configuration object (optional - can be passed to individual methods)
        """
        self.logger = get_training_logger()
        self.config = config
        self.logger.info("📄 CheckpointManager initialized")

    def save_model_safely(
        self, trainer: Any, output_dir: str, model_path: Optional[str] = None
    ) -> bool:
        """
        Safely save model with proper HuggingFace compatibility.

        Args:
            trainer: The trainer instance
            output_dir: Directory to save the model
            model_path: Path to the original model (optional, will use config if not provided)

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"💾 Saving model to {output_dir}...")

        # Validate trainer has save_model method
        if not hasattr(trainer, "save_model"):
            raise ValueError("Trainer must have save_model method")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use trainer's built-in save method
        trainer.save_model(output_dir)

        # Save additional components
        self._save_image_processor(output_dir, model_path)
        self._save_checkpoint_metadata(output_dir, model_path)

        self.logger.info(f"✅ Model saved successfully to {output_dir}")
        return True

    # REMOVED: _fallback_save_model method - FAIL FAST approach
    # Save failures should stop training immediately

    def _save_image_processor(
        self, output_dir: str, model_path: Optional[str] = None
    ) -> None:
        """Save image processor to output directory."""
        # Use provided model_path or get from config with validation
        if model_path is None:
            if self.config is None:
                raise ValueError(
                    "Config is required when model_path is not provided. "
                    "Initialize CheckpointManager with config or provide model_path."
                )
                
            if not hasattr(self.config, "model_path"):
                raise ValueError("Config missing required attribute: model_path")
                
            model_path = self.config.model_path
            
        if not model_path:
            raise ValueError("model_path cannot be empty")

        # Load processor with validation
        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load processor from {model_path}: {e}") from e

        # Validate image processor exists
        if not hasattr(processor, "image_processor"):
            raise ValueError(f"Processor from {model_path} missing image_processor attribute")
            
        if processor.image_processor is None:
            raise ValueError(f"Processor from {model_path} has None image_processor")

        # Save image processor
        processor.image_processor.save_pretrained(output_dir)
        self.logger.info(f"💾 Image processor saved to: {output_dir}")

    def _save_checkpoint_metadata(
        self, output_dir: str, model_path: Optional[str] = None
    ) -> None:
        """Save checkpoint metadata for tracking."""
        # Use provided model_path or get from config with validation
        if model_path is None:
            if self.config is None:
                raise ValueError(
                    "Config is required when model_path is not provided. "
                    "Initialize CheckpointManager with config or provide model_path."
                )
                
            if not hasattr(self.config, "model_path"):
                raise ValueError("Config missing required attribute: model_path")
                
            model_path = self.config.model_path
            
        if not model_path:
            raise ValueError("model_path cannot be empty")
            
        # Validate config has required attributes
        if self.config is None:
            raise ValueError("Config is required for checkpoint metadata")
            
        if not hasattr(self.config, "coordinate_tokens_enabled"):
            raise ValueError("Config missing required attribute: coordinate_tokens_enabled")

        # Create metadata with explicit values
        metadata = {
            "checkpoint_type": "bbu_training",
            "config_system": "unified",
            "model_architecture": "Qwen2.5-VL-BBU",
            "training_framework": "transformers_bbu_custom",
            "creation_timestamp": str(
                torch.cuda.Event().query() if torch.cuda.is_available() else "cpu"
            ),
            "model_path": model_path,
            "coordinate_tokens_enabled": self.config.coordinate_tokens_enabled,
        }

        # Save metadata to file
        metadata_path = Path(output_dir) / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"📋 Checkpoint metadata saved to: {metadata_path}")

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint integrity and compatibility.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            True if valid, False otherwise
        """
        self.logger.info(f"🔍 Validating checkpoint: {checkpoint_path}")

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            self.logger.error(
                f"❌ Checkpoint directory does not exist: {checkpoint_path}"
            )
            return False

        # Check for required files
        required_files = [
            "config.json",
            "pytorch_model.bin",
        ]

        missing_files = []
        for file_name in required_files:
            if not (checkpoint_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            self.logger.error(f"❌ Missing required files: {missing_files}")
            return False

        # Check metadata if available
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.info(
                    f"📋 Checkpoint metadata: {metadata.get('checkpoint_type', 'unknown')}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to read checkpoint metadata: {e}")

        self.logger.info(f"✅ Checkpoint validation passed: {checkpoint_path}")
        return True

    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            Dictionary with checkpoint information, or None if invalid
        """
        if not self.validate_checkpoint(checkpoint_path):
            return None

        checkpoint_dir = Path(checkpoint_path)
        info = {
            "path": str(checkpoint_dir),
            "size_mb": sum(
                f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file()
            )
            / (1024 * 1024),
            "files": [f.name for f in checkpoint_dir.iterdir() if f.is_file()],
        }

        # Add metadata if available
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    info["metadata"] = json.load(f)
            except Exception:
                pass

        return info
