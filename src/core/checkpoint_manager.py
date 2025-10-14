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
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors import safe_open
from transformers.models.auto.processing_auto import AutoProcessor

# Removed explicit config import - config will be passed as parameter
from src.logger_utils import get_training_logger


class CheckpointManager:
    """Manager for model checkpoints and saving/loading operations."""

    def __init__(self, config=None, tokenizer=None, image_processor=None, logger=None):
        """
        Initialize checkpoint manager.

        Args:
            config: Configuration object (optional - can be passed to individual methods)
            tokenizer: Model tokenizer (can be provided later)
            image_processor: Image processor (can be provided later)
            logger: Logger instance (optional)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.logger = logger or get_training_logger()
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

    def save_checkpoint(
        self,
        model: Any,
        output_dir: str,
        training_args: Any = None,
        state_dict: Optional[dict] = None,
    ) -> None:
        """
        Save complete checkpoint in sharded form with coordinate token support.

        This method combines all checkpoint saving logic:
        1. Save model weights (sharded for large models)
        2. Save tokenizer with coordinate token persistence
        3. Save image processor configuration
        4. Copy essential files from base model
        5. Validate checkpoint integrity

        Args:
            model: The model to save
            output_dir: Directory to save checkpoint
            training_args: Training arguments to save
            state_dict: Optional state dict override
        """
        if output_dir is None:
            raise ValueError("output_dir cannot be None")

        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"💾 Saving checkpoint to {output_dir}")

        # 1. Save model weights (sharded)
        self._save_model_weights(model, output_dir)

        # 2. Save tokenizer with coordinate token persistence
        if self.tokenizer is not None:
            self._save_tokenizer_with_coordinate_tokens(output_dir)
        else:
            self.logger.warning("⚠️ No tokenizer provided - skipping tokenizer save")

        # 3. Save image processor
        if self.image_processor is not None:
            self._save_image_processor_config(output_dir)
        else:
            self.logger.warning(
                "⚠️ No image processor provided - skipping processor save"
            )

        # 4. Copy essential files from base model
        self.copy_base_model_files(output_dir)

        # 5. Save training arguments
        if training_args is not None:
            torch.save(training_args, os.path.join(output_dir, "training_args.bin"))
            self.logger.info("💾 Training arguments saved")

        # 6. Validate model-tokenizer consistency
        self.validate_model_consistency(model)

        # 7. Verify checkpoint integrity
        self.verify_checkpoint_integrity(output_dir)

        self.logger.info(f"✅ Checkpoint saved successfully → {output_dir}")

    def _save_model_weights(self, model: Any, output_dir: str) -> None:
        """Save model weights in sharded format."""
        # Determine which model to save based on coordinate token configuration
        if (
            not hasattr(model, "coordinate_tokens_enabled")
            or not model.coordinate_tokens_enabled
        ):
            # For coordinate tokens disabled, save the full Qwen2.5-VL model
            self.logger.info(
                "💾 Saving complete Qwen2.5-VL model with visual tower (sharded)…"
            )
            model_to_save = model
        else:
            # For coordinate tokens enabled, save the enhanced model
            self.logger.info("💾 Saving base Qwen2.5-VL model (sharded)…")
            # Use base_model if available, otherwise use the model itself
            model_to_save = model.base_model if hasattr(model, "base_model") else model

        # Save with sharding for large models
        model_to_save.save_pretrained(
            output_dir, max_shard_size="2GB", safe_serialization=True
        )
        self.logger.info("✅ Model weights saved (sharded)")

    def _save_tokenizer_with_coordinate_tokens(self, output_dir: str) -> None:
        """Save tokenizer with coordinate token persistence."""
        self.logger.info("💾 Saving tokenizer...")

        # Ensure coordinate tokens are persisted before saving
        self.persist_coordinate_tokens()

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info("✅ Tokenizer saved with coordinate tokens")

    def persist_coordinate_tokens(self) -> None:
        """
        Ensure coordinate tokens added via UnifiedTokenManager are persisted in tokenizer vocabulary.

        This is critical for coordinate token mode functionality - without this fix,
        coordinate tokens exist in model embeddings but are missing from vocabulary files,
        causing unstable inference behavior.
        """
        if self.tokenizer is None:
            self.logger.warning(
                "⚠️ No tokenizer provided - skipping coordinate token persistence"
            )
            return

        # Use explicit config - no fallback patterns
        if self.config is None:
            self.logger.warning(
                "⚠️ No config provided - skipping coordinate token persistence"
            )
            return

        if not hasattr(self.config, "coordinate_tokens_enabled"):
            raise ValueError("coordinate_tokens_enabled must be explicitly configured")

        # Only persist coordinate tokens if coordinate mode is enabled
        if not self.config.coordinate_tokens_enabled:
            self.logger.debug("🎯 Coordinate tokens disabled, skipping persistence")
            return

        if not hasattr(self.config, "max_coord_value"):
            raise ValueError(
                "max_coord_value must be explicitly configured when coordinate tokens are enabled"
            )

        max_coord_value = self.config.max_coord_value

        # Check if coordinate tokens already exist in tokenizer vocabulary
        vocab = self.tokenizer.get_vocab()
        coord_0_token = "<|coord_0|>"

        if coord_0_token in vocab:
            self.logger.info(
                f"✅ Coordinate tokens already present in vocabulary (range: {max_coord_value})"
            )
            return

        # If coordinate tokens are missing, add them to ensure persistence
        self.logger.warning(
            "⚠️ Coordinate tokens missing from vocabulary - adding for persistence"
        )

        # Generate coordinate tokens [0, max_coord_value-1]
        coord_tokens = [f"<|coord_{i}|>" for i in range(max_coord_value)]

        # Add coordinate tokens in batches to avoid memory issues
        batch_size = 1000
        total_added = 0

        for i in range(0, len(coord_tokens), batch_size):
            batch = coord_tokens[i : i + batch_size]
            num_added = self.tokenizer.add_tokens(batch)
            total_added += num_added

        self.logger.info(
            f"✅ Added {total_added} coordinate tokens to tokenizer for persistence"
        )

        # Verify coordinate tokens are now present
        updated_vocab = self.tokenizer.get_vocab()
        if coord_0_token in updated_vocab:
            coord_start_id = updated_vocab[coord_0_token]
            self.logger.info(
                f"🎯 Coordinate token persistence verified: range [{coord_start_id}, {coord_start_id + max_coord_value})"
            )
        else:
            self.logger.error("❌ Failed to add coordinate tokens to vocabulary")

    def _save_image_processor_config(self, output_dir: str) -> None:
        """Save image processor configuration preserving base model settings."""
        if self.image_processor is None:
            raise RuntimeError(
                "Image processor is None - cannot save preprocessor config!"
            )

        self.logger.info("💾 Saving image processor...")

        # Load base config from pretrained model and only override specific values
        if self.config is None:
            raise ValueError("Config is required for image processor saving")

        if not hasattr(self.config, "model_path"):
            raise ValueError("Config missing 'model_path' attribute")

        base_model_path = self.config.model_path
        if base_model_path is None or base_model_path == "":
            raise ValueError("model_path cannot be None or empty in config")

        base_preproc_path = os.path.join(base_model_path, "preprocessor_config.json")

        if os.path.exists(base_preproc_path):
            with open(base_preproc_path, "r", encoding="utf-8") as f:
                ip_cfg = json.load(f)
        else:
            raise RuntimeError(
                f"Base preprocessor config not found: {base_preproc_path}"
            )

        # Only override the values that might have changed during training
        if hasattr(self.image_processor, "min_pixels"):
            ip_cfg["min_pixels"] = self.image_processor.min_pixels
        if hasattr(self.image_processor, "max_pixels"):
            ip_cfg["max_pixels"] = self.image_processor.max_pixels

        # Verify all other critical attributes exist
        critical_attrs = [
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "image_mean",
            "image_std",
        ]
        for attr_name in critical_attrs:
            if attr_name not in ip_cfg:
                raise RuntimeError(
                    f"Critical attribute missing from preprocessor config: {attr_name}"
                )

        # Save the configuration
        with open(
            os.path.join(output_dir, "preprocessor_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ip_cfg, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"   ✅ Image processor config saved with {len(ip_cfg)} parameters (preserving base config)"
        )

    def copy_base_model_files(self, output_dir: str) -> None:
        """Copy essential files from base model to match pretrained model structure."""
        self.logger.info("💾 Copying essential files from base model...")

        if self.config is None:
            raise ValueError("Config is required for copying base model files")

        # Get base model path from config
        base_model_path = self.config.model_path
        if not os.path.exists(base_model_path):
            raise RuntimeError(f"Base model path not found: {base_model_path}")

        # Files to copy from pretrained model directory structure
        essential_files = [
            "generation_config.json",
        ]

        # Optional files for full compatibility (not required for functionality)
        optional_files = [
            "chat_template.json",  # Not used - we have custom system prompts
            "LICENSE",
            "README.md",
        ]

        # Copy essential files
        missing_source_files = []
        base_model_path_str = str(base_model_path)
        output_dir_str = str(output_dir)

        for file_name in essential_files:
            file_name_str = str(file_name)
            src_path = os.path.join(base_model_path_str, file_name_str)
            dst_path = os.path.join(output_dir_str, file_name_str)

            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                self.logger.info(f"   ✅ Copied {file_name}")
            elif os.path.exists(dst_path):
                self.logger.info(f"   ✅ {file_name} already exists")
            else:
                self.logger.error(f"   ❌ {file_name} not found in base model")
                missing_source_files.append(file_name)

        if missing_source_files:
            raise RuntimeError(
                f"Essential files missing from base model {base_model_path}: {missing_source_files}"
            )

        # Copy optional files (best effort, don't fail if missing)
        for file_name in optional_files:
            file_name_str = str(file_name)
            src_path = os.path.join(base_model_path_str, file_name_str)
            dst_path = os.path.join(output_dir_str, file_name_str)

            if os.path.exists(dst_path):
                self.logger.info(f"   ✅ {file_name} already exists (optional)")
            elif os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                self.logger.info(f"   ✅ Copied {file_name} (optional)")
            else:
                self.logger.info(
                    f"   ℹ️ Optional file {file_name} not found in base model"
                )

        self.logger.info("✅ Base model files copied")

    def verify_checkpoint_integrity(self, output_dir: str) -> None:
        """Verify that all necessary components were saved in the checkpoint."""
        self.logger.info("🔍 Verifying saved checkpoint components...")

        # Check essential files exist
        essential_files = [
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "generation_config.json",
        ]

        missing_files = []
        for file_name in essential_files:
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                self.logger.info(f"   ✅ {file_name} exists")
            else:
                self.logger.error(f"   ❌ {file_name} MISSING")
                missing_files.append(file_name)

        if missing_files:
            raise RuntimeError(f"Essential checkpoint files missing: {missing_files}")

        # Check model weights and verify visual components
        safetensor_files = [
            f for f in os.listdir(output_dir) if f.endswith(".safetensors")
        ]
        if not safetensor_files:
            raise RuntimeError("No safetensor model files found in checkpoint!")

        self.logger.info(f"   ✅ Found {len(safetensor_files)} safetensor files")

        # Check for visual tower weights in the first safetensor file
        first_file = os.path.join(output_dir, safetensor_files[0])
        visual_keys = []
        total_keys = 0

        with safe_open(first_file, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            total_keys = len(all_keys)

            # Check for visual parameters
            for key in all_keys:
                if "visual" in key:
                    visual_keys.append(key)

        self.logger.info(f"   ✅ Total parameters saved: {total_keys}")

        # Safe fallback: Vision tower check for wrapped models
        if not visual_keys:
            # Check if we're dealing with a wrapped model that stores visual components differently
            wrapped_visual_keys = [
                key
                for key in all_keys
                if "base_model" in key and "visual" in key.lower()
            ]
            if wrapped_visual_keys:
                self.logger.info(
                    f"   ✅ Found visual parameters in wrapped model: {len(wrapped_visual_keys)}"
                )
                visual_keys = wrapped_visual_keys
            else:
                self.logger.warning(
                    "⚠️ No visual tower parameters found in checkpoint! "
                    "This may indicate the model is wrapped or uses a different structure. "
                    "Proceeding with caution - verify model loads correctly."
                )

        self.logger.info(f"   ✅ Visual tower parameters found: {len(visual_keys)}")
        if visual_keys:
            self.logger.info(f"      Examples: {visual_keys[:3]}...")

    def validate_model_consistency(self, model: Any) -> None:
        """
        Validate that tokenizer vocabulary and model embeddings are consistent.

        This is critical for coordinate token functionality - ensures that:
        1. Model embedding dimensions match tokenizer vocabulary size
        2. Coordinate tokens exist in both tokenizer and model
        3. Model can properly process coordinate token inputs
        """
        if self.tokenizer is None:
            self.logger.warning(
                "⚠️ No tokenizer provided - skipping consistency validation"
            )
            return

        self.logger.info("🔍 Validating tokenizer-model consistency...")

        # Get vocabulary size from tokenizer
        tokenizer_vocab_size = len(self.tokenizer)
        vocab = self.tokenizer.get_vocab()

        # Get model embedding dimensions
        model_to_check = model
        if hasattr(model, "base_model"):
            model_to_check = model.base_model

        # Check input embeddings
        if hasattr(model_to_check, "model") and hasattr(
            model_to_check.model, "embed_tokens"
        ):
            embed_weight = model_to_check.model.embed_tokens.weight
            model_vocab_size = embed_weight.shape[0]

            if model_vocab_size != tokenizer_vocab_size:
                raise RuntimeError(
                    f"CRITICAL: Model embedding size ({model_vocab_size}) doesn't match "
                    f"tokenizer vocabulary size ({tokenizer_vocab_size}). "
                    f"This will cause coordinate token inference failures."
                )

            self.logger.info(
                f"✅ Input embeddings consistent: {model_vocab_size} tokens"
            )

        # Check output LM head
        if hasattr(model_to_check, "lm_head"):
            lm_head_weight = model_to_check.lm_head.weight
            lm_head_vocab_size = lm_head_weight.shape[0]

            if lm_head_vocab_size != tokenizer_vocab_size:
                raise RuntimeError(
                    f"CRITICAL: LM head size ({lm_head_vocab_size}) doesn't match "
                    f"tokenizer vocabulary size ({tokenizer_vocab_size}). "
                    f"This will cause coordinate token generation failures."
                )

            self.logger.info(f"✅ LM head consistent: {lm_head_vocab_size} tokens")

        # Validate coordinate tokens if enabled
        if (
            self.config
            and hasattr(self.config, "coordinate_tokens_enabled")
            and self.config.coordinate_tokens_enabled
        ):
            max_coord_value = getattr(self.config, "max_coord_value", 1000)
            coord_0_token = "<|coord_0|>"
            coord_max_token = f"<|coord_{max_coord_value - 1}|>"

            # Check that coordinate tokens exist in vocabulary
            if coord_0_token not in vocab:
                raise RuntimeError(
                    f"CRITICAL: Coordinate token {coord_0_token} missing from vocabulary. "
                    f"Coordinate token persistence failed."
                )

            if coord_max_token not in vocab:
                raise RuntimeError(
                    f"CRITICAL: Coordinate token {coord_max_token} missing from vocabulary. "
                    f"Not all coordinate tokens were added properly."
                )

            coord_start_id = vocab[coord_0_token]
            coord_end_id = vocab[coord_max_token]

            self.logger.info(
                f"✅ Coordinate tokens validated: range [{coord_start_id}, {coord_end_id}] "
                f"({coord_end_id - coord_start_id + 1} tokens)"
            )

            # Verify coordinate tokens are contiguous
            expected_tokens = coord_end_id - coord_start_id + 1
            if expected_tokens != max_coord_value:
                self.logger.warning(
                    f"⚠️ Coordinate tokens may not be contiguous: "
                    f"expected {max_coord_value}, found range of {expected_tokens}"
                )

        self.logger.info("✅ Tokenizer-model consistency validation passed")

    def resize_model_embeddings(self, model: Any) -> None:
        """
        Resize model embeddings to match tokenizer vocabulary size.

        This is critical when coordinate tokens are added to the tokenizer
        and the model embeddings need to be updated accordingly.
        """
        if self.tokenizer is None:
            self.logger.warning("⚠️ No tokenizer provided - skipping embedding resize")
            return

        # Get current vocabulary size
        current_vocab_size = len(self.tokenizer)

        # Resize model embeddings if the model supports it
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(current_vocab_size)
            self.logger.info(
                f"✅ Model embeddings resized to match tokenizer: {current_vocab_size} tokens"
            )
        else:
            self.logger.warning("⚠️ Model doesn't support resize_token_embeddings")

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set the tokenizer for checkpoint operations."""
        self.tokenizer = tokenizer
        self.logger.debug("🔧 Tokenizer updated in CheckpointManager")

    def set_image_processor(self, image_processor: Any) -> None:
        """Set the image processor for checkpoint operations."""
        self.image_processor = image_processor
        self.logger.debug("🔧 Image processor updated in CheckpointManager")

    def set_config(self, config: Any) -> None:
        """Set the configuration for checkpoint operations."""
        self.config = config
        self.logger.debug("🔧 Config updated in CheckpointManager")
