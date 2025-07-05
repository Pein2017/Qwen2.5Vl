"""
Checkpoint Manager for BBU Training System

Centralized model saving, loading, and checkpoint management.
Extracted from trainer utilities for better separation of concerns.

Key Features:
- Safe model saving with HuggingFace compatibility
- Checkpoint validation and metadata
- Model configuration preservation  
- Error handling and fallback mechanisms
- Support for both training and inference checkpoints
"""

from typing import Any, Optional, Dict
from pathlib import Path
import json

import torch
from transformers import AutoProcessor

from src.config import config, get_config_manager
from src.logger_utils import get_training_logger


class CheckpointManager:
    """Manager for model checkpoints and saving/loading operations."""
    
    def __init__(self, use_new_config: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            use_new_config: Whether to use new domain-specific config system
        """
        self.use_new_config = use_new_config
        self.logger = get_training_logger()
        
        # Get configuration
        if use_new_config:
            try:
                self.config = get_config_manager()
                self.logger.info("✅ CheckpointManager using new domain-specific configuration")
            except RuntimeError:
                self.logger.warning("⚠️  New config system not available, falling back to legacy")
                self.config = config
                self.use_new_config = False
        else:
            self.config = config
            self.logger.info("📄 CheckpointManager using legacy configuration system")
    
    def save_model_safely(self, trainer: Any, output_dir: str) -> bool:
        """
        Safely save model with proper HuggingFace compatibility.
        
        Args:
            trainer: The trainer instance
            output_dir: Directory to save the model
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"💾 Saving model to {output_dir}...")
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Use trainer's built-in save method
            trainer.save_model(output_dir)
            
            # Save additional components
            self._save_image_processor(output_dir)
            self._save_checkpoint_metadata(output_dir)
            
            self.logger.info(f"✅ Model saved successfully to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            return self._fallback_save_model(trainer, output_dir)
    
    def _fallback_save_model(self, trainer: Any, output_dir: str) -> bool:
        """
        Fallback method for saving model when standard approach fails.
        
        Args:
            trainer: The trainer instance
            output_dir: Directory to save the model
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.warning("🔄 Attempting fallback save method...")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(
                trainer.model.state_dict(), 
                output_path / "pytorch_model.bin"
            )
            
            # Save config
            trainer.model.config.save_pretrained(output_dir)
            
            # Save additional metadata
            self._save_checkpoint_metadata(output_dir)
            
            self.logger.info(f"✅ Model saved via fallback method to {output_dir}")
            return True
            
        except Exception as fallback_error:
            self.logger.error(f"❌ Fallback save also failed: {fallback_error}")
            return False
    
    def _save_image_processor(self, output_dir: str) -> None:
        """Save image processor to output directory."""
        try:
            model_path = (
                self.config.model.model_path if self.use_new_config
                else self.config.model_path
            )
            
            processor = AutoProcessor.from_pretrained(model_path)
            processor.image_processor.save_pretrained(output_dir)
            self.logger.info(f"💾 Image processor saved to: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save image processor: {e}")
    
    def _save_checkpoint_metadata(self, output_dir: str) -> None:
        """Save checkpoint metadata for tracking."""
        try:
            metadata = {
                'checkpoint_type': 'bbu_training',
                'config_system': 'new' if self.use_new_config else 'legacy',
                'model_architecture': 'Qwen2.5-VL-BBU',
                'training_framework': 'transformers_bbu_custom',
                'creation_timestamp': str(torch.cuda.Event().query() if torch.cuda.is_available() else 'cpu'),
            }
            
            # Add configuration info
            if self.use_new_config:
                metadata.update({
                    'model_path': self.config.model.model_path,
                    'detection_enabled': self.config.detection.detection_enabled,
                })
            else:
                metadata.update({
                    'model_path': self.config.model_path,
                    'detection_enabled': self.config.detection_enabled,
                })
            
            metadata_path = Path(output_dir) / "checkpoint_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"📋 Checkpoint metadata saved to: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save checkpoint metadata: {e}")
    
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
            self.logger.error(f"❌ Checkpoint directory does not exist: {checkpoint_path}")
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
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"📋 Checkpoint metadata: {metadata.get('checkpoint_type', 'unknown')}")
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
            'path': str(checkpoint_dir),
            'size_mb': sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()) / (1024 * 1024),
            'files': [f.name for f in checkpoint_dir.iterdir() if f.is_file()],
        }
        
        # Add metadata if available
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    info['metadata'] = json.load(f)
            except Exception:
                pass
        
        return info