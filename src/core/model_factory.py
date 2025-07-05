"""
Model Factory for BBU Training System

Centralized model creation, configuration, and initialization.
Extracted from trainer_factory.py for better separation of concerns.

Key Features:
- Unified model creation interface
- Support for both legacy and new config systems  
- Automatic patch application and optimization setup
- Detection head configuration
- Hardware-specific optimizations (Flash Attention, dtype handling)
"""

from typing import Any, Tuple, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoProcessor

from src.config import config, get_config_manager
from src.logger_utils import get_training_logger
from src.models.wrapper import Qwen25VLWithDetection
from src.models.patches import apply_comprehensive_qwen25_fixes


class ModelFactory:
    """Factory class for creating and configuring models."""
    
    def __init__(self, use_new_config: bool = False):
        """
        Initialize model factory.
        
        Args:
            use_new_config: Whether to use new domain-specific config system
        """
        self.use_new_config = use_new_config
        self.logger = get_training_logger()
        
        # Get configuration
        if use_new_config:
            try:
                self.config = get_config_manager()
                self.logger.info("✅ ModelFactory using new domain-specific configuration")
            except RuntimeError:
                self.logger.warning("⚠️  New config system not available, falling back to legacy")
                self.config = config
                self.use_new_config = False
        else:
            self.config = config
            self.logger.info("📄 ModelFactory using legacy configuration system")
    
    def create_model(self) -> Qwen25VLWithDetection:
        """
        Create and configure the main model.
        
        Returns:
            Configured Qwen25VLWithDetection instance
        """
        self.logger.info("🤖 Creating model...")
        
        # Get model configuration
        model_config = self._get_model_config()
        detection_config = self._get_detection_config()
        
        # Create model
        model = Qwen25VLWithDetection.from_pretrained(
            model_config['model_path'],
            torch_dtype=model_config['torch_dtype'],
            attn_implementation=model_config['attn_implementation'],
            detection_enabled=detection_config['detection_enabled'],
            **detection_config['detection_params']
        )
        
        # Apply patches and optimizations
        self._apply_model_optimizations(model)
        
        # Disable caching during training
        model.config.use_cache = False
        
        self.logger.info(f"✅ Model created: {model_config['model_path']}")
        return model
    
    def create_tokenizer_and_processor(self) -> Tuple[Any, Any]:
        """
        Create tokenizer and image processor.
        
        Returns:
            Tuple of (tokenizer, image_processor)
        """
        self.logger.info("🔤 Creating tokenizer and processor...")
        
        model_path = self._get_model_path()
        
        # Create processor (includes both tokenizer and image processor)
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        
        # CRITICAL: Set padding_side='left' for Flash Attention compatibility
        # Qwen2.5-VL requires left padding for Flash Attention to work correctly
        tokenizer.padding_side = 'left'
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.logger.info("✅ Tokenizer and processor created")
        return tokenizer, image_processor
    
    def _get_model_config(self) -> dict:
        """Get model configuration parameters."""
        if self.use_new_config:
            return {
                'model_path': self.config.model.model_path,
                'torch_dtype': self._parse_dtype(self.config.model.torch_dtype),
                'attn_implementation': self.config.model.attn_implementation,
            }
        else:
            return {
                'model_path': self.config.model_path,
                'torch_dtype': self._parse_dtype(self.config.torch_dtype),
                'attn_implementation': self.config.attn_implementation,
            }
    
    def _get_detection_config(self) -> dict:
        """Get detection head configuration parameters."""
        if self.use_new_config:
            return {
                'detection_enabled': self.config.detection.detection_enabled,
                'detection_params': {
                    'num_queries': self.config.detection.detection_num_queries,
                    'max_caption_length': self.config.detection.detection_max_caption_length,
                }
            }
        else:
            return {
                'detection_enabled': self.config.detection_enabled,
                'detection_params': {
                    'num_queries': self.config.detection_num_queries,
                    'max_caption_length': self.config.detection_max_caption_length,
                }
            }
    
    def _get_model_path(self) -> str:
        """Get model path from configuration."""
        if self.use_new_config:
            return self.config.model.model_path
        else:
            return self.config.model_path
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse string dtype to torch dtype."""
        if dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float16":
            return torch.float16
        else:
            return torch.float32
    
    def _apply_model_optimizations(self, model: Qwen25VLWithDetection) -> None:
        """Apply patches and optimizations to the model."""
        self.logger.info("🔧 Applying model patches and optimizations...")
        
        # Apply comprehensive fixes
        apply_comprehensive_qwen25_fixes()
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 Model Statistics:")
        self.logger.info(f"   Total parameters: {total_params:,}")
        self.logger.info(f"   Trainable parameters: {trainable_params:,}")
        self.logger.info(f"   Trainable ratio: {trainable_params/total_params:.2%}")
        
    @classmethod
    def create_model_and_processors(cls, use_new_config: bool = False) -> Tuple[Qwen25VLWithDetection, Any, Any]:
        """
        Convenience method to create model, tokenizer, and processor in one call.
        
        Args:
            use_new_config: Whether to use new domain-specific config system
            
        Returns:
            Tuple of (model, tokenizer, image_processor)
        """
        factory = cls(use_new_config=use_new_config)
        model = factory.create_model()
        tokenizer, image_processor = factory.create_tokenizer_and_processor()
        return model, tokenizer, image_processor