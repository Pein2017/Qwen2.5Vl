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

from typing import Any, Tuple

from transformers import AutoProcessor

from src.logger_utils import get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes
from src.models.wrapper import Qwen25VLWithDetection


class ModelFactory:
    """Factory class for creating and configuring models."""

    def __init__(self, config_obj=None):
        """
        Initialize model factory.

        Args:
            config_obj: Configuration object (required)
        """
        self.logger = get_training_logger()

        if config_obj is None:
            from src.config import config

            self.config = config
        else:
            self.config = config_obj

        self.logger.info("🏭 ModelFactory initialized")

    def create_model(self) -> Qwen25VLWithDetection:
        """
        Create and configure the main model.

        Returns:
            Configured Qwen25VLWithDetection instance
        """
        from src.models.wrapper import CoordinateConfig

        self.logger.info("🤖 Creating model...")

        # Strict validation - fail fast
        if not hasattr(self.config, "model_path"):
            raise ValueError("model_path not found in config")
        if not self.config.model_path:
            raise ValueError("model_path cannot be empty")

        import os

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(
                f"Model path does not exist: {self.config.model_path}"
            )

        # Create coordinate config if coordinate tokens are enabled
        coordinate_config = None
        coordinate_tokens_enabled = getattr(
            self.config, "coordinate_tokens_enabled", False
        )

        if coordinate_tokens_enabled:
            self.logger.info(
                "🚀 Coordinate tokens enabled - creating coordinate config"
            )

            # Validate coordinate token configuration
            max_coord_value = getattr(
                self.config, "coordinate_config_max_coord_value", 2048
            )
            coordinate_loss_weight = getattr(
                self.config, "coordinate_config_coordinate_loss_weight", 1.0
            )
            regular_loss_weight = getattr(
                self.config, "coordinate_config_regular_loss_weight", 1.0
            )
            temperature = getattr(
                self.config, "coordinate_config_soft_expectation_temperature", 1.0
            )

            self.logger.info(f"   📊 Coordinate config:")
            self.logger.info(f"      Max coord value: {max_coord_value}")
            self.logger.info(f"      Coordinate loss weight: {coordinate_loss_weight}")
            self.logger.info(f"      Regular loss weight: {regular_loss_weight}")
            self.logger.info(f"      Temperature: {temperature}")

            coordinate_config = CoordinateConfig(
                enable_coordinate_tokens=True,
                max_coord_value=max_coord_value,
                coordinate_loss_weight=coordinate_loss_weight,
                regular_loss_weight=regular_loss_weight,
                soft_expectation_temperature=temperature,
            )
        else:
            self.logger.info("📄 Coordinate tokens disabled - using standard model")

        # Create tokenizer first (required)
        tokenizer, _ = self.create_tokenizer_and_processor()

        # Create model
        model = Qwen25VLWithDetection.from_pretrained(
            model_path=self.config.model_path,
            tokenizer=tokenizer,
            coordinate_config=coordinate_config,
            attn_implementation=getattr(
                self.config, "attn_implementation", "flash_attention_2"
            ),
        )

        # Apply optimizations
        if getattr(self.config, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            self.logger.info("✅ Gradient checkpointing enabled")

        # Disable caching during training
        model.config.use_cache = False

        self.logger.info(f"✅ Model created: {self.config.model_path}")
        return model

    def create_tokenizer_and_processor(self) -> Tuple[Any, Any]:
        """
        Create tokenizer and image processor.

        Returns:
            Tuple of (tokenizer, image_processor)
        """
        self.logger.info("🔤 Creating tokenizer and processor...")

        model_path = self.config.model_path

        # Create processor (includes both tokenizer and image processor)
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor

        # CRITICAL: Set padding_side='left' for Flash Attention compatibility
        # Qwen2.5-VL requires left padding for Flash Attention to work correctly
        if tokenizer.padding_side != 'left':
            self.logger.warning(f"🔧 Fixing tokenizer padding_side: {tokenizer.padding_side} -> left")
        tokenizer.padding_side = "left"
        self.logger.info(f"[PADDING_SIDE_CHECK] Tokenizer padding side set to: {tokenizer.padding_side}")

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.logger.info("✅ Tokenizer and processor created")
        return tokenizer, image_processor

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
        self.logger.info(f"   Trainable ratio: {trainable_params / total_params:.2%}")

    @classmethod
    def create_model_and_processors(
        cls, use_new_config: bool = False
    ) -> Tuple[Qwen25VLWithDetection, Any, Any]:
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
