import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from src.config.base import Config
from src.logger_utils import get_model_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype with explicit mapping."""
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    if dtype_str not in dtype_mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_mapping.keys())}"
        )

    return dtype_mapping[dtype_str]


class ModelWrapper:
    """Wrapper for Qwen2.5VL model with simplified device handling."""

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.logger = logger or get_model_logger()
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load_all(self):
        """Load model, tokenizer, and image processor."""
        self.logger.info(
            f"Loading model components for {self.config.model_size} model..."
        )

        # Apply comprehensive Qwen2.5-VL fixes
        self.logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
        if not apply_comprehensive_qwen25_fixes():
            raise RuntimeError("Failed to apply Qwen2.5-VL fixes")

        # Convert torch_dtype from config
        torch_dtype = _get_torch_dtype(self.config.torch_dtype)
        self.logger.info(f"🔧 Using torch_dtype: {torch_dtype}")

        # Load model
        self.logger.info(
            f"📥 Loading {self.config.model_size} model from {self.config.model_path}"
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            cache_dir=self.config.cache_dir,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=torch_dtype,
        )

        # Verify all patches
        self.logger.info("🔍 Verifying all patches...")
        if not verify_qwen25_patches():
            raise RuntimeError("Patch verification failed")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            cache_dir=self.config.cache_dir,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.logger.info(
            f"✅ Tokenizer loaded with max_length: {self.config.model_max_length}"
        )

        # Load image processor
        processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.image_processor = processor.image_processor

        # CRITICAL FIX: Use pixel constraints from data_conversion/vision_process.py
        # These values match exactly what was used during data preparation
        # Import the constants from data conversion
        import os
        import sys

        data_conversion_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data_conversion"
        )
        sys.path.insert(0, data_conversion_path)

        try:
            from vision_process import IMAGE_FACTOR, MAX_PIXELS, MIN_PIXELS

            # Apply the exact same pixel constraints used during data conversion
            self.image_processor.min_pixels = MIN_PIXELS  # 4 * 28 * 28 = 3136
            self.image_processor.max_pixels = MAX_PIXELS  # 128 * 28 * 28 = 100352

            # Also set size constraints if the processor supports them
            if hasattr(self.image_processor, "size"):
                if isinstance(self.image_processor.size, dict):
                    self.image_processor.size["min_pixels"] = MIN_PIXELS
                    self.image_processor.size["max_pixels"] = MAX_PIXELS
                else:
                    # Create size dict if it doesn't exist
                    self.image_processor.size = {
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    }

            self.logger.info(
                f"✅ Image processor configured with data_conversion/vision_process.py constants:"
            )
            self.logger.info(
                f"   min_pixels: {self.image_processor.min_pixels} (4 * 28 * 28)"
            )
            self.logger.info(
                f"   max_pixels: {self.image_processor.max_pixels} (128 * 28 * 28)"
            )
            self.logger.info(f"   image_factor: {IMAGE_FACTOR}")

        except ImportError as e:
            self.logger.error(
                f"❌ Failed to import from data_conversion/vision_process.py: {e}"
            )
            self.logger.error(
                "   Using fallback values from vision_process.py in project root"
            )

            # Fallback to the values from the project root vision_process.py
            from vision_process import MAX_PIXELS, MIN_PIXELS

            self.image_processor.min_pixels = MIN_PIXELS
            self.image_processor.max_pixels = MAX_PIXELS

        finally:
            # Clean up sys.path
            if data_conversion_path in sys.path:
                sys.path.remove(data_conversion_path)

        self.logger.info(
            f"   patch_size: {self.image_processor.patch_size if hasattr(self.image_processor, 'patch_size') else 'Not set'}"
        )
        self.logger.info(
            f"   merge_size: {self.image_processor.merge_size if hasattr(self.image_processor, 'merge_size') else 'Not set'}"
        )

        if hasattr(self.image_processor, "size"):
            self.logger.info(f"   size constraints: {self.image_processor.size}")

        # Configure training
        self._configure_training()

        self.logger.info(
            f"✅ All {self.config.model_size} components loaded successfully"
        )

        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"   Total parameters: {total_params:,}")
        self.logger.info(f"   Model dtype: {self.model.dtype}")
        self.logger.info(f"   Model device: {next(self.model.parameters()).device}")

        return self.model, self.tokenizer, self.image_processor

    def _configure_training(self):
        """Configure which components to train."""
        # Vision encoder
        for param in self.model.visual.named_parameters():
            param[1].requires_grad = self.config.tune_vision

        # Vision-language connector
        for param in self.model.visual.merger.named_parameters():
            param[1].requires_grad = self.config.tune_mlp

        # Language model
        for param in self.model.model.named_parameters():
            param[1].requires_grad = self.config.tune_llm
        self.model.lm_head.requires_grad = self.config.tune_llm

        # Print stats
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(
            f"Parameters: {trainable_params:,}/{total_params:,} trainable ({trainable_params / total_params * 100:.1f}%)"
        )

    def create_parameter_groups(self):
        """Create parameter groups with simplified learning rate configuration."""
        if not self.config.use_differential_lr:
            # All active modules use the same learning rate
            return [{"params": [p for p in self.model.parameters() if p.requires_grad]}]

        parameter_groups = []

        # Vision encoder parameters
        if self.config.tune_vision:
            vision_params = [
                p
                for name, p in self.model.visual.named_parameters()
                if p.requires_grad and "merger" not in name
            ]
            if vision_params:
                parameter_groups.append(
                    {
                        "params": vision_params,
                        "lr": self.config.vision_lr,
                        "name": "vision_encoder",
                    }
                )
                self.logger.info(
                    f"✅ Vision encoder: {len(vision_params)} parameters, lr={self.config.vision_lr}"
                )

        # Vision-language connector parameters
        if self.config.tune_mlp:
            mlp_params = [
                p
                for name, p in self.model.visual.merger.named_parameters()
                if p.requires_grad
            ]
            if mlp_params:
                parameter_groups.append(
                    {
                        "params": mlp_params,
                        "lr": self.config.mlp_lr,
                        "name": "vision_language_connector",
                    }
                )
                self.logger.info(
                    f"✅ Vision-language connector: {len(mlp_params)} parameters, lr={self.config.mlp_lr}"
                )

        # Language model parameters
        if self.config.tune_llm:
            llm_params = [
                p for name, p in self.model.model.named_parameters() if p.requires_grad
            ]
            # Add lm_head parameters
            if self.model.lm_head.requires_grad:
                llm_params.extend(
                    [p for p in self.model.lm_head.parameters() if p.requires_grad]
                )

            if llm_params:
                parameter_groups.append(
                    {
                        "params": llm_params,
                        "lr": self.config.llm_lr,
                        "name": "language_model",
                    }
                )
                self.logger.info(
                    f"✅ Language model: {len(llm_params)} parameters, lr={self.config.llm_lr}"
                )

        if not parameter_groups:
            raise ValueError("No modules enabled for training")

        # Log training configuration summary
        active_modules = []
        if self.config.tune_vision:
            active_modules.append(f"Vision({self.config.vision_lr})")
        if self.config.tune_mlp:
            active_modules.append(f"MLP({self.config.mlp_lr})")
        if self.config.tune_llm:
            active_modules.append(f"LLM({self.config.llm_lr})")

        self.logger.info(f"🎯 Training modules: {', '.join(active_modules)}")
        self.logger.info(f"📊 Created {len(parameter_groups)} parameter groups")

        return parameter_groups
