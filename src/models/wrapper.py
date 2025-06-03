import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from src.config.base import Config
from src.logging import get_model_logger
from src.models.patches import apply_smart_mrope_fix, verify_smart_mrope_patch


def convert_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string torch_dtype to actual torch.dtype."""
    if isinstance(dtype_str, torch.dtype):
        return dtype_str

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    if dtype_str in dtype_mapping:
        return dtype_mapping[dtype_str]
    else:
        # Try to get from torch directly
        if hasattr(torch, dtype_str):
            return getattr(torch, dtype_str)
        else:
            raise ValueError(f"Unsupported torch_dtype: {dtype_str}")


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

        # Apply the smart mRoPE fix
        self.logger.info("🔧 Applying smart mRoPE fix...")
        if not apply_smart_mrope_fix():
            raise RuntimeError("Failed to apply smart mRoPE fix")

        # Convert torch_dtype from config
        torch_dtype = convert_torch_dtype(self.config.torch_dtype)
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
        self.model.config.use_cache = False

        # Verify mRoPE patch
        self.logger.info("🔍 Verifying mRoPE patch...")
        if not verify_smart_mrope_patch():
            raise RuntimeError("mRoPE patch verification failed")

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
        self.image_processor.max_pixels = self.config.max_pixels
        self.image_processor.min_pixels = self.config.min_pixels

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
