"""
Qwen2.5-VL Model Wrapper with Detection Capabilities

This module provides a wrapper around the official Qwen2.5-VL model
that adds object detection capabilities while preserving all original functionality.
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

from src.config import config


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16,  # Default to bfloat16 for auto
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


class Qwen25VLWithDetection(nn.Module):
    """
    Wrapper around official Qwen2.5-VL model with detection capabilities.

    This wrapper adds a detection head while preserving all original functionality
    of the Qwen2.5-VL model for generation tasks.

    SIMPLIFIED: Only supports loading from official model path with randomly initialized detection head.
    """

    def __init__(
        self,
        base_model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
    ):
        super().__init__()

        # Store tokenizer for detection head initialization
        self.tokenizer = tokenizer

        # Load official Qwen2.5-VL model with proper configuration
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=_get_torch_dtype(config.torch_dtype),
            attn_implementation=config.attn_implementation,
        )

        # Initialize detection head if enabled
        self.detection_head = None
        if config.detection_enabled:
            self.detection_enabled = True  # Set flag for forward method
            self._init_detection_head()
        else:
            self.detection_enabled = False

        # Share token embedding from base model
        if self.detection_head is not None:
            self.detection_head.set_token_embedding(
                self.base_model.get_input_embeddings()
            )

        # Store our custom config for internal use, but expose base model config for DeepSpeed
        self._custom_config = config

        # Move detection head to same device as base model
        device = next(self.base_model.parameters()).device

        # Move to device (dtype already set during initialization)
        if self.detection_head is not None:
            self.detection_head = self.detection_head.to(device=device)

    def _init_detection_head(self):
        """Initialize the detection head with proper configuration."""
        from src.config import config
        from src.models.detection_head import DetectionHead

        # Get configuration - use the correct attribute names from base_flat.yaml
        num_queries = config.detection_num_queries
        max_caption_length = config.detection_max_caption_length
        # Use the tokenizer stored in the instance instead of config.tokenizer
        tokenizer = self.tokenizer
        target_dtype = _get_torch_dtype(config.torch_dtype)

        # Add detection head using official config with correct dtype (randomly initialized)
        self.detection_head = DetectionHead(
            config=self.base_model.config,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
            detection_decoder_nhead=config.detection_decoder_nhead,
            detection_decoder_dim_feedforward_factor=config.detection_decoder_dim_feedforward_factor,
            detection_decoder_num_layers=config.detection_decoder_num_layers,
            detection_caption_decoder_nhead=config.detection_caption_decoder_nhead,
            detection_caption_decoder_dim_feedforward_factor=config.detection_caption_decoder_dim_feedforward_factor,
            detection_caption_decoder_num_layers=config.detection_caption_decoder_num_layers,
            detection_head_dropout=config.detection_head_dropout,
            dtype=target_dtype,
        )

        # Share token embedding from base model
        self.detection_head.set_token_embedding(self.base_model.get_input_embeddings())

    def forward(self, **inputs):
        """
        Forward pass that preserves all functionality and returns combined loss during training.
        """

        # Store original ground truth objects for detection loss (don't pop them)
        # The trainer will handle detection loss computation

        # Remove ground truth objects from model inputs (but keep them in original inputs)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)

        # Standard Qwen2.5-VL forward pass with all parameters preserved
        outputs = self.base_model(**model_inputs)

        # The trainer will handle detection loss computation, so we just return the base model outputs
        # This ensures no duplicate loss computation
        return outputs

    def generate(self, **kwargs):
        """Disable detection during generation to maintain compatibility"""
        old_detection_enabled = self.detection_enabled
        self.detection_enabled = False
        try:
            return self.base_model.generate(**kwargs)
        finally:
            self.detection_enabled = old_detection_enabled

    def prepare_inputs_for_generation(self, **kwargs):
        """Delegate to base model's preparation method"""
        return self.base_model.prepare_inputs_for_generation(**kwargs)

    def get_rope_index(self, **kwargs):
        """Delegate to base model's RoPE calculation"""
        return self.base_model.get_rope_index(**kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        """Delegate to base model for token embedding resizing"""
        return self.base_model.resize_token_embeddings(new_num_tokens)

    @property
    def device(self):
        """Return the device of the base model"""
        return next(self.base_model.parameters()).device

    def train(self, mode=True):
        """Override train mode to handle both base model and detection head"""
        super().train(mode)
        self.base_model.train(mode)
        self.detection_head.train(mode)
        return self

    def eval(self):
        """Override eval mode to handle both base model and detection head"""
        super().eval()
        self.base_model.eval()
        self.detection_head.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_disable()

    def load_detection_head_weights(self, detection_head_path: str):
        """
        Load detection head weights from a saved checkpoint.

        Args:
            detection_head_path: Path to the detection head weights file (.pth)
        """
        import os

        import torch

        if not os.path.exists(detection_head_path):
            raise FileNotFoundError(
                f"Detection head weights not found: {detection_head_path}"
            )

        # Load detection head state dict
        detection_state_dict = torch.load(detection_head_path, map_location=self.device)

        # Load weights into detection head
        self.detection_head.load_state_dict(detection_state_dict)

        print(f"✅ Detection head weights loaded from: {detection_head_path}")

    def save_detection_head_weights(self, output_dir: str):
        """
        Save detection head weights to a directory.

        Args:
            output_dir: Directory to save detection head weights
        """
        import json
        import os

        import torch

        os.makedirs(output_dir, exist_ok=True)

        # Save detection head weights
        detection_state_dict = self.detection_head.state_dict()
        detection_path = os.path.join(output_dir, "detection_head.pth")
        torch.save(detection_state_dict, detection_path)

        # Save detection head config with UNIFIED filename (same as trainer)
        detection_config = {
            "num_queries": self.detection_head.num_queries,
            "max_caption_length": self.detection_head.max_caption_length,
            "hidden_size": self.detection_head.hidden_size,
            "vocab_size": self.detection_head.vocab_size,
            "detection_enabled": True,
            "checkpoint_type": "unified",  # Marker for unified checkpoint
        }

        # Use the UNIFIED config filename (same as trainer)
        config_path = os.path.join(output_dir, "detection_config.json")
        with open(config_path, "w") as f:
            json.dump(detection_config, f, indent=2)

        print(f"✅ Detection head weights saved to: {detection_path}")
        print(f"✅ Detection head config saved to: {config_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_queries: int = None,
        max_caption_length: int = None,
        tokenizer=None,
        **kwargs,
    ):
        """
        Unified loading method that automatically detects checkpoint type.

        This method works with:
        1. Base Qwen2.5-VL models (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        2. Unified checkpoints with detection head (created by our trainer)

        Args:
            model_path: Path to model (base model or checkpoint directory)
            num_queries: Number of detection queries (auto-detected from checkpoint)
            max_caption_length: Max caption length (auto-detected from checkpoint)
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments

        Returns:
            Qwen25VLWithDetection: Model with appropriate weights loaded
        """

        # Check what type of checkpoint this is
        checkpoint_info = cls._analyze_checkpoint(model_path)

        if checkpoint_info["type"] == "unified":
            return cls._load_unified_checkpoint(
                model_path, num_queries, max_caption_length, tokenizer, **kwargs
            )
        else:  # base model
            return cls._load_base_model(
                model_path, num_queries, max_caption_length, tokenizer, **kwargs
            )

    @classmethod
    def _analyze_checkpoint(cls, model_path: str) -> dict:
        """Analyze checkpoint to determine its type and available components."""
        import json
        import os

        # Check for unified checkpoint markers
        detection_config_path = os.path.join(model_path, "detection_config.json")
        detection_head_path = os.path.join(model_path, "detection_head.pth")

        if os.path.exists(detection_config_path) and os.path.exists(
            detection_head_path
        ):
            # Load detection config to get parameters
            with open(detection_config_path, "r") as f:
                detection_config = json.load(f)

            return {
                "type": "unified",
                "has_detection": True,
                "detection_config": detection_config,
                "detection_head_path": detection_head_path,
            }
        else:
            # Base model without detection head
            return {
                "type": "base",
                "has_detection": False,
            }

    @classmethod
    def _load_unified_checkpoint(
        cls,
        model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        **kwargs,
    ):
        """Load from unified checkpoint created by our trainer."""
        checkpoint_info = cls._analyze_checkpoint(model_path)
        detection_config = checkpoint_info["detection_config"]

        # Use config values if not explicitly provided
        if num_queries is None:
            num_queries = detection_config.get("num_queries", 100)
        if max_caption_length is None:
            max_caption_length = detection_config.get("max_caption_length", 32)

        print(f"🔄 Loading unified checkpoint from: {model_path}")
        print(f"   Detection queries: {num_queries}")
        print(f"   Max caption length: {max_caption_length}")

        # Create model with checkpoint as base model path
        model = cls(
            base_model_path=model_path,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
        )

        # Load detection head weights
        model.load_detection_head_weights(checkpoint_info["detection_head_path"])

        print(f"✅ Unified checkpoint loaded successfully")
        return model

    @classmethod
    def _load_base_model(
        cls,
        model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        **kwargs,
    ):
        """Load base model without detection head (randomly initialized)."""
        # Use defaults if not provided
        if num_queries is None:
            num_queries = 100
        if max_caption_length is None:
            max_caption_length = 32

        print(f"🔄 Loading base model from: {model_path}")
        print(f"   Detection head will be randomly initialized")
        print(f"   Detection queries: {num_queries}")
        print(f"   Max caption length: {max_caption_length}")

        # Create model with randomly initialized detection head
        model = cls(
            base_model_path=model_path,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
        )

        print(f"✅ Base model loaded with random detection head")
        return model

    @staticmethod
    def inspect_checkpoint(model_path: str) -> dict:
        """
        Inspect a checkpoint to understand its type and contents.

        This is a utility function to help users understand what type of
        checkpoint they have without loading the full model.

        Args:
            model_path: Path to model or checkpoint directory

        Returns:
            dict: Information about the checkpoint
        """
        import os

        checkpoint_info = Qwen25VLWithDetection._analyze_checkpoint(model_path)

        # Add more detailed information
        result = {
            "path": model_path,
            "type": checkpoint_info["type"],
            "has_detection_head": checkpoint_info["has_detection"],
            "description": "",
            "files_found": [],
        }

        # Check what files exist
        common_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "detection_head.pth",
            "detection_config.json",
        ]

        for file in common_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                result["files_found"].append(file)

        # Add descriptions
        if checkpoint_info["type"] == "unified":
            result["description"] = (
                "Unified checkpoint with both base model and detection head"
            )
            if "detection_config" in checkpoint_info:
                result["detection_config"] = checkpoint_info["detection_config"]
        else:
            result["description"] = "Base Qwen2.5-VL model without detection head"

        return result

    @staticmethod
    def print_checkpoint_info(model_path: str):
        """
        Print human-readable information about a checkpoint.

        Args:
            model_path: Path to model or checkpoint directory
        """
        info = Qwen25VLWithDetection.inspect_checkpoint(model_path)

        print(f"📁 Checkpoint Analysis: {model_path}")
        print(f"   Type: {info['type'].upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Has Detection Head: {'✅' if info['has_detection_head'] else '❌'}")

        if info["files_found"]:
            print(f"   Files Found:")
            for file in info["files_found"]:
                print(f"     - {file}")

        if "detection_config" in info:
            config = info["detection_config"]
            print(f"   Detection Config:")
            print(f"     - Queries: {config.get('num_queries', 'N/A')}")
            print(
                f"     - Max Caption Length: {config.get('max_caption_length', 'N/A')}"
            )
            print(f"     - Hidden Size: {config.get('hidden_size', 'N/A')}")
            print(f"     - Vocab Size: {config.get('vocab_size', 'N/A')}")

        print()

    def set_detection_loss_fn(self, detection_loss_fn):
        """Set the detection loss function for use in forward pass."""
        self._detection_loss_fn = detection_loss_fn

    @property
    def config(self):
        """Return the base model's config for DeepSpeed compatibility"""
        return self.base_model.config
