"""
Simplified Qwen2.5-VL Model Wrapper

This is a refactored version of the original wrapper.py that uses focused components
for better separation of concerns and maintainability.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger
from src.models.coordinate_handler import CoordinateHandler
from src.models.detection_integration import DetectionIntegration
from src.models.loss_manager import LossManager
from src.models.model_adapter import ModelAdapter
from src.models.patches import apply_comprehensive_qwen25_fixes


# Global storage for coordinate loss to bridge wrapper and training LossManager
_GLOBAL_COORDINATE_LOSS_STORAGE = {
    "last_coordinate_l1_loss": None,
    "last_step": None,
}


def get_global_coordinate_loss() -> Optional[float]:
    """Get the last coordinate loss from global storage."""
    return _GLOBAL_COORDINATE_LOSS_STORAGE.get("last_coordinate_l1_loss")


def clear_global_coordinate_loss():
    """Clear the global coordinate loss storage."""
    global _GLOBAL_COORDINATE_LOSS_STORAGE
    _GLOBAL_COORDINATE_LOSS_STORAGE["last_coordinate_l1_loss"] = None
    _GLOBAL_COORDINATE_LOSS_STORAGE["last_step"] = None


@dataclass
class CoordinateConfig:
    """Configuration for coordinate token extension."""

    max_coord_value: int = 1024
    coord_token_init_std: float = 0.02
    coordinate_loss_weight: float = 1.0
    regular_loss_weight: float = 1.0
    soft_expectation_temperature: float = 1.0
    enable_coordinate_tokens: bool = False
    coordinate_tokens_enabled: bool = False  # Alias for global config compatibility
    use_official_box_tokens: bool = (
        True  # Always use official <|box_start|> and <|box_end|>
    )

    # Multi-geometry extensions
    enable_multi_geometry: bool = False
    max_line_coordinates: int = 50


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16,
    }

    if dtype_str.lower() not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str.lower()]


class Qwen25VLWithDetection(nn.Module):
    """
    Simplified wrapper around official Qwen2.5-VL model with detection capabilities.

    This wrapper uses focused components for:
    - Coordinate token handling (CoordinateHandler)
    - Detection integration (DetectionIntegration)
    - Model adaptations (ModelAdapter)
    - Loss management (LossManager)
    """

    def __init__(
        self,
        base_model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer: PreTrainedTokenizerBase,
        attn_implementation: str = "",
        coordinate_config: Optional[CoordinateConfig] = None,
        config=None,
        use_cache: bool = False,
    ) -> None:
        super().__init__()

        # Strict validation
        if not base_model_path:
            raise ValueError("base_model_path cannot be empty")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if not hasattr(tokenizer, "get_vocab"):
            raise ValueError("tokenizer must have get_vocab method")
        if coordinate_config is not None and not isinstance(
            coordinate_config, CoordinateConfig
        ):
            raise ValueError("coordinate_config must be CoordinateConfig instance")

        # Store core attributes
        self.tokenizer = tokenizer
        self.logger = get_training_logger()
        self.num_queries = num_queries
        self.max_caption_length = max_caption_length
        self.use_cache = use_cache

        # Set up coordinate configuration
        self.coordinate_config = coordinate_config or CoordinateConfig()
        self.coordinate_tokens_enabled = self.coordinate_config.enable_coordinate_tokens

        # Initialize vocabulary sizes
        self.original_vocab_size = len(tokenizer.get_vocab())
        self.extended_vocab_size = self.original_vocab_size

        self.logger.info(
            f"🔧 Initializing Qwen2.5-VL wrapper with {self.original_vocab_size} vocab tokens"
        )

        # Load base model
        self.base_model = self._load_base_model(
            base_model_path, attn_implementation, config
        )

        # Initialize focused components
        self._initialize_components()

        # Set up coordinate tokens if enabled
        if self.coordinate_tokens_enabled:
            self._setup_coordinate_system()

        self.logger.info("✅ Qwen2.5-VL wrapper initialization complete")

    def _load_base_model(
        self, model_path: str, attn_implementation: str, config: Any
    ) -> Qwen2_5_VLForConditionalGeneration:
        """Load the base Qwen2.5-VL model."""
        self.logger.info(f"🔧 Loading base model from {model_path}")

        # Apply patches first
        apply_comprehensive_qwen25_fixes()

        # Load model - let HuggingFace load its own config, don't pass our BBUConfig
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=attn_implementation or "flash_attention_2",
            use_cache=self.use_cache,
            trust_remote_code=True,
        )

        self.logger.info("✅ Base model loaded successfully")
        return base_model

    def _initialize_components(self):
        """Initialize the focused components."""
        # Initialize coordinate handler
        self.coordinate_handler = CoordinateHandler(
            coordinate_config=self.coordinate_config,
            tokenizer=self.tokenizer,
            original_vocab_size=self.original_vocab_size,
            logger=self.logger,
        )

        # Initialize detection integration
        self.detection_integration = DetectionIntegration(
            coordinate_config=self.coordinate_config,
            coordinate_handler=self.coordinate_handler,
            logger=self.logger,
        )

        # Initialize model adapter
        self.model_adapter = ModelAdapter(
            base_model=self.base_model,
            coordinate_config=self.coordinate_config,
            tokenizer=self.tokenizer,
            original_vocab_size=self.original_vocab_size,
            extended_vocab_size=self.extended_vocab_size,  # Will be updated in setup
            logger=self.logger,
        )

        # Initialize loss manager (will be set up after coordinate system)
        self.loss_manager = None

    def _setup_coordinate_system(self):
        """Set up the coordinate token system using focused components."""
        self.logger.info("🔧 Setting up coordinate token system")

        # Set up coordinate tokens
        self.coordinate_handler.setup_coordinate_tokens(base_model=self.base_model)
        self.extended_vocab_size = self.coordinate_handler.extended_vocab_size

        # Update model adapter with new extended vocab size
        self.model_adapter.extended_vocab_size = self.extended_vocab_size

        # Create extended embeddings and LM head
        self.model_adapter.create_extended_embeddings()
        self.model_adapter.create_extended_lm_head()

        # Validate extensions
        self.model_adapter.validate_extensions()

        # Update coordinate handler with tokenizer tokens
        self.coordinate_handler.update_coordinate_manager_geometry_tokens(
            self.tokenizer
        )

        # Initialize loss manager with all components
        self.loss_manager = LossManager(
            coordinate_config=self.coordinate_config,
            coordinate_handler=self.coordinate_handler,
            detection_integration=self.detection_integration,
            logger=self.logger,
        )

        self.logger.info(
            f"✅ Coordinate system setup complete: {self.extended_vocab_size} total tokens"
        )

    def forward(self, **kwargs) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Forward pass with coordinate-aware processing."""
        # Reset loss components for this forward pass
        if self.loss_manager:
            self.loss_manager.reset_loss_components_for_forward_pass()

        # Handle coordinate token forward pass
        if self.coordinate_tokens_enabled and "labels" in kwargs:
            self.logger.debug(
                f"🎯 Taking coordinate token forward path (coordinate_tokens_enabled={self.coordinate_tokens_enabled}, has_labels={'labels' in kwargs})"
            )
            return self._forward_with_coordinate_tokens(**kwargs)
        else:
            self.logger.debug(
                f"🎯 Taking extended embeddings only path (coordinate_tokens_enabled={self.coordinate_tokens_enabled}, has_labels={'labels' in kwargs})"
            )
            return self._forward_with_extended_embeddings_only(**kwargs)

    def _forward_with_coordinate_tokens(
        self, **kwargs
    ) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Forward pass with coordinate token processing."""
        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels")

        if input_ids is None or labels is None:
            raise ValueError("input_ids and labels required for coordinate training")

        # Get coordinate mask
        coord_mask = self.coordinate_handler.get_coordinate_mask(input_ids)
        self.logger.debug(
            f"🎯 Coordinate mask: {coord_mask.sum().item()} coordinate tokens detected out of {input_ids.numel()} total tokens"
        )

        # Standard model forward pass
        outputs = self.base_model(**kwargs)

        # Compute coordinate-aware loss if labels provided
        if labels is not None:
            logits = outputs.logits
            device = logits.device

            # Use loss manager for coordinate-aware loss computation
            if self.loss_manager:
                total_loss = self.loss_manager.compute_coordinate_aware_loss(
                    logits, labels, coord_mask
                )

                # Update outputs with computed loss
                outputs.loss = total_loss

                # Store coordinate loss in a way that can be accessed by training LossManager
                # Since dataclass doesn't support dynamic attributes, we'll use a different approach
                coordinate_l1_loss = self.loss_manager._last_coordinate_l1_loss

                # Create a new outputs object with the coordinate loss included
                # We'll modify the outputs object to include coordinate loss as a custom attribute
                try:
                    # Try to set the attribute directly (may work for some output types)
                    outputs._coordinate_l1_loss = torch.tensor(
                        coordinate_l1_loss, device=device
                    )
                    self.logger.debug(
                        f"🎯 Successfully attached _coordinate_l1_loss to outputs: {coordinate_l1_loss}"
                    )
                except (AttributeError, TypeError):
                    # If direct assignment fails, store in a global registry
                    self.logger.debug(
                        f"🎯 Direct attribute assignment failed, using global storage for coordinate loss: {coordinate_l1_loss}"
                    )
                    # Store in wrapper for access by training manager
                    self._last_coordinate_l1_loss_for_training = coordinate_l1_loss

                # Always store in global storage as a fallback
                global _GLOBAL_COORDINATE_LOSS_STORAGE
                _GLOBAL_COORDINATE_LOSS_STORAGE["last_coordinate_l1_loss"] = (
                    coordinate_l1_loss
                )
                _GLOBAL_COORDINATE_LOSS_STORAGE["last_step"] = getattr(
                    self, "_current_step", 0
                )
                self.logger.debug(
                    f"🎯 Stored coordinate loss in global storage: {coordinate_l1_loss}"
                )

        return outputs

    def _forward_with_extended_embeddings_only(
        self, **kwargs
    ) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Forward pass without coordinate token processing."""
        # Reset loss components
        if self.loss_manager:
            self.loss_manager.reset_loss_components_to_zero()

        # Standard forward pass
        outputs = self.base_model(**kwargs)

        # Attach zero coordinate losses if loss manager exists
        if self.loss_manager:
            self.loss_manager.attach_coordinate_losses_to_outputs(outputs)

        return outputs

    def generate(self, **kwargs):
        """Delegate generation to base model."""
        return self.base_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, **kwargs):
        """Delegate input preparation to base model."""
        return self.base_model.prepare_inputs_for_generation(**kwargs)

    def get_rope_index(self, **kwargs):
        """Delegate rope index to base model."""
        return self.base_model.get_rope_index(**kwargs)

    @property
    def device(self):
        """Get device of the model."""
        return next(self.base_model.parameters()).device

    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        self.base_model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.base_model.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_queries: int = 100,
        max_caption_length: int = 256,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        coordinate_config: Optional[CoordinateConfig] = None,
        attn_implementation: str = "",
        config=None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Load pre-trained model with coordinate extensions."""
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)

        return cls(
            base_model_path=model_path,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
            coordinate_config=coordinate_config,
            attn_implementation=attn_implementation,
            config=config,
            use_cache=use_cache,
            **kwargs,
        )

    @property
    def config(self):
        """Get model config."""
        if hasattr(self.base_model, "config"):
            config = self.base_model.config

            # Update vocab size if extended
            if self.coordinate_tokens_enabled and self.extended_vocab_size:
                config = config.__class__(**config.to_dict())
                config.vocab_size = self.extended_vocab_size

            return config
        return None

    def get_last_coordinate_losses(self) -> Dict[str, torch.Tensor]:
        """Get last coordinate losses."""
        if self.loss_manager:
            return self.loss_manager.get_last_coordinate_losses()
        else:
            device = self.device
            return {
                "_llm_loss": torch.tensor(0.0, device=device),
                "_coordinate_l1_loss": torch.tensor(0.0, device=device),
            }

    def get_coordinate_tokenizer_utils(self) -> Dict[str, Any]:
        """Get coordinate tokenizer utilities."""
        if self.coordinate_handler:
            return self.coordinate_handler.get_coordinate_tokenizer_utils()
        else:
            return {}

    def validate_sample_tokens(self, input_ids, sample_info=None):
        """Validate coordinate tokens in sample."""
        if self.coordinate_handler:
            self.coordinate_handler.validate_coordinate_tokens(input_ids, sample_info)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model with coordinate extensions."""
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save coordinate extensions if enabled
        if self.coordinate_tokens_enabled and self.model_adapter:
            extension_metadata = self.model_adapter.get_extension_metadata()
            extension_state = self.model_adapter.get_extension_state_dict()

            # Save extension data
            coord_save_path = os.path.join(save_directory, "coordinate_extensions.pt")
            torch.save(
                {
                    "metadata": extension_metadata,
                    "state_dict": extension_state,
                },
                coord_save_path,
            )

            self.logger.info(f"💾 Coordinate extensions saved to {coord_save_path}")

    def get_input_embeddings(self):
        """Get input embeddings (extended if coordinate tokens enabled)."""
        if self.coordinate_tokens_enabled and self.model_adapter.extended_embeddings:
            return self.model_adapter.extended_embeddings
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        """Get output embeddings (extended if coordinate tokens enabled)."""
        if self.coordinate_tokens_enabled and self.model_adapter.extended_lm_head:
            return self.model_adapter.extended_lm_head
        return self.base_model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings to accommodate new tokens."""
        self.logger.info(
            f"🔧 Wrapper delegating resize_token_embeddings: {new_num_tokens} tokens"
        )

        # Delegate to base model
        result = self.base_model.resize_token_embeddings(new_num_tokens)

        # Update our vocabulary sizes
        self.extended_vocab_size = new_num_tokens

        self.logger.info(f"✅ Wrapper embeddings resized to {new_num_tokens}")
        return result


# Dummy classes for compatibility
class DummyOptim:
    """Dummy optimizer for compatibility."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]

    def zero_grad(self):
        pass

    def step(self):
        pass


class DummyScheduler:
    """Dummy scheduler for compatibility."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1

    def get_last_lr(self):
        return [0.001]
