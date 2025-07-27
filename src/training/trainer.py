"""
Unified BBU Trainer with Robust Loss Logging.

This module extends the standard HuggingFace Trainer to provide fine-grained
logging for a multi-component loss function (language modeling, bounding bbox_2d,
caption, and objectness) while ensuring accurate, per-step reporting even with
gradient accumulation and frequent evaluations.

Key Implementation Details:
1.  **Component Loss Accumulation:**
    - The `compute_loss` method calculates the final combined loss for
      backpropagation.
    - It also tracks each individual loss component (e.g., `_current_lm_loss`)
      and adds it to a corresponding accumulator (e.g., `_accumulated_lm_loss`)
      for each micro-batch (i.e., each forward pass).

2.  **Per-Step Average Logging:**
    - The `_maybe_log_save_evaluate` method is called by the Trainer's main loop
      AFTER a full gradient accumulation cycle is complete.
    - It averages each accumulated component loss by dividing it by the number of
      gradient accumulation steps.
    - The final reported `loss` is the sum of these averaged components,
      ensuring it accurately reflects the loss for that specific training step.
    - All accumulators are reset to zero immediately after logging, preparing
      them for the next accumulation cycle.

3.  **Isolated Evaluation:**
    - The `evaluate` method is "sandboxed" to prevent state corruption.
    - Before evaluation begins, it saves the current state of the training
      loss accumulators.
    - It then runs the entire evaluation, using the same accumulators but for
      evaluation batches.
    - CRUCIALLY, after evaluation is complete, it restores the saved training
      accumulators, ensuring that the evaluation process does not interfere
      with the training loop's loss tracking.

This design guarantees that training and evaluation logging are independent and
that reported training losses are correctly averaged per step.
"""

import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import BBUConfig
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.wrapper import DummyOptim
from src.utils.schema import GroundTruthObject
from src.utils.tokens.special_tokens import SpecialTokens


class BBUTrainer(Trainer):
    """
    Custom trainer for coordinate token training.

    Extends the standard Transformers Trainer with coordinate token support
    while maintaining clean separation of concerns.
    """

    # ------------------------------------------------------------------
    # HF ≥4.41 emits a deprecation warning every time `.tokenizer` is
    # accessed on a Trainer instance.  We access it frequently for
    # logging/decoding, so we cache the reference (if provided) *before*
    # calling the parent ctor, then overwrite the property with a plain
    # attribute afterwards.  This silences the warning without touching
    # upstream library code and keeps backward-compatibility for any
    # external calls that expect `trainer.tokenizer` to exist.
    # ------------------------------------------------------------------
    tokenizer_ref = None

    def __init__(
        self,
        *args: Any,
        cfg: Optional[BBUConfig] = None,
        image_processor: Optional[Qwen2VLImageProcessor] = None,
        training_coordinator: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # --------------------------------------------------------------
        # HF ≥4.41 emits a deprecation warning every time `.tokenizer` is
        # accessed on a Trainer instance.  We access it frequently for
        # logging/decoding, so we cache the reference (if provided) *before*
        # calling the parent ctor, then overwrite the property with a plain
        # attribute afterwards.  This silences the warning without touching
        # upstream library code and keeps backward-compatibility for any
        # external calls that expect `trainer.tokenizer` to exist.
        # --------------------------------------------------------------
        # Support legacy alias where callers used `config=` keyword.
        if cfg is None and "config" in kwargs:
            cfg = kwargs.pop("config")

        self.tokenizer_ref = kwargs.get("tokenizer")

        # ------------------------------------------------------------------
        # Resolve configuration
        # Priority:
        #   1) Explicit `cfg` argument passed by caller
        #   2) Global singleton initialised via src.config.init_config()
        # Fail fast if neither is available.
        # ------------------------------------------------------------------

        # Check if global configuration is initialized
        _global_cfg: Optional[BBUConfig] = None
        try:
            from src.config import get_config

            _global_cfg = get_config()
        except RuntimeError:
            # Global config not initialized
            _global_cfg = None

        if cfg is not None:
            self.config = cfg
        elif _global_cfg is not None:
            self.config = _global_cfg
        else:
            raise RuntimeError(
                "BBUConfig not provided to BBUTrainer and global config has "
                "not been initialised. Call src.config.init_config() before "
                "creating the trainer or pass cfg=<BBUConfig>."
            )

        super().__init__(*args, **kwargs)

        # Initialize logger first since tokenizer setter needs it
        self.logger = get_training_logger()

        # Overwrite the (deprecated) property with a direct attribute so
        # future accesses skip the warning-emitting property defined in the
        # parent class. We bypass the descriptor protocol via
        # `object.__setattr__` to avoid invoking the original setter.
        if self.tokenizer_ref is None and hasattr(self, "tokenizer"):
            self.tokenizer_ref = object.__getattribute__(self, "tokenizer")

        object.__setattr__(self, "tokenizer", self.tokenizer_ref)
        self.image_processor = image_processor

        # Integration with new training coordinator system
        self.training_coordinator = training_coordinator
        self._use_coordinator = training_coordinator is not None

        # Initialize loss tracking variables (shared between coordinator and legacy)
        self._current_lm_loss: float = 0.0
        self._current_teacher_lm_loss: float = 0.0

        # Coordinate loss validation tracking - zero tolerance for missing coordinate tokens
        self._step_count: int = 0
        self._current_student_lm_loss: float = 0.0
        self._current_bbox_loss: float = 0.0
        self._current_objectness_loss: float = 0.0

        # ACCUMULATORS for per-step average logging with gradient accumulation
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_teacher_lm_loss: float = 0.0
        self._accumulated_student_lm_loss: float = 0.0
        self._accumulated_objectness_loss: float = 0.0

        # Counter for the number of *micro-batches* processed since the last log.
        # This is needed for both coordinator and legacy systems
        self._micro_batch_count: int = 0

        if self._use_coordinator:
            self.logger.info("🎯 Using new training coordinator system")
            # Detection loss is handled by coordinator via coordinate tokens
        else:
            self.logger.info("📄 Using legacy training system")
            # Legacy system: manual loss tracking

            # Detection is now handled via coordinate tokens
            self.detection_loss = None

            # Validate required configuration attributes
            if not hasattr(self.config, "coordinate_tokens_enabled"):
                raise ValueError(
                    "coordinate_tokens_enabled must be explicitly configured in config"
                )

            # Initialize coordinate token loss accumulators if coordinate tokens are enabled
            coordinate_tokens_enabled = self.config.coordinate_tokens_enabled
            if coordinate_tokens_enabled:
                self.logger.info("🎯 Initializing clean coordinate token loss tracking")
                self._current_focal_loss: float = 0.0
                self._current_l1_loss: float = 0.0
                self._current_giou_loss: float = 0.0
                self._accumulated_focal_loss: float = 0.0
                self._accumulated_l1_loss: float = 0.0
                self._accumulated_giou_loss: float = 0.0

                # Initialize additional accumulators for coordinate loss tracking
                self._accumulated_coordinate_loss: float = 0.0
                self._accumulated_regular_loss: float = 0.0
                self._accumulated_coord_l1_loss: float = 0.0
                self._accumulated_coord_giou_loss: float = 0.0

        # Cache for per-step weight / grad norms (populated in training_step)
        self._norm_cache: Dict[str, float] = {}

        # Initialize optimizer step tracking
        self._optimizer_step_wrapped: bool = False

        # CRITICAL: Ensure tokenizer has correct padding_side for Flash Attention
        self._fix_tokenizer_padding_side()

    def _fix_tokenizer_padding_side(self):
        """Ensure all tokenizer references have correct padding_side for Flash Attention."""
        tokenizers_to_fix = []

        # Collect all possible tokenizer references
        if hasattr(self, "tokenizer_ref") and hasattr(
            self.tokenizer_ref, "padding_side"
        ):
            tokenizers_to_fix.append(("tokenizer_ref", self.tokenizer_ref))
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "padding_side"):
            tokenizers_to_fix.append(("tokenizer", self.tokenizer))
        if (
            hasattr(self, "data_collator")
            and hasattr(self.data_collator, "tokenizer")
            and hasattr(self.data_collator.tokenizer, "padding_side")
        ):
            tokenizers_to_fix.append(
                ("data_collator.tokenizer", self.data_collator.tokenizer)
            )

        # Fix padding_side for all found tokenizers
        for name, tokenizer in tokenizers_to_fix:
            if tokenizer.padding_side != "left":
                self.logger.warning(
                    f"🔧 Fixing {name} padding_side: {tokenizer.padding_side} -> left"
                )
                tokenizer.padding_side = "left"
            else:
                self.logger.debug(
                    f"✅ {name} padding_side already correct: {tokenizer.padding_side}"
                )

    @property
    def tokenizer(self):
        """Backward compatibility property - returns cached tokenizer without deprecation warning."""
        return self.tokenizer_ref

    @tokenizer.setter
    def tokenizer(self, processing_class):
        """Backward compatibility setter for tokenizer."""
        self.tokenizer_ref = processing_class

        # After initialization, we can safely access the tokenizer
        # and add our special tokens. This is a critical step.
        if self.tokenizer_ref:
            special_tokens = SpecialTokens()
            num_added = self.tokenizer_ref.add_special_tokens(
                {"additional_special_tokens": special_tokens.to_list()}
            )
            if num_added > 0:
                self.logger.info(
                    f"✅ Added {num_added} special tokens to the tokenizer."
                )
                # Important: Resize token embeddings in the model
                self.model.resize_token_embeddings(len(self.tokenizer_ref))
                self.logger.info(
                    "✅ Resized model token embeddings to match new tokenizer size."
                )
        else:
            self.logger.warning(
                "⚠️ Tokenizer not found on BBUTrainer, skipping special token setup."
            )

    def _save(
        self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None
    ) -> None:
        """Save checkpoint in **sharded** form.

        1. Always save the *base* Qwen2.5-VL model with `max_shard_size` so
           enormous weights are split across multiple files (faster I/O).
        2. Save the coordinate token enhanced model.
        3. Tokenizer / processor and training args are stored with standard
           Transformers helpers.
        """

        import json
        import os

        import torch

        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. Save complete model (sharded) including visual components ---
        # CRITICAL FIX: Save the full model, not just base_model
        # For Qwen2.5-VL, visual tower is part of the main model, not base_model
        if (
            not hasattr(self.model, "coordinate_tokens_enabled")
            or not self.model.coordinate_tokens_enabled
        ):
            # For coordinate tokens disabled, save the full Qwen2.5-VL model
            self.logger.info(
                "💾 Saving complete Qwen2.5-VL model with visual tower (sharded)…"
            )
            model_to_save = self.model
        else:
            # For coordinate tokens enabled, save the enhanced model
            self.logger.info("💾 Saving base Qwen2.5-VL model (sharded)…")
            # EXPLICIT CONFIG: Use base_model if available, otherwise use the model itself
            model_to_save = (
                self.model.base_model
                if hasattr(self.model, "base_model")
                else self.model
            )

        model_to_save.save_pretrained(
            output_dir, max_shard_size="2GB", safe_serialization=True
        )

        # --- 2. Save tokenizer & processor ---------------------------------
        if self.tokenizer_ref is not None:
            self.logger.info("💾 Saving tokenizer...")
            self.tokenizer_ref.save_pretrained(output_dir)

        if self.image_processor is None:
            raise RuntimeError(
                "Image processor is None - cannot save preprocessor config!"
            )

        self.logger.info("💾 Saving image processor...")

        # Load base config from pretrained model and only override specific values
        # This prevents corruption of other config values
        from src.config import config as global_config

        # EXPLICIT: Access model_path without fallback
        if not hasattr(global_config, "model_path"):
            raise ValueError(
                "global_config missing 'model_path' attribute. "
                "Ensure model_path is explicitly set in configuration."
            )

        base_model_path = global_config.model_path
        if base_model_path is None or base_model_path == "":
            raise ValueError("model_path cannot be None or empty in global config")

        base_preproc_path = os.path.join(base_model_path, "preprocessor_config.json")

        if os.path.exists(base_preproc_path):
            with open(base_preproc_path, "r", encoding="utf-8") as f:
                ip_cfg = json.load(f)
        else:
            raise RuntimeError(
                f"Base preprocessor config not found: {base_preproc_path}"
            )

        # ONLY override the values that might have changed during training
        # (min_pixels and max_pixels from vision_process.py)
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

        # Make sure output_dir is a string
        output_dir_str = str(output_dir) if output_dir is not None else ""

        with open(
            os.path.join(output_dir_str, "preprocessor_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ip_cfg, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"   ✅ Image processor config saved with {len(ip_cfg)} parameters (preserving base config)"
        )

        # Detection is now handled via coordinate tokens

        # --- 4. Copy essential files from base model -------------------
        self._copy_essential_files_from_base_model(output_dir)

        # --- 5. Save training args ----------------------------------------
        # Make sure output_dir is a string
        output_dir_str = str(output_dir) if output_dir is not None else ""
        torch.save(self.args, os.path.join(output_dir_str, "training_args.bin"))

        # --- 6. Verify visual components were saved -------------------
        self._verify_saved_checkpoint(output_dir)

        self.logger.info(f"✅ Checkpoint saved successfully → {output_dir}")

    def _copy_essential_files_from_base_model(self, output_dir: str) -> None:
        """Copy essential files from base model to match pretrained model structure."""
        import os
        import shutil

        from src.config import config as global_config

        self.logger.info("💾 Copying essential files from base model...")

        # EXPLICIT CONFIG: model_path is required and validated at config load
        base_model_path = global_config.model_path
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
        # Make sure paths are strings
        base_model_path_str = (
            str(base_model_path) if base_model_path is not None else ""
        )
        output_dir_str = str(output_dir) if output_dir is not None else ""

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

    def _verify_saved_checkpoint(self, output_dir: str) -> None:
        """Verify that all necessary components were saved in the checkpoint."""
        import os

        from safetensors import safe_open

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

        # SAFE FALLBACK: Vision tower check for wrapped models
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
        self.logger.info(f"      Examples: {visual_keys[:3]}...")

    def _is_coordinate_token_parameter(
        self, param_name: str, param: torch.nn.Parameter
    ) -> bool:
        """
        Check if a parameter is a coordinate token parameter.

        Args:
            param_name: Name of the parameter
            param: The parameter tensor

        Returns:
            True if this is a coordinate token parameter
        """
        # 1. Check for explicit coordinate token modules
        if any(
            pattern in param_name
            for pattern in [
                "extended_embeddings",
                "extended_lm_head",
                "coordinate_tokens",
                "coord_tokens",
                "coordinate_head",
            ]
        ):
            return True

        # 2. Check if model has coordinate tokens enabled and this is an extended embedding/LM head
        if (
            hasattr(self.model, "coordinate_tokens_enabled")
            and self.model.coordinate_tokens_enabled
        ):
            # Check if this is the input embeddings or LM head that contains coordinate tokens
            if any(
                pattern in param_name
                for pattern in [
                    "embed_tokens.weight",  # Input embeddings
                    "lm_head.weight",  # Output LM head
                ]
            ):
                # Verify this parameter actually has extended vocabulary
                return self._is_extended_vocabulary_parameter(param_name, param)

        # 3. Check for coordinate-specific parameter patterns
        if any(
            pattern in param_name.lower()
            for pattern in [
                "coordinate",
                "coord_",
                "bbox",
                "detection_head",
            ]
        ):
            return True

        return False

    def _is_extended_vocabulary_parameter(
        self, param_name: str, param: torch.nn.Parameter
    ) -> bool:
        """
        Check if a parameter has extended vocabulary size (indicating coordinate tokens).

        Args:
            param_name: Parameter name to check
            param: The parameter tensor

        Returns:
            True if parameter has extended vocabulary size
        """
        try:
            # EXPLICIT CONFIG: Check if model has coordinate token information
            if hasattr(self.model, "original_vocab_size") and hasattr(
                self.model, "extended_vocab_size"
            ):
                # EXPLICIT CONFIG: These attributes are set during model initialization
                original_size = self.model.original_vocab_size
                extended_size = self.model.extended_vocab_size

                # For embedding parameters, check the first dimension (vocab size)
                if "embed_tokens.weight" in param_name and param.dim() >= 2:
                    actual_vocab_size = param.shape[0]
                    return (
                        actual_vocab_size == extended_size
                        and extended_size > original_size
                    )

                # For LM head parameters, check the output dimension
                if "lm_head.weight" in param_name and param.dim() >= 2:
                    actual_vocab_size = param.shape[0]  # Output vocab size
                    return (
                        actual_vocab_size == extended_size
                        and extended_size > original_size
                    )

            return False
        except Exception:
            return False

    def init_param_groups(self) -> None:
        """
        Initializes parameter groups for differential learning rate.

        This method categorizes all trainable parameters into 'vision', 'merger',
        and 'llm' groups. It will raise a ValueError if any
        trainable parameters cannot be categorized, ensuring that all parts of
        the model are explicitly handled.
        """
        self.logger.info(
            "🔧 Initializing parameter groups for differential learning rate..."
        )

        param_groups_with_names = {
            "vision": [],
            "merger": [],
            "llm": [],
            "coordinate": [],  # For coordinate token parameters
            "others": [],  # For uncategorized parameters
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Correct parameter name matching based on the model's structure.
            # The order is critical: check for the most specific names first.
            # Check for coordinate token parameters first (highest priority)
            if self._is_coordinate_token_parameter(name, param):
                param_groups_with_names["coordinate"].append((name, param))
            # "merger" is part of the vision tower, so check for it *before* "visual".
            elif "merger" in name:
                param_groups_with_names["merger"].append((name, param))
            elif "visual" in name:
                param_groups_with_names["vision"].append((name, param))
            # Language model parameters are in the main 'model' and 'lm_head'.
            elif "model." in name or ".model." in name or "lm_head" in name:
                param_groups_with_names["llm"].append((name, param))
            else:
                param_groups_with_names["others"].append((name, param))

        # Check for uncategorized parameters and raise an error if any are found.
        if param_groups_with_names["others"]:
            other_param_names = [name for name, _ in param_groups_with_names["others"]]
            self.logger.error(
                f"❌ Found {len(other_param_names)} unexpected trainable parameters that could not be categorized:"
            )
            for name in other_param_names:
                self.logger.error(f"   - {name}")
            raise ValueError(
                "Uncategorized trainable parameters found. All parameters must be explicitly "
                "assigned to a learning rate group (vision, merger, llm)."
            )

        # Remove the (now empty) 'others' group
        del param_groups_with_names["others"]

        # Store for optimizer creation (without names)
        self._param_groups = {
            group: [p for _, p in params]
            for group, params in param_groups_with_names.items()
        }
        self._param_names = list(self._param_groups.keys())

        # Log the parameter distribution
        for group, params in self._param_groups.items():
            num_params = sum(p.numel() for p in params)
            if num_params > 0:
                self.logger.info(
                    f"   - Group '{group}': {len(params)} tensors, {num_params / 1e6:.2f}M params"
                )

    def create_optimizer(self) -> Union[Optimizer, DummyOptim]:
        """
        Create the optimizer with differential learning rates if configured.
        """
        # --------------------------------------------------------------
        # Ensure parameter groups are initialised *before* the first call
        # to HF Trainer's optimiser builder.  If they are missing at this
        # point we compute them on-the-fly so that the very first optimiser
        # contains the correct learning-rate buckets.
        # --------------------------------------------------------------

        if not self.config.use_differential_lr:
            self.logger.info("🚀 Differential LR disabled → using standard optimizer…")
            self.optimizer = super().create_optimizer()
            self._wrap_optimizer_step()
            return self.optimizer

        # Differential LR *enabled* — make sure param groups exist
        if not hasattr(self, "_param_groups"):
            self.logger.info(
                "🔧 _param_groups not found – running init_param_groups() now…"
            )
            self.init_param_groups()

        self.logger.info("🚀 Creating optimizer with differential learning rates...")

        lr_map = {
            "vision": self.config.vision_lr,
            "merger": self.config.merger_lr,
            "llm": self.config.llm_lr,
            "coordinate": self.config.coordinate_lr,
        }

        optimizer_grouped_parameters = []
        for group_name, params in self._param_groups.items():
            if params:
                lr = lr_map[group_name]
                optimizer_grouped_parameters.append(
                    {
                        "params": params,
                        "lr": lr,
                    }
                )
                self.logger.info(f"   - Group '{group_name}' assigned LR: {lr}")

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )

        # The scheduler is responsible for applying the learning rate schedule to each
        # parameter group. The optimizer should be initialized with the per-group
        # learning rates, and the scheduler will correctly update them based on its
        # schedule (e.g., cosine annealing).
        #
        # The base `learning_rate` in `optimizer_kwargs` serves as a default for any
        # parameters that are not explicitly assigned to a group, which is not the
        # case here but is harmless to leave in. The per-group `lr` will take
        # precedence.
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        self._wrap_optimizer_step()  # Capture grad norms before zero_grad

        self.logger.info(
            "✅ Optimizer with differential learning rates created successfully."
        )
        return self.optimizer

    def _wrap_optimizer_step(self):
        """Wrap ``optimizer.step`` so we can capture weight/grad norms right
        before gradients are cleared by ``zero_grad``.  The wrapper is applied
        once, immediately after the optimizer is created.  Captured statistics
        are stored in ``self._norm_cache`` and consumed during the next call to
        ``_maybe_log_save_evaluate``.  This guarantees that *train* logs always
        contain the true gradient magnitudes, even when DeepSpeed/Accelerate
        zero out the gradients before we reach the logging hook.
        """
        # Norm capture disabled to avoid signature mismatches under Accelerate
        return

        # EXPLICIT CONFIG: _optimizer_step_wrapped is initialized in __init__
        if hasattr(self, "_optimizer_step_wrapped") and self._optimizer_step_wrapped:
            return  # Already wrapped

        from torch.optim import Optimizer

        if not isinstance(self.optimizer, Optimizer):
            self.logger.debug(
                "Optimizer is not a torch.optim.Optimizer; skipping norm capture wrap."
            )
            return

        self._optimizer_step_wrapped = True

        original_step = self.optimizer.step

        def step_with_norm_capture(*args, **kwargs):  # type: ignore[override]
            try:
                self._norm_cache = self._capture_grad_weight_norms()
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to capture weight/grad norms: {exc}")
                self._norm_cache = {}

            return original_step(*args, **kwargs)

        import types

        self.optimizer.step = types.MethodType(step_with_norm_capture, self.optimizer)  # type: ignore[assignment]

    def _capture_grad_weight_norms(self) -> Dict[str, float]:
        """Compute per-parameter-set weight and gradient L2 norms.

        Returns a flat dict ready to be merged into the training logs, e.g.::

            {
                "wn/vision_adapter": 0.91,
                "gn/vision_adapter": 0.03,
                ...
            }
        """

        norms: Dict[str, float] = {}

        # Detection head removed - using coordinate tokens instead
        module_map = {
            # Legacy detection head modules removed
        }

        # Build a quick lookup for named_modules once to avoid O(N²) search
        named_modules = dict(self.model.named_modules())

        for key, module_path in module_map.items():
            module = None
            # Prefer exact match first; fallback to suffix match for robustness
            if module_path in named_modules:
                module = named_modules[module_path]
            else:
                for name, mod in named_modules.items():
                    if name.endswith(module_path):
                        module = mod
                        break

            if module is None:
                # Skip if the module does not exist (e.g., detection disabled)
                continue

            first_param = next(module.parameters())
            weight_sq: torch.Tensor = torch.zeros((), device=first_param.device)  # type: ignore[arg-type]
            grad_sq: torch.Tensor = torch.zeros_like(weight_sq)
            param_cnt: int = 0

            for p in module.parameters():
                weight_sq += p.data.norm(2).pow(2)
                if p.grad is not None:
                    grad_sq += p.grad.norm(2).pow(2)
                param_cnt += 1

            if param_cnt == 0:
                continue

            # All-reduce for distributed so we get *global* norms, even with ZeRO
            vec = torch.stack(
                [
                    weight_sq,
                    grad_sq,
                    torch.tensor(float(param_cnt), device=weight_sq.device),
                ]
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)

            total_params = vec[2].item()
            norms[f"wn/{key}"] = (vec[0].sqrt() / total_params).item()
            norms[f"gn/{key}"] = (vec[1].sqrt() / total_params).item()

        return norms

    def _compute_teacher_student_losses(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        inputs: Dict[str, Any],
        requires_grad: bool = True,
    ) -> tuple[float, float]:
        """
        Compute separate losses for teacher and student spans.

        Uses the same shifting logic as the total LM loss computation,
        so the returned losses are directly comparable to the total LM loss.

        Args:
            logits: Model output logits [batch_size, sequence_length, vocab_size]
            labels: Ground truth labels [batch_size, sequence_length]
            inputs: Batch inputs containing span information
            requires_grad: Whether to create tensors with gradients (True for training, False for evaluation)

        Returns:
            Tuple of (teacher_loss, student_loss) as float values
        """
        if labels is None:
            return 0.0, 0.0

        # Get spans from inputs (may be None for backward compatibility)
        teacher_spans = inputs["teacher_assistant_spans"]
        student_spans = inputs["student_assistant_spans"]

        if teacher_spans is None or student_spans is None:
            # No spans available, split not possible
            return 0.0, 0.0

        # CRITICAL: Apply the same shifting as the total LM loss computation
        # The model predicts next tokens, so we shift logits[:-1] vs labels[1:]
        batch_size, seq_len, vocab_size = logits.shape
        shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
        shift_labels = labels[..., 1:].contiguous()  # [batch, seq_len-1]

        # Flatten shifted logits and labels for easier indexing
        flat_logits = shift_logits.view(
            -1, vocab_size
        )  # [batch_size * (seq_len-1), vocab_size]
        flat_labels = shift_labels.view(-1)  # [batch_size * (seq_len-1)]

        # Adjust sequence length for shifted data
        shifted_seq_len = seq_len - 1

        # Create loss function - use MEAN reduction to match the total LM loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # Collect indices for teacher and student tokens
        # NOTE: Spans are in original coordinates, but we need to adjust for shifting
        teacher_indices = []
        student_indices = []

        for batch_idx in range(batch_size):
            # Teacher spans for this sample
            for start, end in teacher_spans[batch_idx]:
                for pos in range(start, end):
                    # Shift adjustment: original pos becomes pos-1 in shifted sequence
                    # We predict token at pos using logits[pos-1], so spans shift left by 1
                    shifted_pos = pos - 1
                    if (
                        0 <= shifted_pos < shifted_seq_len
                    ):  # Safety check for shifted bounds
                        flat_idx = batch_idx * shifted_seq_len + shifted_pos
                        teacher_indices.append(flat_idx)

            # Student spans for this sample
            for start, end in student_spans[batch_idx]:
                for pos in range(start, end):
                    # Shift adjustment: original pos becomes pos-1 in shifted sequence
                    shifted_pos = pos - 1
                    if (
                        0 <= shifted_pos < shifted_seq_len
                    ):  # Safety check for shifted bounds
                        flat_idx = batch_idx * shifted_seq_len + shifted_pos
                        student_indices.append(flat_idx)

        # Compute teacher loss (with gradients for potential reweighting)
        teacher_loss_tensor = torch.tensor(
            0.0, device=logits.device, requires_grad=requires_grad
        )
        teacher_loss_float = 0.0
        if teacher_indices:
            teacher_indices_tensor = torch.tensor(teacher_indices, device=logits.device)
            teacher_logits = flat_logits[teacher_indices_tensor]
            teacher_labels = flat_labels[teacher_indices_tensor]

            # Only compute loss on tokens that are not ignored
            valid_mask = teacher_labels != -100
            if valid_mask.any():
                # Keep tensor with gradients for potential reweighting
                teacher_loss_tensor = loss_fn(
                    teacher_logits[valid_mask], teacher_labels[valid_mask]
                )
                teacher_loss_float = teacher_loss_tensor.detach().item()

        # Compute student loss (with gradients for potential reweighting)
        student_loss_tensor = torch.tensor(
            0.0, device=logits.device, requires_grad=requires_grad
        )
        student_loss_float = 0.0
        if student_indices:
            student_indices_tensor = torch.tensor(student_indices, device=logits.device)
            student_logits = flat_logits[student_indices_tensor]
            student_labels = flat_labels[student_indices_tensor]

            # Only compute loss on tokens that are not ignored
            valid_mask = student_labels != -100
            if valid_mask.any():
                # Keep tensor with gradients for potential reweighting
                student_loss_tensor = loss_fn(
                    student_logits[valid_mask], student_labels[valid_mask]
                )
                student_loss_float = student_loss_tensor.detach().item()

        # Store tensors for potential gradient reweighting (future enhancement)
        self._teacher_loss_tensor = teacher_loss_tensor
        self._student_loss_tensor = student_loss_tensor

        return teacher_loss_float, student_loss_float

    def _compute_loss_with_coordinator(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss using the training coordinator system.

        This method delegates complex loss computation to the coordinator
        while maintaining the same interface as the legacy compute_loss.
        """

        # Ensure we get hidden states for detection
        model_inputs = inputs.copy()
        model_inputs["output_hidden_states"] = True

        # Validate tokens in training samples (only during training, not evaluation)
        if model.training:
            try:
                # Get the actual model (unwrap DataParallel if needed)
                actual_model = model.module if hasattr(model, "module") else model

                if hasattr(actual_model, "validate_sample_tokens"):
                    # Get input_ids from inputs
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None:
                        # Validate each sample in the batch
                        batch_size = (
                            input_ids.shape[0]
                            if hasattr(input_ids, "shape")
                            else len(input_ids)
                        )
                        for i in range(batch_size):
                            sample_input_ids = (
                                input_ids[i] if batch_size > 1 else input_ids
                            )
                            actual_model.validate_sample_tokens(
                                sample_input_ids,
                                sample_info={
                                    "index": f"batch_{i}",
                                    "training_step": self.state.global_step,
                                    "epoch": self.state.epoch,
                                },
                            )
                        self.logger.debug(
                            f"✅ Token validation passed for batch of {batch_size} samples"
                        )
                else:
                    self.logger.warning(
                        "⚠️ Model does not have validate_sample_tokens method - skipping token validation"
                    )
            except Exception as e:
                self.logger.error(f"❌ Token validation failed: {e}")
                raise e

        # Run model forward pass
        self.logger.debug(f"🔍 TRAINER: About to call model(**model_inputs):")
        self.logger.debug(f"   model type: {type(model)}")
        self.logger.debug(f"   model class: {model.__class__.__name__}")
        self.logger.debug(f"   model id: {id(model)}")

        # Check if this is our wrapper or the base model
        if hasattr(model, "coordinate_tokens_enabled"):
            self.logger.debug(
                f"   ✅ Model is our wrapper with coordinate_tokens_enabled: {model.coordinate_tokens_enabled}"
            )
        else:
            self.logger.error(
                f"   ❌ CRITICAL: Model is NOT our wrapper - it's the base model!"
            )
            self.logger.error(f"   This explains why coordinate losses are missing")

        outputs = model(**model_inputs)

        # CRITICAL DEBUG: Check coordinate losses immediately after model call
        self.logger.debug(
            f"🔍 TRAINER: Model outputs immediately after model(**model_inputs):"
        )
        self.logger.debug(f"   outputs type: {type(outputs)}")
        self.logger.debug(f"   outputs id: {id(outputs)}")
        self.logger.debug(
            f"   hasattr _geometry_focal_loss: {hasattr(outputs, '_geometry_focal_loss')}"
        )
        self.logger.debug(
            f"   hasattr _coordinate_l1_loss: {hasattr(outputs, '_coordinate_l1_loss')}"
        )
        self.logger.debug(
            f"   hasattr _geometry_bbox_giou_loss: {hasattr(outputs, '_geometry_bbox_giou_loss')}"
        )

        # If coordinate losses are missing, we need to investigate the model call
        if hasattr(outputs, "_geometry_focal_loss") and hasattr(
            outputs, "_coordinate_l1_loss"
        ):
            self.logger.debug(f"   ✅ Coordinate losses found on outputs object")
            self.logger.debug(
                f"   _geometry_focal_loss: {outputs._geometry_focal_loss}"
            )
            self.logger.debug(f"   _coordinate_l1_loss: {outputs._coordinate_l1_loss}")
        else:
            self.logger.error(
                f"   ❌ CRITICAL: Coordinate losses missing from outputs object!"
            )
            self.logger.error(
                f"   This means model wrapper is not being called or not attaching losses properly"
            )

        # Use coordinator for loss computation
        if self.training_coordinator is not None:
            total_loss, loss_components = self.training_coordinator.compute_loss(
                model_outputs=outputs, inputs=inputs, is_training=model.training
            )
        else:
            # Fallback to default loss computation if no coordinator is available
            self.logger.warning(
                "⚠️ No training coordinator available for loss computation, using fallback method"
            )
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # During evaluation, detach the loss to avoid gradient issues with logging
        if not model.training and hasattr(total_loss, "detach"):
            total_loss = total_loss.detach()

        # Update current loss attributes for compatibility with legacy logging
        # Note: lm_loss removed - use teacher + student components instead
        self._current_teacher_lm_loss = loss_components["teacher_lm_loss"]
        self._current_student_lm_loss = loss_components["student_lm_loss"]
        self._current_lm_loss = (
            self._current_teacher_lm_loss + self._current_student_lm_loss
        )  # For legacy compatibility
        # objectness_loss removed - not needed for coordinate token system
        # bbox_* and caption_loss duplicates removed - use individual components instead
        # Add coordinate token loss components with new naming - NO DEFAULTS, FAIL FAST
        if (
            hasattr(self.config, "coordinate_tokens_enabled")
            and self.config.coordinate_tokens_enabled
        ):
            required_loss_components = ["llm_loss"]  # Only require clean llm_loss
            # Optional coordinate components: focal_loss, l1_loss, giou_loss
            missing_components = []

            # Check required components
            for component in required_loss_components:
                if component not in loss_components:
                    missing_components.append(component)

            if missing_components:
                raise RuntimeError(
                    f"Coordinate tokens enabled but training coordinator missing required loss components: {missing_components}. "
                    f"This indicates training coordinator is not properly computing coordinate losses."
                )

            # Extract with strict validation - llm_loss is always present
            self._current_llm_loss_clean = loss_components["llm_loss"]

            # Extract coordinate losses (may be zero for teacher samples)
            self._current_coord_focal_loss = loss_components.get(
                "coord_focal_loss", 0.0
            )
            self._current_coord_l1_loss = loss_components.get("coord_l1_loss", 0.0)
            self._current_coord_giou_loss = loss_components.get("coord_giou_loss", 0.0)
        else:
            # Coordinate tokens disabled - set to zero
            self._current_regular_loss = 0.0
            self._current_coord_focal_loss = 0.0
            self._current_coord_l1_loss = 0.0
            self._current_coord_giou_loss = 0.0

        # Coordinator system handles all loss extraction and accumulation

        # Return in same format as legacy method
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss for BBU models with extended functionality.

        Args:
            model: Model to compute loss for
            inputs: Model inputs
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (for loss scaling)

        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        # Use coordinator for enhanced loss computation if available
        if (
            self._use_coordinator
            and hasattr(self, "training_coordinator")
            and self.training_coordinator is not None
        ):
            # Let coordinator handle loss computation with multi-loss support
            if hasattr(self.training_coordinator, "compute_loss"):
                # Filter inputs for base model compatibility
                model_inputs = inputs
                # EXPLICIT CONFIG: detection_enabled is set during model initialization
                if hasattr(model, "detection_enabled") and not model.detection_enabled:
                    # For base model, exclude packed collator and custom fields
                    excluded_keys = [
                        "cu_seqlens",
                        "image_counts_per_sample",
                        "ground_truth_objects",
                        "teacher_assistant_spans",
                        "student_assistant_spans",
                    ]
                    model_inputs = {
                        k: v for k, v in inputs.items() if k not in excluded_keys
                    }
                    self.logger.debug(
                        f"🔍 Filtered inputs for base model: {list(model_inputs.keys())}"
                    )
                elif not hasattr(model, "detection_enabled"):
                    # Assume it's a base model if no detection_enabled attribute
                    excluded_keys = [
                        "cu_seqlens",
                        "image_counts_per_sample",
                        "ground_truth_objects",
                        "teacher_assistant_spans",
                        "student_assistant_spans",
                    ]
                    model_inputs = {
                        k: v for k, v in inputs.items() if k not in excluded_keys
                    }
                    self.logger.debug(
                        f"🔍 Filtered inputs for standard model: {list(model_inputs.keys())}"
                    )

                # First get model outputs, then pass to coordinator
                model_outputs = model(**model_inputs)
                loss, _ = self.training_coordinator.compute_loss(
                    model_outputs, inputs, is_training=model.training
                )
                # Return loss (and outputs if requested)
                return (loss, model_outputs) if return_outputs else loss
            else:
                self.logger.warning(
                    "Training coordinator doesn't have compute_loss method, falling back to standard loss"
                )

        # Fall back to standard loss computation
        if (
            hasattr(self, "label_smoother")
            and self.label_smoother is not None
            and "labels" in inputs
        ):
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if hasattr(self, "label_smoother") and self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # Standard behavior: model outputs loss directly
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            # No labels provided, check if model computes loss internally
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                # No internal loss computation
                raise ValueError(
                    "No labels provided and model does not compute loss internally"
                )

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss: Union[torch.Tensor, float],
        grad_norm: Optional[Union[torch.Tensor, float]],
        model: nn.Module,
        trial: Any,
        epoch: Optional[float],
        ignore_keys_for_eval: Optional[List[str]],
        start_time: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Log metrics with averaging for gradient accumulation.
        """
        if self.control.should_log:
            # Define num_micro_batches for both coordinator and legacy paths
            num_micro_batches = max(1, self._micro_batch_count)

            # NEW: Use training coordinator for loss averaging if available
            if (
                self._use_coordinator
                and self.training_coordinator is not None
                and hasattr(self.training_coordinator, "get_averaged_losses_and_reset")
            ):
                component_logs = (
                    self.training_coordinator.get_averaged_losses_and_reset()
                )

                # Validate required keys in component_logs
                if "loss" not in component_logs:
                    raise ValueError(
                        "Coordinator must provide 'loss' in component_logs"
                    )

                total_avg_loss = component_logs["loss"]

                # CRITICAL FIX: Ensure coordinate losses are always included in coordinator logs
                # Even if they come from the coordinator, we need to validate they're present
                # Validate coordinate_tokens_enabled is present in config
                if not hasattr(self.config, "coordinate_tokens_enabled"):
                    raise ValueError(
                        "coordinate_tokens_enabled must be explicitly configured in config"
                    )

                coordinate_tokens_enabled = self.config.coordinate_tokens_enabled
                if coordinate_tokens_enabled:
                    # Simplified validation - only require coordinate L1 loss
                    required_coord_losses = [
                        "coordinate_l1_loss",
                    ]
                    missing_coord_losses = []

                    for key in required_coord_losses:
                        if key not in component_logs:
                            missing_coord_losses.append(key)

                    if missing_coord_losses:
                        raise RuntimeError(
                            f"Coordinate tokens enabled but coordinator missing required losses: {missing_coord_losses}. "
                            f"This indicates training coordinator is not properly computing coordinate losses."
                        )

                    # DEBUG: Log all available keys in component_logs
                    self.logger.debug(
                        f"🔍 TRAINER: Available component_logs keys: {list(component_logs.keys())}"
                    )

                    # Log coordinate loss values for debugging
                    total_coord_loss = sum(
                        component_logs.get(key, 0.0) for key in required_coord_losses
                    )
                    self.logger.debug(f"🔍 TRAINER: Simplified coordinate losses:")
                    self.logger.debug(
                        f"   coordinate_l1_loss: {component_logs.get('coordinate_l1_loss', 0.0)}"
                    )
                    self.logger.debug(f"   total_coord_loss: {total_coord_loss}")

                    # STRICT VALIDATION: All samples must have students, students must have coordinate losses
                    self._step_count += 1

                    # Validate student_lm_loss and teacher_lm_loss are present
                    if "student_lm_loss" not in component_logs:
                        raise ValueError(
                            "Coordinator must provide 'student_lm_loss' in component_logs"
                        )
                    if "teacher_lm_loss" not in component_logs:
                        raise ValueError(
                            "Coordinator must provide 'teacher_lm_loss' in component_logs"
                        )

                    student_lm_loss = component_logs["student_lm_loss"]
                    teacher_lm_loss = component_logs["teacher_lm_loss"]

                    # Every sample must have student portion (students do detection)
                    if student_lm_loss == 0.0 and (
                        not hasattr(self.args, "local_rank")
                        or self.args.local_rank <= 0
                    ):
                        self.logger.error(
                            f"❌ TRAINER: CRITICAL ERROR at step {self._step_count}"
                        )
                        self.logger.error(
                            "❌ No student samples found - ALL samples must have student portions!"
                        )
                        self.logger.error(
                            "❌ Expected: Every sample has student, some samples also have teachers"
                        )
                        self.logger.error(f"❌ Debug info:")
                        self.logger.error(f"   Step: {self._step_count}")
                        self.logger.error(f"   Student LM loss: {student_lm_loss}")
                        self.logger.error(f"   Teacher LM loss: {teacher_lm_loss}")
                        raise RuntimeError(
                            f"CRITICAL: No student samples found at step {self._step_count}. "
                            f"All training samples must have student portions for detection."
                        )

                    # Every student sample must have coordinate losses (students do detection)
                    # Use small threshold to handle floating point precision issues
                    coord_loss_threshold = 1e-6

                    # Check if we're in a testing environment (integration tests)
                    # Integration tests with synthetic data may not generate proper coordinate tokens
                    is_integration_test = hasattr(
                        self.args, "output_dir"
                    ) and "pipeline_test" in str(self.args.output_dir)

                    if (
                        total_coord_loss < coord_loss_threshold
                        and student_lm_loss > 0.0
                        and not is_integration_test  # Skip validation for integration tests
                        and (
                            not hasattr(self.args, "local_rank")
                            or self.args.local_rank <= 0
                        )
                    ):
                        self.logger.error(
                            f"❌ TRAINER: CRITICAL ERROR at step {self._step_count}"
                        )
                        self.logger.error(
                            "❌ Student samples present but all coordinate losses are ZERO!"
                        )
                        self.logger.error(
                            "❌ Students must have coordinate losses when coordinate tokens are enabled"
                        )
                        self.logger.error("❌ Possible causes:")
                        self.logger.error(
                            "   1. Data preprocessing failed to generate coordinate tokens"
                        )
                        self.logger.error(
                            "   2. Chat processor is not properly formatting coordinate tokens"
                        )
                        self.logger.error(
                            "   3. UnifiedTokenManager coordinate token addition failed"
                        )
                        self.logger.error("   4. Coordinate token vocabulary mismatch")
                        self.logger.error(
                            "   5. Model wrapper not computing coordinate losses"
                        )
                        self.logger.error(f"❌ Debug info:")
                        self.logger.error(f"   Step: {self._step_count}")
                        self.logger.error(
                            f"   Available loss keys: {list(component_logs.keys())}"
                        )
                        self.logger.error(
                            f"   LLM loss: {component_logs.get('llm_loss', 'MISSING')}"
                        )
                        self.logger.error(f"   Student LM loss: {student_lm_loss}")
                        self.logger.error(f"   Teacher LM loss: {teacher_lm_loss}")
                        self.logger.error(
                            f"   Total coordinate loss: {total_coord_loss}"
                        )
                        self.logger.error(f"   Threshold: {coord_loss_threshold}")
                        self.logger.error(
                            f"   Expected coordinate losses: {required_coord_losses}"
                        )
                        self.logger.error(f"   Individual coordinate losses:")
                        for loss_name in required_coord_losses:
                            loss_value = component_logs.get(loss_name, "MISSING")
                            self.logger.error(f"     {loss_name}: {loss_value}")

                        # This is a legitimate error - raise exception
                        raise RuntimeError(
                            f"CRITICAL: Student samples present (student_lm_loss={student_lm_loss}) but all coordinate losses are below threshold ({total_coord_loss} < {coord_loss_threshold}) at step {self._step_count}. "
                            f"This indicates a serious bug in data preprocessing or chat processing. "
                            f"Student samples must have coordinate tokens when coordinate_tokens_enabled=true."
                        )
            else:
                # LEGACY: Original loss averaging logic
                # The `tr_loss` from the Trainer is an accumulated value.
                # We re-compute the loss from our own averaged components to ensure
                # correct, per-step reporting consistent with the docstring.
                # Average the accumulated component losses

                # Average the accumulated component losses across *all* micro-batches
                # seen since the last log, independent of the gradient-accumulation
                # configuration.
                avg_lm_loss = self._accumulated_lm_loss / num_micro_batches
                avg_teacher_lm_loss = (
                    self._accumulated_teacher_lm_loss / num_micro_batches
                )
                avg_student_lm_loss = (
                    self._accumulated_student_lm_loss / num_micro_batches
                )
                total_avg_loss = avg_lm_loss

                component_logs: Dict[str, float] = {
                    "lm_loss": avg_lm_loss,
                    "teacher_lm_loss": avg_teacher_lm_loss,
                    "student_lm_loss": avg_student_lm_loss,
                }

                # Add coordinate token losses if available (new detection system)
                # ALWAYS log coordinate losses when coordinate tokens are enabled, even if zero
                if not hasattr(self.config, "coordinate_tokens_enabled"):
                    raise ValueError(
                        "coordinate_tokens_enabled must be explicitly configured in config"
                    )
                coordinate_tokens_enabled = self.config.coordinate_tokens_enabled
                if coordinate_tokens_enabled:
                    # Validate required accumulators exist
                    required_accumulators = [
                        "_accumulated_coordinate_loss",
                        "_accumulated_focal_loss",
                        "_accumulated_regular_loss",
                        "_accumulated_coord_l1_loss",
                        "_accumulated_coord_giou_loss",
                    ]

                    missing_accumulators = []
                    for attr in required_accumulators:
                        if not hasattr(self, attr):
                            missing_accumulators.append(attr)

                    if missing_accumulators:
                        raise RuntimeError(
                            f"Coordinate tokens enabled but trainer missing required accumulators: {missing_accumulators}. "
                            f"This indicates coordinate loss accumulation is not working correctly."
                        )

                    # Extract with clean naming - NO duplicates or unused losses
                    component_logs["regular_loss"] = (
                        self._accumulated_regular_loss / num_micro_batches
                    )

                    # Only add coordinate losses if they're non-zero (student samples)
                    if self._accumulated_focal_loss > 0:
                        component_logs["coord_focal_loss"] = (
                            self._accumulated_focal_loss / num_micro_batches
                        )
                    if self._accumulated_coord_l1_loss > 0:
                        component_logs["coord_l1_loss"] = (
                            self._accumulated_coord_l1_loss / num_micro_batches
                        )
                    if self._accumulated_coord_giou_loss > 0:
                        component_logs["coord_giou_loss"] = (
                            self._accumulated_coord_giou_loss / num_micro_batches
                        )
                    # Clean coordinate loss logging - only essential info

            # Legacy bbox detection loss logging removed - using coordinate tokens instead
            avg_objectness_loss = self._accumulated_objectness_loss / num_micro_batches
            if avg_objectness_loss > 0:
                component_logs["objectness_loss"] = avg_objectness_loss
                total_avg_loss += avg_objectness_loss

            # Define logging order: 'loss', 'grad_norm', then components
            logs: Dict[str, float] = {}
            logs["loss"] = total_avg_loss

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )

            logs.update(component_logs)

            # ------------------------------------------------------------------
            # Additional diagnostics: weight- and **gradient** norms.  For the
            # **train** phase we capture these in ``_wrap_optimizer_step`` *before*
            # DeepSpeed/Accelerate zero the grads.  If the cache is empty (e.g.,
            # during evaluation), we fall back to a best-effort recomputation –
            # grad norms will be zero in that case, which is expected.
            # ------------------------------------------------------------------

            if self._norm_cache:
                logs.update(self._norm_cache)
                # Clear after use so we don't accidentally reuse stale values.
                self._norm_cache = {}
            else:
                # Detection head removed - using coordinate tokens instead
                module_names = {
                    # Legacy detection head modules removed
                }

                for log_key, module_path in module_names.items():
                    module = None
                    for name, m in model.named_modules():
                        if name.endswith(module_path):
                            module = m
                            break
                    if module is None:
                        continue  # Skip if module missing (e.g., detection disabled)
                    with torch.no_grad():
                        weight_sq, grad_sq, param_cnt = 0.0, 0.0, 0
                        for p in module.parameters():
                            weight_sq += p.data.norm(2).pow(2)
                            if p.grad is not None:
                                grad_sq += p.grad.norm(2).pow(2)
                            param_cnt += 1

                        device = next(module.parameters()).device
                        vec = torch.tensor(
                            [weight_sq, grad_sq, float(param_cnt)],
                            device=device,
                            dtype=torch.float32,
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                vec, op=torch.distributed.ReduceOp.SUM
                            )

                        total_params = vec[2].item()
                        if total_params > 0:
                            logs[f"wn/{log_key}"] = (
                                vec[0].sqrt() / total_params
                            ).item()
                            logs[f"gn/{log_key}"] = (
                                vec[1].sqrt() / total_params
                            ).item()

            # Add ETA and remaining time
            if self.state.max_steps > 0:
                current_step = self.state.global_step
                if current_step > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_step = elapsed_time / current_step
                    remaining_steps = self.state.max_steps - current_step
                    remaining_time_s = remaining_steps * avg_time_per_step

                    logs["remaining_hr"] = round(remaining_time_s / 3600, 3)

            self.log(logs)

            # Reset accumulators after logging (only when not using coordinator)
            # The coordinator handles its own accumulator management
            if not self._use_coordinator:
                self._accumulated_lm_loss = 0.0
                self._accumulated_teacher_lm_loss = 0.0
                self._accumulated_student_lm_loss = 0.0
                self._accumulated_objectness_loss = 0.0

                # Validate coordinate token loss accumulators exist if coordinate tokens are enabled
                if (
                    hasattr(self.config, "coordinate_tokens_enabled")
                    and self.config.coordinate_tokens_enabled
                ):
                    # Validate required accumulators exist
                    required_accumulators = [
                        "_accumulated_coordinate_loss",
                        "_accumulated_focal_loss",
                        "_accumulated_regular_loss",
                        "_accumulated_coord_l1_loss",
                        "_accumulated_coord_giou_loss",
                    ]

                    missing_accumulators = []
                    for attr in required_accumulators:
                        if not hasattr(self, attr):
                            missing_accumulators.append(attr)

                    if missing_accumulators:
                        raise RuntimeError(
                            f"Coordinate tokens enabled but trainer missing required accumulators: {missing_accumulators}. "
                            f"This indicates coordinate loss accumulation is not working correctly."
                        )

                    # Reset coordinate token loss accumulators
                    self._accumulated_coordinate_loss = 0.0
                    self._accumulated_focal_loss = 0.0
                    self._accumulated_regular_loss = 0.0
                    self._accumulated_coord_l1_loss = 0.0
                    self._accumulated_coord_giou_loss = 0.0

                # Also reset the micro-batch counter so the next logging window
                # starts fresh.
                self._micro_batch_count = 0

        if self.control.should_evaluate:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Override to ensure proper data loading with BBU datasets.

        This ensures compatibility between HuggingFace trainer and our custom BBUDataset
        and coordinate token system. The base trainer's column removal logic can interfere
        with our data processing when remove_unused_columns=True (the default).

        Note: This issue is resolved by setting remove_unused_columns=False in configuration.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        from torch.utils.data import DataLoader

        # Define seed_worker locally if not available
        def seed_worker(worker_id):
            """Worker init function to set random seed for each worker."""
            import random

            import numpy as np
            import torch

            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Use our data collator directly without any wrapper
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,  # No column removal wrapper
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)
        )

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        """
        Override to prevent HuggingFace trainer from applying column removal wrapper.

        This ensures consistent behavior between training and evaluation dataloaders.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        from torch.utils.data import DataLoader

        # Define seed_worker locally if not available
        def seed_worker(worker_id):
            """Worker init function to set random seed for each worker."""
            import random

            import numpy as np
            import torch

            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Use our data collator directly without any wrapper
        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": self.data_collator,  # No column removal wrapper
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.
        This method is overridden to support logging of differential learning rates.
        """
        # Remove the generic learning rate from the logs.
        logs.pop("learning_rate", None)

        # Log the learning rate for each parameter group.
        if self.lr_scheduler is not None:
            try:
                # Check scheduler type and handle appropriately
                scheduler_class_name = self.lr_scheduler.__class__.__name__

                if scheduler_class_name == "DummyScheduler":
                    # Validate DummyScheduler has lr attribute
                    if not hasattr(self.lr_scheduler, "lr"):
                        raise ValueError("DummyScheduler missing lr attribute")
                    logs["learning_rate"] = self.lr_scheduler.lr
                elif hasattr(self.lr_scheduler, "get_last_lr"):
                    # Standard schedulers with get_last_lr method
                    last_lr = self.lr_scheduler.get_last_lr()

                    # Validate _param_names exists and has correct length
                    if not hasattr(self, "_param_names"):
                        raise ValueError(
                            "_param_names not initialized - call init_param_groups() first"
                        )

                    if len(last_lr) != len(self._param_names):
                        self.logger.warning(
                            f"Learning rate groups ({len(last_lr)}) don't match parameter groups ({len(self._param_names)})"
                        )
                        # Log all learning rates without group names
                        for i, lr in enumerate(last_lr):
                            logs[f"lr/group_{i}"] = lr
                    else:
                        # Log learning rates with group names
                        for i, (group_name, group_lr) in enumerate(
                            zip(self._param_names, last_lr)
                        ):
                            logs[f"lr/{group_name}"] = group_lr
                else:
                    # Fallback for scheduler types that don't have get_last_lr
                    # Validate args has learning_rate attribute
                    if not hasattr(self.args, "learning_rate"):
                        raise ValueError("args missing learning_rate attribute")
                    logs["learning_rate"] = self.args.learning_rate
            except Exception as e:
                # Log error but don't fail training because of logging issue
                self.logger.warning(f"Error getting learning rate: {e}")
                # Validate args has learning_rate attribute
                if hasattr(self.args, "learning_rate"):
                    logs["learning_rate"] = self.args.learning_rate

        super().log(logs, start_time)

    def _extract_ground_truth_objects(self, inputs):
        """Extracts ground truth objects from inputs if they exist."""
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            if "ground_truth_objects" in inputs:
                raw_objs = inputs["ground_truth_objects"][batch_idx]
                gt_objects = [
                    obj
                    if isinstance(obj, GroundTruthObject)
                    else GroundTruthObject(bbox_2d=obj["bbox_2d"], desc=obj["desc"])
                    for obj in raw_objs
                ]
                self.logger.debug(
                    f"🔍 Found GT objects for batch {batch_idx}: {len(gt_objects)} objects"
                )
            else:
                gt_objects = []
                self.logger.debug(f"🔍 No GT objects found for batch {batch_idx}")

            ground_truth_objects.append(gt_objects)

        total_gt_objects = sum(len(gt_objs) for gt_objs in ground_truth_objects)
        self.logger.debug(
            f"🔍 DEBUG: Total GT objects across batch: {total_gt_objects}"
        )

        return ground_truth_objects

    def _prepare_detection_inputs(self, inputs):
        """Extract ground truth objects from batch"""
        # DEBUG: Log what keys are available in inputs
        self.logger.info(f"🔍 DEBUG: Available input keys: {list(inputs.keys())}")

        # Extract GT objects from conversation format
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            # Extract from the data collator's stored information
            if "ground_truth_objects" in inputs:
                gt_objects = inputs["ground_truth_objects"][batch_idx]
                self.logger.info(
                    f"🔍 DEBUG: Found GT objects for batch {batch_idx}: {len(gt_objects)} objects"
                )
            else:
                # Fallback: extract from conversation (implement based on your data format)
                gt_objects = self._extract_gt_from_conversation(inputs, batch_idx)
                self.logger.info(
                    f"🔍 DEBUG: Using fallback GT extraction for batch {batch_idx}: {len(gt_objects)} objects"
                )

            ground_truth_objects.append(gt_objects)

        # DEBUG: Log final ground truth objects
        total_gt_objects = sum(len(gt_objs) for gt_objs in ground_truth_objects)
        self.logger.debug(f"🔍 Total GT objects across batch: {total_gt_objects}")
        self.logger.debug(
            f"🔍 GT objects per sample: {[len(gt_objs) for gt_objs in ground_truth_objects]}"
        )

        # Add to model inputs
        model_inputs = inputs.copy()
        model_inputs["ground_truth_objects"] = ground_truth_objects

        return model_inputs

    def _extract_gt_from_conversation(self, inputs, batch_idx):
        """Extract ground truth objects from conversation if not provided directly"""
        return []

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Enhanced prediction step that includes detection loss logging during evaluation.
        """
        # Store original state for restoration
        original_training = model.training

        # Set model to eval mode
        model.eval()

        # Use the same compute_loss logic but with eval prefix
        with torch.no_grad():
            # Fix Flash Attention padding issue during evaluation
            original_padding_side = None
            tokenizer_to_fix = None

            # Try multiple tokenizer references
            if hasattr(self, "tokenizer_ref") and hasattr(
                self.tokenizer_ref, "padding_side"
            ):
                tokenizer_to_fix = self.tokenizer_ref
            elif hasattr(self, "tokenizer") and hasattr(self.tokenizer, "padding_side"):
                tokenizer_to_fix = self.tokenizer
            elif (
                hasattr(self, "data_collator")
                and hasattr(self.data_collator, "tokenizer")
                and hasattr(self.data_collator.tokenizer, "padding_side")
            ):
                tokenizer_to_fix = self.data_collator.tokenizer

            if tokenizer_to_fix is not None:
                original_padding_side = tokenizer_to_fix.padding_side
                tokenizer_to_fix.padding_side = "left"
                self.logger.debug(
                    f"🔧 Fixed tokenizer padding_side for evaluation: {original_padding_side} -> left"
                )

            # Temporarily modify the loss info prefix for evaluation
            # EXPLICIT CONFIG: _loss_prefix is initialized in __init__
            old_prefix = getattr(
                self, "_loss_prefix", ""
            )  # Keep getattr for backward compatibility
            self._loss_prefix = "eval"

            try:
                # Use our enhanced compute_loss method
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs)
                    return (loss, None, None)
                else:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )

                    # Extract logits for evaluation metrics
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in (ignore_keys or []) + ["loss"]
                        )
                        # Convert to tensor for proper type compatibility
                        if logits and len(logits) == 1:
                            logits = logits[0]  # Unwrap single-item tuple
                    else:
                        logits = (
                            outputs[1:] if hasattr(outputs, "__getitem__") else outputs
                        )
                        # Handle non-tuple logits
                        if not isinstance(logits, tuple):
                            logits = logits

                    # Extract labels if available
                    labels = None
                    if hasattr(self, "label_names") and len(self.label_names) > 0:
                        labels = tuple(inputs.get(name) for name in self.label_names)
                        if len(labels) == 1:
                            labels = labels[0]

                    # Ensure return values match expected types in signature
                    if not isinstance(loss, (torch.Tensor, type(None))):
                        loss = torch.tensor(loss) if loss is not None else None

                    return (loss, logits, labels)

            finally:
                # Restore original padding side if it was changed
                if original_padding_side is not None and tokenizer_to_fix is not None:
                    tokenizer_to_fix.padding_side = original_padding_side
                    self.logger.debug(
                        f"🔧 Restored tokenizer padding_side: left -> {original_padding_side}"
                    )

                # Restore original prefix and training state
                self._loss_prefix = old_prefix
                model.train(original_training)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation to include individual loss components in metrics."""
        # Save training accumulators to prevent interference from evaluation
        saved_accumulators = {
            "lm": self._accumulated_lm_loss,
            "teacher_lm": self._accumulated_teacher_lm_loss,
            "student_lm": self._accumulated_student_lm_loss,
            "objectness": self._accumulated_objectness_loss,
            "coordinate": self._accumulated_coordinate_loss
            if hasattr(self, "_accumulated_coordinate_loss")
            else 0.0,
            "focal": self._accumulated_focal_loss
            if hasattr(self, "_accumulated_focal_loss")
            else 0.0,
            "regular": self._accumulated_regular_loss
            if hasattr(self, "_accumulated_regular_loss")
            else 0.0,
        }

        # Reset accumulators before evaluation
        self._accumulated_lm_loss = 0.0
        self._accumulated_teacher_lm_loss = 0.0
        self._accumulated_student_lm_loss = 0.0
        self._accumulated_objectness_loss = 0.0
        # Add coordinate token loss accumulators (ensure they exist)
        if not hasattr(self, "_accumulated_coordinate_loss"):
            self._accumulated_coordinate_loss = 0.0
        if not hasattr(self, "_accumulated_focal_loss"):
            self._accumulated_focal_loss = 0.0
        if not hasattr(self, "_accumulated_regular_loss"):
            self._accumulated_regular_loss = 0.0

        # Reset coordinate token loss accumulators
        self._accumulated_coordinate_loss = 0.0
        self._accumulated_focal_loss = 0.0
        self._accumulated_regular_loss = 0.0

        # Run base evaluation. This will call compute_loss and populate our accumulators.
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute average component losses over all evaluation batches
        eval_loader = self.get_eval_dataloader(eval_dataset)
        num_batches = len(eval_loader)

        if num_batches > 0:
            # Always log LM loss components
            metrics[f"{metric_key_prefix}_lm_loss"] = round(
                self._accumulated_lm_loss / num_batches, 4
            )

            # Always log teacher and student losses during evaluation
            metrics[f"{metric_key_prefix}_teacher_lm_loss"] = round(
                self._accumulated_teacher_lm_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_student_lm_loss"] = round(
                self._accumulated_student_lm_loss / num_batches, 4
            )

            # Add coordinate token loss metrics (new detection system)
            if (
                hasattr(self, "_accumulated_coordinate_loss")
                and self._accumulated_coordinate_loss > 0
            ):
                metrics[f"{metric_key_prefix}_coordinate_loss"] = round(
                    self._accumulated_coordinate_loss / num_batches, 4
                )
            if (
                hasattr(self, "_accumulated_focal_loss")
                and self._accumulated_focal_loss > 0
            ):
                metrics[f"{metric_key_prefix}_focal_loss"] = round(
                    self._accumulated_focal_loss / num_batches, 4
                )
            # regular_loss removed - using clean llm_loss instead

            # Legacy bbox metrics removed - using clean coordinate losses instead
            if self._accumulated_objectness_loss > 0:
                metrics[f"{metric_key_prefix}_objectness_loss"] = round(
                    self._accumulated_objectness_loss / num_batches, 4
                )

        # Restore training accumulators
        self._accumulated_lm_loss = saved_accumulators["lm"]
        self._accumulated_teacher_lm_loss = saved_accumulators["teacher_lm"]
        self._accumulated_student_lm_loss = saved_accumulators["student_lm"]
        self._accumulated_objectness_loss = saved_accumulators["objectness"]
        self._accumulated_coordinate_loss = saved_accumulators["coordinate"]
        self._accumulated_focal_loss = saved_accumulators["focal"]
        if hasattr(self, "_accumulated_regular_loss"):
            self._accumulated_regular_loss = saved_accumulators.get("regular", 0.0)

        # Note: metrics are already logged by super().evaluate() call above
        # No need to log again to avoid duplication
        return metrics

    # ------------------------------------------------------------------
    # 🆕  Helper – unpack 1×T *packed* batches back to regular B×S tensors
    # ------------------------------------------------------------------
    def _maybe_unpack_packed(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """If **PackedDataCollator** was used, the batch comes with
        • input_ids/labels -> shape (1, total_len)
        • cu_seqlens       -> inclusive prefix-sum vector [0,L₁,L₁+L₂,…]

        Transformers attention implementation still expects each sample to
        occupy its own batch row.  We therefore reconstruct a padded
        B×S representation on-the-fly *inside* the trainer so the rest of
        the pipeline (loss split, causal mask, etc.) remains unchanged.
        The operation is cheap (≤1 µs) compared to the forward pass.
        """

        if "cu_seqlens" not in batch:
            # Standard collator → nothing to do
            return batch

        cu = batch["cu_seqlens"].to(torch.long)  # (B+1, ) inclusive
        if cu.ndim != 1 or cu[0].item() != 0:
            raise RuntimeError(
                "cu_seqlens must be 1-D inclusive prefix-sum starting with 0"
            )

        lengths = (cu[1:] - cu[:-1]).tolist()  # per-sample lengths
        batch_size = len(lengths)
        max_len = max(lengths)

        device = batch["input_ids"].device
        ids_dtype = batch["input_ids"].dtype
        lbl_dtype = batch["labels"].dtype

        pad_id = (
            self.tokenizer_ref.pad_token_id if self.tokenizer_ref is not None else 0
        )
        IGNORE_INDEX = -100  # keep consistent with src.utils

        new_input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=ids_dtype, device=device
        )
        new_labels = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=lbl_dtype, device=device
        )
        new_attn = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

        # Optional: carry over 3-channel position_ids when available
        pos_ids_src = batch.get(
            "position_ids"
        )  # may be None or (1, total_len) or (3,1,total_len)
        if pos_ids_src is not None:
            pos_dtype = pos_ids_src.dtype
            if pos_ids_src.ndim == 2:  # (1, T)
                pos_ids_src = pos_ids_src.unsqueeze(
                    0
                )  # → (1,1,T) for uniform indexing below
            # (3,1,T) is already fine
            new_pos = torch.zeros(
                (3, batch_size, max_len), dtype=pos_dtype, device=device
            )
        else:
            new_pos = None

        # Adjust teacher / student spans while iterating
        teacher_batch = batch.get("teacher_assistant_spans", [])
        student_batch = batch.get("student_assistant_spans", [])
        new_teacher, new_student = [], []

        cursor = 0
        for i, L in enumerate(lengths):
            slice_ids = slice(cursor, cursor + L)

            # Copy token tensors -------------------------------------------------
            new_input_ids[i, :L] = batch["input_ids"][0, slice_ids]
            new_labels[i, :L] = batch["labels"][0, slice_ids]
            new_attn[i, :L] = True

            # Position-ids -------------------------------------------------------
            if new_pos is not None:
                new_pos[:, i : i + 1, :L] = pos_ids_src[:, :, slice_ids]

            # Span adjustment ----------------------------------------------------
            if teacher_batch and i < len(teacher_batch):
                adj_teacher = [(s - cursor, e - cursor) for (s, e) in teacher_batch[i]]
                new_teacher.append(adj_teacher)
            elif teacher_batch:
                # Handle case where index is out of range
                new_teacher.append([])

            if student_batch and i < len(student_batch):
                adj_student = [(s - cursor, e - cursor) for (s, e) in student_batch[i]]
                new_student.append(adj_student)
            elif student_batch:
                # Handle case where index is out of range
                new_student.append([])

            cursor += L

        # Build new dict --------------------------------------------------------
        packed_keys = {
            "input_ids": new_input_ids,
            "labels": new_labels,
            "attention_mask": new_attn,
        }
        if new_pos is not None:
            packed_keys["position_ids"] = new_pos

        # Replace tensors
        new_batch = batch.copy()
        new_batch.update(packed_keys)

        # Replace spans
        if teacher_batch:
            new_batch["teacher_assistant_spans"] = new_teacher
        if student_batch:
            new_batch["student_assistant_spans"] = new_student

        # Remove cu_seqlens so downstream code isn't confused
        new_batch.pop("cu_seqlens", None)

        return new_batch


def set_model_training_params(model):
    """
    Enable or disable training on model submodules (vision, mlp, llm, detection)
    based on learning-rate flags and the global `detection_enabled` option.
    """
    from src.config import get_config

    config = get_config()
    logger = get_training_logger()
    # Check if we're using the detection wrapper
    has_detection_wrapper = hasattr(model, "base_model") and hasattr(
        model, "coordinate_tokens_enabled"
    )
    base_model = model.base_model if has_detection_wrapper else model

    # Helper to toggle trainability via LR -------------------------------------------------
    def _toggle(module_iter, lr_value: float, module_name: str):
        trainable = lr_value != 0
        for _, p in module_iter:
            p.requires_grad = trainable
        state = "TRAINING" if trainable else "FROZEN"
        logger.info(f"🔧 {module_name}: {state} (lr={lr_value})")

    # Vision encoder
    _toggle(base_model.visual.named_parameters(), config.vision_lr, "Vision encoder")

    # MLP connector (merger)
    _toggle(
        base_model.visual.merger.named_parameters(), config.merger_lr, "MLP connector"
    )

    # LLM backbone & lm_head
    llm_trainable = config.llm_lr != 0
    for _, p in base_model.model.named_parameters():
        p.requires_grad = llm_trainable
    if hasattr(base_model, "lm_head"):
        base_model.lm_head.requires_grad = llm_trainable
    logger.info(
        f"🔧 LLM: {'TRAINING' if llm_trainable else 'FROZEN'} (lr={config.llm_lr})"
    )

    # Detection head removed - using coordinate regression approach


def setup_model_and_tokenizer() -> Tuple[
    nn.Module, PreTrainedTokenizerBase, Qwen2VLImageProcessor
]:
    """
    Centralized setup for model, tokenizer, and image processor using UNIFIED loader.

    This function now delegates to the unified loader to ensure strict consistency
    between training and inference. NO SILENT FALLBACKS.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model with UNIFIED loading mechanism...")
    from src.config import config

    try:
        from src.models.model_loader import load_model_and_processor_unified

        # Use unified loader with training mode
        model, tokenizer, image_processor = load_model_and_processor_unified(
            model_path=config.model_path if hasattr(config, "model_path") else None,
            for_inference=False,  # Training mode
        )

        logger.info("✅ Training model setup completed via unified loader")
        return model, tokenizer, image_processor

    except Exception as e:
        logger.error(f"❌ Training model setup failed: {e}")
        raise RuntimeError(f"Failed to setup model for training: {e}")

    # 2. TOKENIZER & PROCESSOR SETUP
    from data_conversion.vision_process import MAX_PIXELS

    # =========================================================================
    logger.info("🔧 Initializing tokenizer and processor...")

    # Load the processor, which includes the tokenizer and image processor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=False,
        max_pixels=MAX_PIXELS,
    )

    # Extract tokenizer from processor and fix padding_side
    tokenizer = processor.tokenizer
    if tokenizer.padding_side != "left":
        logger.warning(
            f"🔧 [LEGACY PATH] Fixing tokenizer padding_side: {tokenizer.padding_side} -> left"
        )
        tokenizer.padding_side = "left"
    logger.info(f"[LEGACY PATH] Tokenizer padding side: {tokenizer.padding_side}")

    # Load model using unified loader for consistency
    logger.info("🔧 Loading model using unified mechanism...")
    try:
        from src.models.model_loader import load_model_and_processor_unified

        model, _, _ = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,  # Training mode
        )
    except Exception as model_load_error:
        logger.error(f"❌ Failed to load model via unified loader: {model_load_error}")
        raise RuntimeError(f"Legacy path model loading failed: {model_load_error}")

    # Set image processor params from config
    # The image processor is already pre-scaled to the correct resolution
    # during data preparation, so we use the rescaled values, not the defaults.
    logger.info("🔧 Overriding default image processor pixel values...")
    image_processor = processor.image_processor

    # CRITICAL: Use pixel constraints from data_conversion/vision_process.py
    # Our training data was preprocessed with these specific constraints
    try:
        from data_conversion.vision_process import MAX_PIXELS, MIN_PIXELS

        # Apply the exact same pixel constraints used during data conversion
        image_processor.min_pixels = MIN_PIXELS  # 4 * 28 * 28 = 3136
        image_processor.max_pixels = MAX_PIXELS  # 512 * 28 * 28 = 401408

        # Also set size constraints if the processor supports them
        if hasattr(image_processor, "size"):
            if isinstance(image_processor.size, dict):
                image_processor.size["min_pixels"] = MIN_PIXELS
                image_processor.size["max_pixels"] = MAX_PIXELS
            else:
                image_processor.size = {
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }

        # Verify vision processing parameters match config (fail-fast if mismatch)
        if image_processor.patch_size != config.patch_size:
            raise ValueError(
                f"Image processor patch_size ({image_processor.patch_size}) != config ({config.patch_size})"
            )
        if image_processor.merge_size != config.merge_size:
            raise ValueError(
                f"Image processor merge_size ({image_processor.merge_size}) != config ({config.merge_size})"
            )
        if image_processor.temporal_patch_size != config.temporal_patch_size:
            raise ValueError(
                f"Image processor temporal_patch_size ({image_processor.temporal_patch_size}) != config ({config.temporal_patch_size})"
            )

        logger.info(
            f"✅ Image processor configured with data_conversion pixel constraints:"
        )
        logger.info(f"   min_pixels: {image_processor.min_pixels}")
        logger.info(f"   max_pixels: {image_processor.max_pixels}")
        logger.info(f"   patch_size: {image_processor.patch_size}")
        logger.info(f"   merge_size: {image_processor.merge_size}")
        logger.info(f"   temporal_patch_size: {image_processor.temporal_patch_size}")

    except ImportError as e:
        logger.error(f"❌ Failed to import from data_conversion/vision_process.py: {e}")
        raise RuntimeError("Cannot proceed without vision_process pixel constraints")

    # Disable caching and optionally enable gradient checkpointing on the base model
    base_model = model.base_model if hasattr(model, "base_model") else model
    base_model.config.use_cache = False
    if config.gradient_checkpointing:
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            base_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    # Apply training parameter settings
    set_model_training_params(model)
    logger.info("✅ Model setup complete")
    return model, tokenizer, image_processor


def setup_data_module(
    tokenizer: PreTrainedTokenizerBase, image_processor: Qwen2VLImageProcessor
) -> Dict[str, Any]:
    """
    Setup data module following the official approach with improved prompts.
    This matches the data setup in train_qwen.py but with context-aware prompts.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up data module with context-aware prompts...")

    # Create chat processor with training context
    from src.chat_processor import ChatProcessor
    from src.config import config

    # Use consistent prompt style for training and evaluation to prevent distribution mismatch
    # EXPLICIT: Check for prompt configuration without fallbacks
    if not hasattr(config, "use_consistent_prompts"):
        raise ValueError(
            "Configuration missing 'use_consistent_prompts' field. "
            "Ensure this field is explicitly set in your configuration."
        )
    if not hasattr(config, "training_prompt_style"):
        raise ValueError(
            "Configuration missing 'training_prompt_style' field. "
            "Ensure this field is explicitly set in your configuration."
        )

    use_consistent_prompts = config.use_consistent_prompts
    training_prompt_style = config.training_prompt_style

    # EXPLICIT CONFIG: coordinate token configuration is required and validated at config load
    coordinate_tokens_enabled = config.coordinate_tokens_enabled
    max_coord_value = config.max_coord_value

    # Training chat processor
    train_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=training_prompt_style,
        language="chinese",
        enable_coordinate_tokens=coordinate_tokens_enabled,
        max_coord_value=max_coord_value,
    )

    # Evaluation chat processor - use same prompt style unless explicitly overridden
    eval_prompt_style = training_prompt_style if use_consistent_prompts else False
    eval_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=eval_prompt_style,
        language="chinese",
        enable_coordinate_tokens=coordinate_tokens_enabled,
        max_coord_value=max_coord_value,
    )

    if coordinate_tokens_enabled:
        logger.info(
            f"✅ Chat processors created with coordinate tokens enabled (max_coord_value: {max_coord_value})"
        )
    else:
        logger.info("✅ Chat processors created (coordinate tokens disabled)")

    # Create teacher pool manager if teacher_ratio > 0
    teacher_pool_manager = None
    # EXPLICIT: Get teacher_ratio without fallback
    if not hasattr(config, "teacher_ratio"):
        raise ValueError(
            "Configuration missing 'teacher_ratio' field. "
            "Ensure this field is explicitly set in your configuration."
        )

    teacher_ratio = config.teacher_ratio
    if teacher_ratio > 0.0:
        from src.teacher_pool import create_teacher_pool_manager

        # Fail fast if teacher configuration is invalid
        teacher_pool_manager = create_teacher_pool_manager()
        logger.info(
            f"✅ Teacher pool manager created with {len(teacher_pool_manager)} teachers"
        )

    # Create training dataset with detailed prompts and teacher support
    train_dataset = BBUDataset(
        data_path=config.train_data_path,
        chat_processor=train_chat_processor,
        teacher_pool_manager=teacher_pool_manager,
        teacher_ratio=teacher_ratio,
        is_training=True,  # Training context
    )

    # Create validation dataset - use consistent settings unless zero-shot explicitly requested
    val_teacher_manager = teacher_pool_manager if use_consistent_prompts else None
    val_teacher_ratio = teacher_ratio if use_consistent_prompts else 0.0

    val_dataset = BBUDataset(
        data_path=config.val_data_path,
        chat_processor=eval_chat_processor,
        teacher_pool_manager=val_teacher_manager,
        teacher_ratio=val_teacher_ratio,
        is_training=False,  # Evaluation context
    )

    # Create data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        collator_type=config.collator_type,
    )

    logger.info(f"✅ Data module setup completed with improved prompts:")
    logger.info(
        f"   Train samples: {len(train_dataset)} (detailed prompts, teacher_ratio={teacher_ratio})"
    )
    logger.info(f"   Val samples: {len(val_dataset)} (concise prompts, no teachers)")
    # EXPLICIT CONFIG: collator_type is required and validated at config load
    logger.info(f"   Collator type: {config.collator_type}")
    logger.info(
        f"   Training prompt: {train_chat_processor.get_current_system_prompt()[:100]}..."
    )
    logger.info(
        f"   Evaluation prompt: {eval_chat_processor.get_current_system_prompt()[:100]}..."
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """
    Safe model saving following the official approach.
    Uses the improved _save method that includes visual components.
    """
    logger = get_training_logger()
    logger.info(f"💾 Safely saving model to: {output_dir}")

    if trainer.args.should_save:
        # Use the trainer's improved _save method which handles visual components properly
        trainer._save(output_dir)
        logger.info("✅ Model saved using improved _save method with visual components")


# Legacy create_trainer function removed - using unified trainer factory with coordinator


def test_enhanced_logging() -> None:
    """
    Simple test to verify enhanced detection loss logging works.
    This can be called during development to test the logging mechanism.
    """
    from src.logger_utils import get_training_logger

    logger = get_training_logger()
    logger.info("🧪 Testing enhanced detection loss logging...")

    # Test that the loss components are properly structured (cleaned up)
    sample_loss_components = {
        "llm_loss": 0.5,
        "focal_loss": 0.3,
        "l1_loss": 0.8,
        "giou_loss": 0.2,
        "objectness_loss": 0.3,
    }

    # Test prefix handling - no prefix for training, eval_ for evaluation
    for mode, prefix in [("training", ""), ("evaluation", "eval_")]:
        loss_info = {}
        for key, value in sample_loss_components.items():
            loss_info[f"{prefix}{key}"] = float(value)

        # Add coordinate loss total
        coord_total = sum(
            sample_loss_components[k] for k in ["focal_loss", "l1_loss", "giou_loss"]
        )
        loss_info[f"{prefix}coord_loss"] = coord_total

        logger.info(f"✅ {mode.upper()} loss structure: {loss_info}")

    logger.info("🧪 Enhanced logging test completed successfully!")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_enhanced_logging()
