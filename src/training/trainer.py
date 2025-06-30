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

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import config
from src.config.global_config import DirectConfig
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches
from src.schema import GroundTruthObject

from ..tokens.special_tokens import SpecialTokens


class BBUTrainer(Trainer):
    """
    Custom trainer that integrates object detection loss when configured.

    Extends the standard Transformers Trainer to add object detection capabilities
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
        cfg: Optional[DirectConfig] = None,
        image_processor: Optional[Qwen2VLImageProcessor] = None,
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

        tokenizer_ref = kwargs.get("tokenizer")

        # ------------------------------------------------------------------
        # Resolve configuration
        # Priority:
        #   1) Explicit `cfg` argument passed by caller
        #   2) Global singleton initialised via src.config.init_config()
        # Fail fast if neither is available.
        # ------------------------------------------------------------------
        from src.config import get_config

        # Attempt to fetch already-initialised global configuration, if any.
        try:
            _global_cfg: Optional[DirectConfig] = get_config()
        except RuntimeError:
            _global_cfg = None

        if cfg is not None:
            self.config = cfg
        elif _global_cfg is not None:
            self.config = _global_cfg
        else:
            raise RuntimeError(
                "DirectConfig not provided to BBUTrainer and global config has "
                "not been initialised. Call src.config.init_config() before "
                "creating the trainer or pass cfg=<DirectConfig>."
            )

        super().__init__(*args, **kwargs)

        # Overwrite the (deprecated) property with a direct attribute so
        # future accesses skip the warning-emitting property defined in the
        # parent class. We bypass the descriptor protocol via
        # `object.__setattr__` to avoid invoking the original setter.
        if tokenizer_ref is None and hasattr(self, "tokenizer"):
            tokenizer_ref = object.__getattribute__(self, "tokenizer")

        object.__setattr__(self, "tokenizer", tokenizer_ref)

        self.logger = get_training_logger()
        self.image_processor = image_processor

        # Simple attributes to store the latest loss components from a single forward pass
        self._current_lm_loss: float = 0.0
        self._current_teacher_lm_loss: float = 0.0
        self._current_student_lm_loss: float = 0.0
        self._current_bbox_loss: float = 0.0
        self._current_caption_loss: float = 0.0
        self._current_objectness_loss: float = 0.0
        self._current_bbox_l1_loss: float = 0.0
        self._current_bbox_giou_loss: float = 0.0

        # ACCUMULATORS for per-step average logging with gradient accumulation
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_teacher_lm_loss: float = 0.0
        self._accumulated_student_lm_loss: float = 0.0
        self._accumulated_bbox_l1_loss: float = 0.0
        self._accumulated_bbox_giou_loss: float = 0.0
        self._accumulated_caption_loss: float = 0.0
        self._accumulated_objectness_loss: float = 0.0

        # Counter for the number of *micro-batches* processed since the last log.
        # This allows us to report true per-batch averages even when `logging_steps`
        # spans multiple optimizer steps (and thus multiple gradient-accumulation
        # cycles).
        self._micro_batch_count: int = 0

        # Initialize object detection loss if configured
        self.detection_loss = None
        if self.config.detection_enabled:
            self._init_detection_loss()

        # Cache for per-step weight / grad norms (populated in training_step)
        self._norm_cache: Dict[str, float] = {}

        # After initialization, we can safely access the tokenizer
        # and add our special tokens. This is a critical step.
        if self.tokenizer:
            special_tokens = SpecialTokens()
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens.to_list()}
            )
            if num_added > 0:
                self.logger.info(
                    f"✅ Added {num_added} special tokens to the tokenizer."
                )
                # Important: Resize token embeddings in the model
                self.model.resize_token_embeddings(len(self.tokenizer))
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
        2. If a detection head is present **and enabled**, save its weights
           separately as `detection_head.pth` plus a small JSON config.
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
            hasattr(self.model, "detection_enabled")
            and not self.model.detection_enabled
        ):
            # For detection_enabled=False, save the full Qwen2.5-VL model
            self.logger.info(
                "💾 Saving complete Qwen2.5-VL model with visual tower (sharded)…"
            )
            model_to_save = self.model
        else:
            # For detection wrapper, save base model + detection head separately
            self.logger.info("💾 Saving base Qwen2.5-VL model (sharded)…")
            model_to_save = getattr(self.model, "base_model", self.model)

        model_to_save.save_pretrained(
            output_dir, max_shard_size="2GB", safe_serialization=True
        )

        # --- 2. Save tokenizer & processor ---------------------------------
        if self.tokenizer is not None:
            self.logger.info("💾 Saving tokenizer...")
            self.tokenizer.save_pretrained(output_dir)

        if self.image_processor is None:
            raise RuntimeError(
                "Image processor is None - cannot save preprocessor config!"
            )

        self.logger.info("💾 Saving image processor...")

        # Load base config from pretrained model and only override specific values
        # This prevents corruption of other config values
        from src.config import config as global_config

        base_model_path = global_config.model_path
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

        with open(
            os.path.join(output_dir, "preprocessor_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ip_cfg, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"   ✅ Image processor config saved with {len(ip_cfg)} parameters (preserving base config)"
        )

        # --- 3. Save detection head (if any) -------------------------------
        if getattr(self.model, "detection_enabled", False) and hasattr(
            self.model, "detection_head"
        ):
            self._save_detection_components(output_dir)

        # --- 4. Copy essential files from base model -------------------
        self._copy_essential_files_from_base_model(output_dir)

        # --- 5. Save training args ----------------------------------------
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # --- 6. Verify visual components were saved -------------------
        self._verify_saved_checkpoint(output_dir)

        self.logger.info(f"✅ Checkpoint saved successfully → {output_dir}")

    def _copy_essential_files_from_base_model(self, output_dir: str) -> None:
        """Copy essential files from base model to match pretrained model structure."""
        import os
        import shutil

        from src.config import config as global_config

        self.logger.info("💾 Copying essential files from base model...")

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
        for file_name in essential_files:
            src_path = os.path.join(base_model_path, file_name)
            dst_path = os.path.join(output_dir, file_name)

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
            src_path = os.path.join(base_model_path, file_name)
            dst_path = os.path.join(output_dir, file_name)

            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    shutil.copy2(src_path, dst_path)
                    self.logger.info(f"   ✅ Copied {file_name} (optional)")
                except Exception as e:
                    self.logger.info(
                        f"   ℹ️ Failed to copy optional file {file_name}: {e}"
                    )
            elif os.path.exists(dst_path):
                self.logger.info(f"   ✅ {file_name} already exists (optional)")
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
            all_keys = f.keys()
            total_keys = len(list(all_keys))

            # Re-open to iterate (safe_open keys() is a generator)
            with safe_open(first_file, framework="pt", device="cpu") as f2:
                for key in f2.keys():
                    if "visual" in key:
                        visual_keys.append(key)

        self.logger.info(message=f"   ✅ Total parameters saved: {total_keys}")

        # FAIL FAST: Vision tower must be present for vision-language model
        if not visual_keys:
            raise RuntimeError(
                "No visual tower parameters found in checkpoint! "
                "This indicates the model saving failed to include vision components. "
                "Vision-language models require visual tower weights for proper inference."
            )

        self.logger.info(f"   ✅ Visual tower parameters found: {len(visual_keys)}")
        self.logger.info(f"      Examples: {visual_keys[:3]}...")

    def _save_detection_components(self, output_dir: str) -> None:
        """Save detection head weights and configuration."""
        import json
        import os

        import torch

        # Save detection head weights
        detection_state_dict = self.model.detection_head.state_dict()
        detection_path = os.path.join(output_dir, "detection_head.pth")
        torch.save(detection_state_dict, detection_path)

        # ------------------------------------------------------------------
        # Save a CLEAN `preprocessor_config.json` for reload. We start from the
        # base model's pristine file and only override the pixel limits the
        # user may have tweaked. This prevents invalid key combinations that
        # trigger `get_size_dict` errors at load time.
        # ------------------------------------------------------------------

        if self.image_processor:
            base_model_dir = getattr(self.model.base_model, "name_or_path", None)
            src_preproc = (
                os.path.join(base_model_dir, "preprocessor_config.json")
                if base_model_dir
                else None
            )
            dst_preproc = os.path.join(output_dir, "preprocessor_config.json")

            if src_preproc and os.path.exists(src_preproc):
                with open(src_preproc, "r", encoding="utf-8") as f:
                    preproc_cfg = json.load(f)
            else:
                preproc_cfg = {}

            # Update only the pixel constraints the training pipeline may have
            # changed. Keep other keys untouched and REMOVE any nested `size`
            # dict which is not accepted by HF utils.
            if hasattr(self.image_processor, "min_pixels"):
                preproc_cfg["min_pixels"] = int(self.image_processor.min_pixels)
            if hasattr(self.image_processor, "max_pixels"):
                preproc_cfg["max_pixels"] = int(self.image_processor.max_pixels)

            # Ensure allowed size keys only. Prefer `longest_edge` if user set
            # something via `.size`.
            if "size" in preproc_cfg:
                size_dict = preproc_cfg["size"]
                clean_size = {}
                for key in (
                    "height",
                    "width",
                    "shortest_edge",
                    "longest_edge",
                    "max_height",
                    "max_width",
                ):
                    if key in size_dict:
                        clean_size[key] = size_dict[key]
                        break  # keep only the first valid key set
                if clean_size:
                    preproc_cfg["size"] = clean_size
                else:
                    preproc_cfg.pop("size")

            with open(dst_preproc, "w", encoding="utf-8") as f:
                json.dump(preproc_cfg, f, indent=2, ensure_ascii=False)

            self.logger.info("💾 Cleaned preprocessor_config.json saved.")

        # Save detection head configuration with UNIFIED filename
        detection_config = {
            "num_queries": self.model.detection_head.num_queries,
            "max_caption_length": self.model.detection_head.max_caption_length,
            "hidden_size": self.model.detection_head.hidden_size,
            "vocab_size": self.model.detection_head.vocab_size,
            "detection_enabled": True,
            "checkpoint_type": "unified",  # Marker for unified checkpoint
        }

        # Use the UNIFIED config filename (not legacy)
        config_path = os.path.join(output_dir, "detection_config.json")
        with open(config_path, "w") as f:
            json.dump(detection_config, f, indent=2)

        # ------------------------------------------------------------------
        # Ensure essential HF files exist (config.json, generation_config.json).
        # If the wrapper model cannot create them automatically (because it is
        # not a PreTrainedModel), copy them from the original base model dir so
        # that `from_pretrained()` works without manual intervention.
        # ------------------------------------------------------------------

        base_model_dir = getattr(self.model.base_model, "name_or_path", None)
        if base_model_dir and os.path.isdir(base_model_dir):
            for fname in ["config.json", "generation_config.json"]:
                src = os.path.join(base_model_dir, fname)
                dst = os.path.join(output_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        import shutil

                        shutil.copy(src, dst)
                        self.logger.info(
                            f"💾 Copied missing {fname} from base model directory."
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to copy {fname}: {e}")

        # Fail-fast: verify that essential HF files are present post-save. This
        # guards against broken checkpoints that would later crash
        # `from_pretrained()` during evaluation or inference.
        for critical in ["config.json", "generation_config.json"]:
            critical_path = os.path.join(output_dir, critical)
            assert os.path.exists(critical_path), (
                f"Checkpoint incomplete – expected {critical_path} to exist. "
                "Make sure save_pretrained() produced the file or it was copied "
                "from the base model directory."
            )

        self.logger.info(f"💾 Detection head saved to: {detection_path}")
        self.logger.info(f"💾 Detection config saved to: {config_path}")

    def _init_detection_loss(self) -> None:
        """Initialize object detection loss with config parameters."""
        from src.detection_loss import DetectionLoss

        # Initialize with tokenizer for caption loss computation
        self.detection_loss = DetectionLoss(
            bbox_weight=self.config.detection_bbox_weight,
            giou_weight=self.config.detection_giou_weight,
            objectness_weight=self.config.detection_objectness_weight,
            caption_weight=self.config.detection_caption_weight,
            tokenizer=self.tokenizer,
            focal_loss_gamma=self.config.detection_focal_loss_gamma,
            focal_loss_alpha=self.config.detection_focal_loss_alpha,
        )

        self.logger.info("🎯 Detection loss initialized in BBUTrainer")
        self.logger.info(f"   bbox_weight: {self.config.detection_bbox_weight}")
        self.logger.info(f"   giou_weight: {self.config.detection_giou_weight}")
        self.logger.info(
            f"   objectness_weight: {self.config.detection_objectness_weight}"
        )
        self.logger.info(f"   caption_weight: {self.config.detection_caption_weight}")
        self.logger.info(
            f"   focal_loss_gamma: {self.config.detection_focal_loss_gamma}"
        )
        self.logger.info(
            f"   focal_loss_alpha: {self.config.detection_focal_loss_alpha}"
        )

    def init_param_groups(self) -> None:
        """
        Initializes parameter groups for differential learning rate.

        This method categorizes all trainable parameters into 'vision', 'merger',
        'llm', and 'detection' groups. It will raise a ValueError if any
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
            "detection": [],
            "adapter": [],  # detection adapters (vision & lang)
            "others": [],  # For uncategorized parameters
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Correct parameter name matching based on the model's structure.
            # The order is critical: check for the most specific names first.
            if "detection_head" in name:
                if ".adapter" in name:
                    param_groups_with_names["adapter"].append((name, param))
                else:
                    param_groups_with_names["detection"].append((name, param))
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
                "assigned to a learning rate group (vision, merger, llm, detection)."
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

    def create_optimizer(self) -> Optimizer:
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
            "detection": self.config.detection_lr,
            "adapter": getattr(self.config, "adapter_lr", self.config.detection_lr),
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

        if getattr(self, "_optimizer_step_wrapped", False):
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

        module_map = {
            "vision_adapter": "detection_head.vision_adapter",
            "lang_adapter": "detection_head.adapter",
            "bbox_head": "detection_head.bbox_head",
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
    ) -> tuple[float, float]:
        """
        Compute separate losses for teacher and student spans.

        Uses the same shifting logic as the total LM loss computation,
        so the returned losses are directly comparable to the total LM loss.

        Args:
            logits: Model output logits [batch_size, sequence_length, vocab_size]
            labels: Ground truth labels [batch_size, sequence_length]
            inputs: Batch inputs containing span information

        Returns:
            Tuple of (teacher_loss, student_loss) as float values
        """
        if labels is None:
            return 0.0, 0.0

        # Get spans from inputs (may be None for backward compatibility)
        teacher_spans = inputs.get("teacher_assistant_spans")
        student_spans = inputs.get("student_assistant_spans")

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

        # Compute teacher loss
        teacher_loss = 0.0
        if teacher_indices:
            teacher_indices_tensor = torch.tensor(teacher_indices, device=logits.device)
            teacher_logits = flat_logits[teacher_indices_tensor]
            teacher_labels = flat_labels[teacher_indices_tensor]

            # Only compute loss on tokens that are not ignored
            valid_mask = teacher_labels != -100
            if valid_mask.any():
                # Use mean reduction directly - no manual averaging needed
                teacher_loss = loss_fn(
                    teacher_logits[valid_mask], teacher_labels[valid_mask]
                ).item()

        # Compute student loss
        student_loss = 0.0
        if student_indices:
            student_indices_tensor = torch.tensor(student_indices, device=logits.device)
            student_logits = flat_logits[student_indices_tensor]
            student_labels = flat_labels[student_indices_tensor]

            # Only compute loss on tokens that are not ignored
            valid_mask = student_labels != -100
            if valid_mask.any():
                # Use mean reduction directly - no manual averaging needed
                student_loss = loss_fn(
                    student_logits[valid_mask], student_labels[valid_mask]
                ).item()

        return teacher_loss, student_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss with end-to-end object detection integration.
        The Trainer handles loss accumulation and averaging, so we just
        return the final combined loss. Individual components are stored
        as attributes for logging.
        """
        # ------------------------------------------------------------------
        # Increment *micro-batch* counter **before** any early returns so that
        # every forward pass is accounted for. We only count batches during
        # training to avoid contaminating the counter with evaluation steps.
        # ------------------------------------------------------------------
        if model.training:
            self._micro_batch_count += 1

        # Unpack 1×T packed batches (if any) back to B×S for correct masking
        inputs = self._maybe_unpack_packed(inputs)

        # Extract and store GT objects separately
        ground_truth_objects = self._extract_ground_truth_objects(inputs)

        # Fail-fast sanity check: detection enabled but batch has no GT boxes
        if self.config.detection_enabled and not any(
            len(gt) > 0 for gt in ground_truth_objects
        ):
            raise ValueError(
                "Detection training is enabled but the current batch contains no ground-truth objects. "
                "Verify that your dataset JSON provides the 'objects' field for every sample."
            )

        # Prepare clean inputs for model (remove GT objects)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)
        model_inputs.pop("cu_seqlens", None)
        # Remove teacher-student span fields - these are for trainer use only, not model input
        model_inputs.pop("teacher_assistant_spans", None)
        model_inputs.pop("student_assistant_spans", None)

        # Ensure we get hidden states for detection
        model_inputs["output_hidden_states"] = True

        outputs = model(**model_inputs)

        # Debug: catch NaN LM loss and log sample information
        lm_loss = outputs.loss
        if torch.isnan(lm_loss):
            # Count valid labels (not IGNORE_INDEX = -100)
            labels = inputs["labels"]
            valid_label_count = (
                int((labels != -100).sum().item()) if labels is not None else 0
            )
            self.logger.error(
                f"🚨 Detected NaN LM loss. Valid labels in batch: {valid_label_count}"
            )
            # Log first sample tokens and labels
            if labels is not None:
                sample_input = (
                    inputs["input_ids"][0].tolist()
                    if inputs.get("input_ids") is not None
                    else []
                )
                sample_labels = labels[0].tolist() if labels is not None else []
                self.logger.error(f"🚨 input_ids: {sample_input}")
                self.logger.error(f"🚨 labels: {sample_labels}")
                # Decode full conversation
                try:
                    full_text = self.tokenizer.decode(
                        sample_input, skip_special_tokens=False
                    )
                    self.logger.error(f"🚨 Full text: {full_text}")
                except Exception as e:
                    self.logger.error(f"🚨 Error decoding tokens: {e}")
            # Raise error to halt training for debugging
            raise RuntimeError("LM loss is NaN, see previous error logs for details.")
        self._current_lm_loss = lm_loss.item()
        self._accumulated_lm_loss += self._current_lm_loss

        # ------------------------------------------------------------------
        # Teacher-Student Loss Splitting
        # ------------------------------------------------------------------
        (
            self._current_teacher_lm_loss,
            self._current_student_lm_loss,
        ) = self._compute_teacher_student_losses(
            outputs.logits, inputs.get("labels"), inputs
        )
        self._accumulated_teacher_lm_loss += self._current_teacher_lm_loss
        self._accumulated_student_lm_loss += self._current_student_lm_loss

        # Detection loss computation
        total_detection_loss = 0.0

        # ------------------------------------------------------------------
        # Dynamic detection head freeze: optionally skip detection loss during
        # the first `detection_freeze_epochs` to let the caption head warm up.
        # ------------------------------------------------------------------

        # Enable detection head training from the first epoch
        detection_training_enabled = self.config.detection_enabled

        if (
            detection_training_enabled
            and self.detection_loss is not None
            and ground_truth_objects
            and any(len(gt) > 0 for gt in ground_truth_objects)
        ):
            # Prepare full list of LLM hidden states for detection
            hidden_states_list = list(
                outputs.hidden_states
            )  # convert tuple to list for shape checking
            # --------------------------------------------------------------
            # 🔄  Re-use vision features computed *inside* the Qwen2.5-VL
            # forward pass (exposed via the new `vision_embeds` attribute).
            # This removes the need to call `model.base_model.visual(...)`
            # a second time and therefore keeps a single, clean autograd
            # graph flowing from both lm_loss *and* detection_loss back to
            # the vision tower.
            # --------------------------------------------------------------

            vision_feats = getattr(outputs, "vision_embeds", None)

            # Fail-fast: the detection head requires vision features computed
            # inside the main forward pass.  If they are not present we raise
            # an explicit error instead of silently re-computing them, which
            # would create a second autograd graph and likely break ZeRO /
            # gradient checkpointing.
            if vision_feats is None:
                raise RuntimeError(
                    "Detection is enabled but `vision_embeds` are missing from the model output. "
                    "Ensure that the patched Qwen2.5-VL forward method returns the vision features "
                    "(see modeling_qwen2_5_vl.py) before enabling detection training."
                )

            # Detection head will pick and adapt the correct memory internally
            detection_outputs = model.detection_head(
                hidden_states_list,
                attention_mask=model_inputs.get("attention_mask"),
                vision_feats=vision_feats,
                ground_truth_objects=ground_truth_objects,
                training=model.training,
            )

            detection_loss_components = self.detection_loss(
                detection_outputs, ground_truth_objects
            )

            # The detection_loss module returns weighted losses
            total_detection_loss = detection_loss_components["total_loss"]

            # Store unweighted components for logging and accumulate them
            self._current_bbox_loss = self.detection_loss.last_bbox_loss.item()
            self._current_bbox_l1_loss = self.detection_loss.last_l1_loss.item()
            self._current_bbox_giou_loss = self.detection_loss.last_giou_loss.item()
            self._current_caption_loss = self.detection_loss.last_caption_loss.item()
            self._current_objectness_loss = (
                self.detection_loss.last_objectness_loss.item()
            )
            self._accumulated_caption_loss += self._current_caption_loss
            self._accumulated_objectness_loss += self._current_objectness_loss
            self._accumulated_bbox_l1_loss += self._current_bbox_l1_loss
            self._accumulated_bbox_giou_loss += self._current_bbox_giou_loss
        else:
            # Reset detection losses if not computed
            self._current_bbox_loss = 0.0
            self._current_bbox_l1_loss = 0.0
            self._current_bbox_giou_loss = 0.0
            self._current_caption_loss = 0.0
            self._current_objectness_loss = 0.0

        # Total loss for backpropagation
        total_loss = lm_loss + total_detection_loss

        # ------------------------------------------------------------------
        # One-off debug: show the raw chat template for the FIRST train and
        # FIRST eval step.  This helps verify that the data-pipeline inserts
        # system / user / assistant roles and vision tokens correctly.
        # ------------------------------------------------------------------
        def _log_sample_once(mode: str):
            flag_name = f"_debug_sample_logged_{mode}"
            if getattr(self, flag_name, False):
                return

            setattr(self, flag_name, True)

            # Take the 0-th sample of the current batch
            sample_ids = inputs["input_ids"][0].tolist()
            sample_labels = inputs["labels"][0].tolist() if "labels" in inputs else None

            # Decode full sequence with special tokens
            full_text = self.tokenizer.decode(sample_ids, skip_special_tokens=False)

            # Replace long runs of <IMAGE_PAD>
            pad_token = "<|image_pad|>"

            # --- Enhanced compression: merge ANY whitespace-separated run of <|image_pad|> ---
            pad_pattern = rf"(?:\s*{re.escape(pad_token)}\s*)+"

            def _compress_pad(match):
                n = match.group(0).count(pad_token)
                # Preserve a single leading space if the run started with one so that tokens stay separated.
                prefix_space = " " if match.group(0).startswith(" ") else ""
                return f"{prefix_space}{pad_token}*{n}"

            full_text = re.sub(pad_pattern, _compress_pad, full_text)

            # ------------------------------------------------------------------
            # ALSO compress long runs of <|endoftext|> for readability ---------
            # ------------------------------------------------------------------

            eot_token = SpecialTokens.ENDOFTEXT
            eot_pattern = rf"(?:\s*{re.escape(eot_token)}\s*)+"

            def _compress_eot(match):
                n = match.group(0).count(eot_token)
                prefix_space = " " if match.group(0).startswith(" ") else ""
                return f"{prefix_space}{eot_token}*{n}"

            full_text = re.sub(eot_pattern, _compress_eot, full_text)

            # Readability: newline before each assistant turn in the *full* text
            full_text = full_text.replace(
                "<|im_start|>assistant",
                "\n<|im_start|>assistant",
            )

            # Build target string (tokens where label != IGNORE_INDEX)
            target_text = ""
            if sample_labels is not None:
                tgt_ids = [
                    tid for tid, lab in zip(sample_ids, sample_labels) if lab != -100
                ]
                if tgt_ids:
                    target_text = self.tokenizer.decode(
                        tgt_ids, skip_special_tokens=False
                    )

                    # Apply the same <|endoftext|> compression to target_text
                    target_text = re.sub(eot_pattern, _compress_eot, target_text)

                    # Insert a newline before every assistant turn for
                    # readability.  We know the literal marker in the chat
                    # template is "<|im_start|>assistant" so we can operate on
                    # the decoded string directly without converting token
                    # IDs back and forth.
                    target_text = target_text.replace(
                        "<|im_start|>assistant",
                        "\n<|im_start|>assistant",
                    )
            self.logger.info("📝 ===== Sample conversation ({}) =====".format(mode))
            self.logger.info(full_text)
            if target_text:
                self.logger.info("📝 Teacher and target answers (labels≠IGNORE):")
                self.logger.info(target_text)
            self.logger.info("📝 ================================")

        # Log once per mode
        _log_sample_once("train" if model.training else "eval")

        return (total_loss, outputs) if return_outputs else total_loss

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
            # The `tr_loss` from the Trainer is an accumulated value.
            # We re-compute the loss from our own averaged components to ensure
            # correct, per-step reporting consistent with the docstring.
            # Average the accumulated component losses
            num_micro_batches = max(1, self._micro_batch_count)

            # Average the accumulated component losses across *all* micro-batches
            # seen since the last log, independent of the gradient-accumulation
            # configuration.
            avg_lm_loss = self._accumulated_lm_loss / num_micro_batches
            avg_teacher_lm_loss = self._accumulated_teacher_lm_loss / num_micro_batches
            avg_student_lm_loss = self._accumulated_student_lm_loss / num_micro_batches
            total_avg_loss = avg_lm_loss

            component_logs: Dict[str, float] = {
                "lm_loss": avg_lm_loss,
                "teacher_lm_loss": avg_teacher_lm_loss,
                "student_lm_loss": avg_student_lm_loss,
            }

            if self.config.detection_enabled:
                avg_bbox_l1_loss = self._accumulated_bbox_l1_loss / num_micro_batches
                avg_bbox_giou_loss = (
                    self._accumulated_bbox_giou_loss / num_micro_batches
                )
                avg_caption_loss = self._accumulated_caption_loss / num_micro_batches
                avg_objectness_loss = (
                    self._accumulated_objectness_loss / num_micro_batches
                )

                component_logs["bbox_l1_loss"] = avg_bbox_l1_loss
                component_logs["bbox_giou_loss"] = avg_bbox_giou_loss
                component_logs["caption_loss"] = avg_caption_loss
                component_logs["objectness_loss"] = avg_objectness_loss

                # Reconstruct the total detection loss from its averaged, weighted components.
                # This ensures the final logged 'loss' accurately reflects the value used for backprop.
                # NOTE: This assumes the accumulated components are WEIGHTED. If they are not,
                # this sum will not match the true loss.

                # The `loss` passed to compute_loss is lm_loss + weighted detection loss
                # The total loss for logging should be calculated from averaged components.
                # Here we assume the logged components are the primary ones.
                total_avg_loss += (
                    avg_bbox_l1_loss
                    + avg_bbox_giou_loss
                    + avg_caption_loss
                    + avg_objectness_loss
                )

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
                module_names = {
                    "vision_adapter": "detection_head.vision_adapter",
                    "lang_adapter": "detection_head.adapter",
                    "bbox_head": "detection_head.bbox_head",
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

            # Reset accumulators after logging
            self._accumulated_lm_loss = 0.0
            self._accumulated_teacher_lm_loss = 0.0
            self._accumulated_student_lm_loss = 0.0
            self._accumulated_caption_loss = 0.0
            self._accumulated_objectness_loss = 0.0
            self._accumulated_bbox_l1_loss = 0.0
            self._accumulated_bbox_giou_loss = 0.0

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

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        This method is overridden to support logging of differential learning rates.
        """
        # Remove the generic learning rate from the logs.
        logs.pop("learning_rate", None)

        # Log the learning rate for each parameter group.
        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()
            for i, group_lr in enumerate(last_lr):
                group_name = self._param_names[i]
                logs[f"lr/{group_name}"] = group_lr

        super().log(logs)

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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Enhanced prediction step that includes detection loss logging during evaluation.
        """
        # Store original state for restoration
        original_training = model.training

        # Set model to eval mode
        model.eval()

        # Use the same compute_loss logic but with eval prefix
        with torch.no_grad():
            # Temporarily modify the loss info prefix for evaluation
            old_prefix = getattr(self, "_loss_prefix", "")
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
                    else:
                        logits = (
                            outputs[1:] if hasattr(outputs, "__getitem__") else outputs
                        )

                    # Extract labels if available
                    labels = None
                    if hasattr(self, "label_names") and len(self.label_names) > 0:
                        labels = tuple(inputs.get(name) for name in self.label_names)
                        if len(labels) == 1:
                            labels = labels[0]

                    return (loss, logits, labels)

            finally:
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
            "bbox_l1": self._accumulated_bbox_l1_loss,
            "bbox_giou": self._accumulated_bbox_giou_loss,
            "caption": self._accumulated_caption_loss,
            "objectness": self._accumulated_objectness_loss,
        }

        # Reset accumulators before evaluation
        self._accumulated_lm_loss = 0.0
        self._accumulated_teacher_lm_loss = 0.0
        self._accumulated_student_lm_loss = 0.0
        self._accumulated_caption_loss = 0.0
        self._accumulated_objectness_loss = 0.0

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
            metrics[f"{metric_key_prefix}_lm_loss"] = round(
                self._accumulated_lm_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_lm_teacher_loss"] = round(
                self._accumulated_teacher_lm_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_lm_student_loss"] = round(
                self._accumulated_student_lm_loss / num_batches, 4
            )
            if self.config.detection_enabled:
                metrics[f"{metric_key_prefix}_bbox_l1_loss"] = round(
                    self._accumulated_bbox_l1_loss / num_batches, 4
                )
                metrics[f"{metric_key_prefix}_bbox_giou_loss"] = round(
                    self._accumulated_bbox_giou_loss / num_batches, 4
                )
                metrics[f"{metric_key_prefix}_caption_loss"] = round(
                    self._accumulated_caption_loss / num_batches, 4
                )
                metrics[f"{metric_key_prefix}_objectness_loss"] = round(
                    self._accumulated_objectness_loss / num_batches, 4
                )

        # Restore training accumulators
        self._accumulated_lm_loss = saved_accumulators["lm"]
        self._accumulated_teacher_lm_loss = saved_accumulators["teacher_lm"]
        self._accumulated_student_lm_loss = saved_accumulators["student_lm"]
        self._accumulated_caption_loss = saved_accumulators["caption"]
        self._accumulated_objectness_loss = saved_accumulators["objectness"]

        # Log extended metrics
        self.log(metrics)
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

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
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
            if teacher_batch:
                adj_teacher = [(s - cursor, e - cursor) for (s, e) in teacher_batch[i]]
                new_teacher.append(adj_teacher)
            if student_batch:
                adj_student = [(s - cursor, e - cursor) for (s, e) in student_batch[i]]
                new_student.append(adj_student)

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
    logger = get_training_logger()
    has_detection_head = hasattr(model, "detection_head")
    base_model = model.base_model if has_detection_head else model

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

    # Detection head (optional)
    if config.detection_enabled and has_detection_head:
        _toggle(
            model.detection_head.named_parameters(),
            config.detection_lr,
            "Detection head",
        )
    else:
        logger.info("🔧 Detection head: DISABLED (config.detection_enabled False)")


def setup_model_and_tokenizer() -> Tuple[
    nn.Module, PreTrainedTokenizerBase, Qwen2VLImageProcessor
]:
    """
    Centralized setup for model, tokenizer, and image processor.
    - Applies necessary patches for Qwen2.5VL.
    - Initializes tokenizer with custom chat template and special tokens.
    - Initializes the model with appropriate quantization and settings.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model with unified loading mechanism...")

    # Apply comprehensive Qwen2.5-VL fixes FIRST
    logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
    if not apply_comprehensive_qwen25_fixes():
        raise RuntimeError("Failed to apply Qwen2.5-VL fixes")

    # Load tokenizer first (needed for detection head)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Load the model: either full detection wrapper or pure Qwen2.5-VL
    if config.detection_enabled:
        from src.models.wrapper import Qwen25VLWithDetection

        logger.info(
            f"🔄 Loading Qwen2.5-VL with detection head from: {config.model_path}"
        )
        model = Qwen25VLWithDetection.from_pretrained(
            model_path=config.model_path,
            num_queries=config.detection_num_queries,
            max_caption_length=config.detection_max_caption_length,
            tokenizer=tokenizer,
        )
    else:
        # Load base Qwen2.5-VL model without detection head
        from transformers import Qwen2_5_VLForConditionalGeneration

        from src.models.wrapper import _get_torch_dtype  # reuse dtype helper

        logger.info(
            f"🔄 Loading base Qwen2.5-VL model from: {config.model_path} (no detection)"
        )
        # Use *padded* attention mask – therefore we switch to the safer
        #       implementation="flash_attention_2" (no Flash-Attention patch required).
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype=_get_torch_dtype(config.torch_dtype),
            attn_implementation=config.attn_implementation,
        )
        # Flag for downstream checks
        model.detection_enabled = False

    # Verify patches
    logger.info("🔍 Verifying Qwen2.5-VL patches...")
    if not verify_qwen25_patches():
        raise RuntimeError("Patch verification failed")

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

    # Training chat processor (detailed prompts)
    train_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=True,  # Use detailed training prompts
        language="chinese",
    )

    # Evaluation chat processor (concise prompts)
    eval_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=False,  # Use concise evaluation prompts
        language="chinese",
    )

    # Create teacher pool manager if teacher_ratio > 0
    teacher_pool_manager = None
    if config.teacher_ratio > 0.0:
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
        teacher_ratio=config.teacher_ratio,
        is_training=True,  # Training context
    )

    # Create validation dataset with concise prompts and no teachers
    val_dataset = BBUDataset(
        data_path=config.val_data_path,
        chat_processor=eval_chat_processor,
        teacher_pool_manager=None,  # No teachers for validation
        teacher_ratio=0.0,  # No teachers for validation
        is_training=False,  # Evaluation context
    )

    # Create data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        collator_type=config.collator_type,
    )

    logger.info(f"✅ Data module setup completed with improved prompts:")
    logger.info(
        f"   Train samples: {len(train_dataset)} (detailed prompts, teacher_ratio={config.teacher_ratio})"
    )
    logger.info(f"   Val samples: {len(val_dataset)} (concise prompts, no teachers)")
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


def create_trainer(
    training_args: Optional[TrainingArguments] = None, **kwargs: Any
) -> BBUTrainer:
    """Creates a unified BBU trainer with all necessary components."""
    from src.training.trainer import BBUTrainer, set_model_training_params

    # Setup model and tokenizer
    model, tokenizer, image_processor = setup_model_and_tokenizer()

    # Set requires_grad for trainable components based on config
    set_model_training_params(model)

    # Setup data module (dataset and collator)
    data_module = setup_data_module(tokenizer, image_processor)

    # Create trainer
    from src.config import get_config

    trainer = BBUTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        image_processor=image_processor,
        cfg=get_config(),  # Explicitly pass DirectConfig instance
        **data_module,
    )

    # Initialize param groups for differential learning rate
    if config.use_differential_lr:
        trainer.init_param_groups()

    # Set the detection loss function in the model wrapper after trainer creation
    if config.detection_enabled and hasattr(trainer.model, "set_detection_loss_fn"):
        trainer.model.set_detection_loss_fn(trainer.detection_loss)
        trainer.logger.info("🎯 Detection loss function set in model wrapper")

    # ------------------------------------------------------------------
    # Register BestCheckpointCallback to keep the best N checkpoints based
    # on evaluation loss (metric: eval_loss).  The limit N comes from the
    # YAML field `save_total_limit` so experimenters can control retention
    # directly from the config file without touching code.
    # ------------------------------------------------------------------
    from src.training.callbacks import BestCheckpointCallback

    best_ckpt_cb = BestCheckpointCallback(
        save_total_limit=config.save_total_limit,
        metric_name="eval_loss",
        greater_is_better=False,
    )
    trainer.add_callback(best_ckpt_cb)

    trainer.logger.info("✅ BBU trainer created successfully")
    return trainer


def test_enhanced_logging() -> None:
    """
    Simple test to verify enhanced detection loss logging works.
    This can be called during development to test the logging mechanism.
    """
    from src.logger_utils import get_training_logger

    logger = get_training_logger()
    logger.info("🧪 Testing enhanced detection loss logging...")

    # Test that the loss components are properly structured
    sample_loss_components = {
        "total_loss": 1.5,
        "bbox_loss": 0.8,
        "caption_loss": 0.4,
        "objectness_loss": 0.3,
    }

    # Test prefix handling - no prefix for training, eval_ for evaluation
    for mode, prefix in [("training", ""), ("evaluation", "eval_")]:
        loss_info = {}
        for key, value in sample_loss_components.items():
            if key != "total_loss":  # Skip total_loss to avoid duplication
                loss_info[f"{prefix}detection_{key}"] = float(value)

        # Add other components
        loss_info[f"{prefix}lm_loss"] = 0.5
        loss_info[f"{prefix}detection_loss"] = 1.2
        loss_info[f"{prefix}weighted_detection_loss"] = 0.12

        logger.info(f"✅ {mode.upper()} loss structure: {loss_info}")

    logger.info("🧪 Enhanced logging test completed successfully!")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_enhanced_logging()
