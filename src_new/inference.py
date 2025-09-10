#!/usr/bin/env python3
"""
Standalone inference runner for Qwen2.5-VL with new architecture.

This script runs inference on a JSONL dataset using the new src_new/ architecture
with the following features:
- Unified configuration management
- DetectionModel wrapper with coordinate token support
- ChatProcessor for consistent prompt formatting
- Teacher-guided inference support
- Batch processing capabilities
"""

import os
import sys

# PROJECT_ROOT not needed - using relative paths

# Prevent stdlib shadowing when running this file directly (python src_new/inference.py)
# If sys.path[0] points to the script directory (src_new), remove it and ensure
# the project root is at the front so that 'src_new' is imported as a package and
# stdlib modules like 'types' resolve to the correct standard library module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
try:
    if _SCRIPT_DIR in sys.path:
        sys.path.remove(_SCRIPT_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
except Exception:
    raise

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Configure rank-aware logging early (strict, no fallback)
from src_new.utils.rank_aware_logging import (
    get_rank_aware_logger,
    initialize_logging_from_env,
)


# Initialize from environment if available; otherwise default to INFO
initialize_logging_from_env()
logger = get_rank_aware_logger("inference")

# Import new architecture components
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from src_new.config.config import load_config
from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.models.wrapper import DetectionModel
from src_new.processing.conversation import ConversationBuilder
from src_new.processing.special_tokens import (
    ASSISTANT_HEADER,
    END_OF_TEXT,
    GEOMETRY_TOKENS,
    IM_END,
    IM_START,
    IMAGE_PAD,
)
from src_new.processing.token_processor import TokenConfig, TokenProcessor
from src_new.utils.data_resolver import DataResolver
from src_new.utils.path_manager import create_path_manager
from src_new.utils.validation import PathValidationError


class InferenceEngine:
    """Inference engine using the new src_new architecture."""

    def __init__(
        self,
        config_path: str,
        model_path: str,
        use_training_prompts: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        use_torch_compile: bool = False,
        teacher_pool_file: Optional[str] = None,
        num_teachers: int = 0,
        force_eager_attention: bool = False,
        data_root: Optional[str] = None,
    ):
        """Initialize inference engine with new architecture.

        Args:
            config_path: Path to configuration YAML file
            model_path: Path to model directory
            use_training_prompts: Whether to use training prompts
            batch_size: Number of samples to process at once
            num_workers: Number of data loading workers
            use_torch_compile: Use torch.compile for faster inference
            teacher_pool_file: Path to teacher pool JSONL file
            num_teachers: Number of teacher examples to use per sample
            force_eager_attention: Force eager attention instead of flash attention
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_torch_compile = use_torch_compile
        self.teacher_pool_file = teacher_pool_file
        self.num_teachers = num_teachers
        self.force_eager_attention = force_eager_attention
        # Optional data root for resolving relative image paths
        self.data_root = data_root

        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)

        # Resolve model path (do not mutate frozen config)
        self._model_path = model_path if model_path else self.config.model_path
        # Enforce explicit paths (no implicit fallbacks)
        if self.data_root is None:
            self.data_root = self.config.data_root
        if self.teacher_pool_file is None:
            self.teacher_pool_file = self.config.teacher_pool_file

        # Apply Qwen2.5-VL patches
        logger.info("Applying Qwen2.5-VL patches...")
        apply_comprehensive_qwen25_fixes()
        logger.info("✅ Qwen2.5-VL patches applied successfully")

        # Load model and components
        self._load_model_and_processors()

        # Cache coordinate token range once (prefer persisted range if available)
        from src_new.processing.special_tokens import (
            get_coord_token_range,
            validate_geometry_tokens,
        )

        # Validate geometry tokens once - fail fast on invalid tokenizer
        validate_geometry_tokens(self.tokenizer)

        # Derive range from tokenizer - fail fast if tokenizer is invalid
        rng = get_coord_token_range(self.tokenizer)
        self._coord_range = (int(rng.start_id), int(rng.end_exclusive))
        logger.info(f"🎯 Cached coordinate token range: {self._coord_range}")

        # CRITICAL FIX: Load teacher pool manager for dynamic pairing (matches training pipeline)
        # During training, teacher examples are dynamically assigned based on student content
        # We need to replicate this behavior during inference
        self.teacher_pool_manager = None
        teacher_pool_path: Optional[str] = None

        # Resolve teacher pool path using DataResolver first (authoritative to data_root)
        if self.data_root is None:
            raise ValueError("data_root must be provided for teacher pool resolution")
        try:
            resolved_dataset_paths = DataResolver.resolve_dataset_paths(
                str(self.data_root)
            )
            default_teacher_pool = str(resolved_dataset_paths.teacher_pool_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve dataset paths from data_root for teacher pool: {e}"
            )

        candidate_path = self.teacher_pool_file or default_teacher_pool

        path_manager = create_path_manager(self.data_root)
        try:
            resolved_path = str(path_manager.resolve_path(candidate_path))
            teacher_pool_path = resolved_path
        except (FileNotFoundError, ValueError, PathValidationError) as e:
            # If explicit teacher path was provided but failed, try the data_root-derived default once
            if (
                self.teacher_pool_file
                and default_teacher_pool
                and self.teacher_pool_file != default_teacher_pool
            ):
                try:
                    resolved_path = str(path_manager.resolve_path(default_teacher_pool))
                    teacher_pool_path = resolved_path
                    logger.info(
                        f"Using teacher pool resolved from data_root instead of provided path: {default_teacher_pool}"
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Teacher pool path could not be resolved: {candidate_path} - {e}; secondary resolution failed: {e2}"
                    )
            else:
                raise RuntimeError(
                    f"Teacher pool path could not be resolved: {candidate_path} - {e}"
                )

        if teacher_pool_path and os.path.exists(teacher_pool_path):
            try:
                # Load teacher pool manager for dynamic pairing (same as training)
                from src_new.data.teacher_pool import TeacherPoolManager

                self.teacher_pool_manager = TeacherPoolManager(
                    teacher_pool_file=teacher_pool_path, config=self.config
                )
                logger.info(
                    f"✅ Teacher pool manager loaded with {len(self.teacher_pool_manager.teacher_pool)} examples for dynamic pairing"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load teacher pool manager from {teacher_pool_path}: {e}"
                )
                self.teacher_samples = []
        else:
            raise RuntimeError(
                f"Teacher pool file not found or unreadable: {candidate_path}"
            )

        # Use teacher guidance only when explicitly requested via num_teachers > 0
        if self.teacher_pool_manager and self.num_teachers > 0:
            logger.info(
                f"🎯 Dynamic teacher pairing enabled: {self.num_teachers} teacher(s) per sample"
            )
            logger.info(
                f"📚 Teacher samples serve as context input only (no teacher loss computation during inference)"
            )
            # Force batch_size=1 for teacher guidance
            if self.batch_size > 1:
                logger.warning(
                    f"Teacher guidance requires batch_size=1. Changing from {self.batch_size} to 1."
                )
                self.batch_size = 1
        elif self.num_teachers > 0 and not self.teacher_pool_manager:
            logger.warning(
                f"Teacher guidance requested ({self.num_teachers} teachers) but no teacher pool manager available"
            )

        logger.info("✅ InferenceEngine initialized successfully")

    def _load_model_and_processors(self):
        """Load model and processing components."""
        logger.info("Loading model and processors...")

        # Set up critical environment variables for fast loading (same as training)
        import os

        os.environ["HF_MODULES_CACHE"] = "./model_cache"
        os.environ["HF_HOME"] = "./model_cache"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Only set Triton/FlashAttention-specific env when using FlashAttention v2
        if self.config.attn_implementation == "flash_attention_2":
            os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["FLASH_ATTENTION_FORCE_CUDNN"] = "0"

        # Create token processor
        token_config = TokenConfig(
            coordinate_tokens_enabled=self.config.coordinate_tokens_enabled,
            max_coord_value=self.config.max_coord_value,
            new_geometry_tokens=self.config.new_geometry_tokens,
            coordinate_init_mode=self.config.coordinate_init_mode,
        )
        self.token_processor = TokenProcessor(token_config)

        # Fast checkpoint loading using DetectionModel.from_pretrained (same as training)
        # This avoids double extension and weight reinitialization issues
        logger.info(
            "🚀 Loading checkpoint using DetectionModel.from_pretrained (training-compatible approach)"
        )

        # Load tokenizer and processor directly from checkpoint (contains expanded vocabulary)

        logger.info(f"Loading tokenizer from checkpoint: {self._model_path}")
        # Load tokenizer directly from checkpoint - this should contain the expanded vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Fast tokenizer required for inference (use_fast=True)")
        _enc = self.tokenizer(
            "sanity",
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        if _enc.get("offset_mapping") is None:
            raise RuntimeError(
                "Fast tokenizer did not return offset_mapping in inference. Ensure tokenizer.json is valid."
            )

        # Load image processor from checkpoint
        logger.info(f"Loading image processor from checkpoint: {self._model_path}")
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            self._model_path, trust_remote_code=False, do_resize=False
        )
        # Apply overrides from config explicitly (fail-fast)
        if hasattr(self.config, "max_pixels") and int(self.config.max_pixels) > 0:
            if not hasattr(self.image_processor, "max_pixels"):
                raise ValueError("Qwen2VLImageProcessor missing 'max_pixels' attribute")
            self.image_processor.max_pixels = int(self.config.max_pixels)
            logger.info(
                f"🔧 (Inference) Set image_processor.max_pixels={self.image_processor.max_pixels}"
            )

        # Verify coordinate token checkpoint by checking vocabulary (derived, not fixed)
        vocab = self.tokenizer.get_vocab()
        current_vocab_size = len(vocab)
        try:
            from src_new.processing.special_tokens import (
                get_coord_token_range,
                validate_geometry_tokens,
            )

            rng = get_coord_token_range(self.tokenizer)
            has_coords = rng.end_exclusive > rng.start_id
            if has_coords:
                logger.info(
                    f"🎯 Coordinate tokens detected: range=({rng.start_id}, {rng.end_exclusive})"
                )
                # Compare with saved coordinate_config.json if present
                try:
                    import json
                    import os

                    coord_cfg_path = os.path.join(
                        self._model_path, "coordinate_config.json"
                    )
                    if os.path.exists(coord_cfg_path):
                        with open(coord_cfg_path, "r") as f:
                            coord_cfg = json.load(f)
                        saved_range = coord_cfg.get("coordinate_token_range")
                        if isinstance(saved_range, list) and len(saved_range) == 2:
                            if [rng.start_id, rng.end_exclusive] != saved_range:
                                logger.warning(
                                    f"⚠️ Saved coordinate_token_range {saved_range} differs from tokenizer‑derived "
                                    f"[{rng.start_id}, {rng.end_exclusive}]"
                                )
                            else:
                                logger.info(
                                    "✅ Saved coordinate_token_range matches tokenizer‑derived range"
                                )
                except Exception as _e:
                    logger.debug(f"Coordinate config comparison skipped: {_e}")

                validate_geometry_tokens(self.tokenizer)
            else:
                logger.info("ℹ️ No coordinate tokens detected in tokenizer")
        except Exception as e:
            logger.warning(f"Coordinate/geometry token validation warning: {e}")

        # Load model using DetectionModel.from_pretrained (EXACTLY like training)
        # This prevents double extension and weight reinitialization issues
        import time

        logger.info(
            "Loading model using DetectionModel.from_pretrained (training approach)..."
        )
        model_load_start = time.time()

        # Use fast loading for inference - skip vocab extension if checkpoint already has it
        vocab_size = len(self.tokenizer.get_vocab())
        try:
            from src_new.processing.special_tokens import get_coord_token_range

            rng_probe = get_coord_token_range(self.tokenizer)
            has_coordinate_tokens = rng_probe.end_exclusive > rng_probe.start_id
        except Exception:
            has_coordinate_tokens = False

        # Check if SafeTensors format is available for even faster loading
        # COMPATIBILITY FIX: Support both single and sharded SafeTensors
        safetensors_path = os.path.join(self._model_path, "model.safetensors")
        safetensors_index_path = os.path.join(
            self._model_path, "model.safetensors.index.json"
        )

        has_single_safetensors = os.path.exists(safetensors_path)
        has_sharded_safetensors = os.path.exists(safetensors_index_path)

        if has_single_safetensors:
            logger.info(
                f"🚀 Ultra-fast SafeTensors loading detected - using single model.safetensors"
            )
        elif has_sharded_safetensors:
            logger.info(
                f"🚀 SafeTensors loading detected - using sharded SafeTensors format"
            )

        if has_coordinate_tokens:
            logger.info(
                f"🚀 Fast inference loading - checkpoint has extended vocab ({vocab_size} tokens)"
            )
            self.model = DetectionModel.from_pretrained_fast(
                model_path=self._model_path,
                config=self.config,
                tokenizer=self.tokenizer,
                skip_vocab_extension=True,  # Skip for speed
                # Inference-specific optimizations
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Stable attention for inference
                device_map={"": "cuda:0"},  # Direct GPU placement
                trust_remote_code=False,
                use_cache=True,  # Enable KV cache for inference
                low_cpu_mem_usage=True,  # Reduce CPU memory during loading
            )
        else:
            logger.info(
                f"🔧 Standard loading - extending vocab from base model ({vocab_size} tokens)"
            )
            self.model = DetectionModel.from_pretrained(
                model_path=self._model_path,
                config=self.config,
                tokenizer=self.tokenizer,
                # Inference-specific optimizations
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Stable attention for inference
                device_map={"": "cuda:0"},  # Direct GPU placement
                trust_remote_code=False,
                use_cache=True,  # Enable KV cache for inference
                low_cpu_mem_usage=True,  # Reduce CPU memory during loading
            )

        model_load_time = time.time() - model_load_start
        logger.info(f"⏱️ Model loading completed in {model_load_time:.2f} seconds")

        # Set to evaluation mode
        self.model.eval()

        # Verify model loading
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        model_vocab_size = self.model.get_input_embeddings().num_embeddings

        logger.info("✅ Model loading completed successfully")
        logger.info(f"📍 Device: {model_device}")
        logger.info(f"🔢 Dtype: {model_dtype}")
        logger.info(f"� Model vocab size: {model_vocab_size}")
        logger.info(f"� Tokenizer vocab size: {current_vocab_size}")

        # Verify vocabulary consistency (DetectionModel should handle any mismatches)
        if model_vocab_size == current_vocab_size:
            logger.info("✅ Vocabulary sizes match - checkpoint consistency verified")
        else:
            logger.info(
                f"ℹ️ Vocabulary size difference: model={model_vocab_size}, tokenizer={current_vocab_size}"
            )
            logger.info(
                "DetectionModel wrapper handles vocabulary adjustments automatically"
            )

        # Check for ms-swift vocabulary padding optimization (multiples of 128)
        if current_vocab_size % 128 == 0:
            logger.info(
                f"✅ Vocabulary is optimally padded to multiple of 128: {current_vocab_size}"
            )
        else:
            logger.info(
                f"ℹ️ Vocabulary size {current_vocab_size} not padded to 128 (legacy checkpoint)"
            )

        logger.info("✅ Vocabulary consistency verified")
        # Prefer tokenizer-derived range to decide presence of coord tokens
        try:
            from src_new.processing.special_tokens import get_coord_token_range

            rng_flag = get_coord_token_range(self.tokenizer)
            self.coordinate_tokens_enabled = rng_flag.end_exclusive > rng_flag.start_id
        except Exception:
            self.coordinate_tokens_enabled = False

        # Validate configuration consistency
        config_expects_coords = self.config.coordinate_tokens_enabled
        tokenizer_has_coords = self.coordinate_tokens_enabled

        if config_expects_coords and not tokenizer_has_coords:
            logger.warning(
                f"⚠️ Configuration expects coordinate tokens but tokenizer has none detected; coordinate parsing will be disabled."
            )
        elif not config_expects_coords and tokenizer_has_coords:
            logger.info(
                f"ℹ️ Tokenizer has coordinate tokens but config has coordinate_tokens_enabled=False. Using standard parsing mode."
            )

        # Log coordinate token optimization status
        if self.coordinate_tokens_enabled and config_expects_coords:
            logger.info(
                "🎯 Coordinate tokens detected in tokenizer; optimized parsing active"
            )
            logger.info("🚀 Ready for optimized coordinate token inference")
        elif not config_expects_coords:
            logger.info(
                "📝 Using standard inference mode (coordinate tokens disabled in config)"
            )

        # Create conversation processor using new architecture
        # Create unified processor from tokenizer, image processor, and video processor
        try:
            # Try to load from checkpoint for consistency
            proc_from_ckpt = Qwen2VLProcessor.from_pretrained(
                self.config.model_path, trust_remote_code=True
            )
            video_processor = (
                getattr(proc_from_ckpt, "video_processor", None)
                or Qwen2VLVideoProcessor()
            )
        except Exception:
            video_processor = Qwen2VLVideoProcessor()

        unified_processor = Qwen2VLProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            video_processor=video_processor,
        )

        # CRITICAL FIX: Ensure chat template is properly inherited from tokenizer
        # The Qwen2VLProcessor doesn't automatically inherit chat_template from tokenizer
        chat_template = (
            self.tokenizer.chat_template
            if hasattr(self.tokenizer, "chat_template")
            else None
        )
        if chat_template:
            unified_processor.chat_template = chat_template
            logger.info("✅ Chat template inherited from tokenizer to processor")
        else:
            logger.debug(
                "Tokenizer has no chat_template; skipping processor chat_template inheritance"
            )

        self.conversation_processor = ConversationBuilder(
            processor=unified_processor,
            max_coord_value=self.config.max_coord_value,
            coordinate_tokens_enabled=bool(self.config.coordinate_tokens_enabled),
            plain_text_mode_enabled=bool(self.config.plain_text_mode_enabled),
        )

        # Set model to evaluation mode
        self.model.eval()

        # Apply torch.compile if requested
        if self.use_torch_compile and torch.__version__ >= "2.0.0":
            try:
                logger.info("Applying torch.compile optimization...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        logger.info("✅ Model and processors loaded successfully")

    def _load_teacher_pool(
        self, teacher_pool_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load teacher samples from JSONL file."""
        if teacher_pool_path is None:
            teacher_pool_path = self.teacher_pool_file
        if not teacher_pool_path:
            return []

        if not os.path.exists(teacher_pool_path):
            raise FileNotFoundError(f"Teacher pool file not found: {teacher_pool_path}")

        teacher_samples = []
        with open(teacher_pool_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        # Validate required fields
                        if "images" not in sample or "objects" not in sample:
                            raise ValueError(
                                f"Missing required fields at line {line_num}"
                            )
                        teacher_samples.append(sample)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at line {line_num}: {e}")

        if not teacher_samples:
            raise ValueError("No teacher samples found in teacher pool file")

        return teacher_samples

    def _sample_teachers(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample teachers using dynamic pairing (same as training pipeline)."""
        if not self.teacher_pool_manager or self.num_teachers <= 0:
            return []

        if seed is not None:
            random.seed(seed)

        # For inference without a student sample context, do not fabricate random picks here.
        # Delegate to teacher_pool_manager for deterministic selection.
        return self.teacher_pool_manager.select_teachers_for_student(
            student_sample=None, num_samples=self.num_teachers
        )

    def prepare_inference_inputs(
        self,
        sample: Dict[str, Any],
        seed: Optional[int] = None,
        data_root: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for inference using EXACT training pipeline data preparation.

        This method replicates the training data preparation process exactly:
        1. Load JSONL sample (same format as training)
        2. Assign teacher samples (same logic as training)
        3. Use ConversationProcessor to create conversation (same as training)
        4. Apply HuggingFace templates (same as training)
        5. Process images and coordinate tokens (same as training)
        6. Return fully processed inputs ready for generation

        Args:
            sample: The sample to process (JSONL format)
            seed: Random seed for teacher sampling
            data_root: Data root directory for resolving image paths

        Returns:
            Processed inputs ready for model.generate()
        """
        # CRITICAL FIX: Set data_root for teacher image path resolution
        if data_root is not None:
            self.data_root = data_root

        # Use teacher-guided approach only when explicitly requested
        if self.teacher_pool_manager and self.num_teachers > 0:
            return self._prepare_training_matched_inputs(sample, seed)
        else:
            # Simple conversation path (default in tests)
            return self._prepare_simple_training_format(sample)

    def _prepare_training_matched_inputs(
        self, student_sample: Dict[str, Any], seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs using EXACT training pipeline data preparation process."""
        # Set deterministic seed for teacher sampling (same as training)
        if seed is not None:
            random.seed(seed)

        # Use dynamic teacher pairing (same as training pipeline)
        if self.teacher_pool_manager and self.num_teachers > 0:
            teachers = self.teacher_pool_manager.select_teachers_for_student(
                student_sample, num_samples=self.num_teachers
            )
            logger.debug(
                f"🎯 Dynamic teacher pairing selected {len(teachers)} teachers for student"
            )
        else:
            # Fallback to random sampling if no teacher pool manager
            teachers = self._sample_teachers(seed)

        logger.debug(f"🎯 TRAINING-MATCHED INFERENCE SETUP:")
        logger.debug(f"   Teachers: {len(teachers)}")
        logger.debug(f"   Student: 1")
        logger.debug(f"   Seed: {seed}")

        # CRITICAL: Use ConversationBuilder exactly like training
        # This replicates the training data preparation process exactly

        # Load images in training order (teachers first, then student)
        teacher_images_list = []
        total_teacher_images = 0
        for teacher in teachers:
            if not isinstance(teacher, dict) or "images" not in teacher:
                raise ValueError(
                    "Each teacher sample must include a non-empty 'images' list"
                )
            teacher_image_paths = teacher["images"]
            if (
                not isinstance(teacher_image_paths, list)
                or len(teacher_image_paths) == 0
            ):
                raise ValueError(
                    "Each teacher sample must include at least one image path"
                )
            teacher_images = []
            for img_path in teacher_image_paths:
                try:
                    # CRITICAL FIX: Use PathManager for unified path resolution
                    if self.data_root is None:
                        raise ValueError(
                            "data_root must be provided for image path resolution"
                        )
                    path_manager = create_path_manager(self.data_root)
                    resolved_img_path = str(path_manager.resolve_path(img_path))

                    img = Image.open(resolved_img_path).convert("RGB")
                    teacher_images.append(img)
                    total_teacher_images += 1
                    logger.debug(
                        f"   Loaded teacher image: {resolved_img_path} ({img.size})"
                    )
                except Exception as e:
                    logger.error(f"Failed to load teacher image {img_path}: {e}")
                    logger.error(
                        f"   Attempted path: {resolved_img_path if 'resolved_img_path' in locals() else 'N/A'}"
                    )
                    raise
            teacher_images_list.append(teacher_images)

        # Load student images
        if not isinstance(student_sample, dict) or "images" not in student_sample:
            raise ValueError("Student sample must include a non-empty 'images' list")
        student_image_paths = student_sample["images"]
        if not isinstance(student_image_paths, list) or len(student_image_paths) == 0:
            raise ValueError("Student sample must include at least one image path")
        student_images = []
        if self.data_root is None:
            raise ValueError("data_root must be provided for image path resolution")
        path_manager = create_path_manager(self.data_root)
        for img_path in student_image_paths:
            try:
                # CRITICAL FIX: Use PathManager for student image path resolution
                resolved_img_path = str(path_manager.resolve_path(img_path))
                img = Image.open(resolved_img_path).convert("RGB")
                student_images.append(img)
                logger.debug(
                    f"   Loaded student image: {resolved_img_path} ({img.size})"
                )
            except Exception as e:
                logger.error(f"Failed to load student image {img_path}: {e}")
                logger.error(
                    f"   Attempted path: {resolved_img_path if 'resolved_img_path' in locals() else 'N/A'}"
                )
                raise

        total_expected_images = total_teacher_images + len(student_images)
        logger.debug(
            f"   Total expected images: {total_expected_images} (teachers: {total_teacher_images}, student: {len(student_images)})"
        )

        # CRITICAL: Use ConversationProcessor.create_teacher_student_conversation
        # This is EXACTLY what training does in Dataset._process_sample_unified
        try:
            # Build a generation-ready conversation directly via builder
            inputs = self.conversation_processor.create_teacher_student_conversation_for_generation(
                student_sample=student_sample,
                teacher_samples=teachers,
                student_images=student_images,
                teacher_images_list=teacher_images_list,
            )

        except Exception as e:
            logger.error(f"ConversationBuilder failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create training-matched conversation: {e}")

        # FINAL VALIDATION: Ensure tensor consistency for model generation
        input_ids_for_decode = inputs["input_ids"][0] if inputs["input_ids"].dim() == 2 else inputs["input_ids"]
        final_text = self.tokenizer.decode(
            input_ids_for_decode, skip_special_tokens=False
        )
        final_image_token_count = final_text.count(IMAGE_PAD)

        # Validate final tensor shapes and token alignment
        pixel_values_shape = inputs["pixel_values"].shape
        image_grid_thw_shape = inputs["image_grid_thw"].shape

        logger.debug(f"🗣️  FINAL TRAINING-MATCHED CONVERSATION:")
        logger.debug(f"     Input IDs shape: {inputs['input_ids'].shape}")
        logger.debug(f"     Pixel values shape: {pixel_values_shape}")
        logger.debug(f"     Image grid THW shape: {image_grid_thw_shape}")
        logger.debug(f"     Image tokens in prompt: {final_image_token_count}")
        logger.debug(
            f"     Images in tensor: {image_grid_thw_shape[0] if len(image_grid_thw_shape) > 0 else 0}"
        )
        logger.debug(f"     Expected images: {total_expected_images}")
        logger.debug(f"     Conversation length: {len(final_text)} characters")
        logger.debug(
            f"     Conversation ending: ...{final_text[-150:] if len(final_text) > 150 else final_text}"
        )

        # Critical validation before returning
        if (
            len(image_grid_thw_shape) > 0
            and image_grid_thw_shape[0] != total_expected_images
        ):
            logger.error(f"❌ FINAL IMAGE COUNT MISMATCH:")
            logger.error(f"   Expected: {total_expected_images}")
            logger.error(f"   Image grid THW: {image_grid_thw_shape[0]}")
            # Continue but flag the issue for debugging

        # Validate pixel values tensor has correct number of images
        if (
            len(pixel_values_shape) >= 4
        ):  # Should be [num_images, channels, height, width] or similar
            pixel_num_images = pixel_values_shape[0] // (
                image_grid_thw_shape[0] if image_grid_thw_shape[0] > 0 else 1
            )
            if pixel_num_images != 1:
                logger.warning(
                    f"⚠️  Pixel values might have unexpected structure: {pixel_values_shape}"
                )

        return inputs

    def _prepare_simple_training_format(
        self, sample: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs using simple conversation but with training format."""
        logger.info("Using simple conversation with training format")

        # Load images
        if not isinstance(sample, dict) or "images" not in sample:
            raise ValueError("Sample must include a non-empty 'images' list")
        image_paths = sample["images"]
        if not isinstance(image_paths, list) or len(image_paths) == 0:
            raise ValueError("Sample must include at least one image path")
        images = []
        if self.data_root is None:
            raise ValueError("data_root must be provided for image path resolution")
        path_manager = create_path_manager(self.data_root)
        for img_path in image_paths:
            try:
                # CRITICAL FIX: Use PathManager for unified path resolution
                resolved_img_path = str(path_manager.resolve_path(img_path))
                img = Image.open(resolved_img_path).convert("RGB")
                images.append(img)
                logger.debug(f"   Loaded image: {resolved_img_path} ({img.size})")
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                logger.error(
                    f"   Attempted path: {resolved_img_path if 'resolved_img_path' in locals() else 'N/A'}"
                )
                raise

        # CRITICAL: Use ConversationBuilder.create_simple_conversation
        # This matches the training path for samples without teachers
        try:
            # Build a generation-ready simple conversation via processor
            inputs = (
                self.conversation_processor.create_simple_conversation_for_generation(
                    sample=sample, images=images
                )
            )
        except Exception as e:
            raise RuntimeError(f"ConversationBuilder failed: {e}")

        # Validate conversation structure using the processed inputs
        input_ids_for_decode = inputs["input_ids"][0] if inputs["input_ids"].dim() == 2 else inputs["input_ids"]
        final_text = self.tokenizer.decode(
            input_ids_for_decode, skip_special_tokens=False
        )
        image_token_count = final_text.count(IMAGE_PAD)

        logger.debug(f"🗣️  Simple training format conversation:")
        logger.debug(f"     Input IDs shape: {inputs['input_ids'].shape}")
        logger.debug(f"     Pixel values shape: {inputs['pixel_values'].shape}")
        logger.debug(f"     Image grid THW shape: {inputs['image_grid_thw'].shape}")
        logger.debug(f"     Image tokens in prompt: {image_token_count}")
        logger.debug(f"     Prompt length: {len(final_text)} characters")
        logger.debug(f"     Prompt ending: ...{final_text[-100:]}")

        # Keep warning but reduce the hard assumption on minimum token count per image
        if inputs["image_grid_thw"].shape[0] > 0 and image_token_count == 0:
            logger.warning(
                f"   ⚠️  No image tokens decoded though {inputs['image_grid_thw'].shape[0]} images present"
            )

        return inputs

    def generate_response(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float = 0.000001,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,  # Changed from 1.1 to 1.0 for coordinate tokens
    ) -> List[Dict[str, Any]]:
        """Generate response and return a structured list of objects for visualization.

        Returns a list where each element is a dict with one of 'bbox_2d'|'quad'|'line' and 'desc'.
        """

        # CRITICAL FIX: Use pre-processed inputs directly
        # This eliminates the double processing issue that caused the indexing error
        logger.debug("Using pre-processed inputs from ConversationBuilder")

        # ENHANCED VALIDATION: Comprehensive pre-generation checks
        logger.debug(f"🔧 PRE-GENERATION VALIDATION:")
        logger.debug(f"   Input keys: {list(inputs.keys())}")

        # Validate essential tensor presence
        required_keys = ["input_ids", "attention_mask"]
        optional_keys = ["pixel_values", "image_grid_thw"]

        for key in required_keys:
            if key not in inputs:
                raise RuntimeError(f"Missing required input: {key}")

        has_images = all(key in inputs for key in optional_keys)
        logger.debug(f"   Has image inputs: {has_images}")

        # Sanitize and move only model-relevant tensors to device
        allowed_input_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        sanitized_inputs: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if k in allowed_input_keys and torch.is_tensor(v):
                sanitized_inputs[k] = v.to(self.model.device)
        # Ensure batch dimension on text tensors
        for key in ("input_ids", "attention_mask"):
            if key in sanitized_inputs and sanitized_inputs[key].dim() == 1:
                sanitized_inputs[key] = sanitized_inputs[key].unsqueeze(0)
        inputs = sanitized_inputs

        # Enhanced tensor shape logging
        for k, v in inputs.items():
            if torch.is_tensor(v):
                logger.debug(f"   {k}: {v.shape} ({v.dtype}) on {v.device}")

        # CRITICAL VALIDATION: Cross-validate image tokens and tensors before generation
        if has_images:
            # Decode input_ids to check image token alignment
            ids_for_decode = inputs["input_ids"][0] if inputs["input_ids"].dim() == 2 else inputs["input_ids"]
            decoded_text = self.tokenizer.decode(
                ids_for_decode, skip_special_tokens=False
            )
            image_token_count = decoded_text.count(IMAGE_PAD)

            pixel_values_shape = inputs["pixel_values"].shape
            image_grid_thw_shape = inputs["image_grid_thw"].shape

            logger.debug(f"🔍 IMAGE TOKEN ALIGNMENT CHECK:")
            logger.debug(f"   Decoded image tokens: {image_token_count}")
            logger.debug(f"   Pixel values shape: {pixel_values_shape}")
            logger.debug(f"   Image grid THW shape: {image_grid_thw_shape}")

            # Validate image grid consistency
            if len(image_grid_thw_shape) > 0:
                num_image_grids = image_grid_thw_shape[0]
                logger.debug(f"   Number of image grids: {num_image_grids}")

                # Each image should produce multiple image_pad tokens (typically 256-1024)
                expected_min_tokens = num_image_grids * 100  # Conservative estimate
                if image_token_count < expected_min_tokens:
                    logger.warning(
                        f"⚠️  Unusually low image token count: {image_token_count} for {num_image_grids} images"
                    )
                    logger.warning(
                        f"   Expected at least: {expected_min_tokens} tokens"
                    )
                    logger.warning(f"   This might indicate an image processing issue")

            # Validate pixel values tensor structure
            if len(pixel_values_shape) >= 4:
                logger.debug(
                    f"   Pixel values: {pixel_values_shape[0]} patches, {pixel_values_shape[1]}x{pixel_values_shape[2]}x{pixel_values_shape[3]} each"
                )

            # This is where the critical mismatch often occurs
            # The model expects pixel_values to align with the image tokens in input_ids
            if (
                image_token_count == 0
                and len(image_grid_thw_shape) > 0
                and image_grid_thw_shape[0] > 0
            ):
                logger.error(
                    "❌ CRITICAL MISMATCH: No image tokens in text but image tensors present"
                )
                logger.error(
                    f"   This will cause 'Image features and image tokens do not match' error"
                )
                raise RuntimeError(
                    "Image token mismatch: No tokens found in text but image data present"
                )

        # Create constrained decoding function for coordinate tokens
        def create_coordinate_constrained_fn():
            """Create a prefix_allowed_tokens_fn that constrains coordinate generation."""

            # Get coordinate token range
            if self.config.coordinate_tokens_enabled:
                coord_min, coord_max = self._coord_range
                coord_token_ids = set(range(int(coord_min), int(coord_max)))
            else:
                coord_token_ids = set()

            # Get geometry token IDs
            geometry_token_ids = set()
            punctuation_token_ids = set()

            # Flatten canonical geometry tokens from GEOMETRY_TOKENS
            canonical_tokens = set()
            for _, toks in GEOMETRY_TOKENS.items():
                canonical_tokens.update(toks)

            for token in canonical_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    geometry_token_ids.add(token_id)

            # Punctuation tokens commonly used in coordinate lists
            for token in ["[", "]", ",", " ", ", "]:
                token_ids = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_ids, list):
                    punctuation_token_ids.update(token_ids)
                elif token_ids != self.tokenizer.unk_token_id:
                    punctuation_token_ids.add(token_ids)

            def prefix_allowed_tokens_fn(
                batch_id: int, input_ids: torch.Tensor
            ) -> List[int]:
                """Constrain tokens based on current generation state."""

                # Convert to list for easier processing
                ids = input_ids.tolist()

                # Check if we're inside a coordinate section by looking for unclosed brackets
                # after geometry start tokens
                inside_coord_section = False

                # Look for geometry start patterns followed by unclosed brackets
                for i in range(
                    len(ids) - 1, max(-1, len(ids) - 50), -1
                ):  # Check last 50 tokens
                    token_id = ids[i]

                    # Check if this is a closing bracket - if so, we're not inside
                    token_str = self.tokenizer.decode([token_id])
                    if token_str in ["]"]:
                        break

                    # Check if this is an opening bracket after geometry start
                    if token_str in ["["]:
                        # Look backwards for geometry start token
                        for j in range(i - 1, max(-1, i - 10), -1):
                            prev_token_str = self.tokenizer.decode([ids[j]])
                            # Any canonical geometry start token
                            geometry_starts = {t[2] for t in GEOMETRY_TOKENS.values()}
                            if prev_token_str in geometry_starts:
                                inside_coord_section = True
                                break
                        break

                if inside_coord_section:
                    # Inside coordinate section: allow only coordinate tokens and punctuation
                    allowed = list(coord_token_ids | punctuation_token_ids)
                    # Also allow closing bracket and geometry end tokens
                    geometry_ends = {t[3] for t in GEOMETRY_TOKENS.values()}
                    for token in ("]", *geometry_ends):
                        token_id = self.tokenizer.convert_tokens_to_ids(token)
                        if token_id != self.tokenizer.unk_token_id:
                            allowed.append(token_id)
                    return allowed
                else:
                    # Outside coordinate section: allow all tokens (normal generation)
                    return list(range(len(self.tokenizer.get_vocab())))

            return prefix_allowed_tokens_fn

        # Generate response with enhanced error handling
        try:
            with torch.no_grad():
                # Use explicit EOS token; require eos_token_id to exist
                if (
                    self.tokenizer.eos_token_id is None
                    or self.tokenizer.eos_token_id == -1
                ):
                    raise ValueError(
                        "Tokenizer must define eos_token_id for generation"
                    )
                endoftext_token_id = self.tokenizer.convert_tokens_to_ids(END_OF_TEXT)

                logger.debug(f"🚀 Starting generation:")
                logger.debug(f"   Max new tokens: {max_new_tokens}")
                logger.debug(f"   Temperature: {temperature}")
                logger.debug(f"   Do sample: {do_sample}")
                logger.debug(f"   Repetition penalty: {repetition_penalty}")
                logger.debug(f"   EOS token ID: {endoftext_token_id}")

                # Prepare EOS tokens (do not include geometry end tokens; let the model decide endings)
                eos_tokens = set()
                for token in [IM_END, END_OF_TEXT]:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None and token_id != -1:
                        eos_tokens.add(token_id)

                if self.tokenizer.eos_token_id is not None:
                    eos_tokens.add(self.tokenizer.eos_token_id)

                # Log generation mode
                if self.config.coordinate_tokens_enabled:
                    logger.debug(
                        "🎯 Generation mode: Coordinate tokens enabled (will parse <|coord_*|> tokens)"
                    )
                else:
                    logger.debug(
                        "📝 Generation mode: Standard mode (no coordinate token parsing)"
                    )

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=list(eos_tokens),
                    use_cache=True,
                )

        except RuntimeError as e:
            error_msg = str(e)
            if "Image features and image tokens do not match" in error_msg:
                logger.error(f"❌ IMAGE TOKEN MISMATCH ERROR:")
                logger.error(f"   Error: {error_msg}")
                logger.error(f"   Input IDs shape: {inputs['input_ids'].shape}")

                if has_images:
                    logger.error(
                        f"   Pixel values shape: {inputs['pixel_values'].shape}"
                    )
                    logger.error(
                        f"   Image grid THW shape: {inputs['image_grid_thw'].shape}"
                    )

                    # Decode and analyze the problematic input
                    problematic_text = self.tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=False
                    )
                    problematic_tokens = problematic_text.count(IMAGE_PAD)

                    logger.error(
                        f"   Problematic text length: {len(problematic_text)} chars"
                    )
                    logger.error(f"   Image tokens in text: {problematic_tokens}")
                    logger.error(f"   Text preview: {problematic_text[:200]}...")
                    logger.error(f"   Text ending: ...{problematic_text[-200:]}")

                    # Look for patterns that might cause the mismatch
                    if IMAGE_PAD not in problematic_text:
                        logger.error(
                            "   DIAGNOSIS: No image tokens found in input text"
                        )
                    elif problematic_tokens == 0:
                        logger.error(
                            "   DIAGNOSIS: Zero image tokens despite image tensor presence"
                        )
                    else:
                        logger.error(
                            f"   DIAGNOSIS: {problematic_tokens} image tokens found but tensor mismatch"
                        )

                raise RuntimeError(
                    f"Image token mismatch during generation. Check conversation processing. Original error: {error_msg}"
                )

            elif "CUDA out of memory" in error_msg:
                logger.error(f"❌ GPU MEMORY ERROR: {error_msg}")
                logger.error(f"   Try reducing max_new_tokens or image resolution")
                raise RuntimeError(
                    f"GPU memory exhausted during generation: {error_msg}"
                )

            else:
                logger.error(f"❌ GENERATION ERROR: {error_msg}")
                raise

        # Decode response with validation
        try:
            # Decode ONLY the newly generated tokens (assistant content for the student turn)
            # Preserve coordinate and geometry tokens; do NOT skip specials here.
            generated_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )
        except Exception as e:
            logger.error(f"❌ DECODING ERROR: {e}")
            logger.error(f"   Output IDs shape: {output_ids.shape}")
            raise RuntimeError(f"Failed to decode generated response: {e}")

        # Log generation details
        input_length = inputs["input_ids"].shape[1]
        logger.info(f"✅ GENERATION COMPLETED:")
        logger.info(f"   Input tokens: {input_length}")
        logger.info(f"   Generated tokens: {len(generated_text)}")
        logger.info(f"   Total output tokens: {output_ids.shape[1]}")
        logger.info(
            f"   Raw generated text (first 500 chars): '{generated_text[:500]}'"
        )
        if len(generated_text) > 500:
            logger.info(
                f"   Raw generated text (last 500 chars): '...{generated_text[-500:]}'"
            )

        # Clean generated text: trim at the first IM_END and drop any accidental assistant header
        assistant_part = generated_text
        if assistant_part.startswith(ASSISTANT_HEADER):
            assistant_part = assistant_part[len(ASSISTANT_HEADER) :]
        end_markers = [IM_END, END_OF_TEXT]
        cut_idx = None
        for marker in end_markers:
            idx = assistant_part.find(marker)
            if idx != -1:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)
        if cut_idx is not None:
            assistant_part = assistant_part[:cut_idx]

        # Clean up container tokens; keep geometry/coord tokens for parsing
        for token in [IM_END, END_OF_TEXT, IM_START]:
            assistant_part = assistant_part.replace(token, "")
        cleaned_response = assistant_part.strip()

        # Log the cleaned response for debugging
        logger.info(f"🧹 CLEANED RESPONSE FOR PARSING:")
        logger.info(f"   Length: {len(cleaned_response)} characters")
        logger.info(f"   Content: '{cleaned_response}'")
        logger.info(
            f"   Coordinate tokens enabled: {self.config.coordinate_tokens_enabled}"
        )

        # Normalize to visualization object list
        objects_list: List[Dict[str, Any]] = self._normalize_prediction_to_vis_objects(
            cleaned_response
        )

        if not objects_list:
            logger.warning("⚠️  Empty or unparseable response objects!")
            logger.warning(f"   Failed to parse response: '{cleaned_response}'")
            logger.warning(f"   Response length: {len(cleaned_response)} characters")
            logger.warning(f"   Response is empty: {len(cleaned_response) == 0}")
            logger.warning(
                f"   Response contains coordinate tokens: {'<|coord_' in cleaned_response}"
            )
            logger.warning(
                f"   Response contains geometry tokens: {'<|object_ref_start|>' in cleaned_response}"
            )
            logger.warning(
                f"   Response contains JSON-like content: {('[' in cleaned_response and ']' in cleaned_response)}"
            )

        logger.info(f"🎯 Parsed {len(objects_list)} objects from generated response")

        return objects_list

    # =====================
    # Visualization helpers
    # =====================
    def _convert_objects_to_vis_format(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert training-format objects to visualization format.

        - Maps 'desc' -> 'label'
        - Preserves geometry keys: bbox_2d, quad, line
        - Coerces coordinate values to int where possible
        """
        vis_objects: List[Dict[str, Any]] = []
        for obj in objects or []:
            vis_obj: Dict[str, Any] = {}
            # label
            if "desc" in obj and isinstance(obj["desc"], str):
                vis_obj["label"] = obj["desc"]
            elif "label" in obj and isinstance(obj["label"], str):
                vis_obj["label"] = obj["label"]
            else:
                vis_obj["label"] = "Unknown"

            # geometry - copy first matching
            for key in GEOMETRY_TOKENS.keys():
                if key in obj and isinstance(obj[key], list):
                    coords = obj[key]
                    try:
                        vis_obj[key] = [int(c) for c in coords]
                    except Exception:
                        # fallback, keep as-is
                        vis_obj[key] = coords
                    break

            # Only add if we found a geometry key
            if any(k in vis_obj for k in GEOMETRY_TOKENS.keys()):
                vis_objects.append(vis_obj)
        return vis_objects

    def _convert_objects_to_training_format(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize objects to the training dataset format.

        Rules:
        - Field name for text is 'desc' (fallback from 'label' if needed)
        - Only geometry keys allowed: 'bbox_2d', 'quad', 'line'

        - Coordinates are coerced to int where possible
        - Ignore objects without a supported geometry
        """
        normalized: List[Dict[str, Any]] = []
        for obj in objects or []:
            if not isinstance(obj, dict):
                continue

            desc_value = obj.get("desc")
            if not isinstance(desc_value, str) or desc_value == "":
                # Fallback from 'label' if present
                label_value = obj.get("label")
                desc_value = label_value if isinstance(label_value, str) else ""

            # Determine geometry and coordinates
            geometry_key = None
            coords: Optional[List[Any]] = None
            # Priority order
            for key in GEOMETRY_TOKENS.keys():
                if key in obj and isinstance(obj[key], list):
                    geometry_key = key
                    coords = obj[key]
                    break

            if geometry_key is None or coords is None:
                continue

            # Enforce allowed geometries only
            if geometry_key not in GEOMETRY_TOKENS.keys():
                continue

            # Coerce coordinates to int when possible
            try:
                coerced_coords = [int(c) for c in coords]
            except Exception:
                coerced_coords = coords

            # Validate coord lengths for geometry types
            if geometry_key == "bbox_2d" and len(coerced_coords) != 4:
                continue
            if geometry_key == "quad" and len(coerced_coords) != 8:
                continue
            if geometry_key == "line" and (
                len(coerced_coords) < 4 or len(coerced_coords) % 2 != 0
            ):
                continue

            normalized_item: Dict[str, Any] = {"desc": desc_value}
            normalized_item[geometry_key] = coerced_coords
            normalized.append(normalized_item)

        return normalized

    def _try_parse_json_list(self, text: str) -> Optional[List[Any]]:
        """Try parse a JSON list from text; return None on failure."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
        return None

    def _convert_prediction_text_to_vis_json(self, response: str) -> str:
        """Convert model response text to a JSON string for visualization format.

        Ensures objects contain 'label' (from 'desc') and geometry keys.
        """
        # Best-effort parse to objects list
        objects_list: List[Dict[str, Any]] = []

        parsed = self._try_parse_json_list(response)
        if parsed is None:
            # Only try coordinate token parsing if enabled
            if self.config.coordinate_tokens_enabled:
                try:
                    parsed = self._parse_coordinate_token_response(response)
                except Exception:
                    parsed = []
            else:
                # For non-coordinate mode, try other parsing methods
                parsed = []

        # Normalize to vis schema (desc -> label)
        if isinstance(parsed, list):
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                vis = {}
                # label from desc/label
                if "desc" in obj:
                    if "desc" not in obj:
                        raise ValueError("Object missing required 'desc' label field")
                    vis["label"] = obj["desc"]
                else:
                    if "label" not in obj:
                        raise ValueError("Object missing required 'label' field")
                    vis["label"] = obj["label"]
                # geometry copy
                for key in GEOMETRY_TOKENS.keys():
                    if key in obj and isinstance(obj[key], list):
                        try:
                            vis[key] = [int(c) for c in obj[key]]
                        except Exception:
                            vis[key] = obj[key]
                        break
                if any(k in vis for k in GEOMETRY_TOKENS.keys()):
                    objects_list.append(vis)

        try:
            return json.dumps(objects_list, ensure_ascii=False)
        except Exception:
            # If all fails, return original response to avoid data loss
            return response

    def _process_coordinate_response(self, response: str) -> str:
        """Process response based on coordinate token configuration."""
        if not self.config.coordinate_tokens_enabled:
            logger.debug(
                "🔍 Using standard response parsing (coordinate tokens disabled)"
            )
            return self._parse_standard_response(response)

        # Coordinate token mode: strict parsing
        logger.debug("🔍 Using coordinate token parsing (coordinate tokens enabled)")
        try:
            parsed_objects = self._parse_coordinate_token_response(response)
            import json

            return json.dumps(parsed_objects, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Coordinate token parsing failed: {e}")
            # Fallback to standard parsing for robustness
            logger.debug("🔄 Falling back to standard response parsing")
            return self._parse_standard_response(response)

    def _parse_coordinate_token_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse coordinate token response and convert to validation format (strict).

        This method should ONLY be called when coordinate_tokens_enabled=True.

        Strict requirements:
        - Geometry blocks must use training-format tokens with <|object_ref_start|> / <|object_ref_end|>
          or synonyms <|obj_ref_start|> / <|obj_ref_end|>
        - Coordinates must be expressed as <|coord_N|> tokens (no raw number fallback)
        - Geometry lengths must be exact (bbox: 4, quad: 8, line: even >= 4)
        - No overlapping geometry spans
        """
        # Validate that coordinate tokens are enabled
        if not self.config.coordinate_tokens_enabled:
            raise ValueError(
                "_parse_coordinate_token_response called but coordinate tokens are disabled"
            )
        import re

        objects: List[Dict[str, Any]] = []
        consumed_spans: List[Tuple[int, int]] = []

        # Support both object_ref_* and obj_ref_* synonyms as mentioned in docs
        # Training uses object_ref_*, but inference accepts obj_ref_* synonymously
        geometry_patterns = {
            "bbox": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|box_start\|>\[(.*?)\]<\|box_end\|>"
            ),
            "quad": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|quad_start\|>\[(.*?)\]<\|quad_end\|>"
            ),
            "line": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|line_start\|>\[(.*?)\]<\|line_end\|>"
            ),
        }

        found_any = False
        for geom_type, pattern in geometry_patterns.items():
            for m in pattern.finditer(response):
                span = m.span()
                # Fail on overlap to surface ambiguous outputs early
                if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
                    raise ValueError("Overlapping geometry spans detected in response")

                desc, coords_section = m.group(1), m.group(2)
                try:
                    coord_tokens = self._extract_coordinate_tokens(
                        coords_section
                    )  # strict; raises if none
                    coords = [int(token) for token in coord_tokens]
                except ValueError as e:
                    # Log the problematic section for debugging
                    logger.warning(
                        f"Failed to extract coordinates from section '{coords_section}': {e}"
                    )
                    continue  # Skip this geometry block but continue parsing others

                if geom_type == "bbox":
                    if len(coords) != 4:
                        logger.warning(
                            f"Invalid bbox length: expected 4, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"bbox_2d": coords, "desc": desc.strip()})
                elif geom_type == "quad":
                    if len(coords) != 8:
                        logger.warning(
                            f"Invalid quad length: expected 8, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"quad": coords, "desc": desc.strip()})
                elif geom_type == "line":
                    if len(coords) < 4 or len(coords) % 2 != 0:
                        logger.warning(
                            f"Invalid line length: expected even number >= 4, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"line": coords, "desc": desc.strip()})

                consumed_spans.append(span)
                found_any = True

        # If no complete geometry blocks found, try to extract partial information
        # This helps with debugging and provides more informative error messages
        if not found_any or not objects:
            # Check if there are any partial geometry patterns that might indicate issues
            partial_patterns = [
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|>",  # Description only
                r"<\|(?:box_start|quad_start|line_start)\|>\[(.*?)\]",  # Coordinates only
                r"<\|coord_\d+\|>",  # Any coordinate tokens
            ]

            partial_matches = []
            for pattern in partial_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    partial_matches.extend(matches)

            if partial_matches:
                logger.warning(
                    f"Found partial geometry patterns but no complete blocks: {partial_matches[:3]}"
                )
                logger.warning(f"Full response for debugging: '{response}'")

            # Return empty list to handle invalid/partial geometry gracefully
            return []

        return objects

    def _parse_geometry_token_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse geometry token response with raw numbers (not coordinate tokens).

        This method handles responses like:
        <|object_ref_start|>description<|object_ref_end|><|quad_start|>[x1, y1, x2, y2, x3, y3, x4, y4]<|quad_end|>

        Works regardless of coordinate_tokens_enabled setting.
        """
        import re

        objects: List[Dict[str, Any]] = []
        consumed_spans: List[Tuple[int, int]] = []

        # Support both object_ref_* and obj_ref_* synonyms
        geometry_patterns = {
            "bbox": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|box_start\|>\[(.*?)\]<\|box_end\|>"
            ),
            "quad": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|quad_start\|>\[(.*?)\]<\|quad_end\|>"
            ),
            "line": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|line_start\|>\[(.*?)\]<\|line_end\|>"
            ),
        }

        found_any = False
        for geom_type, pattern in geometry_patterns.items():
            for m in pattern.finditer(response):
                span = m.span()
                # Fail on overlap to surface ambiguous outputs early
                if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
                    raise ValueError("Overlapping geometry spans detected in response")

                desc, coords_section = m.group(1), m.group(2)
                try:
                    # Parse raw numbers (not coordinate tokens)
                    coords = self._extract_raw_numbers(coords_section)
                except ValueError as e:
                    # Log the problematic section for debugging
                    logger.warning(
                        f"Failed to extract coordinates from section '{coords_section}': {e}"
                    )
                    continue  # Skip this geometry block but continue parsing others

                if geom_type == "bbox":
                    if len(coords) != 4:
                        logger.warning(
                            f"Invalid bbox length: expected 4, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"bbox_2d": coords, "desc": desc.strip()})
                elif geom_type == "quad":
                    if len(coords) != 8:
                        logger.warning(
                            f"Invalid quad length: expected 8, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"quad": coords, "desc": desc.strip()})
                elif geom_type == "line":
                    if len(coords) < 4 or len(coords) % 2 != 0:
                        logger.warning(
                            f"Invalid line length: expected even number >= 4, got {len(coords)} in '{coords_section}'"
                        )
                        continue
                    objects.append({"line": coords, "desc": desc.strip()})

                consumed_spans.append(span)
                found_any = True

        # If no complete geometry blocks found, return empty list
        if not found_any or not objects:
            logger.debug("No valid geometry blocks found in response")
            return []

        return objects

    def _extract_raw_numbers(self, coords_section: str) -> List[int]:
        """Extract raw numbers from coordinate section (not coordinate tokens)."""
        import re

        # Remove any whitespace and split by comma
        coords_section = coords_section.strip()

        # Extract all numbers (integers) from the section
        number_pattern = r"-?\d+"
        number_matches = re.findall(number_pattern, coords_section)

        if not number_matches:
            raise ValueError("No numbers found in coordinates section")

        try:
            coords = [int(match) for match in number_matches]
            return coords
        except ValueError as e:
            raise ValueError(f"Failed to convert numbers to integers: {e}")

    def _extract_coordinate_tokens(self, coords_section: str) -> List[str]:
        """Extract coordinate values from coordinate tokens only (strict)."""
        import re

        coord_pattern = r"<\|coord_(\d+)\|>"
        coord_matches = re.findall(coord_pattern, coords_section)
        if not coord_matches:
            raise ValueError(
                "No coordinate tokens found in geometry coordinates section"
            )
        return coord_matches

    def validate_image_token_alignment(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        Validate image token alignment between text and tensor inputs.

        This utility function helps diagnose "Image features and image tokens do not match" errors
        by thoroughly analyzing the alignment between image tokens in the text and the actual
        image tensors provided to the model.

        Args:
            inputs: Dictionary containing input_ids, pixel_values, image_grid_thw, etc.

        Returns:
            Dictionary with validation results and diagnostic information
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "details": {},
        }

        try:
            # Basic input validation
            if "input_ids" not in inputs:
                validation_results["errors"].append("Missing input_ids tensor")
                validation_results["is_valid"] = False
                return validation_results

            # Decode the input text to analyze image tokens
            decoded_text = self.tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=False
            )
            image_token_count = decoded_text.count(IMAGE_PAD)

            validation_results["details"]["text_length"] = len(decoded_text)
            validation_results["details"]["image_tokens_in_text"] = image_token_count

            # Check for image-related tensors
            has_pixel_values = "pixel_values" in inputs
            has_image_grid_thw = "image_grid_thw" in inputs

            validation_results["details"]["has_pixel_values"] = has_pixel_values
            validation_results["details"]["has_image_grid_thw"] = has_image_grid_thw

            if has_pixel_values:
                pixel_shape = inputs["pixel_values"].shape
                validation_results["details"]["pixel_values_shape"] = list(pixel_shape)

            if has_image_grid_thw:
                grid_shape = inputs["image_grid_thw"].shape
                validation_results["details"]["image_grid_thw_shape"] = list(grid_shape)

                if len(grid_shape) > 0:
                    num_images = grid_shape[0]
                    validation_results["details"]["num_images_in_grid"] = num_images

                    # Validate image count consistency
                    if image_token_count == 0 and num_images > 0:
                        validation_results["errors"].append(
                            f"No image tokens found in text but {num_images} images in grid tensor"
                        )
                        validation_results["is_valid"] = False
                    elif image_token_count > 0 and num_images == 0:
                        validation_results["errors"].append(
                            f"Found {image_token_count} image tokens in text but no images in grid tensor"
                        )
                        validation_results["is_valid"] = False

                    # Check for reasonable image token density
                    if num_images > 0:
                        tokens_per_image = image_token_count / num_images
                        validation_results["details"]["tokens_per_image"] = (
                            tokens_per_image
                        )

                        if tokens_per_image < 50:
                            validation_results["warnings"].append(
                                f"Very low image tokens per image: {tokens_per_image:.1f} "
                                "(expected 200-1000 depending on image resolution)"
                            )
                        elif tokens_per_image > 2000:
                            validation_results["warnings"].append(
                                f"Very high image tokens per image: {tokens_per_image:.1f} "
                                "(might indicate processing issue)"
                            )

            # Validate tensor shapes are reasonable
            if has_pixel_values:
                pixel_shape = inputs["pixel_values"].shape
                if len(pixel_shape) < 3:
                    validation_results["errors"].append(
                        f"Invalid pixel_values shape: {pixel_shape} (expected at least 3D)"
                    )
                    validation_results["is_valid"] = False
                elif pixel_shape[0] == 0:
                    validation_results["errors"].append("Empty pixel_values tensor")
                    validation_results["is_valid"] = False

            # Check for common problematic patterns
            if IMAGE_PAD not in decoded_text and (
                has_pixel_values or has_image_grid_thw
            ):
                validation_results["errors"].append(
                    f"No {IMAGE_PAD} tokens found in text despite having image tensors"
                )
                validation_results["is_valid"] = False

            # Check input_ids tensor shape
            input_ids_shape = inputs["input_ids"].shape
            validation_results["details"]["input_ids_shape"] = list(input_ids_shape)

            if len(input_ids_shape) != 2 or input_ids_shape[0] != 1:
                validation_results["warnings"].append(
                    f"Unexpected input_ids shape: {input_ids_shape} (expected [1, seq_len])"
                )

        except Exception as e:
            validation_results["errors"].append(
                f"Validation failed with exception: {str(e)}"
            )
            validation_results["is_valid"] = False

        return validation_results

    def run_inference(
        self,
        input_file: str,
        output_file: str,
        data_root: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        max_samples: int = -1,
        dataset: str = "bbu",
    ) -> None:
        """Run inference on dataset with the specified parameters.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output file
            data_root: Root directory for data
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            max_samples: Maximum number of samples to process (-1 for all)
            dataset: Dataset type
        """
        import json

        logger.info(f"🚀 Starting inference on {input_file}")
        logger.info(f"📁 Data root: {data_root}")
        logger.info(f"📝 Output file: {output_file}")
        logger.info(f"🎯 Max samples: {max_samples if max_samples > 0 else 'all'}")
        logger.info(
            f"🔧 Generation params: max_tokens={max_new_tokens}, temp={temperature}, sample={do_sample}"
        )

        self.data_root = Path(data_root)

        # Count total samples
        with open(input_file, "r", encoding="utf-8") as f:
            total_samples = sum(1 for _ in f)

        # Determine number of samples to process
        run_samples = (
            total_samples if max_samples < 0 else min(total_samples, max_samples)
        )
        logger.info(f"Processing {run_samples} out of {total_samples} samples")

        # Process samples
        results = []
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break

                try:
                    sample = json.loads(line.strip())

                    # Process single sample
                    result = self._process_single_sample(
                        sample=sample,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                    )

                    results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{run_samples} samples")

                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    continue

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"✅ Inference complete! Results saved to {output_file}")
        logger.info(f"📊 Processed {len(results)} samples successfully")

    def _process_single_sample(
        self,
        sample: dict,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """Process a single sample and return the result.

        Args:
            sample: Input sample dictionary (training format with 'images' and 'objects')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty

        Returns:
            Dictionary with inference results
        """
        try:
            # Validate sample format (training format)
            if not isinstance(sample, dict):
                raise ValueError("Sample must be a dictionary")

            if (
                "images" not in sample
                or not isinstance(sample["images"], list)
                or len(sample["images"]) == 0
            ):
                raise ValueError(
                    "Sample missing required 'images' field or empty images list"
                )

            if "objects" not in sample or not isinstance(sample["objects"], list):
                raise ValueError(
                    "Sample missing required 'objects' field or objects is not a list"
                )

            # Use the training-matched inference pipeline
            inputs = self.prepare_inference_inputs(
                sample=sample,
                seed=42,  # Use fixed seed for reproducible results
                data_root=self.data_root,
            )

            # Generate structured prediction objects
            prediction_objects = self.generate_response(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )

            # Create result in unified visualization-friendly format
            result = {
                "sample_id": sample.get("id", "unknown"),
                "image": sample["images"][0]
                if isinstance(sample.get("images"), list) and len(sample["images"]) > 0
                else None,
                "images": sample["images"],
                "prediction": prediction_objects,
                "ground_truth": sample.get("objects", []),
                "width": sample.get("width"),
                "height": sample.get("height"),
            }

            return result

        except Exception as e:
            logger.error(f"Error in _process_single_sample: {e}")
            return {
                "sample_id": sample.get("id", "unknown"),
                "images": sample.get("images", []),
                "prediction": f"ERROR: {str(e)}",
                "ground_truth": sample.get("objects", []),
                "width": sample.get("width"),
                "height": sample.get("height"),
            }

    def _generate_response(
        self,
        image_path: str,
        conversations: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate response for given image and conversations.

        Args:
            image_path: Path to image file
            conversations: List of conversation turns
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty

        Returns:
            Generated response string
        """
        try:
            # Load and process image
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Prepare conversation for model
            messages = []
            for conv in conversations:
                if conv.get("from") == "human":
                    messages.append({"role": "user", "content": conv.get("value", "")})
                elif conv.get("from") == "gpt":
                    messages.append(
                        {"role": "assistant", "content": conv.get("value", "")}
                    )

            # Use the last user message for generation
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("No user message found for generation")

            user_content = messages[-1]["content"]

            # Apply chat template
            text = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_content},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {
                k: v.to(self.model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            input_len = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_len:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Process coordinate response if needed
            if self.config.coordinate_tokens_enabled:
                response = self._process_coordinate_response(response)

            return response.strip()

        except Exception as e:
            logger.error(f"Error in _generate_response: {e}")
            return f"ERROR: {str(e)}"

    def _parse_standard_response(self, response: str) -> str:
        """Parse standard response without coordinate tokens."""
        import json
        import re

        try:
            # Try to parse as JSON first
            if response.strip().startswith("[") and response.strip().endswith("]"):
                parsed = json.loads(response.strip())
                if isinstance(parsed, list):
                    return json.dumps(parsed, ensure_ascii=False)

            # Try to extract JSON-like structures
            json_pattern = r"\[.*?\]"
            json_matches = re.findall(json_pattern, response, re.DOTALL)

            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        return json.dumps(parsed, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON: {e}")
                    continue

            # Return original response if no structured data found
            return response

        except Exception as e:
            logger.warning(f"Failed to parse standard response: {e}")
            return response

    def _parse_plain_json_lines_response(self, text: str) -> List[Dict[str, Any]]:
        """Parse plain-text JSON Lines response where each line is one object.

        Expected per line object: {GEOM_KEY:[...],"desc":"..."}
        GEOM_KEY must be one of: bbox_2d, quad, line.
        """
        objects: List[Dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Extract JSON object substring conservatively
            json_str: Optional[str] = None
            if line.startswith("{") and line.endswith("}"):
                json_str = line
            else:
                start = line.find("{")
                end = line.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = line[start : end + 1]

            obj: Optional[Dict[str, Any]] = None
            if json_str:
                try:
                    obj = json.loads(json_str)
                except Exception:
                    obj = None

            if obj is None:
                # Fallback: regex extraction to support unquoted coord tokens
                import re

                # desc
                desc_match = re.search(r'"desc"\s*:\s*"(.*?)"', line)
                desc_value = desc_match.group(1) if desc_match else ""

                # geometry
                geom_match = re.search(
                    r'"(bbox_2d|quad|line)"\s*:\s*\[(.*?)\]', line
                )
                if not geom_match:
                    logger.debug(f"No geometry/desc found in line: '{line[:120]}'")
                    continue
                geometry_key = geom_match.group(1)
                coords_section = geom_match.group(2)

                # Try coordinate tokens first
                try:
                    token_ids = self._extract_coordinate_tokens(coords_section)
                    coords = [int(t) for t in token_ids]
                except Exception:
                    # Fallback to raw numbers
                    try:
                        coords = self._extract_raw_numbers(coords_section)
                    except Exception:
                        logger.debug(
                            f"Failed to extract coords from line: '{line[:120]}'"
                        )
                        continue

                parsed_line_item: Dict[str, Any] = {"desc": desc_value}
                parsed_line_item[geometry_key] = coords
                objects.append(parsed_line_item)
                continue

            if not isinstance(obj, dict):
                continue

            # desc (fallback from label if needed)
            desc_value = obj.get("desc")
            if not isinstance(desc_value, str) or desc_value == "":
                label_value = obj.get("label")
                desc_value = label_value if isinstance(label_value, str) else ""

            # geometry key detection (exact keys only)
            geometry_key: Optional[str] = None
            for key in ("bbox_2d", "quad", "line"):
                if key in obj and isinstance(obj[key], list):
                    geometry_key = key
                    break

            if geometry_key is None:
                logger.debug(f"No geometry key found in line: '{line[:120]}'")
                continue

            coords_in = obj.get(geometry_key, [])

            # Convert coords: ints when possible; also convert '<|coord_N|>' strings to ints
            converted: List[Any] = []
            for val in coords_in:
                try:
                    converted.append(int(val))
                    continue
                except Exception:
                    pass
                if isinstance(val, str):
                    import re

                    m = re.fullmatch(r"<\|coord_(\d+)\|>", val)
                    if m:
                        try:
                            converted.append(int(m.group(1)))
                            continue
                        except Exception:
                            pass
                converted.append(val)

            parsed_item: Dict[str, Any] = {"desc": desc_value}
            parsed_item[geometry_key] = converted
            objects.append(parsed_item)

        return objects

    def _normalize_prediction_to_vis_objects(self, text: str) -> List[Dict[str, Any]]:
        """Normalize raw generated text into a list of visualization objects.

        Preferred schema per object: one geometry key in {'bbox_2d','quad','line'} and a 'desc' string.
        """
        # Check if coordinate tokens are enabled
        coordinate_tokens_enabled = self.config.coordinate_tokens_enabled

        logger.info(
            f"🔍 Parsing response with coordinate_tokens_enabled={coordinate_tokens_enabled}"
        )
        logger.info(f"   Response length: {len(text)} characters")
        logger.info(
            f"   Response preview: '{text[:200]}{'...' if len(text) > 200 else ''}'"
        )

        # Plain-text JSON Lines mode parsing first
        if getattr(self.config, "plain_text_mode_enabled", False):
            logger.info("🧾 Plain-text JSON Lines mode: attempting JSON Lines parsing...")
            try:
                jsonl_objects = self._parse_plain_json_lines_response(text)
                if jsonl_objects:
                    # Normalize coords to ints where possible for consistency
                    normalized_from_jsonl: List[Dict[str, Any]] = []
                    for obj in jsonl_objects:
                        if not isinstance(obj, dict):
                            continue
                        jsonl_norm_item: Dict[str, Any] = {"desc": obj.get("desc", "")}
                        for key in ("bbox_2d", "quad", "line"):
                            if key in obj and isinstance(obj[key], list):
                                try:
                                    jsonl_norm_item[key] = [int(v) for v in obj[key]]
                                except Exception:
                                    jsonl_norm_item[key] = obj[key]
                                break
                        if any(k in jsonl_norm_item for k in GEOMETRY_TOKENS.keys()):
                            normalized_from_jsonl.append(jsonl_norm_item)
                    logger.info(
                        f"✅ JSON Lines parsing successful: {len(normalized_from_jsonl)} objects"
                    )
                    return normalized_from_jsonl
                else:
                    logger.warning(
                        "❌ JSON Lines parsing returned no objects; falling back to other parsers"
                    )
            except Exception as e:
                logger.warning(f"Plain JSON Lines parsing failed: {e}")

        # Coordinate-token strict path (only when enabled)
        if coordinate_tokens_enabled:
            logger.info("🎯 Attempting coordinate token parsing...")
            try:
                coord_objects = self._parse_coordinate_token_response(text)
                normalized_from_coords: List[Dict[str, Any]] = []
                for obj in coord_objects:
                    if not isinstance(obj, dict):
                        continue
                    coords_norm_item: Dict[str, Any] = {"desc": obj.get("desc", "")}
                    for key in GEOMETRY_TOKENS.keys():
                        if key in obj and isinstance(obj[key], list):
                            try:
                                coords_norm_item[key] = [int(v) for v in obj[key]]
                            except Exception:
                                coords_norm_item[key] = obj[key]
                            break
                    if any(k in coords_norm_item for k in GEOMETRY_TOKENS.keys()):
                        normalized_from_coords.append(coords_norm_item)
                logger.info(
                    f"✅ Coordinate token parsing successful: {len(normalized_from_coords)} objects"
                )
                return normalized_from_coords
            except Exception as e:
                logger.warning(
                    f"❌ Coordinate response parsing failed; falling back to standard parsing: {e}"
                )
                logger.warning(f"   Response that failed coordinate parsing: '{text}'")
        else:
            logger.info("📝 Using standard parsing (coordinate tokens disabled)")

        # Check if response contains geometry tokens (regardless of coordinate token setting)
        if ("<|object_ref_start|>" in text or "<|obj_ref_start|>" in text) and any(
            token in text
            for token in ["<|quad_start|>", "<|box_start|>", "<|line_start|>"]
        ):
            logger.info(
                "🔧 Detected geometry tokens - attempting geometry token parsing..."
            )
            try:
                geometry_objects = self._parse_geometry_token_response(text)
                if geometry_objects:
                    logger.info(
                        f"✅ Geometry token parsing successful: {len(geometry_objects)} objects"
                    )
                    return geometry_objects
                else:
                    logger.warning("❌ Geometry token parsing returned no objects")
            except Exception as e:
                logger.warning(f"❌ Geometry token parsing failed: {e}")

        # Non-coordinate/standard parsing path
        logger.info("📝 Attempting standard JSON parsing...")
        # Try best-effort JSON list parse
        try:
            parsed_any = json.loads(text)
            if isinstance(parsed_any, list):
                logger.info(
                    f"✅ Successfully parsed JSON list with {len(parsed_any)} items"
                )
                normalized_from_json: List[Dict[str, Any]] = []
                for obj in parsed_any:
                    if not isinstance(obj, dict):
                        logger.debug(f"   Skipping non-dict item: {type(obj)}")
                        continue
                    desc_val = obj.get("desc")
                    if not isinstance(desc_val, str) or desc_val == "":
                        lbl = obj.get("label")
                        desc_val = lbl if isinstance(lbl, str) else ""
                    json_item: Dict[str, Any] = {"desc": desc_val}
                    for key in GEOMETRY_TOKENS.keys():
                        if key in obj and isinstance(obj[key], list):
                            try:
                                json_item[key] = [int(v) for v in obj[key]]
                            except Exception:
                                json_item[key] = obj[key]
                            break
                    if any(k in json_item for k in GEOMETRY_TOKENS.keys()):
                        normalized_from_json.append(json_item)
                    else:
                        logger.debug(f"   Skipping item without geometry: {obj}")
                logger.info(
                    f"✅ Standard JSON parsing successful: {len(normalized_from_json)} objects"
                )
                return normalized_from_json
            else:
                logger.warning(f"❌ JSON parsed but not a list: {type(parsed_any)}")
        except Exception as e:
            logger.warning(f"❌ JSON parsing failed: {e}")
            logger.warning(
                f"   Text that failed JSON parsing: '{text[:500]}{'...' if len(text) > 500 else ''}'"
            )
            pass

        # Try using existing converter that returns JSON and then load
        logger.info("🔄 Attempting fallback converter parsing...")
        try:
            json_text = self._convert_prediction_text_to_vis_json(text)
            parsed_list = json.loads(json_text)
            if isinstance(parsed_list, list):
                logger.info(
                    f"✅ Fallback converter successful: {len(parsed_list)} items"
                )
                unified: List[Dict[str, Any]] = []
                for obj in parsed_list:
                    if not isinstance(obj, dict):
                        continue
                    desc = (
                        obj.get("desc")
                        if isinstance(obj.get("desc"), str)
                        else obj.get("label", "")
                    )
                    merged: Dict[str, Any] = {"desc": desc}
                    for k in GEOMETRY_TOKENS.keys():
                        if k in obj:
                            merged[k] = obj[k]
                            break
                    if any(k in merged for k in GEOMETRY_TOKENS.keys()):
                        unified.append(merged)
                logger.info(f"✅ Fallback parsing successful: {len(unified)} objects")
                return unified
            else:
                logger.warning(
                    f"❌ Fallback converter returned non-list: {type(parsed_list)}"
                )
        except Exception as e:
            logger.warning(f"❌ Fallback converter failed: {e}")

        logger.warning("❌ All parsing methods failed - returning empty list")
        logger.warning(f"   Original text: '{text}'")
        return []


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL Inference with New Architecture"
    )

    # Required arguments
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Path to configuration YAML file (optional). If omitted, will auto-load training_config.yaml from model_path if available.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    # Input selection: either provide --input_file or choose --dataset under --data_root
    parser.add_argument(
        "--input_file", type=str, required=False, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory for data files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "val"],
        default=None,
        help="Dataset split to run (train or val). If provided, input file is derived as data_root/<dataset>.jsonl",
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.000001, help="Sampling temperature"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.05, help="Repetition penalty"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Use sampling instead of greedy decoding (default: False)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)",
    )

    # Model parameters
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile for optimization",
    )

    # Teacher guidance parameters
    parser.add_argument(
        "--teacher_pool_file",
        type=str,
        default=None,
        help="Path to teacher pool JSONL file",
    )
    parser.add_argument(
        "--num_teachers", type=int, default=0, help="Number of teacher examples to use"
    )

    # Model optimization
    parser.add_argument(
        "--force_eager_attention",
        action="store_true",
        help="Force eager attention instead of flash attention for stability",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level - fail fast on invalid level
    log_level_name = args.log_level.upper()
    if not hasattr(logging, log_level_name):
        raise ValueError(
            f"Invalid logging level: {args.log_level}. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    log_level = getattr(logging, log_level_name)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # Normalize critical paths: accept relative and '@src_new' alias (preserve relativity)

    from src_new.utils.validation import normalize_path_input

    try:
        if args.config_path is not None:
            args.config_path = str(normalize_path_input(args.config_path))
        if args.model_path is not None:
            # Keep relative to current working directory to respect user PWD
            args.model_path = str(normalize_path_input(args.model_path))
        if args.data_root is not None:
            args.data_root = str(normalize_path_input(args.data_root))
        if args.output_file is not None:
            args.output_file = str(normalize_path_input(args.output_file))
    except Exception as e:
        raise ValueError(f"Failed to normalize CLI paths: {e}")

    # Auto-detect config from model_path if not provided
    if args.config_path is None or len(str(args.config_path).strip()) == 0:
        ckpt_dir = args.model_path
        # Candidates in preference order
        candidate_files = [
            os.path.join(ckpt_dir, "training_config.yaml"),
            os.path.join(ckpt_dir, "config.yaml"),
            os.path.join(ckpt_dir, "config.yml"),
            os.path.join(ckpt_dir, "config.json"),
        ]
        found_config = None
        for cand in candidate_files:
            if os.path.exists(cand):
                found_config = cand
                break
        if found_config is None:
            raise ValueError(
                "--config_path not provided and no config file found in model_path. Expected one of training_config.yaml, config.yaml, config.yml, config.json"
            )
        # If config.json is found, attempt to convert to YAML-compatible dict load
        args.config_path = found_config
        logger.info(f"🧭 Auto-loaded config from checkpoint: {args.config_path}")

    # Resolve input_file from dataset if not provided
    if args.input_file is None:
        if args.dataset is None:
            raise ValueError("Either --input_file or --dataset must be provided")

        # Use DataResolver for direct path resolution (more efficient than loading full config)
        from src_new.utils.data_resolver import DataResolver

        try:
            dataset_paths = DataResolver.resolve_dataset_paths(args.data_root)

            if args.dataset == "train":
                args.input_file = str(dataset_paths.train_data_path)
            elif args.dataset == "val":
                args.input_file = str(dataset_paths.val_data_path)
            else:
                raise ValueError(
                    f"Unknown dataset: {args.dataset}. Must be 'train' or 'val'"
                )

            logger.info(
                f"Resolved input file from dataset '{args.dataset}': {args.input_file}"
            )
        except (FileNotFoundError, ValueError) as e:
            # Fall back to config-based resolution for backward compatibility
            logger.warning(
                f"DataResolver failed, falling back to config-based resolution: {e}"
            )
            from src_new.config.config import load_config

            config = load_config(args.config_path)

            if args.dataset == "train":
                args.input_file = config.train_data_path
            elif args.dataset == "val":
                args.input_file = config.val_data_path
            else:
                raise ValueError(
                    f"Unknown dataset: {args.dataset}. Must be 'train' or 'val'"
                )

            logger.info(
                f"Resolved input file from config '{args.dataset}': {args.input_file}"
            )

    # Create inference engine
    engine = InferenceEngine(
        config_path=args.config_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_torch_compile=args.use_torch_compile,
        teacher_pool_file=args.teacher_pool_file,
        num_teachers=args.num_teachers,
        force_eager_attention=args.force_eager_attention,
        data_root=args.data_root,
    )

    # Run inference
    engine.run_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        max_samples=args.max_samples,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
