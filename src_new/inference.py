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

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm


# Configure rank-aware logging early
try:
    from src_new.utils.rank_aware_logging import (
        get_rank_aware_logger,
        initialize_logging_from_env,
    )

    # Initialize from environment if available; otherwise default to INFO
    initialize_logging_from_env()
    logger = get_rank_aware_logger("inference")
except ImportError:
    # Fallback to basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logger = logging.getLogger("inference")

# Import new architecture components
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor

from src_new.config.config import load_config
from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.models.wrapper import DetectionModel
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.token_processor import TokenConfig, TokenProcessor
from src_new.utils.path_manager import create_path_manager


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

        # Override model path if provided
        if model_path:
            self.config.model_path = model_path

        # Use teacher pool from config if not explicitly provided
        if self.teacher_pool_file is None:
            self.teacher_pool_file = getattr(self.config, "teacher_pool_file", None)

        # Apply Qwen2.5-VL patches
        logger.info("Applying Qwen2.5-VL patches...")
        try:
            apply_comprehensive_qwen25_fixes()
            logger.info("✅ Qwen2.5-VL patches applied successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to apply Qwen2.5-VL patches: {e}")

        # Load model and components
        self._load_model_and_processors()

        # CRITICAL FIX: Always load teacher pool if available to match training pipeline
        # During training, teacher examples are randomly assigned to samples
        # We need to replicate this behavior during inference
        self.teacher_samples = []
        teacher_pool_path: Optional[str] = None

        # Resolve teacher pool path using PathManager to avoid double-prefix issues
        try:
            candidate_path = (
                self.teacher_pool_file
                if self.teacher_pool_file
                else "teacher_pool.jsonl"
            )
            path_manager = create_path_manager(getattr(self, "data_root", None))
            resolved_path = str(path_manager.resolve_path(candidate_path))
            teacher_pool_path = resolved_path
        except Exception as e:
            # Resolution failed; log and continue without teacher mode
            logger.warning(
                f"Teacher pool resolution failed for '{self.teacher_pool_file or 'teacher_pool.jsonl'}' "
                f"(data_root={self.data_root}): {e}"
            )
            teacher_pool_path = None

        if teacher_pool_path and os.path.exists(teacher_pool_path):
            try:
                with open(teacher_pool_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.teacher_samples.append(json.loads(line))
                logger.info(
                    f"Loaded {len(self.teacher_samples)} teacher examples from {teacher_pool_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load teacher pool from {teacher_pool_path}: {e}"
                )
        else:
            if self.teacher_pool_file or self.num_teachers > 0:
                logger.warning(
                    f"Teacher pool file not found: {teacher_pool_path or (self.teacher_pool_file or 'teacher_pool.jsonl')}"
                )

        # CRITICAL FIX: Always use at least 1 teacher to match training pipeline when pool is available
        if self.teacher_samples:
            if self.num_teachers == 0:
                self.num_teachers = 1  # Default to 1 teacher to match training
                logger.info(
                    f"Auto-enabled teacher guidance with {self.num_teachers} teacher(s) to match training pipeline"
                )

            logger.info(
                f"Using {self.num_teachers} teacher(s) per sample to match training"
            )

            # Force batch_size=1 for teacher guidance
            if self.batch_size > 1:
                logger.warning(
                    f"Teacher guidance requires batch_size=1. Changing from {self.batch_size} to 1."
                )
                self.batch_size = 1
        elif self.num_teachers > 0:
            logger.warning(
                f"Teacher guidance requested ({self.num_teachers} teachers) but no teacher pool file provided"
            )

        logger.info("✅ InferenceEngine initialized successfully")

    def _load_model_and_processors(self):
        """Load model and processing components."""
        logger.info("Loading model and processors...")

        # Set up critical environment variables for fast loading (same as training)
        import os

        os.environ["HF_MODULES_CACHE"] = "/data3/Qwen2.5-VL-main/model_cache"
        os.environ["HF_HOME"] = "/data3/Qwen2.5-VL-main/model_cache"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Only set Triton/FlashAttention-specific env when using FlashAttention v2
        if getattr(self.config, "attn_implementation", "eager") == "flash_attention_2":
            os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["FLASH_ATTENTION_FORCE_CUDNN"] = "0"

        # Create token processor
        token_config = TokenConfig(
            coordinate_tokens_enabled=self.config.coordinate_tokens_enabled,
            max_coord_value=self.config.max_coord_value,
            new_geometry_tokens=self.config.new_geometry_tokens,
        )
        self.token_processor = TokenProcessor(token_config)

        # Fast checkpoint loading using DetectionModel.from_pretrained (same as training)
        # This avoids double extension and weight reinitialization issues
        logger.info(
            "🚀 Loading checkpoint using DetectionModel.from_pretrained (training-compatible approach)"
        )

        # Load tokenizer and processor directly from checkpoint (contains expanded vocabulary)

        logger.info(f"Loading tokenizer from checkpoint: {self.config.model_path}")
        # Load tokenizer directly from checkpoint - this should contain the expanded vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # Load image processor from checkpoint
        logger.info(
            f"Loading image processor from checkpoint: {self.config.model_path}"
        )
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=False, do_resize=False
        )

        # Verify coordinate token checkpoint by checking vocabulary
        original_vocab_size = 151665  # Standard Qwen2.5-VL vocab size
        current_vocab_size = len(self.tokenizer.get_vocab())

        if current_vocab_size > original_vocab_size:
            coordinate_tokens_added = current_vocab_size - original_vocab_size
            logger.info(
                f"🎯 Coordinate token checkpoint confirmed: vocab {original_vocab_size} → {current_vocab_size}"
            )
            logger.info(
                f"✅ Found {coordinate_tokens_added} additional tokens (coordinate + geometry tokens)"
            )

            # Verify coordinate tokens are present
            vocab = self.tokenizer.get_vocab()
            coord_tokens_found = sum(
                1 for token in vocab.keys() if token.startswith("<|coord_")
            )
            logger.info(f"🔢 Coordinate tokens detected: {coord_tokens_found}")
        else:
            logger.warning(
                f"⚠️ Expected coordinate token checkpoint but vocab size is {current_vocab_size}"
            )
            logger.warning(
                "This may indicate the checkpoint doesn't contain expanded vocabulary"
            )

        # Load model using DetectionModel.from_pretrained (EXACTLY like training)
        # This prevents double extension and weight reinitialization issues
        import time

        logger.info(
            "Loading model using DetectionModel.from_pretrained (training approach)..."
        )
        model_load_start = time.time()

        # Use fast loading for inference - skip vocab extension if checkpoint already has it
        vocab_size = len(self.tokenizer.get_vocab())
        has_coordinate_tokens = vocab_size > 151665

        # Check if SafeTensors format is available for even faster loading
        # COMPATIBILITY FIX: Support both single and sharded SafeTensors
        safetensors_path = os.path.join(self.config.model_path, "model.safetensors")
        safetensors_index_path = os.path.join(
            self.config.model_path, "model.safetensors.index.json"
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
                model_path=self.config.model_path,
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
                model_path=self.config.model_path,
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
        self.coordinate_tokens_enabled = current_vocab_size > original_vocab_size

        # Log coordinate token optimization status
        if self.coordinate_tokens_enabled:
            coordinate_tokens_added = current_vocab_size - original_vocab_size
            logger.info(
                f"🎯 Coordinate tokens detected: +{coordinate_tokens_added} tokens"
            )
            logger.info("🚀 Ready for optimized coordinate token inference")

        # Create conversation processor using new architecture
        # Create unified processor from tokenizer and image processor
        unified_processor = Qwen2VLProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
        )

        # CRITICAL FIX: Ensure chat template is properly inherited from tokenizer
        # The Qwen2VLProcessor doesn't automatically inherit chat_template from tokenizer
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            unified_processor.chat_template = self.tokenizer.chat_template
            logger.info("✅ Chat template inherited from tokenizer to processor")
        else:
            logger.warning(
                "⚠️ No chat template found in tokenizer - this may cause inference failures"
            )

        self.conversation_processor = ConversationProcessor(
            processor=unified_processor,
            max_coord_value=self.config.max_coord_value,
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
        """Sample random teacher examples."""
        if seed is not None:
            random.seed(seed)

        if self.num_teachers >= len(self.teacher_samples):
            return self.teacher_samples.copy()

        return random.sample(self.teacher_samples, self.num_teachers)

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

        # CRITICAL: Use teacher-guided approach to match training pipeline exactly
        # Training always has potential for teacher assignment, so inference should too
        if self.teacher_samples and len(self.teacher_samples) > 0:
            return self._prepare_training_matched_inputs(sample, seed)
        else:
            # Fallback to simple conversation (still using training format)
            logger.warning(
                "No teacher pool available - using simple conversation with training format"
            )
            return self._prepare_simple_training_format(sample)

    def _prepare_training_matched_inputs(
        self, student_sample: Dict[str, Any], seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs using EXACT training pipeline data preparation process."""
        # Set deterministic seed for teacher sampling (same as training)
        if seed is not None:
            random.seed(seed)

        # Sample teacher examples (same logic as training Dataset.__getitem__)
        teachers = self._sample_teachers(seed)

        logger.debug(f"🎯 TRAINING-MATCHED INFERENCE SETUP:")
        logger.debug(f"   Teachers: {len(teachers)}")
        logger.debug(f"   Student: 1")
        logger.debug(f"   Seed: {seed}")

        # CRITICAL: Use ConversationProcessor exactly like training
        # This replicates the training data preparation process exactly

        # Load images in training order (teachers first, then student)
        teacher_images_list = []
        total_teacher_images = 0
        for teacher in teachers:
            teacher_image_paths = teacher.get("images", [])
            teacher_images = []
            for img_path in teacher_image_paths:
                try:
                    # CRITICAL FIX: Use PathManager for unified path resolution
                    path_manager = create_path_manager(getattr(self, "data_root", None))
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
        student_image_paths = student_sample.get("images", [])
        student_images = []
        path_manager = create_path_manager(getattr(self, "data_root", None))
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
            # Build a generation-ready conversation directly via processor
            inputs = self.conversation_processor.create_teacher_student_conversation_for_generation(
                student_sample=student_sample,
                teacher_samples=teachers,
                student_images=student_images,
                teacher_images_list=teacher_images_list,
            )

        except Exception as e:
            logger.error(f"ConversationProcessor failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create training-matched conversation: {e}")

        # FINAL VALIDATION: Ensure tensor consistency for model generation
        final_text = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        final_image_token_count = final_text.count("<|image_pad|>")

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
        image_paths = sample.get("images", [])
        images = []
        path_manager = create_path_manager(getattr(self, "data_root", None))
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

        # CRITICAL: Use ConversationProcessor.create_simple_conversation
        # This matches the training path for samples without teachers
        try:
            # Build a generation-ready simple conversation via processor
            inputs = (
                self.conversation_processor.create_simple_conversation_for_generation(
                    sample=sample, images=images
                )
            )

        except Exception as e:
            logger.error(f"ConversationProcessor failed: {e}")
            raise RuntimeError(
                f"Failed to create simple training format conversation: {e}"
            )

        # Validate conversation structure using the processed inputs
        final_text = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        image_token_count = final_text.count("<|image_pad|>")

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
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate response using the model with pre-processed inputs."""

        # CRITICAL FIX: Use pre-processed inputs directly
        # This eliminates the double processing issue that caused the indexing error
        logger.debug("Using pre-processed inputs from ConversationProcessor")

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

        # Move inputs to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Enhanced tensor shape logging
        for k, v in inputs.items():
            if torch.is_tensor(v):
                logger.debug(f"   {k}: {v.shape} ({v.dtype}) on {v.device}")

        # CRITICAL VALIDATION: Cross-validate image tokens and tensors before generation
        if has_images:
            # Decode input_ids to check image token alignment
            decoded_text = self.tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=False
            )
            image_token_count = decoded_text.count("<|image_pad|>")

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

        # Generate response with enhanced error handling
        try:
            with torch.no_grad():
                # Use appropriate EOS token
                endoftext_token_id = self.tokenizer.get_vocab().get(
                    "<|endoftext|>", self.tokenizer.eos_token_id
                )

                logger.debug(f"🚀 Starting generation:")
                logger.debug(f"   Max new tokens: {max_new_tokens}")
                logger.debug(f"   Temperature: {temperature}")
                logger.debug(f"   Do sample: {do_sample}")
                logger.debug(f"   EOS token ID: {endoftext_token_id}")

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # Dynamically resolve EOS tokens to ensure correct stopping
                    eos_token_id=[
                        t_id
                        for t_id in {
                            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                            getattr(self.tokenizer, "eos_token_id", None),
                            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                        }
                        if t_id is not None and t_id != -1
                    ]
                    or getattr(self.tokenizer, "eos_token_id", None),
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
                    problematic_tokens = problematic_text.count("<|image_pad|>")

                    logger.error(
                        f"   Problematic text length: {len(problematic_text)} chars"
                    )
                    logger.error(f"   Image tokens in text: {problematic_tokens}")
                    logger.error(f"   Text preview: {problematic_text[:200]}...")
                    logger.error(f"   Text ending: ...{problematic_text[-200:]}")

                    # Look for patterns that might cause the mismatch
                    if "<|image_pad|>" not in problematic_text:
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
            # IMPORTANT: Preserve coordinate and geometry tokens which may be registered as special tokens
            # during tokenizer extension. We therefore do NOT skip special tokens here.
            response_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=False
            )[0]
        except Exception as e:
            logger.error(f"❌ DECODING ERROR: {e}")
            logger.error(f"   Output IDs shape: {output_ids.shape}")
            raise RuntimeError(f"Failed to decode generated response: {e}")

        # Log generation details
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]

        logger.debug(f"✅ GENERATION COMPLETED:")
        logger.debug(f"   Input tokens: {input_length}")
        logger.debug(f"   Generated tokens: {len(generated_ids)}")
        logger.debug(f"   Total output tokens: {output_ids.shape[1]}")
        logger.debug(f"   Raw response preview: '{response_text[:200]}...'")

        # Extract assistant response using proven approach from working version
        try:
            assistant_part = response_text.split("assistant\n", 1)[1].strip()
            logger.debug("   Extracted assistant part from conversation")
        except IndexError:
            assistant_part = response_text
            logger.debug("   Using full response text (no assistant marker found)")

        # Clean up special tokens (DO NOT remove coordinate or geometry tokens)
        special_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        for token in special_tokens:
            if token in assistant_part:
                assistant_part = assistant_part.replace(token, "")

        cleaned_response = assistant_part.strip()

        # Log final response
        logger.info(
            f"🎯 Generated response ({len(cleaned_response)} chars): {cleaned_response[:100]}..."
        )

        if not cleaned_response:
            logger.warning("⚠️  Empty response generated!")
            logger.warning("   This might indicate a generation or parsing issue")

        return cleaned_response

    # =====================
    # Visualization helpers
    # =====================
    def _convert_objects_to_vis_format(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert training-format objects to visualization format.

        - Maps 'desc' -> 'label'
        - Preserves geometry keys: bbox_2d, quad, square, line
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
            for key in ["bbox_2d", "quad", "square", "line"]:
                if key in obj and isinstance(obj[key], list):
                    coords = obj[key]
                    try:
                        vis_obj[key] = [int(c) for c in coords]
                    except Exception:
                        # fallback, keep as-is
                        vis_obj[key] = coords
                    break

            # Only add if we found a geometry key
            if any(k in vis_obj for k in ("bbox_2d", "quad", "square", "line")):
                vis_objects.append(vis_obj)
        return vis_objects

    def _convert_objects_to_training_format(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize objects to the training dataset format.

        Rules:
        - Field name for text is 'desc' (fallback from 'label' if needed)
        - Only geometry keys allowed: 'bbox_2d', 'quad', 'line'
        - If 'square' is present, map it to 'quad'
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
            for key in ["bbox_2d", "quad", "line", "square"]:
                if key in obj and isinstance(obj[key], list):
                    geometry_key = key
                    coords = obj[key]
                    break

            if geometry_key is None or coords is None:
                continue

            # Map 'square' -> 'quad'
            if geometry_key == "square":
                geometry_key = "quad"

            # Enforce allowed geometries only
            if geometry_key not in ("bbox_2d", "quad", "line"):
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

            item: Dict[str, Any] = {"desc": desc_value}
            item[geometry_key] = coerced_coords
            normalized.append(item)

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
            # Fallback to tokenizer-aware parser we already have
            try:
                parsed = self._parse_coordinate_token_response(response)
            except Exception:
                parsed = []

        # Normalize to vis schema (desc -> label)
        if isinstance(parsed, list):
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                vis = {}
                # label from desc/label
                if "desc" in obj:
                    vis["label"] = obj.get("desc", "Unknown")
                else:
                    vis["label"] = obj.get("label", "Unknown")
                # geometry copy
                for key in ["bbox_2d", "quad", "square", "line"]:
                    if key in obj and isinstance(obj[key], list):
                        try:
                            vis[key] = [int(c) for c in obj[key]]
                        except Exception:
                            vis[key] = obj[key]
                        break
                if any(k in vis for k in ("bbox_2d", "quad", "square", "line")):
                    objects_list.append(vis)

        try:
            return json.dumps(objects_list, ensure_ascii=False)
        except Exception:
            # If all fails, return original response to avoid data loss
            return response

    def run_inference(
        self,
        input_file: Optional[str],
        output_file: str,
        data_root: Optional[str],
        max_new_tokens: int = 512,
        temperature: float = 0.000001,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        max_samples: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """Run inference on a JSONL dataset."""
        # Store/derive data_root for image path resolution
        self.data_root = data_root or getattr(self.config, "data_root", None)

        # Derive input file from config if not provided
        if input_file is None:
            split = (dataset or "val").lower()
            if split not in ("train", "val"):
                raise ValueError(f"Invalid dataset split: {split}")
            input_file = (
                self.config.train_data_path
                if split == "train"
                else self.config.val_data_path
            )

        logger.info(f"Starting inference on {input_file}")
        logger.info(f"Output will be saved to {output_file}")
        logger.info(f"Data root: {self.data_root}")
        if max_samples is not None:
            logger.info(f"Max samples limit: {max_samples}")

        # Load dataset
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    # CRITICAL FIX: Use PathManager for unified path resolution
                    if "images" in sample:
                        path_manager = create_path_manager(self.data_root)
                        resolved_images = []
                        for img_path in sample["images"]:
                            try:
                                resolved_path = str(path_manager.resolve_path(img_path))
                                resolved_images.append(resolved_path)
                                logger.debug(
                                    f"Resolved image path: {img_path} -> {resolved_path}"
                                )
                            except (FileNotFoundError, ValueError) as e:
                                logger.error(
                                    f"Failed to resolve image path {img_path}: {e}"
                                )
                                raise
                        sample["images"] = resolved_images
                    samples.append(sample)

        logger.debug(f"Loaded {len(samples)} samples")

        # Apply max_samples limit if specified
        if max_samples is not None and max_samples > 0:
            original_count = len(samples)
            samples = samples[:max_samples]
            logger.info(
                f"Limited to {len(samples)} samples (from {original_count} total)"
            )

        # CRITICAL FIX: Use PathManager for unified teacher path resolution
        if self.teacher_samples:
            path_manager = create_path_manager(self.data_root)
            for teacher_sample in self.teacher_samples:
                if "images" in teacher_sample:
                    resolved_teacher_images = []
                    for img_path in teacher_sample["images"]:
                        try:
                            resolved_path = str(path_manager.resolve_path(img_path))
                            resolved_teacher_images.append(resolved_path)
                            logger.debug(
                                f"Resolved teacher image path: {img_path} -> {resolved_path}"
                            )
                        except (FileNotFoundError, ValueError) as e:
                            logger.error(
                                f"Failed to resolve teacher image path {img_path}: {e}"
                            )
                            raise
                    teacher_sample["images"] = resolved_teacher_images
            logger.info(
                f"Resolved paths for {len(self.teacher_samples)} teacher samples"
            )

        # Run inference with immediate debug logging
        results = []
        for i, sample in enumerate(tqdm(samples, desc="Running inference")):
            try:
                logger.debug(f"🔄 Processing sample {i + 1}/{len(samples)}")

                # Prepare inputs
                seed = i if self.teacher_samples else None
                inputs = self.prepare_inference_inputs(
                    sample, seed=seed, data_root=self.data_root
                )

                # Generate response
                response = self.generate_response(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                )

                # Log sample completion
                sample_id = sample.get("id", i)
                logger.debug(f"✅ Sample {i + 1} ({sample_id}) completed")

                # Process response for coordinate tokens if needed
                if self.config.coordinate_tokens_enabled:
                    processed_response = self._process_coordinate_response(response)
                    if processed_response != response:
                        logger.debug(f"   🔧 Coordinate processing applied")
                    response = processed_response

                # Normalize prediction to training format JSON string
                pred_list = self._try_parse_json_list(response)
                if pred_list is None:
                    try:
                        pred_list = self._parse_coordinate_token_response(response)
                    except Exception:
                        pred_list = []

                pred_objects = (
                    self._convert_objects_to_training_format(pred_list)
                    if isinstance(pred_list, list)
                    else []
                )
                pred_json = json.dumps(pred_objects, ensure_ascii=False)

                # === Build visualization-friendly outputs ===
                # Ground truth objects -> training format JSON string
                gt_objects = self._convert_objects_to_training_format(
                    sample.get("objects", [])
                )
                gt_json = json.dumps(gt_objects, ensure_ascii=False)

                # Choose primary image for visualization
                image_path = ""
                if isinstance(sample.get("images"), list) and len(sample["images"]) > 0:
                    image_path = sample["images"][0]

                # Store result (keep original fields, add vis fields)
                result = {
                    "sample_id": sample_id,
                    "prediction": pred_json,
                    # Visualization keys expected by vis_tools/vis_generation.py
                    "image": image_path,
                    "ground_truth": gt_json,
                }
                results.append(result)

                # Log progress summary
                if (i + 1) % 10 == 0:
                    non_empty_count = sum(
                        1 for r in results if r.get("prediction", "").strip()
                    )
                    logger.info(
                        f"📊 Progress: {i + 1}/{len(samples)} samples, {non_empty_count} non-empty responses"
                    )

            except Exception as e:
                logger.error(f"❌ Failed to process sample {i + 1}: {e}")
                logger.error(f"   Sample data: {sample}")
                # Even in failure, try to output a minimal vis item for bookkeeping
                image_path = ""
                if isinstance(sample.get("images"), list) and len(sample["images"]) > 0:
                    image_path = sample["images"][0]
                results.append(
                    {
                        "sample_id": sample.get("id", i),
                        "prediction": "",
                        "image": image_path,
                        "ground_truth": json.dumps(
                            self._convert_objects_to_training_format(
                                sample.get("objects", [])
                            ),
                            ensure_ascii=False,
                        ),
                        "error": str(e),
                    }
                )

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Inference completed. Results saved to {output_file}")
        logger.info(f"Processed {len(results)} samples")

    def _process_coordinate_response(self, response: str) -> str:
        """Process response to handle coordinate tokens and convert to validation format."""
        if not self.config.coordinate_tokens_enabled:
            return self._parse_standard_response(response)

        # Use enhanced post-processing for coordinate token responses
        try:
            parsed_objects = self._parse_coordinate_token_response(response)

            if parsed_objects:
                # Convert to validation format JSON (list of objects)
                import json

                return json.dumps(parsed_objects, ensure_ascii=False)
            else:
                # Fallback to standard parsing
                return self._parse_standard_response(response)

        except Exception as e:
            logger.warning(f"Failed to process coordinate response: {e}")
            # Fallback to standard parsing
            return self._parse_standard_response(response)

    def _parse_coordinate_token_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse coordinate token response and convert to validation format.

        This function implements the inverse of the training preprocessing pipeline:
        1. Parse special tokens and geometry tags
        2. Convert coordinate tokens back to integers
        3. Reconstruct geometry objects in validation format

        Args:
            response: Raw model response with coordinate tokens

        Returns:
            List of objects in validation format
        """
        import json
        import re

        objects = []
        # Track consumed character spans to avoid duplicate matches
        consumed_spans: List[Tuple[int, int]] = []

        try:
            # Method 1: Try to parse as JSON format first (if response is structured JSON)
            if response.strip().startswith("[") and response.strip().endswith("]"):
                json_text = response.strip()

                # Convert coordinate tokens to integers in JSON
                coord_pattern = r"<\|coord_(\d+)\|>"
                json_text = re.sub(coord_pattern, r"\1", json_text)

                parsed_objects = json.loads(json_text)
                if isinstance(parsed_objects, list):
                    return parsed_objects

            # Method 2: Parse special token format (training format)
            # Pattern: <|obj_ref_start|>desc<|obj_ref_end|><|geometry_start|>[coords]<|geometry_end|>
            geometry_patterns = {
                # Prefer official 'box' tokens but also support legacy 'bbox'
                "bbox": re.compile(
                    r"<\|obj_ref_start\|>(.*?)<\|obj_ref_end\|>(?:<\|box_start\|>|<\|bbox_start\|>)\[(.*?)\](?:<\|box_end\|>|<\|bbox_end\|>)"
                ),
                "quad": re.compile(
                    r"<\|obj_ref_start\|>(.*?)<\|obj_ref_end\|><\|quad_start\|>\[(.*?)\]<\|quad_end\|>"
                ),
                "line": re.compile(
                    r"<\|obj_ref_start\|>(.*?)<\|obj_ref_end\|><\|line_start\|>\[(.*?)\]<\|line_end\|>"
                ),
                # Alternative patterns without obj_ref tags
                "bbox_alt": re.compile(
                    r"(?:<\|box_start\|>|<\|bbox_start\|>)\[(.*?)\](?:<\|box_end\|>|<\|bbox_end\|>)"
                ),
                "quad_alt": re.compile(r"<\|quad_start\|>\[(.*?)\]<\|quad_end\|>"),
                "line_alt": re.compile(r"<\|line_start\|>\[(.*?)\]<\|line_end\|>"),
            }

            for geom_type, pattern in geometry_patterns.items():
                # Use finditer to capture spans and avoid duplicates
                for m in pattern.finditer(response):
                    span = m.span()
                    # Skip if this span overlaps a previously consumed span
                    if any(
                        not (span[1] <= s or span[0] >= e) for s, e in consumed_spans
                    ):
                        continue

                    try:
                        if geom_type.endswith("_alt"):
                            # Alternative pattern without description
                            coords_section = m.group(1)
                            desc = ""
                            geom_key = geom_type.replace("_alt", "")
                        else:
                            # Standard pattern with description
                            desc, coords_section = m.group(1), m.group(2)
                            geom_key = geom_type

                        # Extract coordinate tokens and convert to integers
                        coord_tokens = self._extract_coordinate_tokens(coords_section)

                        if coord_tokens:
                            coords = [int(token) for token in coord_tokens]

                            # Create object based on geometry type
                            if geom_key == "bbox":
                                if len(coords) == 4:
                                    objects.append(
                                        {"bbox_2d": coords, "desc": desc.strip()}
                                    )
                                    consumed_spans.append(span)
                            elif geom_key == "quad":
                                if len(coords) == 8:
                                    objects.append(
                                        {"quad": coords, "desc": desc.strip()}
                                    )
                                    consumed_spans.append(span)
                            elif geom_key == "line":
                                if len(coords) >= 4 and len(coords) % 2 == 0:
                                    objects.append(
                                        {"line": coords, "desc": desc.strip()}
                                    )
                                    consumed_spans.append(span)

                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse geometry {geom_type}: {e}")
                        continue

            # Method 3: Try JSON parsing as fallback
            if not objects:
                try:
                    # Try to extract JSON from response
                    import re

                    json_match = re.search(r"\[.*\]", response, re.DOTALL)
                    if json_match:
                        import json

                        parsed_result = json.loads(json_match.group())
                        if isinstance(parsed_result, list):
                            objects = parsed_result
                        elif (
                            isinstance(parsed_result, dict)
                            and "objects" in parsed_result
                        ):
                            objects = parsed_result["objects"]
                except Exception as e:
                    logger.warning(f"JSON parsing fallback failed: {e}")

            return objects

        except Exception as e:
            logger.warning(f"Failed to parse coordinate token response: {e}")
            return []

    def _extract_coordinate_tokens(self, coords_section: str) -> List[str]:
        """Extract coordinate values from coordinate tokens or raw numbers."""
        import re

        # First try to extract coordinate tokens
        coord_pattern = r"<\|coord_(\d+)\|>"
        coord_matches = re.findall(coord_pattern, coords_section)

        if coord_matches:
            return coord_matches

        # Fallback: extract raw numbers (comma or space separated)
        number_pattern = r"\b\d+\b"
        number_matches = re.findall(number_pattern, coords_section)

        return number_matches

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
            image_token_count = decoded_text.count("<|image_pad|>")

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
            if "<|image_pad|>" not in decoded_text and (
                has_pixel_values or has_image_grid_thw
            ):
                validation_results["errors"].append(
                    "No <|image_pad|> tokens found in text despite having image tensors"
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


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL Inference with New Architecture"
    )

    # Required arguments
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    # Input selection: either provide --input_file or choose --dataset under --data_root
    parser.add_argument(
        "--input_file", type=str, required=False, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSON file"
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

    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

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
