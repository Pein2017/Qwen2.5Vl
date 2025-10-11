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
from src_new.config.config import load_config
from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.processing.conversation import ConversationBuilder
from src_new.processing.parse_generated import (
    convert_objects_to_vis,
    convert_response_to_vis_json,
    deduplicate_objects,
    parse_geometry_response,
)
from src_new.processing.special_tokens import (
    ASSISTANT_HEADER,
    END_OF_TEXT,
    GEOMETRY_TOKENS,
    IM_END,
    IM_START,
    IMAGE_PAD,
)

# Import prompts from training pipeline to ensure consistency
from src_new.processing.templates import CONSTANTS, get_system_prompt
from src_new.utils.data_resolver import DataResolver
from src_new.utils.hf_components import build_hf_components
from src_new.utils.path_manager import create_path_manager
from src_new.utils.validation import PathValidationError


def _normalize_prediction_to_vis_objects_impl(
    engine: "InferenceEngine", text: str
) -> List[Dict[str, Any]]:
    """Normalize raw generated text into a list of visualization objects."""

    logger.info("🔍 Parsing response text for geometry objects")
    logger.info(f"   Response length: {len(text)} characters")
    logger.info(
        f"   Response preview: '{text[:200]}{'...' if len(text) > 200 else ''}'"
    )

    if "\n" in text or "\r" in text:
        try:
            text = text.replace("\r", "").replace("\n", "")
        except Exception:
            pass
    try:
        text = engine._normalize_punctuations(text)
    except Exception:
        pass

    logger.info("📝 Using standard parsing (coordinate tokens removed)")

    has_geometry_tokens = (
        "<|object_ref_start|>" in text or "<|obj_ref_start|>" in text
    ) and any(
        token in text for token in ["<|quad_start|>", "<|box_start|>", "<|line_start|>"]
    )

    if has_geometry_tokens:
        logger.info(
            "🔧 Detected geometry tokens - attempting strict geometry parsing..."
        )
        geometry_objects = parse_geometry_response(text, tolerant=False)
        if geometry_objects:
            logger.info(
                f"✅ Geometry token parsing successful: {len(geometry_objects)} objects"
            )
            return geometry_objects
        logger.warning("❌ Geometry token parsing returned no objects (strict)")

        logger.info("🔄 Attempting tolerant geometry parsing...")
        tolerant_objects = parse_geometry_response(text, tolerant=True)
        if tolerant_objects:
            logger.info(
                f"✅ Tolerant geometry parsing successful: {len(tolerant_objects)} objects"
            )
            return tolerant_objects
        logger.warning("❌ Tolerant geometry parsing failed or returned no objects")

    logger.info("📝 Attempting standard JSON parsing...")
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
                for key in GEOMETRY_TOKENS:
                    if key in obj and isinstance(obj[key], list):
                        try:
                            json_item[key] = [int(v) for v in obj[key]]
                        except Exception:
                            json_item[key] = obj[key]
                        break
                if any(k in json_item for k in GEOMETRY_TOKENS):
                    normalized_from_json.append(json_item)
                else:
                    logger.debug(f"   Skipping item without geometry: {obj}")
            logger.info(
                f"✅ Standard JSON parsing successful: {len(normalized_from_json)} objects"
            )
            return deduplicate_objects(normalized_from_json)
        else:
            logger.warning(f"❌ JSON parsed but not a list: {type(parsed_any)}")
    except Exception as exc:
        logger.warning(f"❌ JSON parsing failed: {exc}")
        logger.warning(
            f"   Text that failed JSON parsing: '{text[:500]}{'...' if len(text) > 500 else ''}'"
        )

    logger.info("🔄 Attempting fallback converter parsing...")
    try:
        json_text = engine._convert_prediction_text_to_vis_json(text)
        parsed_list = json.loads(json_text)
        if isinstance(parsed_list, list):
            logger.info(f"✅ Fallback converter successful: {len(parsed_list)} items")
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
                for key in GEOMETRY_TOKENS:
                    if key in obj and isinstance(obj[key], list):
                        merged[key] = obj[key]
                        break
                if any(k in merged for k in GEOMETRY_TOKENS):
                    unified.append(merged)
            if unified:
                logger.info(f"✅ Fallback parsing successful: {len(unified)} objects")
                return deduplicate_objects(unified)
            logger.warning("❌ Fallback parsing produced no valid objects")
        else:
            logger.warning(
                f"❌ Fallback converter returned non-list: {type(parsed_list)}"
            )
    except Exception as exc:
        logger.warning(f"❌ Fallback converter failed: {exc}")

    logger.warning("❌ All parsing methods failed - returning empty list")
    logger.warning(f"   Original text: '{text}'")
    return []


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
        generation_variant: str = "dense",
        use_global_teacher: bool = False,
        global_teacher_seed: Optional[int] = None,
        global_teacher_index: Optional[int] = None,
        global_teacher_file: Optional[str] = None,
        config_auto_loaded: bool = False,
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
            config_auto_loaded: Whether config was auto-loaded from checkpoint (internal flag)
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_torch_compile = use_torch_compile
        self.teacher_pool_file = teacher_pool_file
        # Force-disable teacher pairing for inference
        if num_teachers and int(num_teachers) > 0:
            logger.warning(
                f"Teacher pairing is disabled in inference; ignoring num_teachers={num_teachers}"
            )
        self.num_teachers = 0
        self.force_eager_attention = force_eager_attention
        # Optional data root for resolving relative image paths
        self.data_root = data_root
        # Variant: 'dense' (objects) or 'summary' (one-line CN)
        v = str(generation_variant).strip().lower()
        if v not in {"dense", "summary"}:
            raise ValueError(
                f"Unsupported generation_variant: {generation_variant!r}. Must be one of {{'dense', 'summary'}}."
            )
        self.generation_variant = v
        # Global teacher ablation configuration
        self.use_global_teacher = bool(use_global_teacher)
        self.global_teacher_seed = (
            int(global_teacher_seed) if global_teacher_seed is not None else None
        )
        self.global_teacher_index = (
            int(global_teacher_index) if global_teacher_index is not None else None
        )
        self.global_teacher_file = global_teacher_file
        self._global_teacher_sample: Optional[Dict[str, Any]] = None

        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)

        # CRITICAL FIX: Override config's model_path when auto-loaded from checkpoint
        # When config is auto-loaded from checkpoint directory, the config's model_path
        # field still points to the original base model, not the checkpoint itself.
        # Always prioritize CLI's model_path over config's model_path.
        if config_auto_loaded and model_path:
            logger.info(
                f"🔄 Auto-loaded config detected - overriding config's model_path"
            )
            logger.info(
                f"   Config's model_path: {getattr(self.config, 'model_path', 'N/A')}"
            )
            logger.info(f"   CLI's model_path: {model_path}")
            logger.info(f"   Using CLI's model_path for inference")
            self._model_path = model_path
        else:
            # Standard path resolution when config is explicitly provided
            self._model_path = model_path if model_path else self.config.model_path

        # Log final resolved model path
        logger.info(f"✅ Resolved model path: {self._model_path}")

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

        from src_new.processing.special_tokens import validate_geometry_tokens

        # Validate geometry tokens once - fail fast on invalid tokenizer
        validate_geometry_tokens(self.tokenizer)

        # Load teacher pool manager only when teacher guidance is requested
        self.teacher_pool_manager = None
        if self.num_teachers > 0:
            teacher_pool_path: Optional[str] = None

            # Resolve teacher pool path using DataResolver first (authoritative to data_root)
            if self.data_root is None:
                raise ValueError(
                    "data_root must be provided for teacher pool resolution when num_teachers>0"
                )
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
                        resolved_path = str(
                            path_manager.resolve_path(default_teacher_pool)
                        )
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
            else:
                raise RuntimeError(
                    f"Teacher pool file not found or unreadable: {candidate_path}"
                )

            # Use teacher guidance only when explicitly requested via num_teachers > 0
            # Always skip teacher guidance in inference
        logger.info(
            "ℹ️ Inference: teacher-student pairing is disabled; using single-turn prompts only"
        )

        logger.info("✅ InferenceEngine initialized successfully")

    def _load_model_and_processors(self):
        """Load model and processing components."""
        logger.info("Loading model and processors via shared HF loader...")

        # Enforce bf16-only policy in inference
        try:
            if not bool(getattr(self.config, "bf16", True)):
                raise ValueError(
                    "Inference requires bf16; set training.bf16: true in the config"
                )
            if bool(getattr(self.config, "fp16", False)):
                raise ValueError(
                    "Inference disallows fp16; set training.fp16: false and use bf16"
                )
            dtype_norm = str(getattr(self.config, "torch_dtype", "")).lower()
            if dtype_norm not in {"bfloat16", "bf16"}:
                raise ValueError("Inference requires model.torch_dtype=bfloat16")
        except Exception:
            raise

        if self.config.attn_implementation == "flash_attention_2":
            os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
            os.environ.setdefault("FLASH_ATTENTION_FORCE_CUDNN", "0")

        device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}

        max_pixels = getattr(self.config, "max_pixels", None)
        max_pixels_int = None
        if max_pixels is not None:
            try:
                max_pixels_int = int(max_pixels)
            except (TypeError, ValueError):
                max_pixels_int = None
            else:
                if max_pixels_int <= 0:
                    max_pixels_int = None

        attn_impl = getattr(self.config, "attn_implementation", None)
        if not attn_impl:
            attn_impl = "flash_attention_2"
        components = build_hf_components(
            model_path=self._model_path,
            model_config=self.config,
            attn_implementation=attn_impl,
            image_max_pixels=max_pixels_int,
            bf16=bool(getattr(self.config, "bf16", True)),
            force_eager_attention=bool(self.force_eager_attention),
            device_map=device_map,
        )

        self.tokenizer = components.tokenizer
        self.image_processor = components.image_processor
        self.processor = components.processor
        self.model = components.model

        param = next(self.model.parameters())
        logger.info(
            "Loaded tokenizer/model parity | device=%s | dtype=%s | bf16=%s",
            param.device,
            param.dtype,
            components.bf16_enabled,
        )

        current_vocab_size = len(self.tokenizer.get_vocab())
        model_vocab_size = self.model.get_input_embeddings().num_embeddings
        logger.info(
            "Tokenizer vocab=%d | Model vocab=%d",
            current_vocab_size,
            model_vocab_size,
        )

        from src_new.processing.special_tokens import validate_geometry_tokens

        validate_geometry_tokens(self.tokenizer)

        try:
            self.system_prompt = get_system_prompt(
                format_mode="special_tokens",
            )
            logger.info("✅ Cached system prompt from training templates for inference")
        except Exception as exc:
            self.system_prompt = None
            logger.warning(
                "Failed to build system prompt from training templates: %s", exc
            )

        self.conversation_processor = ConversationBuilder(
            processor=self.processor,
        )

        self.model.eval()

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
            teacher_pool_path = (
                str(self.teacher_pool_file)
                if self.teacher_pool_file is not None
                else None
            )
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

    def _select_global_teacher_sample(
        self,
        teacher_file: str,
    ) -> Tuple[Dict[str, Any], Optional[int]]:
        """Select and cache a single global teacher sample from a JSONL file.

        Returns the selected sample and its zero-based index within teacher_file (or None if unknown).
        Selection uses global_teacher_index if provided; otherwise uses global_teacher_seed for reproducibility.
        """
        if teacher_file is None or len(str(teacher_file).strip()) == 0:
            raise ValueError("Global teacher mode requires a valid teacher_file")

        # Determine total lines for random index when needed
        total_lines = 0
        with open(teacher_file, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        if total_lines == 0:
            raise ValueError(f"Teacher file is empty: {teacher_file}")

        # Choose index
        if self.global_teacher_index is not None:
            if (
                self.global_teacher_index < 0
                or self.global_teacher_index >= total_lines
            ):
                raise ValueError(
                    f"global_teacher_index out of range: {self.global_teacher_index} not in [0,{total_lines - 1}]"
                )
            chosen_index = int(self.global_teacher_index)
        else:
            rng = random.Random(self.global_teacher_seed)
            chosen_index = rng.randrange(total_lines)

        # Fetch the chosen sample
        selected_sample: Optional[Dict[str, Any]] = None
        with open(teacher_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == chosen_index:
                    try:
                        candidate = json.loads(line.strip())
                    except Exception as e:
                        raise ValueError(
                            f"Failed to parse teacher sample at line {idx}: {e}"
                        )
                    if not isinstance(candidate, dict):
                        raise ValueError("Teacher sample must be a JSON object")
                    # Basic validation; dense pipeline expects images and objects
                    if (
                        "images" not in candidate
                        or not isinstance(candidate["images"], list)
                        or len(candidate["images"]) == 0
                    ):
                        raise ValueError(
                            "Teacher sample missing required 'images' list"
                        )
                    # Do not strictly require 'objects' in summary mode
                    if self.generation_variant == "dense":
                        if "objects" not in candidate or not isinstance(
                            candidate["objects"], list
                        ):
                            raise ValueError(
                                "Teacher sample missing required 'objects' list for dense mode"
                            )
                    selected_sample = candidate
                    break

        if selected_sample is None:
            raise RuntimeError("Failed to retrieve selected global teacher sample")

        # Cache on self for reuse
        self._global_teacher_sample = selected_sample
        return selected_sample, chosen_index

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

        # If explicit teachers are provided on the sample, honor them (global-teacher mode)
        try:
            explicit_teachers = (
                sample.get("teacher_samples") if isinstance(sample, dict) else None
            )
            if isinstance(explicit_teachers, list) and len(explicit_teachers) > 0:
                return self._prepare_with_explicit_teachers(sample, explicit_teachers)
        except Exception as _e:
            logger.debug(f"Explicit teachers check failed/ignored: {_e}")

        # Use teacher-guided approach only when explicitly requested via teacher_pool_manager
        if self.teacher_pool_manager and self.num_teachers > 0:
            return self._prepare_training_matched_inputs(sample, seed)

        # Simple conversation path (default)
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
        input_ids_for_decode = (
            inputs["input_ids"][0]
            if inputs["input_ids"].dim() == 2
            else inputs["input_ids"]
        )
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

        # Build conversation based on generation variant
        try:
            if self.generation_variant == "summary":
                inputs = self.conversation_processor.create_summary_conversation_for_generation(
                    images=images
                )
            else:
                inputs = self.conversation_processor.create_simple_conversation_for_generation(
                    sample=sample, images=images
                )
        except Exception as e:
            raise RuntimeError(f"ConversationBuilder failed: {e}")

        # Validate conversation structure using the processed inputs
        input_ids_for_decode = (
            inputs["input_ids"][0]
            if inputs["input_ids"].dim() == 2
            else inputs["input_ids"]
        )
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
        allowed_input_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        }
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
        debug_mode = logger.isEnabledFor(logging.DEBUG)

        # CRITICAL VALIDATION: Cross-validate image tokens and tensors before generation
        if has_images and debug_mode:
            # Decode input_ids to check image token alignment
            ids_for_decode = (
                inputs["input_ids"][0]
                if inputs["input_ids"].dim() == 2
                else inputs["input_ids"]
            )
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

        # No decoding constraints in inference; model generation is unconstrained.

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
                # Use only <|im_end|> for EOS (training parity)
                eos_tokens = set()
                im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
                if im_end_id is None or im_end_id == -1:
                    raise ValueError("Tokenizer missing IM_END token id for generation")
                eos_tokens.add(int(im_end_id))

                logger.debug(
                    "📝 Generation mode: standard geometry tokens (coordinate tokens removed)"
                )

                # No decode-time token bans; allow full vocabulary per user request

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "repetition_penalty": repetition_penalty,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": list(eos_tokens),
                    "use_cache": True,
                }
                safe_inputs = self._filter_generate_inputs(inputs)
                output_ids = self.model.generate(**safe_inputs, **gen_kwargs)

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
            # Compute input length robustly for 1D/2D input_ids
            if (
                "input_ids" in safe_inputs
                and hasattr(safe_inputs["input_ids"], "dim")
                and safe_inputs["input_ids"].dim() == 2
            ):
                input_len = int(safe_inputs["input_ids"].shape[1])
            else:
                input_len = int(inputs["input_ids"].shape[-1])
            generated_text = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=False
            )
        except Exception as e:
            logger.error(f"❌ DECODING ERROR: {e}")
            logger.error(f"   Output IDs shape: {output_ids.shape}")
            raise RuntimeError(f"Failed to decode generated response: {e}")

        # Log generation details
        input_length = int(inputs["input_ids"].shape[-1])
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
        # Trim only at the first IM_END (training parity)
        idx = assistant_part.find(IM_END)
        if idx != -1:
            assistant_part = assistant_part[:idx]

        # Clean up container tokens; keep geometry/coord tokens for parsing
        for token in [IM_END, IM_START]:
            assistant_part = assistant_part.replace(token, "")
        cleaned_response = assistant_part.strip()

        # Expose raw/clean text for downstream result dumping
        self._last_raw_generated_text = generated_text
        self._last_clean_generated_text = cleaned_response

        # Log the cleaned response for debugging
        logger.info(f"🧹 CLEANED RESPONSE FOR PARSING:")
        logger.info(f"   Length: {len(cleaned_response)} characters")
        logger.info(f"   Content: '{cleaned_response}'")
        # Normalize to visualization object list
        objects_list: List[Dict[str, Any]] = self._normalize_prediction_to_vis_objects(
            self, cleaned_response
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

    def generate_summary_text(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float = 0.000001,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate one-line Chinese summary text (no geometry/special tokens)."""
        # Reuse generation pipeline but return cleaned string
        try:
            with torch.no_grad():
                # EOS: <|im_end|> only
                im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
                if im_end_id is None or im_end_id == -1:
                    raise ValueError("Tokenizer missing IM_END token id for generation")
                eos_tokens = [int(im_end_id)]
                # No decode-time token bans in summary mode per user request
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "repetition_penalty": repetition_penalty,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": eos_tokens,
                    "use_cache": True,
                }
                safe_inputs = self._filter_generate_inputs(inputs)
                # Ensure batch dimension for text tensors
                if (
                    "input_ids" in safe_inputs
                    and hasattr(safe_inputs["input_ids"], "dim")
                    and safe_inputs["input_ids"].dim() == 1
                ):
                    safe_inputs["input_ids"] = safe_inputs["input_ids"].unsqueeze(0)
                if (
                    "attention_mask" in safe_inputs
                    and hasattr(safe_inputs["attention_mask"], "dim")
                    and safe_inputs["attention_mask"].dim() == 1
                ):
                    safe_inputs["attention_mask"] = safe_inputs[
                        "attention_mask"
                    ].unsqueeze(0)
                # Compute input length from batched input_ids
                if "input_ids" in safe_inputs and hasattr(
                    safe_inputs["input_ids"], "shape"
                ):
                    input_len = int(safe_inputs["input_ids"].shape[-1])
                else:
                    input_len = int(inputs["input_ids"].shape[-1])
                output_ids = self.model.generate(**safe_inputs, **gen_kwargs)
                # Normalize output shape to [1, seq]
                if hasattr(output_ids, "dim") and output_ids.dim() == 1:
                    output_ids = output_ids.unsqueeze(0)
                # Decode only newly generated tokens
                generated_text = self.tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=False
                )
                # Trim at IM_END and strip container tokens
                idx = generated_text.find(IM_END)
                if idx != -1:
                    generated_text = generated_text[:idx]
                for token in [IM_END, IM_START, END_OF_TEXT]:
                    generated_text = generated_text.replace(token, "")
                return generated_text.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""

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
        """Convert model response text to a JSON string for visualization format."""
        parsed = self._try_parse_json_list(response)
        if isinstance(parsed, list):
            normalized: List[Dict[str, Any]] = []
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("desc") or obj.get("label")
                if label is None:
                    continue
                normalized_obj: Dict[str, Any] = {"desc": label}
                for key in GEOMETRY_TOKENS:
                    if key in obj and isinstance(obj[key], list):
                        try:
                            normalized_obj[key] = [int(c) for c in obj[key]]
                        except Exception:
                            normalized_obj[key] = obj[key]
                        break
                if any(k in normalized_obj for k in GEOMETRY_TOKENS):
                    normalized.append(normalized_obj)
            if normalized:
                vis_objects = convert_objects_to_vis(normalized)
                try:
                    return json.dumps(vis_objects, ensure_ascii=False)
                except Exception:
                    return response
        return convert_response_to_vis_json(
            response,
            coordinate_mode=False,
            tolerant_geometry=True,
        )

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

        # Global teacher ablation mode: select and cache a single teacher
        global_teacher: Optional[Dict[str, Any]] = None
        global_teacher_index: Optional[int] = None
        if getattr(self, "use_global_teacher", False):
            teacher_source = self.global_teacher_file or input_file
            try:
                global_teacher, global_teacher_index = (
                    self._select_global_teacher_sample(teacher_source)
                )
                logger.info(
                    f"🧪 Global-teacher ablation enabled: selected index {global_teacher_index} from '{teacher_source}'"
                )
                # Optional: summarize teacher for logs
                try:
                    num_imgs = (
                        len(global_teacher.get("images", []))
                        if isinstance(global_teacher.get("images"), list)
                        else 0
                    )
                    logger.info(
                        f"   Teacher has {num_imgs} image(s); fields: {list(global_teacher.keys())}"
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to select global teacher sample: {e}")
                raise

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
                    # Skip the chosen teacher sample in global-teacher mode to avoid self-pairing
                    if (
                        global_teacher is not None
                        and global_teacher_index is not None
                        and i == global_teacher_index
                    ):
                        logger.debug(
                            f"Skipping teacher sample at index {i} during student loop (global-teacher mode)"
                        )
                        continue

                    # Inject teacher into sample if enabled
                    if global_teacher is not None:
                        # Ensure immutability of cached teacher
                        import copy

                        teacher_copy = copy.deepcopy(global_teacher)
                        # Attach explicit teacher for exact reproduction path
                        sample["teacher_samples"] = [teacher_copy]

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

            if self.generation_variant == "dense":
                if "objects" not in sample or not isinstance(sample["objects"], list):
                    raise ValueError(
                        "Sample missing required 'objects' field or objects is not a list"
                    )

            # Use the generation pipeline
            inputs = self.prepare_inference_inputs(
                sample=sample,
                seed=42,  # Use fixed seed for reproducible results
                data_root=self.data_root,
            )

            if self.generation_variant == "summary":
                summary_text = self.generate_summary_text(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                )
                result = {
                    "sample_id": sample.get("id", "unknown"),
                    "image": sample["images"][0]
                    if isinstance(sample.get("images"), list)
                    and len(sample["images"]) > 0
                    else None,
                    "images": sample["images"],
                    "summary_text": summary_text,
                    "ground_truth": sample.get("summary", None),
                    "width": sample.get("width"),
                    "height": sample.get("height"),
                }
                return result

            # Dense objects path
            prediction_objects = self.generate_response(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )

            result = {
                "sample_id": sample.get("id", "unknown"),
                "image": sample["images"][0]
                if isinstance(sample.get("images"), list) and len(sample["images"]) > 0
                else None,
                "images": sample["images"],
                "prediction": prediction_objects,
                "prediction_text_raw": getattr(self, "_last_raw_generated_text", None),
                "ground_truth": sample.get("objects", []),
                "width": sample.get("width"),
                "height": sample.get("height"),
            }

            return result

        except Exception as e:
            logger.error(f"Error in _process_single_sample: {e}")
            # In summary mode, return summary-shaped error to avoid downstream parsing issues
            if getattr(self, "generation_variant", "dense") == "summary":
                return {
                    "sample_id": sample.get("id", "unknown"),
                    "images": sample.get("images", []),
                    "summary_text": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("summary", None),
                    "width": sample.get("width"),
                    "height": sample.get("height"),
                }
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

            # Use training-consistent prompts: system from get_system_prompt, user from CONSTANTS
            user_prompt = CONSTANTS.get("BASE_USER_PROMPT", "")
            sys_prompt = self.system_prompt or get_system_prompt()

            # Build messages exactly like ConversationProcessor.create_inference_conversation
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image"},
                    ],
                },
            ]

            # Apply chat template with images param for proper <|image_pad|> handling
            text = self.processor.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                images=[image],
            )

            # Process inputs via the unified HF processor
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

    _normalize_prediction_to_vis_objects = staticmethod(
        _normalize_prediction_to_vis_objects_impl
    )

    def _filter_generate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only keys accepted by generate/forward and move tensors to device.

        Allowed keys for Qwen2.5-VL generate: input_ids, attention_mask, pixel_values, image_grid_thw
        """
        allowed_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        filtered: Dict[str, Any] = {}
        for key, value in inputs.items():
            if key in allowed_keys:
                if torch.is_tensor(value):
                    filtered[key] = value.to(self.model.device)
                else:
                    filtered[key] = value
        return filtered


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

    # Global teacher ablation (single teacher used for all students)
    parser.add_argument(
        "--use_global_teacher",
        action="store_true",
        default=False,
        help="Enable global-teacher ablation: pick one sample as teacher and inject into all conversations",
    )
    parser.add_argument(
        "--global_teacher_seed",
        type=int,
        default=None,
        help="Random seed for selecting the global teacher sample (ignored if index is provided)",
    )
    parser.add_argument(
        "--global_teacher_index",
        type=int,
        default=None,
        help="Zero-based index of teacher sample to use from the teacher JSONL (overrides seed)",
    )
    parser.add_argument(
        "--global_teacher_file",
        type=str,
        default=None,
        help="Optional JSONL file to choose the global teacher from (defaults to input_file)",
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

    parser.add_argument(
        "--generation_variant",
        type=str,
        choices=["dense", "summary"],
        default="dense",
        help="Generation variant: 'dense' for objects + geometry, 'summary' for one-line Chinese summary",
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
    config_auto_loaded = False
    if args.config_path is None or len(str(args.config_path).strip()) == 0:
        config_auto_loaded = True
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
        logger.info(
            f"   ⚠️  Note: CLI's --model_path will override config's model_path field"
        )

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
        generation_variant=args.generation_variant,
        use_global_teacher=args.use_global_teacher,
        global_teacher_seed=args.global_teacher_seed,
        global_teacher_index=args.global_teacher_index,
        global_teacher_file=args.global_teacher_file,
        config_auto_loaded=config_auto_loaded,
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

# Backward compatibility alias
InferenceEngine._normalize_prediction_to_vis_objects = staticmethod(
    _normalize_prediction_to_vis_objects_impl
)
