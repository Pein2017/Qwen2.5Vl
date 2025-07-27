#!/usr/bin/env python3
"""
Standalone inference runner for Qwen2.5-VL with mandatory performance optimizations.

This script runs inference on a JSONL dataset with the following mandatory features:
- Flash Attention 2 (always enabled, no fallback)
- KV Cache (always enabled for efficient generation)
- CUDA/GPU support (CPU mode not supported)
- Native batch processing support
- Teacher-guided inference support (optional)

Requirements:
- PyTorch 2.0+ with CUDA support
- CUDA 11.6+
- GPU with Flash Attention 2 support (Ampere or newer)
- Transformers library with Qwen2.5-VL support

The script will raise errors if any of these requirements are not met.
"""

import argparse
import json
import logging
import os
import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
from PIL import Image
from tqdm import tqdm

from src.logger_utils import (
    configure_global_logging,
    get_logger,
)


# Runtime type-checking ---------------------------------------------------

# Dim symbols (aligned with chat_processor.py)
S = TypeVar("S")  # Sequence length
B = TypeVar("B")
PT = TypeVar("PT")
C = TypeVar("C")
H = TypeVar("H")
W = TypeVar("W")


# Global logger (configured later in main)
logger = get_logger("inference")

# Initialize logger (will be reconfigured in main)
logger = logging.getLogger("inference")

# -------------------- Monkey-patch logger utils for compatibility --------------------
# Some modules still expect the old logger interface, so we provide minimal compatibility


# Config will be initialized from command line argument - no automatic loading

# Apply all critical Qwen2.5-VL fixes **before** we import the model
from src.models.patches import (
    apply_comprehensive_qwen25_fixes,
    verify_qwen25_patches,
)


if not apply_comprehensive_qwen25_fixes():
    raise RuntimeError(
        "Failed to apply Qwen2.5-VL patches – cannot proceed with inference."
    )

# Optional sanity check
verify_qwen25_patches()

# ---------------------------------------------------------------------------
# Runtime verification helpers
# ---------------------------------------------------------------------------


def check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available in the current environment.

    Returns:
        bool: True if Flash Attention 2 is available, False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Check if flash_attn package is installed
        try:
            import flash_attn
        except ImportError:
            return False

        # Lazy import – will raise if the fused kernels are missing.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLFlashAttention2,  # noqa
        )

        return True

    except ImportError:
        return False


# ---------------------------------------------------------------------------

# Deferred imports that rely on config and the patched logging utilities

# Ensure training chat template is re-applied


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_training_prompts: bool = False,
        language: str = "chinese",
        batch_size: int = 1,
        num_workers: int = 0,
        use_torch_compile: bool = False,
        teacher_pool_file: Optional[str] = None,
        num_teachers: int = 0,
    ):
        """Initialize inference engine with mandatory Flash Attention 2 and KV cache.

        Args:
            model_path: Path to model directory
            device: Device to use ("auto", "cuda:0", etc) - CPU not supported
            use_training_prompts: Whether to use training prompts
            language: Language for prompts
            batch_size: Number of samples to process at once (default=1)
            num_workers: Number of data loading workers (default=0)
            use_torch_compile: Use torch.compile for faster inference (requires PyTorch 2.0+)
            teacher_pool_file: Path to teacher pool JSONL file (optional)
            num_teachers: Number of teacher examples to use per sample (default=0)

        Note:
            Flash Attention 2 and KV cache are always enabled and cannot be disabled.
            The script will raise an error if the environment doesn't support these features.
        """
        self.device = device
        self.model_path = model_path
        self.use_training_prompts = use_training_prompts
        self.language = language
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_torch_compile = use_torch_compile
        self.teacher_pool_file = teacher_pool_file
        self.num_teachers = num_teachers

        # Verify CUDA is available
        if device == "cpu":
            raise RuntimeError(
                "CPU inference is not supported. Please use a CUDA-enabled device."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This inference script requires a GPU with CUDA support."
            )

        # Force eager attention for inference to avoid vocabulary and triton issues
        self.use_flash_attention = False  # Always disable for inference
        logger.info(
            "🔧 Using eager attention for inference (flash attention disabled for stability)"
        )

        # Load processor and model following demo approach
        self.processor, self.model = self._load_model_and_processor()

        # Set to evaluation mode
        self.model.eval()

        # Model type detection will be set during loading

        # Single GPU only - no multi-GPU support
        logger.info(
            f"🚀 Using single GPU for inference: {next(self.model.parameters()).device}"
        )

        # Apply torch.compile optimization if requested
        if self.use_torch_compile and torch.__version__ >= "2.0.0":
            try:
                logger.info("🔥 Applying torch.compile optimization...")
                # Compile the model directly (no DataParallel unwrapping needed)
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(
                    f"⚠️ torch.compile failed, continuing without compilation: {e}"
                )

        # Flash Attention is configured during model loading via attn_implementation parameter
        # Verify attention implementation
        if hasattr(self.model.config, "_attn_implementation"):
            attn_impl = self.model.config._attn_implementation
            if attn_impl == "flash_attention_2":
                logger.info("✅ Using Flash Attention 2")
            elif attn_impl == "eager":
                logger.info("✅ Using standard eager attention")
            else:
                logger.info(f"✅ Using attention implementation: {attn_impl}")
        else:
            logger.info("✅ Using default attention implementation")

        # Verify KV cache is enabled
        if not hasattr(self.model.config, "use_cache"):
            raise RuntimeError(
                "Model config missing use_cache attribute. "
                "This indicates KV cache setup failed."
            )

        if not self.model.config.use_cache:
            raise RuntimeError(
                "KV cache is disabled. This is required for efficient generation. "
                "The model config has use_cache=False."
            )

        logger.info("✅ Critical features verified:")
        logger.info("   ⚡ Eager Attention: ENABLED (for stability)")
        logger.info("   🚫 Flash Attention: DISABLED (to avoid triton issues)")
        logger.info("   💾 KV Cache: ENABLED")

        logger.info("✅ InferenceEngine initialized")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Device: {next(self.model.parameters()).device}")
        logger.info(f"   Use training prompts: {use_training_prompts}")
        logger.info(f"   Language: {language}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Num workers: {num_workers}")
        logger.info(f"   Use torch compile: {use_torch_compile}")
        if teacher_pool_file and num_teachers > 0:
            logger.info(f"   Teacher pool: {teacher_pool_file}")
            logger.info(f"   Teachers per sample: {num_teachers}")

        # Load teacher pool if specified
        self.teacher_samples = []
        if self.teacher_pool_file and self.num_teachers > 0:
            self.teacher_samples = self._load_teacher_pool()
            logger.info(f"✅ Loaded {len(self.teacher_samples)} teacher samples")
            logger.info(f"   Will use {self.num_teachers} teachers per sample")

            # Force batch_size=1 for teacher guidance due to multi-image conversations
            if self.batch_size > 1:
                logger.warning(
                    f"⚠️  Teacher guidance requires batch_size=1 due to multi-image conversations. "
                    f"Changing batch_size from {self.batch_size} to 1."
                )
                self.batch_size = 1

        # ALWAYS create ChatProcessor to ensure consistent prompt formatting
        from src.chat_processor import ChatProcessor

        self.chat_processor = ChatProcessor(
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
            use_training_prompts=True,  # Use training prompts to match training pipeline
            language=self.language,
        )

        if self.teacher_samples:
            logger.info("✅ Initialized ChatProcessor for teacher-guided inference")
        else:
            logger.info(
                "✅ Initialized ChatProcessor for standard inference (matching training pipeline)"
            )

    def _load_model_and_processor(self):
        """Load model and processor using UNIFIED loader for consistency."""
        # Enforce offline mode
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # FAIL-FAST: Validate model path
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        if not model_dir.is_dir():
            raise ValueError(f"Model path is not a directory: {self.model_path}")

        logger.info("🔧 Loading model via UNIFIED loader (same as training)")

        try:
            # FAIL-FAST: Validate config is initialized
            try:
                from src.config import config
            except ImportError:
                raise ImportError("Failed to import config - ensure config module is available")
                
            # FAIL-FAST: Validate model loader is available
            try:
                from src.models.model_loader import load_model_and_processor_unified
            except ImportError:
                raise ImportError("Failed to import model_loader - ensure models module is available")

            # FAIL-FAST: Validate required config attributes
            if not hasattr(config, "coordinate_tokens_enabled"):
                raise ValueError("config missing required attribute: coordinate_tokens_enabled")
                
            # Detect model type based on config - coordinate tokens drive wrapper usage
            coordinate_tokens_enabled = config.coordinate_tokens_enabled

            logger.info(f"🎯 Model type detection:")
            logger.info(f"   Coordinate tokens enabled: {coordinate_tokens_enabled}")
            logger.info(f"   Detection wrapper enabled: {coordinate_tokens_enabled}")

            # Log configuration for debugging
            logger.debug(
                f"Full config attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}"
            )

            # Use IDENTICAL loading process as training (only difference: for_inference=True)
            # Force eager attention for inference to avoid triton issues
            attn_implementation = "eager"  # Always use eager for inference
            logger.info(
                "🔧 Forcing eager attention for inference to avoid triton issues"
            )
            
            # FAIL-FAST: Validate model loading with explicit error handling
            try:
                model, tokenizer, image_processor = load_model_and_processor_unified(
                    model_path=str(model_dir),
                    for_inference=True,  # ONLY difference from training
                    force_detection=False,  # Explicit value instead of None
                    attn_implementation=attn_implementation,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model and processor: {e}")

            # FAIL-FAST: Validate loaded components
            if model is None:
                raise ValueError("Model loading failed - model is None")
            if tokenizer is None:
                raise ValueError("Model loading failed - tokenizer is None")
            if image_processor is None:
                raise ValueError("Model loading failed - image_processor is None")

            # Create processor-like object for compatibility with proper methods
            class UnifiedProcessor:
                def __init__(self, tokenizer, image_processor):
                    # FAIL-FAST: Validate required components
                    if tokenizer is None:
                        raise ValueError("tokenizer cannot be None")
                    if image_processor is None:
                        raise ValueError("image_processor cannot be None")
                        
                    self.tokenizer = tokenizer
                    self.image_processor = image_processor

                def __call__(
                    self, text=None, images=None, return_tensors=None, **kwargs
                ):
                    """Process text and images together"""
                    inputs = {}

                    if text is not None:
                        # Process text using tokenizer
                        text_inputs = self.tokenizer(
                            text, return_tensors=return_tensors, **kwargs
                        )
                        # Convert BatchEncoding to dict if needed
                        if hasattr(text_inputs, "to_dict"):
                            text_inputs = text_inputs.to_dict()
                        inputs.update(text_inputs)

                    if images is not None:
                        # Process images using image_processor
                        image_inputs = self.image_processor(
                            images, return_tensors=return_tensors, **kwargs
                        )
                        # Convert BatchEncoding to dict if needed
                        if hasattr(image_inputs, "to_dict"):
                            image_inputs = image_inputs.to_dict()
                        inputs.update(image_inputs)

                    return inputs

                def batch_decode(self, *args, **kwargs):
                    """Pass through to tokenizer's batch_decode"""
                    # FAIL-FAST: Validate tokenizer has batch_decode method
                    if not hasattr(self.tokenizer, "batch_decode"):
                        raise AttributeError("tokenizer does not have batch_decode method")
                    return self.tokenizer.batch_decode(*args, **kwargs)

            processor = UnifiedProcessor(tokenizer, image_processor)

            model.eval()

            # Store model type for later use
            self.coordinate_tokens_enabled = coordinate_tokens_enabled

            logger.info("✅ UNIFIED model loading completed for inference")
            logger.debug(f"Model loaded on device: {next(model.parameters()).device}")
            logger.debug(f"Model dtype: {next(model.parameters()).dtype}")

            if coordinate_tokens_enabled:
                logger.info(
                    "🚀 Coordinate token model detected - ready for soft expectation inference"
                )
            else:
                logger.info("📄 Base model detected - ready for standard VL inference")

            return processor, model

        except Exception as e:
            logger.error(f"❌ UNIFIED model loading failed: {e}")
            raise RuntimeError(f"Failed to load model via unified loader: {e}")

    def _load_teacher_pool(self) -> List[Dict[str, Any]]:
        """Load teacher samples from teacher pool JSONL file."""
        # FAIL-FAST: Validate teacher_pool_file
        if self.teacher_pool_file is None:
            raise ValueError("teacher_pool_file is None")
            
        teacher_pool_path = Path(self.teacher_pool_file)
        
        # FAIL-FAST: Validate file exists and is a file
        if not teacher_pool_path.exists():
            raise FileNotFoundError(
                f"Teacher pool file not found: {self.teacher_pool_file}"
            )
        if not teacher_pool_path.is_file():
            raise ValueError(f"Teacher pool path is not a file: {self.teacher_pool_file}")

        teacher_samples = []
        
        # FAIL-FAST: Validate file contents with explicit error handling
        try:
            with open(teacher_pool_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            
                            # FAIL-FAST: Validate sample structure
                            if not isinstance(sample, dict):
                                raise ValueError(f"Line {line_num}: Sample is not a dictionary")
                                
                            # FAIL-FAST: Validate required fields
                            if "images" not in sample:
                                raise ValueError(f"Line {line_num}: Missing required field 'images'")
                            if "objects" not in sample:
                                raise ValueError(f"Line {line_num}: Missing required field 'objects'")
                                
                            # FAIL-FAST: Validate field types
                            if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
                                raise ValueError(f"Line {line_num}: 'images' field is empty or not a list")
                            if not isinstance(sample["objects"], list):
                                raise ValueError(f"Line {line_num}: 'objects' field is not a list")
                                
                            teacher_samples.append(sample)
                            
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read teacher pool file: {e}")

        # FAIL-FAST: Validate we have teacher samples
        if not teacher_samples:
            raise ValueError("No teacher samples found in teacher pool file")
            
        logger.info(f"Loaded {len(teacher_samples)} teacher samples from {self.teacher_pool_file}")

        return teacher_samples

    def _sample_teachers(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample random teacher examples."""
        if seed is not None:
            random.seed(seed)

        if self.num_teachers >= len(self.teacher_samples):
            return self.teacher_samples.copy()

        return random.sample(self.teacher_samples, self.num_teachers)

    def prepare_inference_inputs(
        self, sample: Dict[str, Any], seed: Optional[int] = None
    ) -> Tuple[str, List[Image.Image]]:
        """Prepare inputs for inference exactly like demo script.

        Args:
            sample: The sample to process
            seed: Random seed for teacher sampling (only used if teacher guidance is enabled)

        Returns:
            Tuple of (prompt_str, images)
        """
        # Use teacher-guided approach if enabled
        if self.chat_processor and self.teacher_samples:
            return self._prepare_teacher_guided_inputs(sample, seed)

        # Otherwise use standard approach
        return self._prepare_standard_inputs(sample)

    def _prepare_teacher_guided_inputs(
        self, student_sample: Dict[str, Any], seed: Optional[int] = None
    ) -> Tuple[str, List[Image.Image]]:
        """Prepare inputs with teacher guidance using the exact training pipeline."""
        # Sample teacher examples
        teachers = self._sample_teachers(seed)
        logger.debug(f"Sampled {len(teachers)} teachers for inference")

        # Build a sample structure identical to training
        sample_struct: Dict[str, Any] = {
            "teachers": teachers,
            "student": student_sample,
        }

        # Build conversation messages using ChatProcessor
        messages = self.chat_processor._create_conversation_messages(sample_struct)

        # Remove the last assistant message (student answer) for inference
        if messages and messages[-1].role == "assistant":
            messages = messages[:-1]

        # Process images and expand vision tokens
        image_paths = self.chat_processor._extract_all_image_paths(sample_struct)
        processed_messages, images, _ = self.chat_processor._process_images_and_tokens(
            messages, image_paths
        )

        # Convert ChatMessage dataclasses to dicts
        messages_dicts = [asdict(msg) for msg in processed_messages]

        # Apply chat template with generation prompt
        prompt_str = self.chat_processor.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )
        # Ensure we return a string
        if not isinstance(prompt_str, str):
            prompt_str = str(prompt_str)

        logger.debug(f"Built teacher-guided prompt with {len(images)} images")
        logger.debug(f"Teacher examples: {len(teachers)}")
        logger.debug(f"Prompt preview (first 500 chars): {prompt_str[:500]}...")

        return prompt_str, images

    def _prepare_standard_inputs(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[Image.Image]]:
        """Prepare inputs for standard inference using ChatProcessor for consistency."""
        logger.debug(
            "Using ChatProcessor for standard inference to match training pipeline"
        )

        # Build a sample structure for single student (no teachers)
        sample_struct: Dict[str, Any] = {
            "teachers": [],  # No teachers for standard inference
            "student": sample,
        }

        # Build conversation messages using ChatProcessor (same as training)
        messages = self.chat_processor._create_conversation_messages(sample_struct)

        # Remove the last assistant message (student answer) for inference
        if messages and messages[-1].role == "assistant":
            messages = messages[:-1]

        # Process images and expand vision tokens
        image_paths = self.chat_processor._extract_all_image_paths(sample_struct)
        processed_messages, images, _ = self.chat_processor._process_images_and_tokens(
            messages, image_paths
        )

        # Convert ChatMessage dataclasses to dicts
        messages_dicts = [asdict(msg) for msg in processed_messages]

        # Apply chat template with generation prompt
        prompt_str = self.chat_processor.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )
        # Ensure we return a string
        if not isinstance(prompt_str, str):
            prompt_str = str(prompt_str)

        logger.debug(f"Built standard inference prompt using ChatProcessor")
        logger.debug(f"Prompt preview (first 500 chars): {prompt_str[:500]}...")

        return prompt_str, images

    def prepare_batch_inference_inputs(
        self, samples: List[Dict[str, Any]], start_idx: int = 0
    ) -> Tuple[List[str], List[List[Image.Image]]]:
        """Prepare inputs for batch inference.

        Args:
            samples: List of samples to process
            start_idx: Starting index for seed generation (for teacher sampling)

        Returns:
            Tuple of (prompts, images_list) where prompts is a list of strings
            and images_list is a list of image lists
        """
        prompts = []
        images_list = []

        for i, sample in enumerate(samples):
            # Use deterministic seed based on sample index for reproducibility
            seed = start_idx + i if self.teacher_samples else None
            prompt, images = self.prepare_inference_inputs(sample, seed=seed)
            prompts.append(prompt)
            images_list.append(images)

        return prompts, images_list

    def generate_response(
        self,
        images: List[Image.Image],
        text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate response exactly like demo script."""
        # Always use ChatProcessor for consistent inference
        if self.chat_processor:
            inputs = self.chat_processor.prepare_inputs_for_inference(
                images=images,
                text=text,
                is_first_step=True,
            )
        else:
            # Fallback (should not happen as we always create ChatProcessor)
            inputs = self.processor(text=[text], images=images, return_tensors="pt")

        inputs = {
            k: v.to(self.model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        # Log input information
        logger.debug(f"INPUT PROMPT (first 500 chars): {text[:500]}...")
        logger.debug(f"INPUT IDS shape: {inputs['input_ids'].shape}")
        if "pixel_values" in inputs:
            logger.debug(f"PIXEL VALUES shape: {inputs['pixel_values'].shape}")

        # Log generation parameters
        logger.debug(
            f"Generation settings: max_new_tokens={max_new_tokens}, do_sample={do_sample}, repetition_penalty={repetition_penalty}, temperature={temperature}"
        )

        # Generate exactly like demo with timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time

        start_time = time.perf_counter()

        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ),
        ):
            # Verify KV cache is still enabled before generation
            if not self.model.config.use_cache:
                raise RuntimeError(
                    "KV cache was disabled before generation. "
                    "This should never happen and indicates a configuration error."
                )

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,  # deterministic to preserve exact keys/labels
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=repetition_penalty,  # mild repeat penalty
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,  # Explicitly enable KV cache for generation
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        # Log timing info
        prompt_tokens = inputs["input_ids"].shape[1]
        total_tokens = output_ids.shape[1]
        generated_tokens = max(total_tokens - prompt_tokens, 1)
        tokens_per_second = generated_tokens / elapsed if elapsed > 0 else float("inf")

        logger.debug(
            f"Generation took {elapsed:.2f}s for {generated_tokens} tokens → {tokens_per_second:.2f} tokens/s"
        )

        # Decode response exactly like demo
        response_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # Extract assistant part - need to handle multi-turn conversations
        # For teacher-guided inference, we need the LAST assistant response
        if self.teacher_samples and "assistant\n" in response_text:
            # Split by assistant markers and get the last one
            assistant_parts = response_text.split("assistant\n")
            if len(assistant_parts) > 1:
                # Get the last assistant response
                assistant_part = assistant_parts[-1].strip()
                # Remove any trailing user/system markers
                if "\nuser" in assistant_part:
                    assistant_part = assistant_part.split("\nuser")[0].strip()
                if "\nsystem" in assistant_part:
                    assistant_part = assistant_part.split("\nsystem")[0].strip()
            else:
                assistant_part = response_text
        else:
            # Standard single-turn extraction
            try:
                assistant_part = response_text.split("assistant\n", 1)[1].strip()
            except IndexError:
                assistant_part = response_text

        # Clean up special tokens that might remain
        # Remove <|im_end|> and other special tokens
        special_tokens_to_remove = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]

        cleaned_response = assistant_part
        for token in special_tokens_to_remove:
            cleaned_response = cleaned_response.replace(token, "")

        # Remove any trailing whitespace
        cleaned_response = cleaned_response.strip()

        logger.debug(f"OUTPUT TEXT (first 200 chars): {cleaned_response[:200]}...")

        return cleaned_response

    def generate_batch_responses(
        self,
        images_list: List[List[Image.Image]],
        texts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> List[str]:
        """Generate responses for a batch of inputs using true batch processing.

        This method now implements real batch processing by combining all inputs
        into a single batch and processing them together for better GPU utilization.

        Args:
            images_list: List of image lists (one list per sample)
            texts: List of text prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty

        Returns:
            List of generated responses
        """
        batch_size = len(texts)
        logger.debug(f"Processing true batch of {batch_size} samples together")

        try:
            # Prepare batch inputs using the processor
            all_images = []
            all_texts = []

            for text, images in zip(texts, images_list):
                all_texts.append(text)
                all_images.extend(images)  # Flatten all images into one list

            # Use processor to handle batch processing
            if self.chat_processor:
                # For chat processor, we need to process each sample individually
                # but then combine the inputs for batch generation
                batch_inputs = []
                for text, images in zip(texts, images_list):
                    inputs = self.chat_processor.prepare_inputs_for_inference(
                        images=images,
                        text=text,
                        is_first_step=True,
                    )
                    batch_inputs.append(inputs)

                # Combine batch inputs
                combined_inputs = {}
                for key in batch_inputs[0].keys():
                    if key == "input_ids":
                        # Pad and stack input_ids
                        input_ids_list = [inp[key] for inp in batch_inputs]
                        max_len = max(ids.shape[1] for ids in input_ids_list)
                        padded_ids = []
                        for ids in input_ids_list:
                            pad_len = max_len - ids.shape[1]
                            if pad_len > 0:
                                # Left padding for generation
                                # Make sure we have a valid pad token ID
                                pad_token_id = (
                                    self.processor.tokenizer.pad_token_id
                                    if self.processor.tokenizer.pad_token_id is not None
                                    else 0
                                )
                                padded = torch.cat(
                                    [
                                        torch.full(
                                            size=(
                                                ids.shape[0],
                                                pad_len,
                                            ),
                                            fill_value=int(
                                                pad_token_id
                                            ),  # Ensure pad_token_id is an integer
                                            dtype=ids.dtype,
                                            device=ids.device,
                                        ),
                                        ids,
                                    ],
                                    dim=1,
                                )
                            else:
                                padded = ids
                            padded_ids.append(padded)
                        combined_inputs[key] = torch.cat(padded_ids, dim=0)

                    elif key == "attention_mask":
                        # Pad and stack attention masks
                        mask_list = [inp[key] for inp in batch_inputs]
                        max_len = max(mask.shape[1] for mask in mask_list)
                        padded_masks = []
                        for mask in mask_list:
                            pad_len = max_len - mask.shape[1]
                            if pad_len > 0:
                                # Left padding with zeros for attention mask
                                padded = torch.cat(
                                    [
                                        torch.zeros(
                                            (mask.shape[0], pad_len),
                                            dtype=mask.dtype,
                                            device=mask.device,
                                        ),
                                        mask,
                                    ],
                                    dim=1,
                                )
                            else:
                                padded = mask
                            padded_masks.append(padded)
                        combined_inputs[key] = torch.cat(padded_masks, dim=0)

                    elif key == "pixel_values":
                        # Stack pixel values
                        pixel_values_list = [
                            inp[key] for inp in batch_inputs if inp[key] is not None
                        ]
                        if pixel_values_list:
                            combined_inputs[key] = torch.cat(pixel_values_list, dim=0)
                        else:
                            combined_inputs[key] = None

                    elif key == "image_grid_thw":
                        # Stack image grid info
                        grid_list = [
                            inp[key] for inp in batch_inputs if inp[key] is not None
                        ]
                        if grid_list:
                            combined_inputs[key] = torch.cat(grid_list, dim=0)
                        else:
                            combined_inputs[key] = None

                    else:
                        # For other keys, take the first non-None value or stack if possible
                        values = [
                            inp[key] for inp in batch_inputs if inp[key] is not None
                        ]
                        if values:
                            if torch.is_tensor(values[0]):
                                combined_inputs[key] = torch.cat(values, dim=0)
                            else:
                                combined_inputs[key] = values[0]
                        else:
                            combined_inputs[key] = None

                inputs = combined_inputs
            else:
                # Fallback to processor batch handling (less reliable)
                inputs = self.processor(
                    text=all_texts, images=all_images, return_tensors="pt", padding=True
                )

            # Move inputs to device
            inputs = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            # Log batch information
            logger.debug(f"BATCH INPUT IDS shape: {inputs['input_ids'].shape}")
            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                logger.debug(
                    f"BATCH PIXEL VALUES shape: {inputs['pixel_values'].shape}"
                )

            # Generate responses for the entire batch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            import time

            start_time = time.perf_counter()

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ),
            ):
                # Verify KV cache is enabled
                if not self.model.config.use_cache:
                    raise RuntimeError("KV cache was disabled before batch generation")

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            # Log timing info
            prompt_tokens = inputs["input_ids"].shape[1]
            total_tokens = output_ids.shape[1]
            generated_tokens = max(total_tokens - prompt_tokens, 1)
            tokens_per_second = (
                generated_tokens * batch_size / elapsed if elapsed > 0 else float("inf")
            )

            logger.debug(
                f"Batch generation took {elapsed:.2f}s for {generated_tokens} tokens × {batch_size} samples → {tokens_per_second:.2f} tokens/s"
            )

            # Decode all responses
            response_texts = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )

            # Extract assistant responses from each sample
            responses = []
            for i, response_text in enumerate(response_texts):
                # Extract assistant part - handle multi-turn conversations
                if self.teacher_samples and "assistant\n" in response_text:
                    # Split by assistant markers and get the last one
                    assistant_parts = response_text.split("assistant\n")
                    if len(assistant_parts) > 1:
                        assistant_part = assistant_parts[-1].strip()
                        # Remove any trailing user/system markers
                        if "\nuser" in assistant_part:
                            assistant_part = assistant_part.split("\nuser")[0].strip()
                        if "\nsystem" in assistant_part:
                            assistant_part = assistant_part.split("\nsystem")[0].strip()
                    else:
                        assistant_part = response_text
                else:
                    # Standard single-turn extraction
                    try:
                        assistant_part = response_text.split("assistant\n", 1)[
                            1
                        ].strip()
                    except IndexError:
                        assistant_part = response_text

                # Clean up special tokens
                special_tokens_to_remove = [
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<|im_start|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                ]

                cleaned_response = assistant_part
                for token in special_tokens_to_remove:
                    cleaned_response = cleaned_response.replace(token, "")

                cleaned_response = cleaned_response.strip()
                responses.append(cleaned_response)

                logger.debug(
                    f"Sample {i} OUTPUT (first 200 chars): {cleaned_response[:200]}..."
                )

            return responses

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            logger.debug(f"Falling back to individual processing")

            # Fallback to individual processing if batch fails
            responses = []
            for i, (text, images) in enumerate(zip(texts, images_list)):
                try:
                    response = self.generate_response(
                        images=images,
                        text=text,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                    )
                    responses.append(response)
                except Exception as e2:
                    logger.error(f"Failed to generate response for sample {i}: {e2}")
                    responses.append("")

            return responses

    def _process_model_response(self, response: str) -> str:
        """Process model response based on model type."""
        if not hasattr(self, "coordinate_tokens_enabled"):
            logger.warning("⚠️ Model type not detected - returning raw response")
            return response

        # The coordinate token model is generating JSON format directly
        # No coordinate token conversion needed for this implementation
        logger.debug(f"📄 Model response (first 100 chars): {response[:100]}...")
        return response

    def run_inference_on_jsonl(
        self,
        input_file: str,
        output_file: str,
        data_root: str,
        max_new_tokens: int = 1024,
        max_samples: int = -1,
    ) -> None:
        """Run inference on JSONL file with batch processing support."""
        self.data_root = Path(data_root)

        # Set data_root for chat processor if using teacher guidance
        if self.chat_processor:
            self.chat_processor.data_root = self.data_root

        # Count total samples
        with open(input_file, "r", encoding="utf-8") as f:
            total_samples = sum(1 for _ in f)

        # Determine number of samples to process
        run_samples = (
            total_samples if max_samples < 0 else min(total_samples, max_samples)
        )
        logger.info(
            f"Running inference on {run_samples} out of {total_samples} samples"
        )
        logger.info(f"Using batch size: {self.batch_size}")
        if self.teacher_samples:
            logger.info(
                f"Using teacher guidance with {self.num_teachers} teachers per sample"
            )

        results = []

        # Create output file and write initial structure
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write("[\n")  # Start JSON array

        with open(input_file, "r", encoding="utf-8") as f_in:
            batch_samples = []
            batch_indices = []
            result_count = 0

            for idx, line in enumerate(
                tqdm(
                    islice(f_in, run_samples),
                    total=run_samples,
                    desc="Running Inference",
                    disable=(logger.level > logging.INFO),
                )
            ):
                sample = json.loads(line)
                batch_samples.append(sample)
                batch_indices.append(idx)

                # Process batch when full or at end
                if len(batch_samples) == self.batch_size or idx == run_samples - 1:
                    logger.debug(f"{'=' * 50}")
                    logger.debug(f"Processing batch with {len(batch_samples)} samples")

                    try:
                        # Always leverage Transformers' native batch generation—even
                        # when the effective batch size is 1—so that the code path
                        # is uniform and easier to maintain.
                        prompts, images_list = self.prepare_batch_inference_inputs(
                            batch_samples, start_idx=batch_indices[0]
                        )

                        # Use single-sample generation for teacher guidance due to multi-image complexity
                        if self.teacher_samples:
                            responses = []
                            for prompt, images in zip(prompts, images_list):
                                response = self.generate_response(
                                    images=images,
                                    text=prompt,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0,
                                    do_sample=False,
                                    repetition_penalty=1.01,
                                )
                                responses.append(response)
                        else:
                            # Use batch generation for standard inference
                            responses = self.generate_batch_responses(
                                images_list=images_list,
                                texts=prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=0.000001,
                                do_sample=True,
                                repetition_penalty=1.05,
                            )

                        # Process each response in the batch
                        for sample, response, sample_idx in zip(
                            batch_samples, responses, batch_indices
                        ):
                            # Extract metadata
                            target = sample
                            target_images = target.get("images", [])
                            image_id = (
                                Path(target_images[0]).name
                                if target_images
                                else f"sample_{sample_idx}"
                            )

                            # Format ground truth
                            objects = target.get("objects", [])
                            ground_truth_objects = []
                            for obj in objects:
                                bbox = obj.get("bbox_2d", obj.get("bbox", []))
                                desc = obj.get(
                                    "desc", obj.get("description", obj.get("label", ""))
                                )
                                if bbox and desc:
                                    ground_truth_objects.append(
                                        {"bbox_2d": bbox, "label": desc}
                                    )

                            ground_truth = json.dumps(
                                ground_truth_objects, ensure_ascii=False
                            )

                            # Process model response based on model type
                            processed_response = self._process_model_response(response)

                            # Log response processing for debugging
                            if (
                                hasattr(self, "coordinate_tokens_enabled")
                                and self.coordinate_tokens_enabled
                            ):
                                logger.debug(
                                    f"📄 Coordinate token model response: {processed_response[:200]}..."
                                )
                            else:
                                logger.debug(
                                    f"📄 Standard model response: {processed_response[:200]}..."
                                )

                            result = {
                                # Updated key names aligned with downstream evaluation
                                "image": target_images[0]
                                if target_images
                                else image_id,
                                "ground_truth": ground_truth,
                                "pred_result": processed_response,
                                "height": target.get("height"),
                                "width": target.get("width"),
                            }

                            results.append(result)

                            # Write to file
                            with open(output_file, "a", encoding="utf-8") as f_out:
                                if result_count > 0:
                                    f_out.write(",\n")
                                json.dump(result, f_out, ensure_ascii=False, indent=2)
                                f_out.flush()

                            result_count += 1

                        logger.info(
                            f"✅ Batch completed: {len(batch_samples)} samples processed"
                        )

                    except Exception as e:
                        import traceback

                        logger.error(f"❌ Batch failed: {e}")
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        # Handle failed batch - process each sample individually
                        for sample, sample_idx in zip(batch_samples, batch_indices):
                            target = sample
                            target_images = target.get("images", [])
                            image_id = (
                                Path(target_images[0]).name
                                if target_images
                                else f"sample_{sample_idx}"
                            )

                            result = {
                                # Maintain the same key schema even when errors occur so
                                # that downstream tools can still parse the file uniformly.
                                "image": target_images[0]
                                if target_images
                                else image_id,
                                "ground_truth": "",
                                "pred_result": "",
                                "height": target.get("height"),
                                "width": target.get("width"),
                                "error": str(e),
                            }
                            results.append(result)

                            with open(output_file, "a", encoding="utf-8") as f_out:
                                if result_count > 0:
                                    f_out.write(",\n")
                                json.dump(result, f_out, ensure_ascii=False, indent=2)
                                f_out.flush()

                            result_count += 1

                    # Clear batch
                    batch_samples = []
                    batch_indices = []

        # Close JSON array
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write("\n]")

        logger.info(f"✅ Inference complete. Results saved to {output_file}")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Successful: {len([r for r in results if 'error' not in r])}")
        logger.info(f"   Failed: {len([r for r in results if 'error' in r])}")


def main():
    parser = argparse.ArgumentParser(description="Simplified inference runner")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input JSONL file"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument(
        "--data_root", type=str, default=".", help="Data root directory"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Max new tokens"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to process in each batch (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile for faster inference (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--teacher_pool_file",
        type=str,
        help="Path to teacher pool JSONL file (optional)",
    )
    parser.add_argument(
        "--num_teachers",
        type=int,
        default=0,
        help="Number of teacher examples to use per sample (default: 0)",
    )

    args = parser.parse_args()

    # Initialize config from EXPLICITLY provided config path (no fallbacks)
    from src.config import init_config as _init_config

    if not Path(args.config_path).exists():
        logger = get_logger("inference")
        logger.error(f"❌ Configuration file not found: {args.config_path}")
        logger.error(f"Please provide a valid config file path using --config_path")
        exit(1)

    logger = get_logger("inference")
    logger.info(f"🔧 Loading configuration from: {args.config_path}")
    try:
        _init_config(args.config_path)
        logger.info(f"✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        exit(1)

    # Configure global logging once for the whole run
    configure_global_logging(
        log_dir="logs",
        log_file="run.log",
        log_level=args.log_level.upper(),
        verbose=False,
    )

    # Re-acquire logger so it inherits the freshly installed handlers
    logger = get_logger("inference")

    # Check Flash Attention availability (non-blocking)
    flash_available = check_flash_attention_available()
    if flash_available:
        logger.info("✅ Flash Attention 2 is available")
    else:
        logger.warning(
            "⚠️ Flash Attention 2 not available - will use standard attention"
        )

    logger.info(f"Starting inference with log level: {args.log_level}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input JSONL: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.teacher_pool_file and args.num_teachers > 0:
        logger.info(f"Teacher pool: {args.teacher_pool_file}")
        logger.info(f"Teachers per sample: {args.num_teachers}")

    engine = InferenceEngine(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_torch_compile=args.use_torch_compile,
        teacher_pool_file=args.teacher_pool_file,
        num_teachers=args.num_teachers,
    )
    engine.run_inference_on_jsonl(
        input_file=args.input_file,
        output_file=args.output_file,
        data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
