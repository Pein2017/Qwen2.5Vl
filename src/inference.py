#!/usr/bin/env python3
# ! Ignore this file for now
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
from transformers import AutoProcessor

import src.logger_utils as _logger_utils

# Runtime type-checking ---------------------------------------------------

# Dim symbols (aligned with chat_processor.py)
S = TypeVar("S")  # Sequence length
B = TypeVar("B")
PT = TypeVar("PT")
C = TypeVar("C")
H = TypeVar("H")
W = TypeVar("W")


# -------------------- Setup proper logging --------------------
def setup_logging(log_level: str = "info") -> logging.Logger:
    """Configure a single persistent *infer* logger.

    The logger always writes to *logs/infer.log* (overwriting on every run) and
    simultaneously streams logs to the console.  No timestamp-based filenames
    are used – this keeps the number of generated files minimal.
    """

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    level = level_map.get(log_level.lower(), logging.INFO)

    # Clean previous handlers to avoid duplicate logs when `setup_logging` is
    # invoked multiple times (e.g. during interactive usage).
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Ensure log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "infer.log")  # Overwrite each run

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Expose a dedicated "infer" child logger for downstream modules
    logger = logging.getLogger("infer")
    logger.setLevel(level)
    return logger


# Initialize logger (will be reconfigured in main)
logger = logging.getLogger(__name__)

# -------------------- Monkey-patch logger utils for compatibility --------------------
# Some modules still expect the old logger interface, so we provide minimal compatibility


class _CompatLogger:
    def __init__(self, logger):
        self._logger = logger

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)


# Initialize config with default settings
from pathlib import Path as _Path

from src.config import init_config as _init_config

_default_cfg_path = _Path(__file__).resolve().parents[1] / "configs" / "base_flat.yaml"
_init_config(str(_default_cfg_path))

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


def verify_flash_attention_available() -> None:
    """Ensure Flash Attention 2 is present in the current environment.

    The check intentionally remains lightweight – we import the expected module
    and inspect CUDA availability.  A `RuntimeError` is raised on failure so
    that the caller can surface the issue immediately.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Flash Attention 2 requires CUDA. "
                "CPU inference is not supported."
            )

        # Lazy import – will raise if the fused kernels are missing.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLFlashAttention2,  # noqa: F401 – imported for availability check only
        )

    except ImportError as e:
        raise RuntimeError(
            "Flash Attention 2 is not available in the environment: "
            f"{e}. Make sure the `flash-attn` package is installed and that "
            "your GPU/driver/CUDA stack is compatible."
        )


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

        # Verify CUDA is available (required for Flash Attention 2)
        if device == "cpu":
            raise RuntimeError(
                "CPU inference is not supported. Flash Attention 2 requires CUDA. "
                "Please use a CUDA-enabled device."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This inference script requires a GPU with CUDA support "
                "for Flash Attention 2 and optimal performance."
            )

        # Load processor and model following demo approach
        self.processor, self.model = self._load_model_and_processor()

        # Set to evaluation mode
        self.model.eval()

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

        # Flash Attention is already configured during model loading via attn_implementation parameter
        # No need to manually set it here

        # Final verification of critical features
        # Verify Flash Attention 2 is enabled
        if not hasattr(self.model.config, "_attn_implementation"):
            raise RuntimeError(
                "Model config missing _attn_implementation attribute. "
                "This indicates Flash Attention 2 setup failed."
            )

        if self.model.config._attn_implementation != "flash_attention_2":
            raise RuntimeError(
                f"Expected Flash Attention 2 but got: {self.model.config._attn_implementation}. "
                "Flash Attention 2 is required for optimal performance."
            )

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
        logger.info("   ⚡ Flash Attention 2: ENABLED")
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

        # Create chat processor if using teacher guidance
        self.chat_processor = None
        if self.teacher_samples:
            from src.chat_processor import ChatProcessor

            self.chat_processor = ChatProcessor(
                tokenizer=self.processor.tokenizer,
                image_processor=self.processor.image_processor,
                use_training_prompts=True,  # Always use training prompts for teacher guidance
                language=self.language,
            )
            logger.info("✅ Initialized ChatProcessor for teacher-guided inference")

    def _load_model_and_processor(self):
        """Load model and processor exactly like demo script."""
        # Enforce offline mode
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Load processor first (slow tokenizer, vision preproc) - exactly like demo
        processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, use_fast=False
        )

        # Configure tokenizer for proper batch generation with Flash Attention
        processor.tokenizer.padding_side = "left"

        # Ensure pad token is set for left padding
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            logger.info("✅ Pad token set to EOS token for proper padding")

        logger.info(
            "✅ Tokenizer padding side set to 'left' for Flash Attention compatibility"
        )

        # Override chat_template with training template via collator - exactly like demo
        try:
            from src.reference.qwen2_5vl_collator import Qwen2_5VLCollator

            _ = Qwen2_5VLCollator(
                processor
            )  # mutates processor.tokenizer.chat_template
            logger.info("✅ Chat template overridden with training collator template.")
        except Exception as exc:
            logger.warning(
                "⚠️  Could not import training collator – proceeding with checkpoint template."
            )
            logger.warning(f"   Reason: {exc}")

        # Load model exactly like demo - try adapter first, then fallback
        model = self._load_model_like_demo(model_dir)
        model.eval()

        logger.debug(f"Model loaded on device: {next(model.parameters()).device}")
        logger.debug(f"Model dtype: {next(model.parameters()).dtype}")

        return processor, model

    def _load_model_like_demo(self, model_path: Path):
        """Load model exactly like demo script - adapter-aware."""
        # 1) If checkpoint already contains full fused weights -> normal load
        adapter_config_file = model_path / "adapter_config.json"

        # ALWAYS use Flash Attention 2 - no fallback
        attn_implementation = "flash_attention_2"
        logger.info(f"🔧 Using attention implementation: {attn_implementation}")

        # Prefer adapter path if adapter_config.json exists
        if adapter_config_file.exists():
            try:
                from peft import PeftConfig, PeftModel
            except ImportError:
                raise ImportError(
                    "PEFT is required for loading adapter models. "
                    "Please install it with: pip install peft"
                )

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_model_path = Path(peft_cfg.base_model_name_or_path or "")

            if not base_model_path.exists():
                # Assume base model was saved inside the same dir
                base_model_path = model_path

            from transformers import Qwen2_5_VLForConditionalGeneration

            logger.info(f"🔧 Loading base model from {base_model_path}")

            try:
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch.bfloat16
                    if torch.cuda.is_available()
                    else torch.float32,
                    device_map=None,  # Single GPU only - no multi-GPU device mapping
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    use_cache=True,  # Enable KV cache by default
                )
                # Move to single GPU if available
                if torch.cuda.is_available():
                    base_model = base_model.to("cuda:0")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load base model with Flash Attention 2: {e}. "
                    "Ensure your environment supports Flash Attention 2."
                )

            logger.info(f"🔧 Loading PEFT adapter weights from {model_path}")
            model = PeftModel.from_pretrained(base_model, str(model_path))

            # Optionally merge LoRA weights for inference efficiency
            try:
                model = model.merge_and_unload()
                logger.info("✅ LoRA adapter merged into base model for inference.")
            except Exception as exc:
                logger.warning(
                    f"⚠️  Could not merge adapter (continuing with PEFT model): {exc}"
                )

            # Verify Flash Attention is enabled
            if (
                not hasattr(model.config, "_attn_implementation")
                or model.config._attn_implementation != "flash_attention_2"
            ):
                raise RuntimeError(
                    "Flash Attention 2 was not properly enabled on the model. "
                    "This is required for optimal performance."
                )

            # Verify KV cache is enabled
            if not model.config.use_cache:
                raise RuntimeError(
                    "KV cache is not enabled. This is required for efficient generation."
                )

            return model

        # 2) If no adapter config, attempt AutoPeftModelForCausalLM
        try:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                device_map=None,  # Single GPU only - no multi-GPU device mapping
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                use_cache=True,  # Enable KV cache by default
            )
            # Move to single GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda:0")

            # Verify Flash Attention is enabled
            if (
                not hasattr(model.config, "_attn_implementation")
                or model.config._attn_implementation != "flash_attention_2"
            ):
                raise RuntimeError(
                    "Flash Attention 2 was not properly enabled on the model. "
                    "This is required for optimal performance."
                )

            # Verify KV cache is enabled
            if not model.config.use_cache:
                raise RuntimeError(
                    "KV cache is not enabled. This is required for efficient generation."
                )

            logger.info(
                "✅ Loaded model via AutoPeftModelForCausalLM (adapter already fused)."
            )
            return model
        except ImportError:
            pass  # PEFT not installed, try plain model
        except (ValueError, OSError):
            pass  # Fall back to plain model

        # 3) Plain full-weights load
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                device_map=None,  # Single GPU only - no multi-GPU device mapping
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                use_cache=True,  # Enable KV cache by default
            )
            # Move to single GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda:0")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with Flash Attention 2: {e}. "
                "Ensure your environment supports Flash Attention 2."
            )

        # Verify Flash Attention is enabled
        if (
            not hasattr(model.config, "_attn_implementation")
            or model.config._attn_implementation != "flash_attention_2"
        ):
            raise RuntimeError(
                "Flash Attention 2 was not properly enabled on the model. "
                "This is required for optimal performance."
            )

        # Verify KV cache is enabled
        if not model.config.use_cache:
            raise RuntimeError(
                "KV cache is not enabled. This is required for efficient generation."
            )

        logger.info("✅ Loaded full Qwen2.5-VL model (no adapter detected).")
        logger.info(f"   Attention implementation: {attn_implementation}")
        logger.info(f"   KV cache enabled: {model.config.use_cache}")
        return model

    def _load_teacher_pool(self) -> List[Dict[str, Any]]:
        """Load teacher samples from teacher pool JSONL file."""
        teacher_pool_path = Path(self.teacher_pool_file)
        if not teacher_pool_path.exists():
            raise FileNotFoundError(
                f"Teacher pool file not found: {self.teacher_pool_file}"
            )

        teacher_samples = []
        with open(teacher_pool_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    teacher_samples.append(sample)

        if not teacher_samples:
            raise ValueError("No teacher samples found in teacher pool file")

        return teacher_samples

    def _sample_teachers(self, seed: int = None) -> List[Dict[str, Any]]:
        """Sample random teacher examples."""
        if seed is not None:
            random.seed(seed)

        if self.num_teachers >= len(self.teacher_samples):
            return self.teacher_samples.copy()

        return random.sample(self.teacher_samples, self.num_teachers)

    def prepare_inference_inputs(
        self, sample: Dict[str, Any], seed: int = None
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
        self, student_sample: Dict[str, Any], seed: int = None
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
        prompt_str: str = self.chat_processor.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

        logger.debug(f"Built teacher-guided prompt with {len(images)} images")
        logger.debug(f"Teacher examples: {len(teachers)}")
        logger.debug(f"Prompt preview (first 500 chars): {prompt_str[:500]}...")

        return prompt_str, images

    def _prepare_standard_inputs(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[Image.Image]]:
        """Prepare inputs for standard inference (original implementation)."""
        # Get the target data (handles both direct format and nested format)
        target = sample.get("target", sample.get("student", sample))

        # Load images
        image_paths = target.get("images", [])
        images = []

        for img_path in image_paths:
            full_path = (
                Path(self.data_root) / img_path
                if hasattr(self, "data_root")
                else Path(img_path)
            )
            if full_path.exists():
                image = Image.open(full_path).convert("RGB")
                images.append(image)
            else:
                logger.warning(f"Image not found: {full_path}")

        # Create the Chinese prompt exactly like demo
        question_cn = """
你是专业的通信机房BBU设备检测AI助手，负责精确识别和定位图像中的所有相关设备和部件。

**检测任务**:识别图像中所有目标对象的位置和类型。

**目标对象分类**:

1. **BBU设备**
   - bbu基带处理单元/华为
   - bbu基带处理单元/中兴  
   - bbu基带处理单元/爱立信

2. **螺丝连接点**
   - 螺丝连接点/BBU安装螺丝
   - 螺丝连接点/CPRI光缆和BBU连接点
   - 螺丝连接点/地排处螺丝
   - 螺丝连接点/BBU接地线机柜接地端
   - 螺丝连接点/BBU尾纤和ODF连接点

3. **线缆**
   - 线缆/光纤
   - 线缆/非光纤

4. **机柜部件**
   - 机柜空间
   - 挡风板
   - 标签贴纸

**检测要求**:
1. 仔细观察图像中的每个区域，识别所有目标对象
2. 准确标注边界框，确保完全包含目标对象
3. 使用上述标准分类标签，保持格式一致性
4. 忽略图像中的水印、时间戳等叠加信息
5. 对于部分遮挡的对象，标注可见部分的边界

**输出格式**:严格按照JSON数组格式输出:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "标准分类标签"}
]
```

坐标说明:(x1,y1)为左上角，(x2,y2)为右下角，使用绝对像素坐标。
"""

        # Create messages exactly like demo
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_cn},
                ],
            }
        ]

        # Apply chat template exactly like demo
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
        # Use ChatProcessor for teacher-guided inference (images are already processed)
        if self.chat_processor and self.teacher_samples:
            inputs = self.chat_processor.prepare_inputs_for_inference(
                images=images,
                text=text,
                is_first_step=True,
            )
        else:
            # Standard inference: prepare inputs exactly like demo
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

        # Extract assistant part exactly like demo
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
        """Generate responses for a batch of inputs using native transformers batch support.

        Transformers' model.generate() already supports batch processing natively.
        We just need to prepare the inputs correctly.

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
        # Handle single-image case (most common for this task)
        # Flatten images for batch processing
        all_images = []
        for images in images_list:
            if images:
                all_images.append(images[0])  # Assuming one image per sample
            else:
                # Create a dummy image if none provided
                all_images.append(Image.new("RGB", (224, 224), color="white"))

        # Prepare batch inputs - transformers handles batching natively
        inputs = self.processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device with non-blocking transfer
        inputs = {
            k: v.to(self.model.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        # Log batch information
        logger.debug(f"BATCH SIZE: {len(texts)}")
        logger.debug(f"INPUT IDS shape: {inputs['input_ids'].shape}")
        if "pixel_values" in inputs:
            logger.debug(f"PIXEL VALUES shape: {inputs['pixel_values'].shape}")

        # Generate with timing
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

            # Native batch generation - transformers handles batching automatically
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,  # Explicitly enable KV cache for generation
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        # Log timing info
        batch_size = len(texts)
        # Calculate tokens generated per sample
        prompt_tokens = inputs["input_ids"].shape[1]
        total_output_tokens = output_ids.shape[1]
        generated_tokens = max(total_output_tokens - prompt_tokens, 1)
        total_generated = generated_tokens * batch_size
        tokens_per_second = total_generated / elapsed if elapsed > 0 else float("inf")

        logger.debug(
            f"Batch generation took {elapsed:.2f}s for {total_generated} tokens "
            f"({batch_size} samples × {generated_tokens} tokens/sample) → {tokens_per_second:.2f} tokens/s"
        )

        # Decode responses - batch_decode handles batches natively
        response_texts = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # Clean responses
        cleaned_responses = []
        for response_text in response_texts:
            try:
                assistant_part = response_text.split("assistant\n", 1)[1].strip()
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

            cleaned_responses.append(cleaned_response.strip())

        return cleaned_responses

    def run_inference_on_jsonl(
        self,
        input_jsonl: str,
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
        with open(input_jsonl, "r", encoding="utf-8") as f:
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

        with open(input_jsonl, "r", encoding="utf-8") as f_in:
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
                        for i, (sample, response, sample_idx) in enumerate(
                            zip(batch_samples, responses, batch_indices)
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

                            result = {
                                # Updated key names aligned with downstream evaluation
                                "image": target_images[0]
                                if target_images
                                else image_id,
                                "ground_truth": ground_truth,
                                "pred_result": response,
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
                        logger.error(f"❌ Batch failed: {e}")
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
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True, help="Input JSONL file"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Data root directory"
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

    # Setup logging with specified level
    global logger
    logger = setup_logging(args.log_level)

    # Verify Flash Attention 2 availability *after* logging is configured so
    # that any raised errors are captured in the log file.  Also make sure the
    # compatibility loggers point to the freshly configured *infer* logger.
    verify_flash_attention_available()

    # Ensure legacy utility functions use the updated *infer* logger
    _logger_utils.get_logger = lambda *_, **__: _CompatLogger(logger)
    _logger_utils.get_chat_logger = lambda *_, **__: _CompatLogger(logger)
    _logger_utils.get_model_logger = lambda *_, **__: _CompatLogger(logger)
    _logger_utils.get_patches_logger = lambda *_, **__: _CompatLogger(logger)

    logger.info(f"Starting inference with log level: {args.log_level}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input JSONL: {args.input_jsonl}")
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
        input_jsonl=args.input_jsonl,
        output_file=args.output_file,
        data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
