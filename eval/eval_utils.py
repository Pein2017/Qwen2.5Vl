"""
Evaluation utilities for Qwen2.5-VL models.

This module contains utilities specifically for evaluation and inference,
extracted from the training framework to maintain clean separation of concerns.
"""

import json
import logging
import os

# Import training constants and utilities
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rope2d import get_rope_index_25
from src.utils import (
    CHAT_TEMPLATE,
    DEFAULT_BASE_MODEL_PATH,
)


class EvaluationLogger:
    """Simple logger for evaluation tasks."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_name: str = None,
        verbose: bool = True,
    ):
        self.log_dir = log_dir
        self.verbose = verbose

        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename if not provided
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"eval_{timestamp}.log"

        self.log_file = os.path.join(log_dir, log_name)

        # Create logger
        self.logger = logging.getLogger(f"eval_{id(self)}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def get_log_file(self) -> str:
        return self.log_file


class SimpleModelLoader:
    """Simple model loader for evaluation tasks."""

    def __init__(
        self,
        model_path: str,
        base_model_path: str = DEFAULT_BASE_MODEL_PATH,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        logger: Optional[EvaluationLogger] = None,
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.logger = logger or EvaluationLogger()

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        """Load the model with eager attention for inference stability."""
        self.logger.info(f"Loading model from {self.model_path}")

        # Use eager attention for inference stability
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",  # Use eager for inference stability
            device_map=None,  # Don't use device_map for inference
        )

        # Manually move to specified device
        if self.device != "auto":
            self.model = self.model.to(self.device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda:0")
        else:
            self.model = self.model.to("cpu")

        self.logger.info("Model loaded successfully")
        return self.model

    def load_processor_and_tokenizer(self) -> Tuple[AutoProcessor, AutoTokenizer]:
        """Load processor and tokenizer from base model path."""
        self.logger.info("Loading processor and tokenizer")

        # Load processor from base model
        self.processor = AutoProcessor.from_pretrained(self.base_model_path)

        # Load tokenizer from base model and set chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )

        # Set the chat template to match training setup
        self.tokenizer.chat_template = CHAT_TEMPLATE
        self.processor.chat_template = CHAT_TEMPLATE

        # Configure image processor with same settings as training
        self.processor.image_processor.max_pixels = 1003520
        self.processor.image_processor.min_pixels = 784

        self.logger.info("Processor and tokenizer loaded successfully")
        return self.processor, self.tokenizer

    def load_all(
        self,
    ) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer]:
        """Load model, processor, and tokenizer."""
        model = self.load_model()
        processor, tokenizer = self.load_processor_and_tokenizer()
        return model, processor, tokenizer


class SimpleDataPreprocessor:
    """Simple data preprocessor for evaluation tasks."""

    def __init__(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        logger: Optional[EvaluationLogger] = None,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.logger = logger or EvaluationLogger()

    def preprocess_image(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Preprocess image for inference.

        Args:
            image: PIL Image

        Returns:
            Tuple of (image_tensor, grid_thw, grid_thw_merged)
        """
        # Process image
        visual_processed = self.processor.image_processor.preprocess(
            image, return_tensors="pt"
        )
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]

        grid_thw = visual_processed["image_grid_thw"][0]

        # Calculate vision tokens
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        grid_thw_merged = grid_thw.prod() // (merge_size**2)

        return image_tensor, grid_thw, grid_thw_merged.item()

    def preprocess_for_inference(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        Complete preprocessing pipeline for inference.

        Args:
            image: PIL Image
            system_prompt: System prompt
            user_prompt: User prompt (may contain <image>)

        Returns:
            Tuple of (model_inputs, input_height, input_width)
        """
        # Preprocess image
        image_tensor, grid_thw, grid_thw_merged = self.preprocess_image(image)

        # Replace <image> with vision tokens in user prompt
        if "<image>" in user_prompt:
            vision_tokens = (
                "<|vision_start|>"
                + "<|image_pad|>" * grid_thw_merged
                + "<|vision_end|>"
            )
            user_prompt_processed = user_prompt.replace("<image>", vision_tokens)
        else:
            user_prompt_processed = user_prompt

        # Create conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_processed},
        ]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        attention_mask = torch.ones_like(input_ids)

        # Calculate position IDs
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        position_ids, _ = get_rope_index_25(
            spatial_merge_size=merge_size,
            input_ids=input_ids,
            image_grid_thw=grid_thw.unsqueeze(0),
        )

        # Create model inputs
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_tensor.unsqueeze(0),
            "image_grid_thw": grid_thw.unsqueeze(0),
            "position_ids": position_ids,
        }

        # Calculate input dimensions for coordinate scaling
        input_height = grid_thw[1] * 14
        input_width = grid_thw[2] * 14

        return inputs, input_height.item(), input_width.item()


class SimpleOutputManager:
    """Simple output manager for evaluation results."""

    def __init__(self, output_dir: str, run_name: str = None):
        self.output_dir = output_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)

        # File paths
        self.log_file = os.path.join(output_dir, "eval.log")

    def save_responses(self, responses: List[Dict[str, Any]], filename: str) -> str:
        """Save responses to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False, default=str)
        return filepath

    def save_metrics(self, metrics: Dict[str, Any], filename: str) -> str:
        """Save evaluation metrics."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return filepath

    def get_log_file(self) -> str:
        """Get log file path."""
        return self.log_file
