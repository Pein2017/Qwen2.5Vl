#!/usr/bin/env python3
"""
Inference module for Qwen2.5-VL model with proper input handling.

This module demonstrates the correct way to prepare inputs for both
model.forward() and model.generate() methods, following the official
Qwen2.5-VL implementation patterns.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.chat_processor import ChatProcessor
from src.logger_utils import get_model_logger

logger = get_model_logger()


class Qwen25VLInference:
    """
    Inference wrapper for Qwen2.5-VL with proper input handling.

    Demonstrates the key differences between forward() and generate() input preparation.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device

        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )

        self.processor = AutoProcessor.from_pretrained(model_path)

        # Initialize chat processor for advanced use cases
        self.chat_processor = ChatProcessor(
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
            model_max_length=8192,
        )

        logger.info(f"✅ Qwen2.5-VL inference initialized on {device}")

    def prepare_inputs_for_forward(
        self, images: List[Image.Image], text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model.forward() - used during training.

        Key characteristics:
        - Always includes pixel_values and image_grid_thw when images are present
        - Used for training where we need gradients through vision encoder
        - Only includes parameters accepted by the official model
        """
        # Use the official processor
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=False
        )

        # Move to device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)

        # Filter to only include valid model parameters
        valid_model_params = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "labels",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "rope_deltas",
            "cache_position",
            "second_per_grid_ts",
        }

        filtered_inputs = {
            key: value for key, value in inputs.items() if key in valid_model_params
        }

        logger.debug(f"🔧 FORWARD INPUTS:")
        logger.debug(f"   input_ids shape: {filtered_inputs['input_ids'].shape}")
        logger.debug(
            f"   pixel_values shape: {filtered_inputs.get('pixel_values', torch.empty(0)).shape}"
        )
        logger.debug(
            f"   image_grid_thw shape: {filtered_inputs.get('image_grid_thw', torch.empty(0)).shape}"
        )

        return filtered_inputs

    def prepare_inputs_for_generate(
        self, images: List[Image.Image], text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model.generate() - used during inference.

        Key characteristics:
        - Includes pixel_values and image_grid_thw for the initial prefill
        - The model's prepare_inputs_for_generation will handle subsequent steps
        - Only includes parameters accepted by the official model
        """
        # Use the official processor for the initial step
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=False
        )

        # Move to device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)

        # Filter to only include valid generation parameters
        valid_generation_params = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }

        filtered_inputs = {
            key: value
            for key, value in inputs.items()
            if key in valid_generation_params
        }

        logger.debug(f"🔧 GENERATE INPUTS (initial):")
        logger.debug(f"   input_ids shape: {filtered_inputs['input_ids'].shape}")
        logger.debug(
            f"   pixel_values shape: {filtered_inputs.get('pixel_values', torch.empty(0)).shape}"
        )
        logger.debug(
            f"   image_grid_thw shape: {filtered_inputs.get('image_grid_thw', torch.empty(0)).shape}"
        )

        return filtered_inputs

    def forward_pass(
        self,
        images: List[Image.Image],
        text: str,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a forward pass (training-style) through the model.
        """
        inputs = self.prepare_inputs_for_forward(images, text)

        if labels is not None:
            inputs["labels"] = labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
        }

    def generate_response(
        self,
        images: List[Image.Image],
        text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response using model.generate().

        This method properly handles the vision input lifecycle:
        1. Initial step: includes pixel_values and image_grid_thw
        2. Subsequent steps: vision inputs are automatically set to None
        """
        inputs = self.prepare_inputs_for_generate(images, text)

        # Generate with proper parameters
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]

        response = self.processor.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        )

        return response

    def process_sample_from_jsonl(
        self, sample: Dict, data_root: str = "./"
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sample from your JSONL format for training.

        This shows how to use your ChatProcessor for training data preparation.
        """
        # Update chat processor data root
        self.chat_processor.data_root = Path(data_root)

        # Process the sample
        processed = self.chat_processor.process_sample(sample)

        # Move to device
        for key, value in processed.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(self.device)

        return processed

    def demonstrate_input_differences(self, images: List[Image.Image], text: str):
        """
        Demonstrate the key differences between forward and generate input preparation.
        """
        logger.info("🔍 DEMONSTRATING INPUT DIFFERENCES:")

        # 1. Forward inputs (training)
        logger.info("\n1️⃣ FORWARD INPUTS (Training):")
        forward_inputs = self.prepare_inputs_for_forward(images, text)

        logger.info(f"   input_ids: {forward_inputs['input_ids'].shape}")
        logger.info(f"   pixel_values: {forward_inputs.get('pixel_values', 'None')}")
        logger.info(
            f"   image_grid_thw: {forward_inputs.get('image_grid_thw', 'None')}"
        )

        # 2. Generate inputs (inference)
        logger.info("\n2️⃣ GENERATE INPUTS (Inference - Initial Step):")
        generate_inputs = self.prepare_inputs_for_generate(images, text)

        logger.info(f"   input_ids: {generate_inputs['input_ids'].shape}")
        logger.info(f"   pixel_values: {generate_inputs.get('pixel_values', 'None')}")
        logger.info(
            f"   image_grid_thw: {generate_inputs.get('image_grid_thw', 'None')}"
        )

        # 3. Show what happens during generation
        logger.info("\n3️⃣ DURING GENERATION:")
        logger.info("   - First step: Uses pixel_values and image_grid_thw")
        logger.info(
            "   - Subsequent steps: model.prepare_inputs_for_generation sets them to None"
        )
        logger.info("   - This prevents the 'shape [0, 4, -1] is invalid' error")


def load_images_from_paths(
    image_paths: List[str], data_root: str = "./"
) -> List[Image.Image]:
    """Load images from file paths."""
    images = []
    data_root = Path(data_root)

    for path in image_paths:
        full_path = data_root / path
        if full_path.exists():
            images.append(Image.open(full_path).convert("RGB"))
        else:
            logger.warning(f"Image not found: {full_path}")

    return images


def example_usage():
    """Example usage of the inference wrapper."""

    # Initialize inference
    inference = Qwen25VLInference(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"
    )

    # Example images and text
    images = [Image.new("RGB", (224, 224), color="red")]  # Dummy image
    text = "What do you see in this image?"

    # Demonstrate input differences
    inference.demonstrate_input_differences(images, text)

    # Example 1: Forward pass (training-style)
    logger.info("\n🔧 FORWARD PASS EXAMPLE:")
    forward_result = inference.forward_pass(images, text)
    logger.info(f"   Logits shape: {forward_result['logits'].shape}")

    # Example 2: Generation (inference-style)
    logger.info("\n🔧 GENERATION EXAMPLE:")
    response = inference.generate_response(images, text, max_new_tokens=50)
    logger.info(f"   Generated response: {response}")

    # Example 3: Process JSONL sample
    logger.info("\n🔧 JSONL PROCESSING EXAMPLE:")
    sample = {
        "examples": [
            {
                "images": ["example1.jpg"],
                "objects": [{"box": [10, 10, 50, 50], "desc": "test object"}],
            }
        ],
        "target": {
            "images": ["target.jpg"],
            "objects": [{"box": [20, 20, 60, 60], "desc": "target object"}],
        },
    }

    try:
        processed = inference.process_sample_from_jsonl(
            sample, data_root="./ds_rescaled/"
        )
        logger.info(f"   Processed sample keys: {list(processed.keys())}")
    except Exception as e:
        logger.warning(f"   JSONL processing failed (expected with dummy data): {e}")


if __name__ == "__main__":
    example_usage()
