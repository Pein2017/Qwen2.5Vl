#!/usr/bin/env python3
# ! Ignore this file for now
"""
Standalone inference runner for Qwen2.5-VL.

This script runs inference on a JSONL dataset, processing each sample
through the model to generate text predictions. It uses the ChatProcessor
to construct the correct chat template with few-shot examples.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Runtime type-checking ---------------------------------------------------
from typeguard import typechecked

from data_conversion.vision_process import (
    MAX_PIXELS,
    MIN_PIXELS,
)
from src.chat_processor import ChatProcessor
from src.config import init_config
from src.logger_utils import get_model_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches
from src.models.wrapper import Qwen25VLWithDetection
from src.schema import (
    AttentionMaskType,
    LLMTokenType,
    assert_detection_head_outputs,
    assert_tensor_shape,
)
from src.tokens.special_tokens import SpecialTokens

# Dim symbols (aligned with chat_processor.py)
S = TypeVar("S")  # Sequence length
B = TypeVar("B")
PT = TypeVar("PT")
C = TypeVar("C")
H = TypeVar("H")
W = TypeVar("W")

logger = get_model_logger()

# Apply all critical Qwen2.5-VL fixes **before** we import the model so that
# the patched functions are in effect as soon as the module is loaded.

# Apply once at import time
if not apply_comprehensive_qwen25_fixes():
    raise RuntimeError(
        "Failed to apply Qwen2.5-VL patches – cannot proceed with inference."
    )

# Optional sanity check
verify_qwen25_patches()

# Global inference mode: 'generate' or 'detect'
INFERENCE_MODE = "generate"


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_training_prompts: bool = False,  # New parameter
        language: str = "chinese",  # New parameter
    ):
        """
        Initialize inference engine with context-aware prompts.

        Args:
            model_path: Path to the trained model
            device: Device to run inference on
            torch_dtype: Torch data type for inference
            use_training_prompts: Whether to use detailed training prompts for inference
            language: Language for prompts ("chinese" or "english")
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_training_prompts = use_training_prompts
        self.language = language

        # Load model and tokenizer
        self.model, self.tokenizer, self.image_processor = (
            self._load_model_and_tokenizer(model_path)
        )

        # Create chat processor with inference context
        self.chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            merge_size=self.image_processor.merge_size,
            max_length=self.tokenizer.model_max_length,
            use_training_prompts=use_training_prompts,
            language=language,
        )

        # Set inference context
        context = "inference"  # Always use inference context for standalone inference
        self.chat_processor.set_context(context)

        logger.info(f"✅ InferenceEngine initialized with context-aware prompts")
        logger.info(f"   Use training prompts: {use_training_prompts}")
        logger.info(f"   Language: {language}")
        logger.info(
            f"   System prompt: {self.chat_processor.get_current_system_prompt()[:100]}..."
        )

    def set_prompt_context(
        self, context: str = "inference", use_training_prompts: bool = None
    ):
        """
        Update the prompt context for different inference scenarios.

        Args:
            context: "inference", "evaluation", or "few_shot"
            use_training_prompts: Override the training prompts setting
        """
        if use_training_prompts is not None:
            self.use_training_prompts = use_training_prompts
            self.chat_processor.use_training_prompts = use_training_prompts

        self.chat_processor.set_context(context)
        logger.info(f"🔄 Updated inference context to: {context}")
        logger.info(
            f"   Using training prompts: {self.chat_processor.use_training_prompts}"
        )

    def predict_single_image(
        self,
        image_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        use_few_shot: bool = False,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Predict objects in a single image with context-aware prompts.

        Args:
            image_path: Path to the input image
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            use_few_shot: Whether to use few-shot examples
            few_shot_examples: List of few-shot examples

        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            if use_few_shot and few_shot_examples:
                logger.debug(
                    f"🔍 Using few-shot with {len(few_shot_examples)} teacher samples"
                )

                # Create sample with teachers
                sample = {
                    "teachers": few_shot_examples,
                    "student": {"images": [image_path]},
                }
            else:
                sample = {"student": {"images": [image_path]}}

            # Process sample through chat processor
            processed_sample = self.chat_processor.process_sample(sample)

            # Create inputs for generation
            input_ids = processed_sample["input_ids"].unsqueeze(0)  # Add batch dim
            pixel_values = processed_sample.get("pixel_values")
            image_grid_thw = processed_sample.get("image_grid_thw")

            # Build model inputs
            model_inputs = {"input_ids": input_ids}

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values.unsqueeze(
                    0
                )  # Add batch dim

            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

            # Log generation info
            logger.debug(f"🚀 GENERATION INFO:")
            logger.debug(f"   Input sequence length: {input_ids.shape[-1]}")
            logger.debug(f"   Has vision input: {pixel_values is not None}")
            logger.debug(
                f"   Vision tensor shape: {pixel_values.shape if pixel_values is not None else None}"
            )
            logger.debug(f"   Max new tokens: {max_new_tokens}")
            logger.debug(f"   Temperature: {temperature}")
            logger.debug(f"   Few-shot mode: {'Yes' if use_few_shot else 'No'}")
            logger.debug(
                f"   Num teachers: {len(few_shot_examples) if few_shot_examples else 0}"
            )

            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **{
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()
                    },
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][model_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            # Parse JSON response
            from .response_parser import parse_detection_response

            parsed_objects = parse_detection_response(generated_text)

            return {
                "image_path": image_path,
                "objects": parsed_objects,
                "raw_response": generated_text,
                "context": "few_shot" if use_few_shot else "inference",
                "num_examples": len(few_shot_examples) if few_shot_examples else 0,
                "prompt_type": "training"
                if self.chat_processor.use_training_prompts
                else "evaluation",
            }

        except Exception as e:
            logger.error(f"❌ Prediction failed for {image_path}: {e}")
            return {
                "image_path": image_path,
                "objects": [],
                "raw_response": "",
                "error": str(e),
                "context": "few_shot" if use_few_shot else "inference",
                "num_examples": len(few_shot_examples) if few_shot_examples else 0,
                "prompt_type": "training"
                if self.chat_processor.use_training_prompts
                else "evaluation",
            }

    @typechecked
    def _prepare_inference_prompt(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[Image.Image]]:
        """Prepare the prompt and images for a single inference sample."""
        messages = self.chat_processor._create_conversation_messages(sample)

        # Remove the final assistant response (ground-truth) so that we generate it.
        inference_messages = messages[:-1]

        # Replace <image> placeholders with proper vision tokens **and** load the images
        image_paths = self.chat_processor._extract_all_image_paths(sample)
        (
            processed_messages,
            images,
            _image_dims,
        ) = self.chat_processor._process_images_and_tokens(
            inference_messages, image_paths
        )

        # Serialise ``ChatMessage`` objects to dicts for the HF chat template.
        text_prompt = self.tokenizer.apply_chat_template(
            [asdict(msg) for msg in processed_messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        return text_prompt, images

    @typechecked
    def generate_response(
        self,
        images: List[Image.Image],
        text: str,
        max_new_tokens: int,
        *,
        temperature: float = 1e-6,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Generate a response using model.generate()."""
        # Use ChatProcessor helper to build inputs exactly like training / official inference
        inputs_dict = self.chat_processor.prepare_inputs_for_inference(
            images=images, text=text, is_first_step=True
        )

        # ------------------------------------------------------------------
        # Move *all* tensor inputs to the very same device as the model
        # parameters.  This is more reliable than using the user-provided
        # `self.device` string because the model might be offloaded to a
        # specific CUDA device ("cuda:0", "cuda:1", …) by Accelerate's
        # internal logic when `device_map` was set.  Fetching the device
        # from the first parameter guarantees consistency.
        # ------------------------------------------------------------------

        target_device = next(self.model.parameters()).device  # e.g. cuda:0

        for key, value in inputs_dict.items():
            if torch.is_tensor(value):
                inputs_dict[key] = value.to(device=target_device, non_blocking=True)

        inputs = inputs_dict

        # Use automatic mixed-precision (bfloat16) during the forward pass to
        # further speed-up inference while keeping memory usage low.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]

        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        return response

    @assert_tensor_shape
    @typechecked
    def predict_detection(
        self, images: List[Image.Image], text: str
    ) -> List[Dict[str, Any]]:
        """Run detection head inference to produce bounding boxes and captions."""
        # Fail-fast if detection head not available
        if not getattr(self.model, "detection_enabled", False) or not hasattr(
            self.model, "detection_head"
        ):
            raise RuntimeError(
                "Detection head not available on the model. Please ensure you loaded a model trained with detection_enabled=True."
            )
        # Prepare inputs for detection
        inputs_dict = self.chat_processor.prepare_inputs_for_inference(
            images=images, text=text, is_first_step=True
        )
        # Move all tensor inputs to correct device
        target_device = next(self.model.parameters()).device
        for key, value in inputs_dict.items():
            if torch.is_tensor(value):
                inputs_dict[key] = value.to(device=target_device, non_blocking=True)

        # Forward base model to get hidden states
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.base_model(
                **inputs_dict, output_hidden_states=True, return_dict=True
            )
        # Use full hidden state list for two-stream detection
        hidden_states_list: list[LLMTokenType] = list(
            outputs.hidden_states
        )  # convert tuple to list for shape checking
        attention_mask: AttentionMaskType = inputs_dict.get("attention_mask")

        # Extract raw vision features from the visual tower
        pixel_vals = inputs_dict.get("pixel_values")
        grid_thw = inputs_dict.get("image_grid_thw")
        if pixel_vals is not None and grid_thw is not None:
            vision_feats = self.model.base_model.visual(pixel_vals, grid_thw)
        else:
            vision_feats = None
        # Invoke detection head with both language and vision features
        detection_outputs = self.model.detection_head(
            hidden_states_list,
            attention_mask,
            training=False,
            vision_feats=vision_feats,
        )
        assert_detection_head_outputs(detection_outputs)

        # Extract and process predictions
        pred_boxes = detection_outputs["pred_boxes"][0]  # (N,4) normalized
        pred_logits_obj = detection_outputs.get("pred_objectness")[0]  # (N,) logits
        # Compute objectness probabilities
        obj_scores = torch.sigmoid(pred_logits_obj)
        caption_logits = detection_outputs["caption_logits"][0]  # (N,L,V)
        token_ids = caption_logits.argmax(dim=-1)  # (N,L)

        # Determine target image dimensions (last image)
        widths = [img.size[0] for img in images]
        heights = [img.size[1] for img in images]
        target_w, target_h = widths[-1], heights[-1]

        results: List[Dict[str, Any]] = []
        # Apply objectness thresholding and EOS trimming per query
        eos_id = self.tokenizer.eos_token_id
        obj_threshold = 0.5
        for idx in range(token_ids.shape[0]):
            # Skip low-confidence predictions
            if obj_scores[idx].item() < obj_threshold:
                continue
            # Box conversion
            x1_norm, y1_norm, x2_norm, y2_norm = pred_boxes[idx].tolist()
            x1, y1 = int(x1_norm * target_w), int(y1_norm * target_h)
            x2, y2 = int(x2_norm * target_w), int(y2_norm * target_h)
            # Trim tokens at EOS
            seq = token_ids[idx].tolist()
            if eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            # Skip if no tokens
            if not seq:
                continue
            # Decode trimmed tokens to text
            label = self.tokenizer.decode(seq, skip_special_tokens=True)
            results.append({"bbox_2d": [x1, y1, x2, y2], "label": label})
        return results

    def run_inference_on_jsonl(
        self,
        input_jsonl: str,
        output_file: str,
        data_root: str,
        max_new_tokens: int,
    ) -> None:
        """Run inference on a JSONL file and save the results."""
        self.chat_processor.data_root = Path(data_root)

        # Compute total number of samples once so that tqdm can display an
        # accurate progress bar with time-left estimation. This extra pass is
        # lightweight compared to model inference.
        with open(input_jsonl, "r", encoding="utf-8") as _f:
            total_samples = sum(1 for _ in _f)

        with (
            open(input_jsonl, "r", encoding="utf-8") as f_in,
            open(output_file, "w", encoding="utf-8") as f_out,
        ):
            # Start JSON array
            f_out.write("[\n")

            for idx, line in enumerate(
                tqdm(f_in, total=total_samples, desc="Running Inference", unit="sample")
            ):
                sample = json.loads(line)

                # Prepare prompt and images
                prompt, images = self._prepare_inference_prompt(sample)

                # Run inference according to mode
                if INFERENCE_MODE == "generate":
                    inference_result = self.generate_response(
                        images, prompt, max_new_tokens
                    )
                elif INFERENCE_MODE == "detect":
                    inference_result = self.predict_detection(images, prompt)
                else:
                    raise RuntimeError(f"Unknown INFERENCE_MODE: {INFERENCE_MODE}")

                # Build result dict in the requested format
                target = sample.get("target", sample)
                target_images = target.get("images", [])
                target_image_name = (
                    Path(target_images[0]).name if target_images else "N/A"
                )

                result = {
                    "id": target_image_name,
                    "ground_truth": self.chat_processor._format_objects_response(
                        target.get("objects", [])
                    ),
                    "result": inference_result,
                }

                # Stream JSON list to disk: write comma for all but first element
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(result, ensure_ascii=False))

                # Persist the just-computed result to disk right away so that
                # partial outputs are not lost if the process is interrupted.
                f_out.flush()
                os.fsync(f_out.fileno())

            # Close JSON array
            f_out.write("\n]\n")

        logger.info(f"✅ Inference complete. Results saved to {output_file}")

    def _validate_special_tokens(self, tokenizer):
        """Fail-fast check for special token ID mismatches or vocabulary shift."""
        mismatches = []

        tokens = SpecialTokens()

        for name, expected_id in [
            ("image_start_id", tokens.image_start_id),
            ("image_end_id", tokens.image_end_id),
            ("image_pad_id", tokens.image_pad_id),
        ]:
            actual_id = getattr(tokenizer, name, None)
            if actual_id != expected_id:
                mismatches.append(f"{name}: expected {expected_id}, got {actual_id}")

        if mismatches:
            raise ValueError(f"❌ Special token ID mismatches: {mismatches}")

        logger.info("✅ Special token validation passed")

    def _load_model_and_tokenizer(self, model_path: str):
        """Load model, tokenizer, and image processor from checkpoint."""
        # Enforce offline mode for HuggingFace
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"❌ Model path does not exist: {model_path}")

        # Load processor first to get exact tokenizer
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Align image processor constants with data conversion pipeline
        ip = processor.image_processor
        ip.min_pixels = MIN_PIXELS
        ip.max_pixels = MAX_PIXELS

        logger.info(
            f"🔧 Image-processor constants patched: min_pixels={ip.min_pixels}, "
            f"max_pixels={ip.max_pixels}"
        )

        logger.info("🚀 Loading Qwen2.5-VL model (with DetectionHead) from disk …")

        # Load model with detection head
        model = Qwen25VLWithDetection.from_pretrained(
            model_path,
            load_detection_head=True,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
            tokenizer=processor.tokenizer,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Move to device and set eval mode
        model.to(torch.device(self.device))
        if getattr(model, "detection_enabled", False):
            model.eval()
        else:
            model.base_model.eval()

        logger.info("✅ Model weights & detection head loaded.")

        # Validate special tokens
        self._validate_special_tokens(processor.tokenizer)

        return model, processor.tokenizer, processor.image_processor

    def _extract_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from sample for result tracking."""
        student = sample.get("student", sample.get("target", sample))
        student_images = student.get("images", [])
        student_image_name = Path(student_images[0]).name if student_images else "N/A"

        return {
            "id": student_image_name,
            "ground_truth": self._normalize_ground_truth_for_eval(
                student.get("objects", [])
            ),
        }


def load_images_from_paths(
    image_paths: List[str], data_root: Path
) -> List[Image.Image]:
    """Load images from file paths."""
    images = []
    for path in image_paths:
        full_path = data_root / path
        if full_path.exists():
            images.append(Image.open(full_path).convert("RGB"))
        else:
            logger.warning(f"Image not found: {full_path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a JSONL dataset.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output predictions.",
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory for images."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base_flat.yaml",
        help="Path to the flat YAML config file (required by ChatProcessor).",
    )
    args = parser.parse_args()

    # Initialize global configuration so ChatProcessor can access values like data_root, language, etc.
    init_config(args.config_path)

    inference = InferenceEngine(model_path=args.model_path, device=args.device)
    inference.run_inference_on_jsonl(
        input_jsonl=args.input_jsonl,
        output_file=args.output_file,
        data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
