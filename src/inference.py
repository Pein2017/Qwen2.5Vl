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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

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


class Qwen25VLInference:
    """Inference wrapper for Qwen2.5-VL."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        # Enforce offline mode for HuggingFace
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.model_path = model_path
        self.device = device

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"❌ Model path does not exist: {model_path}")

        # ------------------------------------------------------------------
        # First, load the processor so we have the *exact* tokenizer that was
        # saved with the checkpoint.  This guarantees identical vocabulary &
        # special-token IDs.
        # ------------------------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------
        # 🔧 ALIGN IMAGE-PROCESSOR CONSTANTS WITH DATA-CONVERSION PIPELINE
        # ------------------------------------------------------------------
        # The JPEGs fed into inference are *already* pre-rescaled by
        # `data_conversion/vision_process.py`.
        # If we keep the HuggingFace defaults (min_pixels=56*56, …), the
        # processor may silently up- or down-scale the images again which
        # breaks the strict (<|image_pad|> ↔ vision token) alignment enforced
        # by Qwen2.5-VL.
        #
        # By copying the constants we guarantee that:
        #   1. ChatProcessor._calculate_image_tokens() uses the exact same
        #      resizing logic as the *actual* image preprocessing.
        #   2. `pixel_values` fed into the Vision Transformer match the token
        #      placeholders, preventing runtime errors such as:
        #         "Image features and image tokens do not match".
        # ------------------------------------------------------------------

        ip = self.processor.image_processor
        ip.min_pixels = MIN_PIXELS
        ip.max_pixels = MAX_PIXELS

        # For reference we still log the current values so mis-configurations
        # surface immediately in the console.
        logger.info(
            f"🔧 Image-processor constants patched: min_pixels={ip.min_pixels}, "
            f"max_pixels={ip.max_pixels}"
        )

        logger.info("🚀 Loading Qwen2.5-VL model (with DetectionHead) from disk …")

        self.model = Qwen25VLWithDetection.from_pretrained(
            model_path,
            load_detection_head=True,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            tokenizer=self.processor.tokenizer,
            local_files_only=True,
            trust_remote_code=True,
        )
        # Move entire model (base + detection head) to the inference device
        self.model.to(torch.device(self.device))
        # Ensure we are in inference mode right away (disables dropout etc.)
        if getattr(self.model, "detection_enabled", False):
            self.model.eval()
        else:
            self.model.base_model.eval()

        logger.info("✅ Model weights & detection head loaded.")

        self.chat_processor = ChatProcessor(
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
        )

        # ------------------------------------------------------------------
        # Validate special tokens & vocabulary alignment to surface silent
        # mismatches that would corrupt generation outputs.
        # ------------------------------------------------------------------

        self._validate_special_tokens()

        logger.info(f"✅ Qwen2.5-VL inference initialised on {device}")

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

        # Use the tokenizer chat template to build the final prompt that ends with the assistant tag
        text_prompt = self.processor.tokenizer.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True
        )

        return text_prompt, images

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
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]

        response = self.processor.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        )
        return response

    @assert_tensor_shape
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
        eos_id = self.processor.tokenizer.eos_token_id
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
            label = self.processor.tokenizer.decode(seq, skip_special_tokens=True)
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

    def _validate_special_tokens(self) -> None:
        """Fail-fast check for special token ID mismatches or vocabulary shift."""

        tok = self.processor.tokenizer

        mismatches = []
        for token, expected_id in SpecialTokens.TOKEN_IDS.items():
            actual_id = tok.convert_tokens_to_ids(token)
            if actual_id != expected_id:
                mismatches.append((token, expected_id, actual_id))

        if mismatches:
            details = ", ".join(
                [f"{t}: expected {e}, got {a}" for t, e, a in mismatches]
            )
            msg = (
                "❌ Special-token ID mismatch detected. This indicates that the "
                "tokenizer vocabulary differs from the one used during training. "
                f"Details: {details}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("✅ Special-token IDs match training vocabulary – safe to proceed.")


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

    inference = Qwen25VLInference(model_path=args.model_path, device=args.device)
    inference.run_inference_on_jsonl(
        input_jsonl=args.input_jsonl,
        output_file=args.output_file,
        data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
