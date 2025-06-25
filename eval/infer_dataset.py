#!/usr/bin/env python3
"""
Pure inference script for Qwen2.5-VL model with FAST inference settings.

This script ONLY generates raw model responses - no evaluation, no parsing, no metrics.
It saves the raw responses in JSON format for later evaluation by eval_dataset.py.

Role: Pure model inference with FAST inference settings (not training settings)
Output: Raw responses JSON file

Usage:
    python eval/infer_dataset.py \
        --model_path output/checkpoint-XXX \
        --validation_jsonl 521_qwen_val.jsonl \
        --output_file raw_responses.json \
        --max_new_tokens 4096
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path for training components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import training components to ensure exact alignment
from src.config.base import Config
from src.models.wrapper import ModelWrapper
from src.utils import (
    DEFAULT_BASE_MODEL_PATH,
    UnifiedLogger,
    extract_prompts_from_conversation,
    find_ground_truth_response,
    load_jsonl,
)


class FastInferenceEngine:
    """
    Fast inference engine for raw response generation.

    Role: ONLY generate raw model responses using FAST inference settings
    - No evaluation logic
    - No response parsing
    - No metrics calculation
    - Just pure model inference with SPEED optimizations

    Key differences from training:
    - use_cache=True (FAST inference, not training)
    - Flash attention properly enabled
    - Optimized for speed, not training alignment
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: str = DEFAULT_BASE_MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 2048,  # Increased to allow longer, more detailed responses
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.max_new_tokens = max_new_tokens

        print(f"🚀 Loading model with FAST inference settings from {model_path}")
        print(f"📝 Max new tokens: {max_new_tokens}")

        # Create training config to ensure exact alignment
        self.config = Config()
        self.config.model_path = model_path
        self.config.model_max_length = 8192  # Allow longer sequences for inference
        self.config.max_pixels = 1003520  # Same as training
        self.config.min_pixels = 784  # Same as training

        # Create logger (inference mode)
        self.logger = UnifiedLogger(
            log_dir="logs",
            verbose=True,
            log_level=20,  # INFO
            is_training=False,  # This is inference
        )

        # Use training's ModelWrapper for exact alignment
        self.model_wrapper = ModelWrapper(self.config, self.logger)

        # Load model, tokenizer, and image processor using training method
        self.model, self.tokenizer, self.image_processor = self.model_wrapper.load_all()

        # CRITICAL: Match training config use_cache=False
        self.model.config.use_cache = False
        print("🚀 TRAINING-ALIGNED INFERENCE: use_cache=False (matches training)")

        # Move to device if specified
        if device != "auto":
            self.model = self.model.to(device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda:0")

        # Set model to eval mode for inference
        self.model.eval()

        print("✅ Model loaded with FAST inference settings")
        print(
            f"   - Flash attention: {getattr(self.model.config, 'attn_implementation', getattr(self.model.config, '_attn_implementation', 'unknown'))}"
        )
        print(f"   - Torch dtype: {self.model.dtype}")
        print(f"   - Use cache: {self.model.config.use_cache} (FAST inference)")
        print(f"   - Model max length: {self.tokenizer.model_max_length}")
        print(f"   - Max pixels: {self.image_processor.max_pixels}")
        print(f"   - Min pixels: {self.image_processor.min_pixels}")

    def _post_process_output(self, output_text: str) -> str:
        """
        Post-process output to remove repetition and detect proper completion.

        Args:
            output_text: Raw model output

        Returns:
            Cleaned output text
        """
        # Remove leading/trailing whitespace
        output_text = output_text.strip()

        # Check for character-level repetition (like repeated exclamation marks)
        if self._has_character_repetition(output_text):
            print("⚠️ Detected character repetition in output, attempting to clean...")
            # Try to find any valid JSON at the beginning
            json_match = self._extract_first_json(output_text)
            if json_match:
                return json_match
            else:
                # If no JSON found, return empty string to indicate failure
                return ""

        # If output starts with JSON array, try to find proper completion
        if output_text.startswith("["):
            # Find the first complete JSON array
            bracket_count = 0
            end_pos = -1

            for i, char in enumerate(output_text):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                # Extract the complete JSON array
                json_part = output_text[:end_pos]

                # Only clean if there's obvious repetition
                if self._has_obvious_repetition(json_part):
                    return self._remove_repetitive_objects(json_part)
                else:
                    return json_part

        # For non-JSON, just return as-is (minimal processing)
        return output_text

    def _has_character_repetition(self, text: str) -> bool:
        """Check if text has excessive character repetition."""
        if len(text) < 10:
            return False

        # Check for repeated characters (like !!!!!!)
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # If any single character makes up more than 80% of the text, it's repetitive
        for char, count in char_counts.items():
            if count / len(text) > 0.8:
                return True

        # Check for repeated patterns
        text_sample = text[:200]  # Check first 200 chars
        for pattern_len in [1, 2, 3, 4, 5]:
            if len(text_sample) >= pattern_len * 10:
                pattern = text_sample[:pattern_len]
                repeated_pattern = pattern * (len(text_sample) // pattern_len)
                if text_sample.startswith(repeated_pattern[: len(text_sample)]):
                    return True

        return False

    def _extract_first_json(self, text: str) -> str:
        """Try to extract the first valid JSON array from text."""
        import re

        # Look for JSON array pattern
        json_pattern = r"\[.*?\]"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                import json

                # Try to parse as JSON
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        return ""

    def _has_obvious_repetition(self, json_text: str) -> bool:
        """Check if JSON has obvious repetition (same object repeated many times)."""
        try:
            import json

            data = json.loads(json_text)

            if isinstance(data, list) and len(data) > 5:
                # Check for identical consecutive objects
                for i in range(len(data) - 1):
                    if data[i] == data[i + 1]:
                        return True

                # Check if more than 30% of objects are identical
                if len(data) > 10:
                    unique_objects = []
                    for obj in data:
                        if obj not in unique_objects:
                            unique_objects.append(obj)

                    if len(unique_objects) / len(data) < 0.7:
                        return True

            return False

        except (json.JSONDecodeError, TypeError):
            return False

    def _is_reasonable_json(self, json_text: str) -> bool:
        """Check if JSON text is reasonable (not too repetitive)."""
        try:
            import json

            data = json.loads(json_text)

            if isinstance(data, list) and len(data) > 0:
                # Check for excessive repetition
                if len(data) > 50:  # Too many objects
                    return False

                # Check for identical objects
                seen_objects = set()
                for obj in data:
                    if isinstance(obj, dict) and "bbox_2d" in obj:
                        # Create a signature for the object
                        bbox = (
                            tuple(obj["bbox_2d"])
                            if isinstance(obj["bbox_2d"], list)
                            else None
                        )
                        desc = obj.get("description", "")
                        signature = (bbox, desc)

                        if signature in seen_objects:
                            return False  # Found duplicate
                        seen_objects.add(signature)

                return True

        except (json.JSONDecodeError, TypeError):
            pass

        return False

    def _remove_repetitive_objects(self, json_text: str) -> str:
        """Remove repetitive objects from JSON array."""
        try:
            import json

            data = json.loads(json_text)

            if isinstance(data, list):
                unique_objects = []
                seen_signatures = set()

                for obj in data:
                    if isinstance(obj, dict) and "bbox_2d" in obj:
                        # Create signature
                        bbox = (
                            tuple(obj["bbox_2d"])
                            if isinstance(obj["bbox_2d"], list)
                            else None
                        )
                        desc = obj.get("description", "")
                        signature = (bbox, desc)

                        if signature not in seen_signatures:
                            unique_objects.append(obj)
                            seen_signatures.add(signature)

                        # Limit to reasonable number
                        if len(unique_objects) >= 20:
                            break

                return json.dumps(unique_objects, ensure_ascii=False)

        except (json.JSONDecodeError, TypeError):
            pass

        return json_text

    def _is_repetitive_line(self, line: str, existing_lines: list) -> bool:
        """Check if a line is repetitive compared to existing lines."""
        for existing in existing_lines[-3:]:  # Check last 3 lines
            if line == existing:
                return True
            # Check for high similarity
            if len(line) > 20 and len(existing) > 20:
                # Simple similarity check
                common_chars = sum(1 for a, b in zip(line, existing) if a == b)
                similarity = common_chars / max(len(line), len(existing))
                if similarity > 0.8:
                    return True
        return False

    def generate_response(
        self, image_paths: list, system_prompt: str, user_prompt: str
    ) -> Tuple[str, Dict]:
        """
        Generate raw response using EXACTLY the same preprocessing as training.

        Supports both single-image and multi-image inputs.

        Role: Pure inference - no evaluation logic

        Returns:
            Tuple of (raw_response_text, metadata)
        """
        try:
            # Handle both single image path (string) and multiple image paths (list)
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            # Count <image> tokens in user prompt
            image_token_count = user_prompt.count("<image>")

            print(
                f"🖼️ Processing {len(image_paths)} images for {image_token_count} <image> tokens"
            )

            # Load and preprocess all images EXACTLY like training
            all_pixel_values = []
            all_grid_thw = []
            all_vision_tokens = []

            for i, image_path in enumerate(image_paths):
                if i >= image_token_count:
                    # Don't process more images than <image> tokens
                    break

                image = Image.open(image_path).convert("RGB")

                # Use training's image processor with exact settings
                visual_processed = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )
                image_tensor = visual_processed["pixel_values"]
                if isinstance(image_tensor, list):
                    image_tensor = image_tensor[0]

                grid_thw = visual_processed["image_grid_thw"][0]

                all_pixel_values.append(image_tensor)
                all_grid_thw.append(grid_thw)

                # Calculate vision tokens EXACTLY like training
                grid_thw_merged = grid_thw.prod() // self.image_processor.merge_size**2
                vision_tokens = (
                    "<|vision_start|>"
                    + "<|image_pad|>" * grid_thw_merged.item()
                    + "<|vision_end|>"
                )
                all_vision_tokens.append(vision_tokens)

            # Replace <image> tokens with vision tokens EXACTLY like training
            user_prompt_processed = user_prompt
            for i, vision_tokens in enumerate(all_vision_tokens):
                # Replace one <image> token at a time
                user_prompt_processed = user_prompt_processed.replace(
                    "<image>", vision_tokens, 1
                )

            # Handle pixel values and grid_thw EXACTLY like training
            if all_pixel_values:
                if len(all_pixel_values) == 1:
                    # Single image - keep original format like training
                    pixel_values = all_pixel_values[0]
                    image_grid_thw = all_grid_thw[0].unsqueeze(0)
                else:
                    # Multiple images - concatenate like training
                    pixel_values = torch.cat(all_pixel_values, dim=0)
                    image_grid_thw = torch.stack(all_grid_thw, dim=0)
            else:
                # Fallback: use first image if no <image> tokens found
                image = Image.open(image_paths[0]).convert("RGB")
                visual_processed = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )
                pixel_values = visual_processed["pixel_values"]
                if isinstance(pixel_values, list):
                    pixel_values = pixel_values[0]
                image_grid_thw = visual_processed["image_grid_thw"]
                user_prompt_processed = user_prompt

            # Create conversation EXACTLY like training (with double system message)
            # Training uses: 1) "You are a helpful assistant." 2) actual system prompt
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_processed},
            ]

            # Apply chat template EXACTLY like training
            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # Create attention mask EXACTLY like training
            attention_mask = torch.ones_like(input_ids)

            # Prepare model inputs EXACTLY like training
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with CONSERVATIVE parameters (stable and anti-repetition)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    # CONSERVATIVE INFERENCE SETTINGS (stable and anti-repetition):
                    do_sample=False,  # Deterministic for stability
                    temperature=None,  # No temperature (deterministic)
                    top_p=None,  # No nucleus sampling
                    top_k=None,  # No top-k sampling
                    num_beams=1,  # No beam search (faster)
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # CRITICAL: Match training config
                    repetition_penalty=1.15,  # Strong repetition prevention
                    length_penalty=1.0,  # Neutral length penalty
                    early_stopping=False,  # Let model complete naturally
                    min_new_tokens=10,  # Minimum reasonable output
                    no_repeat_ngram_size=4,  # Prevent 4-gram repetition
                )

            # Decode generated tokens
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]

            output_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            # Post-process to remove repetition and detect completion
            output_text = self._post_process_output(output_text)

            # Create metadata (for debugging, not evaluation)
            metadata = {
                "input_tokens": input_ids.size(1),
                "output_tokens": output_ids.size(1),
                "generated_tokens": output_ids.size(1) - input_ids.size(1),
                "num_images": len(all_pixel_values) if all_pixel_values else 1,
                "image_token_count": image_token_count,
                "grid_thw": image_grid_thw.tolist(),
                "merge_size": self.image_processor.merge_size,
                "max_new_tokens": self.max_new_tokens,
                "generation_config": {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                    "num_beams": 1,
                    "use_cache": False,  # Match training config
                    "repetition_penalty": 1.15,
                    "length_penalty": 1.0,
                    "early_stopping": False,
                    "no_repeat_ngram_size": 4,
                },
            }

            # Clean up GPU memory
            if torch.cuda.is_available():
                del output_ids, generated_ids
                torch.cuda.empty_cache()

            return output_text, metadata

        except Exception as e:
            print(f"❌ Error generating response for {image_path}: {e}")
            return "", {"error": str(e)}

    def process_dataset(
        self, validation_jsonl: str, output_file: str, max_samples: int
    ) -> None:
        """
        Process entire validation dataset and save raw responses.

        Role: Pure inference - no evaluation, just raw response generation

        Args:
            validation_jsonl: Path to validation JSONL file
            output_file: Path to save raw responses
            max_samples: Maximum number of samples to process
        """
        print(f"📊 Loading validation data from {validation_jsonl}")
        samples = load_jsonl(validation_jsonl)

        if max_samples != -1:
            samples = samples[:max_samples]
            print(f"🔢 Processing {max_samples} samples (limited)")
        else:
            print(f"🔢 Processing {len(samples)} samples")

        results = []
        skipped_count = 0
        error_count = 0

        # Create progress bar
        progress_bar = tqdm(
            enumerate(samples),
            total=len(samples),
            desc="🔄 Generating responses",
            unit="sample",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for i, sample in progress_bar:
            try:
                # Extract prompts and ground truth (for saving, not evaluation)
                conversations = sample.get("conversations", [])

                # Handle both unified format (images array) and legacy format (image string)
                image_paths = []
                if "images" in sample and sample["images"]:
                    # Unified format: use all images for multi-image inference
                    image_paths = sample["images"]
                elif "image" in sample:
                    # Legacy format: single image
                    image_paths = [sample["image"]]

                if not conversations or not image_paths:
                    skipped_count += 1
                    progress_bar.set_postfix(
                        {
                            "✅": len(results),
                            "⚠️": skipped_count,
                            "❌": error_count,
                            "Current": os.path.basename(image_paths[0])
                            if image_paths
                            else "unknown",
                        }
                    )
                    continue

                system_prompt, user_prompt = extract_prompts_from_conversation(
                    conversations
                )
                ground_truth = find_ground_truth_response(conversations)

                if not user_prompt or not ground_truth:
                    skipped_count += 1
                    progress_bar.set_postfix(
                        {
                            "✅": len(results),
                            "⚠️": skipped_count,
                            "❌": error_count,
                            "Current": os.path.basename(image_paths[0])
                            if image_paths
                            else "unknown",
                        }
                    )
                    continue

                # Generate raw response (pure inference)
                raw_response, metadata = self.generate_response(
                    image_paths, system_prompt, user_prompt
                )

                # Save raw result (no evaluation here)
                result = {
                    "sample_id": i,
                    "image_paths": image_paths,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "ground_truth": ground_truth,
                    "prediction": raw_response,  # Raw model output
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # Update progress bar with current stats
                progress_bar.set_postfix(
                    {
                        "✅": len(results),
                        "⚠️": skipped_count,
                        "❌": error_count,
                        "Current": os.path.basename(image_paths[0])
                        if image_paths
                        else "unknown",
                    }
                )

            except Exception as e:
                error_count += 1
                progress_bar.set_postfix(
                    {
                        "✅": len(results),
                        "⚠️": skipped_count,
                        "❌": error_count,
                        "Error": str(e)[:20] + "..." if len(str(e)) > 20 else str(e),
                    }
                )
                continue

        progress_bar.close()

        # Save raw results (no metrics calculation here)
        print(f"💾 Saving {len(results)} raw responses to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ FAST inference completed! Raw responses saved to {output_file}")
        print(f"📈 Successfully processed: {len(results)}/{len(samples)} samples")
        print(f"⚠️  Skipped: {skipped_count} samples")
        print(f"❌ Errors: {error_count} samples")
        print(
            "📄 Next step: Use eval_dataset.py to parse responses and calculate metrics"
        )


def main():
    """Main inference function - pure response generation only."""
    parser = argparse.ArgumentParser(
        description="FAST inference script for Qwen2.5-VL model (optimized for speed)"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--validation_jsonl",
        type=str,
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save raw responses JSON file",
    )

    # Optional arguments
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Path to base model for processor/tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,  # Increased to allow longer, more detailed responses
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    if not os.path.exists(args.validation_jsonl):
        raise FileNotFoundError(
            f"Validation file does not exist: {args.validation_jsonl}"
        )

    print("🎯 FAST INFERENCE MODE - Optimized for speed:")
    print("   - Model loading: flash_attention_2, torch.bfloat16, use_cache=TRUE")
    print("   - Tokenizer: padding_side='right', use_fast=False")
    print("   - Image processor: max_pixels=1003520, min_pixels=784")
    print("   - Generation: do_sample=False, use_cache=TRUE (FAST)")
    print(f"   - Max new tokens: {args.max_new_tokens}")
    print("   - Role: ONLY generate raw responses (no evaluation)")

    # Initialize fast inference engine
    engine = FastInferenceEngine(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    # Process dataset (pure inference only)
    engine.process_dataset(
        validation_jsonl=args.validation_jsonl,
        output_file=args.output_file,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
