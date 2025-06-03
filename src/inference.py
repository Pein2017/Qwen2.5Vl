"""
Unified inference engine that uses the same preprocessing pipeline as training.

This module provides fast, optimized inference using the same components
and optimizations as training, ensuring consistency and performance.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch

from src.config.base import Config
from src.models.wrapper import ModelWrapper

from .preprocessing import create_preprocessor
from .utils import (
    UnifiedLogger,
    extract_prompts_from_conversation,
    find_ground_truth_response,
)


class InferenceEngine:
    """
    Unified inference engine that uses the same preprocessing and optimizations as training.

    Key benefits:
    - Same preprocessing pipeline as training (no inconsistencies)
    - Uses training's flash attention optimizations
    - Supports both single-image and multi-image inputs
    - Fast inference with training-optimized components
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 2048,
        data_root: str = "./",
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.data_root = data_root

        print(f"🚀 Loading unified inference engine from {model_path}")
        print(f"📝 Max new tokens: {max_new_tokens}")

        # Create config for model loading
        self.config = Config()
        self.config.model_path = model_path
        self.config.data_root = data_root

        # Create logger (inference mode)
        self.logger = UnifiedLogger(log_dir="logs", verbose=True, is_training=False)

        # Load model components using training's ModelWrapper
        self.model_wrapper = ModelWrapper(self.config, self.logger)
        self.model, self.tokenizer, self.image_processor = self.model_wrapper.load_all()

        # Create preprocessor (same as training)
        self.preprocessor = create_preprocessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            data_root=data_root,
            model_max_length=self.config.model_max_length,
        )

        # Move to device
        if device != "auto":
            self.model = self.model.to(device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda:0")

        # Set model to eval mode
        self.model.eval()

        print("✅ Unified inference engine loaded")
        print(
            f"   - Flash attention: {getattr(self.model.config, 'attn_implementation', 'unknown')}"
        )
        print(f"   - Torch dtype: {self.model.dtype}")
        print(f"   - Use cache: {self.model.config.use_cache}")
        print(f"   - Model max length: {self.tokenizer.model_max_length}")

    def generate_response(
        self, image_paths: Union[str, List[str]], system_prompt: str, user_prompt: str
    ) -> Tuple[str, Dict]:
        """
        Generate response using unified preprocessing pipeline.

        Args:
            image_paths: Single image path or list of image paths
            system_prompt: System message
            user_prompt: User message with <image> tokens

        Returns:
            Tuple of (generated_text, metadata)
        """
        try:
            # Use unified preprocessor (same as training)
            inputs = self.preprocessor.process_sample_for_inference(
                image_paths=image_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Generate with optimized parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.model.config.use_cache,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    early_stopping=False,
                    min_new_tokens=10,
                    no_repeat_ngram_size=3,
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

            # Create metadata
            metadata = {
                "input_tokens": inputs["input_ids"].size(1),
                "output_tokens": output_ids.size(1),
                "generated_tokens": output_ids.size(1) - inputs["input_ids"].size(1),
                "num_images": len(image_paths) if isinstance(image_paths, list) else 1,
                "max_new_tokens": self.max_new_tokens,
                "generation_config": {
                    "do_sample": False,
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "early_stopping": False,
                    "no_repeat_ngram_size": 3,
                },
            }

            # Clean up GPU memory
            if torch.cuda.is_available():
                del output_ids, generated_ids
                torch.cuda.empty_cache()

            return output_text, metadata

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "", {"error": str(e)}

    def process_dataset(
        self, validation_jsonl: str, output_file: str, max_samples: Optional[int] = None
    ) -> None:
        """
        Process entire validation dataset using unified pipeline.

        Args:
            validation_jsonl: Path to validation JSONL file
            output_file: Path to save results
            max_samples: Maximum number of samples to process
        """
        import json
        import os

        from tqdm import tqdm

        from .utils import load_jsonl

        print(f"📊 Loading validation data from {validation_jsonl}")
        samples = load_jsonl(validation_jsonl)

        if max_samples is not None:
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
        )

        for i, sample in progress_bar:
            try:
                # Extract data using training utilities
                conversations = sample.get("conversations", [])

                # Handle both unified format (images array) and legacy format (image string)
                if "images" in sample and sample["images"]:
                    image_paths = sample["images"]
                elif "image" in sample:
                    image_paths = [sample["image"]]
                else:
                    skipped_count += 1
                    continue

                if not conversations:
                    skipped_count += 1
                    continue

                # Extract prompts using training utilities
                system_prompt, user_prompt = extract_prompts_from_conversation(
                    conversations
                )
                ground_truth = find_ground_truth_response(conversations)

                if not user_prompt or not ground_truth:
                    skipped_count += 1
                    continue

                # Generate response using unified pipeline
                raw_response, metadata = self.generate_response(
                    image_paths, system_prompt, user_prompt
                )

                # Save result
                result = {
                    "sample_id": i,
                    "image_paths": image_paths,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "ground_truth": ground_truth,
                    "prediction": raw_response,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # Update progress
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

        # Save results
        print(f"💾 Saving {len(results)} results to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("✅ Unified inference completed!")
        print(f"📈 Successfully processed: {len(results)}/{len(samples)} samples")
        print(f"⚠️  Skipped: {skipped_count} samples")
        print(f"❌ Errors: {error_count} samples")


def create_inference_engine(
    model_path: str,
    device: str = "auto",
    max_new_tokens: int = 2048,
    data_root: str = "./",
) -> InferenceEngine:
    """
    Factory function to create a unified inference engine.

    Args:
        model_path: Path to the model
        device: Device to load model on
        max_new_tokens: Maximum tokens to generate
        data_root: Root directory for image paths

    Returns:
        UnifiedInferenceEngine instance
    """
    return InferenceEngine(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        data_root=data_root,
    )
