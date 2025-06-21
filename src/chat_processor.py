#!/usr/bin/env python3
"""
Unified Chat Processor for Qwen2.5-VL BBU Dataset

This module provides a clean, single-purpose processor for the simplified JSONL format:
- Loads JSONL with 'examples' and 'target' structure
- Creates proper few-shot chat templates
- Uses pure JSON format for object detection (compatible with Qwen2.5-VL)
- Manages vision token expansion
- Prepares data for training with proper masking

Pipeline: Simplified JSONL → ChatProcessor → Training-ready samples
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import config
from src.logger_utils import get_chat_logger
from src.prompt import (
    CHINESE_BASE_PROMPT,
    CHINESE_CANDIDATES_SECTION,
    CHINESE_FEW_SHOT_SECTION,
    ENGLISH_BASE_PROMPT,
    ENGLISH_CANDIDATES_SECTION,
    ENGLISH_FEW_SHOT_SECTION,
)
from src.rope2d import get_rope_index_25  # NEW: explicit RoPE position ids
from src.schema import assert_chat_processor_output
from src.tokens import SpecialTokens

logger = get_chat_logger()


class ChatProcessor:
    """
    Single-purpose processor for BBU dataset chat template creation.

    Handles the complete pipeline from simplified JSONL to training-ready format.
    Uses pure JSON format for object detection output (Qwen2.5-VL compatible).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Qwen2VLImageProcessor,
    ) -> None:
        """
        Initialize the chat processor using global configuration.

        Args:
            tokenizer: Qwen2.5-VL tokenizer
            image_processor: Qwen2.5-VL image processor
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # Store data root from global config
        self.data_root = Path(config.data_root)

        # Initialize special tokens (only for vision tokens, not for object detection)
        self.tokens = SpecialTokens()

        # Get language from global config
        self.language = config.language

        # Build system prompt using global config
        self.system_prompt = self._build_system_prompt()

        # Log configuration
        logger.info(f"✅ ChatProcessor initialized:")
        logger.info(f"   Language: {self.language}")
        logger.info(f"   Data root: {config.data_root}")
        logger.info(f"   Model max length: {config.max_total_length}")
        logger.info(f"   Use candidates: {config.use_candidates}")
        logger.info(f"   Output format: Pure JSON (Qwen2.5-VL compatible)")

        # Log a sample of the system prompt
        logger.info(f"📄 System prompt sample (first 500 chars):")
        logger.info(f"   {repr(self.system_prompt[:500])}")
        logger.info(f"📄 System prompt sample (last 200 chars):")
        logger.info(f"   {repr(self.system_prompt[-200:])}")

    def _build_system_prompt(self) -> str:
        """Build system prompt with pure JSON format for object detection."""

        if self.language == "chinese":
            base_prompt = CHINESE_BASE_PROMPT
            candidates_template = CHINESE_CANDIDATES_SECTION
            few_shot_section = CHINESE_FEW_SHOT_SECTION
        else:  # English
            base_prompt = ENGLISH_BASE_PROMPT
            candidates_template = ENGLISH_CANDIDATES_SECTION
            few_shot_section = ENGLISH_FEW_SHOT_SECTION

        # Add candidate phrases if provided
        if config.use_candidates and config.candidates_file:
            candidates_path = Path(config.candidates_file)
            if candidates_path.exists():
                with open(candidates_path, "r", encoding="utf-8") as f:
                    candidates_data = json.load(f)

                phrase_list = candidates_data.get("phrase_list", [])

                # Format as numbered list for better readability
                formatted_phrases = "\n".join(
                    [f"{i + 1}) {phrase}" for i, phrase in enumerate(phrase_list)]
                )

                # Format the template with the phrases
                candidates_section = candidates_template.format(
                    formatted_phrases=formatted_phrases
                )
                base_prompt += candidates_section
            else:
                logger.warning(f"Candidates file not found: {config.candidates_file}")

        return base_prompt + few_shot_section

    def process_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single sample from simplified JSONL format.

        Args:
            raw_sample: Sample with 'examples' and 'target' structure

        Returns:
            Dict containing input_ids, labels, attention_mask, pixel_values, image_grid_thw, and ground_truth_objects
        """
        # 1. Create conversation messages
        conversation_messages = self._create_conversation_messages(raw_sample)

        # 2. Process images, expand vision tokens, and get image dimensions
        (
            processed_conversation,
            images,
            image_dims,
        ) = self._process_images_and_tokens(
            conversation_messages, self._extract_all_image_paths(raw_sample)
        )

        # 3. Apply chat template and tokenize
        input_ids, labels = self._tokenize_conversation(processed_conversation)

        # 4. Process images for model input
        pixel_values, image_grid_thw = self._process_images_for_model(images)

        # 5. Extract and normalize ground truth objects for the target image
        ground_truth_objects = self._extract_and_normalize_ground_truth(
            raw_sample, image_dims
        )

        # NEW 5.1 Compute explicit 3-D RoPE position_ids (batch dim=1)
        attention_mask_single = torch.ones_like(input_ids).unsqueeze(0)  # (1, S)
        # get_rope_index_* expects (batch, seq) for input_ids
        position_ids, _ = get_rope_index_25(
            spatial_merge_size=self.image_processor.merge_size,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask_single,
        )
        # Keep returned shape (3, 1, S) – collator will pad/concat later

        sample_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": position_ids,  # <-- added
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ground_truth_objects": ground_truth_objects,
        }

        # Fail-fast shape validation (raises AssertionError on mismatch)
        assert_chat_processor_output(sample_dict)

        return sample_dict

    def _create_conversation_messages(
        self, sample: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create conversation messages from sample data."""
        messages = []

        # Add system message
        messages.append({"role": "system", "content": self.system_prompt})

        # Process examples if present
        if "examples" in sample:
            for example in sample["examples"]:
                # User message with image
                user_content = "<image>"
                messages.append({"role": "user", "content": user_content})

                # Assistant response with JSON format
                objects = example.get("objects", [])
                sorted_objects = self._sort_objects_by_position(objects)
                assistant_response = self._format_objects_response(sorted_objects)
                messages.append({"role": "assistant", "content": assistant_response})

        # Add target
        target = sample.get("target", sample)  # Fallback to sample itself if no target
        user_content = "<image>"
        messages.append({"role": "user", "content": user_content})

        # Target response
        target_objects = target.get("objects", [])
        sorted_target_objects = self._sort_objects_by_position(target_objects)
        target_response = self._format_objects_response(sorted_target_objects)
        messages.append({"role": "assistant", "content": target_response})

        return messages

    def _extract_all_image_paths(self, sample: Dict[str, Any]) -> List[str]:
        """Extract all image paths from sample."""
        image_paths = []

        # Extract from examples
        if "examples" in sample:
            for example in sample["examples"]:
                image_paths.extend(example.get("images", []))

        # Extract from target
        target = sample.get("target", sample)
        image_paths.extend(target.get("images", []))

        return image_paths

    def _sort_objects_by_position(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort objects by position (top-to-bottom, left-to-right)."""

        def sort_key(obj):
            box = obj.get("box", [0, 0, 0, 0])
            return (box[1], box[0])  # Sort by y first, then x

        return sorted(objects, key=sort_key)

    def _format_objects_response(self, objects: List[Dict[str, Any]]) -> str:
        """Format objects into JSON array format."""
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            box = obj.get("box", [0, 0, 0, 0])
            desc = obj.get("desc", "unknown")

            # Create JSON object
            json_obj = {"bbox_2d": box, "label": desc}
            json_objects.append(json_obj)

        # Return formatted JSON array
        return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))

    def _process_images_and_tokens(
        self, conversation: List[Dict[str, str]], image_paths: List[str]
    ) -> Tuple[List[Dict[str, str]], List[Image.Image], List[Tuple[int, int]]]:
        """
        Process images and expand vision tokens in conversation.

        Replaces <image> placeholders with proper vision token sequences.
        """
        processed_conversation = []
        images = []
        image_dims = []
        image_index = 0

        for message in conversation:
            content = message["content"]

            # Process image placeholders
            while "<image>" in content and image_index < len(image_paths):
                # Load image
                image_path = self.data_root / image_paths[image_index]
                # Fail-fast image loading; errors will raise
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                image_dims.append(image.size)  # (width, height)
                image_index += 1

                # Calculate number of vision tokens required for this image
                num_vision_tokens = self._calculate_image_tokens(image)

                # Create vision token sequence
                vision_token_sequence = (
                    self.tokens.VISION_START
                    + self.tokens.IMAGE_PAD * num_vision_tokens
                    + self.tokens.VISION_END
                )

                # Replace placeholder with vision tokens
                content = content.replace("<image>", vision_token_sequence, 1)

            processed_conversation.append({"role": message["role"], "content": content})

        # Final validation
        if image_index != len(image_paths):
            logger.warning(
                f"Mismatch between image placeholders and image paths. "
                f"Found {image_index} placeholders, but {len(image_paths)} images."
            )

        return processed_conversation, images, image_dims

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Return the number of <|image_pad|> tokens required for ``image``.

        This replicates the official Qwen2-VL preprocessing logic *exactly* but
        in pure-python so we do **not** need to call ``self.image_processor`` or
        allocate any dummy tensors.

        Steps
        -----
        1.  Compute the *resized* image resolution using the same `smart_resize`
            rule as the upstream `Qwen2VLImageProcessor` – this guarantees we
            match the vision encoder.
        2.  Convert the final resolution to a spatial patch grid
            ``(grid_h, grid_w)`` where each patch covers ``patch_size`` pixels.
        3.  Account for the vision->LLM spatial merge layer (``merge_size``).
        4.  Return the **post-merge** token count that the text prompt must
            reserve using ``<|image_pad|>`` placeholders.
        """
        # Explicit parameter extraction – all attributes must exist on the
        # `image_processor`. Fail fast if the pipeline is mis-configured.
        required_attrs = [
            ("patch_size", int),
            ("merge_size", int),
            ("min_pixels", int),
            ("max_pixels", int),
        ]

        for attr_name, attr_type in required_attrs:
            if not hasattr(self.image_processor, attr_name):
                raise AttributeError(
                    f"image_processor is missing required attribute '{attr_name}'. "
                    "Ensure it is correctly initialised with all hyper-parameters."
                )

        patch_size: int = self.image_processor.patch_size  # type: ignore[attr-defined]
        merge_size: int = self.image_processor.merge_size  # type: ignore[attr-defined]
        min_pixels: int = self.image_processor.min_pixels  # type: ignore[attr-defined]
        max_pixels: int = self.image_processor.max_pixels  # type: ignore[attr-defined]

        # Import the *exact* smart_resize helper used by the official recipe.
        # This is safe and avoids code duplication while keeping full parity.
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        # 1)  Determine the post-resize resolution (guaranteed to be divisible
        #     by `patch_size * merge_size`).
        resized_h, resized_w = smart_resize(
            image.height,
            image.width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # 2)  Spatial patch grid (pre-merge)
        grid_h = resized_h // patch_size
        grid_w = resized_w // patch_size

        # 3)  Account for the spatial merge that groups `merge_size²` patches
        #     into **one** LLM token.
        num_tokens = (grid_h * grid_w) // (merge_size**2)

        # Sanity check – must be >0 for any valid image.
        if num_tokens <= 0:
            raise ValueError(
                f"Calculated non-positive vision token count ({num_tokens}) for image size "
                f"{image.size}. Check smart_resize / patch_size configuration."
            )

        return num_tokens

    def _extract_and_normalize_ground_truth(
        self, sample: Dict[str, Any], image_dims: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts GT objects from the target and normalizes their bounding boxes.
        """
        target = sample.get("target", sample)
        target_objects = target.get("objects", [])

        # The last image in the list corresponds to the target.
        if not image_dims or not target_objects:
            return []

        target_image_dims = image_dims[-1]
        width, height = target_image_dims

        normalized_objects = []
        for obj in target_objects:
            box = obj.get("box")
            desc = obj.get("desc")

            if box is None or desc is None:
                continue

            # Validate box format and coordinates
            if not (isinstance(box, list) and len(box) == 4) or not (
                0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            ):
                logger.error(
                    f"Invalid or out-of-bounds box: {box} for image size {width}x{height}."
                )
                raise RuntimeError(
                    f"Invalid or out-of-bounds box: {box} for image size {width}x{height}."
                )

            normalized_box = [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ]
            normalized_objects.append({"box": normalized_box, "desc": desc})

        return normalized_objects

    def _tokenize_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert `conversation` to `input_ids` & `labels`.

        *Implementation notes*
        ---------------------------------
        • We call ``apply_chat_template`` **once** on the *entire* message list
          to avoid the tokenizer injecting its fallback system prompt
          ("You are a helpful assistant.") before every user/assistant turn.

        • To keep full control, we temporarily replace the tokenizer's
          ``chat_template`` with a *minimal* variant lifted from the official
          finetuning script.  No default system prompt, no auto generation
          marker – just a linear dump of messages.
        """

        IGNORE_INDEX = -100

        # ------------------------------------------------------------------
        # 1)  Temporarily swap the chat template
        # ------------------------------------------------------------------
        original_template = getattr(self.tokenizer, "chat_template", None)

        minimal_template = (
            "{% for message in messages %}"
            "{{'<' + '|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

        self.tokenizer.chat_template = minimal_template  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # 2)  Encode full conversation once (no labels yet)
        # ------------------------------------------------------------------
        full_ids: List[int] = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )

        # ------------------------------------------------------------------
        # 3)  Build label mask per-message – re-use the same minimal template
        # ------------------------------------------------------------------
        labels: List[int] = []
        for msg in conversation:
            # Tokenise single message *with the minimal template* so the
            # length exactly matches its slice inside ``full_ids``.
            msg_ids: List[int] = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False
            )

            if msg["role"] in {"user", "system"}:
                labels.extend([IGNORE_INDEX] * len(msg_ids))
            else:
                msg_labels = msg_ids.copy()
                if len(msg_labels) >= 3:
                    msg_labels[:3] = [IGNORE_INDEX] * 3
                labels.extend(msg_labels)

        # ------------------------------------------------------------------
        # 4)  Restore original template and convert to tensors
        # ------------------------------------------------------------------
        if original_template is not None:
            self.tokenizer.chat_template = original_template  # type: ignore[attr-defined]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        label_ids = torch.tensor(labels, dtype=torch.long)

        assert input_ids.shape == label_ids.shape, (
            "Input/label length mismatch after tokenisation."  # noqa: E501
        )

        return input_ids, label_ids

    def _process_images_for_model(
        self, images: List[Image.Image]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process images for model input using official approach with data_conversion pixel settings."""
        if not images:
            return None, None

        # Log original image sizes for debugging
        logger.debug(f"🖼️ PROCESSING {len(images)} IMAGES:")
        for i, img in enumerate(images):
            logger.debug(f"   Image {i}: {img.size} (W*H), mode={img.mode}")

        # Use processor directly like official QwenVL implementation
        # The processor is already configured with data_conversion/vision_process.py values
        processed = self.image_processor.preprocess(images, return_tensors="pt")

        pixel_values = processed["pixel_values"]
        image_grid_thw_np = processed.get("image_grid_thw")

        # Convert to torch.LongTensor immediately to avoid numpy propagation
        image_grid_thw: Optional[torch.Tensor]
        if image_grid_thw_np is not None:
            image_grid_thw = torch.as_tensor(image_grid_thw_np, dtype=torch.long)
        else:
            image_grid_thw = None

        # CRITICAL: Log the vision token analysis
        logger.debug(f"🚨 VISION TOKEN ANALYSIS:")
        logger.debug(f"   Input images: {len(images)} images")
        logger.debug(f"   Original sizes: {[img.size for img in images]}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(f"   Vision tokens generated (pre-merge): {pixel_values.shape[0]}")

        if image_grid_thw is not None:
            logger.debug(f"   image_grid_thw shape: {image_grid_thw.shape}")
            logger.debug(f"   image_grid_thw values: {image_grid_thw.tolist()}")

            # Calculate both pre-merge and post-merge token counts for clarity
            # EXPLICIT: Get merge_size from image processor - no defaults
            if not hasattr(self.image_processor, "merge_size"):
                raise AttributeError(
                    "image_processor missing required attribute 'merge_size'."
                )
            merge_size = self.image_processor.merge_size  # type: ignore[attr-defined]
            merge_length = merge_size**2

            total_pre_merge = 0
            total_post_merge = 0

            for i, grid in enumerate(image_grid_thw):
                t, h, w = grid.tolist()
                pre_merge_tokens = t * h * w
                post_merge_tokens = pre_merge_tokens // merge_length
                total_pre_merge += pre_merge_tokens
                total_post_merge += post_merge_tokens

                logger.debug(
                    f"   Image {i}: grid=({t},{h},{w}) → {pre_merge_tokens} pre-merge → {post_merge_tokens} final tokens"
                )

            logger.debug(
                f"   TOTAL: {total_pre_merge} pre-merge → {total_post_merge} final tokens (merge_size={merge_size}²={merge_length})"
            )
            logger.debug(
                f"   Vision token efficiency: {total_post_merge}/{total_pre_merge} = {total_post_merge / total_pre_merge:.1%}"
            )
        else:
            logger.debug(f"   No image_grid_thw available")

        # CRITICAL: Ensure bf16 precision for pixel_values
        # Removed forced cast to bf16 – rely on model.config.torch_dtype instead
        logger.debug(f"🔍 pixel_values dtype: {pixel_values.dtype} (no manual cast)")

        # Debug: Log the shapes to understand the processing
        logger.debug(f"🖼️ OFFICIAL IMAGE PROCESSING:")
        logger.debug(f"   Number of images: {len(images)}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(
            f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        # Fail-fast: ensure pixel_values patches match grid_thw pre-merge count
        if image_grid_thw is not None:
            # pre-merge patches per image = t*h*w
            grid = image_grid_thw  # Tensor[N_images, 3]
            if not torch.is_tensor(grid):
                grid = torch.as_tensor(grid, dtype=torch.long)
            pre_merge_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item()
            actual_patches = pixel_values.shape[0]
            assert actual_patches == pre_merge_counts, (
                f"Mismatch in image preprocessing: pixel_values has {actual_patches} patches, "
                f"but grid_thw implies {pre_merge_counts} patches. Check merge_size and smart_resize logic."
            )

        return pixel_values, image_grid_thw

    def prepare_inputs_for_inference(
        self, images: List[Image.Image], text: str, is_first_step: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for inference following the official Qwen2.5-VL pattern.

        Args:
            images: List of PIL images
            text: Input text prompt
            is_first_step: Whether this is the first generation step (prefill)

        Returns:
            Dict containing properly formatted inputs for model.generate()
        """
        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=False)

        # Prepare base inputs
        model_inputs = {
            "input_ids": text_inputs["input_ids"],
        }

        # Add attention_mask only if it exists
        if (
            "attention_mask" in text_inputs
            and text_inputs["attention_mask"] is not None
        ):
            model_inputs["attention_mask"] = text_inputs["attention_mask"]

        # Handle vision inputs based on generation step
        if is_first_step and images:
            # First step: process images normally
            pixel_values, image_grid_thw = self._process_images_for_model(images)

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values

            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

            # --- Compute 3-D position_ids for generation prefill ---------
            ids_for_rope = model_inputs["input_ids"].clone()
            attn_mask_for_rope = model_inputs.get("attention_mask")
            if attn_mask_for_rope is None:
                attn_mask_for_rope = torch.ones_like(ids_for_rope)

            # get_rope_index_25 returns (3,B,S)
            pos_ids, _ = get_rope_index_25(
                spatial_merge_size=self.image_processor.merge_size,
                input_ids=ids_for_rope,
                image_grid_thw=image_grid_thw,
                attention_mask=attn_mask_for_rope,
            )
            model_inputs["position_ids"] = pos_ids

            logger.debug(f"🔥 FIRST STEP: Added vision inputs")
            logger.debug(
                f"   pixel_values shape: {pixel_values.shape if pixel_values is not None else None}"
            )
            logger.debug(
                f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
            )
        else:
            # Subsequent steps: don't add vision inputs at all
            # The model's prepare_inputs_for_generation will handle this
            logger.debug(f"🔥 SUBSEQUENT STEP: No vision inputs added")

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
            for key, value in model_inputs.items()
            if key in valid_generation_params and value is not None
        }

        logger.debug(f"🔧 FILTERED INFERENCE INPUTS: {list(filtered_inputs.keys())}")

        # ------------------------------------------------------------------
        # FAIL-FAST CHECK ‑ ensure prompt & vision tensors stay in perfect sync
        # ------------------------------------------------------------------
        if "pixel_values" in filtered_inputs:
            from src.tokens.special_tokens import SpecialTokens

            # --------------------------------------------------------------
            # Reconstruct the *post-merge* vision token count from the
            # supplied `image_grid_thw` so we compare like-for-like.
            # --------------------------------------------------------------
            if "image_grid_thw" not in filtered_inputs:
                raise ValueError(
                    "prepare_inputs_for_inference received `pixel_values` but "
                    "no corresponding `image_grid_thw`. Cannot validate vision/text alignment."
                )

            image_grid_thw = filtered_inputs["image_grid_thw"]  # (N, 3)

            if not torch.is_tensor(image_grid_thw):
                image_grid_thw = torch.as_tensor(image_grid_thw)

            if not hasattr(self.image_processor, "merge_size"):
                raise AttributeError(
                    "image_processor missing required attribute 'merge_size'."
                )
            merge_size = self.image_processor.merge_size  # type: ignore[attr-defined]
            merge_length = merge_size**2

            num_vision_tokens = int(
                (image_grid_thw.prod(dim=1) // merge_length).sum().item()
            )
            num_prompt_tokens = text.count(SpecialTokens.IMAGE_PAD)

            assert num_vision_tokens == num_prompt_tokens, (
                "Mismatch between vision tensors and <|image_pad|> tokens. "
                f"Vision tokens (post-merge): {num_vision_tokens} ≠ image_pad tokens: {num_prompt_tokens}. "
                "Check _calculate_image_tokens & image preprocessing pipeline."
            )
            # Additional safety: ensure pixel_values tensor length matches pre-merge patch count
            pv_len = filtered_inputs["pixel_values"].shape[0]
            expected_pre_merge = num_vision_tokens * merge_length
            assert pv_len == expected_pre_merge, (
                "pixel_values tensor length does not match expected pre-merge patch count. "
                f"pixel_values patches: {pv_len} ≠ num_vision_tokens ({num_vision_tokens}) × merge_size² ({merge_length}) = {expected_pre_merge}. "
                "Verify image_processor.merge_size and preprocessing alignment."
            )
            logger.debug(
                f"✅ pixel_values length check passed: {pv_len} patches ↔ {expected_pre_merge} expected pre-merge patches"
            )
            logger.debug(
                f"✅ Vision/Text sync check passed: {num_vision_tokens} vision tokens ↔ {num_prompt_tokens} IMAGE_PAD tokens"
            )

        return filtered_inputs


def create_chat_processor(
    tokenizer: PreTrainedTokenizerBase,
    image_processor: Qwen2VLImageProcessor,
    data_root: str = "./",
    model_max_length: int = 8192,
    use_candidates: bool = False,
    candidates_file: Optional[str] = None,
) -> ChatProcessor:
    """
    Factory function to create ChatProcessor.

    Args:
        tokenizer: Qwen2.5-VL tokenizer
        image_processor: Qwen2.5-VL image processor
        data_root: Root directory for image paths
        model_max_length: Maximum sequence length
        use_candidates: Whether to use candidate phrases
        candidates_file: Path to candidate phrases JSON

    Returns:
        Configured ChatProcessor instance
    """
    return ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        model_max_length=model_max_length,
        use_candidates=use_candidates,
        candidates_file=candidates_file,
    )
