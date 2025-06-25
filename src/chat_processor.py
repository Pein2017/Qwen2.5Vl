import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import torch
from PIL import Image
from torchtyping import TensorType

# Runtime & shape-checking ----------------------------------------------
from typeguard import typechecked

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
from src.schema import ChatMessage, ChatProcessorOutput, GroundTruthObject
from src.tokens import SpecialTokens

logger = get_chat_logger()

# Dimensional symbols for torchtyping ----------------------------------
S = TypeVar("S")  # Sequence length
B = TypeVar("B")  # Batch size
C_TOK = TypeVar("C_TOK")  # Channels (avoid single ambiguous)
PT = TypeVar("PT")  # Flattened patch tokens count (replaces ambiguous I)
E = TypeVar("E")  # Embedding dimension for flattened vision tokens
N_IMG = TypeVar("N_IMG")  # Number of images in the sample
H = TypeVar("H")  # Height
W = TypeVar("W")  # Width


class ChatProcessor:
    """
    Single-purpose processor for BBU dataset chat template creation.

    Handles the complete pipeline from simplified JSONL to training-ready format.
    Uses pure JSON format for object detection output (Qwen2.5-VL compatible).
    """

    def __init__(self, tokenizer, image_processor, **kwargs):
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

        # ---------------- Optional kwargs ----------------
        # Many call-sites (trainer / inference) pass extra kwargs such as
        # merge_size, max_length, use_training_prompts, language …
        # We keep only what is actually needed to stay compatible.
        self.use_training_prompts: bool = kwargs.get("use_training_prompts", False)

        # Prefer explicit arg over global config
        self.language: str = kwargs.get("language", config.language)

        # Initialize special tokens (vision-only)
        self.tokens = SpecialTokens()

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

        # Default context
        self._current_context: str = "training"

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

    def process_sample(self, raw_sample: Dict[str, Any]) -> ChatProcessorOutput:
        """
        Process a single sample from teacher/student structured format.

        Args:
            raw_sample: Sample with 'teachers' (List[Sample]) and 'student' (Sample) structure

        Returns:
            ChatProcessorOutput containing input_ids, labels, attention_mask, pixel_values, image_grid_thw, and ground_truth_objects
        """
        # Debug: Log the sample structure
        teachers = raw_sample["teachers"]
        logger.debug(f"📝 Processing sample: {len(teachers)} teachers + 1 student")

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

        # 5. Extract and normalize ground truth objects for the student
        ground_truth_objects = self._extract_and_normalize_ground_truth(
            raw_sample, image_dims
        )

        return ChatProcessorOutput(
            input_ids=input_ids,
            labels=labels,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            ground_truth_objects=ground_truth_objects,
        )

    def _create_conversation_messages(
        self, sample: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Return a validated list of :class:`ChatMessage` objects built from *sample*."""

        messages: list[ChatMessage] = []

        # 1) System prompt ----------------------------------------------------------------
        messages.append(ChatMessage(role="system", content=self.system_prompt))

        # 2) Teacher examples --------------------------------------------------------------
        teachers: Sequence[Dict[str, Any]] = sample.get("teachers", [])
        for teacher in teachers:
            # User uploads an image ------------------------------------------------------
            messages.append(ChatMessage(role="user", content="<image>"))

            # Assistant returns detection JSON -----------------------------------------
            objects = teacher.get("objects", [])
            sorted_objects = self._sort_objects_by_position(objects)
            assistant_response = self._format_objects_response(sorted_objects)
            messages.append(ChatMessage(role="assistant", content=assistant_response))

        # 3) Student target ---------------------------------------------------------------
        student = sample.get("student", sample)
        messages.append(ChatMessage(role="user", content="<image>"))

        student_objects = student.get("objects", [])
        sorted_student_objects = self._sort_objects_by_position(student_objects)
        student_response = self._format_objects_response(sorted_student_objects)
        messages.append(ChatMessage(role="assistant", content=student_response))

        return messages

    def _extract_all_image_paths(self, sample: Dict[str, Any]) -> List[str]:
        """Collect **all** image paths referenced in *sample* (teachers + student)."""

        image_paths: list[str] = []

        for teacher in sample.get("teachers", []):
            image_paths.extend(teacher.get("images", []))

        image_paths.extend(sample.get("student", sample).get("images", []))

        return image_paths

    def _sort_objects_by_position(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort objects by position (top-to-bottom, left-to-right)."""

        def sort_key(obj):
            box = obj.get("bbox_2d", [0, 0, 0, 0])
            return (box[1], box[0])  # Sort by y first, then x

        return sorted(objects, key=sort_key)

    def _format_objects_response(self, objects: List[Dict[str, Any]]) -> str:
        """Format objects into JSON array format."""
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            box = obj.get("bbox_2d", [0, 0, 0, 0])
            desc = obj.get("desc", "unknown")

            # Create JSON object
            json_obj = {"bbox_2d": box, "label": desc}
            json_objects.append(json_obj)

        # Return formatted JSON array
        return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))

    def _process_images_and_tokens(
        self, conversation: List[ChatMessage], image_paths: List[str]
    ) -> Tuple[List[ChatMessage], List[Image.Image], List[Tuple[int, int]]]:
        """
        Process images and expand vision tokens in conversation.

        Replaces <image> placeholders with proper vision token sequences.
        """
        processed_conversation: list[ChatMessage] = []
        images = []
        image_dims = []
        image_index = 0

        for message in conversation:
            content = message.content

            # Process image placeholders
            while "<image>" in content and image_index < len(image_paths):
                # Load image
                image_path = self.data_root / image_paths[image_index]
                # Fail-fast: raise explicit error if the image cannot be loaded.
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                image_dims.append(image.size)  # (width, height)

                # Calculate number of vision tokens required for this image
                num_vision_tokens = self._calculate_image_tokens(image)

                # Create vision token sequence using helper that inserts spaces between <|image_pad|> tokens
                # This prevents the tokenizer from returning `None` IDs for contiguous special tokens.
                vision_token_sequence = self.tokens.format_vision_tokens(
                    num_vision_tokens
                )

                # Replace placeholder with vision tokens
                content = content.replace("<image>", vision_token_sequence, 1)
                image_index += 1

            processed_conversation.append(
                ChatMessage(role=message.role, content=content)
            )

        # Final validation
        if image_index != len(image_paths):
            logger.warning(
                f"Mismatch between image placeholders and image paths. "
                f"Found {image_index} placeholders, but {len(image_paths)} images."
            )

        return processed_conversation, images, image_dims

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Return the exact number of <|image_pad|> tokens the processor will emit for *image*.

        Older logic estimated this figure from a dummy tensor shaped like the
        image.  That breaks if the HF image-processor performs an internal
        resize or uses a different patch/merge configuration.  We now let the
        processor do its real preprocessing and read the `image_grid_thw`
        metadata that the model itself will consume during the forward pass.
        """
        # Run the *actual* preprocessing pipeline for a single image.  This is
        # comparatively cheap (<1 ms for 896×1344) and guarantees the grid is
        # consistent with training/inference.
        processed = self.image_processor.preprocess([image], return_tensors="pt")

        grid_thw = processed["image_grid_thw"][0]  # (t, h, w)

        merge_size = getattr(self.image_processor, "merge_size", 2)
        tokens_per_merge = merge_size**2

        # Number of flattened patch tokens after the spatial-merge step that
        # the vision tower applies internally.
        num_tokens: int = int(grid_thw.prod().item() // tokens_per_merge)

        return num_tokens

    def _extract_and_normalize_ground_truth(
        self, sample: Dict[str, Any], image_dims: List[Tuple[int, int]]
    ) -> List[GroundTruthObject]:
        """Return a list of :class:`src.schema.GroundTruthObject` instances.

        The bounding boxes are converted from absolute pixel coordinates to the
        *normalised* \[0,1] range expected by the detection loss.
        """
        student = sample.get("student", sample)
        student_objects = student.get("objects", [])

        # The last image in the list corresponds to the student.
        if not image_dims or not student_objects:
            return []

        student_image_dims = image_dims[-1]
        width, height = student_image_dims

        normalized_objects: list[GroundTruthObject] = []
        for obj in student_objects:
            box = obj.get("bbox_2d")
            desc = obj.get("desc")

            if box is None or desc is None:
                raise ValueError(f"Invalid object: {obj}")

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

            # Build structured GT object (automatically validated)
            normalized_objects.append(
                GroundTruthObject(bbox_2d=normalized_box, desc=desc)
            )

        return normalized_objects

    @typechecked
    def _tokenize_conversation(
        self, conversation: List[ChatMessage]
    ) -> Tuple[TensorType["B", "S"], TensorType["B", "S"]]:
        """
        Tokenize conversation and create labels with proper masking.
        """
        # ``apply_chat_template`` expects a ``List[dict]`` – convert once here.
        formatted_text = self.tokenizer.apply_chat_template(
            [asdict(msg) for msg in conversation],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Debug: Log the formatted text before adding endoftext
        logger.debug(
            f"📄 Formatted text before endoftext: {repr(formatted_text[-100:])}"
        )

        # Check if endoftext is already present
        if not formatted_text.endswith(self.tokens.ENDOFTEXT):
            # Add end of text token only if not already present
            formatted_text += self.tokens.ENDOFTEXT
            logger.debug(f"✅ Added ENDOFTEXT token")
        else:
            logger.debug(f"✅ ENDOFTEXT token already present")

        # Debug: Log the formatted text after adding endoftext
        logger.debug(
            f"📄 Formatted text after endoftext: {repr(formatted_text[-100:])}"
        )
        logger.debug(f"🔍 ENDOFTEXT token: {repr(self.tokens.ENDOFTEXT)}")

        # NOTE: Using `return_tensors="pt"` here leads to a hard failure when the tokenizer
        # encounters any `None` values in the produced python lists (typically caused by
        # special-token mis-alignment or exceedingly long inputs).  Instead we first obtain
        # the raw python lists from the tokenizer *without* tensor conversion and only then
        # convert to `torch.Tensor` once we are confident the data structure is correct.

        tokenized = self.tokenizer(
            formatted_text,
            padding=False,
            truncation=False,
            add_special_tokens=False,  # we explicitly bake all special tokens into the prompt
        )

        input_ids_list = tokenized["input_ids"]

        # `input_ids_list` is usually a list with a single sub-list when the input is a
        # single string.  We nevertheless handle both `[List[int]]` and `List[int]` for
        # maximum robustness.
        if len(input_ids_list) == 0:
            raise RuntimeError(
                "Tokenizer returned empty input_ids list – cannot proceed."
            )

        if isinstance(input_ids_list[0], list):
            # Typical case: [[int, int, …]]
            flat_ids: List[int] = input_ids_list[0]
        else:
            # Edge case: already flat [int, int, …]
            flat_ids = input_ids_list  # type: ignore[assignment]

        # Fail-fast if any element is None (this triggers the earlier crash in torch.tensor)
        if any(tok is None for tok in flat_ids):
            raise ValueError(
                "Tokenizer produced `None` token IDs – check that all special tokens are "
                "present in the tokenizer vocabulary. Offending IDs: "
                f"{[tok for tok in flat_ids if tok is None]}"
            )

        # Convert to tensor (1D)
        input_ids_1d = torch.tensor(flat_ids, dtype=torch.long)

        # Create labels (copy of input_ids)
        labels_1d = input_ids_1d.clone()

        # Mask non-assistant tokens → only assistant messages contribute to loss
        labels_1d = self._mask_non_assistant_tokens(
            labels_1d, conversation, formatted_text
        )

        # We **do not** unmask `<|endoftext|>` because in this project that token
        # serves the dual role of *padding* as well as a legacy data delimiter.
        # The actual generation stop token is `<|im_end|>` (tokenizer
        # `eos_token`).  Leaving `<|endoftext|>` masked ensures the language
        # model does not learn to emit padding tokens during generation.

        # ------------------------------------------------------------------
        # Collators (especially PackedDataCollator) expect an explicit batch
        # dimension (B=1) so that concatenation along *dim=1* works without
        # additional squeezing/unsqueezing steps.  We therefore add a leading
        # dimension **after** all 1-D processing is complete.
        # ------------------------------------------------------------------

        input_ids = input_ids_1d.unsqueeze(0)  # (1, S)
        labels = labels_1d.unsqueeze(0)  # (1, S)

        return input_ids, labels

    @typechecked
    def _mask_non_assistant_tokens(
        self,
        labels: TensorType["S"],
        conversation: List[ChatMessage],
        formatted_text: str,
    ) -> TensorType["S"]:
        """Return a version of ``labels`` where only tokens belonging to **assistant**
        messages remain; all others are replaced by ``-100`` so they do not
        contribute to the LM loss.

        This includes assistant messages coming from *teacher* examples **and** the
        final student answer.  The algorithm:

        1. Initialise a full mask (`-100`) the same shape as ``labels``.
        2. Iterate over the conversation sequentially.  For every assistant
           message, locate its byte-offset in ``formatted_text`` *after* the last
           match to avoid duplicates.
        3. Re-tokenise the prefix and the assistant content itself (with
           ``add_special_tokens=False``) to compute the span boundaries in token
           space.
        4. Copy the original token IDs from the untouched ``labels`` tensor back
           into the masked tensor for that assistant span.
        5. Advance the search cursor and continue until all assistant messages
           are processed.
        """

        # Preserve originals to restore assistant spans later
        original_ids = labels.clone()

        # Mask everything
        labels[:] = -100

        # We iteratively rebuild the template token-by-token, thereby knowing
        # the *exact* start/end token indices of every assistant span without
        # performing substring searches over ``formatted_text``.  This makes
        # the complexity strictly O(#tokens) instead of O(#messages × len(text)).

        token_offset: int = 0

        for msg in conversation:
            # ------------------------------------------------------------------
            # Re-tokenise *prefix*, *content* and *suffix* **exactly** as they
            # appear inside the global chat template so that `token_offset`
            # remains perfectly aligned with the flattened conversation.
            # ------------------------------------------------------------------

            prefix_str = f"{self.tokens.IM_START}{msg.role}\n"
            # The HF chat template appends a *newline* after <|im_end|> for every
            # message.  We must replicate that byte-for-byte to keep token
            # alignment in sync with the template; otherwise the running
            # offset drifts by one token per message which manifests as
            # leftover "assistant" prefixes in the label preview.
            suffix_str = f"{self.tokens.IM_END}\n"

            # Token counts ----------------------------------------------------
            prefix_tokens = self.tokenizer(
                prefix_str,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            content_tokens = self.tokenizer(
                msg.content,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            suffix_tokens = self.tokenizer(
                suffix_str,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            # Flatten helper --------------------------------------------------
            def _flatten(lst):
                return lst[0] if lst and isinstance(lst[0], list) else lst

            prefix_tokens = _flatten(prefix_tokens)
            content_tokens = _flatten(content_tokens)
            suffix_tokens = _flatten(suffix_tokens)

            # Un-mask assistant *content* (+ optional suffix) -----------------
            if msg.role == "assistant" and content_tokens:
                start_idx = token_offset + len(prefix_tokens)
                end_idx = start_idx + len(content_tokens)

                labels[start_idx:end_idx] = original_ids[start_idx:end_idx]

                # Optionally unmask the immediate `<|im_end|>` token (first
                # token of the suffix), preserving the rest as -100 so the
                # model explicitly learns to emit the terminator but not the
                # closing `<|im_start|>` of the next turn.
                if suffix_tokens:
                    im_end_token_id = suffix_tokens[0]
                    if (
                        end_idx < labels.size(0)
                        and original_ids[end_idx] == im_end_token_id
                    ):
                        labels[end_idx] = im_end_token_id

            # Advance offset by *full* message length ------------------------
            token_offset += (
                len(prefix_tokens) + len(content_tokens) + len(suffix_tokens)
            )

        return labels

    @typechecked
    def _process_images_for_model(
        self, images: List[Image.Image]
    ) -> Tuple[
        Optional[TensorType["PT", "E"]],  # (tokens, embed_dim)
        Optional[TensorType["N_IMG", 3]],  # (num_images, 3) grid spec
    ]:
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
        image_grid_thw = processed.get("image_grid_thw")

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
            if hasattr(self.image_processor, "merge_size"):
                merge_size = self.image_processor.merge_size
            else:
                # Use Qwen2.5-VL default
                merge_size = 2
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
        else:
            logger.debug(f"   No image_grid_thw available")

        # CRITICAL: Ensure bf16 precision for pixel_values
        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)
            logger.debug(
                f"🔧 Converted pixel_values from {processed['pixel_values'].dtype} to bf16: {pixel_values.dtype}"
            )
        else:
            logger.debug(f"✅ pixel_values already in bf16: {pixel_values.dtype}")

        # Debug: Log the shapes to understand the processing
        logger.debug(f"🖼️ OFFICIAL IMAGE PROCESSING:")
        logger.debug(f"   Number of images: {len(images)}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(
            f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        return pixel_values, image_grid_thw

    @typechecked
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
        # Tokenize text without immediate tensor conversion for robustness
        raw_text_tokens = self.tokenizer(
            text, padding=False, truncation=False, add_special_tokens=False
        )

        # Convert to tensor manually (single sequence expected)
        if isinstance(raw_text_tokens["input_ids"][0], list):
            flat_text_ids = raw_text_tokens["input_ids"][0]
        else:
            flat_text_ids = raw_text_tokens["input_ids"]  # type: ignore[assignment]

        text_input_ids = torch.tensor(flat_text_ids, dtype=torch.long).unsqueeze(0)

        attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)

        text_inputs = {"input_ids": text_input_ids, "attention_mask": attention_mask}

        # Handle vision inputs based on generation step
        if is_first_step and images:
            # First step: process images normally
            pixel_values, image_grid_thw = self._process_images_for_model(images)

            if pixel_values is not None:
                text_inputs["pixel_values"] = pixel_values

            if image_grid_thw is not None:
                text_inputs["image_grid_thw"] = image_grid_thw

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
            for key, value in text_inputs.items()
            if key in valid_generation_params and value is not None
        }

        logger.debug(f"🔧 FILTERED INFERENCE INPUTS: {list(filtered_inputs.keys())}")

        return filtered_inputs

    # ------------------------------------------------------------------
    # Backwards-compat helpers expected by Dataset / Inference / Trainer
    # ------------------------------------------------------------------

    def set_context(self, context: str = "training") -> None:
        """No-op context setter kept for external compatibility."""
        self._current_context = context

    def get_current_system_prompt(self) -> str:
        """Return the system prompt currently in use (helper for logging)."""
        return self.system_prompt


def create_chat_processor(
    tokenizer,
    image_processor,
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
