#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace-first conversation processor for Qwen2.5-VL.

Simplified, self-contained implementation that:
- Uses processor.apply_chat_template() for conversation formatting
- Converts objects to strings via CoordinateTokenConverter
- Provides dense-captioning builders and two additional detection-style variants
- Supports teacher-student and simple flows; includes inference builder
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from src_new.types import FormatMode, ConversationVariant

import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor

from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.templates import CONSTANTS, get_system_prompt
from src_new.processing.special_tokens import IMAGE_PAD
from src_new.processing.variants import create_default_variant_registry
from src_new.utils.rank_aware_logging import get_rank_aware_logger

logger = get_rank_aware_logger("processing.conversation")


class ConversationError(Exception):
    pass


class ConversationStructureError(ConversationError):
    pass


class ImageTokenMismatchError(ConversationStructureError):
    def __init__(self, message: str, expected_count: int, actual_count: int):
        super().__init__(message)
        self.expected_count = expected_count
        self.actual_count = actual_count


class TeacherStudentValidationError(ConversationError):
    """Error during teacher-student conversation validation."""
    pass


class ConversationTruncationError(ConversationError):
    """Error when conversation exceeds maximum length."""
    pass


@dataclass
class ConversationValidationResult:
    """Result of conversation validation."""
    is_valid: bool
    error_message: Optional[str] = None
    conversation_type: Optional[ConversationType] = None
    token_count: Optional[int] = None


class ConversationType(Enum):
    """Type of conversation being processed."""
    SIMPLE = "simple"
    TEACHER_STUDENT = "teacher_student"
    INFERENCE = "inference"
    MULTI_TEACHER = "multi_teacher"


class ConversationValidator:
    """Validator for conversation structure and content."""
    
    def __init__(self):
        pass
    
    def validate(self, conversation: Dict[str, Any]) -> ConversationValidationResult:
        """Validate a conversation structure."""
        # Basic validation implementation
        if not isinstance(conversation, dict):
            return ConversationValidationResult(
                is_valid=False,
                error_message="Conversation must be a dictionary"
            )
        
        return ConversationValidationResult(is_valid=True)

    @staticmethod
    def validate_conversation_structure(
        messages: List[Dict],
        images: List[Image.Image],
        conversation_type: ConversationType,
    ) -> ConversationValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        try:
            if not messages:
                errors.append("Empty messages list")

            if not isinstance(images, list):
                errors.append(f"Images must be a list, got {type(images)}")

            image_placeholder_count = 0
            user_turns = 0
            assistant_turns = 0
            system_turns = 0

            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    errors.append(f"Message {i} must be a dict, got {type(message)}")
                    continue
                if "role" not in message or "content" not in message:
                    errors.append(f"Message {i} missing 'role' or 'content'")
                    continue
                role = message["role"]
                content = message["content"]
                if role == "system":
                    system_turns += 1
                elif role == "user":
                    user_turns += 1
                    if isinstance(content, list):
                        image_placeholder_count += sum(
                            1
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "image"
                        )
                elif role == "assistant":
                    assistant_turns += 1

            details.update(
                {
                    "user_turns": user_turns,
                    "assistant_turns": assistant_turns,
                    "system_turns": system_turns,
                    "image_placeholder_count": image_placeholder_count,
                }
            )

            if conversation_type == ConversationType.SIMPLE:
                if user_turns != 1:
                    errors.append(
                        f"Simple conversation must have exactly 1 user turn, got {user_turns}"
                    )
                if assistant_turns != 1:
                    errors.append(
                        f"Simple conversation must have exactly 1 assistant turn, got {assistant_turns}"
                    )
                if len(images) != 1:
                    errors.append(
                        f"Simple conversation must have exactly 1 image, got {len(images)}"
                    )
            elif conversation_type == ConversationType.TEACHER_STUDENT:
                if user_turns < 2:
                    errors.append(
                        f"Teacher-student conversation must have at least 2 user turns, got {user_turns}"
                    )
                if assistant_turns < 2:
                    errors.append(
                        f"Teacher-student conversation must have at least 2 assistant turns, got {assistant_turns}"
                    )
                if len(images) < 2:
                    errors.append(
                        f"Teacher-student conversation must have at least 2 images, got {len(images)}"
                    )
            elif conversation_type == ConversationType.INFERENCE:
                if assistant_turns > 0:
                    warnings.append(
                        f"Inference conversation has {assistant_turns} assistant turns (unexpected)"
                    )
                if len(images) == 0:
                    errors.append("Inference conversation must have at least 1 image")

            if image_placeholder_count != len(images):
                errors.append(
                    f"Image placeholder count ({image_placeholder_count}) != actual image count ({len(images)})"
                )

            return ConversationValidationResult(
                is_valid=len(errors) == 0,
                error_message=", ".join(errors),
                conversation_type=conversation_type,
                token_count=len(messages),
            )
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            return ConversationValidationResult(
                is_valid=False,
                error_message=f"Validation failed with exception: {str(e)}",
                conversation_type=conversation_type,
                token_count=len(messages) if isinstance(messages, list) else 0,
            )

    @staticmethod
    def validate_image_token_consistency(
        text: str, images: List[Image.Image]
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        image_token_count = text.count(IMAGE_PAD)
        if image_token_count != len(images):
            errors.append(
                f"Image token count ({image_token_count}) != provided image count ({len(images)})"
            )
        return len(errors) == 0, errors


class ConversationProcessor:
    """
    HuggingFace-first conversation processor.

    Wrapper around official HuggingFace processor with coordinate token conversion.
    """

    processor: Qwen2_5_VLProcessor
    coordinate_converter: CoordinateTokenConverter

    def __init__(
        self,
        processor: Qwen2_5_VLProcessor,
    ) -> None:
        if processor is None:
            raise ValueError("processor cannot be None")
        self.processor = processor
        self._format_mode: str = FormatMode.SPECIAL_TOKENS.value
        self.coordinate_converter = CoordinateTokenConverter(
            format_mode=self._format_mode,
        )
        
        # Initialize variant registry for conversation variants
        self._variant_registry = create_default_variant_registry(self.coordinate_converter)
        
        # Initialize system prompt
        self._system_prompt = get_system_prompt(
            format_mode=self._format_mode,
        )

    # ---------------- Variant user-text helpers are centralized in geometry_text/variants ----------------

    # ------------- Core utilities -------------
    def _validate_messages_and_images(
        self, messages: List[Dict], images: List[Image.Image]
    ) -> int:
        if not messages:
            raise ConversationStructureError("Empty messages list")
        if not isinstance(images, list):
            raise ConversationStructureError("images must be provided as a list")
        image_placeholders = 0
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ConversationStructureError(
                    f"Invalid message at index {i}: {msg!r}"
                )
            content = msg["content"]
            if isinstance(content, list):
                image_placeholders += sum(
                    1 for item in content if isinstance(item, dict) and item.get("type") == "image"
                )
        if image_placeholders == 0 and images:
            raise ImageTokenMismatchError(
                "Messages do not reference images but images were provided",
                expected_count=0,
                actual_count=len(images),
            )
        if image_placeholders != len(images):
            raise ImageTokenMismatchError(
                f"Image placeholder count ({image_placeholders}) != actual image count ({len(images)})",
                expected_count=image_placeholders,
                actual_count=len(images),
            )
        return image_placeholders

    def _interleave_images_with_turns(
        self,
        messages: List[Dict],
        all_images: List[Image.Image],
        placeholder_count: int,
    ) -> Tuple[List[Dict], List[Image.Image]]:
        if not messages:
            raise ConversationStructureError("Messages cannot be empty")
        if placeholder_count == 0:
            if all_images:
                raise ImageTokenMismatchError(
                    "No image placeholders in messages but images were provided",
                    expected_count=0,
                    actual_count=len(all_images),
                )
            return [dict(msg) for msg in messages], []
        if not all_images:
            raise ImageTokenMismatchError(
                "Messages reference image placeholders but images list is empty",
                expected_count=placeholder_count,
                actual_count=0,
            )
        validated_messages: List[Dict] = []
        ordered_images: List[Image.Image] = []
        image_index = 0
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ConversationStructureError("Each message must include 'role' and 'content'")
            validated_messages.append(dict(msg))
            content = msg["content"]
            if isinstance(content, list):
                img_count = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
                if img_count:
                    end = image_index + img_count
                    if end > len(all_images):
                        raise ImageTokenMismatchError(
                            "Not enough images for provided image placeholders",
                            expected_count=end,
                            actual_count=len(all_images),
                        )
                    ordered_images.extend(all_images[image_index:end])
                    image_index = end
        if image_index != len(all_images):
            raise ImageTokenMismatchError(
                f"Image usage mismatch: used {image_index} images but provided {len(all_images)}",
                expected_count=len(all_images),
                actual_count=image_index,
            )
        return validated_messages, ordered_images

    def _apply_chat_template_safe(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
        add_generation_prompt: bool,
    ) -> Tuple[str, List[Image.Image]]:
        """Validate, interleave, and safely apply the chat template."""
        placeholder_count = self._validate_messages_and_images(messages, images)
        vm, oi = self._interleave_images_with_turns(messages, images, placeholder_count)
        proc_any: Any = self.processor
        text = proc_any.apply_chat_template(
            vm,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            images=oi,
        )
        return text, oi

    def _process_text_and_images(
        self, text: str, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        try:
            has_images = bool(images)
            processor_kwargs: Dict[str, Any] = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True,
                "truncation": False,
            }
            if has_images:
                processor_kwargs["images"] = images
            outputs = self.processor(**processor_kwargs)
            if not isinstance(outputs, dict):
                try:
                    outputs = outputs.data  # type: ignore[attr-defined]
                except Exception:
                    outputs = {}
        except Exception as e:
            raise RuntimeError(f"Processor call failed: {type(e).__name__}: {e}")
        # Keep 2D batch-first tensors (no squeezing) to standardize shapes across the pipeline
        # Validate that image placeholders in text match provided images
        image_token_count = text.count(IMAGE_PAD)
        if image_token_count != len(images):
            raise ImageTokenMismatchError(
                f"Image token count ({image_token_count}) != provided image count ({len(images)})",
                expected_count=image_token_count,
                actual_count=len(images),
            )
        # Ensure required keys exist
        if "input_ids" not in outputs or "attention_mask" not in outputs:
            raise RuntimeError("Processor did not produce required text tensors")
        if images and ("pixel_values" not in outputs or "image_grid_thw" not in outputs):
            raise RuntimeError(
                "Processor did not return required image tensors (pixel_values, image_grid_thw)"
            )
        # Attach conversation text
        outputs["conversation_text"] = text
        # Attach offset_mapping from tokenizer strictly; fail if unavailable
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor missing tokenizer for offset mapping")
        tokenized = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        try:
            om = tokenized.get("offset_mapping")
        except Exception:
            om = None
        if om is None:
            raise RuntimeError("Tokenizer did not return offset_mapping")
        outputs["offset_mapping"] = om[0]
        return outputs

    # ------------- Dense captioning builders (baseline) -------------
    def create_simple_conversation(
        self, sample: Dict[str, Any], images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        if "objects" not in sample or not sample["objects"]:
            raise ConversationStructureError("Sample missing non-empty 'objects'")
        assistant_render = self.coordinate_converter.convert_objects_to_tokens(
            sample["objects"]
        )
        assistant_text = assistant_render["text"] if isinstance(assistant_render, dict) else assistant_render
        # Dense caption variant: user contains only image(s)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": ([{"type": "image"}] * len(images))},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            if logger.isEnabledFor(logging.DEBUG):
                placeholders = sum(
                    1
                    for m in messages
                    if m.get('role') == 'user'
                    for c in (m.get('content') if isinstance(m.get('content'), list) else [])
                    if isinstance(c, dict) and c.get('type') == 'image'
                )
                logger.debug("[conv] simple: images=%d placeholders=%d", len(images), placeholders)
        except Exception:
            pass
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=False
        )
        out = self._process_text_and_images(text, oi)
        return out

    def create_teacher_student_conversation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        if not teacher_samples or not teacher_images_list:
            raise TeacherStudentValidationError("teacher_samples/images cannot be empty")
        if len(teacher_samples) != len(teacher_images_list):
            raise TeacherStudentValidationError("Mismatch teachers vs images lists")
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        all_images: List[Image.Image] = []
        # Teacher turns: user with only image; assistant full dense outputs
        turn_group_spans = []
        turn_roles = []
        for t_sample, t_images in zip(teacher_samples, teacher_images_list):
            if "objects" not in t_sample or not t_sample["objects"]:
                continue
            t_assistant_render = self.coordinate_converter.convert_objects_to_tokens(
                t_sample["objects"]
            )
            t_assistant = t_assistant_render["text"] if isinstance(t_assistant_render, dict) else t_assistant_render
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": t_assistant},
                ]
            )
            all_images.extend(t_images[:1])
            if isinstance(t_assistant_render, dict) and t_assistant_render.get("group_char_spans"):
                turn_group_spans.append(t_assistant_render["group_char_spans"])
                turn_roles.append("teacher")
        # Student turn: user with only image; assistant full dense outputs (training)
        if "objects" not in student_sample or not student_sample["objects"]:
            raise TeacherStudentValidationError("Student sample missing objects")
        s_assistant_render = self.coordinate_converter.convert_objects_to_tokens(
            student_sample["objects"]
        )
        s_assistant = s_assistant_render["text"] if isinstance(s_assistant_render, dict) else s_assistant_render
        messages.extend(
            [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": s_assistant},
            ]
        )
        all_images.extend(student_images[:1])
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=all_images, add_generation_prompt=False
        )
        try:
            placeholder_count = 0
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), list):
                    placeholder_count += sum(1 for item in m["content"] if isinstance(item, dict) and item.get("type") == "image")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[conv] teacher_student: teachers=%d images=%d placeholders=%d", len(teacher_samples), len(all_images), placeholder_count)
        except Exception:
            pass
        out = self._process_text_and_images(text, oi)
        # Precompute spans for teacher-student
        try:
            from src_new.processing.span_builder import build_assistant_spans_token_aligned as _build_spans
            t_spans, s_spans = _build_spans(
                conversation_text=text,
                offset_mapping=out.get("offset_mapping"),
                tokenizer=self.processor.tokenizer,
                input_ids_expanded=out["input_ids"],
                has_teachers=True,
                num_teachers=len(teacher_samples),
                include_eos=True,
            )
            out["teacher_assistant_spans"] = t_spans
            out["student_assistant_spans"] = s_spans
        except Exception:
            pass
        return out

    def create_inference_conversation(
        self, user_prompt: str, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ConversationStructureError("user_prompt must be non-empty string")
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},
                ],
            },
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=True
        )
        try:
            pl = 0
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), list):
                    pl += sum(1 for item in m["content"] if isinstance(item, dict) and item.get("type") == "image")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[conv] inference: images=%d placeholders=%d", len(images), pl)
        except Exception:
            pass
        return self._process_text_and_images(text, oi)

    def create_simple_conversation_for_generation(
        self, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """Build generation inputs matching training dense-caption setup (image-only user).

        Messages:
        - system: training system prompt
        - user: image placeholder only (no free-form text)
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=True
        )
        return self._process_text_and_images(text, oi)

    def create_summary_conversation_for_generation(
        self, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """Build generation inputs for one-line Chinese summary (summary variant).

        Messages:
        - system: SUMMARY_SYSTEM_PROMPT
        - user: SUMMARY_USER_PROMPT + image
        """
        try:
            sys_prompt = self._get_system_prompt_for_variant("summary")
        except Exception:
            from src_new.processing.templates import SUMMARY_SYSTEM_PROMPT as _SSP  # type: ignore
            sys_prompt = _SSP
        from src_new.processing.templates import CONSTANTS as _C
        user_text = _C.get("SUMMARY_USER_PROMPT", "请只输出一行摘要：")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image"}]},
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=True
        )
        return self._process_text_and_images(text, oi)

    def create_teacher_student_conversation_for_generation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        enable_recovery: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not teacher_samples or not teacher_images_list:
            raise TeacherStudentValidationError("teacher_samples/images cannot be empty")
        if len(teacher_samples) != len(teacher_images_list):
            raise TeacherStudentValidationError("Mismatch teachers vs images lists")
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        all_images: List[Image.Image] = []
        for t_sample, t_images in zip(teacher_samples, teacher_images_list):
            objs = t_sample.get("objects", [])
            if not objs:
                if enable_recovery:
                    continue
                raise TeacherStudentValidationError("Empty teacher objects")
            t_assistant = self.coordinate_converter.convert_objects_to_tokens(objs)
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": t_assistant},
                ]
            )
            all_images.extend(t_images[:1])
        if "objects" not in student_sample or not student_sample["objects"]:
            raise TeacherStudentValidationError("Student sample missing objects")
        messages.append({"role": "user", "content": [{"type": "image"}]})
        all_images.extend(student_images[:1])
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=all_images, add_generation_prompt=True
        )
        return self._process_text_and_images(text, oi)

    # ------------- Unified variant dispatcher -------------
    def _get_variant_handlers(self, variant: Union[str, ConversationVariant]):
        v = getattr(variant, "value", variant)
        handler = self._variant_registry.get(str(v))
        try:
            handler_name = type(handler).__name__
        except Exception:
            handler_name = str(handler)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("🧩 Variant resolved: '%s' -> handler=%s", v, handler_name)
        return handler.build_user_text, handler.build_assistant_text

    def _get_system_prompt_for_variant(self, variant: Union[str, ConversationVariant]) -> str:
        try:
            v = str(getattr(variant, "value", variant)).strip().lower()
        except Exception:
            v = str(variant).strip().lower()
        if v == "summary":
            try:
                return CONSTANTS.get("SUMMARY_SYSTEM_PROMPT", self._system_prompt)
            except Exception:
                return self._system_prompt
        return self._system_prompt

    def _build_user_content(
        self,
        user_spec: Optional[Union[str, Dict[str, Any]]],
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        if user_spec is None:
            if not images:
                raise ImageTokenMismatchError(
                    "Variant requested image-only turn but images list is empty",
                    expected_count=1,
                    actual_count=0,
                )
            return [{"type": "image"} for _ in images]

        include_image = True
        text_value: Optional[str]
        if isinstance(user_spec, dict):
            text_value = user_spec.get("text")
            include_image = bool(user_spec.get("include_image", True))
        else:
            text_value = user_spec

        content: List[Dict[str, Any]] = []
        if text_value:
            content.append({"type": "text", "text": text_value})

        if include_image:
            if not images:
                raise ImageTokenMismatchError(
                    "Variant requested image placeholder but no images were provided",
                    expected_count=1,
                    actual_count=0,
                )
            content.extend({"type": "image"} for _ in images)

        return content

    def _build_simple_conversation_unified(
        self, sample: Dict[str, Any], images: List[Image.Image], variant: Union[str, ConversationVariant]
    ) -> Dict[str, torch.Tensor]:
        if "objects" not in sample or not sample["objects"]:
            raise ConversationStructureError("Sample missing non-empty 'objects'")
        logger.debug(
            f"🗣️ Building simple conversation (variant='{getattr(variant, 'value', variant)}', images={len(images)})"
        )

        # Direct summary path: use precomputed sample['summary']
        try:
            vkey = str(getattr(variant, "value", variant)).strip().lower()
        except Exception:
            vkey = str(variant).strip().lower()
        if vkey == "summary":
            sys_prompt = self._get_system_prompt_for_variant(variant)
            user_text = CONSTANTS.get("SUMMARY_USER_PROMPT", "请只输出一行摘要：")
            assistant_text = str(sample.get("summary", "")).strip()
            if not assistant_text:
                raise ConversationStructureError("Summary variant requires precomputed 'summary' in sample")
            user_content = [{"type": "text", "text": user_text}, {"type": "image"}]
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_text},
            ]
            text, oi = self._apply_chat_template_safe(
                messages=messages, images=images, add_generation_prompt=False
            )
            out = self._process_text_and_images(text, oi)
            try:
                from src_new.processing.span_builder import build_assistant_spans_token_aligned as _build_spans
                t_spans, s_spans = _build_spans(
                    conversation_text=text,
                    offset_mapping=out.get("offset_mapping"),
                    tokenizer=self.processor.tokenizer,
                    input_ids_expanded=out["input_ids"],
                    has_teachers=False,
                    num_teachers=0,
                    include_eos=True,
                )
                out["teacher_assistant_spans"] = t_spans
                out["student_assistant_spans"] = s_spans
            except Exception:
                pass
            return out

        user_text_fn, assistant_fn = self._get_variant_handlers(variant)
        user_spec = user_text_fn(sample["objects"])  # type: ignore[arg-type]
        assistant_render = assistant_fn(sample["objects"])  # may be dict in plain mode
        assistant_text = (
            assistant_render["text"]
            if isinstance(assistant_render, dict) and "text" in assistant_render
            else assistant_render
        )
        user_content = self._build_user_content(user_spec, images)
        sys_prompt = self._get_system_prompt_for_variant(variant)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=False
        )
        out = self._process_text_and_images(text, oi)
        try:
            from src_new.processing.span_builder import build_assistant_spans_token_aligned as _build_spans
            t_spans, s_spans = _build_spans(
                conversation_text=text,
                offset_mapping=out.get("offset_mapping"),
                tokenizer=self.processor.tokenizer,
                input_ids_expanded=out["input_ids"],
                has_teachers=False,
                num_teachers=0,
                include_eos=True,
            )
            out["teacher_assistant_spans"] = t_spans
            out["student_assistant_spans"] = s_spans
        except Exception:
            pass
        return out

    def _build_teacher_student_conversation_unified(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        variant: Union[str, ConversationVariant],
    ) -> Dict[str, torch.Tensor]:
        if not teacher_samples or not teacher_images_list:
            raise TeacherStudentValidationError("teacher_samples/images cannot be empty")
        if len(teacher_samples) != len(teacher_images_list):
            raise TeacherStudentValidationError("Mismatch teachers vs images lists")
        logger.debug(
            f"🗣️ Building teacher-student conversation (variant='{getattr(variant, 'value', variant)}', teachers={len(teacher_samples)}, student_images={len(student_images)})"
        )

        # Direct summary path: use precomputed 'summary' for teachers and student
        try:
            vkey = str(getattr(variant, "value", variant)).strip().lower()
        except Exception:
            vkey = str(variant).strip().lower()
        if vkey == "summary":
            sys_prompt = self._get_system_prompt_for_variant(variant)
            sum_messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
            sum_images: List[Image.Image] = []
            valid_teachers = 0

            # Teacher turns
            for t_sample, t_images in zip(teacher_samples, (teacher_images_list or [])):
                objs = t_sample.get("objects")
                if not isinstance(objs, list) or not objs:
                    logger.warning(f"Skipping teacher sample with empty/invalid objects: {type(objs)}")
                    continue
                t_summary = str(t_sample.get("summary", "")).strip()
                if not t_summary:
                    raise TeacherStudentValidationError("Summary variant requires precomputed 'summary' in teacher sample")
                sum_messages.extend(
                    [
                        {"role": "user", "content": [{"type": "image"}]},
                        {"role": "assistant", "content": t_summary},
                    ]
                )
                sum_images.append(t_images[0])
                valid_teachers += 1

            if valid_teachers == 0:
                raise TeacherStudentValidationError("No valid teacher samples found after filtering")

            # Student turn
            if "objects" not in student_sample or not student_sample["objects"]:
                raise TeacherStudentValidationError("Student sample missing objects")
            s_u_text = CONSTANTS.get("SUMMARY_USER_PROMPT", "请只输出一行摘要：")
            s_summary = str(student_sample.get("summary", "")).strip()
            if not s_summary:
                raise TeacherStudentValidationError("Summary variant requires precomputed 'summary' in student sample")
            s_u_content = [{"type": "text", "text": s_u_text}, {"type": "image"}]
            sum_messages.extend(
                [
                    {"role": "user", "content": s_u_content},
                    {"role": "assistant", "content": s_summary},
                ]
            )
            sum_images.extend(student_images[:1])

            text, oi = self._apply_chat_template_safe(
                messages=sum_messages, images=sum_images, add_generation_prompt=False
            )
            out = self._process_text_and_images(text, oi)
            # Precompute spans for teacher-student
            try:
                from src_new.processing.span_builder import build_assistant_spans_token_aligned as _build_spans
                t_spans, s_spans = _build_spans(
                    conversation_text=text,
                    offset_mapping=out.get("offset_mapping"),
                    tokenizer=self.processor.tokenizer,
                    input_ids_expanded=out["input_ids"],
                    has_teachers=True,
                    num_teachers=valid_teachers,
                    include_eos=True,
                )
                out["teacher_assistant_spans"] = t_spans
                out["student_assistant_spans"] = s_spans
            except Exception:
                pass
            return out

        user_text_fn, assistant_fn = self._get_variant_handlers(variant)
        sys_prompt = self._get_system_prompt_for_variant(variant)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        all_images: List[Image.Image] = []
        turn_group_spans: List[Dict[str, Any]] = []
        turn_roles: List[str] = []

        # Teacher examples first
        valid_teachers = 0
        for t_sample, t_images in zip(teacher_samples, teacher_images_list or []):
            objs = t_sample.get("objects")
            if not isinstance(objs, list) or not objs:
                # Skip empty teacher samples rather than failing
                logger.warning(f"Skipping teacher sample with empty/invalid objects: {type(objs)}")
                continue
            
            t_render = assistant_fn(objs)
            t_assistant = (
                t_render["text"] if isinstance(t_render, dict) and "text" in t_render else t_render
            )
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": t_assistant},
                ]
            )
            all_images.append(t_images[0])
            if isinstance(t_render, dict) and t_render.get("group_char_spans"):
                turn_group_spans.append(t_render["group_char_spans"])
                turn_roles.append("teacher")
            valid_teachers += 1

        # Ensure we have at least one valid teacher
        if valid_teachers == 0:
            raise TeacherStudentValidationError("No valid teacher samples found after filtering")

        # Student turn
        if "objects" not in student_sample or not student_sample["objects"]:
            raise TeacherStudentValidationError("Student sample missing objects")
        s_u_spec = user_text_fn(student_sample["objects"])  # type: ignore[index]
        s_render = assistant_fn(student_sample["objects"])  # type: ignore[index]
        s_assistant = (
            s_render["text"] if isinstance(s_render, dict) and "text" in s_render else s_render
        )
        student_image_slice = student_images[:1]
        s_u_content = self._build_user_content(s_u_spec, student_image_slice)
        messages.extend(
            [
                {"role": "user", "content": s_u_content},
                {"role": "assistant", "content": s_assistant},
            ]
        )
        if any(isinstance(item, dict) and item.get("type") == "image" for item in s_u_content):
            all_images.extend(student_image_slice)
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=all_images, add_generation_prompt=False
        )
        out = self._process_text_and_images(text, oi)
        # Precompute spans for teacher-student
        try:
            from src_new.processing.span_builder import build_assistant_spans_token_aligned as _build_spans
            t_spans, s_spans = _build_spans(
                conversation_text=text,
                offset_mapping=out.get("offset_mapping"),
                tokenizer=self.processor.tokenizer,
                input_ids_expanded=out["input_ids"],
                has_teachers=True,
                num_teachers=valid_teachers,
                include_eos=True,
            )
            out["teacher_assistant_spans"] = t_spans
            out["student_assistant_spans"] = s_spans
        except Exception:
            pass
        return out

    def create_conversation(
        self,
        sample: Dict[str, Any],
        images: List[Image.Image],
        variant: Union[str, ConversationVariant],
        teacher_samples: Optional[List[Dict[str, Any]]] = None,
        teacher_images_list: Optional[List[List[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified public entrypoint for conversation creation.

        When teacher_samples/teacher_images_list are provided and non-empty, builds
        a teacher-student conversation; otherwise builds a simple conversation.
        """
        logger.debug(
            f"🧵 create_conversation: variant='{getattr(variant, 'value', variant)}', teachers={bool(teacher_samples and teacher_images_list)}"
        )
        if teacher_samples and teacher_images_list:
            return self._build_teacher_student_conversation_unified(
                student_sample=sample,
                teacher_samples=teacher_samples,
                student_images=images,
                teacher_images_list=teacher_images_list,
                variant=variant,
            )
        return self._build_simple_conversation_unified(sample=sample, images=images, variant=variant)
