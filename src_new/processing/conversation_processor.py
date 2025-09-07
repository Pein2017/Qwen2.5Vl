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

from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLProcessor
from dataclasses import dataclass
from enum import Enum

from .coordinate_converter import CoordinateTokenConverter
from .templates import CONSTANTS, get_system_prompt
from .special_tokens import IMAGE_PAD
from .variants import create_default_variant_registry
from ..utils.rank_aware_logging import get_rank_aware_logger

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


class TeacherStudentValidationError(ConversationStructureError):
    pass


class ConversationTruncationError(ConversationStructureError):
    pass


@dataclass
class ConversationValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    image_count: int
    turn_count: int
    details: Dict[str, Any]


class ConversationType(Enum):
    SIMPLE = "simple"
    TEACHER_STUDENT = "teacher_student"
    INFERENCE = "inference"
    MULTI_TEACHER = "multi_teacher"


class ConversationValidator:
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
                errors=errors,
                warnings=warnings,
                image_count=len(images),
                turn_count=len(messages),
                details=details,
            )
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            return ConversationValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                image_count=len(images) if isinstance(images, list) else 0,
                turn_count=len(messages) if isinstance(messages, list) else 0,
                details=details,
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

    processor: Qwen2VLProcessor
    coordinate_tokens_enabled: bool
    coordinate_converter: CoordinateTokenConverter

    def __init__(
        self,
        processor: Qwen2VLProcessor,
        max_coord_value: int,
        coordinate_tokens_enabled: bool,
    ) -> None:
        if processor is None:
            raise ValueError("processor cannot be None")
        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be a positive integer, got {max_coord_value!r}"
            )
        if not isinstance(coordinate_tokens_enabled, bool):
            raise ValueError(
                f"coordinate_tokens_enabled must be a bool, got {type(coordinate_tokens_enabled)}"
            )
        self.processor = processor
        self.coordinate_tokens_enabled = coordinate_tokens_enabled
        self.coordinate_converter = CoordinateTokenConverter(
            max_coord_value=max_coord_value,
            coordinate_tokens_enabled=coordinate_tokens_enabled,
        )
        # Cache system prompt once (stable per instance)
        self._system_prompt: str = get_system_prompt(self.coordinate_tokens_enabled)
        # Variant registry
        self._variant_registry = create_default_variant_registry(self.coordinate_converter)

    # ---------------- Variant user-text helpers are centralized in geometry_text/variants ----------------

    # ------------- Core utilities -------------
    def _validate_messages_and_images(
        self, messages: List[Dict], images: List[Image.Image]
    ) -> None:
        if not messages:
            raise ConversationStructureError("Empty messages list")
        if not isinstance(images, list) or not images:
            raise ConversationStructureError("images must be a non-empty list")
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
        if image_placeholders != len(images):
            raise ImageTokenMismatchError(
                f"Image placeholder count ({image_placeholders}) != actual image count ({len(images)})",
                expected_count=image_placeholders,
                actual_count=len(images),
            )

    def _interleave_images_with_turns(
        self, messages: List[Dict], all_images: List[Image.Image]
    ) -> Tuple[List[Dict], List[Image.Image]]:
        if not messages or not all_images:
            raise ConversationStructureError("Messages and images cannot be empty")
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
        self._validate_messages_and_images(messages, images)
        vm, oi = self._interleave_images_with_turns(messages, images)
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
            outputs = self.processor(
                text=[text], images=images, return_tensors="pt", padding=True
            )
            if not isinstance(outputs, dict):
                try:
                    outputs = outputs.data  # type: ignore[attr-defined]
                except Exception:
                    outputs = {}
        except Exception as e:
            raise RuntimeError(f"Processor call failed: {type(e).__name__}: {e}")
        # Squeeze batch dimension on text tensors for downstream span/mask code
        if isinstance(outputs.get("input_ids"), torch.Tensor) and outputs["input_ids"].dim() == 2 and outputs["input_ids"].shape[0] == 1:
            outputs["input_ids"] = outputs["input_ids"].squeeze(0)
        if isinstance(outputs.get("attention_mask"), torch.Tensor) and outputs["attention_mask"].dim() == 2 and outputs["attention_mask"].shape[0] == 1:
            outputs["attention_mask"] = outputs["attention_mask"].squeeze(0)
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
        assistant_text = self.coordinate_converter.convert_objects_to_tokens(
            sample["objects"]
        )
        # Dense caption variant: user contains only image(s)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": ([{"type": "image"}] * len(images))},
            {"role": "assistant", "content": assistant_text},
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=False
        )
        return self._process_text_and_images(text, oi)

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
        for t_sample, t_images in zip(teacher_samples, teacher_images_list):
            if "objects" not in t_sample or not t_sample["objects"]:
                continue
            t_assistant = self.coordinate_converter.convert_objects_to_tokens(
                t_sample["objects"]
            )
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": t_assistant},
                ]
            )
            all_images.extend(t_images[:1])
        # Student turn: user with only image; assistant full dense outputs (training)
        if "objects" not in student_sample or not student_sample["objects"]:
            raise TeacherStudentValidationError("Student sample missing objects")
        s_assistant = self.coordinate_converter.convert_objects_to_tokens(
            student_sample["objects"]
        )
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
        return self._process_text_and_images(text, oi)

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
    def _get_variant_handlers(self, variant: str):
        handler = self._variant_registry.get(variant)
        try:
            handler_name = type(handler).__name__
        except Exception:
            handler_name = str(handler)
        logger.debug(f"🧩 Variant resolved: '{variant}' -> handler={handler_name}")
        return handler.build_user_text, handler.build_assistant_text

    def _build_simple_conversation_unified(
        self, sample: Dict[str, Any], images: List[Image.Image], variant: str
    ) -> Dict[str, torch.Tensor]:
        if "objects" not in sample or not sample["objects"]:
            raise ConversationStructureError("Sample missing non-empty 'objects'")
        logger.debug(
            f"🗣️ Building simple conversation (variant='{variant}', images={len(images)})"
        )
        user_text_fn, assistant_fn = self._get_variant_handlers(variant)
        user_text = user_text_fn(sample["objects"])  # may be None for dense
        assistant_text = assistant_fn(sample["objects"])  # always a string
        if user_text is None:
            user_content = ([{"type": "image"}] * len(images))
        else:
            user_content = [{"type": "text", "text": user_text}, {"type": "image"}]
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=images, add_generation_prompt=False
        )
        return self._process_text_and_images(text, oi)

    def _build_teacher_student_conversation_unified(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        variant: str,
    ) -> Dict[str, torch.Tensor]:
        if not teacher_samples or not teacher_images_list:
            raise TeacherStudentValidationError("teacher_samples/images cannot be empty")
        if len(teacher_samples) != len(teacher_images_list):
            raise TeacherStudentValidationError("Mismatch teachers vs images lists")
        logger.debug(
            f"🗣️ Building teacher-student conversation (variant='{variant}', teachers={len(teacher_samples)}, student_images={len(student_images)})"
        )
        user_text_fn, assistant_fn = self._get_variant_handlers(variant)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        all_images: List[Image.Image] = []
        # Teachers
        for t_sample, t_images in zip(teacher_samples, teacher_images_list):
            objs = t_sample.get("objects", [])
            if not objs:
                continue
            u_text = user_text_fn(objs)
            t_assistant = assistant_fn(objs)
            if u_text is None:
                u_content = [{"type": "image"}]
            else:
                u_content = [{"type": "text", "text": u_text}, {"type": "image"}]
            messages.extend(
                [
                    {"role": "user", "content": u_content},
                    {"role": "assistant", "content": t_assistant},
                ]
            )
            all_images.extend(t_images[:1])
        # Student
        if "objects" not in student_sample or not student_sample["objects"]:
            raise TeacherStudentValidationError("Student sample missing objects")
        s_u_text = user_text_fn(student_sample["objects"]) 
        s_assistant = assistant_fn(student_sample["objects"]) 
        if s_u_text is None:
            s_u_content = [{"type": "image"}]
        else:
            s_u_content = [{"type": "text", "text": s_u_text}, {"type": "image"}]
        messages.extend(
            [
                {"role": "user", "content": s_u_content},
                {"role": "assistant", "content": s_assistant},
            ]
        )
        all_images.extend(student_images[:1])
        text, oi = self._apply_chat_template_safe(
            messages=messages, images=all_images, add_generation_prompt=False
        )
        return self._process_text_and_images(text, oi)

    def create_conversation(
        self,
        sample: Dict[str, Any],
        images: List[Image.Image],
        variant: str,
        teacher_samples: Optional[List[Dict[str, Any]]] = None,
        teacher_images_list: Optional[List[List[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified public entrypoint for conversation creation.

        When teacher_samples/teacher_images_list are provided and non-empty, builds
        a teacher-student conversation; otherwise builds a simple conversation.
        """
        logger.debug(
            f"🧵 create_conversation: variant='{variant}', teachers={bool(teacher_samples and teacher_images_list)}"
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
