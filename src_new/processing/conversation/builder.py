from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..conversation_processor import ConversationProcessor


class ConversationBuilder:
    """Thin wrapper around ConversationProcessor providing the new API surface."""

    _impl: ConversationProcessor

    def __init__(
        self,
        processor: Any,
        max_coord_value: int,
        coordinate_tokens_enabled: bool,
    ) -> None:
        self._impl = ConversationProcessor(
            processor=processor,
            max_coord_value=max_coord_value,
            coordinate_tokens_enabled=coordinate_tokens_enabled,
        )

    # Simple delegations to preserve behavior
    def create_simple_conversation(
        self, sample: Dict[str, Any], images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_simple_conversation(sample, images)

    def create_teacher_student_conversation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_teacher_student_conversation(
            student_sample, teacher_samples, student_images, teacher_images_list
        )

    def create_multi_teacher_conversation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        max_conversation_length: Optional[int] = None,
        teacher_selection_strategy: str = "all",
        enable_caching: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_multi_teacher_conversation(
            student_sample,
            teacher_samples,
            student_images,
            teacher_images_list,
            max_conversation_length,
            teacher_selection_strategy,
            enable_caching,
        )

    def create_inference_conversation(
        self, user_prompt: str, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_inference_conversation(user_prompt, images)

    def create_conversation_with_truncation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        max_tokens: int = 2048,
        truncation_strategy: str = "reduce_teachers",
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_conversation_with_truncation(
            student_sample,
            teacher_samples,
            student_images,
            teacher_images_list,
            max_tokens,
            truncation_strategy,
        )

    def create_teacher_student_conversation_for_generation(
        self,
        student_sample: Dict[str, Any],
        teacher_samples: List[Dict[str, Any]],
        student_images: List[Image.Image],
        teacher_images_list: List[List[Image.Image]],
        enable_recovery: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_teacher_student_conversation_for_generation(
            student_sample,
            teacher_samples,
            student_images,
            teacher_images_list,
            enable_recovery,
        )

    def create_simple_conversation_for_generation(
        self, sample: Dict[str, Any], images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        return self._impl.create_simple_conversation_for_generation(sample, images)
