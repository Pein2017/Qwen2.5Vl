#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat processor for Qwen2.5-VL conversation building and tokenization.

Implements stateless conversation building with multi-turn teacher-student support,
chat template integration, and loss masking strategy.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from .templates import TemplateManager
from .token_processor import TokenProcessor


logger = logging.getLogger(__name__)

# Ignore index for loss computation (standard PyTorch convention)
IGNORE_INDEX = -100


@dataclass
class ChatConfig:
    """Configuration for chat processing."""

    language: str = "chinese"
    use_training_prompts: bool = False
    teacher_ratio: float = 0.7
    max_teacher_examples: int = 3
    coordinate_tokens_enabled: bool = False


@dataclass
class ProcessedSample:
    """Processed training sample ready for model input."""

    input_ids: torch.Tensor  # [seq_len]
    attention_mask: torch.Tensor  # [seq_len]
    labels: torch.Tensor  # [seq_len] with IGNORE_INDEX for masked positions
    images: Optional[torch.Tensor] = None  # Will be handled by image processor
    coordinate_mask: Optional[torch.Tensor] = None  # [seq_len] for coordinate tokens
    has_teachers: bool = False
    num_teachers: int = 0
    teacher_spans: Optional[List[Tuple[int, int]]] = (
        None  # [(start, end), ...] for teacher assistant responses
    )
    student_spans: Optional[List[Tuple[int, int]]] = (
        None  # [(start, end), ...] for student assistant responses
    )


class ChatProcessor:
    """
    Handles conversation building and tokenization for Qwen2.5-VL training.

    Supports both standalone detection and teacher-student learning scenarios.
    Implements stateless processing with proper loss masking for training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        token_processor: TokenProcessor,
        config: ChatConfig,
    ) -> None:
        """
        Initialize chat processor.

        Args:
            tokenizer: Pre-trained tokenizer (may be extended with new tokens)
            token_processor: Token processor for coordinate handling
            config: Chat processing configuration
        """
        self.tokenizer = tokenizer
        self.token_processor = token_processor
        self.config = config
        self.template_manager = TemplateManager(
            language=config.language,
            token_processor=token_processor,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Cache important token IDs for efficient processing
        self._cache_special_token_ids()

    def _cache_special_token_ids(self) -> None:
        """Cache special token IDs for efficient processing."""
        vocab = self.tokenizer.get_vocab()

        # Chat template tokens
        self.im_start_id = vocab.get("<|im_start|>", None)
        self.im_end_id = vocab.get("<|im_end|>", None)

        # Try to get system/user/assistant tokens
        self.system_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None

        # These might be encoded as separate tokens
        system_ids = self.tokenizer.encode("system", add_special_tokens=False)
        user_ids = self.tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = self.tokenizer.encode("assistant", add_special_tokens=False)

        if len(system_ids) == 1:
            self.system_token_id = system_ids[0]
        if len(user_ids) == 1:
            self.user_token_id = user_ids[0]
        if len(assistant_ids) == 1:
            self.assistant_token_id = assistant_ids[0]

        logger.info(
            f"Cached special tokens - im_start: {self.im_start_id}, im_end: {self.im_end_id}"
        )

    def build_conversation(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build conversation for a single sample without teacher examples.

        Args:
            sample: Sample dictionary with objects and metadata

        Returns:
            List of conversation messages
        """
        return self.template_manager.get_standalone_conversation(sample)

    def add_teacher_examples(
        self, conversation: List[Dict[str, str]], teachers: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Add teacher examples to create multi-turn teacher-student conversation.

        Args:
            conversation: Base conversation (typically standalone)
            teachers: List of teacher example dictionaries

        Returns:
            Extended conversation with teacher examples
        """
        if not teachers:
            return conversation

        # Extract the student sample from the original conversation
        student_sample = self._parse_student_sample_from_conversation(conversation)

        # Create new teacher-student conversation
        return self.template_manager.format_teacher_student_conversation(
            teachers, student_sample
        )

    def _parse_student_sample_from_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Parse student sample from standalone conversation.

        Args:
            conversation: Standalone conversation with student response

        Returns:
            Student sample dictionary with parsed objects
        """
        # Get the last assistant message as student response
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                student_response = msg["content"]
                # Parse the response to extract objects
                return self._parse_objects_from_response(student_response)

        # Fallback to empty sample if no assistant message found
        return {"objects": []}

    def _parse_objects_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse objects from assistant response text.

        Args:
            response: Assistant response containing object annotations

        Returns:
            Sample dictionary with parsed objects
        """
        import json
        import re

        objects = []

        try:
            # Try to parse as JSON first (if response is in JSON format)
            if response.strip().startswith("[") and response.strip().endswith("]"):
                parsed_objects = json.loads(response.strip())
                if isinstance(parsed_objects, list):
                    objects = parsed_objects
            else:
                # Parse object tokens from response
                # Look for patterns like <|obj_ref_start|>desc<|obj_ref_end|><|bbox_start|>[coords]<|bbox_end|>
                obj_pattern = r"<\|obj_ref_start\|>(.*?)<\|obj_ref_end\|><\|bbox_start\|>\[(.*?)\]<\|bbox_end\|>"
                matches = re.findall(obj_pattern, response)

                for desc, coords_str in matches:
                    try:
                        # Parse coordinates
                        if self.config.coordinate_tokens_enabled:
                            # Parse coordinate tokens
                            coord_tokens = [
                                token.strip() for token in coords_str.split(",")
                            ]
                            coords = self.token_processor.tokens_to_coordinates(
                                coord_tokens
                            )
                        else:
                            # Parse numeric coordinates
                            coords = [float(x.strip()) for x in coords_str.split(",")]

                        if len(coords) == 4:  # bbox_2d
                            objects.append({"desc": desc.strip(), "bbox_2d": coords})
                    except (ValueError, AttributeError) as e:
                        logger.warning(
                            f"Failed to parse coordinates from '{coords_str}': {e}"
                        )
                        continue

        except Exception as e:
            logger.warning(f"Failed to parse objects from response: {e}")

        return {"objects": objects}

    def tokenize_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> ProcessedSample:
        """
        Tokenize conversation and create training sample with loss masking.

        Args:
            conversation: List of conversation messages

        Returns:
            ProcessedSample ready for training
        """
        # Apply chat template
        chat_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Tokenize the full conversation
        encoding = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels with loss masking and span tracking
        labels, teacher_spans, student_spans = self.create_loss_mask_with_spans(
            input_ids, conversation
        )

        # Create coordinate mask if coordinate tokens are enabled
        coordinate_mask = None
        if self.config.coordinate_tokens_enabled:
            coordinate_mask = self.token_processor.create_coordinate_mask(
                input_ids, self.tokenizer
            )

        # Count teachers (simple heuristic based on conversation length)
        num_teachers = max(
            0, (len(conversation) - 3) // 2
        )  # Subtract system + user + assistant, divide by 2
        has_teachers = num_teachers > 0

        return ProcessedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            coordinate_mask=coordinate_mask,
            has_teachers=has_teachers,
            num_teachers=num_teachers,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )

    def create_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create loss mask where only assistant responses contribute to loss.

        This is a simplified version that doesn't track spans.
        Use create_loss_mask_with_spans for full functionality.

        Args:
            input_ids: Token sequence [seq_len]

        Returns:
            Labels tensor with IGNORE_INDEX for masked positions
        """
        labels, _, _ = self.create_loss_mask_with_spans(input_ids, [])
        return labels

    def create_loss_mask_with_spans(
        self, input_ids: torch.Tensor, conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Create loss mask with teacher-student span tracking.

        Args:
            input_ids: Token sequence [seq_len]
            conversation: Original conversation messages for span identification

        Returns:
            Tuple of (labels, teacher_spans, student_spans)
        """
        labels = input_ids.clone()
        teacher_spans = []
        student_spans = []

        if self.im_start_id is None or self.im_end_id is None:
            logger.warning(
                "Chat template tokens not found - cannot create proper loss mask"
            )
            return labels, teacher_spans, student_spans

        # Convert to list for easier processing
        token_list = input_ids.tolist()

        # Determine which assistant responses are teachers vs student
        assistant_indices = self._get_assistant_message_indices(conversation)

        # Find conversation boundaries and mask non-assistant content
        in_assistant = False
        assistant_count = 0
        assistant_start = None
        i = 0

        while i < len(token_list):
            current_token = token_list[i]

            if current_token == self.im_start_id:
                # Check next token to determine role
                if i + 1 < len(token_list):
                    next_token = token_list[i + 1]

                    # Check if this is assistant start
                    if next_token == self.assistant_token_id:
                        in_assistant = True
                        assistant_start = i + 2  # Start after <|im_start|>assistant
                        # Mask the im_start and assistant tokens themselves
                        labels[i] = IGNORE_INDEX
                        labels[i + 1] = IGNORE_INDEX
                        i += 2
                        continue
                    else:
                        # This is system or user - mask everything until im_end
                        in_assistant = False

                # Mask current token and continue
                labels[i] = IGNORE_INDEX

            elif current_token == self.im_end_id:
                # End of assistant response - record span
                if in_assistant and assistant_start is not None:
                    assistant_end = i  # End before <|im_end|>

                    # Determine if this is teacher or student span
                    if assistant_count < len(assistant_indices) - 1:
                        # This is a teacher response
                        teacher_spans.append((assistant_start, assistant_end))
                    else:
                        # This is the student response (last assistant message)
                        student_spans.append((assistant_start, assistant_end))

                    assistant_count += 1
                    assistant_start = None

                # Mask the im_end token
                labels[i] = IGNORE_INDEX
                in_assistant = False

            elif not in_assistant:
                # Mask system and user content
                labels[i] = IGNORE_INDEX

            # If in_assistant is True, we keep the token for loss computation

            i += 1

        return labels, teacher_spans, student_spans

    def _get_assistant_message_indices(
        self, conversation: List[Dict[str, str]]
    ) -> List[int]:
        """
        Get indices of assistant messages in conversation.

        Args:
            conversation: List of conversation messages

        Returns:
            List of indices where assistant messages occur
        """
        return [i for i, msg in enumerate(conversation) if msg["role"] == "assistant"]

    def should_assign_teachers(self) -> bool:
        """
        Determine whether to assign teacher examples based on teacher ratio.

        Returns:
            True if teachers should be assigned
        """
        return random.random() < self.config.teacher_ratio

    def process_sample_with_teacher_pool(
        self,
        sample: Dict[str, Any],
        teacher_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> ProcessedSample:
        """
        Process sample with optional teacher examples from teacher pool.

        Args:
            sample: Student sample to process
            teacher_pool: Pool of available teacher examples

        Returns:
            Processed sample ready for training
        """
        # Build base conversation
        conversation = self.build_conversation(sample)

        # Add teacher examples if conditions are met
        if teacher_pool and self.should_assign_teachers() and len(teacher_pool) > 0:
            # Randomly select teachers
            num_teachers = min(
                random.randint(1, self.config.max_teacher_examples), len(teacher_pool)
            )

            selected_teachers = random.sample(teacher_pool, num_teachers)
            conversation = self.add_teacher_examples(conversation, selected_teachers)

        # Tokenize and return processed sample
        return self.tokenize_conversation(conversation)

    def batch_process_samples(
        self,
        samples: List[Dict[str, Any]],
        teacher_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ProcessedSample]:
        """
        Process a batch of samples.

        Args:
            samples: List of samples to process
            teacher_pool: Pool of available teacher examples

        Returns:
            List of processed samples
        """
        processed_samples = []

        for sample in samples:
            try:
                processed = self.process_sample_with_teacher_pool(sample, teacher_pool)
                processed_samples.append(processed)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                # Create a minimal valid sample to avoid breaking the batch
                conversation = self.build_conversation({"objects": []})
                processed = self.tokenize_conversation(conversation)
                processed_samples.append(processed)

        return processed_samples

    def get_loss_mask_statistics(self, labels: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics about loss masking for debugging.

        Args:
            labels: Labels tensor with masked positions

        Returns:
            Dictionary with masking statistics
        """
        total_tokens = labels.numel()
        masked_tokens = (labels == IGNORE_INDEX).sum().item()
        training_tokens = total_tokens - masked_tokens

        return {
            "total_tokens": total_tokens,
            "masked_tokens": masked_tokens,
            "training_tokens": training_tokens,
            "mask_ratio": masked_tokens / total_tokens if total_tokens > 0 else 0.0,
            "training_ratio": training_tokens / total_tokens
            if total_tokens > 0
            else 0.0,
        }

    def validate_processed_sample(self, sample: ProcessedSample) -> bool:
        """
        Validate that processed sample is correctly formatted.

        Args:
            sample: Processed sample to validate

        Returns:
            True if sample is valid
        """
        try:
            # Check tensor shapes match
            seq_len = sample.input_ids.shape[0]
            if sample.attention_mask.shape[0] != seq_len:
                logger.error("Attention mask shape mismatch")
                return False

            if sample.labels.shape[0] != seq_len:
                logger.error("Labels shape mismatch")
                return False

            if (
                sample.coordinate_mask is not None
                and sample.coordinate_mask.shape[0] != seq_len
            ):
                logger.error("Coordinate mask shape mismatch")
                return False

            # Check that some tokens are available for training
            training_tokens = (sample.labels != IGNORE_INDEX).sum().item()
            if training_tokens == 0:
                logger.warning("No training tokens available - all tokens are masked")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating sample: {e}")
            return False
