#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified debug logging utilities for Qwen2.5-VL training pipeline.

This module provides essential debug logging with one-time logging per training run.
All logging is handled at the trainer level for proper training/evaluation context.

Key Features:
- One-time logging: exactly once for training and once for evaluation
- Pre-tokenization text logging with coordinate token verification
- Loss mask analysis with teacher-student span tracking
- Simple global flags with no complex session management
"""

import logging
import re
from typing import List, Optional, Tuple

import torch


# Configure logger using rank-aware system
def get_debug_logger():
    """Get rank-aware logger for debug logging module."""
    try:
        from .rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("debug_logging")
    except ImportError:
        # Fallback to centralized config system
        try:
            from ..config.config import _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL

            logger = logging.getLogger("debug_logging")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(_GLOBAL_LOG_LEVEL)
                _CONFIGURED_LOGGERS.add("debug_logging")
            return logger
        except ImportError:
            # Final fallback to standard logging
            fallback_logger = logging.getLogger(__name__)
            if not fallback_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
                )
                handler.setFormatter(formatter)
                fallback_logger.addHandler(handler)
                # Use the root logger's level or DEBUG if explicitly set
                root_level = logging.getLogger().getEffectiveLevel()
                fallback_logger.setLevel(root_level)
            return fallback_logger


logger = get_debug_logger()


class DebugLogger:
    """
    Simplified debug logging for Qwen2.5-VL training pipeline.

    Provides one-time logging for essential debugging information during training.
    All logging is controlled by simple global flags to ensure exactly one log
    per training run for each type of debug information.
    """

    def __init__(self):
        """Initialize debug logger with simple one-time logging control."""
        # Simple global flags - log exactly once for the entire training run
        self.logged_training_sample = False
        self.logged_evaluation_sample = False
        self.logged_training_loss_mask = False
        self.logged_evaluation_loss_mask = False

        # Essential patterns for basic validation
        self.coord_token_pattern = re.compile(r"<\|coord_(\d+)\|>")
        self.im_start_pattern = re.compile(r"<\|im_start\|>")
        self.im_end_pattern = re.compile(r"<\|im_end\|>")
        self.assistant_pattern = re.compile(r"<\|im_start\|>assistant\n")

    def should_log_training_sample(self) -> bool:
        """Check if we should log a training sample (only once per training run)."""
        if not self.logged_training_sample:
            self.logged_training_sample = True
            return True
        return False

    def should_log_evaluation_sample(self) -> bool:
        """Check if we should log an evaluation sample (only once per training run)."""
        if not self.logged_evaluation_sample:
            self.logged_evaluation_sample = True
            return True
        return False

    def should_log_training_loss_mask(self) -> bool:
        """Check if we should log training loss mask analysis (only once per training run)."""
        if not self.logged_training_loss_mask:
            self.logged_training_loss_mask = True
            return True
        return False

    def should_log_evaluation_loss_mask(self) -> bool:
        """Check if we should log evaluation loss mask analysis (only once per training run)."""
        if not self.logged_evaluation_loss_mask:
            self.logged_evaluation_loss_mask = True
            return True
        return False

    def start_training_run(self) -> None:
        """
        Initialize debug logging for a new training run.

        This should be called once at the start of training to ensure
        we get exactly one training sample and one evaluation sample logged.
        """
        logger.info(
            "🚀 Debug logging: Started new training run - will log one training and one evaluation sample"
        )

    def reset_for_new_run(self) -> None:
        """
        Reset all logging flags for a completely new training run.

        This is useful for testing or when starting a fresh training session.
        """
        self.logged_training_sample = False
        self.logged_evaluation_sample = False
        self.logged_training_loss_mask = False
        self.logged_evaluation_loss_mask = False
        logger.info("🔄 Debug logging: Reset all flags for new training run")

    def log_pre_tokenization_text(
        self,
        chat_text: str,
        sample_id: Optional[str] = None,
        is_training: bool = True,
        has_teachers: bool = False,
    ) -> None:
        """
        Log pre-tokenization text with comprehensive format validation.

        Logs exactly once for training and once for evaluation during the entire training run.
        Shows the complete conversation structure with coordinate tokens and special tokens.

        Args:
            chat_text: Final text ready for tokenization (after coordinate wrapping)
            sample_id: Sample identifier for tracking
            is_training: Whether this is during training
            has_teachers: Whether sample contains teacher examples
        """

        # Simple one-time logging check
        if is_training:
            if not self.should_log_training_sample():
                return
        else:
            if not self.should_log_evaluation_sample():
                return

        sample_id = sample_id or "UNKNOWN"
        mode = "TRAINING" if is_training else "EVALUATION"

        logger.debug("=" * 80)
        logger.debug(
            f"🔍 PRE-TOKENIZATION CONVERSATION ANALYSIS [{sample_id}] ({mode})"
        )
        logger.debug("=" * 80)

        # Log the complete conversation text
        self._log_complete_conversation_text(chat_text, sample_id, has_teachers)

        # Log conversation structure analysis
        self._log_conversation_structure_analysis(chat_text, sample_id)

        # Log coordinate token analysis
        self._log_coordinate_token_analysis(chat_text, sample_id)

        logger.debug("=" * 80)

    def _log_complete_conversation_text(
        self, chat_text: str, sample_id: str, has_teachers: bool
    ) -> None:
        """
        Log the complete conversation text with clear formatting and compacted image tokens.

        Args:
            chat_text: Complete conversation text
            sample_id: Sample identifier
            has_teachers: Whether this is a teacher-student conversation
        """
        conversation_type = "TEACHER-STUDENT" if has_teachers else "SIMPLE"
        logger.debug(
            f"📄 COMPLETE CONVERSATION TEXT [{sample_id}] ({conversation_type}):"
        )

        # Compact image_pad tokens for better readability and format as string
        compacted_text = self._compact_image_pad_tokens(chat_text)
        formatted_text = repr(compacted_text)
        logger.debug(f"Conversation string: {formatted_text}")

    def _compact_image_pad_tokens(self, text: str) -> str:
        """
        Compact consecutive image_pad tokens for better readability.

        Converts sequences like:
        <|image_pad|><|image_pad|><|image_pad|><|image_pad|>...

        To:
        <image_pad>*N

        Args:
            text: Original conversation text

        Returns:
            Text with compacted image_pad tokens
        """
        import re

        # Pattern to match consecutive <|image_pad|> tokens
        image_pad_pattern = r"(?:<\|image_pad\|>)+"

        def replace_image_pads(match):
            # Count the number of image_pad tokens in the match
            full_match = match.group(0)
            count = full_match.count("<|image_pad|>")
            return f"<image_pad>*{count}"

        # Replace all consecutive image_pad sequences
        compacted = re.sub(image_pad_pattern, replace_image_pads, text)

        return compacted

    def _log_conversation_structure_analysis(
        self, chat_text: str, sample_id: str
    ) -> None:
        """
        Log detailed conversation structure analysis.

        Args:
            chat_text: Complete conversation text
            sample_id: Sample identifier
        """
        logger.debug(f"🔍 CONVERSATION STRUCTURE ANALYSIS [{sample_id}]:")

        lines = chat_text.split("\n")
        conversation_segments = []
        current_segment = None

        for i, line in enumerate(lines):
            if "<|im_start|>" in line:
                if "system" in line:
                    current_segment = {"type": "SYSTEM", "start_line": i, "content": []}
                elif "user" in line:
                    current_segment = {"type": "USER", "start_line": i, "content": []}
                elif "assistant" in line:
                    current_segment = {
                        "type": "ASSISTANT",
                        "start_line": i,
                        "content": [],
                    }
                else:
                    current_segment = {
                        "type": "UNKNOWN",
                        "start_line": i,
                        "content": [],
                    }
            elif "<|im_end|>" in line:
                if current_segment:
                    current_segment["end_line"] = i
                    conversation_segments.append(current_segment)
                    current_segment = None
            elif current_segment:
                current_segment["content"].append(line)

        # Log each conversation segment
        for i, segment in enumerate(conversation_segments):
            segment_type = segment["type"]
            start_line = segment["start_line"]
            end_line = segment["end_line"]
            content_lines = len(segment["content"])

            logger.debug(
                f"  Segment {i + 1}: {segment_type} [lines {start_line}-{end_line}] ({content_lines} content lines)"
            )

            # Show content preview for each segment
            if segment["content"]:
                preview = " ".join(segment["content"])[:100]
                if len(preview) == 100:
                    preview += "..."
                logger.debug(f"    Content: {preview}")

                # Count special tokens in this segment
                segment_text = "\n".join(segment["content"])
                coord_count = len(self.coord_token_pattern.findall(segment_text))
                image_count = segment_text.count("<|image_pad|>")

                if coord_count > 0:
                    logger.debug(f"    🎯 Coordinate tokens: {coord_count}")
                if image_count > 0:
                    # Show compacted format for image tokens
                    logger.debug(
                        f"    🖼️ Image tokens: {image_count} (shown as <image_pad>*{image_count})"
                    )

        logger.debug(f"  Total conversation segments: {len(conversation_segments)}")

    def _log_coordinate_token_analysis(self, chat_text: str, sample_id: str) -> None:
        """
        Log detailed coordinate token analysis.

        Args:
            chat_text: Complete conversation text
            sample_id: Sample identifier
        """
        logger.debug(f"🎯 COORDINATE TOKEN ANALYSIS [{sample_id}]:")

        # Find all coordinate tokens
        coord_matches = self.coord_token_pattern.findall(chat_text)
        coord_values = [int(match) for match in coord_matches]

        if coord_values:
            logger.debug(f"  Total coordinate tokens: {len(coord_values)}")
            logger.debug(
                f"  Coordinate value range: {min(coord_values)} - {max(coord_values)}"
            )
            logger.debug(
                f"  Sample coordinate values: {coord_values[:10]}{'...' if len(coord_values) > 10 else ''}"
            )

            # Analyze coordinate token distribution
            coord_ranges = {
                "0-255": sum(1 for v in coord_values if 0 <= v <= 255),
                "256-511": sum(1 for v in coord_values if 256 <= v <= 511),
                "512-767": sum(1 for v in coord_values if 512 <= v <= 767),
                "768-1023": sum(1 for v in coord_values if 768 <= v <= 1023),
                "1024+": sum(1 for v in coord_values if v >= 1024),
            }

            logger.debug("  Coordinate value distribution:")
            for range_name, count in coord_ranges.items():
                if count > 0:
                    logger.debug(
                        f"    {range_name}: {count} tokens ({count / len(coord_values) * 100:.1f}%)"
                    )
        else:
            logger.debug("  No coordinate tokens found")

        # Count other special tokens
        image_count = chat_text.count("<|image_pad|>")
        obj_ref_start_count = chat_text.count("<|obj_ref_start|>")
        obj_ref_end_count = chat_text.count("<|obj_ref_end|>")
        box_start_count = chat_text.count("<|box_start|>")
        box_end_count = chat_text.count("<|box_end|>")

        logger.debug(f"  Other special tokens:")
        logger.debug(f"    <|image_pad|>: {image_count}")
        logger.debug(
            f"    <|obj_ref_start|>/<|obj_ref_end|>: {obj_ref_start_count}/{obj_ref_end_count}"
        )
        logger.debug(
            f"    <|box_start|>/<|box_end|>: {box_start_count}/{box_end_count}"
        )

    def _log_conversation_summary(self, chat_text: str, sample_id: str) -> None:
        """
        Log essential conversation information for debugging.

        Args:
            chat_text: Complete conversation text
            sample_id: Sample identifier
        """
        # Count basic conversation elements
        im_start_count = len(self.im_start_pattern.findall(chat_text))
        im_end_count = len(self.im_end_pattern.findall(chat_text))
        assistant_count = len(self.assistant_pattern.findall(chat_text))
        coord_tokens = len(self.coord_token_pattern.findall(chat_text))
        image_tokens = chat_text.count("<|image_pad|>")

        logger.info(f"Conversation summary [{sample_id}]:")
        logger.info(f"  - Length: {len(chat_text)} characters")
        logger.info(
            f"  - Conversation turns: {im_start_count} start, {im_end_count} end"
        )
        logger.info(f"  - Assistant messages: {assistant_count}")
        logger.info(f"  - Coordinate tokens: {coord_tokens}")
        logger.info(f"  - Image tokens: {image_tokens}")

        # Show a sample of the text (first 200 chars)
        sample_text = chat_text[:200] + "..." if len(chat_text) > 200 else chat_text
        logger.info(f"  - Sample text: {repr(sample_text)}")

        # Basic validation warnings
        if im_start_count != im_end_count:
            logger.warning(f"Unbalanced conversation tags [{sample_id}]")
        if assistant_count == 0:
            logger.warning(f"No assistant messages found [{sample_id}]")
        elif assistant_count > 1:
            logger.info(f"Multi-assistant conversation (teacher-student) [{sample_id}]")

    def log_loss_mask_analysis(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_spans: List[Tuple[int, int]],
        student_spans: List[Tuple[int, int]],
        sample_id: Optional[str] = None,
        is_training: bool = True,
        tokenizer=None,
        coordinate_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Log comprehensive loss mask analysis with detailed token-level information.

        Logs exactly once for training and once for evaluation during the entire training run.
        Shows which tokens participate in learning vs. are masked, with clear indicators.

        Args:
            input_ids: Token sequence
            labels: Labels with loss masking applied
            teacher_spans: Teacher response token spans
            student_spans: Student response token spans
            sample_id: Sample identifier
            is_training: Whether this is during training
            tokenizer: Tokenizer for token decoding (optional)
            coordinate_mask: Mask for coordinate tokens (optional)
        """

        # Simple one-time logging check
        if is_training:
            if not self.should_log_training_loss_mask():
                return
        else:
            if not self.should_log_evaluation_loss_mask():
                return

        sample_id = sample_id or "UNKNOWN"
        mode = "TRAINING" if is_training else "EVALUATION"

        logger.debug("=" * 80)
        logger.debug(f"🎯 LOSS MASK ANALYSIS [{sample_id}] ({mode})")
        logger.debug("=" * 80)

        # Calculate mask statistics
        total_tokens = len(labels)
        masked_tokens = (labels == -100).sum().item()
        unmasked_tokens = total_tokens - masked_tokens

        # Count teacher and student tokens (spans use exclusive end index)
        teacher_token_count = sum(end - start for start, end in teacher_spans)
        student_token_count = sum(end - start for start, end in student_spans)

        logger.debug(f"📊 MASK STATISTICS [{sample_id}]:")
        logger.debug(f"  Total tokens: {total_tokens}")
        logger.debug(
            f"  Masked tokens (-100): {masked_tokens} ({masked_tokens / total_tokens * 100:.1f}%)"
        )
        logger.debug(
            f"  Unmasked tokens (learning): {unmasked_tokens} ({unmasked_tokens / total_tokens * 100:.1f}%)"
        )
        logger.debug(f"  Teacher response tokens: {teacher_token_count}")
        logger.debug(f"  Student response tokens: {student_token_count}")

        # Coordinate token analysis
        if coordinate_mask is not None:
            coord_count = coordinate_mask.sum().item()
            coord_in_unmasked = (coordinate_mask & (labels != -100)).sum().item()
            logger.debug(
                f"  Coordinate tokens (L1 loss): {coord_count} ({coord_count / total_tokens * 100:.1f}%)"
            )
            logger.debug(f"  Coordinate tokens in learning region: {coord_in_unmasked}")

        # Verify span coverage
        total_span_tokens = teacher_token_count + student_token_count
        if total_span_tokens != unmasked_tokens:
            logger.debug(f"⚠️ SPAN COVERAGE MISMATCH:")
            logger.debug(f"  Expected unmasked tokens: {total_span_tokens}")
            logger.debug(f"  Actual unmasked tokens: {unmasked_tokens}")
            logger.debug(f"  Difference: {total_span_tokens - unmasked_tokens} tokens")
        else:
            logger.debug(f"✅ Span coverage verified: {total_span_tokens} tokens match")

        # Log detailed span information
        self._log_detailed_span_analysis(
            input_ids, labels, teacher_spans, student_spans, sample_id, tokenizer
        )

        # Log token-by-token analysis (first 50 tokens)
        self._log_token_by_token_analysis(
            input_ids, labels, coordinate_mask, sample_id, tokenizer
        )

        logger.debug("=" * 80)

    def _log_detailed_span_analysis(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_spans: List[Tuple[int, int]],
        student_spans: List[Tuple[int, int]],
        sample_id: str,
        tokenizer=None,
    ) -> None:
        """
        Log detailed analysis of teacher and student spans.

        Args:
            input_ids: Token sequence
            labels: Labels with masking
            teacher_spans: Teacher response spans
            student_spans: Student response spans
            sample_id: Sample identifier
            tokenizer: Tokenizer for decoding
        """
        logger.debug(f"🔍 DETAILED SPAN ANALYSIS [{sample_id}]:")

        # Analyze teacher spans
        if teacher_spans:
            logger.debug(f"  👨‍🏫 TEACHER SPANS ({len(teacher_spans)} spans):")
            for i, (start, end) in enumerate(teacher_spans):
                span_length = end - start
                span_tokens = (
                    input_ids[start:end]
                    if start < len(input_ids) and end <= len(input_ids)
                    else None
                )

                logger.debug(
                    f"    Span {i + 1}: [{start}, {end}) - {span_length} tokens"
                )

                if tokenizer and span_tokens is not None:
                    try:
                        span_text = tokenizer.decode(
                            span_tokens, skip_special_tokens=False
                        )
                        preview = (
                            span_text[:100] + "..."
                            if len(span_text) > 100
                            else span_text
                        )
                        logger.debug(f"      Text: {repr(preview)}")
                    except Exception:
                        logger.debug(f"      Text: [decode failed]")

        # Analyze student spans
        if student_spans:
            logger.debug(f"  👨‍🎓 STUDENT SPANS ({len(student_spans)} spans):")
            for i, (start, end) in enumerate(student_spans):
                span_length = end - start
                span_tokens = (
                    input_ids[start:end]
                    if start < len(input_ids) and end <= len(input_ids)
                    else None
                )

                logger.debug(
                    f"    Span {i + 1}: [{start}, {end}) - {span_length} tokens"
                )

                if tokenizer and span_tokens is not None:
                    try:
                        span_text = tokenizer.decode(
                            span_tokens, skip_special_tokens=False
                        )
                        preview = (
                            span_text[:100] + "..."
                            if len(span_text) > 100
                            else span_text
                        )
                        logger.debug(f"      Text: {repr(preview)}")
                    except Exception:
                        logger.debug(f"      Text: [decode failed]")

        if not teacher_spans and not student_spans:
            logger.debug("  No teacher or student spans found")

    def _log_token_by_token_analysis(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor],
        sample_id: str,
        tokenizer=None,
        max_tokens: int = 50,
    ) -> None:
        """
        Log token-by-token analysis showing learning vs. masked tokens.

        Args:
            input_ids: Token sequence
            labels: Labels with masking
            coordinate_mask: Coordinate token mask
            sample_id: Sample identifier
            tokenizer: Tokenizer for decoding
            max_tokens: Maximum number of tokens to analyze
        """
        logger.debug(
            f"🔤 TOKEN-BY-TOKEN ANALYSIS [{sample_id}] (first {max_tokens} tokens):"
        )
        logger.debug("  Format: Token_ID 'text' -> STATUS [COORD if coordinate token]")

        analysis_length = min(max_tokens, len(input_ids), len(labels))

        for i in range(analysis_length):
            token_id = input_ids[i].item()
            label = labels[i].item()

            # Determine token status
            if label == -100:
                status = "MASKED"
            else:
                status = "LEARNING"

            # Check if coordinate token
            coord_status = ""
            if coordinate_mask is not None and coordinate_mask[i]:
                coord_status = " [COORD]"

            # Decode token text
            token_text = "[decode failed]"
            if tokenizer:
                try:
                    token_text = tokenizer.decode([token_id])
                    # Escape special characters for logging
                    token_text = repr(token_text)
                except Exception:
                    pass

            logger.debug(
                f"    Token {i:3d}: {token_id:6d} {token_text:20s} -> {status:8s}{coord_status}"
            )

        if len(input_ids) > max_tokens:
            logger.debug(
                f"    ... ({len(input_ids) - max_tokens} more tokens not shown)"
            )

        # Summary of token types in the analyzed range
        analyzed_tokens = labels[:analysis_length]
        analyzed_masked = (analyzed_tokens == -100).sum().item()
        analyzed_learning = analysis_length - analyzed_masked

        logger.debug(f"  Summary of first {analysis_length} tokens:")
        logger.debug(
            f"    MASKED: {analyzed_masked} tokens ({analyzed_masked / analysis_length * 100:.1f}%)"
        )
        logger.debug(
            f"    LEARNING: {analyzed_learning} tokens ({analyzed_learning / analysis_length * 100:.1f}%)"
        )

    def reconfigure_logger(self, force_debug: bool = False) -> None:
        """Reconfigure logger to use current global log level."""
        global logger
        logger = get_debug_logger()

        # Force DEBUG level if requested
        if force_debug:
            logger.setLevel(logging.DEBUG)
            for handler in logger.handlers:
                handler.setLevel(logging.DEBUG)

        logger.info("Debug logger reconfigured with current global log level")
        logger.info(f"Current logger level: {logger.level}")
        logger.info(f"Current logger effective level: {logger.getEffectiveLevel()}")

    def log_comprehensive_conversation_analysis(
        self,
        full_conversation: str,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_spans: List[Tuple[int, int]],
        student_spans: List[Tuple[int, int]],
        sample_id: Optional[str] = None,
        is_training: bool = True,
        tokenizer=None,
        coordinate_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Log comprehensive conversation analysis with full text and detailed span mapping.

        Logs exactly once for training and once for evaluation during the entire training run.
        Shows complete conversation text and maps token spans back to character-level text segments.

        Args:
            full_conversation: Complete conversation text without truncation
            input_ids: Token sequence
            labels: Labels with loss masking applied
            teacher_spans: Teacher response token spans
            student_spans: Student response token spans
            sample_id: Sample identifier
            is_training: Whether this is during training
            tokenizer: Tokenizer for token decoding (required for span mapping)
            coordinate_mask: Mask for coordinate tokens (optional)
        """

        # Simple one-time logging check
        if is_training:
            if not self.should_log_training_loss_mask():
                return
        else:
            if not self.should_log_evaluation_loss_mask():
                return

        sample_id = sample_id or "UNKNOWN"
        mode = "TRAINING" if is_training else "EVALUATION"

        logger.debug("=" * 100)
        logger.debug(f"🎯 COMPREHENSIVE CONVERSATION ANALYSIS [{sample_id}] ({mode})")
        logger.debug("=" * 100)

        # Log full conversation text without truncation (with compacted image tokens)
        logger.debug("📄 FULL CONVERSATION TEXT:")
        compacted_conversation = self._compact_image_pad_tokens(full_conversation)
        # Format as proper string for better log visualization
        formatted_conversation = repr(compacted_conversation)
        logger.debug(f"Conversation string: {formatted_conversation}")

        # Calculate basic statistics
        total_tokens = len(labels)
        masked_tokens = (labels == -100).sum().item()
        unmasked_tokens = total_tokens - masked_tokens
        teacher_token_count = sum(end - start for start, end in teacher_spans)
        student_token_count = sum(end - start for start, end in student_spans)

        logger.debug(f"\n📊 CONVERSATION STATISTICS [{sample_id}]:")
        logger.debug(f"  Full conversation length: {len(full_conversation)} characters")
        logger.debug(f"  Total tokens: {total_tokens}")
        logger.debug(
            f"  Masked tokens (-100): {masked_tokens} ({masked_tokens / total_tokens * 100:.1f}%)"
        )
        logger.debug(
            f"  Unmasked tokens (learning): {unmasked_tokens} ({unmasked_tokens / total_tokens * 100:.1f}%)"
        )
        logger.debug(f"  Teacher response tokens: {teacher_token_count}")
        logger.debug(f"  Student response tokens: {student_token_count}")

        # Perform comprehensive span analysis
        if tokenizer is not None:
            self._log_comprehensive_span_analysis(
                full_conversation,
                input_ids,
                labels,
                teacher_spans,
                student_spans,
                coordinate_mask,
                sample_id,
                tokenizer,
            )
        else:
            logger.debug("⚠️ No tokenizer provided - skipping detailed span analysis")

        logger.debug("=" * 100)

    def _log_comprehensive_span_analysis(
        self,
        full_conversation: str,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_spans: List[Tuple[int, int]],
        student_spans: List[Tuple[int, int]],
        coordinate_mask: Optional[torch.Tensor],
        sample_id: str,
        tokenizer,
    ) -> None:
        """
        Log comprehensive span analysis with character-level mapping.

        Args:
            full_conversation: Complete conversation text
            input_ids: Token sequence
            labels: Labels with masking
            teacher_spans: Teacher response token spans
            student_spans: Student response token spans
            coordinate_mask: Coordinate token mask
            sample_id: Sample identifier
            tokenizer: Tokenizer for decoding
        """
        logger.debug(f"\n🎯 COMPREHENSIVE SPAN ANALYSIS [{sample_id}]:")

        # Create token-to-character mapping
        char_spans = self._create_token_to_char_mapping(
            input_ids, tokenizer, full_conversation
        )

        # Extract and log teacher spans
        if teacher_spans:
            logger.debug(f"\n👨‍🏫 TEACHER RESPONSE SPANS ({len(teacher_spans)} spans):")
            teacher_texts = []
            teacher_coord_texts = []

            for i, (start, end) in enumerate(teacher_spans):
                span_text = self._extract_span_text(
                    start, end, char_spans, full_conversation
                )
                teacher_texts.append(span_text)

                # Extract coordinate tokens from this span
                coord_text = self._extract_coordinate_tokens_from_span(
                    start, end, input_ids, coordinate_mask, tokenizer
                )
                if coord_text:
                    teacher_coord_texts.append(coord_text)

                logger.debug(f"  Teacher Span {i + 1} [tokens {start}-{end}]:")
                # Format as string for better log visualization
                if len(span_text) > 200:
                    display_text = span_text[:200] + "..."
                else:
                    display_text = span_text
                logger.debug(f"    Text string: {repr(display_text)}")
                if coord_text:
                    logger.debug(f"    Coordinates: {coord_text}")

            # Summary of all teacher text
            all_teacher_text = " ".join(teacher_texts)
            logger.debug(f"  📝 All Teacher Text ({len(all_teacher_text)} chars):")
            if len(all_teacher_text) > 300:
                display_text = all_teacher_text[:300] + "..."
            else:
                display_text = all_teacher_text
            logger.debug(f"    Teacher text string: {repr(display_text)}")

        # Extract and log student spans
        if student_spans:
            logger.debug(f"\n👨‍🎓 STUDENT RESPONSE SPANS ({len(student_spans)} spans):")
            student_texts = []
            student_coord_texts = []

            for i, (start, end) in enumerate(student_spans):
                span_text = self._extract_span_text(
                    start, end, char_spans, full_conversation
                )
                student_texts.append(span_text)

                # Extract coordinate tokens from this span
                coord_text = self._extract_coordinate_tokens_from_span(
                    start, end, input_ids, coordinate_mask, tokenizer
                )
                if coord_text:
                    student_coord_texts.append(coord_text)

                logger.debug(f"  Student Span {i + 1} [tokens {start}-{end}]:")
                # Format as string for better log visualization
                if len(span_text) > 200:
                    display_text = span_text[:200] + "..."
                else:
                    display_text = span_text
                logger.debug(f"    Text string: {repr(display_text)}")
                if coord_text:
                    logger.debug(f"    Coordinates: {coord_text}")

            # Summary of all student text
            all_student_text = " ".join(student_texts)
            logger.debug(f"  📝 All Student Text ({len(all_student_text)} chars):")
            if len(all_student_text) > 300:
                display_text = all_student_text[:300] + "..."
            else:
                display_text = all_student_text
            logger.debug(f"    Student text string: {repr(display_text)}")

        # Extract and log all valid learning spans
        valid_spans = teacher_spans + student_spans
        if valid_spans:
            logger.debug(f"\n✅ ALL VALID LEARNING SPANS ({len(valid_spans)} spans):")
            all_valid_texts = []

            for start, end in sorted(valid_spans):
                span_text = self._extract_span_text(
                    start, end, char_spans, full_conversation
                )
                all_valid_texts.append(span_text)

            all_valid_text = " ".join(all_valid_texts)
            logger.debug(f"  📝 All Learning Text ({len(all_valid_text)} chars):")
            if len(all_valid_text) > 400:
                display_text = all_valid_text[:400] + "..."
            else:
                display_text = all_valid_text
            logger.debug(f"    Learning text string: {repr(display_text)}")

        # Extract and log coordinate token analysis
        if coordinate_mask is not None:
            self._log_coordinate_span_analysis(
                input_ids, coordinate_mask, tokenizer, sample_id
            )

    def _create_token_to_char_mapping(
        self, input_ids: torch.Tensor, tokenizer, full_conversation: str
    ) -> List[Tuple[int, int]]:
        """
        Create mapping from token indices to character spans in the full conversation.

        Args:
            input_ids: Token sequence
            tokenizer: Tokenizer for decoding
            full_conversation: Complete conversation text

        Returns:
            List of (start_char, end_char) tuples for each token
        """
        char_spans = []
        current_pos = 0

        for i in range(len(input_ids)):
            try:
                # Decode this single token
                token_text = tokenizer.decode([input_ids[i]], skip_special_tokens=False)

                # Find this token in the conversation starting from current position
                token_start = full_conversation.find(token_text, current_pos)
                if token_start != -1:
                    token_end = token_start + len(token_text)
                    char_spans.append((token_start, token_end))
                    current_pos = token_end
                else:
                    # Token not found - use current position
                    char_spans.append((current_pos, current_pos))
            except Exception:
                # Decoding failed - use current position
                char_spans.append((current_pos, current_pos))

        return char_spans

    def _extract_span_text(
        self,
        start_token: int,
        end_token: int,
        char_spans: List[Tuple[int, int]],
        full_conversation: str,
    ) -> str:
        """
        Extract text corresponding to a token span.

        Args:
            start_token: Start token index
            end_token: End token index (exclusive)
            char_spans: Token to character mapping
            full_conversation: Complete conversation text

        Returns:
            Text corresponding to the token span
        """
        if start_token >= len(char_spans) or end_token > len(char_spans):
            return "[span out of bounds]"

        if start_token >= end_token:
            return "[invalid span]"

        # Get character range for this token span
        start_char = char_spans[start_token][0]
        end_char = char_spans[end_token - 1][1]

        return full_conversation[start_char:end_char]

    def _extract_coordinate_tokens_from_span(
        self,
        start_token: int,
        end_token: int,
        input_ids: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor],
        tokenizer,
    ) -> str:
        """
        Extract coordinate tokens from a specific span.

        Args:
            start_token: Start token index
            end_token: End token index (exclusive)
            input_ids: Token sequence
            coordinate_mask: Coordinate token mask
            tokenizer: Tokenizer for decoding

        Returns:
            String representation of coordinate tokens in this span
        """
        if coordinate_mask is None:
            return ""

        coord_tokens = []
        for i in range(start_token, min(end_token, len(coordinate_mask))):
            if coordinate_mask[i]:
                try:
                    token_text = tokenizer.decode(
                        [input_ids[i]], skip_special_tokens=False
                    )
                    coord_tokens.append(token_text)
                except Exception:
                    coord_tokens.append(f"[token_{input_ids[i]}]")

        return (
            self._format_coordinate_tokens_as_string(coord_tokens)
            if coord_tokens
            else ""
        )

    def _format_coordinate_tokens_as_string(self, coord_tokens: List[str]) -> str:
        """
        Format coordinate tokens into a readable string representation.

        Converts tokens like ['<|coord_196|>', '<|coord_71|>', '<|coord_205|>']
        into a compact string format like '[196, 71, 205]' or groups them by geometry.

        Args:
            coord_tokens: List of coordinate token strings

        Returns:
            Formatted string representation of coordinate tokens
        """
        if not coord_tokens:
            return ""

        # Extract coordinate values from tokens
        coord_values = []
        for token in coord_tokens:
            if "<|coord_" in token and "|>" in token:
                try:
                    value_str = token.replace("<|coord_", "").replace("|>", "")
                    coord_values.append(int(value_str))
                except ValueError:
                    # Keep original token if parsing fails
                    coord_values.append(token)
            else:
                coord_values.append(token)

        # Format as compact coordinate list
        if len(coord_values) <= 10:
            # Short list - show all values
            return f"[{', '.join(map(str, coord_values))}]"
        else:
            # Long list - show first few and count
            first_few = coord_values[:6]
            remaining_count = len(coord_values) - 6
            return f"[{', '.join(map(str, first_few))}, ...+{remaining_count}]"

    def _format_coordinate_values_as_string(self, coord_values: List[int]) -> str:
        """
        Format coordinate values into a readable string representation.

        Args:
            coord_values: List of coordinate integer values

        Returns:
            Formatted string representation of coordinate values
        """
        if not coord_values:
            return "[]"

        if len(coord_values) <= 10:
            # Short list - show all values
            return f"[{', '.join(map(str, coord_values))}]"
        else:
            # Long list - show first few and count
            first_few = coord_values[:6]
            remaining_count = len(coord_values) - 6
            return f"[{', '.join(map(str, first_few))}, ...+{remaining_count}]"

    def _log_coordinate_span_analysis(
        self,
        input_ids: torch.Tensor,
        coordinate_mask: torch.Tensor,
        tokenizer,
        sample_id: str,
    ) -> None:
        """
        Log detailed coordinate token analysis.

        Args:
            input_ids: Token sequence
            coordinate_mask: Coordinate token mask
            tokenizer: Tokenizer for decoding
            sample_id: Sample identifier
        """
        coord_positions = torch.where(coordinate_mask)[0].tolist()
        if not coord_positions:
            logger.debug(f"\n🔢 COORDINATE TOKENS: None found")
            return

        logger.debug(f"\n🔢 COORDINATE TOKEN ANALYSIS ({len(coord_positions)} tokens):")

        # Group consecutive coordinate tokens
        coord_groups = []
        current_group = [coord_positions[0]]

        for pos in coord_positions[1:]:
            if pos == current_group[-1] + 1:
                current_group.append(pos)
            else:
                coord_groups.append(current_group)
                current_group = [pos]
        coord_groups.append(current_group)

        # Log each group
        for i, group in enumerate(coord_groups):
            group_tokens = []

            for pos in group:
                try:
                    token_text = tokenizer.decode(
                        [input_ids[pos]], skip_special_tokens=False
                    )
                    group_tokens.append(token_text)
                except Exception:
                    group_tokens.append(f"[token_{input_ids[pos]}]")

        # Extract coordinate values
        coord_values = []
        for pos in coord_positions:
            try:
                token_text = tokenizer.decode(
                    [input_ids[pos]], skip_special_tokens=False
                )
                if "<|coord_" in token_text and "|>" in token_text:
                    value_str = token_text.replace("<|coord_", "").replace("|>", "")
                    try:
                        coord_values.append(int(value_str))
                    except ValueError:
                        pass
            except Exception:
                pass

        if coord_values:
            logger.debug(
                f"  📊 Coordinate Values: min={min(coord_values)}, max={max(coord_values)}, count={len(coord_values)}"
            )
            # Use the new formatter for better coordinate display
            formatted_coords = self._format_coordinate_values_as_string(coord_values)
            logger.debug(f"  📊 Sample Values: {formatted_coords}")


# Global debug logger instance
debug_logger = DebugLogger()
