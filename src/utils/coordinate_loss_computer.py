"""
Coordinate Loss Computer for Bbox-Aware Loss Splitting

This module implements advanced coordinate loss computation with proper bbox span detection.
It replaces the scattered coordinate loss logic in the model wrapper with a unified,
optimized system that only applies coordinate losses within <bbox_start> and <bbox_end> spans.

Key Features:
- Bbox-aware coordinate token detection
- Proper soft expectation regression
- Enhanced L1 and GIoU computation
- Efficient batch processing
- Comprehensive loss tracking and metrics
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.logger_utils import get_logger
from src.utils.coordinate_token_manager import CoordinateTokenManager


class CoordinateLossComputer:
    """
    Advanced coordinate loss computation with bbox-aware processing.

    This class handles:
    - Detection of coordinate tokens within bbox spans
    - Soft expectation regression for coordinate prediction
    - L1 and GIoU losses for bbox-level evaluation
    - Efficient batch processing and loss accumulation
    """

    def __init__(self, coordinate_manager: CoordinateTokenManager):
        """
        Initialize coordinate loss computer.

        Args:
            coordinate_manager: Coordinate token manager for token operations
        """
        self.manager = coordinate_manager
        self.config = coordinate_manager.config
        self.logger = get_logger("coordinate_loss_computer")

        # Performance metrics
        self._metrics = {
            "total_forward_passes": 0,
            "bbox_spans_processed": 0,
            "coordinate_tokens_processed": 0,
            "regular_tokens_processed": 0,
            "loss_computations": 0,
            "validation_errors": 0,
        }

        self.logger.info("✅ CoordinateLossComputer initialized")
        self.logger.info(
            f"   Coordinate tokens enabled: {self.config.enable_coordinate_tokens}"
        )
        self.logger.info(f"   Max coordinate value: {self.config.max_coord_value}")

    def compute_coordinate_aware_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute coordinate-aware loss with enhanced debugging and validation.

        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if not self.config.enable_coordinate_tokens:
            self.logger.debug(f"📄 Coordinate tokens disabled, using standard loss")
            return self._compute_standard_loss(logits, labels, attention_mask)

        self.logger.debug(f"🎯 Computing coordinate-aware loss")
        self.logger.debug(
            f"   📊 Input shapes: logits={logits.shape}, labels={labels.shape}"
        )
        self.logger.debug(
            f"   🔧 Config enabled: {self.config.enable_coordinate_tokens}"
        )
        self.logger.debug(
            f"   🎯 Box tokens: start_id={self.config.box_start_id}, end_id={self.config.box_end_id}"
        )
        self._metrics["total_forward_passes"] += 1

        # Input validation
        if not self._validate_inputs(logits, labels):
            self.logger.warning("❌ Input validation failed, returning zero loss")
            return self._create_zero_loss(logits.device, logits.dtype)

        # Shift for causal modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tensors with proper vocabulary size
        flat_logits = shift_logits.view(-1, self.manager.extended_vocab_size)
        flat_labels = shift_labels.view(-1)

        # Filter out ignore_index (-100) with validation
        valid_mask = flat_labels != -100
        if not valid_mask.any():
            self.logger.warning("⚠️ No valid labels found, returning zero loss")
            return self._create_zero_loss(logits.device, logits.dtype)

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        self.logger.debug(
            f"   📊 Token stats: total={flat_labels.size(0)}, valid={valid_labels.size(0)}"
        )

        # Detect bbox spans with enhanced validation - use input_ids for span detection
        self.logger.debug(f"   🔍 DEBUGGING input_ids parameter:")
        self.logger.debug(f"     input_ids is None: {input_ids is None}")
        if input_ids is not None:
            self.logger.debug(f"     input_ids shape: {input_ids.shape}")
            self.logger.debug(f"     input_ids sample: {input_ids[0].tolist()}")
            # USE FULL input_ids WITHOUT SHIFTING to detect bbox spans
            # The span detection needs complete sequences with box_start and box_end
            bbox_spans = self._detect_bbox_spans_batch_enhanced(input_ids)
            # Adjust spans to match shifted sequence indices
            bbox_spans = [
                [(max(0, start - 1), max(0, end - 1)) for start, end in spans]
                for spans in bbox_spans
            ]
            self.logger.debug(f"     🎯 Using full input_ids for bbox span detection")
            self.logger.debug(f"     🔧 Adjusted spans for shifted sequence")
        else:
            # Fallback to labels (may miss spans if box tokens are ignored)
            bbox_spans = self._detect_bbox_spans_batch_enhanced(shift_labels)
            self.logger.debug(
                f"   ⚠️ Using labels for bbox span detection (may miss ignored box tokens)"
            )
        self.logger.debug(f"   📦 Detected bbox spans: {len(bbox_spans)} spans")
        for i, spans in enumerate(bbox_spans):
            self.logger.debug(f"     Sample {i}: {len(spans)} spans")
            for j, (start, end) in enumerate(spans):
                self.logger.debug(f"       Span {j}: [{start}:{end}]")

        # Create coordinate mask with validation
        coordinate_mask_full = self._create_validated_coordinate_mask(
            shift_labels, bbox_spans, valid_mask
        )
        coord_count = (
            coordinate_mask_full.sum().item()
            if coordinate_mask_full.sum().numel() > 0
            else 0
        )
        self.logger.debug(f"   🎯 Coordinate tokens detected: {coord_count}")

        # STRICT VALIDATION: If bbox spans are detected but no coordinate tokens found, raise error
        if (
            bbox_spans
            and any(len(spans) > 0 for spans in bbox_spans)
            and coord_count == 0
        ):
            # Check if coordinate tokens exist in the input but are masked as -100
            coord_token_positions = []
            for i, spans in enumerate(bbox_spans):
                for start, end in spans:
                    tokens_in_span = shift_labels[i, start:end]
                    for j, token in enumerate(tokens_in_span):
                        if (
                            self.coord_token_start
                            <= token.item()
                            < self.coord_token_end
                        ):
                            coord_token_positions.append((i, start + j, token.item()))

            if coord_token_positions:
                # Coordinate tokens exist but are being filtered out
                self.logger.error(
                    f"❌ CRITICAL: Coordinate tokens found in input but filtered out!"
                )
                self.logger.error(
                    f"   Bbox spans detected: {sum(len(spans) for spans in bbox_spans)}"
                )
                self.logger.error(
                    f"   Coordinate token positions: {coord_token_positions}"
                )
                self.logger.error(
                    f"   This means coordinate tokens are being set to -100 in labels!"
                )
                raise RuntimeError(
                    f"Coordinate tokens detected in input ({len(coord_token_positions)} tokens) "
                    f"but filtered out by valid_mask. This indicates coordinate tokens are being "
                    f"set to -100 (ignore index) in labels during data preprocessing."
                )
            else:
                # No coordinate tokens found at all
                self.logger.error(
                    f"❌ CRITICAL: Bbox spans detected but no coordinate tokens found!"
                )
                self.logger.error(f"   Bbox spans: {bbox_spans}")
                self.logger.error(
                    f"   This indicates coordinate token conversion failed!"
                )
                raise RuntimeError(
                    f"Bbox spans detected ({sum(len(spans) for spans in bbox_spans)} spans) "
                    f"but no coordinate tokens found in the expected range "
                    f"[{self.coord_token_start}, {self.coord_token_end})"
                )

        # CRITICAL FIX: After fixing coordinate token labels, update valid_mask to include them
        valid_mask_updated = shift_labels.view(-1) != -100

        # Apply coordinate mask to valid tokens only
        coordinate_mask_valid = coordinate_mask_full[valid_mask_updated]

        self.logger.debug(f"   🔍 DEBUGGING mask alignment:")
        self.logger.debug(f"     Full mask size: {coordinate_mask_full.size(0)}")
        self.logger.debug(f"     Valid mask size: {valid_mask_updated.size(0)}")
        self.logger.debug(
            f"     Valid coordinate mask size: {coordinate_mask_valid.size(0)}"
        )

        # Update valid logits and labels to use the updated valid mask
        valid_logits = flat_logits[valid_mask_updated]
        valid_labels = flat_labels[valid_mask_updated]

        self.logger.debug(f"     Valid logits size: {valid_logits.size(0)}")

        # Debug: Show some sample labels
        sample_size = min(20, valid_labels.size(0))
        sample_labels = valid_labels[:sample_size].tolist()
        self.logger.debug(f"   📋 Sample labels: {sample_labels}")

        # Split tokens with bounds checking
        coord_indices, regular_indices = self._split_token_indices_safely(
            coordinate_mask_valid, valid_logits.size(0)
        )

        self.logger.debug(
            f"   📊 Split: coordinate={coord_indices.size(0)}, regular={regular_indices.size(0)}"
        )

        # Initialize loss computation
        total_loss = torch.tensor(
            0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
        )
        loss_components = self._initialize_loss_components(
            valid_labels.size(0), coord_indices.size(0), regular_indices.size(0)
        )

        # Compute regular token loss with validation
        if regular_indices.size(0) > 0:
            regular_loss = self._compute_regular_token_loss_enhanced(
                valid_logits[regular_indices], valid_labels[regular_indices]
            )
            weighted_regular_loss = self._apply_loss_weight(
                regular_loss, self.config.regular_loss_weight, "regular"
            )
            total_loss = total_loss + weighted_regular_loss
            loss_components["regular_loss"] = regular_loss.item()
            self._metrics["regular_tokens_processed"] += regular_indices.size(0)

        # Compute coordinate token loss with enhanced processing
        if coord_indices.size(0) > 0:
            coord_loss_dict = self._compute_coordinate_token_loss_enhanced(
                valid_logits[coord_indices], valid_labels[coord_indices], bbox_spans
            )

            # Apply improved loss weighting
            total_coord_loss = self._combine_coordinate_losses(coord_loss_dict)
            weighted_coord_loss = self._apply_loss_weight(
                total_coord_loss, self.config.coordinate_loss_weight, "coordinate"
            )
            total_loss = total_loss + weighted_coord_loss

            # Update loss components with validation
            loss_components.update(coord_loss_dict)
            self._metrics["coordinate_tokens_processed"] += coord_indices.size(0)
        else:
            self.logger.debug(f"   ⚠️ No coordinate tokens found in batch")

        # Update metrics and validate final loss
        self._update_metrics_and_validate_loss(bbox_spans, total_loss)

        # STRICT VALIDATION: Ensure coordinate losses are non-zero when expected
        if bbox_spans and any(len(spans) > 0 for spans in bbox_spans):
            # We detected bbox spans, so coordinate losses should be non-zero
            total_coord_loss_value = sum(
                [
                    loss_components.get("focal_loss", 0.0),
                    loss_components.get("l1_loss", 0.0),
                    loss_components.get("giou_loss", 0.0),
                ]
            )
            if total_coord_loss_value == 0.0:
                self.logger.error(
                    f"❌ CRITICAL: Bbox spans detected but all coordinate losses are zero!"
                )
                self.logger.error(f"   Bbox spans: {bbox_spans}")
                self.logger.error(f"   Loss components: {loss_components}")
                self.logger.error(f"   Coordinate indices: {coord_indices.size(0)}")
                self.logger.error(f"   Regular indices: {regular_indices.size(0)}")
                raise RuntimeError(
                    f"Bbox spans detected ({sum(len(spans) for spans in bbox_spans)} spans) "
                    f"but all coordinate losses are zero. This indicates coordinate tokens are not "
                    f"being processed correctly in the loss computation."
                )

        return total_loss, loss_components

    def _compute_standard_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute standard cross-entropy loss when coordinate tokens disabled."""
        # Shift for causal modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten and compute loss
        flat_logits = shift_logits.view(-1, logits.size(-1))
        flat_labels = shift_labels.view(-1)

        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        loss_components = {
            "regular_loss": loss.item(),
            "focal_loss": 0.0,
            "l1_loss": 0.0,
            "giou_loss": 0.0,
        }

        return loss, loss_components

    def _detect_bbox_spans_batch(
        self, labels: torch.Tensor
    ) -> List[List[Tuple[int, int]]]:
        """
        Detect bbox spans in batch of label sequences.

        Args:
            labels: Label tensor (batch_size, seq_len)

        Returns:
            List of bbox spans for each sequence in batch
        """
        batch_bbox_spans = []

        for batch_idx in range(labels.size(0)):
            sequence_spans = self._detect_bbox_spans_single(labels[batch_idx])
            batch_bbox_spans.append(sequence_spans)

        return batch_bbox_spans

    def _detect_bbox_spans_single(self, labels: torch.Tensor) -> List[Tuple[int, int]]:
        """Detect bbox spans in single label sequence."""
        spans = []
        i = 0

        while i < len(labels):
            # Look for box_start token
            if labels[i] == self.config.box_start_id:
                start_idx = i
                i += 1

                # Look for box_end token
                while i < len(labels) and labels[i] != self.config.box_end_id:
                    i += 1

                if i < len(labels):  # Found box_end
                    end_idx = i + 1  # Include box_end token
                    spans.append((start_idx, end_idx))
                    i += 1
                else:
                    # No matching box_end found - incomplete span
                    self.logger.warning(f"Incomplete bbox span starting at {start_idx}")
                    break
            else:
                i += 1

        return spans

    def _create_bbox_aware_coordinate_mask(
        self,
        labels: torch.Tensor,
        bbox_spans: List[List[Tuple[int, int]]],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create coordinate mask only for tokens within bbox spans.

        Args:
            labels: Shifted labels (batch_size, seq_len)
            bbox_spans: Detected bbox spans per sequence
            valid_mask: Mask for valid (non-ignored) tokens

        Returns:
            Boolean mask for coordinate tokens within bbox spans
        """
        batch_size, seq_len = labels.shape
        coordinate_mask = torch.zeros(
            batch_size * seq_len, dtype=torch.bool, device=labels.device
        )

        for batch_idx, spans in enumerate(bbox_spans):
            for start_idx, end_idx in spans:
                # Only mark coordinate tokens within bbox spans (exclude box_start/box_end)
                for pos in range(start_idx + 1, end_idx - 1):
                    if pos < seq_len:
                        flat_idx = batch_idx * seq_len + pos
                        token_id = labels[batch_idx, pos]

                        # Check if this is a coordinate token
                        if self.manager.is_coordinate_token(token_id):
                            coordinate_mask[flat_idx] = True

        # Don't apply valid_mask here - it will be applied at caller level
        return coordinate_mask

    def _compute_regular_token_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard cross-entropy loss for regular tokens."""
        # Ensure labels are within original vocabulary range
        valid_regular_mask = (labels >= 0) & (labels < self.manager.original_vocab_size)

        if not valid_regular_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        regular_logits = logits[valid_regular_mask, : self.manager.original_vocab_size]
        regular_labels = labels[valid_regular_mask]

        # Additional bounds checking for safety
        regular_labels = torch.clamp(
            regular_labels, 0, self.manager.original_vocab_size - 1
        )

        return F.cross_entropy(regular_logits, regular_labels)

    def _validate_inputs(self, logits: torch.Tensor, labels: torch.Tensor) -> bool:
        """Validate input tensors for coordinate loss computation."""
        if logits.size(0) == 0 or labels.size(0) == 0:
            self.logger.warning("⚠️ Empty logits or labels tensor")
            return False

        if logits.size(-1) != self.manager.extended_vocab_size:
            self.logger.warning(
                f"⚠️ Logits vocab size mismatch: {logits.size(-1)} vs {self.manager.extended_vocab_size}"
            )
            return False

        return True

    def _detect_bbox_spans_batch_enhanced(
        self, labels: torch.Tensor
    ) -> List[List[Tuple[int, int]]]:
        """Enhanced bbox span detection with validation."""
        batch_bbox_spans = []

        for batch_idx in range(labels.size(0)):
            sequence_spans = self._detect_bbox_spans_single_enhanced(labels[batch_idx])
            batch_bbox_spans.append(sequence_spans)

        total_spans = sum(len(spans) for spans in batch_bbox_spans)
        self.logger.debug(
            f"   📋 Detected {total_spans} bbox spans across {labels.size(0)} sequences"
        )

        return batch_bbox_spans

    def _detect_bbox_spans_single_enhanced(
        self, labels: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Enhanced single sequence bbox span detection."""
        spans = []
        i = 0

        self.logger.debug(
            f"     🔍 Searching for bbox spans in sequence of length {len(labels)}"
        )
        self.logger.debug(
            f"     🎯 Looking for box_start_id={self.config.box_start_id}, box_end_id={self.config.box_end_id}"
        )

        # Debug: Check if we have any box tokens at all
        start_tokens = (labels == self.config.box_start_id).sum().item()
        end_tokens = (labels == self.config.box_end_id).sum().item()
        self.logger.debug(
            f"     📊 Found {start_tokens} box_start tokens and {end_tokens} box_end tokens"
        )

        # Show first few tokens for debugging
        sample_size = min(10, len(labels))
        sample_tokens = labels[:sample_size].tolist()
        self.logger.debug(f"     📋 First {sample_size} tokens: {sample_tokens}")

        while i < len(labels):
            if labels[i] == self.config.box_start_id:
                start_idx = i
                i += 1

                # Look for box_end token with bounds checking
                while i < len(labels) and labels[i] != self.config.box_end_id:
                    i += 1

                if i < len(labels):
                    end_idx = i + 1
                    span_length = end_idx - start_idx

                    # Debug: Show tokens within the span
                    span_tokens = labels[start_idx:end_idx].tolist()
                    self.logger.debug(
                        f"     📦 Span tokens [{start_idx}:{end_idx}]: {span_tokens}"
                    )

                    if span_length >= 6:  # Minimum: box_start + 4 coords + box_end
                        spans.append((start_idx, end_idx))
                        self.logger.debug(
                            f"     ✅ Valid bbox span: [{start_idx}:{end_idx}] (length={span_length})"
                        )
                    else:
                        self.logger.debug(
                            f"     ⚠️ Short bbox span: [{start_idx}:{end_idx}] (length={span_length})"
                        )
                    i += 1
                else:
                    self.logger.warning(
                        f"⚠️ Incomplete bbox span starting at {start_idx}"
                    )
                    break
            else:
                i += 1

        self.logger.debug(f"     📦 Total valid bbox spans found: {len(spans)}")
        return spans

    def _create_validated_coordinate_mask(
        self,
        labels: torch.Tensor,
        bbox_spans: List[List[Tuple[int, int]]],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create coordinate mask with enhanced validation."""
        batch_size, seq_len = labels.shape
        coordinate_mask = torch.zeros(
            batch_size * seq_len, dtype=torch.bool, device=labels.device
        )

        coord_token_count = 0
        self.logger.debug(f"   🔍 DEBUGGING coordinate mask creation:")
        self.logger.debug(
            f"     Coordinate range: [{self.manager.coord_start_id}, {self.manager.coord_end_id})"
        )

        for batch_idx, spans in enumerate(bbox_spans):
            for span_idx, (start_idx, end_idx) in enumerate(spans):
                # Only mark coordinate tokens within bbox spans (exclude box_start/box_end)
                for pos in range(start_idx + 1, end_idx - 1):
                    if pos < seq_len:
                        flat_idx = batch_idx * seq_len + pos
                        token_id = labels[batch_idx, pos].item()

                        is_coord = self.manager.is_coordinate_token(token_id)
                        if span_idx < 3:  # Debug first few spans
                            self.logger.debug(
                                f"     Pos {pos}: token_id={token_id}, is_coord={is_coord}"
                            )

                        if is_coord:
                            coordinate_mask[flat_idx] = True
                            coord_token_count += 1

        self.logger.debug(
            f"   🎯 Created coordinate mask: {coord_token_count} coordinate tokens"
        )

        # CRITICAL FIX: Handle coordinate tokens with -100 labels correctly
        # In coordinate token training, coordinate tokens should have their token IDs in labels, not -100
        # But if data preprocessing is setting them to -100, we need to fix this here
        if self.config.enable_coordinate_tokens and len(bbox_spans[0]) > 0:
            # Count how many coordinate tokens have -100 in labels
            coord_tokens_with_ignore = 0
            coord_tokens_with_valid_labels = 0

            for batch_idx, spans in enumerate(bbox_spans):
                for start_idx, end_idx in spans:
                    for pos in range(start_idx + 1, end_idx - 1):
                        if pos < labels.shape[1]:
                            token_id = labels[batch_idx, pos].item()
                            if self.manager.is_coordinate_token(token_id):
                                # This coordinate token is in the input
                                label_value = labels[batch_idx, pos].item()
                                if label_value == -100:
                                    coord_tokens_with_ignore += 1
                                else:
                                    coord_tokens_with_valid_labels += 1

            self.logger.debug(f"🔍 Coordinate token label analysis:")
            self.logger.debug(
                f"   Coordinate tokens enabled: {self.config.enable_coordinate_tokens}"
            )
            self.logger.debug(f"   Bbox spans detected: {len(bbox_spans[0])}")
            self.logger.debug(f"   Coordinate tokens in input: {coord_token_count}")
            self.logger.debug(
                f"   Coordinate tokens with -100 labels: {coord_tokens_with_ignore}"
            )
            self.logger.debug(
                f"   Coordinate tokens with valid labels: {coord_tokens_with_valid_labels}"
            )

            if coord_tokens_with_ignore > 0:
                self.logger.warning(
                    f"⚠️ FIXING: Coordinate tokens in input have -100 in labels!"
                )
                self.logger.warning(
                    f"   This means coordinate tokens are being generated in input but labels are set to ignore (-100)"
                )
                self.logger.warning(
                    f"   FIXING: Setting coordinate token labels to match input coordinate tokens"
                )

                # FIX: Set coordinate token labels to match input tokens
                for batch_idx, spans in enumerate(bbox_spans):
                    for start_idx, end_idx in spans:
                        for pos in range(start_idx + 1, end_idx - 1):
                            if pos < labels.shape[1]:
                                token_id = labels[batch_idx, pos].item()
                                if self.manager.is_coordinate_token(token_id):
                                    # Set label to match input coordinate token
                                    labels[batch_idx, pos] = token_id

                self.logger.info(
                    f"✅ FIXED: Set {coord_tokens_with_ignore} coordinate token labels to match input tokens"
                )

                # Show sample span for debugging
                if len(bbox_spans[0]) > 0:
                    start_idx, end_idx = bbox_spans[0][0]
                    span_input_tokens = labels[
                        0, start_idx:end_idx
                    ].tolist()  # This shows input tokens in this context
                    self.logger.debug(
                        f"   Sample span tokens after fix: {span_input_tokens}"
                    )

        return coordinate_mask

    def _split_token_indices_safely(
        self, coordinate_mask: torch.Tensor, max_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Safely split token indices with bounds checking."""
        coord_indices = torch.nonzero(coordinate_mask, as_tuple=False).squeeze(-1)
        regular_indices = torch.nonzero(~coordinate_mask, as_tuple=False).squeeze(-1)

        # Ensure indices are within bounds
        max_valid_idx = max_size - 1
        if coord_indices.numel() > 0:
            coord_indices = coord_indices[coord_indices <= max_valid_idx]
        if regular_indices.numel() > 0:
            regular_indices = regular_indices[regular_indices <= max_valid_idx]

        return coord_indices, regular_indices

    def _initialize_loss_components(
        self, total_tokens: int, coord_tokens: int, regular_tokens: int
    ) -> Dict[str, float]:
        """Initialize loss components dictionary."""
        return {
            "focal_loss": 0.0,
            "regular_loss": 0.0,
            "l1_loss": 0.0,
            "giou_loss": 0.0,
            "detection_loss": 0.0,
            "total_tokens": total_tokens,
            "coordinate_tokens": coord_tokens,
            "regular_tokens": regular_tokens,
        }

    def _compute_regular_token_loss_enhanced(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced regular token loss computation with validation."""
        # Ensure labels are within original vocabulary range
        valid_regular_mask = (labels >= 0) & (labels < self.manager.original_vocab_size)

        if not valid_regular_mask.any():
            self.logger.debug(
                f"   ⚠️ No valid regular tokens in range [0, {self.manager.original_vocab_size})"
            )
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        regular_logits = logits[valid_regular_mask, : self.manager.original_vocab_size]
        regular_labels = labels[valid_regular_mask]

        # Additional bounds checking
        regular_labels = torch.clamp(
            regular_labels, 0, self.manager.original_vocab_size - 1
        )

        regular_loss = F.cross_entropy(regular_logits, regular_labels)
        self.logger.debug(
            f"   📊 Regular loss: {regular_loss.item():.6f} ({valid_regular_mask.sum()} tokens)"
        )

        return regular_loss

    def _apply_loss_weight(
        self, loss: torch.Tensor, weight: float, loss_type: str
    ) -> torch.Tensor:
        """Apply loss weight with logging."""
        weighted_loss = weight * loss
        self.logger.debug(
            f"   ⚔️ {loss_type} loss: {loss.item():.6f} * {weight} = {weighted_loss.item():.6f}"
        )
        return weighted_loss

    def _compute_coordinate_token_loss_enhanced(
        self,
        coord_logits: torch.Tensor,
        coord_labels: torch.Tensor,
        bbox_spans: List[List[Tuple[int, int]]],
    ) -> Dict[str, float]:
        """Enhanced coordinate token loss computation with better validation."""
        if coord_logits.size(0) == 0:
            self.logger.debug(f"   ⚠️ No coordinate tokens to process")
            return {
                "detection_loss": 0.0,
                "focal_loss": 0.0,
                "l1_loss": 0.0,
                "giou_loss": 0.0,
            }

        self.logger.debug(f"   🎯 Processing {coord_logits.size(0)} coordinate tokens")

        # Extract and validate coordinate portion of logits
        coord_start = self.manager.coord_start_id
        coord_end = self.manager.coord_end_id

        if coord_start >= coord_logits.size(-1) or coord_end > coord_logits.size(-1):
            self.logger.warning(
                f"Coordinate range [{coord_start}:{coord_end}] out of bounds for logits size {coord_logits.size(-1)}"
            )
            return {
                "detection_loss": 0.0,
                "focal_loss": 0.0,
                "l1_loss": 0.0,
                "giou_loss": 0.0,
            }

        coord_only_logits = coord_logits[:, coord_start:coord_end]

        # Convert and validate coordinate labels
        coord_indices = coord_labels - coord_start
        coord_indices = torch.clamp(coord_indices, 0, self.config.max_coord_value - 1)

        # Validate indices bounds
        max_valid_idx = coord_only_logits.size(-1) - 1
        if max_valid_idx < 0:
            self.logger.warning(
                f"Invalid coordinate logits size: {coord_only_logits.shape}"
            )
            return {
                "detection_loss": 0.0,
                "focal_loss": 0.0,
                "l1_loss": 0.0,
                "giou_loss": 0.0,
            }

        coord_indices = torch.clamp(coord_indices, 0, max_valid_idx)

        # Enhanced soft expectation computation
        temperature = self.config.soft_expectation_temperature
        soft_weights = F.softmax(coord_only_logits / temperature, dim=-1)

        coord_range = torch.arange(
            coord_only_logits.size(-1), device=coord_logits.device, dtype=torch.float32
        )
        expected_coords = torch.sum(soft_weights * coord_range, dim=-1)

        # Compute individual loss components
        detection_loss = F.l1_loss(expected_coords, coord_indices.float())
        focal_loss = self._compute_focal_loss_enhanced(soft_weights, coord_indices)
        l1_loss, giou_loss = self._compute_bbox_level_losses_enhanced(
            expected_coords, coord_indices.float()
        )

        self.logger.debug(
            f"   📊 Coordinate losses: detection={detection_loss.item():.6f}, focal={focal_loss.item():.6f}, l1={l1_loss.item():.6f}, giou={giou_loss.item():.6f}"
        )

        return {
            "detection_loss": detection_loss.item(),
            "focal_loss": focal_loss.item(),
            "l1_loss": l1_loss.item(),
            "giou_loss": giou_loss.item(),
        }

    def _combine_coordinate_losses(
        self, coord_loss_dict: Dict[str, float]
    ) -> torch.Tensor:
        """Combine coordinate loss components with proper weighting."""
        # Primary loss: detection (L1 between expected and target coordinates)
        detection_weight = 1.0
        focal_weight = 0.1  # Encourage sharp distributions
        l1_weight = 0.2  # Bbox-level L1 loss
        giou_weight = 0.1  # Bbox-level GIoU loss

        total_loss = (
            detection_weight * coord_loss_dict["detection_loss"]
            + focal_weight * coord_loss_dict["focal_loss"]
            + l1_weight * coord_loss_dict["l1_loss"]
            + giou_weight * coord_loss_dict["giou_loss"]
        )

        # Use CPU as fallback device, ensure gradients are enabled
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
            except:
                device = torch.device("cpu")

        self.logger.debug(f"   ⚖️ Combined coordinate loss: {total_loss:.6f}")
        # Create tensor with gradients enabled
        loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
        loss_tensor.requires_grad_(True)
        return loss_tensor

    def _update_metrics_and_validate_loss(
        self, bbox_spans: List[List[Tuple[int, int]]], total_loss: torch.Tensor
    ):
        """Update metrics and validate final loss value."""
        self._metrics["loss_computations"] += 1
        self._metrics["bbox_spans_processed"] += sum(len(spans) for spans in bbox_spans)

        # Validate loss is not NaN or infinite
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.logger.error(f"❌ Invalid loss detected: {total_loss}")
        elif total_loss.item() == 0.0:
            self.logger.debug(
                f"   ⚠️ Zero loss computed - may indicate no trainable tokens"
            )
        else:
            self.logger.debug(f"   ✅ Final total loss: {total_loss.item():.6f}")

    def _compute_focal_loss_enhanced(
        self, soft_weights: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced focal loss computation with better validation."""
        alpha = self.config.focal_loss_alpha
        gamma = self.config.focal_loss_gamma

        # Validate targets are within bounds
        targets = torch.clamp(targets, 0, soft_weights.size(-1) - 1)

        # Get target probabilities with bounds checking
        target_probs = soft_weights.gather(1, targets.unsqueeze(-1)).squeeze(-1)

        # Add numerical stability
        epsilon = 1e-8
        target_probs = torch.clamp(target_probs, epsilon, 1.0 - epsilon)

        # Compute focal weight with stability checks
        focal_weight = alpha * torch.pow(1 - target_probs, gamma)

        # Compute cross-entropy loss with numerical stability
        ce_loss = -torch.log(target_probs)

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss
        mean_focal_loss = focal_loss.mean()

        # Validate result
        if torch.isnan(mean_focal_loss) or torch.isinf(mean_focal_loss):
            self.logger.warning(
                f"⚠️ Invalid focal loss: {mean_focal_loss}, using fallback"
            )
            return torch.tensor(
                0.0, device=soft_weights.device, dtype=soft_weights.dtype
            )

        return mean_focal_loss

    def _compute_bbox_level_losses_enhanced(
        self, predicted_coords: torch.Tensor, target_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced bbox-level loss computation with better validation.

        Args:
            predicted_coords: Predicted coordinates (N,)
            target_coords: Target coordinates (N,)

        Returns:
            Tuple of (l1_loss, giou_loss)
        """
        if predicted_coords.size(0) == 0:
            zero_loss = torch.tensor(
                0.0, device=predicted_coords.device, dtype=predicted_coords.dtype
            )
            return zero_loss, zero_loss

        # Handle coordinates that aren't multiples of 4
        num_coords = predicted_coords.size(0)
        if num_coords % 4 != 0:
            pad_size = 4 - (num_coords % 4)
            if pad_size < 4:
                predicted_coords = F.pad(predicted_coords, (0, pad_size), value=0.0)
                target_coords = F.pad(target_coords, (0, pad_size), value=0.0)
                self.logger.debug(
                    f"   🔄 Padded coordinates from {num_coords} to {predicted_coords.size(0)}"
                )

        # Reshape to bboxes with validation
        num_boxes = predicted_coords.size(0) // 4
        if num_boxes == 0:
            zero_loss = torch.tensor(
                0.0, device=predicted_coords.device, dtype=predicted_coords.dtype
            )
            return zero_loss, zero_loss

        pred_boxes = predicted_coords.view(num_boxes, 4)
        target_boxes = target_coords.view(num_boxes, 4)

        # Validate coordinate ranges
        max_coord = float(self.config.max_coord_value - 1)
        pred_boxes = torch.clamp(pred_boxes, 0, max_coord)
        target_boxes = torch.clamp(target_boxes, 0, max_coord)

        # Normalize for GIoU computation
        pred_boxes_norm = pred_boxes / max_coord
        target_boxes_norm = target_boxes / max_coord

        # Compute losses with validation
        l1_loss = F.l1_loss(pred_boxes, target_boxes)
        giou_loss = self._compute_giou_loss_enhanced(pred_boxes_norm, target_boxes_norm)

        # Validate results
        if torch.isnan(l1_loss) or torch.isinf(l1_loss):
            self.logger.warning(f"⚠️ Invalid L1 loss: {l1_loss}")
            l1_loss = torch.tensor(
                0.0, device=predicted_coords.device, dtype=predicted_coords.dtype
            )

        if torch.isnan(giou_loss) or torch.isinf(giou_loss):
            self.logger.warning(f"⚠️ Invalid GIoU loss: {giou_loss}")
            giou_loss = torch.tensor(
                0.0, device=predicted_coords.device, dtype=predicted_coords.dtype
            )

        self.logger.debug(
            f"   📊 Bbox losses: L1={l1_loss.item():.6f}, GIoU={giou_loss.item():.6f} ({num_boxes} boxes)"
        )

        return l1_loss, giou_loss

    def _compute_giou_loss_enhanced(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced Generalized IoU loss computation with better numerical stability."""
        if pred_boxes.size(0) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        # Ensure boxes are in proper format with enhanced validation
        pred_boxes = torch.stack(
            [
                torch.min(pred_boxes[:, 0], pred_boxes[:, 2]),  # x1
                torch.min(pred_boxes[:, 1], pred_boxes[:, 3]),  # y1
                torch.max(pred_boxes[:, 0], pred_boxes[:, 2]),  # x2
                torch.max(pred_boxes[:, 1], pred_boxes[:, 3]),  # y2
            ],
            dim=1,
        )

        target_boxes = torch.stack(
            [
                torch.min(target_boxes[:, 0], target_boxes[:, 2]),  # x1
                torch.min(target_boxes[:, 1], target_boxes[:, 3]),  # y1
                torch.max(target_boxes[:, 0], target_boxes[:, 2]),  # x2
                torch.max(target_boxes[:, 1], target_boxes[:, 3]),  # y2
            ],
            dim=1,
        )

        # Compute intersection with numerical stability
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Compute union with numerical stability
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        )
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )

        # Clamp areas to prevent negative values
        pred_area = torch.clamp(pred_area, min=1e-6)
        target_area = torch.clamp(target_area, min=1e-6)

        union_area = pred_area + target_area - inter_area
        union_area = torch.clamp(union_area, min=1e-6)

        # Compute IoU with numerical stability
        iou = inter_area / union_area
        iou = torch.clamp(iou, min=0, max=1)

        # Compute enclosing area for GIoU
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        enclose_area = torch.clamp(enclose_area, min=1e-6)

        # Compute GIoU with numerical stability
        giou_term = (enclose_area - union_area) / enclose_area
        giou_term = torch.clamp(giou_term, min=0, max=1)
        giou = iou - giou_term
        giou = torch.clamp(giou, min=-1, max=1)

        # GIoU loss = 1 - GIoU with validation
        giou_loss = 1.0 - giou.mean()

        # Final validation
        if torch.isnan(giou_loss) or torch.isinf(giou_loss):
            self.logger.warning(f"⚠️ Invalid GIoU computation, using fallback")
            return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        return torch.clamp(giou_loss, min=0, max=2)  # Reasonable bounds for GIoU loss

    def _create_zero_loss(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Create zero loss tensor and components."""
        zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
        zero_components = {
            "focal_loss": 0.0,
            "regular_loss": 0.0,
            "l1_loss": 0.0,
            "giou_loss": 0.0,
        }
        return zero_loss, zero_components

    def get_metrics(self) -> Dict[str, int]:
        """Get performance metrics."""
        return self._metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = {
            "total_forward_passes": 0,
            "bbox_spans_processed": 0,
            "coordinate_tokens_processed": 0,
            "regular_tokens_processed": 0,
            "loss_computations": 0,
            "validation_errors": 0,
        }


def create_coordinate_loss_computer(
    coordinate_manager: CoordinateTokenManager,
) -> CoordinateLossComputer:
    """
    Factory function to create coordinate loss computer.

    Args:
        coordinate_manager: Coordinate token manager

    Returns:
        Configured CoordinateLossComputer
    """
    return CoordinateLossComputer(coordinate_manager)
