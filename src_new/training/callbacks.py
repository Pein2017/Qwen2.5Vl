"""
Training callbacks for Qwen2.5-VL.

This module provides callbacks for training including loss tracking and
best checkpoint creation with descriptive naming.

Key Components:
- LossTracker: Tracks loss components with moving averages
- BestCheckpointCallback: Creates best-{step}-{eval_loss} checkpoints
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


def get_callback_logger() -> logging.Logger:
    """Get rank-aware logger for callbacks."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("callbacks")
    except ImportError:
        return logging.getLogger("callbacks")


def get_rank_info() -> tuple[int, int]:
    """
    Get current rank and world size for distributed training.

    Returns:
        Tuple of (rank, world_size)
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except (ImportError, RuntimeError):
        pass

    # Fallback for non-distributed training
    return 0, 1


@dataclass
class LossTracker:
    """
    Track and compute moving averages for multi-component loss.

    This class maintains rolling averages of different loss components
    to provide smooth metrics for monitoring training progress.
    """

    # Configuration
    window_size: int = 100

    # Loss component history
    loss_history: Optional[List[float]] = None
    llm_loss_history: Optional[List[float]] = None
    coordinate_loss_history: Optional[List[float]] = None
    teacher_loss_history: Optional[List[float]] = None
    student_loss_history: Optional[List[float]] = None

    # Granular teacher-student loss component history
    teacher_llm_loss_history: Optional[List[float]] = None
    teacher_l1_loss_history: Optional[List[float]] = None
    student_llm_loss_history: Optional[List[float]] = None
    student_l1_loss_history: Optional[List[float]] = None

    def __post_init__(self):
        """Initialize loss history lists."""
        self.loss_history = []
        self.llm_loss_history = []
        self.coordinate_loss_history = []
        self.teacher_loss_history = []
        self.student_loss_history = []

        # Initialize granular teacher-student loss histories
        self.teacher_llm_loss_history = []
        self.teacher_l1_loss_history = []
        self.student_llm_loss_history = []
        self.student_l1_loss_history = []

    def update(self, loss_components: Any) -> None:
        """
        Update loss history with new components.

        Args:
            loss_components: Loss components to track
        """
        # Extract loss components
        main_loss = self._extract_loss_value(
            loss_components.loss if hasattr(loss_components, "loss") else None
        )
        llm_loss = self._extract_loss_value(
            loss_components.llm_loss if hasattr(loss_components, "llm_loss") else None
        )
        coordinate_loss = self._extract_loss_value(
            loss_components.coordinate_loss
            if hasattr(loss_components, "coordinate_loss")
            else None
        )
        teacher_loss = self._extract_loss_value(
            loss_components.teacher_loss
            if hasattr(loss_components, "teacher_loss")
            else None
        )
        student_loss = self._extract_loss_value(
            loss_components.student_loss
            if hasattr(loss_components, "student_loss")
            else None
        )

        # Extract granular teacher-student loss components
        teacher_llm_loss = self._extract_loss_value(
            loss_components.teacher_llm_loss
            if hasattr(loss_components, "teacher_llm_loss")
            else None
        )
        teacher_l1_loss = self._extract_loss_value(
            loss_components.teacher_l1_loss
            if hasattr(loss_components, "teacher_l1_loss")
            else None
        )
        student_llm_loss = self._extract_loss_value(
            loss_components.student_llm_loss
            if hasattr(loss_components, "student_llm_loss")
            else None
        )
        student_l1_loss = self._extract_loss_value(
            loss_components.student_l1_loss
            if hasattr(loss_components, "student_l1_loss")
            else None
        )

        # Update histories
        self._update_history(self.loss_history, main_loss)
        self._update_history(self.llm_loss_history, llm_loss)
        self._update_history(self.coordinate_loss_history, coordinate_loss)
        self._update_history(self.teacher_loss_history, teacher_loss)
        self._update_history(self.student_loss_history, student_loss)

        # Update granular teacher-student loss histories
        self._update_history(self.teacher_llm_loss_history, teacher_llm_loss)
        self._update_history(self.teacher_l1_loss_history, teacher_l1_loss)
        self._update_history(self.student_llm_loss_history, student_llm_loss)
        self._update_history(self.student_l1_loss_history, student_l1_loss)

    def _extract_loss_value(self, loss: Any) -> Optional[float]:
        """
        Extract float value from loss tensor or return None.

        Args:
            loss: Loss value (tensor, float, or None)

        Returns:
            Float value or None
        """
        if loss is None:
            return None

        if torch.is_tensor(loss):
            return loss.detach().item()

        if isinstance(loss, (int, float)):
            return float(loss)

        return None

    def _update_history(self, history: List[float], value: Optional[float]) -> None:
        """
        Update history list with new value.

        Args:
            history: History list to update
            value: New value to add
        """
        if value is not None:
            history.append(value)
            # Keep history within window size
            if len(history) > self.window_size:
                history.pop(0)

    def get_averages(self) -> Dict[str, float]:
        """
        Get moving averages for all loss components.

        Returns:
            Dictionary with loss component averages
        """
        return {
            "loss": self._compute_average(self.loss_history),
            "llm_loss": self._compute_average(self.llm_loss_history),
            "coordinate_loss": self._compute_average(self.coordinate_loss_history),
            "teacher_loss": self._compute_average(self.teacher_loss_history),
            "student_loss": self._compute_average(self.student_loss_history),
            # Granular teacher-student loss components
            "teacher_llm_loss": self._compute_average(self.teacher_llm_loss_history),
            "teacher_l1_loss": self._compute_average(self.teacher_l1_loss_history),
            "student_llm_loss": self._compute_average(self.student_llm_loss_history),
            "student_l1_loss": self._compute_average(self.student_l1_loss_history),
        }

    def _compute_average(self, history: List[float]) -> Optional[float]:
        """
        Compute average of history list.

        Args:
            history: List of values

        Returns:
            Average value or None if empty
        """
        if not history:
            return None
        return sum(history) / len(history)

    def reset(self) -> None:
        """Reset all loss histories."""
        self.loss_history = []
        self.llm_loss_history = []
        self.coordinate_loss_history = []
        self.teacher_loss_history = []
        self.student_loss_history = []

        # Reset granular teacher-student loss histories
        self.teacher_llm_loss_history = []
        self.teacher_l1_loss_history = []
        self.student_llm_loss_history = []
        self.student_l1_loss_history = []


class BestCheckpointCallback(TrainerCallback):
    """
    DEPRECATED: This callback has been replaced by unified checkpoint management in BBUTrainer.

    The functionality of this callback is now integrated directly into BBUTrainer's
    _save_checkpoint() method, eliminating redundant I/O operations and ensuring
    consistency between regular and best checkpoints.

    Migration: Remove this callback from your training setup. BBUTrainer now handles
    best checkpoint creation automatically using UnifiedCheckpointManager.
    """

    def __init__(self):
        import warnings

        warnings.warn(
            "BestCheckpointCallback is deprecated and will be removed in a future version. "
            "Best checkpoint functionality is now integrated into BBUTrainer. "
            "Remove this callback from your training setup.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.current_best_dir = None
        self.current_best_metric = None
        self.logger = get_callback_logger()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after checkpoint save to check if we have a new best model."""
        # Only process on main process to avoid race conditions
        if not args.should_save:
            return

        # Get current eval_loss from the most recent log entry
        current_eval_loss = self._get_latest_eval_loss(state)
        if current_eval_loss is None:
            return

        # Check if this is a new best (lower eval_loss is better)
        if (
            self.current_best_metric is None
            or current_eval_loss < self.current_best_metric
        ):
            # Get current step
            current_step = state.global_step

            # Find the checkpoint directory that was just saved
            if not args.output_dir:
                return  # No output directory specified
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{current_step}")

            if os.path.exists(checkpoint_dir):
                self._create_best_checkpoint_copy(
                    args.output_dir, checkpoint_dir, current_step, current_eval_loss
                )
                self.current_best_metric = current_eval_loss

    def _get_latest_eval_loss(self, state: TrainerState) -> Optional[float]:
        """Extract the most recent eval_loss from trainer state."""
        if not hasattr(state, "log_history") or not state.log_history:
            return None

        # Search backwards through log history for the most recent eval_loss
        for log_entry in reversed(state.log_history):
            if isinstance(log_entry, dict) and "eval_loss" in log_entry:
                return log_entry["eval_loss"]

        return None

    def _create_best_checkpoint_copy(
        self, output_dir: str, source_checkpoint: str, step: int, eval_loss: float
    ):
        """Create or update the copy of the best checkpoint."""
        # Format eval_loss to 4 decimal places for filename
        loss_str = f"{eval_loss:.4f}"
        new_best_name = f"best-{step}-loss{loss_str}"
        new_best_path = os.path.join(output_dir, new_best_name)

        # Skip if this is already the current best directory
        if self.current_best_dir == new_best_path:
            return

        # Remove old best directory if it exists
        if self.current_best_dir and os.path.exists(self.current_best_dir):
            try:
                shutil.rmtree(self.current_best_dir)
                self.logger.info(
                    f"🗑️ Removed old best checkpoint: {os.path.basename(self.current_best_dir)}"
                )
            except OSError as e:
                self.logger.warning(f"Failed to remove old best checkpoint: {e}")

        # Create new best checkpoint copy
        try:
            shutil.copytree(source_checkpoint, new_best_path)
            self.current_best_dir = new_best_path
            self.logger.info(f"📁 Created best checkpoint copy: {new_best_name}")
            self.logger.info(
                f"📊 Best model at step {step} with eval_loss: {eval_loss:.4f}"
            )

        except OSError as e:
            self.logger.error(f"Failed to create best checkpoint copy: {e}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training to report best checkpoint status."""
        if args.should_save and self.current_best_dir:
            self.logger.info(
                f"✅ Training completed. Best checkpoint available at: {os.path.basename(self.current_best_dir)}"
            )


def create_best_checkpoint_callback() -> BestCheckpointCallback:
    """
    DEPRECATED: Factory function for BestCheckpointCallback.

    This function is deprecated because BestCheckpointCallback has been replaced
    by unified checkpoint management in BBUTrainer.

    Raises:
        NotImplementedError: Always raised with migration instructions
    """
    raise NotImplementedError(
        "BestCheckpointCallback has been deprecated and replaced by unified checkpoint management. "
    )


# === New: Progressive Unfreeze with Coordinate-Slice Masking ===
class ProgressiveUnfreezeCallback(TrainerCallback):
    """
    Freeze vision + LLM at start and train only:
      - visual.merger (MLP aligner)
      - coord-token slice of embed_tokens.weight and lm_head.weight
    Then unfreeze all at the beginning of epoch `freeze_vision_llm_epochs`.

    This integrates with BBUTrainer's optimizer grouping, re-creating
    optimizer/scheduler at the stage boundary for correctness.
    """

    def __init__(
        self, freeze_vision_llm_epochs: int = 1, coord_slice_only: bool = True
    ):
        super().__init__()
        self.freeze_vision_llm_epochs = int(freeze_vision_llm_epochs)
        self.coord_slice_only = bool(coord_slice_only)
        self._logger = get_callback_logger()
        self._mask_handles: List[Any] = []
        self._stage: int = 0  # 0=unset, 1=coord+merger only, 2=unfrozen
        self._trainer_ref = None  # Store trainer reference for HF compatibility

    # ---- Helpers ----
    def _get_base_model(self, trainer) -> Any:
        model = trainer.model
        # DetectionModel wrapper exposes base model under .base_model
        if hasattr(model, "base_model"):
            return model.base_model
        return model

    def _get_coord_token_range(self, trainer) -> Optional[tuple[int, int]]:
        # Fail fast - if coordinate processor is not available, this is a configuration error
        if not hasattr(trainer.model, "coordinate_processor"):
            raise AttributeError(
                "Model does not have coordinate_processor attribute - check model configuration"
            )

        cp = trainer.model.coordinate_processor
        if cp is None:
            raise ValueError(
                "coordinate_processor is None - check model initialization"
            )

        if not hasattr(cp, "coordinate_token_range"):
            raise AttributeError(
                "coordinate_processor does not have coordinate_token_range attribute"
            )

        rng = cp.coordinate_token_range
        if not rng or rng[0] is None or rng[1] is None or rng[0] >= rng[1]:
            raise ValueError(
                f"Invalid coordinate token range: {rng} - check coordinate processor setup"
            )

        return int(rng[0]), int(rng[1])

    def _find_embedding_and_lmhead(
        self, trainer
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        base = self._get_base_model(trainer)
        # Input embeddings
        if not hasattr(base, "get_input_embeddings"):
            raise RuntimeError("Base model does not expose get_input_embeddings()")
        emb = base.get_input_embeddings()
        if emb is None or not hasattr(emb, "weight"):
            raise RuntimeError("Could not access input embedding weight")
        embed_param = emb.weight

        # LM head
        if not hasattr(base, "lm_head") or not hasattr(base.lm_head, "weight"):
            raise RuntimeError("Base model does not expose lm_head.weight")
        lm_head_param = base.lm_head.weight
        return embed_param, lm_head_param

    def _apply_coord_slice_grad_masks(
        self,
        embed_param: torch.nn.Parameter,
        lm_head_param: torch.nn.Parameter,
        coord_start: int,
        coord_end_exclusive: int,
    ) -> None:
        # Build a boolean row mask over vocab entries
        device = embed_param.device
        vocab_rows = (
            embed_param.shape[0] if embed_param.dim() == 2 else embed_param.shape[1]
        )
        row_mask = torch.zeros(vocab_rows, dtype=torch.bool, device=device)
        row_mask[coord_start:coord_end_exclusive] = True

        # For embed_tokens.weight: shape [vocab, hidden]
        def _mask_embed_grad(g: torch.Tensor) -> torch.Tensor:
            return g * row_mask.unsqueeze(1).to(dtype=g.dtype)

        # For lm_head.weight: can be [vocab, hidden] or [hidden, vocab]
        def _mask_lm_head_grad(g: torch.Tensor) -> torch.Tensor:
            if g.dim() != 2:
                return g
            if lm_head_param.shape[0] == vocab_rows:
                # [vocab, hidden]
                return g * row_mask.unsqueeze(1).to(dtype=g.dtype)
            elif lm_head_param.shape[1] == vocab_rows:
                # [hidden, vocab]
                return g * row_mask.to(dtype=g.dtype)
            else:
                return g

        self._mask_handles.append(embed_param.register_hook(_mask_embed_grad))
        self._mask_handles.append(lm_head_param.register_hook(_mask_lm_head_grad))

    def _clear_masks(self) -> None:
        for h in self._mask_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._mask_handles.clear()

    def _stage1_freeze_and_mask(self, trainer) -> None:
        if self._stage != 0:
            return
        model = trainer.model
        # 1) Freeze everything
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 2) Unfreeze MLP aligner (visual.merger) and record
        for name, p in model.named_parameters():
            if "visual.merger" in name:
                p.requires_grad = True

        # 3) Unfreeze embeddings + lm_head and add coordinate-slice grad masks
        try:
            embed_param, lm_head_param = self._find_embedding_and_lmhead(trainer)
        except Exception as e:
            self._logger.error(
                f"[ProgressiveUnfreeze] Failed to access embeddings/LM head: {e}"
            )
            raise

        embed_param.requires_grad = True
        lm_head_param.requires_grad = True

        if self.coord_slice_only:
            coord_range = self._get_coord_token_range(trainer)
            if coord_range is None:
                self._logger.warning(
                    "[ProgressiveUnfreeze] Coordinate token range unavailable; skipping coord grad masks"
                )
            else:
                c0, c1 = coord_range
                self._apply_coord_slice_grad_masks(embed_param, lm_head_param, c0, c1)
                self._logger.info(
                    f"[ProgressiveUnfreeze] Applied coord-slice grad masks for rows [{c0}, {c1})"
                )

        self._stage = 1
        self._logger.info(
            "[ProgressiveUnfreeze] Stage 1 active: training visual.merger and coord slices of embeddings/LM head"
        )

    def _rebuild_optimizer_and_scheduler(self, trainer) -> None:
        # Recreate optimizer and scheduler to reflect newly trainable params
        try:
            trainer.optimizer = None
            # Rebuild optimizer with current requires_grad flags
            trainer.create_optimizer()

            # Recreate scheduler with remaining steps estimate
            import math

            dl = trainer.get_train_dataloader()
            steps_per_epoch = math.ceil(
                len(dl) / max(1, trainer.args.gradient_accumulation_steps)
            )
            # state.epoch is float, we want remaining full epochs
            current_epoch = int(getattr(trainer.state, "epoch", 0) or 0)
            remaining_epochs = max(
                0, int(trainer.args.num_train_epochs) - current_epoch
            )
            remaining_steps = max(steps_per_epoch * remaining_epochs, 1)
            trainer.create_scheduler(remaining_steps, trainer.optimizer)
            self._logger.info(
                f"[ProgressiveUnfreeze] Optimizer and scheduler rebuilt (remaining_steps={remaining_steps})"
            )
        except Exception as e:
            self._logger.warning(
                f"[ProgressiveUnfreeze] Failed to rebuild optimizer/scheduler cleanly: {e}"
            )

    def _stage2_unfreeze_all(self, trainer) -> None:
        if self._stage != 1:
            return
        # Remove grad masks
        self._clear_masks()
        # Unfreeze all
        for _, p in trainer.model.named_parameters():
            p.requires_grad = True
        # Rebuild optimizer/scheduler
        self._rebuild_optimizer_and_scheduler(trainer)
        self._stage = 2
        self._logger.info(
            "[ProgressiveUnfreeze] Stage 2 active: all modules unfrozen for joint training"
        )

    # ---- HF TrainerCallback events ----
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Get trainer from kwargs (newer HF versions) or use stored reference
        tr = kwargs.get("trainer", self._trainer_ref)
        if tr is None:
            raise ValueError(
                "trainer not available - callback was not properly initialized with trainer reference"
            )

        # Store trainer reference for future use if not already stored
        if self._trainer_ref is None:
            self._trainer_ref = tr

        self._stage1_freeze_and_mask(tr)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Get trainer from kwargs (newer HF versions) or use stored reference
        tr = kwargs.get("trainer", self._trainer_ref)
        if tr is None:
            raise ValueError(
                "trainer not available - callback was not properly initialized with trainer reference"
            )
        # Unfreeze at the beginning of epoch == freeze_vision_llm_epochs
        if not hasattr(state, "epoch"):
            raise AttributeError(
                "TrainerState does not have epoch attribute - check trainer setup"
            )

        epoch_value = state.epoch
        if epoch_value is None:
            raise ValueError(
                "TrainerState.epoch is None - check training state initialization"
            )

        current_epoch = int(epoch_value)
        if (
            self.freeze_vision_llm_epochs > 0
            and current_epoch >= self.freeze_vision_llm_epochs
            and self._stage == 1
        ):
            self._stage2_unfreeze_all(tr)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Cleanup any remaining hooks
        self._clear_masks()
        self._logger.info("[ProgressiveUnfreeze] Cleanup complete")
