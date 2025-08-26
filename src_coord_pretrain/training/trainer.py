from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

import torch
from torch.utils.data import Subset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Trainer,
    TrainingArguments,
)


# Apply Qwen2.5 runtime patches if available (must run before model instantiation)
try:
    from src_new.models.patches import apply_comprehensive_qwen25_fixes  # type: ignore

    apply_comprehensive_qwen25_fixes()
except Exception:
    pass

from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
    CoordBootstrapDataset,
    DatasetConfig,
)
from src_coord_pretrain.datasets.collator import (
    CollatorConfig,
    DataCollatorCoordBootstrap,
)


try:
    import yaml
except Exception:
    yaml = None


class HasTokenizer(Protocol):
    """Protocol for objects exposing a HuggingFace tokenizer and save_pretrained method."""

    tokenizer: Any  # PreTrainedTokenizerBase, but relaxed for runtime variants

    def save_pretrained(self, save_directory: str) -> Any: ...


@dataclass(frozen=True)
class TrainConfig:
    config_path: str

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        p = Path(path)
        # Convert relative paths to absolute paths relative to current working directory
        if not p.is_absolute():
            p = p.resolve()
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config YAML")
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config YAML must parse to a dict")
        return cfg


def _validate_coord_tokens(
    processor: HasTokenizer, max_coord_value: int
) -> Dict[str, Any]:
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    missing: List[str] = []
    ids: List[int] = []
    for i in range(int(max_coord_value) + 1):
        t = f"<|coord_{i}|>"
        if t not in vocab:
            missing.append(t)
        else:
            ids.append(int(tokenizer.convert_tokens_to_ids(t)))
    if missing:
        raise RuntimeError(
            f"Missing coordinate tokens in tokenizer: {missing[:5]}... (total {len(missing)})"
        )
    return {"coord_token_ids": ids, "max": max_coord_value}


def _load_processor(model_path: Path) -> HasTokenizer:
    """Load a processor (2.5 first, then 2.0 fallback) and normalize its type to HasTokenizer."""
    try:
        proc = Qwen2_5_VLProcessor.from_pretrained(str(model_path))
    except Exception:
        # Fallback to Qwen2-VL processor if a legacy export used it
        from transformers import (
            Qwen2VLProcessor as _Qwen2VLProcessor,  # local import to avoid top-level dependency
        )

        proc = _Qwen2VLProcessor.from_pretrained(str(model_path))
    if isinstance(proc, tuple):  # type: ignore[reportUnnecessaryIsInstance]
        proc = proc[0]
    return cast(HasTokenizer, proc)


def _build_subsets(
    dataset: CoordBootstrapDataset, val_ratio: float, seed: int
) -> tuple[Subset, Subset]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        raise ValueError("Train set would be empty after split; reduce val_ratio")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _name_is_mlp_aligner(param_name: str) -> bool:
    name = param_name.lower()
    # Broad patterns covering typical projector/connector/aligner modules
    patterns = ["mm_projector", "projector", "connector", "align", "adapter", "mlp"]
    return any(p in name for p in patterns)


class CustomLrTrainer(Trainer):
    """Trainer with differential LR for LLM vs MLP aligner and optional vision freeze."""

    def __init__(
        self, *args, llm_lr: float, mlp_lr: float, freeze_vision: bool, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._llm_lr = float(llm_lr)
        self._mlp_lr = float(mlp_lr)
        self._freeze_vision = bool(freeze_vision)
        self._training_start_time = time.time()  # Track training start time

        # Optionally freeze vision tower parameters
        if self._freeze_vision and hasattr(self.model, "visual"):
            for _, p in self.model.visual.named_parameters():
                p.requires_grad = False


class PhaseATrainer(CustomLrTrainer):
    """Enhanced trainer with Phase A support, unlikelihood training, and advanced loss computation."""

    def __init__(
        self,
        *args,
        unlikelihood_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Single-phase defaults
        self.phase_a_freeze_backbone = True

        # Unlikelihood configuration
        self.unlikelihood_config = unlikelihood_config or {}
        self.unlikelihood_enabled = self.unlikelihood_config.get(
            "unlikelihood_enabled", False
        )
        self.unlikelihood_lambda_digits = self.unlikelihood_config.get(
            "unlikelihood_lambda_digits", 1.0
        )
        self.unlikelihood_lambda_coords = self.unlikelihood_config.get(
            "unlikelihood_lambda_coords", 1.0
        )
        self.unlikelihood_coord_window = self.unlikelihood_config.get(
            "unlikelihood_coord_window", 8
        )

        # Top-K Unlikelihood configuration (advanced features)
        self.ul_topk_noncoord = self.unlikelihood_config.get("ul_topk_noncoord", 100)
        self.ul_topk_coord = self.unlikelihood_config.get("ul_topk_coord", 100)
        self.ul_neighbor_window = self.unlikelihood_config.get("ul_neighbor_window", 8)

        self._current_phase = "B"

        # Loss component tracking for logging
        self._loss_components = {}
        self._additional_loss_functions = {}  # For future extensibility

        # Freeze vision tower and keep embeddings/LM head trainable
        self._apply_phase_a_freezing()

        # Install grad mask so only coordinate token embeddings update
        try:
            self._install_coord_embedding_grad_mask()
        except Exception:
            pass

    def _apply_phase_a_freezing(self):
        """Freeze vision tower; keep embeddings and LM head trainable (text-only path)."""
        if not hasattr(self.model, "model"):
            return

        # Freeze backbone layers (vision) by default
        if hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers:
                for param in layer.parameters():
                    param.requires_grad = True

        if hasattr(self.model, "lm_head"):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

    def _install_coord_embedding_grad_mask(self):
        if not hasattr(self.model, "model") or not hasattr(
            self.model.model, "embed_tokens"
        ):
            return
        coord_ids = self._get_coordinate_token_ids()
        if not coord_ids:
            return
        emb = self.model.model.embed_tokens
        weight = emb.weight
        vocab_size = int(weight.shape[0])  # type: ignore
        keep = torch.zeros(vocab_size, device=weight.device, dtype=weight.dtype)
        for cid in coord_ids:
            if 0 <= int(cid) < vocab_size:
                keep[int(cid)] = 1.0
        mask = keep.view(-1, 1)

        # Zero out gradients for non-coordinate tokens
        def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * mask.to(device=grad.device, dtype=grad.dtype)

        # Remove previous hook if exists
        if hasattr(self, "_embed_grad_hook") and self._embed_grad_hook is not None:
            try:
                self._embed_grad_hook.remove()
            except Exception:
                pass
        self._embed_grad_hook = weight.register_hook(_grad_hook)

    def _get_tokenizer(self):
        pc = getattr(self, "processing_class", None)
        if (
            pc is not None
            and hasattr(pc, "tokenizer")
            and getattr(pc, "tokenizer") is not None
        ):
            return getattr(pc, "tokenizer")
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            return tok
        if hasattr(self, "data_collator") and hasattr(self.data_collator, "tokenizer"):
            return self.data_collator.tokenizer
        td = getattr(self, "train_dataset", None)
        if td is not None:
            tok_ds = getattr(td, "tokenizer", None)
            if tok_ds is not None:
                return tok_ds
        raise RuntimeError(
            "No tokenizer/processing_class available in trainer for decoding/ID lookups"
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to add unlikelihood loss; single-phase training."""
        return super().training_step(model, inputs, num_items_in_batch)

    def _log_phase_info(self, logs: Dict[str, Any]):
        logs["current_phase"] = self._current_phase

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Override compute_loss to add unlikelihood terms and track loss components."""
        # Always obtain outputs to enable UL in both train and eval
        is_eval_mode = not model.training
        if is_eval_mode:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                llm_result = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
        else:
            llm_result = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )

        if isinstance(llm_result, tuple):
            llm_loss, outputs = llm_result
        else:
            llm_loss, outputs = llm_result, None

        # Normalize and validate LLM loss type
        llm_loss_t: torch.Tensor = cast(torch.Tensor, llm_loss)
        if not torch.isfinite(llm_loss_t).all():
            raise RuntimeError(
                f"Non-finite llm_loss detected (phase={self._current_phase}, step={self.state.global_step})"
            )

        # Initialize loss components tracking
        self._loss_components = {"llm_loss": self._safe_float_conversion(llm_loss_t)}

        # Start with LLM loss as total
        total_loss: torch.Tensor = llm_loss_t

        # Add unlikelihood loss if enabled and outputs available
        if self.unlikelihood_enabled and outputs is not None:
            unlikelihood_loss = self._compute_unlikelihood_loss(outputs, inputs)
            self._loss_components["unlikelihood_loss"] = self._safe_float_conversion(
                unlikelihood_loss
            )
            total_loss = total_loss + unlikelihood_loss
        else:
            self._loss_components["unlikelihood_loss"] = 0.0

        # Add any additional registered loss components
        total_loss = self._compute_additional_losses(total_loss, outputs, inputs)

        # Fail-fast: validate total loss
        if not torch.isfinite(total_loss).all():
            raise RuntimeError(
                f"Non-finite total loss detected (phase={self._current_phase}, step={self.state.global_step}). "
                f"Components: {self._loss_components}"
            )
        if float(total_loss.item()) == 0.0:
            raise RuntimeError(
                f"Zero total loss detected (phase={self._current_phase}, step={self.state.global_step}). "
                f"Check LR/scheduler and unfreeze policy. Components: {self._loss_components}"
            )

        # Store total loss
        self._loss_components["loss"] = self._safe_float_conversion(total_loss)

        return (total_loss, outputs) if return_outputs else total_loss

    def _safe_float_conversion(self, value):
        """Safely convert tensor or numeric value to float for logging."""
        if hasattr(value, "item"):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0

    def register_loss_component(self, name: str, loss_function, weight: float = 1.0):
        """Register an additional loss component for future extensibility.

        Args:
            name: Name of the loss component (will appear in logs)
            loss_function: Callable that takes (outputs, inputs) and returns loss tensor
            weight: Weight to apply to this loss component
        """
        self._additional_loss_functions[name] = {
            "function": loss_function,
            "weight": weight,
        }

    def _compute_additional_losses(self, current_total_loss, outputs, inputs):
        """Compute any additional registered loss components."""
        total_loss = current_total_loss

        for loss_name, loss_config in self._additional_loss_functions.items():
            try:
                loss_function = loss_config["function"]
                weight = loss_config["weight"]

                # Compute the additional loss
                additional_loss = loss_function(outputs, inputs)
                weighted_loss = weight * additional_loss

                # Track the loss component
                self._loss_components[loss_name] = self._safe_float_conversion(
                    additional_loss
                )
                self._loss_components[f"{loss_name}_weighted"] = (
                    self._safe_float_conversion(weighted_loss)
                )

                # Add to total loss
                total_loss = total_loss + weighted_loss

            except Exception as e:
                # Log error but don't crash training (only on rank 0 to avoid spam)
                if getattr(self.args, "should_save", True):  # Only rank 0
                    print(
                        f"Warning: Error computing additional loss '{loss_name}': {e}"
                    )
                self._loss_components[loss_name] = 0.0

        return total_loss

    def _compute_unlikelihood_loss(self, outputs, inputs):
        """Compute advanced Top-K Unlikelihood loss with conflict resolution."""
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = inputs.get("labels")

        if labels is None:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Find assistant token positions (labels != -100) and exclude EOS dynamically
        tokenizer = self._get_tokenizer()
        try:
            im_end_id = getattr(tokenizer, "convert_tokens_to_ids", lambda x: None)(
                "<|im_end|>"
            )
        except Exception:
            im_end_id = None
        assistant_mask = labels != -100
        if im_end_id is not None and int(im_end_id) != -1:
            assistant_mask = assistant_mask & (labels != int(im_end_id))
        if not assistant_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Use Top-K approach if configured, otherwise fall back to legacy
        if hasattr(self, "ul_topk_noncoord") and self.ul_topk_noncoord > 0:
            return self._compute_topk_unlikelihood_loss(logits, labels, assistant_mask)
        else:
            return self._compute_legacy_unlikelihood_loss(
                logits, labels, assistant_mask
            )

    def _compute_topk_unlikelihood_loss(self, logits, labels, assistant_mask):
        # Compute EOS id once and pass to coord loss
        tokenizer = self._get_tokenizer()
        try:
            im_end_id = getattr(tokenizer, "convert_tokens_to_ids", lambda x: None)(
                "<|im_end|>"
            )
        except Exception:
            im_end_id = None
        """Compute Top-K Unlikelihood loss with proper conflict resolution."""
        # Get coordinate token IDs via static range if available
        coord_start = getattr(self, "_coord_start_id", 151667)
        coord_end = getattr(self, "_coord_end_id", coord_start + 1024)
        coord_ids_tensor = torch.arange(
            coord_start, coord_end + 1, device=logits.device, dtype=torch.long
        )

        # Compute target masks with conflict resolution
        coord_target_mask = torch.isin(labels, coord_ids_tensor) & assistant_mask
        text_target_mask = (
            (~torch.isin(labels, coord_ids_tensor)) & (labels != -100) & assistant_mask
        )

        # Verify masks are mutually exclusive (fail-fast validation)
        if (coord_target_mask & text_target_mask).any():
            raise RuntimeError(
                "Target masks are not mutually exclusive - this indicates a bug"
            )

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Coordinate targets: apply Top-K non-coordinate token suppression
        if coord_target_mask.any() and self.ul_topk_noncoord > 0:
            coord_loss = self._compute_coordinate_target_loss(
                logits, labels, coord_target_mask, coord_ids_tensor, im_end_id=im_end_id
            )
            weighted = self.unlikelihood_lambda_coords * coord_loss
            total_loss = total_loss + weighted
            # Log specific component (weighted contribution)
            try:
                self._loss_components["unlikelihood_coord"] = (
                    self._safe_float_conversion(weighted)
                )
            except Exception:
                pass

        # Text targets: apply Top-K coordinate token suppression
        if text_target_mask.any() and self.ul_topk_coord > 0:
            text_loss = self._compute_text_target_loss(
                logits, labels, text_target_mask, coord_ids_tensor
            )
            weighted = self.unlikelihood_lambda_digits * text_loss
            total_loss = total_loss + weighted
            # Log specific component (weighted contribution)
            try:
                self._loss_components["unlikelihood_text"] = (
                    self._safe_float_conversion(weighted)
                )
            except Exception:
                pass

        # Track additional metrics at text targets (does not affect loss)
        try:
            log_probs = torch.log_softmax(logits, dim=-1)
            vocab_size = logits.shape[-1]
            valid_coord_ids = coord_ids_tensor[coord_ids_tensor < vocab_size]
            if text_target_mask.any() and valid_coord_ids.numel() > 0:
                text_pos = text_target_mask.nonzero(as_tuple=True)
                selected_log_probs = log_probs[text_pos[0], text_pos[1]]  # [N, V]
                coord_log_probs = selected_log_probs[:, valid_coord_ids]  # [N, K]
                # Average of maximum coordinate-token probability across text positions
                max_coord_logp, _ = coord_log_probs.max(dim=-1)  # [N]
                mean_max_coord_prob = torch.exp(max_coord_logp).mean()
                # Average log mass of all coordinate tokens across text positions
                log_mass_per_pos = torch.logsumexp(coord_log_probs, dim=-1)  # [N]
                mean_log_mass = log_mass_per_pos.mean()
                self._loss_components["unlikelihood_text_positions"] = float(
                    coord_log_probs.shape[0]
                )
                self._loss_components["unlikelihood_text_max_coord_prob"] = (
                    self._safe_float_conversion(mean_max_coord_prob)
                )
                self._loss_components["unlikelihood_text_log_coord_mass"] = (
                    self._safe_float_conversion(mean_log_mass)
                )
            else:
                self._loss_components["unlikelihood_text_positions"] = 0.0
                self._loss_components["unlikelihood_text_max_coord_prob"] = 0.0
                self._loss_components["unlikelihood_text_log_coord_mass"] = 0.0
        except Exception:
            # Do not disrupt training if metric computation fails
            self._loss_components["unlikelihood_text_positions"] = 0.0
            self._loss_components["unlikelihood_text_max_coord_prob"] = 0.0
            self._loss_components["unlikelihood_text_log_coord_mass"] = 0.0

        return total_loss

    def _compute_legacy_unlikelihood_loss(self, logits, labels, assistant_mask):
        """Legacy unlikelihood loss computation for backward compatibility."""
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Digit unlikelihood loss
        if self.unlikelihood_lambda_digits > 0:
            digit_loss = self._compute_digit_unlikelihood(logits, assistant_mask)
            weighted = self.unlikelihood_lambda_digits * digit_loss
            total_loss = total_loss + weighted
            try:
                self._loss_components["unlikelihood_digits"] = (
                    self._safe_float_conversion(weighted)
                )
            except Exception:
                pass

        # Coordinate window unlikelihood loss
        if self.unlikelihood_lambda_coords > 0:
            coord_loss = self._compute_coordinate_unlikelihood(
                logits, labels, assistant_mask
            )
            weighted = self.unlikelihood_lambda_coords * coord_loss
            total_loss = total_loss + weighted
            try:
                self._loss_components["unlikelihood_coords"] = (
                    self._safe_float_conversion(weighted)
                )
            except Exception:
                pass

        return total_loss

    def _compute_coordinate_target_loss(
        self, logits, labels, coord_target_mask, coord_ids_tensor, im_end_id
    ):
        """Compute Top-K non-coordinate token suppression for coordinate targets."""
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        eps = 1e-5  # Numerical stability

        # Get positions where we have coordinate targets
        coord_positions = coord_target_mask.nonzero(as_tuple=True)

        for batch_idx, seq_idx in zip(coord_positions[0], coord_positions[1]):
            position_logits = logits[batch_idx, seq_idx]  # [vocab_size]
            gold_token_id = labels[batch_idx, seq_idx].item()

            # Get probabilities
            probs = torch.softmax(position_logits, dim=-1)

            # Create mask for valid negative tokens (exclude gold, coordinates, special tokens)
            exclude_ids = set(coord_ids_tensor.tolist()) | {gold_token_id}
            if im_end_id is not None and int(im_end_id) != -1:
                exclude_ids.add(int(im_end_id))
            valid_mask = torch.ones(
                logits.shape[-1], dtype=torch.bool, device=logits.device
            )
            for exclude_id in exclude_ids:
                if exclude_id < logits.shape[-1]:
                    valid_mask[exclude_id] = False

            # Select top-K from valid non-coordinate tokens
            if valid_mask.sum() > 0:
                valid_probs = probs.clone()
                valid_probs[~valid_mask] = -float("inf")

                k = min(self.ul_topk_noncoord, valid_mask.sum().item())
                if k > 0:
                    topk_probs, _ = torch.topk(valid_probs, k=k)

                    # Apply unlikelihood to top-K negatives
                    clamped_complement = torch.clamp(1.0 - topk_probs, min=eps)
                    position_loss = -torch.log(clamped_complement).sum()
                    total_loss = total_loss + position_loss

        # Average over coordinate positions
        num_coord_positions = coord_target_mask.sum().float()
        if num_coord_positions > 0:
            return total_loss / num_coord_positions
        else:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    def _compute_text_target_loss(
        self, logits, labels, text_target_mask, coord_ids_tensor
    ):
        """Compute Top-K coordinate token suppression for text targets."""
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        eps = 1e-5  # Numerical stability

        # Get positions where we have text targets
        text_positions = text_target_mask.nonzero(as_tuple=True)

        for batch_idx, seq_idx in zip(text_positions[0], text_positions[1]):
            position_logits = logits[batch_idx, seq_idx]  # [vocab_size]
            gold_token_id = labels[batch_idx, seq_idx].item()

            # Get probabilities
            probs = torch.softmax(position_logits, dim=-1)

            # Create mask for coordinate tokens only (exclude gold if it's accidentally a coord token)
            coord_mask = torch.zeros(
                logits.shape[-1], dtype=torch.bool, device=logits.device
            )
            for coord_id in coord_ids_tensor.tolist():
                if coord_id < logits.shape[-1] and coord_id != gold_token_id:
                    coord_mask[coord_id] = True

            # Select top-K from coordinate tokens
            if coord_mask.sum() > 0:
                coord_probs = probs.clone()
                coord_probs[~coord_mask] = -float("inf")

                k = min(self.ul_topk_coord, coord_mask.sum().item())
                if k > 0:
                    topk_probs, _ = torch.topk(coord_probs, k=k)

                    # Apply unlikelihood to top-K coordinate negatives
                    clamped_complement = torch.clamp(1.0 - topk_probs, min=eps)
                    position_loss = -torch.log(clamped_complement).sum()
                    total_loss = total_loss + position_loss

        # Average over text positions
        num_text_positions = text_target_mask.sum().float()
        if num_text_positions > 0:
            return total_loss / num_text_positions
        else:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    def _compute_digit_unlikelihood(self, logits, assistant_mask):
        """Compute unlikelihood loss for raw digit tokens at assistant positions."""
        # Get digit token IDs (0-9 and common numeric tokens)
        digit_tokens = self._get_digit_token_ids()
        if not digit_tokens:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Convert to tensor
        digit_ids = torch.tensor(digit_tokens, device=logits.device, dtype=torch.long)

        # Get probabilities for digit tokens at assistant positions
        probs = torch.softmax(logits, dim=-1)  # [batch, seq, vocab]
        digit_probs = probs[:, :, digit_ids]  # [batch, seq, num_digits]

        # Sum probabilities of all digit tokens at each position
        digit_prob_sum = digit_probs.sum(dim=-1)  # [batch, seq]

        # Apply unlikelihood: -log(1 - p) for digit tokens at assistant positions
        eps = 1e-8
        unlikelihood = -torch.log(1.0 - digit_prob_sum + eps)

        # Mask to assistant positions only
        masked_loss = unlikelihood * assistant_mask.float()

        # Average over assistant positions
        num_assistant_tokens = assistant_mask.sum().float()
        if num_assistant_tokens > 0:
            return masked_loss.sum() / num_assistant_tokens
        else:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    def _compute_coordinate_unlikelihood(self, logits, labels, assistant_mask):
        """Compute unlikelihood loss for neighboring coordinate tokens."""
        # Find coordinate token positions in labels
        coord_token_ids = self._get_coordinate_token_ids()
        if not coord_token_ids:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        coord_min, coord_max = min(coord_token_ids), max(coord_token_ids)
        coord_positions = (labels >= coord_min) & (labels <= coord_max) & assistant_mask

        if not coord_positions.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        eps = 1e-8

        # For each coordinate token position, suppress neighboring coordinates
        for batch_idx in range(labels.shape[0]):
            for seq_idx in range(labels.shape[1]):
                if not coord_positions[batch_idx, seq_idx]:
                    continue

                target_coord_id = labels[batch_idx, seq_idx].item()
                target_coord_value = (
                    target_coord_id - coord_min
                )  # Convert to coordinate value

                # Define window of neighboring coordinates to suppress
                window_start = max(
                    0, target_coord_value - self.unlikelihood_coord_window
                )
                window_end = min(
                    len(coord_token_ids),
                    target_coord_value + self.unlikelihood_coord_window + 1,
                )

                # Get neighboring coordinate token IDs (excluding target)
                neighbor_ids = []
                for coord_val in range(window_start, window_end):
                    neighbor_id = coord_min + coord_val
                    if neighbor_id != target_coord_id:
                        neighbor_ids.append(neighbor_id)

                if neighbor_ids:
                    # Get probabilities for neighboring coordinates
                    neighbor_tensor = torch.tensor(
                        neighbor_ids, device=logits.device, dtype=torch.long
                    )
                    probs = torch.softmax(logits[batch_idx, seq_idx], dim=-1)
                    neighbor_probs = probs[neighbor_tensor]

                    # Apply unlikelihood to neighbors
                    neighbor_loss = -torch.log(1.0 - neighbor_probs + eps).sum()
                    total_loss = total_loss + neighbor_loss

        # Average over coordinate positions
        num_coord_positions = coord_positions.sum().float()
        if num_coord_positions > 0:
            return total_loss / num_coord_positions
        else:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    def _get_digit_token_ids(self):
        """Get token IDs for raw digit tokens (0-9) using tokenization approach."""
        if not hasattr(self, "_digit_token_ids"):
            digit_tokens = []

            # Get tokenizer via unified accessor
            try:
                tokenizer = self._get_tokenizer()
            except Exception:
                self._digit_token_ids = []
                return self._digit_token_ids

            # Use encode to get token IDs deterministically
            for digit in "0123456789":
                try:
                    ids = tokenizer.encode(digit, add_special_tokens=False)
                    if (
                        isinstance(ids, list) and len(ids) == 1
                    ):  # Single token for this digit
                        digit_tokens.append(ids[0])
                except Exception:
                    # Skip if tokenization fails
                    continue

            self._digit_token_ids = list(set(digit_tokens))  # Remove duplicates

        return self._digit_token_ids

    def _get_coordinate_token_ids(self):
        """Get token IDs for coordinate tokens using known ranges from src_new."""
        if not hasattr(self, "_coordinate_token_ids"):
            # Use known coordinate token range from src_new analysis:
            # <|coord_0|> to <|coord_1024|>: 151667 to 152691
            coord_start_id = 151667
            coord_end_id = 152691  # 151667 + 1024

            # Generate coordinate token IDs in the known range
            coord_tokens = list(range(coord_start_id, coord_end_id + 1))

            # Verify with tokenizer if available (optional validation)
            try:
                tokenizer = self._get_tokenizer()
            except Exception:
                tokenizer = None

            if tokenizer is not None:
                try:
                    vocab = tokenizer.get_vocab()
                    # Verify a few coordinate tokens exist in vocabulary
                    test_tokens = ["<|coord_0|>", "<|coord_100|>", "<|coord_1024|>"]
                    for token in test_tokens:
                        if token in vocab:
                            expected_id = coord_start_id + int(
                                token.split("_")[1].split("|")[0]
                            )
                            if vocab[token] != expected_id:
                                # Fallback to vocabulary-based detection if ranges don't match
                                coord_tokens = []
                                for token_name, token_id in vocab.items():
                                    if token_name.startswith(
                                        "<|coord_"
                                    ) and token_name.endswith("|>"):
                                        coord_tokens.append(token_id)
                                coord_tokens = sorted(coord_tokens)
                                break
                except Exception:
                    # Keep the range-based approach if vocabulary check fails
                    pass

            self._coordinate_token_ids = coord_tokens

        return self._coordinate_token_ids

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with coordinate token validation."""
        # Run standard evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Add coordinate token validation metrics
        if self.eval_dataset is not None:
            validation_metrics = self._compute_validation_metrics(
                eval_dataset or self.eval_dataset
            )
            metrics.update(
                {f"{metric_key_prefix}_{k}": v for k, v in validation_metrics.items()}
            )

        return metrics

    def _compute_validation_metrics(self, eval_dataset):
        """Compute coordinate token validation metrics."""
        self.model.eval()

        total_samples = 0
        identity_correct = 0
        reverse_correct = 0
        identity_total = 0
        reverse_total = 0
        strictness_violations = 0

        # Sample a subset for validation (to avoid long evaluation times)
        max_eval_samples = 100
        eval_indices = list(range(min(len(eval_dataset), max_eval_samples)))

        with torch.no_grad():
            for idx in eval_indices:
                sample = eval_dataset[idx]

                # Get the expected answer from the sample
                input_ids = sample["input_ids"].unsqueeze(0).to(self.model.device)
                # Build labels if not present in dataset sample (dataset returns only inputs; collator creates labels)
                labels_tensor = sample["labels"] if "labels" in sample else None
                if labels_tensor is None:
                    data_collator = getattr(self, "data_collator", None)
                    if data_collator is not None and hasattr(
                        data_collator, "_build_labels_for_sample"
                    ):
                        labels_tensor = data_collator._build_labels_for_sample(
                            sample["input_ids"]
                        )
                    else:
                        raise KeyError("labels")
                labels = labels_tensor.unsqueeze(0).to(self.model.device)

                # Generate response
                try:
                    generated = self._generate_response(input_ids, labels)
                    if generated is None:
                        continue

                    # Validate the generated response
                    is_valid, task_type, expected_value, actual_value = (
                        self._validate_response(generated, labels)
                    )

                    total_samples += 1

                    if not is_valid:
                        strictness_violations += 1
                    else:
                        # Check task-specific accuracy
                        if task_type == "identity" or task_type == "arithmetic":
                            identity_total += 1
                            if actual_value == expected_value:
                                identity_correct += 1
                        elif task_type == "reverse_mapping":
                            reverse_total += 1
                            if actual_value == expected_value:
                                reverse_correct += 1

                except Exception:
                    # Skip problematic samples
                    continue

        # Compute metrics
        metrics = {
            "total_samples": total_samples,
            "strictness_violations": strictness_violations,
            "strictness_accuracy": 1.0
            - (strictness_violations / max(1, total_samples)),
        }

        if identity_total > 0:
            metrics["identity_accuracy"] = identity_correct / identity_total
            metrics["identity_samples"] = identity_total

        if reverse_total > 0:
            metrics["reverse_accuracy"] = reverse_correct / reverse_total
            metrics["reverse_samples"] = reverse_total

        self.model.train()
        return metrics

    def _generate_response(self, input_ids, labels):
        """Generate response for a single sample."""
        # Find the assistant start position (where labels != -100)
        assistant_mask = labels != -100
        if not assistant_mask.any():
            return None

        # Find the start of assistant content
        assistant_start = assistant_mask.nonzero(as_tuple=True)[1][0].item()

        # Generate from the assistant start position
        prompt_ids = input_ids[:, :assistant_start]

        # Generate with constrained parameters
        tok = self._get_tokenizer()
        try:
            im_end_id = getattr(tok, "convert_tokens_to_ids", lambda x: None)(
                "<|im_end|>"
            )
        except Exception:
            im_end_id = None

        pad_id = getattr(tok, "eos_token_id", 0)
        eos_id = im_end_id if im_end_id is not None else getattr(tok, "eos_token_id", 0)

        # Build attention mask explicitly (pad and eos may be identical for Qwen)
        attn_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=10,  # Short responses expected
            do_sample=False,  # Deterministic for evaluation
            temperature=None,  # Explicitly disable temperature for greedy decoding
            top_p=None,  # Explicitly disable top_p for greedy decoding
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            attention_mask=attn_mask,
        )

        # Extract only the generated part
        generated_response = generated_ids[:, prompt_ids.shape[1] :]
        return generated_response

    def _validate_response(self, generated_ids, labels):
        """Validate generated response for coordinate token usage."""
        # Decode the generated response
        tok = self._get_tokenizer()
        generated_text = tok.decode(generated_ids[0], skip_special_tokens=False)

        # Extract expected value from labels
        expected_value = self._extract_expected_value(labels)
        task_type = self._infer_task_type(labels, generated_text)

        # Check strictness: exactly one coordinate token, no raw digits
        is_valid = self._check_strictness(generated_text)

        # Extract actual value from generated response
        actual_value = self._extract_actual_value(generated_text, task_type)

        return is_valid, task_type, expected_value, actual_value

    def _extract_expected_value(self, labels):
        """Extract expected coordinate value from labels."""
        coord_token_ids = self._get_coordinate_token_ids()
        if not coord_token_ids:
            return None

        coord_min = min(coord_token_ids)

        # Find coordinate tokens in labels
        for token_id in labels[0]:
            if token_id.item() in coord_token_ids:
                return token_id.item() - coord_min

        # For reverse mapping, look for raw digits
        tok = self._get_tokenizer()
        labels_text = tok.decode(
            labels[0][labels[0] != -100], skip_special_tokens=False
        )
        import re

        digit_match = re.search(r"\b(\d+)\b", labels_text)
        if digit_match:
            return int(digit_match.group(1))

        return None

    def _infer_task_type(self, labels, _):
        """Infer task type from labels."""
        tok = self._get_tokenizer()
        labels_text = tok.decode(
            labels[0][labels[0] != -100], skip_special_tokens=False
        )

        # Check if labels contain coordinate tokens (forward tasks)
        if "<|coord_" in labels_text:
            return "identity"  # Could be identity or arithmetic

        # Check if labels contain raw digits (reverse mapping)
        import re

        if re.search(r"\b\d+\b", labels_text):
            return "reverse_mapping"

        return "unknown"

    def _check_strictness(self, generated_text):
        """Check if generated text follows coordinate token strictness rules."""
        import re

        # Count coordinate tokens
        coord_tokens = re.findall(r"<\|coord_\d+\|>", generated_text)

        # Count raw digits (not part of coordinate tokens)
        text_without_coords = re.sub(r"<\|coord_\d+\|>", "", generated_text)
        raw_digits = re.findall(r"\b\d+\b", text_without_coords)

        # For coordinate token responses: exactly one coord token, no raw digits
        if coord_tokens:
            return len(coord_tokens) == 1 and len(raw_digits) == 0

        # For raw digit responses (reverse mapping): exactly one number, no coord tokens
        if raw_digits:
            return len(raw_digits) == 1 and len(coord_tokens) == 0

        return False

    def _extract_actual_value(self, generated_text, task_type):
        """Extract actual coordinate value from generated text."""
        import re

        if task_type == "reverse_mapping":
            # Extract raw digit
            digit_match = re.search(r"\b(\d+)\b", generated_text)
            if digit_match:
                return int(digit_match.group(1))
        else:
            # Extract coordinate token value
            coord_match = re.search(r"<\|coord_(\d+)\|>", generated_text)
            if coord_match:
                return int(coord_match.group(1))

        return None

    def create_optimizer(self):
        if self.optimizer is not None:
            return
        decay = set()
        no_decay = set()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            n_lower = n.lower()
            if any(x in n_lower for x in ["bias", "layernorm", "rmsnorm", "norm."]):
                no_decay.add(n)
            else:
                decay.add(n)

        llm_params_decay = []
        llm_params_nodecay = []
        mlp_params_decay = []
        mlp_params_nodecay = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Skip vision parameters entirely when frozen
            if self._freeze_vision and n.startswith("visual."):
                continue
            target_is_mlp = _name_is_mlp_aligner(n)
            if n in decay:
                (mlp_params_decay if target_is_mlp else llm_params_decay).append(p)
            elif n in no_decay:
                (mlp_params_nodecay if target_is_mlp else llm_params_nodecay).append(p)

        weight_decay = self.args.weight_decay
        param_groups = []
        if llm_params_decay:
            param_groups.append(
                {
                    "params": llm_params_decay,
                    "lr": self._llm_lr,
                    "weight_decay": weight_decay,
                    "group_name": "llm",
                }
            )
        if llm_params_nodecay:
            param_groups.append(
                {
                    "params": llm_params_nodecay,
                    "lr": self._llm_lr,
                    "weight_decay": 0.0,
                    "group_name": "llm",
                }
            )
        if mlp_params_decay:
            param_groups.append(
                {
                    "params": mlp_params_decay,
                    "lr": self._mlp_lr,
                    "weight_decay": weight_decay,
                    "group_name": "mlp",
                }
            )
        if mlp_params_nodecay:
            param_groups.append(
                {
                    "params": mlp_params_nodecay,
                    "lr": self._mlp_lr,
                    "weight_decay": 0.0,
                    "group_name": "mlp",
                }
            )

        if not param_groups:
            raise RuntimeError(
                "No trainable parameters found after freezing and grouping."
            )

        from torch.optim import AdamW

        self.optimizer = AdamW(
            param_groups, lr=self._llm_lr, betas=(0.9, 0.999), eps=1e-8
        )

        # Create scaler for mixed precision training if needed
        if self.args.fp16 or self.args.bf16:
            try:
                from torch.amp.grad_scaler import GradScaler

                self.scaler = GradScaler("cuda")
            except ImportError:
                # Fallback for older PyTorch versions
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()

        # Compute total training steps for scheduler when using num_train_epochs
        if self.args.max_steps and self.args.max_steps > 0:
            total_steps = self.args.max_steps
        else:
            dl_len = len(self.get_train_dataloader())
            steps_per_epoch = math.ceil(
                dl_len / max(1, self.args.gradient_accumulation_steps)
            )
            total_steps = int(steps_per_epoch * max(1, int(self.args.num_train_epochs)))
        self.create_scheduler(num_training_steps=total_steps)

    def _calculate_remaining_hours(self) -> float:
        """
        Calculate estimated remaining training time in hours.

        Returns:
            Estimated remaining hours (0.0 if cannot calculate)
        """
        try:
            if not hasattr(self.state, "global_step") or not hasattr(
                self.state, "max_steps"
            ):
                return 0.0

            current_step = self.state.global_step
            max_steps = self.state.max_steps

            if current_step <= 0 or max_steps <= 0 or current_step >= max_steps:
                return 0.0

            # Calculate elapsed time and average time per step
            elapsed_time = time.time() - self._training_start_time
            avg_time_per_step = elapsed_time / current_step

            # Calculate remaining steps and time
            remaining_steps = max_steps - current_step
            remaining_seconds = remaining_steps * avg_time_per_step

            # Convert to hours
            return remaining_seconds / 3600.0

        except Exception:
            return 0.0

    def log(
        self,
        logs: Dict[str, float],
        start_time: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Add phase information
        self._log_phase_info(logs)

        # Add loss components to logs
        self._log_loss_components(logs)

        # Add param-group learning rates to logs
        llm_lr_val: Optional[float] = None
        mlp_lr_val: Optional[float] = None
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                name = pg.get("group_name")
                if name == "llm":
                    llm_lr_val = float(pg.get("lr", 0.0))
                elif name == "mlp":
                    mlp_lr_val = float(pg.get("lr", 0.0))
        if llm_lr_val is not None:
            logs["llm_lr"] = llm_lr_val
        if mlp_lr_val is not None:
            logs["mlp_lr"] = mlp_lr_val
        if self._freeze_vision:
            logs.setdefault("vision_lr", 0.0)

        # Add remaining hours estimation
        remaining_hrs = self._calculate_remaining_hours()
        if remaining_hrs > 0:
            logs["remaining_hrs"] = remaining_hrs

        super().log(logs, start_time, *args, **kwargs)

    def _log_loss_components(self, logs: Dict[str, float]) -> None:
        """Add individual loss components to logs for detailed monitoring."""
        if hasattr(self, "_loss_components") and self._loss_components:
            # Only include raw loss values, no ratios and no ambiguous auxiliary metrics
            for component_name, component_value in self._loss_components.items():
                if component_name == "loss":
                    continue
                # Allow llm_loss, unlikelihood_loss, and specific unlikelihood components
                if component_name == "llm_loss" or component_name.startswith(
                    "unlikelihood"
                ):
                    logs[component_name] = component_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Coord bootstrap training entry")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (absolute or relative)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    # Required config keys (fail-fast)
    required_keys = [
        "model_path",
        "output_dir",
        "data_path",
        "max_coord_value",
        "coordinate_tokens_enabled",
        "coordinate_init_mode",
        "per_device_train_batch_size",
        "learning_rate",
        "max_epochs",
        "eval_steps",
        "seed",
        "llm_lr",
        "mlp_lr",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")

    # Remove Phase A validation; standard single-phase pipeline

    # Validate Unlikelihood configuration
    if cfg.get("unlikelihood_enabled", False):
        for param, param_type in [
            ("unlikelihood_lambda_digits", (int, float)),
            ("unlikelihood_lambda_coords", (int, float)),
            ("unlikelihood_coord_window", int),
        ]:
            if param in cfg:
                value = cfg[param]
                if not isinstance(value, param_type):
                    raise ValueError(
                        f"{param} must be of type {param_type}, got: {type(value)}"
                    )
                if isinstance(value, (int, float)) and value < 0:
                    raise ValueError(f"{param} must be non-negative, got: {value}")

    # Validate reverse mapping ratio
    if "reverse_mapping_ratio" in cfg:
        ratio = cfg["reverse_mapping_ratio"]
        if not isinstance(ratio, (int, float)) or not (0.0 <= ratio <= 1.0):
            raise ValueError(
                f"reverse_mapping_ratio must be between 0.0 and 1.0, got: {ratio}"
            )

    model_path = Path(cfg["model_path"])
    output_dir = Path(cfg["output_dir"])
    data_path = Path(cfg["data_path"])

    # Convert relative paths to absolute paths relative to current working directory
    if not model_path.is_absolute():
        model_path = model_path.resolve()
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    if not data_path.is_absolute():
        data_path = data_path.resolve()

    processor = _load_processor(model_path)
    # Load 2.5 model (will match your expanded checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path))

    # Text-only MRoPE fix: ensure mrope_section sums to head_dim for pure text (no vision)
    try:
        layers = getattr(model, "model").layers  # type: ignore[attr-defined]
        for lyr in layers:
            attn = getattr(lyr, "self_attn", None)
            if (
                attn is not None
                and hasattr(attn, "head_dim")
                and hasattr(attn, "rope_scaling")
            ):
                hd = int(getattr(attn, "head_dim"))
                # Use a single section so [64] * 2 -> [64,64] sums to 128 (matches cos last dim)
                attn.rope_scaling["mrope_section"] = [max(1, hd // 2)]
    except Exception:
        pass

    # Wrap model.forward to drop unexpected kwargs injected by external patches (e.g., num_items_in_batch)
    original_forward = model.forward

    def safe_forward(*args, **kwargs):
        for k in [
            "num_items_in_batch",
            "image_grid_thw",
            "video_grid_thw",
            "pixel_values",
            "pixel_values_videos",
        ]:
            # We do not use images/videos in this text-only module. Drop silently if provided by upstream stacks.
            if k in kwargs:
                kwargs.pop(k, None)
        return original_forward(*args, **kwargs)

    model.forward = safe_forward  # type: ignore[assignment]

    # Validate coord tokens
    max_coord_value = int(cfg["max_coord_value"])
    coord_meta = _validate_coord_tokens(processor, max_coord_value)

    # Dataset & Collator
    ds_conf = DatasetConfig(
        data_path=str(data_path),
        max_coord_value=max_coord_value,
        use_apply_chat_template=True,
    )
    dataset = CoordBootstrapDataset(tokenizer=processor.tokenizer, config=ds_conf)
    collator = DataCollatorCoordBootstrap(
        tokenizer=processor.tokenizer, config=CollatorConfig()
    )

    # Split
    train_subset, eval_subset = _build_subsets(
        dataset,
        val_ratio=float(cfg.get("val_ratio", 0.2)),
        seed=int(cfg["seed"]),
    )

    # Check for potentially problematic config settings
    if cfg.get("save_on_each_node", False):
        print(
            "⚠️ WARNING: save_on_each_node=true in config, but forcing to false for distributed safety"
        )

    # Training args
    args_train = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(
            cfg.get("per_device_eval_batch_size", cfg["per_device_train_batch_size"])
        ),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(
            cfg["learning_rate"]
        ),  # global LR; param groups override with llm_lr/mlp_lr
        num_train_epochs=int(cfg["max_epochs"]),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_steps=int(cfg.get("save_steps", 500)),
        eval_strategy=cfg.get(
            "eval_strategy", "steps"
        ),  # Allow configurable eval strategy
        save_strategy=cfg.get(
            "save_strategy", "steps"
        ),  # Allow configurable save strategy
        eval_steps=int(cfg["eval_steps"]),
        save_on_each_node=False,  # FORCE rank 0 only saving for safety in distributed training
        remove_unused_columns=False,
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        report_to=["none"],
        load_best_model_at_end=False,  # avoid extra save events
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        deepspeed=str(cfg["deepspeed_config"]) if "deepspeed_config" in cfg else None,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),  # Add warmup_ratio support
        lr_scheduler_type=cfg.get(
            "lr_scheduler_type", "cosine"
        ),  # Default to cosine scheduler
        max_grad_norm=0.0,  # Disable gradient clipping to avoid duplicate unscale_ calls
    )

    # Prepare Phase A and Unlikelihood configurations
    unlikelihood_config = {
        "unlikelihood_enabled": cfg.get("unlikelihood_enabled", False),
        "unlikelihood_lambda_digits": cfg.get("unlikelihood_lambda_digits", 1.0),
        "unlikelihood_lambda_coords": cfg.get("unlikelihood_lambda_coords", 1.0),
        "unlikelihood_coord_window": cfg.get("unlikelihood_coord_window", 8),
        # Top-K Unlikelihood parameters
        "ul_topk_noncoord": cfg.get("ul_topk_noncoord", 100),
        "ul_topk_coord": cfg.get("ul_topk_coord", 100),
        "ul_neighbor_window": cfg.get("ul_neighbor_window", 8),
    }

    trainer = PhaseATrainer(
        model=model,
        args=args_train,
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        data_collator=collator,
        llm_lr=float(cfg["llm_lr"]),
        mlp_lr=float(cfg["mlp_lr"]),
        freeze_vision=bool(cfg.get("freeze_vision", True)),
        unlikelihood_config=unlikelihood_config,
    )

    # Static coordinate token ID range from src_new (avoid dynamic setting)
    try:
        trainer._coord_start_id = 151667
        trainer._coord_end_id = trainer._coord_start_id + int(
            cfg["max_coord_value"]
        )  # inclusive
    except Exception:
        pass

    # Use processing_class instead of deprecated tokenizer for speed optimization
    # This follows HuggingFace's new API and enables fast tokenizer optimizations
    try:
        setattr(trainer, "processing_class", processor)
        # Ensure fast tokenizer is used if available for better performance
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "is_fast"):
            if processor.tokenizer.is_fast:
                print("✅ Using fast tokenizer for optimized performance")
            else:
                print(
                    "⚠️ Using slow tokenizer - consider using fast tokenizer for better speed"
                )
    except Exception:
        pass

    # Initial eval (optional quick sanity)
    trainer.evaluate()

    trainer.train()

    # Final eval and save
    final_metrics = trainer.evaluate()

    # Only save checkpoint on rank 0 (main process) to avoid conflicts
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create final checkpoint directory using the same naming convention as regular checkpoints
        final_step = trainer.state.global_step
        final_checkpoint_dir = output_dir / f"checkpoint-{final_step}"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save inference-ready checkpoint (no optimizer states) in checkpoint subfolder
        print(
            f"💾 Saving final inference-ready checkpoint to checkpoint-{final_step}..."
        )
        _save_inference_checkpoint(
            model,
            processor,
            final_checkpoint_dir,
            coord_meta,
            max_coord_value,
            final_metrics,
        )

        print(
            f"✅ Training completed! Final checkpoint saved to {final_checkpoint_dir}"
        )
        print(f"📊 Final metrics: {final_metrics}")


def _save_inference_checkpoint(
    model, processor, output_dir, coord_meta, max_coord_value, final_metrics
):
    """Save inference-ready checkpoint with all essential configuration files."""

    # Save model with SafeTensors format for faster loading
    print("  - Saving model weights (SafeTensors format)...")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,  # Use SafeTensors format
        max_shard_size="10GB",  # Optimize shard size
    )

    # Save processor (includes tokenizer and image processor)
    print("  - Saving processor configuration...")
    processor.save_pretrained(str(output_dir))

    # Save coordinate token metadata
    print("  - Saving coordinate token configuration...")
    with (output_dir / "coord_token_ids.json").open("w", encoding="utf-8") as f:
        json.dump(coord_meta, f, ensure_ascii=False, indent=2)

    # Save comprehensive coordinate configuration for inference
    coord_cfg = {
        "coordinate_tokens_enabled": True,
        "max_coord_value": max_coord_value,
        "coordinate_token_count": int(max_coord_value) + 1,
        "vocab_size_after": int(
            getattr(processor.tokenizer, "vocab_size", len(processor.tokenizer))
        ),
        "coordinate_token_range": [
            min(coord_meta["coord_token_ids"])
            if coord_meta and "coord_token_ids" in coord_meta
            else None,
            max(coord_meta["coord_token_ids"])
            if coord_meta and "coord_token_ids" in coord_meta
            else None,
        ],
        "inference_ready": True,
        "training_completed": True,
    }
    with (output_dir / "coordinate_config.json").open("w", encoding="utf-8") as f:
        json.dump(coord_cfg, f, ensure_ascii=False, indent=2)

    # Save final training metrics
    print("  - Saving training metrics...")
    with (output_dir / "metrics-final.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    # Save chat template separately for easy access
    if hasattr(processor, "chat_template") and processor.chat_template:
        print("  - Saving chat template...")
        with (output_dir / "chat_template.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"chat_template": processor.chat_template},
                f,
                ensure_ascii=False,
                indent=2,
            )

    # Create inference usage instructions
    usage_instructions = {
        "usage": "This checkpoint is ready for inference with src_new pipeline",
        "model_path": str(output_dir),
        "coordinate_tokens_enabled": True,
        "max_coord_value": max_coord_value,
        "checkpoint_info": {
            "checkpoint_type": "final_inference_ready",
            "saved_in_checkpoint_subfolder": True,
            "note": "This follows the same folder structure as regular training checkpoints",
        },
        "example_usage": {
            "python": [
                "from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor",
                f"model = Qwen2_5_VLForConditionalGeneration.from_pretrained('{output_dir}')",
                f"processor = Qwen2_5_VLProcessor.from_pretrained('{output_dir}')",
                "# Model is ready for coordinate token inference",
            ]
        },
        "files_included": [
            "model weights (SafeTensors format)",
            "tokenizer configuration",
            "image processor configuration",
            "coordinate token mappings",
            "chat template",
            "training metrics",
        ],
    }

    with (output_dir / "README_INFERENCE.json").open("w", encoding="utf-8") as f:
        json.dump(usage_instructions, f, ensure_ascii=False, indent=2)

    print(
        f"  ✅ Inference-ready checkpoint saved with {len(usage_instructions['files_included'])} configuration files"
    )


if __name__ == "__main__":
    main()
