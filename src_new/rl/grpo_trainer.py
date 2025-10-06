"""Manual GRPO trainer for Qwen2.5-VL (TRL-free path)."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers.optimization import get_scheduler

from src_new.config.rl_config import EnhancedRLConfig
from src_new.rl import buffer, logprobs, losses, schedules, validators
from src_new.rl import distributed as dist_utils
from src_new.rl.rewards.standardizer import RewardStandardizer
from src_new.training.checkpoint_saver import BestCheckpointManager, CheckpointSaver
from src_new.training.training_state_manager import TrainingStateManager
from src_new.utils.rank_aware_logging import get_rank_aware_logger


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


_LOGGER = get_rank_aware_logger("rl.BBUGRPOTrainer")


@dataclass
class ManualTrainerConfig:
    sample_k: int
    max_new_tokens: int
    min_new_tokens: Optional[int]
    temperature: float
    temperature_schedule: str
    top_p: float
    repetition_penalty: float
    epsilon_low: float
    epsilon_high: float
    beta: float
    loss_type: str
    scale_rewards: bool
    mask_truncated_completions: bool
    max_advantage_magnitude: Optional[float]
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    steps_per_generation: int
    update_steps: int
    logging_steps: int
    save_steps: int
    max_steps: int
    bf16: bool
    standardize_rewards: bool
    cross_rank_advantages: bool
    beta_schedule: Optional[Dict[str, Any]]
    beta_start: float
    max_grad_norm: float


def _ensure_positive(name: str, value: int) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")
    return ivalue


def _get_pad_token_id(tokenizer: Any, model: nn.Module) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or int(pad_id) < 0:
        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None or int(pad_id) < 0:
        raise ValueError("Tokenizer/model must expose a valid pad_token_id")
    return int(pad_id)


class BBUGRPOTrainer:
    """Manual GRPO trainer that mirrors the SFT stack while avoiding TRL."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        processor: Any,
        train_dataset: Any,
        val_dataset: Any,
        reward_functions: Sequence[Callable[..., Sequence[float]]],
        reward_names: Sequence[str],
        reward_weights: Sequence[float],
        enhanced_cfg: EnhancedRLConfig,
        raw_config: Dict[str, Any],
        output_dir: str,
    ) -> None:
        if not reward_functions:
            raise ValueError("Manual trainer requires at least one reward function")
        if len(reward_functions) != len(reward_weights):
            raise ValueError("reward_functions and reward_weights size mismatch")
        if len(reward_functions) != len(reward_names):
            raise ValueError("reward_names size mismatch with reward_functions")

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_functions = list(reward_functions)
        self.reward_func_names = list(reward_names)
        self.reward_weight_list = [float(w) for w in reward_weights]
        self.enhanced_cfg = enhanced_cfg
        self.raw_config = raw_config
        self.output_dir = output_dir

        self.manual_cfg = self._build_manual_cfg(enhanced_cfg, raw_config)
        self.optimizer_cfg = enhanced_cfg.optimizer_config
        self.training_cfg = enhanced_cfg.training_config
        self.checkpoint_cfg = enhanced_cfg.checkpoint_config
        self.logging_cfg = enhanced_cfg.logging_config

        # Training utilities
        self._device = next(self.model.parameters()).device
        self._pad_token_id = _get_pad_token_id(tokenizer, model)
        self._reward_weights_tensor = torch.tensor(
            self.reward_weight_list, dtype=torch.float32, device=self._device
        )
        self._state_manager = TrainingStateManager(enhanced_cfg, model)
        self._best_ckpt_manager = BestCheckpointManager(
            metric_name="reward", greater_is_better=True
        )
        saver_args = SimpleNamespace(
            output_dir=output_dir,
            save_steps=self.checkpoint_cfg.save_steps,
            logging_steps=self.logging_cfg.logging_steps,
            save_total_limit=self.checkpoint_cfg.save_total_limit,
            best_checkpoint_interval_multiplier=10,
            best_checkpoint_min_interval_steps=None,
            should_save=True,
            eval_steps=self.checkpoint_cfg.eval_steps,
        )
        self._checkpoint_saver = CheckpointSaver(
            args=saver_args, checkpoint_manager=self._best_ckpt_manager
        )

        self._global_step = 0
        self._micro_step = 0
        self._start_time = time.time()
        self._buffer_chunks: List[Dict[str, Any]] = []
        self._buffer_chunk_idx = 0
        self._standardizer = (
            RewardStandardizer() if self.manual_cfg.standardize_rewards else None
        )

        self._distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self._distributed else 0
        self.world_size = dist.get_world_size() if self._distributed else 1

        # Derive accumulation strictly from sample_k: one optimizer update per generation
        # Set per-device micro-batch to 1 and use sample_k as micro-steps per update
        # Use local_k (per-rank K) to set accumulation when cross-rank sampling is enabled
        per_rank = max(1, int(self._local_sample_k()))
        self.manual_cfg.per_device_train_batch_size = 1
        self.manual_cfg.gradient_accumulation_steps = per_rank
        self.manual_cfg.steps_per_generation = per_rank
        if self.rank == 0:
            _LOGGER.info(
                "Using local_k=%d (from sample_k=%d, world_size=%d) to set grad_accum=steps_per_generation=%d",
                per_rank,
                self.manual_cfg.sample_k,
                self.world_size,
                per_rank,
            )

        self.ref_model: Optional[nn.Module] = None
        if self.manual_cfg.beta_start > 0.0:
            self.ref_model = copy.deepcopy(model).eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        self._sampler: Optional[DistributedSampler] = None
        self._sampler_epoch: int = 0
        self._temperature_scale: float = 1.0
        self._temperature_history: List[float] = []
        self._reward_history: List[float] = []
        self._clip_ratio_history: List[float] = []
        self._adv_std_history: List[float] = []
        self._adv_max_history: List[float] = []
        self._reward_component_history: Dict[str, List[float]] = {
            name: [] for name in self.reward_func_names
        }
        self._sample_prompt: Optional[str] = None
        self._sample_completion: Optional[str] = None
        self._last_grad_norm: float = 0.0
        self._last_console_summary: Optional[str] = None
        self._logged_ratio_diagnostic: bool = (
            False  # One-time verification of GRPO ratio fix
        )

        # TensorBoard writer (rank 0 only)
        self._tb_writer: Optional[Any] = None
        if self.rank == 0 and SummaryWriter is not None:
            # Extract run_name from output section first, then fallback to root level
            output_section = raw_config.get("output", {})
            run_name = output_section.get("run_name") or raw_config.get(
                "run_name", "rl_run"
            )
            tb_dir = raw_config.get("tb_dir", f"{output_dir}/tensorboard")
            tb_path = f"{tb_dir}/{run_name}"
            self._tb_writer = SummaryWriter(log_dir=tb_path)
            _LOGGER.info(f"TensorBoard logging to: {tb_path}")

        _LOGGER.info(
            "Manual GRPO trainer ready | sample_k=%d | lr=%.2e | max_steps=%d",
            self.manual_cfg.sample_k,
            float(self.optimizer_cfg.base_lr),
            self.manual_cfg.max_steps,
        )

        if (
            self.manual_cfg.steps_per_generation
            % self.manual_cfg.gradient_accumulation_steps
            != 0
        ):
            _LOGGER.warning(
                "steps_per_generation (%d) not divisible by gradient_accumulation_steps (%d);"
                " generation batches will span multiple optimizer updates",
                self.manual_cfg.steps_per_generation,
                self.manual_cfg.gradient_accumulation_steps,
            )

        # Cross-rank sampling is inferred: enabled when world_size > 1
        if self.world_size > 1:
            _LOGGER.info(
                "Cross-rank K sampling enabled | world_size=%d",
                self.world_size,
            )

    @staticmethod
    def _build_manual_cfg(
        enhanced_cfg: EnhancedRLConfig, raw_cfg: Dict[str, Any]
    ) -> ManualTrainerConfig:
        grpo_cfg = enhanced_cfg.grpo_config
        train_cfg = enhanced_cfg.training_config

        raw_training = raw_cfg.get("training", {})
        raw_grpo = raw_cfg.get("grpo", {})
        generation_cfg = raw_cfg.get("generation", {})

        grad_accum = raw_training.get(
            "gradient_accumulation_steps", train_cfg.gradient_accumulation_steps
        )
        per_device_batch = raw_training.get(
            "per_device_train_batch_size", train_cfg.per_device_train_batch_size
        )
        update_steps = raw_training.get("update_steps", grad_accum)
        steps_per_generation = raw_cfg.get("steps_per_generation", update_steps)
        max_steps = raw_training.get("max_steps", train_cfg.max_steps)
        if max_steps <= 0:
            raise ValueError("training.max_steps must be > 0 for manual trainer")

        min_new_tokens = (
            generation_cfg.get("min_new_tokens")
            if isinstance(generation_cfg, dict)
            else None
        )
        temperature_schedule = raw_grpo.get("temperature_schedule", "constant")
        max_adv_magnitude = raw_grpo.get("max_advantage_magnitude")
        beta_start = float(raw_grpo.get("beta_start", grpo_cfg.beta))
        beta_anneal = raw_grpo.get("beta_anneal")
        if beta_start <= 0.0:
            beta_schedule = None
        else:
            beta_schedule = beta_anneal if isinstance(beta_anneal, dict) else None

        normalization_cfg = raw_cfg.get("normalization", {})
        # Default to cross-rank advantages when running distributed
        cross_rank_adv = bool(normalization_cfg.get("cross_rank_advantages", False))
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                cross_rank_adv = True
        except Exception:
            pass

        standardize_rewards = bool(raw_grpo.get("standardize_rewards", False))

        # Distributed K sampling defaults handled at runtime (world_size driven)

        return ManualTrainerConfig(
            sample_k=_ensure_positive("grpo.sample_k", grpo_cfg.sample_k),
            max_new_tokens=_ensure_positive(
                "grpo.max_new_tokens", grpo_cfg.max_new_tokens
            ),
            min_new_tokens=int(min_new_tokens) if min_new_tokens is not None else None,
            temperature=float(grpo_cfg.temperature),
            temperature_schedule=str(temperature_schedule or "constant"),
            top_p=float(grpo_cfg.top_p),
            repetition_penalty=float(grpo_cfg.repetition_penalty),
            epsilon_low=float(grpo_cfg.epsilon_low),
            epsilon_high=float(grpo_cfg.epsilon_high),
            beta=float(grpo_cfg.beta),
            loss_type=str(grpo_cfg.loss_type),
            scale_rewards=bool(grpo_cfg.scale_rewards),
            mask_truncated_completions=bool(grpo_cfg.mask_truncated_completions),
            max_advantage_magnitude=(
                float(max_adv_magnitude) if max_adv_magnitude is not None else None
            ),
            gradient_accumulation_steps=_ensure_positive(
                "training.gradient_accumulation_steps", grad_accum
            ),
            per_device_train_batch_size=_ensure_positive(
                "training.per_device_train_batch_size", per_device_batch
            ),
            steps_per_generation=_ensure_positive(
                "steps_per_generation", steps_per_generation
            ),
            update_steps=_ensure_positive("training.update_steps", update_steps),
            logging_steps=_ensure_positive(
                "logging.logging_steps", enhanced_cfg.logging_config.logging_steps
            ),
            save_steps=_ensure_positive(
                "checkpointing.save_steps", enhanced_cfg.checkpoint_config.save_steps
            ),
            max_steps=_ensure_positive("training.max_steps", max_steps),
            bf16=bool(train_cfg.bf16),
            standardize_rewards=standardize_rewards,
            cross_rank_advantages=cross_rank_adv,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            max_grad_norm=float(enhanced_cfg.optimizer_config.max_grad_norm),
        )

    def _build_optimizer(self) -> AdamW:
        """Build AdamW with separate param groups for LLM, vision, and merger.

        Parameters are grouped by module name patterns so different learning
        rates can be applied as configured in OptimizerConfig.
        """

        params_llm: list[torch.nn.Parameter] = []
        params_vision: list[torch.nn.Parameter] = []
        params_merger: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Merger takes precedence if matched
            if "visual.merger" in name:
                params_merger.append(param)
                continue
            # Vision backbone (excluding merger)
            if ".visual." in name or name.startswith("visual."):
                params_vision.append(param)
                continue
            # Default to LLM group
            params_llm.append(param)

        if not (params_llm or params_vision or params_merger):
            raise ValueError("Model has no trainable parameters after phase freeze")

        # Resolve explicit learning rates per group (no base lr)
        lr_cfg = getattr(self.optimizer_cfg, "learning_rates", None)
        if lr_cfg is None:
            # Fallback to raw YAML for per-group learning rates
            try:
                lr_cfg = (self.raw_config.get("optimizer", {}) or {}).get(
                    "learning_rates"
                )
            except Exception:
                lr_cfg = None
        if not isinstance(lr_cfg, dict):
            raise ValueError(
                "optimizer.learning_rates must be provided in YAML with keys: llm, vision, merger"
            )
        try:
            lr_llm = float(lr_cfg["llm"])  # type: ignore[index]
            lr_vision = float(lr_cfg["vision"])  # type: ignore[index]
            lr_merger = float(lr_cfg["merger"])  # type: ignore[index]
        except Exception as e:
            raise ValueError(
                f"optimizer.learning_rates must define llm, vision, merger (got: {lr_cfg})"
            ) from e

        param_groups = []
        if params_llm:
            param_groups.append(
                {
                    "params": params_llm,
                    "lr": lr_llm,
                    "weight_decay": self.optimizer_cfg.weight_decay,
                }
            )
        if params_vision:
            param_groups.append(
                {
                    "params": params_vision,
                    "lr": lr_vision,
                    "weight_decay": self.optimizer_cfg.weight_decay,
                }
            )
        if params_merger:
            param_groups.append(
                {
                    "params": params_merger,
                    "lr": lr_merger,
                    "weight_decay": self.optimizer_cfg.weight_decay,
                }
            )

        # Prefer PyTorch fused AdamW kernels when available; fallback to foreach, then standard
        try:
            optimizer = AdamW(
                param_groups,
                lr=lr_llm,  # Global lr unused by per-group lrs; set equal to llm for logging
                betas=(self.optimizer_cfg.adam_beta1, self.optimizer_cfg.adam_beta2),
                eps=self.optimizer_cfg.adam_epsilon,
                weight_decay=self.optimizer_cfg.weight_decay,
                fused=True,
                foreach=True,
            )
            _LOGGER.info("Optimizer: using fused AdamW (foreach=True)")
        except TypeError:
            # Older PyTorch may not support fused
            try:
                optimizer = AdamW(
                    param_groups,
                    lr=lr_llm,
                    betas=(
                        self.optimizer_cfg.adam_beta1,
                        self.optimizer_cfg.adam_beta2,
                    ),
                    eps=self.optimizer_cfg.adam_epsilon,
                    weight_decay=self.optimizer_cfg.weight_decay,
                    foreach=True,
                )
                _LOGGER.info("Optimizer: using AdamW (foreach=True)")
            except TypeError:
                optimizer = AdamW(
                    param_groups,
                    lr=lr_llm,
                    betas=(
                        self.optimizer_cfg.adam_beta1,
                        self.optimizer_cfg.adam_beta2,
                    ),
                    eps=self.optimizer_cfg.adam_epsilon,
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
                _LOGGER.info("Optimizer: using standard AdamW")

        # Best-effort summary for debugging
        try:
            num_llm = sum(int(p.numel()) for p in params_llm)
            num_vision = sum(int(p.numel()) for p in params_vision)
            num_merger = sum(int(p.numel()) for p in params_merger)
            _LOGGER.info(
                "Optimizer param groups — llm=%d (lr=%.2e), vision=%d (lr=%.2e), merger=%d (lr=%.2e)",
                num_llm,
                lr_llm,
                num_vision,
                lr_vision,
                num_merger,
                lr_merger,
            )
        except Exception:
            pass

        return optimizer

    def _build_scheduler(self, optimizer: AdamW):
        return get_scheduler(
            name=self.training_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=self.manual_cfg.max_steps,
        )

    def _dataloader(self) -> DataLoader:
        sampler: Optional[DistributedSampler] = None
        if self._distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            self._sampler = sampler
        else:
            self._sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.manual_cfg.per_device_train_batch_size
            * self.manual_cfg.steps_per_generation,
            shuffle=not self._distributed,
            num_workers=self.training_cfg.dataloader_num_workers,
            pin_memory=self.training_cfg.pin_memory,
            collate_fn=lambda batch: batch,
            sampler=sampler,
        )

    def _next_generation_batch(
        self, iterator: Iterator[List[Dict[str, Any]]]
    ) -> tuple[List[Dict[str, Any]], Iterator[List[Dict[str, Any]]]]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self._dataloader())
            batch = next(iterator)
        if not isinstance(batch, list) or not batch:
            raise ValueError(
                "Generation batch must be a non-empty list of dataset records"
            )
        return batch, iterator

    def _prepare_generation_config(self, step: int) -> Dict[str, Any]:
        base_temperature = schedules.temperature_at(
            step=step,
            schedule=self.manual_cfg.temperature_schedule,
            base=self.manual_cfg.temperature,
            total_steps=self.manual_cfg.max_steps,
        )
        temperature = max(base_temperature * self._temperature_scale, 1e-4)
        return {
            "temperature": temperature,
            "top_p": self.manual_cfg.top_p,
            "repetition_penalty": self.manual_cfg.repetition_penalty,
            "max_new_tokens": self.manual_cfg.max_new_tokens,
            "min_new_tokens": self.manual_cfg.min_new_tokens,
        }

    def _beta_for_step(self, step: int) -> float:
        if self.manual_cfg.beta_start <= 0.0:
            return 0.0
        return schedules.beta_at(
            step,
            self.manual_cfg.beta_start,
            anneal=self.manual_cfg.beta_schedule,
        )

    def _local_sample_k(self) -> int:
        """Compute per-rank sample_k when cross-rank sampling is enabled."""
        k_total = int(self.manual_cfg.sample_k)
        if not (self.world_size > 1):
            return k_total
        # Cross-rank sampling: each rank produces ceil(sample_k / world_size)
        base = k_total // self.world_size
        rem = k_total % self.world_size
        return base + (1 if self.rank < rem else 0)

    def train(self) -> None:
        base_seed = int(self.enhanced_cfg.seed)
        rank_seed = dist_utils.seed_for_rank(base_seed)
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        dataloader = self._dataloader()
        dataloader_iter = iter(dataloader)

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        optimizer.zero_grad(set_to_none=True)

        self.model.train()
        scaler = GradScaler(enabled=False)
        use_autocast = bool(self.manual_cfg.bf16)
        autocast_dtype = torch.bfloat16 if use_autocast else torch.float32

        self._buffer_chunks = []
        self._buffer_chunk_idx = 0

        while self._global_step < self.manual_cfg.max_steps:
            # Determine reuse window: generate once per accumulation window and reuse
            reuse_window = max(1, int(self.manual_cfg.steps_per_generation))
            if self._buffer_chunk_idx >= len(self._buffer_chunks):
                if self._distributed and isinstance(self._sampler, DistributedSampler):
                    self._sampler.set_epoch(self._sampler_epoch)
                    self._sampler_epoch += 1
                batch, dataloader_iter = self._next_generation_batch(dataloader_iter)
                # Cross-rank sampling: force all ranks to use the same dataset index when world_size > 1
                if self.world_size > 1:
                    try:
                        # On rank 0, pick an index from the dataset; broadcast to others
                        if self.rank == 0:
                            import random as _rnd

                            idx_val = int(_rnd.randrange(len(self.train_dataset)))
                        else:
                            idx_val = 0
                        idx_tensor = torch.tensor([idx_val], dtype=torch.long)
                        idx_tensor = dist_utils.broadcast_indices(idx_tensor, src=0)
                        idx_b = int(idx_tensor.item())
                        # Rebuild a single-sample batch shared across ranks
                        shared_sample = self.train_dataset[idx_b]
                        batch = [shared_sample]
                    except Exception:
                        pass
                gen_cfg = self._prepare_generation_config(self._global_step)
                local_k = self._local_sample_k()
                # Generate once for the current mini-batch, then reuse for next `reuse_window` micro-steps
                chunks: List[Dict[str, Any]] = []
                for sample in batch:
                    single_generation = buffer.generate_and_score(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        inputs=[sample],
                        reward_fns=self.reward_functions,
                        reward_weights=self._reward_weights_tensor,
                        reward_names=self.reward_func_names,
                        reward_standardizer=self._standardizer,
                        pad_token_id=self._pad_token_id,
                        sample_k=local_k,
                        max_new_tokens=self.manual_cfg.max_new_tokens,
                        min_new_tokens=self.manual_cfg.min_new_tokens,
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                        repetition_penalty=self.manual_cfg.repetition_penalty,
                        mask_truncated_completions=self.manual_cfg.mask_truncated_completions,
                        scale_rewards=self.manual_cfg.scale_rewards,
                        max_advantage_magnitude=self.manual_cfg.max_advantage_magnitude,
                        reward_clip_sigma=self.raw_config.get("rewards_config", {}).get(
                            "clip_sigma"
                        ),
                    )
                    single_generation["temperature"] = gen_cfg["temperature"]
                    chunks.append(single_generation)
                # Build a small ring-buffer of length `reuse_window` by repeating the same chunk
                # This allows reusing generation outputs across micro-steps within the window
                self._buffer_chunks = []
                for _ in range(reuse_window):
                    self._buffer_chunks.extend(chunks)
                self._buffer_chunk_idx = 0

            generation_result = self._buffer_chunks[self._buffer_chunk_idx]
            self._buffer_chunk_idx += 1

            prompt_ids_all = generation_result["prompt_ids"].to(self._device)
            prompt_mask_all = generation_result["prompt_mask"].to(self._device)
            completion_ids_all = generation_result["completion_ids"].to(self._device)
            completion_mask_all = generation_result["completion_mask"].to(self._device)
            advantages_all = generation_result["advantages"].to(self._device)

            pixel_values = generation_result.get("pixel_values")
            image_grid_thw = generation_result.get("image_grid_thw")
            images_per_sample_full = generation_result.get("images_per_sample")

            # Keep pixel_values and image_grid_thw on CPU; slices will be moved inside get_per_token_logps
            # For single-sample streaming, use a single-image count vector for logprob slicing
            if images_per_sample_full is not None:
                try:
                    num_images = int(images_per_sample_full[0].item())
                except Exception:
                    num_images = (
                        int(images_per_sample_full.item())
                        if images_per_sample_full.numel() == 1
                        else 0
                    )
                images_per_sample_single = torch.tensor(
                    [num_images], dtype=torch.long, device=self._device
                )
            else:
                images_per_sample_single = None

            validators.debug_validate_image_alignment(
                tokenizer=self.tokenizer,
                prompt_ids=prompt_ids_all,
                image_grid_thw=image_grid_thw,
            )

            if self.manual_cfg.cross_rank_advantages and self.world_size > 1:
                # Recompute advantages across ranks using gathered rewards for the same sample
                try:
                    rewards_local = generation_result.get("rewards")
                    if rewards_local is not None:
                        rewards_local = rewards_local.detach()
                        # Gather all rewards across ranks and concatenate
                        gathered_rewards = dist_utils.all_gather_rewards(rewards_local)
                        global_mean = gathered_rewards.mean()
                        global_std = gathered_rewards.std(unbiased=False)
                        global_std = torch.clamp(global_std, min=1e-4)
                        # Use global statistics to normalize local rewards
                        local_adv = rewards_local - global_mean
                        if self.manual_cfg.scale_rewards:
                            local_adv = local_adv / global_std
                        if (
                            self.manual_cfg.max_advantage_magnitude is not None
                            and self.manual_cfg.max_advantage_magnitude > 0
                        ):
                            mag = float(self.manual_cfg.max_advantage_magnitude)
                            local_adv = torch.clamp(local_adv, min=-mag, max=mag)
                        advantages_all = local_adv.to(self._device)
                        generation_result["advantages"] = advantages_all
                except Exception:
                    pass

            current_beta = self._beta_for_step(self._global_step)
            generation_result["beta"] = current_beta

            with autocast(
                device_type="cuda", enabled=use_autocast, dtype=autocast_dtype
            ):
                # Prepare single prompt row
                if prompt_ids_all.dim() == 2:
                    prompt_ids_row = prompt_ids_all[0]
                    prompt_mask_row = prompt_mask_all[0]
                else:
                    prompt_ids_row = prompt_ids_all
                    prompt_mask_row = prompt_mask_all

                num_completions = int(completion_ids_all.size(0))
                # Accumulate only a CPU scalar for logging; stream backward per completion
                loss_value_for_log = 0.0
                non_finite = False
                for k in range(num_completions):
                    comp_ids_row = completion_ids_all[k]
                    comp_mask_row = completion_mask_all[k]
                    effective_len = int(comp_mask_row.long().sum().item())
                    if effective_len <= 0:
                        continue
                    comp_ids_eff = comp_ids_row[:effective_len]
                    comp_mask_eff = comp_mask_row[:effective_len].unsqueeze(0)

                    # Build 1-sample inputs
                    input_ids_k = torch.cat(
                        [prompt_ids_row, comp_ids_eff], dim=0
                    ).unsqueeze(0)
                    attention_mask_k = torch.cat(
                        [
                            prompt_mask_row,
                            torch.ones_like(comp_ids_eff, dtype=prompt_mask_row.dtype),
                        ],
                        dim=0,
                    ).unsqueeze(0)

                    per_token_logps_k = logprobs.get_per_token_logps(
                        model=self.model,
                        input_ids=input_ids_k,
                        attention_mask=attention_mask_k,
                        logits_to_keep=effective_len,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        images_per_sample=images_per_sample_single,
                        temperature=generation_result.get("temperature", 1.0),
                    )

                    # Use stored generation logprobs for proper GRPO ratio computation
                    # This ensures ratio = π_current / π_generation (not π_current / π_current!)
                    stored_gen_logps = generation_result.get("generation_logps")
                    if stored_gen_logps is not None and k < stored_gen_logps.size(0):
                        # Extract the k-th completion's generation logprobs
                        old_logps_k = stored_gen_logps[k, :effective_len].to(
                            self._device
                        )
                        # Ensure shapes match for loss computation
                        if old_logps_k.dim() == 1:
                            old_logps_k = old_logps_k.unsqueeze(0)
                        if old_logps_k.size(-1) != per_token_logps_k.size(-1):
                            # Pad or trim to match current logprobs length
                            if old_logps_k.size(-1) < per_token_logps_k.size(-1):
                                pad_len = per_token_logps_k.size(-1) - old_logps_k.size(
                                    -1
                                )
                                old_logps_k = torch.cat(
                                    [
                                        old_logps_k,
                                        torch.zeros(
                                            old_logps_k.size(0),
                                            pad_len,
                                            device=self._device,
                                        ),
                                    ],
                                    dim=-1,
                                )
                            else:
                                old_logps_k = old_logps_k[
                                    :, : per_token_logps_k.size(-1)
                                ]

                        # Diagnostic: verify ratio is not always 1.0 (one-time log)
                        if (
                            not self._logged_ratio_diagnostic
                            and self.rank == 0
                            and k == 0
                        ):
                            ratio_sample = torch.exp(per_token_logps_k - old_logps_k)
                            ratio_mean = float(ratio_sample.mean().item())
                            ratio_std = float(ratio_sample.std().item())
                            _LOGGER.info(
                                "✅ GRPO ratio diagnostic (step=%d): mean=%.4f std=%.4f (should NOT be 1.0±0.0)",
                                self._global_step,
                                ratio_mean,
                                ratio_std,
                            )
                            self._logged_ratio_diagnostic = True
                    else:
                        # Fallback: use current policy (old buggy behavior, but safe)
                        old_logps_k = per_token_logps_k.detach()
                        if (
                            not self._logged_ratio_diagnostic
                            and self.rank == 0
                            and k == 0
                        ):
                            _LOGGER.warning(
                                "⚠️  GRPO fallback: using current policy as old_logps (ratio will be ~1.0)"
                            )

                    per_token_kl_k = None
                    if current_beta > 0.0 and self.ref_model is not None:
                        with torch.no_grad():
                            ref_logps_k = logprobs.get_per_token_logps(
                                model=self.ref_model,
                                input_ids=input_ids_k,
                                attention_mask=attention_mask_k,
                                logits_to_keep=effective_len,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                images_per_sample=images_per_sample_single,
                                temperature=generation_result.get("temperature", 1.0),
                                detach=True,
                            )
                        per_token_kl_k = losses.compute_kl(
                            per_token_logps_k.detach(), ref_logps_k
                        )

                    adv_k = advantages_all[k : k + 1]
                    loss_k = losses.compute_grpo_loss(
                        per_token_logps_k,
                        old_logps_k,
                        adv_k,
                        comp_mask_eff,
                        epsilon_low=self.manual_cfg.epsilon_low,
                        epsilon_high=self.manual_cfg.epsilon_high,
                        loss_type=self.manual_cfg.loss_type,
                        beta=current_beta,
                        per_token_kl=per_token_kl_k,
                    )

                    if not torch.isfinite(loss_k):
                        non_finite = True
                        break

                    # Stream backward per completion to free graphs; average across K
                    scaled_part = loss_k / float(max(num_completions, 1))
                    scaler.scale(scaled_part).backward()
                    try:
                        loss_value_for_log += float(
                            loss_k.detach().float().item()
                        ) / float(max(num_completions, 1))
                    except Exception:
                        pass

                    # Free per-completion temporaries where possible
                    try:
                        del (
                            input_ids_k,
                            attention_mask_k,
                            per_token_logps_k,
                            old_logps_k,
                        )
                        if "ref_logps_k" in locals():
                            del ref_logps_k
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            if "non_finite" in locals() and non_finite:
                _LOGGER.warning(
                    "Non-finite loss detected at step %d; reducing temperature scale",
                    self._global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                self._temperature_scale = max(0.1, self._temperature_scale * 0.9)
                self._buffer_chunks = []
                self._buffer_chunk_idx = 0
                continue

            self._micro_step += 1

            if self._micro_step % self.manual_cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = clip_grad_norm_(
                    self.model.parameters(), self.manual_cfg.max_grad_norm
                )
                try:
                    self._last_grad_norm = float(
                        getattr(grad_norm_tensor, "item", lambda: grad_norm_tensor)()
                    )
                except Exception:
                    # Fallback: best-effort float conversion
                    self._last_grad_norm = (
                        float(grad_norm_tensor) if grad_norm_tensor is not None else 0.0
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]["lr"]
                self._global_step += 1
                self._micro_step = 0

                self._log_step(loss_value_for_log, generation_result, current_lr)
                self._maybe_checkpoint(generation_result)
                self._temperature_scale = min(1.0, self._temperature_scale * 1.01)

        _LOGGER.info("Manual GRPO training finished at step %d", self._global_step)

    def _log_step(
        self, loss_value: float, generation_result: Dict[str, Any], current_lr: float
    ) -> None:
        rewards = generation_result.get("rewards")
        if rewards is None:
            rewards = torch.zeros(1, device=self._device)
        rewards_for_log = rewards.detach()
        if self.world_size > 1:
            rewards_for_log = dist_utils.all_gather_rewards(rewards_for_log)
        reward_mean = float(rewards_for_log.mean().item())
        reward_std = float(rewards_for_log.std(unbiased=False).item())

        metrics = {
            "loss": loss_value,
            "reward": reward_mean,
            "reward_std": reward_std,
        }
        self._state_manager.accumulate_loss_components(metrics)

        if self._global_step % self.manual_cfg.logging_steps == 0:
            adv_tensor = generation_result.get("advantages")
            adv_std = 0.0
            adv_max = 0.0
            if adv_tensor is not None:
                adv_det = adv_tensor.detach()
                if self.world_size > 1:
                    adv_det = dist_utils.all_gather_rewards(adv_det)
                adv_std = float(adv_det.std(unbiased=False).item())
                adv_max = float(adv_det.abs().max().item())

            # ETA (minutes) and epoch progress (0..1)
            elapsed_sec = max(time.time() - self._start_time, 1e-6)
            remaining_steps = max(self.manual_cfg.max_steps - self._global_step, 0)
            steps_done = max(self._global_step, 1)
            eta_minutes = float((elapsed_sec / steps_done) * remaining_steps / 60.0)
            epoch_progress = float(self._global_step) / float(
                max(self.manual_cfg.max_steps, 1)
            )

            logs = self._state_manager.log_training_metrics(
                tr_loss=torch.tensor(loss_value),
                grad_norm=torch.tensor(self._last_grad_norm, dtype=torch.float32),
                model=self.model,
                start_time=self._start_time,
                learning_rate=current_lr,
                trainer_state=SimpleNamespace(
                    global_step=self._global_step, max_steps=self.manual_cfg.max_steps
                ),
            )
            logs.update(
                {
                    "step": self._global_step,
                    "reward": reward_mean,
                    "reward_std": reward_std,
                    "temperature": float(generation_result.get("temperature", 0.0)),
                    "elapsed_min": (time.time() - self._start_time) / 60.0,
                    "learning_rate": current_lr,
                    "beta": generation_result.get("beta", 0.0),
                    "grad_norm": self._last_grad_norm,
                    "eta_minutes": eta_minutes,
                    "epoch": epoch_progress,
                }
            )

            comp_lengths = generation_result.get("completion_lengths")
            if comp_lengths is not None:
                comp_lengths_det = comp_lengths.detach().float()
                if self.world_size > 1:
                    comp_lengths_det = dist_utils.all_gather_rewards(comp_lengths_det)
                logs["completions/mean_length"] = float(comp_lengths_det.mean().item())
                logs["completions/min_length"] = float(comp_lengths_det.min().item())
                logs["completions/max_length"] = float(comp_lengths_det.max().item())

            truncated_flags = generation_result.get("truncated_flags")
            if truncated_flags is not None:
                trunc = truncated_flags.detach().float()
                if self.world_size > 1:
                    trunc = dist_utils.all_gather_rewards(trunc)
                logs["completions/clipped_ratio"] = float(trunc.mean().item())

            terminated_flags = generation_result.get("terminated_with_eos")
            if terminated_flags is not None:
                term = terminated_flags.detach().float()
                if self.world_size > 1:
                    term = dist_utils.all_gather_rewards(term)
                logs["completions/terminated_ratio"] = float(term.mean().item())

            rewards_per_func = generation_result.get("rewards_per_func")
            if rewards_per_func is not None:
                values = rewards_per_func.detach()
                if self.world_size > 1:
                    values = dist_utils.all_gather_rewards(values)
                reward_names = generation_result.get(
                    "reward_names", self.reward_func_names
                )
                for idx, name in enumerate(reward_names):
                    if idx >= values.size(-1):
                        continue
                    col = values[:, idx]
                    logs[f"rewards/{name}/mean"] = float(col.mean().item())
                    logs[f"rewards/{name}/std"] = float(col.std(unbiased=False).item())

            if adv_tensor is not None:
                logs["advantages/std"] = adv_std
                logs["advantages/max_abs"] = adv_max

            # Unified metrics payload
            summary_parts = [
                f"step={self._global_step}",
                f"epoch={epoch_progress:.3f}",
                f"reward={reward_mean:.4f}±{reward_std:.4f}",
                f"lr={current_lr:.3e}",
                f"grad_norm={self._last_grad_norm:.3f}",
                f"eta={eta_minutes:.1f}min",
            ]
            if "advantages/std" in logs:
                summary_parts.append(f"adv_std={logs['advantages/std']:.4f}")
            if "advantages/max_abs" in logs:
                summary_parts.append(f"adv_max={logs['advantages/max_abs']:.4f}")
            if "completions/terminated_ratio" in logs:
                summary_parts.append(
                    f"term_ratio={logs['completions/terminated_ratio']:.3f}"
                )
            if "completions/clipped_ratio" in logs:
                summary_parts.append(
                    f"clip_ratio={logs['completions/clipped_ratio']:.3f}"
                )
            # Include a couple of key reward components if present
            for key in ("parse", "wrappers", "coords", "separators"):
                mk = f"rewards/{key}/mean"
                if mk in logs:
                    summary_parts.append(f"{key}={logs[mk]:.3f}")

            self._last_console_summary = "[" + " ".join(summary_parts) + "]"
            _LOGGER.info(self._last_console_summary)
            self._state_manager.reset_metrics_state()

            # Log to TensorBoard with conventional tags (rank 0 only)
            if self.rank == 0 and self._tb_writer is not None:
                # Core training metrics (without loss)
                self._tb_writer.add_scalar(
                    "train/learning_rate", current_lr, self._global_step
                )
                self._tb_writer.add_scalar(
                    "train/grad_norm", self._last_grad_norm, self._global_step
                )
                self._tb_writer.add_scalar(
                    "train/epoch", epoch_progress, self._global_step
                )

                # RL-specific metrics
                self._tb_writer.add_scalar("reward", reward_mean, self._global_step)
                self._tb_writer.add_scalar("reward_std", reward_std, self._global_step)
                self._tb_writer.add_scalar(
                    "temperature", logs.get("temperature", 0.0), self._global_step
                )
                self._tb_writer.add_scalar(
                    "beta", logs.get("beta", 0.0), self._global_step
                )
                self._tb_writer.add_scalar(
                    "eta_minutes", eta_minutes, self._global_step
                )
                self._tb_writer.add_scalar("step", self._global_step, self._global_step)
                # Diversity metric (optional)
                try:
                    # Compute unique_ratio on first chunk for preview
                    completions_list = generation_result.get("completions") or []
                    if completions_list:
                        distinct = len(set(completions_list))
                        total = len(completions_list)
                        unique_ratio = float(distinct) / float(max(total, 1))
                        self._tb_writer.add_scalar(
                            "completions/unique_ratio", unique_ratio, self._global_step
                        )
                except Exception:
                    pass

                # Per-reward metrics under /rewards
                reward_names = generation_result.get(
                    "reward_names", self.reward_func_names
                )
                for name in reward_names:
                    mean_key = f"rewards/{name}/mean"
                    std_key = f"rewards/{name}/std"
                    if mean_key in logs:
                        self._tb_writer.add_scalar(
                            mean_key, logs[mean_key], self._global_step
                        )
                    if std_key in logs:
                        self._tb_writer.add_scalar(
                            std_key, logs[std_key], self._global_step
                        )

            if self.rank == 0:
                prompts = generation_result.get("prompts") or []
                completions = generation_result.get("completions") or []
                if prompts and completions:
                    preview = completions[0].replace("\n", " ")[:200]
                    _LOGGER.debug("Sample completion: %s", preview)
                    if self._sample_prompt is None:
                        self._sample_prompt = str(prompts[0])
                        self._sample_completion = str(completions[0])

            temperature_logged = float(generation_result.get("temperature", 0.0))
            clip_ratio = float(logs.get("completions/clipped_ratio", 0.0))
            self._temperature_history.append(temperature_logged)
            self._reward_history.append(reward_mean)
            self._clip_ratio_history.append(clip_ratio)
            self._adv_std_history.append(adv_std)
            self._adv_max_history.append(adv_max)
            for name in self.reward_func_names:
                comp_mean = logs.get(f"rewards/{name}/mean")
                if comp_mean is not None:
                    self._reward_component_history.setdefault(name, []).append(
                        float(comp_mean)
                    )

    def _maybe_checkpoint(self, generation_result: Dict[str, Any]) -> None:
        if self.manual_cfg.save_steps <= 0:
            return
        if self._global_step % self.manual_cfg.save_steps != 0:
            return

        # Only rank 0 saves, but all ranks must wait at barrier
        if self.rank == 0:
            rewards = generation_result.get("rewards")
            mean_reward = 0.0
            if rewards is not None:
                rewards_for_ckpt = rewards.detach()
                if self.world_size > 1:
                    rewards_for_ckpt = dist_utils.all_gather_rewards(rewards_for_ckpt)
                mean_reward = float(rewards_for_ckpt.mean().item())
            metrics = {"reward": mean_reward}
            try:
                # Unwrap DDP if present to save underlying model
                model_to_save = getattr(self.model, "module", self.model)
                self._checkpoint_saver.save_checkpoint(
                    model=model_to_save,
                    processing_class=self.tokenizer,
                    processor=self.processor,
                    step=self._global_step,
                    current_metrics=metrics,
                    is_eval_step=False,
                    force_step_save=True,
                    is_deepspeed_enabled=False,
                    training_start_time=self._start_time,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                _LOGGER.warning(
                    "Checkpoint save failed at step %d: %s", self._global_step, exc
                )

        # Barrier: all ranks wait for rank 0 to finish saving before continuing
        if self._distributed:
            dist_utils.barrier()

    @property
    def temperature_history(self) -> List[float]:
        return list(self._temperature_history)

    @property
    def reward_history(self) -> List[float]:
        return list(self._reward_history)

    @property
    def clip_ratio_history(self) -> List[float]:
        return list(self._clip_ratio_history)

    @property
    def adv_std_history(self) -> List[float]:
        return list(self._adv_std_history)

    @property
    def adv_max_history(self) -> List[float]:
        return list(self._adv_max_history)

    @property
    def reward_component_history(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._reward_component_history.items()}

    @property
    def sample_prompt(self) -> Optional[str]:
        return self._sample_prompt

    @property
    def sample_completion(self) -> Optional[str]:
        return self._sample_completion

    def close(self) -> None:
        """Clean up resources (e.g., TensorBoard writer)."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
                _LOGGER.info("TensorBoard writer closed")
            except Exception as e:
                _LOGGER.warning(f"Failed to close TensorBoard writer: {e}")


__all__ = ["BBUGRPOTrainer"]
