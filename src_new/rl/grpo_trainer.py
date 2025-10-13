
"""Manual GRPO trainer for Qwen2.5-VL (TRL-free path)."""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler

from src_new.config.rl_config_v2 import RLConfig
from src_new.rl import buffer, logprobs, losses, schedules
from src_new.rl.eval import compute_absolute_metrics as _abs_metrics
from src_new.rl.eval_utils import dump_samples
from src_new.rl.generation_buffer import GenerationBuffer
from src_new.rl.metrics_aggregator import MetricsAggregator
from src_new.rl.metrics_keys import (
    BUFFER_EFFICIENCY,
    BUFFER_REUSE_ACTIVE,
    BUFFER_REUSE_COUNT,
    BUFFER_STEPS_PER_GEN,
)
from src_new.rl.reward_logger import RewardLogger
from src_new.rl.rewards.standardizer import RewardStandardizer
from src_new.rl.training_state import GRPOTrainingState
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
    prompt_batch_size: int
    trajectories_per_cycle: int
    reward_average_window: int
    max_new_tokens: int
    min_new_tokens: Optional[int]
    temperature: float
    temperature_schedule: str
    top_p: float
    epsilon_low: float
    epsilon_high: float
    beta: float
    loss_type: str
    scale_rewards: bool
    mask_truncated_completions: bool
    max_advantage_magnitude: Optional[float]
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    steps_per_generation: (
        int  # How many optimizer steps to reuse each generation buffer
    )
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
        rl_config: RLConfig,  # v2 config only
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
        self.config = rl_config  # Typed config only

        # Initialize unified reward logger
        self._reward_logger = RewardLogger(reward_names=self.reward_func_names)
        self.output_dir = output_dir
        # Run name from typed config
        self._run_name = rl_config.experiment.run_name

        self.manual_cfg = self._build_manual_cfg(rl_config, train_dataset)
        self._expected_trajectories = int(self.manual_cfg.trajectories_per_cycle)
        self._drop_last_enabled = True

        # Sampling configuration (derive before Accelerator so we can set accumulation)
        # GLOBAL-K semantics: sample_k is total completions per prompt across all ranks
        # Resolve world size and rank from env (works before process group init)
        try:
            _ws_env = int(os.getenv("WORLD_SIZE", "1"))
        except Exception:
            _ws_env = 1
        try:
            _rank_env = int(os.getenv("RANK", "0"))
        except Exception:
            _rank_env = 0

        # Temporarily set placeholders; will be updated after Accelerator creation
        self.world_size = _ws_env
        self.rank = _rank_env
        self._distributed = self.world_size > 1

        # Compute local sample_k for accumulation boundary
        k_total = int(self.manual_cfg.sample_k)
        prompt_batch_size_cfg = max(1, int(self.manual_cfg.prompt_batch_size))
        if self.world_size <= 1:
            _local_k_for_acc = k_total
        else:
            _base = k_total // self.world_size
            _rem = k_total % self.world_size
            if _rem != 0:
                raise ValueError(
                    "sampling.sample_k must be divisible by world_size (GLOBAL-K semantics)"
                )
            _local_k_for_acc = _base
        _local_completions_for_acc = max(
            1, int(_local_k_for_acc) * prompt_batch_size_cfg
        )

        # Instantiate Accelerator with YAML-driven precision and accumulation boundary
        mp = "bf16" if self.manual_cfg.bf16 else "no"

        # Configure NCCL timeout for collective operations (default: 120 seconds)
        nccl_timeout_seconds = int(os.getenv("NCCL_COLLECTIVE_TIMEOUT", "120"))
        init_pg_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=nccl_timeout_seconds)
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        self.accelerator = Accelerator(
            mixed_precision=mp,
            gradient_accumulation_steps=_local_completions_for_acc,
            kwargs_handlers=[init_pg_kwargs, ddp_kwargs],
        )

        # After Accelerator is ready, update distributed attributes
        self.rank = getattr(self.accelerator.state, "process_index", 0)
        self.world_size = getattr(self.accelerator.state, "num_processes", 1)
        self._distributed = self.world_size > 1

        # Training utilities
        self._device = self.accelerator.device
        # Metrics aggregator for consistent cross-rank stats
        try:
            self._metrics_aggregator = MetricsAggregator(self.accelerator, self._device)
        except Exception:
            self._metrics_aggregator = None
        self._pad_token_id = _get_pad_token_id(tokenizer, model)
        self._reward_weights_tensor = torch.tensor(
            self.reward_weight_list, dtype=torch.float32, device=self._device
        )
        self._state_manager = TrainingStateManager(self.config, model)
        self._best_ckpt_manager = BestCheckpointManager(
            metric_name="reward", greater_is_better=True
        )
        saver_args = SimpleNamespace(
            output_dir=self.output_dir,
            save_steps=self.config.checkpointing.save_steps,
            logging_steps=self.config.logging.logging_steps,
            save_total_limit=self.config.checkpointing.save_total_limit,
            best_checkpoint_interval_multiplier=10,
            best_checkpoint_min_interval_steps=None,
            should_save=True,
            eval_steps=None,  # RL uses periodic evaluation, not separate eval_steps
        )
        self._checkpoint_saver = CheckpointSaver(
            args=saver_args, checkpoint_manager=self._best_ckpt_manager
        )

        self._micro_step = 0
        self._start_time = time.time()

        # Unified training state (replaces scattered counters and buffer management)
        self._training_state = GRPOTrainingState(
            global_step=0,
            sampler_pos=0,
            generation_buffer=None,
            buffer_step_idx=0,
        )

        # Build reward standardizer from config when enabled
        if self.manual_cfg.standardize_rewards:
            from src_new.config.rl_config_v2 import StandardizerConfig

            std_cfg = getattr(self.config.rewards.config, "standardizer", None)
            if std_cfg is None:
                raise ValueError(
                    "standardize_rewards is True but 'rewards_config.standardizer' block is missing in YAML"
                )
            else:
                # Accept either a StandardizerConfig instance or a dict
                if isinstance(std_cfg, StandardizerConfig):
                    std_config = std_cfg
                else:
                    std_config = StandardizerConfig.from_dict(std_cfg)
                self._standardizer = RewardStandardizer(
                    mode=std_config.mode,
                    momentum=float(std_config.momentum),
                    warmup_steps=int(std_config.warmup_steps),
                    min_std=float(std_config.min_std),
                    clip_abs=float(std_config.clip),
                )
            try:
                _LOGGER.info(
                    "Reward standardization: ENABLED (mode=%s, momentum=%.3f, warmup=%d, min_std=%.3f, clip=%.1f)",
                    getattr(self._standardizer, "mode", "std_only"),
                    getattr(self._standardizer, "momentum", 0.97),
                    getattr(self._standardizer, "warmup_steps", 10),
                    getattr(self._standardizer, "min_std", 0.05),
                    getattr(self._standardizer, "clip_abs", 3.0),
                )
            except Exception:
                pass
        else:
            self._standardizer = None
        if len(self.train_dataset) < int(self.manual_cfg.prompt_batch_size):
            raise ValueError(
                "prompt_batch.prompt_batch_size exceeds available prompts; "
                f"dataset has {len(self.train_dataset)} prompts but "
                f"requires at least {self.manual_cfg.prompt_batch_size}"
            )
        self._epoch_size: int = len(self.train_dataset)

        # Sampling window (based on current world_size)

        # Derive accumulation strictly from the sampling configuration so each
        # optimizer step always corresponds to a consistent set of completions.
        self._configure_sampling_window()

        # Prompt batch telemetry tracker (per-rank expected trajectories)
        from src_new.rl.prompt_batch_telemetry import PromptBatchTelemetryTracker

        self._expected_trajectories = int(self.local_completions_per_cycle)
        self._prompt_batch_telemetry = PromptBatchTelemetryTracker(
            expected_trajectories_per_cycle=self._expected_trajectories,
            reward_average_window=int(self.manual_cfg.reward_average_window),
        )

        self.ref_model: Optional[nn.Module] = None
        if self.manual_cfg.beta_start > 0.0:
            self.ref_model = copy.deepcopy(model).eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        self._sampler = None
        self._sampler_epoch: int = 0
        self._temperature_scale: float = 1.0
        self._temperature_history: List[float] = []
        self._reward_history: List[float] = []
        self._adv_std_history: List[float] = []
        self._adv_max_history: List[float] = []
        self._reward_component_history: Dict[str, List[float]] = {}

        # TensorBoard writer (initialized lazily in train method)
        self._tb_writer: Optional[Any] = None

        # TensorBoard logger wrapper (will be set in train())
        self._tb_logger: Optional[Any] = None

        # Initialize console metrics collector
        from src_new.rl.console_metrics_collector import ConsoleMetricsCollector

        self._console_collector = ConsoleMetricsCollector()

        # Debug and performance monitoring flags (from environment)
        self._debug_timing: bool = bool(
            os.getenv("DEBUG_TIMING", "0").lower() in ("1", "true", "yes")
        )
        # Generation warning threshold (seconds) - warn if generation takes longer
        # Centralized diagnostics settings (no behavior change)
        try:
            from src_new.rl.diagnostics.settings import SETTINGS as _DBG

            self._generation_warn_threshold = float(_DBG.generation_warn_threshold)
            self._debug_timing = bool(_DBG.debug_timing)
        except Exception:
            self._generation_warn_threshold = float(
                os.getenv("GENERATION_WARN_THRESHOLD", "60.0")
            )

        # Gradient norm tracking (initialized before first step)
        self._last_grad_norm: float = 0.0

        # Console summary cache
        self._last_console_summary: str = ""

        # Ensure we log to console/TensorBoard at most once per optimizer step
        self._last_logged_step: int = -1

        # Evaluation configuration (from config)
        self._eval_enabled: bool = rl_config.evaluation.enabled
        self._eval_every_steps: int = rl_config.evaluation.eval_every_steps
        self._eval_rounds: int = rl_config.evaluation.rounds
        self._eval_per_rank_samples: int = rl_config.evaluation.per_rank_samples
        self._eval_save_samples: int = rl_config.evaluation.save_samples
        self._eval_log_text: bool = rl_config.evaluation.log_text_snippets
        self._eval_seed: int = rl_config.evaluation.seed

        # GRPO ratio diagnostic flag (one-time log to verify trust region)
        self._logged_ratio_diagnostic: bool = False

        # Vision cache for cached vision tensors
        self._vision_cache = None

        # Warn-once flag used in cross-rank advantage normalization
        self._warned_rewards_mismatch: bool = False

    @property
    def _global_step(self) -> int:
        """Backward compatibility property for _global_step."""
        return self._training_state.global_step

    @_global_step.setter
    def _global_step(self, value: int) -> None:
        """Backward compatibility setter for _global_step."""
        self._training_state.global_step = value

    @property
    def _sampler_pos(self) -> int:
        """Backward compatibility property for _sampler_pos."""
        return self._training_state.sampler_pos

    @_sampler_pos.setter
    def _sampler_pos(self, value: int) -> None:
        """Backward compatibility setter for _sampler_pos."""
        self._training_state.sampler_pos = value

    def _log_step(
        self, loss_value: float, generation_result: Dict[str, Any], current_lr: float
    ) -> None:
        # Guard against duplicate logging within the same optimizer step
        if self._last_logged_step == self._global_step:
            return
        # Gather all metrics across ranks FIRST, then log only on rank 0
        rewards = generation_result.get("rewards")
        if rewards is None:
            rewards = torch.zeros(1, device=self._device)
        rewards_for_log = rewards.detach()
        if self.world_size > 1:
            rewards_for_log = self.accelerator.gather(rewards_for_log)
        reward_mean = float(rewards_for_log.mean().item())
        reward_std = float(rewards_for_log.std(unbiased=False).item())

        # Absolute (unnormalized) reward logging
        raw_rewards = generation_result.get("raw_rewards")
        if raw_rewards is not None:
            raw_for_log = raw_rewards.detach()
            if self.world_size > 1:
                raw_for_log = self.accelerator.gather(raw_for_log)
            raw_reward_mean = float(raw_for_log.mean().item())
            raw_reward_std = float(raw_for_log.std(unbiased=False).item())
        else:
            raw_reward_mean = None
            raw_reward_std = None

        metrics = {
            "loss": loss_value,
            "reward": reward_mean,
            "reward_std": reward_std,
        }
        if raw_reward_mean is not None and raw_reward_std is not None:
            metrics["raw_reward"] = raw_reward_mean
            metrics["raw_reward_std"] = raw_reward_std
        self._state_manager.accumulate_loss_components(metrics)

        buffer_is_fresh = bool(generation_result.get("buffer/is_fresh", False))
        if buffer_is_fresh or (self._global_step % self.manual_cfg.logging_steps == 0):
            # Gather advantages across ranks via aggregator
            adv_tensor = generation_result.get("advantages")
            adv_std = 0.0
            adv_max = 0.0
            if adv_tensor is not None:
                try:
                    if self._metrics_aggregator is not None:
                        _adv = self._metrics_aggregator.gather_advantage_stats(
                            generation_result
                        )
                        adv_std = float(_adv.get("std", 0.0))
                        adv_max = float(_adv.get("max_abs", 0.0))
                    else:
                        adv_det = adv_tensor.detach()
                        if adv_det.device.type == "cpu":
                            adv_det = adv_det.to(self.accelerator.device)
                        if self.world_size > 1:
                            adv_det = self.accelerator.gather(adv_det)
                        adv_std = float(adv_det.std(unbiased=False).item())
                        adv_max = float(adv_det.abs().max().item())
                except Exception:
                    pass

            # Gather per-reward components across ranks (normalized)
            rewards_per_func = generation_result.get("rewards_per_func")
            if rewards_per_func is not None:
                rewards_det = rewards_per_func.detach()
                # Move to GPU before gather if needed
                if rewards_det.device.type == "cpu":
                    rewards_det = rewards_det.to(self.accelerator.device)
                if self.world_size > 1:
                    rewards_per_func = self.accelerator.gather(rewards_det)
                else:
                    rewards_per_func = rewards_det

            # Gather raw per-reward components across ranks
            raw_pf = generation_result.get("raw_rewards_per_func")
            if raw_pf is not None:
                raw_pf_det = raw_pf.detach()
                # Move to GPU before gather if needed
                if raw_pf_det.device.type == "cpu":
                    raw_pf_det = raw_pf_det.to(self.accelerator.device)
                if self.world_size > 1:
                    raw_pf = self.accelerator.gather(raw_pf_det)
                else:
                    raw_pf = raw_pf_det

            # Gather termination ratio via aggregator
            try:
                if self._metrics_aggregator is not None:
                    term_ratio = float(
                        self._metrics_aggregator.gather_termination_ratio(
                            generation_result
                        )
                    )
                else:
                    terminated_flags = generation_result.get("terminated_with_eos")
                    if terminated_flags is not None:
                        term = terminated_flags.detach().float()
                        if term.device.type == "cpu":
                            term = term.to(self.accelerator.device)
                        if self.world_size > 1:
                            term = self.accelerator.gather(term)
                        term_ratio = float(term.mean().item())
                    else:
                        term_ratio = 0.0
            except Exception:
                term_ratio = 0.0

            # Warn if truncation is being forced frequently
            try:
                cap_hit = None
                trunc_flags = generation_result.get("truncated_flags")
                trunc_ratio = None
                if trunc_flags is not None:
                    tf = trunc_flags.detach().float()
                    if tf.device.type == "cpu":
                        tf = tf.to(self.accelerator.device)
                    if self.world_size > 1:
                        tf = self.accelerator.gather(tf)
                    trunc_ratio = float(tf.mean().item())
                # Rank-0 only warning
                if self.accelerator.is_main_process:
                    if isinstance(cap_hit, float) and cap_hit > 0.0:
                        self.logger.warning(
                            f"Low EOS termination: term_ratio={term_ratio:.3f}. Consider increasing max_new_tokens or adjusting temperature/top_p."
                        )
                    if trunc_ratio is not None and trunc_ratio > 0.0:
                        self.logger.warning(
                            f"Truncation masked tokens post-generation: truncated_ratio={trunc_ratio:.3f}; term_ratio={term_ratio:.3f}."
                        )
            except Exception:
                pass

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
                    "learning_rate": current_lr,
                    "beta": generation_result.get("beta", 0.0),
                    "grad_norm": self._last_grad_norm,
                    "eta_minutes": eta_minutes,
                    "epoch": epoch_progress,
                    "completions/terminated_ratio": term_ratio,
                    # Tail sanitizer usage (if provided by buffer)
                    "sanitizer/applied_ratio": float(
                        generation_result.get("sanitizer/applied_ratio", 0.0)
                    ),
                    # Pass through zero-length ratio if computed by buffer
                    "completions/zero_len_ratio": float(
                        generation_result.get("completions/zero_len_ratio", 0.0)
                    ),
                }
            )
            # Only keep raw reward metrics if available
            if raw_reward_mean is not None and raw_reward_std is not None:
                logs["raw_reward"] = raw_reward_mean
                logs["raw_reward_std"] = raw_reward_std
            # Remove redundant time metrics from RL logs to reduce clutter
            logs.pop("train_runtime", None)
            logs.pop("train_samples_per_second", None)
            logs.pop("remaining_hrs", None)

            # Add per-reward components to logs (using gathered tensors)
            self._add_reward_components_to_logs(
                logs, rewards_per_func, raw_pf, generation_result
            )

            # Add dynamic length, GT ratios, and clipping to logs
            self._add_supplementary_metrics_to_logs(logs, generation_result)

            # Add advantage metrics
            if adv_tensor is not None:
                logs["advantages/std"] = adv_std
                logs["advantages/max_abs"] = adv_max

            # Console logging - ONLY ON RANK 0 (single concise line per optimizer step)
            if self.accelerator.is_main_process:
                try:
                    eta_hours = float(eta_minutes) / 60.0
                except Exception:
                    eta_hours = 0.0
                _LOGGER.info(
                    "[step=%d reward=%.4f±%.4f raw=%.4f±%.4f lr=%.3e grad=%.3f eta=%.2fh]",
                    self._global_step,
                    reward_mean,
                    reward_std,
                    0.0 if raw_reward_mean is None else raw_reward_mean,
                    0.0 if raw_reward_std is None else raw_reward_std,
                    float(current_lr),
                    float(self._last_grad_norm),
                    eta_hours,
                )
                # Print per-reward raw metrics once per optimizer step
                try:
                    reward_names = generation_result.get(
                        "reward_names", self.reward_func_names
                    )
                    parts = []
                    for name in reward_names:
                        key = f"raw_rewards/{name}/mean"
                        if key in logs:
                            try:
                                val = float(logs[key])
                                parts.append(f"{name}={val:.3f}")
                            except Exception:
                                continue
                    if parts:
                        _LOGGER.info("[REWARDS_RAW] " + " ".join(parts))
                except Exception:
                    pass

            # TensorBoard logging - ONLY ON RANK 0 (once per optimizer step)
            if self.accelerator.is_main_process and self._tb_logger is not None:
                reward_names = generation_result.get(
                    "reward_names", self.reward_func_names
                )
                self._tb_logger.log_all_from_dict(logs, reward_names, self._global_step)

            # History tracking (only on rank 0)
            if self.accelerator.is_main_process:
                self._update_history(
                    generation_result, reward_mean, adv_std, adv_max, logs
                )

            self._state_manager.reset_metrics_state()
        else:
            # Silence compact one-liner to avoid repeated logs within the prompt batch window
            pass

        # Mark this step as logged
        self._last_logged_step = self._global_step

    def _add_reward_components_to_logs(
        self,
        logs: Dict[str, Any],
        rewards_per_func: Optional[torch.Tensor],
        raw_pf: Optional[torch.Tensor],
        generation_result: Dict[str, Any],
    ) -> None:
        """Add per-reward components to logs dict."""
        # Per-reward components (raw/unnormalized)
        if raw_pf is not None:
            names = generation_result.get("reward_names", self.reward_func_names)
            for idx, name in enumerate(names):
                if idx >= raw_pf.size(-1):
                    continue
                col = raw_pf[:, idx]
                logs[f"raw_rewards/{name}/mean"] = float(col.mean().item())
                logs[f"raw_rewards/{name}/std"] = float(col.std(unbiased=False).item())

        # Per-reward components (normalized)
        if rewards_per_func is not None:
            reward_names = generation_result.get("reward_names", self.reward_func_names)
            for idx, name in enumerate(reward_names):
                if idx >= rewards_per_func.size(-1):
                    continue
                col = rewards_per_func[:, idx]
                logs[f"rewards/{name}/mean"] = float(col.mean().item())
                logs[f"rewards/{name}/std"] = float(col.std(unbiased=False).item())

    def _add_supplementary_metrics_to_logs(
        self,
        logs: Dict[str, Any],
        generation_result: Dict[str, Any],
    ) -> None:
        """Add GT ratios and clipping metrics to logs (dynamic caps removed)."""

        # GT-aligned length diagnostics
        try:
            meta_list = generation_result.get("meta") or []
            ratios: list[float] = []
            for m in meta_list:
                if not isinstance(m, dict):
                    continue
                gt_len = m.get("gt_len_tokenizer")
                gen_len = m.get("gen_len_tokenizer")
                if (
                    isinstance(gt_len, int)
                    and gt_len > 0
                    and isinstance(gen_len, int)
                    and gen_len >= 0
                ):
                    ratios.append(float(gen_len) / float(gt_len))
            if ratios:
                lvgt_cfg = self.config.rewards.config.length_vs_gt
                lower = float(lvgt_cfg.lower)
                upper = float(lvgt_cfg.upper)
                arr = torch.tensor(ratios, dtype=torch.float32)
                logs["completions/ratio_to_gt_mean"] = float(arr.mean().item())
                over = (arr > upper).float().mean().item()
                under = (arr < lower).float().mean().item()
                logs["completions/over_upper_ratio"] = float(over)
                logs["completions/under_lower_ratio"] = float(under)
        except Exception:
            pass

        # Tokenizer-based length stats and cap-hit ratio
        for key in (
            "completions/mean_len_tok",
            "gt/mean_len_tok",
            # cap hit ratio removed
        ):
            if key in generation_result:
                try:
                    logs[key] = float(generation_result.get(key, 0.0))
                except Exception:
                    pass
        # Also log truncation ratio if available
        try:
            trunc_flags = generation_result.get("truncated_flags")
            if trunc_flags is not None:
                tf = trunc_flags.detach().float()
                if tf.device.type == "cpu":
                    tf = tf.to(self.accelerator.device)
                if self.world_size > 1:
                    tf = self.accelerator.gather(tf)
                logs["completions/truncated_ratio"] = float(tf.mean().item())
        except Exception:
            pass

        # Policy clipping diagnostics
        try:
            _clip_low = generation_result.get("policy_clip_low_ratio")
            _clip_high = generation_result.get("policy_clip_high_ratio")
            _clip_region = generation_result.get("policy_clip_region_ratio")
            if (
                _clip_low is not None
                and _clip_high is not None
                and _clip_region is not None
            ):
                _vals = torch.tensor(
                    [_clip_low, _clip_high, _clip_region],
                    dtype=torch.float32,
                    device=self._device,
                )
                if self.world_size > 1:
                    _vals = self.accelerator.gather(_vals)
                _vals_mean = _vals.mean(dim=0)
                logs["clip_ratio/low_mean"] = float(_vals_mean[0].item())
                logs["clip_ratio/high_mean"] = float(_vals_mean[1].item())
                logs["clip_ratio/region_mean"] = float(_vals_mean[2].item())
        except Exception:
            pass

        # Buffer reuse metrics (Swift-style)
        try:
            steps_per_gen = self.manual_cfg.steps_per_generation
            logs[BUFFER_STEPS_PER_GEN] = int(steps_per_gen)
            logs[BUFFER_REUSE_COUNT] = int(steps_per_gen)
            logs[BUFFER_EFFICIENCY] = float(steps_per_gen)
            # Clarify: reuse currently not active (no multi-step buffer reuse yet)
            logs[BUFFER_REUSE_ACTIVE] = 0
        except Exception:
            pass

    def _log_to_console(
        self,
        logs: Dict[str, Any],
        generation_result: Dict[str, Any],
        loss_value: float,
        reward_mean: float,
        reward_std: float,
        raw_reward_mean: float,
        raw_reward_std: float,
        current_lr: float,
        eta_minutes: float,
        epoch_progress: float,
        term_ratio: float,
    ) -> None:
        """Format and log comprehensive metrics to console."""
        from src_new.rl.console_formatter import ConsoleFormatter

        reward_names = generation_result.get("reward_names", self.reward_func_names)

        # Collect all metrics via ConsoleMetricsCollector
        console_logs = self._console_collector.collect_all_metrics(
            global_step=self._global_step,
            epoch_progress=epoch_progress,
            loss_value=loss_value,
            reward_mean=reward_mean,
            reward_std=reward_std,
            raw_reward_mean=raw_reward_mean,
            raw_reward_std=raw_reward_std,
            current_lr=current_lr,
            grad_norm=self._last_grad_norm,
            eta_minutes=eta_minutes,
            term_ratio=term_ratio,
            logs=logs,
            generation_result=generation_result,
            reward_names=reward_names,
        )

        # Format and log multi-line summary
        formatter = ConsoleFormatter()
        detailed_summary = formatter.format_detailed_training_summary(
            console_logs, reward_names
        )
        _LOGGER.info("\n" + detailed_summary)

        self._last_console_summary = detailed_summary

    def _update_history(
        self,
        generation_result: Dict[str, Any],
        reward_mean: float,
        adv_std: float,
        adv_max: float,
        logs: Dict[str, Any],
    ) -> None:
        """Update training history tracking."""
        temperature_logged = float(generation_result.get("temperature", 0.0))
        self._temperature_history.append(temperature_logged)
        self._reward_history.append(reward_mean)
        self._adv_std_history.append(adv_std)
        self._adv_max_history.append(adv_max)
        for name in self.reward_func_names:
            comp_mean = logs.get(f"rewards/{name}/mean")
            if comp_mean is not None:
                self._reward_component_history.setdefault(name, []).append(
                    float(comp_mean)
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
        lr_cfg = self.config.optimizer.learning_rates
        lr_llm = float(lr_cfg.llm)
        lr_vision = float(lr_cfg.vision)
        lr_merger = float(lr_cfg.merger)

        param_groups = []
        if params_llm:
            param_groups.append(
                {
                    "params": params_llm,
                    "lr": lr_llm,
                    "weight_decay": self.config.optimizer.weight_decay,
                }
            )
        if params_vision:
            param_groups.append(
                {
                    "params": params_vision,
                    "lr": lr_vision,
                    "weight_decay": self.config.optimizer.weight_decay,
                }
            )
        if params_merger:
            param_groups.append(
                {
                    "params": params_merger,
                    "lr": lr_merger,
                    "weight_decay": self.config.optimizer.weight_decay,
                }
            )

        # Prefer PyTorch fused AdamW kernels when available; fallback to foreach, then standard
        # Note: fused and foreach cannot both be True (runtime error in PyTorch)

        # Build kwargs dict, only including Adam parameters if explicitly configured
        adam_kwargs = {}
        if (
            self.config.optimizer.adam_beta1 is not None
            and self.config.optimizer.adam_beta2 is not None
        ):
            adam_kwargs["betas"] = (
                self.config.optimizer.adam_beta1,
                self.config.optimizer.adam_beta2,
            )
        if self.config.optimizer.adam_epsilon is not None:
            adam_kwargs["eps"] = self.config.optimizer.adam_epsilon

        try:
            optimizer = AdamW(
                param_groups,
                lr=lr_llm,  # Global lr unused by per-group lrs; set equal to llm for logging
                weight_decay=self.config.optimizer.weight_decay,
                fused=True,
                **adam_kwargs,
            )
            _LOGGER.info("Optimizer: using fused AdamW")
        except (TypeError, RuntimeError):
            # Older PyTorch may not support fused, or fused+foreach conflict
            try:
                optimizer = AdamW(
                    param_groups,
                    lr=lr_llm,
                    weight_decay=self.config.optimizer.weight_decay,
                    foreach=True,
                    **adam_kwargs,
                )
                _LOGGER.info("Optimizer: using AdamW (foreach=True)")
            except (TypeError, RuntimeError):
                optimizer = AdamW(
                    param_groups,
                    lr=lr_llm,
                    weight_decay=self.config.optimizer.weight_decay,
                    **adam_kwargs,
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
        # Compute warmup_steps from warmup_ratio
        warmup_steps = int(
            self.config.training.warmup_ratio * self.manual_cfg.max_steps
        )
        _LOGGER.info(
            "Computed warmup_steps=%d from warmup_ratio=%.3f × max_steps=%d",
            warmup_steps,
            self.config.training.warmup_ratio,
            self.manual_cfg.max_steps,
        )

        return get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.manual_cfg.max_steps,
        )

    def _dataloader(self) -> DataLoader:
        # Let Accelerate inject its own distributed sampler during prepare()
        num_workers = self.config.training.dataloader_num_workers
        prefetch = self.config.training.prefetch_factor if num_workers > 0 else None

        return DataLoader(
            self.train_dataset,
            batch_size=self.manual_cfg.per_device_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=prefetch,
            collate_fn=lambda batch: batch,
        )

    def _next_generation_batch(
        self, iterator: Iterator[List[Dict[str, Any]]]
    ) -> tuple[List[Dict[str, Any]], Iterator[List[Dict[str, Any]]]]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self._train_dataloader)
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

    def _optimizer_step(
        self,
        optimizer: Optimizer,
        scheduler: Optional[Any],
        generation_result: Dict[str, Any],
        loss_value_for_log: Any,
    ) -> bool:
        """Perform optimizer step (called at accumulation boundary)."""
        try:
            self._last_grad_norm = float(
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.manual_cfg.max_grad_norm
                )
            )
        except Exception:
            self._last_grad_norm = 0.0

        # Register successful trajectories for telemetry
        # Extract rewards from buffer
        reward_tensor = None
        if self._training_state.generation_buffer is not None:
            rewards_list = self._training_state.generation_buffer.rewards_list
            if rewards_list:
                reward_tensor = torch.cat([r.flatten() for r in rewards_list], dim=0)

        self._prompt_batch_telemetry.register_trajectories(
            count=self._training_state.generation_buffer.get_total_completions()
            if self._training_state.generation_buffer
            else 0,
            rewards=reward_tensor,
            is_valid=True,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        self._global_step += 1

        # prompt_batch_metrics already added to generation_result in training loop
        # No need to recompute here

        self._log_step(loss_value_for_log, generation_result, current_lr)

        # Optionally dump raw generation + GT text as JSONL (rank-0 only)
        try:
            if self.accelerator.is_main_process:
                dump_enabled = os.getenv("DUMP_TEXT_SAMPLES", "0") == "1"
                if dump_enabled and self._training_state.generation_buffer is not None:
                    cadence_env = os.getenv("DUMP_TEXT_CADENCE")
                    try:
                        cadence = (
                            int(cadence_env)
                            if cadence_env is not None
                            else int(self.config.logging.logging_steps)
                        )
                    except Exception:
                        cadence = int(self.config.logging.logging_steps)
                    if cadence > 0 and (self._global_step % cadence == 0):
                        from src_new.rl.text_dump import (
                            append_generation_text_samples,  # local import
                        )

                        out_path = os.getenv("DUMP_TEXT_OUTPUT")
                        if not out_path:
                            try:
                                out_path = os.path.join(
                                    self.output_dir, "gen_text.jsonl"
                                )
                            except Exception:
                                out_path = "gen_text.jsonl"
                        append_generation_text_samples(
                            tokenizer=self.tokenizer,
                            gen_buffer=self._training_state.generation_buffer,
                            output_file=out_path,
                            step=int(self._global_step),
                        )
        except Exception as _e:
            _LOGGER.warning("Text dump skipped due to error: %s", _e)
        if (
            self._eval_enabled
            and self._eval_every_steps > 0
            and self._global_step % self._eval_every_steps == 0
        ):
            try:
                self._evaluate_and_log()
            except Exception as _e:
                _LOGGER.warning("Eval step failed: %s", _e)
        self._maybe_checkpoint(generation_result)
        self._temperature_scale = min(1.0, self._temperature_scale * 1.01)
        logprobs.clear_gpu_memory()
        return True

    def _configure_sampling_window(self) -> None:
        """Resolve per-rank sampling/accumulation configuration."""
        # Prefer env-provided world size/rank when process group may not be initialized
        try:
            _ws_env = int(os.getenv("WORLD_SIZE", "1"))
        except Exception:
            _ws_env = 1
        try:
            _rank_env = int(os.getenv("RANK", "0"))
        except Exception:
            _rank_env = 0
        if self.world_size <= 1 and _ws_env > 1:
            self.world_size = _ws_env
            self.rank = _rank_env

        raw_local_k = int(self._local_sample_k())
        if raw_local_k <= 0:
            raise ValueError(
                "Computed per-rank sample_k is 0. Ensure sampling.sample_k >= world_size and divisible by it."
            )

        self.global_sample_k = int(self.manual_cfg.sample_k)
        self.local_sample_k = raw_local_k
        self.prompts_per_cycle = int(self.manual_cfg.prompt_batch_size)
        completions_per_update = self.global_sample_k
        self.global_completions_per_update = completions_per_update
        self.local_completions_per_cycle = self.local_sample_k * self.prompts_per_cycle

        self.manual_cfg.per_device_train_batch_size = 1
        self.manual_cfg.gradient_accumulation_steps = self.local_completions_per_cycle

        if self.rank == 0:
            _LOGGER.info(
                "Sampling window | sample_k=%d | world_size=%d | per_rank_k=%d | prompts_per_cycle=%d | completions_per_cycle=%d",
                self.global_sample_k,
                self.world_size,
                self.local_sample_k,
                self.prompts_per_cycle,
                self.local_completions_per_cycle,
            )

    def _local_sample_k(self) -> int:
        """Compute per-rank sample_k under GLOBAL-K semantics."""
        k_total = int(self.manual_cfg.sample_k)
        if not (self.world_size > 1):
            return k_total
        if k_total % self.world_size != 0:
            raise ValueError(
                "sampling.sample_k must be divisible by world_size (GLOBAL-K semantics)"
            )
        return k_total // self.world_size

    def _check_cross_rank_buffer_sync(self) -> bool:
        """Check if any rank needs new buffer (cross-rank sync)."""
        need_new = self._training_state.is_buffer_exhausted()
        if self.world_size > 1:
            flag = torch.tensor(
                [1 if need_new else 0], dtype=torch.int32, device=self._device
            )
            flags = self.accelerator.gather(flag).cpu().view(-1)
            need_new = bool(int(flags.max().item()))
        return need_new

    def _check_non_finite_loss(self, loss: torch.Tensor) -> bool:
        """Check for non-finite loss across ranks."""
        is_bad = not torch.isfinite(loss)
        if self.world_size > 1:
            flag = torch.tensor(
                [1 if is_bad else 0], dtype=torch.int32, device=self._device
            )
            flags = self.accelerator.gather(flag).cpu().view(-1)
            is_bad = bool(int(flags.max().item()))
        return is_bad

    def _handle_non_finite_loss(self, optimizer: Optimizer) -> None:
        """Handle non-finite loss by clearing state and resampling."""
        _LOGGER.warning("Non-finite loss; reducing temperature and resampling")
        optimizer.zero_grad(set_to_none=True)
        self._temperature_scale = max(0.1, self._temperature_scale * 0.9)
        self._training_state.reset_buffer()
        # Clear vision cache to free GPU memory
        self._vision_cache = None
        logprobs.clear_gpu_memory()
        if self.world_size > 1:
            self.accelerator.wait_for_everyone()

    def _compute_cross_rank_advantages(self, chunks: List[Dict[str, Any]]) -> None:
        """Compute cross-rank advantage normalization for all chunks.

        Modifies chunks in-place to replace local advantages with global ones.
        This is called once after generation completes, before buffer storage.

        Args:
            chunks: List of generation dicts, each with "rewards" and "advantages"
        """
        if not self.manual_cfg.cross_rank_advantages or self.world_size <= 1:
            return

        # Clear memory before cross-rank computation
        logprobs.clear_gpu_memory()

        if not chunks and self._debug_timing:
            _LOGGER.warning(
                "Rank %d has no chunks to normalize at step %d",
                self.rank,
                self._global_step,
            )

        for single_generation in chunks:
            rewards_local = single_generation.get("rewards")
            if rewards_local is None:
                continue

            rewards_local = rewards_local.detach().to(self._device).flatten()
            rewards_local = rewards_local.contiguous()
            rewards_len = int(rewards_local.numel())

            # Gather reward lengths across ranks
            length_tensor = torch.tensor(
                [rewards_len], dtype=torch.int32, device=self._device
            )
            lengths_all = self.accelerator.gather(length_tensor)
            if lengths_all.device.type != "cpu":
                lengths_all = lengths_all.cpu()
            lengths_all = lengths_all.view(-1)
            max_len = (
                int(lengths_all.max().item())
                if lengths_all.numel() > 0
                else rewards_len
            )

            if self._debug_timing:
                _LOGGER.warning(
                    "Rank %d step %d reward lengths local=%d gathered=%s max=%d",
                    self.rank,
                    self._global_step,
                    rewards_len,
                    lengths_all.tolist() if lengths_all.numel() > 0 else [],
                    max_len,
                )

            if rewards_len != max_len and not self._warned_rewards_mismatch:
                _LOGGER.warning(
                    "Rank %d rewards length %d mismatched (max_len=%d) at step %d; padding/trimming",
                    self.rank,
                    rewards_len,
                    max_len,
                    self._global_step,
                )
                self._warned_rewards_mismatch = True

            # Pad or trim rewards to max_len
            if max_len > 0:
                if rewards_local.size(0) < max_len:
                    pad = torch.zeros(
                        max_len - rewards_local.size(0),
                        dtype=rewards_local.dtype,
                        device=self._device,
                    )
                    rewards_padded = torch.cat([rewards_local, pad], dim=0)
                else:
                    rewards_padded = rewards_local[:max_len]

                # Gather rewards across all ranks
                gathered_rewards = self.accelerator.gather(rewards_padded)
                if gathered_rewards.device.type != "cpu":
                    gathered_rewards = gathered_rewards.cpu()
                gathered_rewards = gathered_rewards.view(self.world_size, max_len)

                # Create mask for valid rewards
                lengths_cpu = lengths_all.view(self.world_size)
                arange_vec = torch.arange(max_len, device=lengths_cpu.device).unsqueeze(
                    0
                )
                mask = arange_vec < lengths_cpu.unsqueeze(1)

                # Compute global mean and std
                if mask.any():
                    masked_vals = gathered_rewards[mask]
                    global_mean = masked_vals.mean()
                    global_std = masked_vals.std(unbiased=False)

                    # Attach global scalar summaries for logging (per prompt)
                    try:
                        gstd = torch.clamp(global_std, min=1e-4)
                        if bool(self.manual_cfg.scale_rewards):
                            global_adv = (masked_vals - global_mean) / gstd
                        else:
                            global_adv = masked_vals - global_mean
                        single_generation["rewards_global/mean"] = float(
                            global_mean.item()
                        )
                        single_generation["rewards_global/std"] = float(
                            global_std.item()
                        )
                        single_generation["advantages_global/std"] = float(
                            global_adv.detach().std(unbiased=False).item()
                        )
                        single_generation["advantages_global/max_abs"] = float(
                            global_adv.detach().abs().max().item()
                        )
                    except Exception:
                        pass
                else:
                    global_mean = torch.tensor(0.0)
                    global_std = torch.tensor(1.0)

                if self._debug_timing:
                    _LOGGER.warning(
                        "Rank %d step %d global_mean=%.4f global_std=%.4f valid=%d",
                        self.rank,
                        self._global_step,
                        float(global_mean),
                        float(global_std),
                        int(mask.sum().item()),
                    )
            else:
                global_mean = torch.tensor(0.0)
                global_std = torch.tensor(1.0)

            # Compute normalized advantages
            global_std = torch.clamp(global_std, min=1e-4)
            adv_local = rewards_local.detach().cpu() - global_mean
            if self.manual_cfg.scale_rewards:
                adv_local = adv_local / global_std
            if (
                self.manual_cfg.max_advantage_magnitude is not None
                and self.manual_cfg.max_advantage_magnitude > 0
            ):
                mag = float(self.manual_cfg.max_advantage_magnitude)
                adv_local = torch.clamp(adv_local, min=-mag, max=mag)

            # Store normalized advantages back in chunk
            single_generation["advantages"] = adv_local.to(self._device)

        # Clear memory after cross-rank computation
        logprobs.clear_gpu_memory()

    def _generate_buffer(
        self,
        dataloader_iter: Iterator,
        optimizer: Optimizer,
    ) -> Optional[GenerationBuffer]:
        """Generate P prompts × K completions, store to CPU buffer for reuse.

        This implements the "generation phase" of Swift-style buffer reuse.
        All completions are generated sequentially (cross-rank broadcast),
        rewards and advantages are computed, and everything is stored on CPU
        for subsequent streaming to GPU during optimization phase.

        Returns:
            GenerationBuffer with CPU tensors, or None if generation fails (slow/non-finite)
        """
        gen_cfg = self._prepare_generation_config(self._global_step)
        local_k = self.local_sample_k
        P = self.prompts_per_cycle

        # Drop-last check at epoch boundary
        if self._epoch_size - self._sampler_pos < P:
            dropped = max(0, self._epoch_size - self._sampler_pos)
            if dropped > 0:
                self._prompt_batch_telemetry.register_dropped_prompts(dropped)
            self._sampler_pos = 0
            if self.world_size > 1:
                self.accelerator.wait_for_everyone()
            return None

        # Storage lists (all CPU)
        prompt_ids_list: List[torch.Tensor] = []
        prompt_mask_list: List[torch.Tensor] = []
        completion_ids_all: List[torch.Tensor] = []
        completion_mask_all: List[torch.Tensor] = []
        old_logps_all: List[torch.Tensor] = []
        advantages_all: List[torch.Tensor] = []
        rewards_all: List[torch.Tensor] = []
        rewards_per_func_all: List[torch.Tensor] = []  # For detailed logging
        raw_rewards_per_func_all: List[torch.Tensor] = []  # For detailed logging
        # Termination/truncation flags per prompt
        terminated_flags_all: List[torch.Tensor] = []
        truncated_flags_all: List[torch.Tensor] = []
        pixel_values_all: List[Optional[torch.Tensor]] = []
        image_grid_thw_all: List[Optional[torch.Tensor]] = []
        images_per_sample_all: List[Optional[torch.Tensor]] = []
        meta_all: List[Dict[str, Any]] = []

        slow_generation = False

        # Generate P prompts sequentially (cross-rank broadcast)
        for prompt_idx in range(P):
            # Cross-rank sampling: force all ranks to use the same dataset index
            if self.world_size > 1:
                idx_val = (
                    int(self._sampler_pos) if self.accelerator.is_main_process else 0
                )
                idx_tensor = torch.tensor(
                    [idx_val], dtype=torch.long, device=self._device
                )
                gathered_idx = self.accelerator.gather(idx_tensor)
                shared_idx = (
                    int(gathered_idx[0].item())
                    if gathered_idx.numel() > 0
                    else int(idx_tensor.item())
                )
                sample = self.train_dataset[shared_idx]
                if self.accelerator.is_main_process:
                    self._sampler_pos += 1
            else:
                if self._sampler_pos >= self._epoch_size:
                    self._sampler_pos = 0
                sample = self.train_dataset[int(self._sampler_pos)]
                self._sampler_pos += 1

            # Generate K completions for this prompt
            gen_start = time.time()
            single_gen = buffer.generate_and_score(
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
                repetition_penalty=self.config.generation.repetition_penalty,
                mask_truncated_completions=self.manual_cfg.mask_truncated_completions,
                scale_rewards=self.manual_cfg.scale_rewards,
                max_advantage_magnitude=self.manual_cfg.max_advantage_magnitude,
                reward_clip_sigma=self.config.rewards.config.clip_sigma,
            )

            gen_duration = time.time() - gen_start
            if gen_duration > self._generation_warn_threshold:
                slow_generation = True
                _LOGGER.warning(
                    "Slow generation %.2fs at step %d prompt %d/%d",
                    gen_duration,
                    self._global_step,
                    prompt_idx + 1,
                    P,
                )

            # Extract and store to CPU (critical: move to CPU immediately!)
            # single_gen already has [1, ...] for prompt_ids/mask, [K, ...] for completions
            prompt_ids_list.append(single_gen["prompt_ids"][0].cpu())
            prompt_mask_list.append(single_gen["prompt_mask"][0].cpu())
            completion_ids_all.append(
                single_gen["completion_ids"].cpu()
            )  # [K, max_comp_len]
            completion_mask_all.append(single_gen["completion_mask"].cpu())
            old_logps_all.append(
                single_gen["generation_logps"].cpu()
            )  # CRITICAL for trust region!
            advantages_all.append(single_gen["advantages"].cpu())  # [K]
            rewards_all.append(single_gen["rewards"].cpu())

            # Per-completion termination/truncation flags (flatten to [K])
            try:
                t = single_gen.get("terminated_with_eos")
                u = single_gen.get("truncated_flags")
                if t is not None:
                    terminated_flags_all.append(t.detach().cpu().long().view(-1))
                if u is not None:
                    truncated_flags_all.append(u.detach().cpu().long().view(-1))
            except Exception:
                pass

            # Per-function rewards for detailed logging
            if "rewards_per_func" in single_gen:
                rewards_per_func_all.append(single_gen["rewards_per_func"].cpu())
            if "raw_rewards_per_func" in single_gen:
                raw_rewards_per_func_all.append(
                    single_gen["raw_rewards_per_func"].cpu()
                )

            # Vision tensors
            pv = single_gen.get("pixel_values")
            pixel_values_all.append(pv.cpu() if pv is not None else None)
            thw = single_gen.get("image_grid_thw")
            image_grid_thw_all.append(thw.cpu() if thw is not None else None)
            ips = single_gen.get("images_per_sample")
            images_per_sample_all.append(ips.cpu() if ips is not None else None)

            # Metadata
            meta_list = single_gen.get("meta", [])
            meta_all.append(meta_list[0] if len(meta_list) > 0 else {})

            # Note: Completion registration and cycle reward tracking removed
            # These are now handled via GenerationBuffer state

        # Check slow generation across ranks
        if self.world_size > 1:
            slow_flag = torch.tensor(
                [1 if slow_generation else 0], dtype=torch.int32, device=self._device
            )
            flags = self.accelerator.gather(slow_flag)
            slow_generation = bool(int(flags.cpu().view(-1).max().item()))

        if slow_generation:
            _LOGGER.warning("Slow generation detected; resampling")
            optimizer.zero_grad(set_to_none=True)
            logprobs.clear_gpu_memory()
            if self.world_size > 1:
                self.accelerator.wait_for_everyone()
            return None

        # Compute cross-rank advantages ONCE for all generated chunks
        # Reconstruct chunks from stored lists for cross-rank processing
        chunks_list = []
        for p_idx in range(P):
            chunk = {
                "prompt_ids": prompt_ids_list[p_idx].unsqueeze(0),
                "prompt_mask": prompt_mask_list[p_idx].unsqueeze(0),
                "completion_ids": completion_ids_all[p_idx],
                "completion_mask": completion_mask_all[p_idx],
                "advantages": advantages_all[p_idx],
                "rewards": rewards_all[p_idx],
            }
            chunks_list.append(chunk)

        # Apply cross-rank normalization (modifies chunks in-place)
        self._compute_cross_rank_advantages(chunks_list)

        # Extract normalized advantages back
        for p_idx, chunk in enumerate(chunks_list):
            advantages_all[p_idx] = chunk["advantages"].cpu()

        # Build buffer with all data on CPU
        current_beta = self._beta_for_step(self._global_step)

        # Pad prompts to same length for stacking
        max_prompt_len = max(t.size(0) for t in prompt_ids_list)
        prompt_ids_padded = []
        prompt_mask_padded = []
        for pid, pmask in zip(prompt_ids_list, prompt_mask_list):
            if pid.size(0) < max_prompt_len:
                pad_len = max_prompt_len - pid.size(0)
                pid = torch.cat(
                    [pid, torch.full((pad_len,), self._pad_token_id, dtype=pid.dtype)]
                )
                pmask = torch.cat([pmask, torch.zeros(pad_len, dtype=pmask.dtype)])
            prompt_ids_padded.append(pid)
            prompt_mask_padded.append(pmask)

        return GenerationBuffer(
            prompt_ids=torch.stack(prompt_ids_padded, dim=0),  # [P, max_prompt_len]
            prompt_mask=torch.stack(prompt_mask_padded, dim=0),
            completion_ids_list=completion_ids_all,  # List of [K, max_comp_len]
            completion_mask_list=completion_mask_all,
            old_logps_list=old_logps_all,  # List of [K, max_comp_len] - CRITICAL!
            advantages_list=advantages_all,  # List of [K] - NOW CROSS-RANK NORMALIZED!
            rewards_list=rewards_all,
            rewards_per_func_list=rewards_per_func_all,  # For detailed logging
            raw_rewards_per_func_list=raw_rewards_per_func_all,  # For detailed logging
            terminated_flags_list=terminated_flags_all,
            truncated_flags_list=truncated_flags_all,
            pixel_values_list=pixel_values_all,
            image_grid_thw_list=image_grid_thw_all,
            images_per_sample_list=images_per_sample_all,
            meta_list=meta_all,
            temperature=gen_cfg["temperature"],
            beta=current_beta,
            steps_per_generation=self.manual_cfg.steps_per_generation,
            current_step_in_cycle=0,
        )

    def _compute_loss_for_single_completion(
        self,
        buffer_data: Dict[str, Any],
    ) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Compute loss for ONE completion (streaming mode from GenerationBuffer).

        This method is called once per completion when using buffer reuse.
        It moves tensors from CPU to GPU one at a time, computes forward pass,
        and returns the loss tensor for backward.

        Returns:
            (loss_tensor, diagnostics_dict) or (None, {}) if skipped
        """
        # Move tensors from CPU to GPU one-by-one
        prompt_ids = buffer_data["prompt_ids"].to(self._device)
        prompt_mask = buffer_data["prompt_mask"].to(self._device)
        comp_ids = buffer_data["completion_ids"].to(self._device)
        comp_mask = buffer_data["completion_mask"].to(self._device)
        old_logps = buffer_data["old_logps"].to(self._device)  # From generation policy!
        adv = buffer_data["advantages"].to(self._device)  # Scalar

        # Compute effective length
        effective_len = int(comp_mask.long().sum().item())
        if effective_len <= 0:
            return None, {}

        # Truncate to effective length
        comp_ids_eff = comp_ids[:effective_len]
        comp_mask_eff = comp_mask[:effective_len].unsqueeze(0)
        old_logps_eff = old_logps[:effective_len]
        if old_logps_eff.dim() == 1:
            old_logps_eff = old_logps_eff.unsqueeze(0)

        # Build input_ids for forward pass
        input_ids = torch.cat([prompt_ids, comp_ids_eff], dim=0).unsqueeze(0)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones_like(comp_ids_eff, dtype=prompt_mask.dtype)], dim=0
        ).unsqueeze(0)

        # Vision tensors (move to GPU)
        pv = buffer_data.get("pixel_values")
        pixel_values = pv.to(self._device) if pv is not None else None
        thw = buffer_data.get("image_grid_thw")
        image_grid_thw = thw.to(self._device) if thw is not None else None
        ips = buffer_data.get("images_per_sample")

        # Build images_per_sample tensor (single image count)
        if ips is not None:
            try:
                num_images = int(ips[0].item()) if ips.dim() > 0 else int(ips.item())
            except Exception:
                num_images = 0
            images_per_sample = torch.tensor(
                [num_images], dtype=torch.long, device=self._device
            )
        else:
            images_per_sample = None

        current_beta = buffer_data.get("beta", 0.0)
        temperature = buffer_data.get("temperature", 1.0)

        # Forward pass (current policy) - MINIMAL scope for autocast
        # CRITICAL MEMORY FIX: Aggressive cleanup of intermediate tensors
        with self.accelerator.autocast():
            per_token_logps = logprobs.get_per_token_logps(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=effective_len,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                images_per_sample=images_per_sample,
                temperature=temperature,
            )

            # CRITICAL: Free vision tensors immediately - can be 100s of MB!
            del pixel_values, image_grid_thw, images_per_sample
            logprobs.clear_gpu_memory()

            # Compute KL if using reference model
            per_token_kl = None
            if current_beta > 0.0 and self.ref_model is not None:
                # Re-create minimal vision tensors for ref model
                pv_ref = pv.to(self._device) if pv is not None else None
                thw_ref = thw.to(self._device) if thw is not None else None
                ips_ref = None
                if ips is not None:
                    try:
                        num_img = (
                            int(ips[0].item()) if ips.dim() > 0 else int(ips.item())
                        )
                    except Exception:
                        num_img = 0
                    ips_ref = torch.tensor(
                        [num_img], dtype=torch.long, device=self._device
                    )

                with torch.no_grad():
                    ref_logps = logprobs.get_per_token_logps(
                        model=self.ref_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        logits_to_keep=effective_len,
                        pixel_values=pv_ref,
                        image_grid_thw=thw_ref,
                        images_per_sample=ips_ref,
                        temperature=temperature,
                        detach=True,
                    )
                # Free ref model vision tensors immediately
                del pv_ref, thw_ref, ips_ref
                logprobs.clear_gpu_memory()

                per_token_kl = losses.compute_kl(per_token_logps, ref_logps)
                del ref_logps

            # GRPO loss (trust region via old_logps from generation!)
            loss = losses.compute_grpo_loss(
                per_token_logps,
                old_logps_eff,  # From generation policy, not current!
                adv.unsqueeze(0),  # [1]
                comp_mask_eff,
                epsilon_low=self.manual_cfg.epsilon_low,
                epsilon_high=self.manual_cfg.epsilon_high,
                loss_type=self.manual_cfg.loss_type,
                beta=current_beta,
                per_token_kl=per_token_kl,
            )

            # Free KL tensor if exists
            if per_token_kl is not None:
                del per_token_kl

        # CRITICAL: Compute diagnostics OUTSIDE autocast with detached tensors
        # This breaks the computation graph and allows aggressive cleanup
        with torch.no_grad():
            per_token_logps_detached = per_token_logps.detach()
            old_logps_detached = old_logps_eff.detach()

            coef_1 = torch.exp(per_token_logps_detached - old_logps_detached)
            adv_is_neg = (adv < 0).view(1, 1).expand_as(coef_1)
            adv_is_pos = (adv > 0).view(1, 1).expand_as(coef_1)
            is_low_clipped = (coef_1 < (1.0 - self.manual_cfg.epsilon_low)) & adv_is_neg
            is_high_clipped = (
                coef_1 > (1.0 + self.manual_cfg.epsilon_high)
            ) & adv_is_pos
            token_mask_bool = comp_mask_eff > 0

            diagnostics = {
                "clip_low_tokens": int((is_low_clipped & token_mask_bool).sum().item()),
                "clip_high_tokens": int(
                    (is_high_clipped & token_mask_bool).sum().item()
                ),
                "clip_region_tokens": int(
                    ((is_low_clipped | is_high_clipped) & token_mask_bool).sum().item()
                ),
                "clip_total_tokens": int(token_mask_bool.sum().item()),
            }

            # Free diagnostic tensors immediately
            del coef_1, adv_is_neg, adv_is_pos, is_low_clipped, is_high_clipped
            del per_token_logps_detached, old_logps_detached, token_mask_bool

        # Free ALL remaining intermediate tensors before returning
        # Keep only 'loss' for backward pass
        del per_token_logps, old_logps_eff, comp_mask_eff
        del input_ids, attention_mask, prompt_ids, prompt_mask
        del comp_ids, comp_mask, old_logps, adv

        return loss, diagnostics

    def train(self) -> None:
        base_seed = int(self.config.experiment.seed)
        rank_seed = base_seed + int(getattr(self.accelerator.state, "process_index", 0))
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        # Initialize TensorBoard writer on main process only
        if self.accelerator.is_main_process and SummaryWriter is not None:
            # tb_dir from config + run_name subfolder
            tb_base_dir = self.config.paths.tb_dir
            tb_log_dir = os.path.join(tb_base_dir, self._run_name)
            try:
                os.makedirs(tb_log_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
                # Initialize structured logger
                from src_new.rl.tensorboard_logger import TensorBoardLogger

                self._tb_logger = TensorBoardLogger(self._tb_writer)
                _LOGGER.info("TensorBoard logging to: %s", tb_log_dir)
            except Exception as e:
                _LOGGER.warning("Failed to initialize TensorBoard writer: %s", e)
                self._tb_writer = None
                self._tb_logger = None

        dataloader = self._dataloader()

        optimizer = self._build_optimizer()
        # Prepare model, optimizer, and dataloader with Accelerator
        self.model, optimizer, dataloader = self.accelerator.prepare(
            self.model, optimizer, dataloader
        )

        # Keep prepared dataloader for iterator resets
        self._train_dataloader = dataloader
        scheduler = self._build_scheduler(optimizer)
        optimizer.zero_grad(set_to_none=True)

        # Move reference model to device (no DDP wrap)
        if self.ref_model is not None:
            try:
                self.ref_model.to(self._device).eval()
            except Exception:
                pass

        dataloader_iter = iter(self._train_dataloader)

        self.model.train()

        # Main training loop with buffer reuse
        # Main training loop with buffer reuse
        while self._training_state.global_step < self.manual_cfg.max_steps:
            # === GENERATION PHASE ===
            if self._training_state.is_buffer_exhausted():
                # Check cross-rank buffer sync
                need_new = self._check_cross_rank_buffer_sync()
                if need_new:
                    # Generate new buffer
                    gen_buffer = self._generate_buffer(dataloader_iter, optimizer)
                    if gen_buffer is None:
                        continue  # Failed generation, resample
                    self._training_state.generation_buffer = gen_buffer
                    self._training_state.buffer_step_idx = 0

            # === OPTIMIZATION PHASE ===
            # Get next completion from buffer
            buffer_data = (
                self._training_state.generation_buffer.get_completion_for_step(
                    self._training_state.buffer_step_idx
                )
            )

            # Compute loss for single completion
            loss, diagnostics = self._compute_loss_for_single_completion(buffer_data)

            if loss is None:
                # Skip zero-length completion
                self._training_state.generation_buffer.increment_step()
                self._training_state.buffer_step_idx += 1
                continue

            # Check for non-finite loss (cross-rank sync)
            if self._check_non_finite_loss(loss):
                self._handle_non_finite_loss(optimizer)
                continue

            # Save loss value for logging BEFORE backward
            loss_value_for_log = float(loss.detach().item())

            # CRITICAL: Use no_sync() for all intermediate accumulation steps
            # Only sync gradients on the final step (when accelerator.sync_gradients is True)
            # This prevents expensive all-reduce operations and reduces memory pressure
            if self.accelerator.sync_gradients:
                # Final accumulation step - allow gradient sync
                self.accelerator.backward(loss)
            else:
                # Intermediate accumulation step - no sync
                with self.accelerator.no_sync(self.model):
                    self.accelerator.backward(loss)

            # CRITICAL: Clear intermediate activations immediately after backward
            # to prevent OOM when processing multiple completions sequentially
            del loss
            logprobs.clear_gpu_memory()  # Clear any cached activations
            torch.cuda.empty_cache()

            # Accumulate diagnostics
            self._training_state.generation_buffer.accumulate_clip_diagnostics(
                **diagnostics
            )

            # Free diagnostics dict
            del diagnostics

            # Increment buffer position
            self._training_state.generation_buffer.increment_step()
            self._training_state.buffer_step_idx += 1

            # === OPTIMIZER STEP ===
            if self.accelerator.sync_gradients:
                # Reconstruct full generation result from buffer for logging
                # The buffer contains rewards_list (P prompts × K completions)
                # We need to flatten this for proper logging
                gen_buffer = self._training_state.generation_buffer

                # Flatten rewards from all prompts in buffer
                all_rewards = torch.cat(
                    [r.to(self._device) for r in gen_buffer.rewards_list], dim=0
                )
                all_advantages = torch.cat(
                    [a.to(self._device) for a in gen_buffer.advantages_list], dim=0
                )

                # Flatten per-function rewards for detailed logging
                all_rewards_per_func = None
                all_raw_rewards_per_func = None
                if gen_buffer.rewards_per_func_list:
                    all_rewards_per_func = torch.cat(
                        [r.to(self._device) for r in gen_buffer.rewards_per_func_list],
                        dim=0,
                    )
                if gen_buffer.raw_rewards_per_func_list:
                    all_raw_rewards_per_func = torch.cat(
                        [
                            r.to(self._device)
                            for r in gen_buffer.raw_rewards_per_func_list
                        ],
                        dim=0,
                    )

                # Build comprehensive generation_result for logging
                generation_result_for_log = {
                    "rewards": all_rewards,
                    "advantages": all_advantages,
                    "temperature": gen_buffer.temperature,
                    "beta": gen_buffer.beta,
                    "reward_names": self.reward_func_names,
                }

                # Add per-function rewards if available
                if all_rewards_per_func is not None:
                    generation_result_for_log["rewards_per_func"] = all_rewards_per_func
                if all_raw_rewards_per_func is not None:
                    generation_result_for_log["raw_rewards_per_func"] = (
                        all_raw_rewards_per_func
                    )

                # Add termination/truncation flags if present in buffer
                try:
                    if (
                        gen_buffer.terminated_flags_list
                        and gen_buffer.truncated_flags_list
                    ):
                        term_flat = torch.cat(
                            [
                                t.to(self._device).view(-1)
                                for t in gen_buffer.terminated_flags_list
                            ],
                            dim=0,
                        )
                        trunc_flat = torch.cat(
                            [
                                u.to(self._device).view(-1)
                                for u in gen_buffer.truncated_flags_list
                            ],
                            dim=0,
                        )
                        generation_result_for_log["terminated_with_eos"] = term_flat
                        generation_result_for_log["truncated_flags"] = trunc_flat
                except Exception:
                    pass

                # Add clipping diagnostics
                clip_diag = gen_buffer.get_clip_diagnostics()
                generation_result_for_log["policy_clip_low_ratio"] = clip_diag[
                    "clip_low_ratio"
                ]
                generation_result_for_log["policy_clip_high_ratio"] = clip_diag[
                    "clip_high_ratio"
                ]
                generation_result_for_log["policy_clip_region_ratio"] = clip_diag[
                    "clip_region_ratio"
                ]

                # Buffer reuse logging metadata to gate full logging to the first reuse step
                try:
                    is_fresh = bool(
                        self._training_state.buffer_step_idx
                        == self.local_completions_per_cycle
                    )
                except Exception:
                    is_fresh = False
                generation_result_for_log["buffer/is_fresh"] = is_fresh
                generation_result_for_log["buffer/step_in_cycle"] = int(
                    self._training_state.buffer_step_idx
                )
                generation_result_for_log["buffer/completions_per_cycle"] = int(
                    self.local_completions_per_cycle
                )
                try:
                    generation_result_for_log["buffer/completions_total"] = int(
                        gen_buffer.get_total_completions()
                    )
                except Exception:
                    pass

                # Add prompt batch telemetry
                prompt_batch_metrics = (
                    self._prompt_batch_telemetry.compute_cycle_metrics()
                )
                generation_result_for_log["prompt_batch_metrics"] = prompt_batch_metrics

                # Perform optimizer step
                stepped = self._optimizer_step(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generation_result=generation_result_for_log,
                    loss_value_for_log=loss_value_for_log,
                )
                if not stepped:
                    continue

        _LOGGER.info("Manual GRPO training finished at step %d", self._global_step)

        # Final safeguard: ensure a checkpoint exists with all auxiliary JSONs
        try:
            if self.accelerator.is_main_process:
                model_to_save = self.accelerator.unwrap_model(self.model)
                self._checkpoint_saver.save_checkpoint(
                    model=model_to_save,
                    processing_class=self.tokenizer,
                    processor=self.processor,
                    step=self._global_step,
                    current_metrics={"reward": 0.0},
                    is_eval_step=False,
                    force_step_save=True,
                    is_deepspeed_enabled=False,
                    training_start_time=self._start_time,
                )
        except Exception as _e:
            _LOGGER.warning("Final checkpoint save skipped due to error: %s", _e)

    def _maybe_checkpoint(self, generation_result: Dict[str, Any]) -> None:
        if self.manual_cfg.save_steps <= 0:
            return
        # Save at exact cadence boundaries OR at the terminal step
        is_cadence_step = (self._global_step % self.manual_cfg.save_steps) == 0
        is_final_step = self._global_step == self.manual_cfg.max_steps
        if not (is_cadence_step or is_final_step):
            return

        # Save only on the main process without cross-rank barriers to avoid hangs
        if not self.accelerator.is_main_process:
            return

        # Emit a clear log so users can see when a save is triggered
        try:
            _LOGGER.info(
                "Checkpoint trigger: step=%d | cadence=%s | final=%s",
                self._global_step,
                str(is_cadence_step),
                str(is_final_step),
            )
        except Exception:
            pass

        rewards = generation_result.get("rewards")
        mean_reward = 0.0
        if rewards is not None:
            try:
                rewards_for_ckpt = rewards.detach()
                mean_reward = float(rewards_for_ckpt.mean().item())
            except Exception:
                mean_reward = 0.0
        metrics = {"reward": mean_reward}
        try:
            # Unwrap Accelerate/DP wrapper before saving
            model_to_save = self.accelerator.unwrap_model(self.model)
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

    def _evaluate_and_log(self) -> None:
        """Run a tiny eval: exactly one generation per rank per round.

        Aggregates metrics on rank-0 and logs to TensorBoard.
        """
        if not self._eval_enabled:
            return
        if self._eval_per_rank_samples != 1:
            # Enforce single-sample per rank
            _LOGGER.warning("Forcing per_rank_samples=1 for eval")
        # No heavy imports needed; use shared metric fn

        device = self._device
        was_training = self.model.training
        self.model.eval()
        torch.manual_seed(self._eval_seed + self._global_step)

        # Containers for per-rank metrics
        local_metrics: Dict[str, float] = {}
        saved_samples: list[dict] = []

        rounds = max(1, int(self._eval_rounds))
        val_len = len(self.val_dataset) if hasattr(self, "val_dataset") else 0
        if val_len <= 0:
            _LOGGER.warning("No validation dataset; skipping eval")
            if was_training:
                self.model.train()
            return

        # Helper to log one sample
        def _eval_one(idx: int) -> None:
            try:
                sample = self.val_dataset[idx]
            except Exception:
                return
            # Inputs from dataset (already processed to tensors by RL dataset)
            ids = sample.get("input_ids")
            mask = sample.get("attention_mask")
            pv = sample.get("pixel_values")
            thw = sample.get("image_grid_thw")
            meta = sample.get("meta")
            if ids is None or mask is None:
                return
            # Build single-row batch without padding to keep memory small
            batch = {
                "input_ids": ids.unsqueeze(0) if ids.dim() == 1 else ids[:1],
                "attention_mask": mask.unsqueeze(0) if mask.dim() == 1 else mask[:1],
            }
            if pv is not None:
                batch["pixel_values"] = pv
            if thw is not None:
                batch["image_grid_thw"] = thw

            # Generation kwargs align with training
            gen_cfg = self._prepare_generation_config(self._global_step)
            try:
                seq = buffer.generation.generate_completions(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch=batch,
                    generation_config=None,
                    eos_token_id=None,
                    max_new_tokens=int(self.manual_cfg.max_new_tokens),
                    temperature=float(gen_cfg["temperature"]),
                    top_p=float(self.manual_cfg.top_p),
                )
            except Exception:
                return
            if seq.dim() == 2:
                out_ids = seq[0]
            else:
                out_ids = seq.view(-1)
            prompt_len = int(batch["input_ids"].shape[-1])
            comp_ids = out_ids[prompt_len:]
            text = self.tokenizer.decode(comp_ids, skip_special_tokens=False)
            # Trim at <|im_end|>
            end_tok = "<|im_end|>"
            cut = text.find(end_tok)
            if cut != -1:
                text = text[:cut]

            # Compute absolute/objective metrics via shared helper
            try:
                abs_metrics = _abs_metrics(text, meta if isinstance(meta, dict) else {})
            except Exception:
                abs_metrics = {}

            # Accumulate
            for k, v in abs_metrics.items():
                local_metrics.setdefault(k, 0.0)
                try:
                    local_metrics[k] += float(v)
                except Exception:
                    pass

            if len(saved_samples) < 1 and self._eval_save_samples > 0:
                saved_samples.append(
                    {
                        "image": sample.get("image")
                        or (
                            sample.get("images", [None])[0]
                            if isinstance(sample.get("images"), list)
                            else None
                        ),
                        "prediction_text_raw": text,
                        "prediction": abs_metrics.get("prediction", []),
                        "ground_truth": meta.get("objects", [])
                        if isinstance(meta, dict)
                        else [],
                        # width/height may be present in meta; include if available
                        "width": meta.get("width") if isinstance(meta, dict) else None,
                        "height": meta.get("height")
                        if isinstance(meta, dict)
                        else None,
                        "metrics": abs_metrics,
                    }
                )

        # Run rounds (one sample per rank each round)
        for r in range(rounds):
            # Pick index deterministically across ranks
            base = (self._global_step * rounds + r) % val_len
            idx = (base + self.rank) % val_len
            _eval_one(int(idx))

        # Build tensors for aggregation
        keys = list(local_metrics.keys())
        vals = [float(local_metrics.get(k, 0.0) / float(rounds)) for k in keys]
        metrics_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
        gathered = self.accelerator.gather(metrics_tensor)
        if gathered.device.type != "cpu":
            gathered = gathered.cpu()
        # Rank-0 reduce and log
        if self.accelerator.is_main_process and gathered.numel() > 0:
            g = gathered.view(-1, len(keys))
            means = g.mean(dim=0)
            try:
                if self._tb_writer is not None:
                    for k, v in zip(keys, means.tolist()):
                        self._tb_writer.add_scalar(
                            f"eval/{k}", float(v), self._global_step
                        )
            except Exception:
                pass
            # Save samples via shared dumper (main process only)
            if (
                self._eval_save_samples > 0
                and saved_samples
                and self.accelerator.is_main_process
            ):
                dump_samples(
                    saved_samples,
                    output_file=None,
                    output_dir=self.output_dir,
                    step=self._global_step,
                    limit=self._eval_save_samples,
                )

        # Restore mode
        if was_training:
            self.model.train()

    @staticmethod
    def _build_manual_cfg(
        rl_config: RLConfig, train_dataset: Any
    ) -> ManualTrainerConfig:
        """Build manual config from strict v2 config - NO defaults.

        Computes max_steps from num_train_epochs and dataset_size.
        """

        # All values from typed config
        sample_k = rl_config.sampling.sample_k
        prompt_batch_size = rl_config.sampling.prompt_batch_size
        trajectories_per_cycle = (
            prompt_batch_size * sample_k
        )  # Will be adjusted by world_size

        grad_accum = rl_config.training.gradient_accumulation_steps

        # Compute max_steps from epochs and dataset size
        dataset_size = rl_config.training.dataset_size
        if dataset_size == -1:
            dataset_size = len(train_dataset)  # Auto-detect

        # Steps per epoch = dataset_size // prompt_batch_size
        steps_per_epoch = max(1, dataset_size // prompt_batch_size)
        max_steps = rl_config.training.num_train_epochs * steps_per_epoch

        _LOGGER.info(
            "Computed max_steps=%d from num_train_epochs=%d × steps_per_epoch=%d (dataset_size=%d, prompt_batch_size=%d)",
            max_steps,
            rl_config.training.num_train_epochs,
            steps_per_epoch,
            dataset_size,
            prompt_batch_size,
        )

        # Beta annealing schedule (computed from ratio after max_steps)
        per_device_train_batch = rl_config.training.per_device_train_batch_size
        standardize_rewards = rl_config.grpo.standardize_rewards

        beta_start = rl_config.grpo.beta_start
        beta_schedule = None
        if beta_start > 0.0 and rl_config.grpo.beta_anneal is not None:
            # Compute anneal steps from ratio
            anneal_steps = int(rl_config.grpo.beta_anneal.ratio * max_steps)
            beta_schedule = {
                "type": rl_config.grpo.beta_anneal.type,
                "steps": anneal_steps,  # Computed from ratio
            }
            _LOGGER.info(
                "Computed beta_anneal_steps=%d from ratio=%.3f × max_steps=%d",
                anneal_steps,
                rl_config.grpo.beta_anneal.ratio,
                max_steps,
            )

        # Steps per generation - required field
        steps_per_gen = rl_config.grpo.steps_per_generation
        if steps_per_gen > 1:
            _LOGGER.warning(
                "steps_per_generation=%d > 1: Buffer reuse is EXPERIMENTAL. "
                "Core infrastructure is ready but full integration pending. Use with caution.",
                steps_per_gen,
            )

        return ManualTrainerConfig(
            sample_k=sample_k,
            prompt_batch_size=prompt_batch_size,
            trajectories_per_cycle=trajectories_per_cycle,
            reward_average_window=rl_config.sampling.reward_average_window,
            max_new_tokens=rl_config.generation.max_new_tokens,
            min_new_tokens=rl_config.generation.min_new_tokens,
            temperature=rl_config.generation.temperature,
            temperature_schedule="constant",
            top_p=rl_config.generation.top_p,
            epsilon_low=rl_config.grpo.epsilon_low,
            epsilon_high=rl_config.grpo.epsilon_high,
            beta=beta_start,
            loss_type=rl_config.grpo.loss_type,
            scale_rewards=rl_config.grpo.scale_rewards,
            mask_truncated_completions=rl_config.grpo.mask_truncated_completions,
            max_advantage_magnitude=rl_config.grpo.max_advantage_magnitude,
            gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=per_device_train_batch,
            steps_per_generation=steps_per_gen,
            logging_steps=rl_config.logging.logging_steps,
            save_steps=rl_config.checkpointing.save_steps,
            max_steps=max_steps,
            bf16=rl_config.training.bf16,
            standardize_rewards=standardize_rewards,
            cross_rank_advantages=rl_config.normalization.cross_rank_advantages,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            max_grad_norm=rl_config.optimizer.max_grad_norm,
        )

    @property
    def temperature_history(self) -> List[float]:
        return list(self._temperature_history)

    @property
    def reward_history(self) -> List[float]:
        return list(self._reward_history)

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
        if self._tb_logger is not None:
            try:
                self._tb_logger.close()
                _LOGGER.info("TensorBoard logger closed")
            except Exception as e:
                _LOGGER.warning(f"Failed to close TensorBoard logger: {e}")


__all__ = ["BBUGRPOTrainer"]
