"""
RL Configuration v2 - Strict validation, no defaults for hyperparameters.

This module provides comprehensive configuration management with:
- NO defaults for any tunable hyperparameters
- Only Optional[T] = None for truly optional fields
- Custom validator with clear error messages
- Backward compatibility detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ConfigValidationError(ValueError):
    """Raised when required config key is missing."""

    pass


def _require(cfg: Dict[str, Any], key: str, context: str = "") -> Any:
    """Require a key to be present, raise clear error if missing."""
    if key not in cfg:
        ctx = f" in {context}" if context else ""
        raise ConfigValidationError(
            f"Missing required config key: '{key}'{ctx}\n"
            f"This value must be explicitly set in your YAML config."
        )
    return cfg[key]


@dataclass(frozen=True)
class PathsConfig:
    """All file paths - all required except ref_model_path."""

    model_path: str
    train_data_path: str
    val_data_path: str
    data_root: str
    output_dir: str
    tb_dir: str
    ref_model_path: Optional[str] = None  # Only optional field (must be last)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "PathsConfig":
        return PathsConfig(
            model_path=_require(cfg, "model_path", "paths"),
            train_data_path=_require(cfg, "train_data_path", "paths"),
            val_data_path=_require(cfg, "val_data_path", "paths"),
            data_root=_require(cfg, "data_root", "paths"),
            output_dir=_require(cfg, "output_dir", "paths"),
            tb_dir=_require(cfg, "tb_dir", "paths"),
            ref_model_path=cfg.get("ref_model_path"),  # Only optional
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata - all required."""

    run_name: str
    seed: int
    tags: List[str] = field(default_factory=list)  # Only structural default

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(
            run_name=_require(cfg, "run_name", "experiment"),
            seed=_require(cfg, "seed", "experiment"),
            tags=cfg.get("tags", []),  # Structural default OK
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration - all required except immutable from base."""

    attn_implementation: str
    torch_dtype: str
    use_cache: bool
    trust_remote_code: bool
    image_max_pixels: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(
            attn_implementation=_require(cfg, "attn_implementation", "model"),
            torch_dtype=_require(cfg, "torch_dtype", "model"),
            use_cache=_require(cfg, "use_cache", "model"),
            trust_remote_code=_require(cfg, "trust_remote_code", "model"),
            image_max_pixels=_require(cfg, "image_max_pixels", "model"),
        )


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling configuration - all required."""

    prompt_batch_size: int
    sample_k: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "SamplingConfig":
        # Migration guard: forbid legacy flag
        if "sample_k_per_rank" in cfg:
            raise ConfigValidationError(
                "'sampling.sample_k_per_rank' has been removed.\n"
                "Define 'sampling.sample_k' as a GLOBAL count (total across ranks).\n"
                "The trainer requires sample_k % world_size == 0 and uses local_k = sample_k / world_size."
            )

        return SamplingConfig(
            prompt_batch_size=_require(cfg, "prompt_batch_size", "sampling"),
            sample_k=_require(cfg, "sample_k", "sampling"),
        )


# DynamicLengthConfig removed — dynamic per-sample caps no longer supported


@dataclass(frozen=True)
class GenerationConfig:
    """Generation config - all required."""

    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "GenerationConfig":
        return GenerationConfig(
            max_new_tokens=_require(cfg, "max_new_tokens", "generation"),
            min_new_tokens=_require(cfg, "min_new_tokens", "generation"),
            temperature=_require(cfg, "temperature", "generation"),
            top_p=_require(cfg, "top_p", "generation"),
            repetition_penalty=_require(cfg, "repetition_penalty", "generation"),
        )


@dataclass(frozen=True)
class BetaAnnealConfig:
    """Beta annealing - epoch-based with ratio."""

    type: str
    ratio: float  # Fraction of training for annealing (e.g., 0.67 for 2/3)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "BetaAnnealConfig":
        # Detect legacy 'steps' usage
        if "steps" in cfg:
            raise ConfigValidationError(
                "Legacy field 'steps' detected in beta_anneal.\n"
                "Beta annealing is now epoch-based. Please use:\n"
                "  - ratio: <float>  # e.g., 0.67 for 2/3 of training\n"
                "See configs/dense_rl/standard.yaml for example."
            )

        anneal_type = _require(cfg, "type", "beta_anneal")
        if anneal_type not in {"linear", "cosine"}:
            raise ConfigValidationError(
                f"beta_anneal.type must be 'linear' or 'cosine', got: {anneal_type}"
            )

        ratio = _require(cfg, "ratio", "beta_anneal")
        if ratio <= 0.0 or ratio > 1.0:
            raise ConfigValidationError(
                f"beta_anneal.ratio must be in (0.0, 1.0], got {ratio}"
            )

        return BetaAnnealConfig(type=anneal_type, ratio=ratio)


@dataclass(frozen=True)
class GRPOConfig:
    """GRPO algorithm - TRL-aligned. No custom standardizer or advantage clipping."""

    epsilon_low: float
    epsilon_high: float
    loss_type: str
    scale_rewards: bool
    mask_truncated_completions: bool
    steps_per_generation: int  # Buffer reuse interval
    beta_start: float = 0.0  # deprecated; ignored by TRL path
    beta_anneal: Optional[BetaAnnealConfig] = None  # tolerated when present

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "GRPOConfig":
        beta_anneal_dict = cfg.get("beta_anneal")
        steps_per_gen = _require(cfg, "steps_per_generation", "grpo")

        # Validate steps_per_generation
        steps_per_gen = int(steps_per_gen)
        if steps_per_gen < 1:
            raise ConfigValidationError(
                f"grpo.steps_per_generation must be >= 1, got {steps_per_gen}"
            )

        return GRPOConfig(
            epsilon_low=_require(cfg, "epsilon_low", "grpo"),
            epsilon_high=_require(cfg, "epsilon_high", "grpo"),
            loss_type=_require(cfg, "loss_type", "grpo"),
            scale_rewards=_require(cfg, "scale_rewards", "grpo"),
            mask_truncated_completions=_require(
                cfg, "mask_truncated_completions", "grpo"
            ),
            steps_per_generation=steps_per_gen,
            beta_anneal=BetaAnnealConfig.from_dict(beta_anneal_dict)
            if beta_anneal_dict
            else None,
            beta_start=float(cfg.get("beta_start", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization config - retained for backward compat; no cross-rank advs."""

    cross_rank_advantages: bool

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "NormalizationConfig":
        # Always false: TRL normalizes per-prompt group locally
        return NormalizationConfig(cross_rank_advantages=False)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration - epoch-based (no max_steps)."""

    num_train_epochs: int
    dataset_size: int  # -1 for auto-detect from len(train_dataset)
    warmup_ratio: float
    lr_scheduler_type: str
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    bf16: bool
    fp16: bool
    gradient_accumulation_steps: int
    per_device_train_batch_size: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "TrainingConfig":
        # Detect legacy max_steps/warmup_steps usage
        if "max_steps" in cfg:
            raise ConfigValidationError(
                "Legacy field 'max_steps' detected in training config.\n"
                "RL training is now epoch-based. Please use:\n"
                "  - num_train_epochs: <int>\n"
                "  - dataset_size: -1  # or explicit size\n"
                "  - warmup_ratio: <float>  # instead of warmup_steps\n"
                "See configs/dense_rl/standard.yaml for example."
            )
        if "warmup_steps" in cfg:
            raise ConfigValidationError(
                "Legacy field 'warmup_steps' detected in training config.\n"
                "Please use 'warmup_ratio' instead (e.g., 0.1 for 10% warmup)."
            )

        num_epochs = _require(cfg, "num_train_epochs", "training")
        if num_epochs <= 0:
            raise ConfigValidationError(
                f"num_train_epochs must be > 0, got {num_epochs}"
            )

        dataset_size = _require(cfg, "dataset_size", "training")
        if dataset_size < -1:
            raise ConfigValidationError(
                f"dataset_size must be >= -1, got {dataset_size}"
            )

        warmup = _require(cfg, "warmup_ratio", "training")
        if warmup < 0.0:
            raise ConfigValidationError(f"warmup_ratio must be >= 0.0, got {warmup}")

        grad_accum = _require(cfg, "gradient_accumulation_steps", "training")
        if grad_accum < 1:
            raise ConfigValidationError(
                f"gradient_accumulation_steps must be >= 1, got {grad_accum}"
            )

        per_device_batch = _require(cfg, "per_device_train_batch_size", "training")
        if per_device_batch < 1:
            raise ConfigValidationError(
                f"per_device_train_batch_size must be >= 1, got {per_device_batch}"
            )

        return TrainingConfig(
            num_train_epochs=num_epochs,
            dataset_size=dataset_size,
            warmup_ratio=warmup,
            lr_scheduler_type=_require(cfg, "lr_scheduler_type", "training"),
            dataloader_num_workers=_require(cfg, "dataloader_num_workers", "training"),
            pin_memory=_require(cfg, "pin_memory", "training"),
            prefetch_factor=_require(cfg, "prefetch_factor", "training"),
            bf16=_require(cfg, "bf16", "training"),
            fp16=_require(cfg, "fp16", "training"),
            gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=per_device_batch,
        )


@dataclass(frozen=True)
class LearningRatesConfig:
    """Learning rates - all required."""

    llm: float
    vision: Optional[float] = None
    merger: Optional[float] = None

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LearningRatesConfig":
        return LearningRatesConfig(
            llm=_require(cfg, "llm", "learning_rates"),
            vision=cfg.get("vision"),
            merger=cfg.get("merger"),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration - all required."""

    type: str
    learning_rates: LearningRatesConfig
    weight_decay: float
    max_grad_norm: float
    adam_beta1: Optional[float] = None  # Optional: use AdamW default (0.9)
    adam_beta2: Optional[float] = None  # Optional: use AdamW default (0.999)
    adam_epsilon: Optional[float] = None  # Optional: use AdamW default (1e-8)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "OptimizerConfig":
        return OptimizerConfig(
            type=_require(cfg, "type", "optimizer"),
            learning_rates=LearningRatesConfig.from_dict(
                _require(cfg, "learning_rates", "optimizer")
            ),
            weight_decay=_require(cfg, "weight_decay", "optimizer"),
            max_grad_norm=_require(cfg, "max_grad_norm", "optimizer"),
            adam_beta1=cfg.get("adam_beta1"),  # Optional
            adam_beta2=cfg.get("adam_beta2"),  # Optional
            adam_epsilon=cfg.get("adam_epsilon"),  # Optional
        )


@dataclass(frozen=True)
class VisionTowerConfig:
    """Vision tower freezing config - all required."""

    freeze_patch_embed: bool
    freeze_bottom_layers: bool
    trainable_top_k_blocks: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "VisionTowerConfig":
        return VisionTowerConfig(
            freeze_patch_embed=_require(cfg, "freeze_patch_embed", "vision_tower"),
            freeze_bottom_layers=_require(cfg, "freeze_bottom_layers", "vision_tower"),
            trainable_top_k_blocks=_require(
                cfg, "trainable_top_k_blocks", "vision_tower"
            ),
        )


@dataclass(frozen=True)
class LLMConfig:
    """LLM freezing config - all required."""

    freeze_bottom_layers: bool
    trainable_top_k_blocks: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LLMConfig":
        return LLMConfig(
            freeze_bottom_layers=_require(cfg, "freeze_bottom_layers", "llm"),
            trainable_top_k_blocks=_require(cfg, "trainable_top_k_blocks", "llm"),
        )


@dataclass(frozen=True)
class MergerConfig:
    """Merger freezing config - all required."""

    freeze: bool

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "MergerConfig":
        return MergerConfig(
            freeze=_require(cfg, "freeze", "merger"),
        )


@dataclass(frozen=True)
class LayerFreezingConfig:
    """Layer freezing configuration - all required."""

    vision_tower: VisionTowerConfig
    llm: LLMConfig
    merger: MergerConfig

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LayerFreezingConfig":
        return LayerFreezingConfig(
            vision_tower=VisionTowerConfig.from_dict(
                _require(cfg, "vision_tower", "layer_config")
            ),
            llm=LLMConfig.from_dict(_require(cfg, "llm", "layer_config")),
            merger=MergerConfig.from_dict(_require(cfg, "merger", "layer_config")),
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration - all required."""

    logging_steps: int
    log_level: str

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LoggingConfig":
        return LoggingConfig(
            logging_steps=_require(cfg, "logging_steps", "logging"),
            log_level=_require(cfg, "log_level", "logging"),
        )


@dataclass(frozen=True)
class CheckpointingConfig:
    """Checkpointing configuration - all required."""

    save_steps: int
    save_total_limit: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "CheckpointingConfig":
        return CheckpointingConfig(
            save_steps=_require(cfg, "save_steps", "checkpointing"),
            save_total_limit=_require(cfg, "save_total_limit", "checkpointing"),
        )


@dataclass(frozen=True)
class LengthVsGTConfig:
    """Length vs GT reward config - minimal Gaussian parameters."""

    use_ratio: bool
    sigma_ratio: float
    sigma_tokens: int
    min_reward: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LengthVsGTConfig":
        # Defaults consistent with plan; enforce presence via _require for explicit config
        return LengthVsGTConfig(
            use_ratio=_require(cfg, "use_ratio", "length_vs_gt"),
            sigma_ratio=_require(cfg, "sigma_ratio", "length_vs_gt"),
            sigma_tokens=_require(cfg, "sigma_tokens", "length_vs_gt"),
            min_reward=_require(cfg, "min_reward", "length_vs_gt"),
        )


@dataclass(frozen=True)
class StandardizerConfig:
    """Per-component standardizer hyperparameters (optional block)."""

    mode: str  # "std_only" or "zscore"
    momentum: float
    warmup_steps: int
    min_std: float
    clip: float

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "StandardizerConfig":
        if cfg is None:
            raise ConfigValidationError(
                "Missing required 'rewards_config.standardizer' block while reward standardization is enabled"
            )
        mode = _require(cfg, "mode", "rewards_config.standardizer")
        if mode not in {"std_only", "zscore"}:
            raise ConfigValidationError(
                "rewards_config.standardizer.mode must be 'std_only' or 'zscore'"
            )
        return StandardizerConfig(
            mode=mode,
            momentum=float(_require(cfg, "momentum", "rewards_config.standardizer")),
            warmup_steps=int(
                _require(cfg, "warmup_steps", "rewards_config.standardizer")
            ),
            min_std=float(_require(cfg, "min_std", "rewards_config.standardizer")),
            clip=float(_require(cfg, "clip", "rewards_config.standardizer")),
        )


@dataclass(frozen=True)
class RewardParamsConfig:
    """Reward-specific hyperparameters - all required."""

    clip_sigma: float
    length_vs_gt: LengthVsGTConfig
    # Optional per-reward params
    # New: line-specific duplicate and pattern penalties (flattened 'line' sub-blocks)
    duplicate_penalty: Optional[Dict[str, Any]] = None
    pattern_penalty: Optional[Dict[str, Any]] = None
    # Assignment-specific params (lambda_caption, alpha, beta_fp, beta_fn, line_buffer_frac)
    assignment_f1: Optional[Dict[str, Any]] = None
    # Optional: standardizer hyperparameters
    standardizer: Optional["StandardizerConfig"] = None

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RewardParamsConfig":
        def _extract_duplicate_penalty(
            root: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            dp = root.get("duplicate_penalty")
            if dp is None:
                return None
            if not isinstance(dp, dict):
                raise ConfigValidationError(
                    "rewards_config.duplicate_penalty must be a mapping when provided"
                )
            line = (
                dp.get("line")
                if isinstance(dp.get("apply_to"), list) and "line" in dp.get("apply_to")
                else dp.get("line")
            )
            if line is not None and not isinstance(line, dict):
                raise ConfigValidationError(
                    "rewards_config.duplicate_penalty.line must be a mapping when provided"
                )
            # Validate known fields when present
            if line is not None:
                for k in ("min_vertex_separation",):
                    if k in line:
                        _ = int(line[k])
                for k in (
                    "zero_length_seg_penalty",
                    "per_duplicate_vertex",
                    "max_penalty",
                ):
                    if k in line:
                        _ = float(line[k])
            return {"line": line} if line is not None else {}

        def _extract_pattern_penalty(root: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            pp = root.get("pattern_penalty")
            if pp is None:
                return None
            if not isinstance(pp, dict):
                raise ConfigValidationError(
                    "rewards_config.pattern_penalty must be a mapping when provided"
                )
            line = pp.get("line")
            if line is not None and not isinstance(line, dict):
                raise ConfigValidationError(
                    "rewards_config.pattern_penalty.line must be a mapping when provided"
                )
            if line is not None:
                for k in (
                    "axis_run_max_ratio",
                    "step_repeat_max_ratio",
                    "per_overshoot",
                    "max_penalty",
                ):
                    if k in line:
                        _ = float(line[k])
            return {"line": line} if line is not None else {}

        # Optional assignment_f1 block
        assignment_cfg = cfg.get("assignment_f1")
        if assignment_cfg is not None and not isinstance(assignment_cfg, dict):
            raise ConfigValidationError(
                "rewards_config.assignment_f1 must be a mapping when provided"
            )
        if isinstance(assignment_cfg, dict):
            for k in (
                "lambda_caption",
                "alpha",
                "beta_fp",
                "beta_fn",
                "line_buffer_frac",
            ):
                if k in assignment_cfg:
                    _ = float(assignment_cfg[k])

        std_cfg = cfg.get("standardizer")
        std_parsed = (
            StandardizerConfig.from_dict(std_cfg) if isinstance(std_cfg, dict) else None
        )

        return RewardParamsConfig(
            clip_sigma=_require(cfg, "clip_sigma", "rewards_config"),
            length_vs_gt=LengthVsGTConfig.from_dict(
                _require(cfg, "length_vs_gt", "rewards_config")
            ),
            duplicate_penalty=_extract_duplicate_penalty(cfg),
            pattern_penalty=_extract_pattern_penalty(cfg),
            assignment_f1=assignment_cfg,
            standardizer=std_parsed,
        )


@dataclass(frozen=True)
class RewardsConfig:
    """Reward configuration - all required."""

    weights: Dict[str, float]
    observe_only: List[str]
    config: RewardParamsConfig

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RewardsConfig":
        """Parse rewards from root config dict."""
        rewards_weights = _require(cfg, "rewards")
        if not isinstance(rewards_weights, dict):
            raise ConfigValidationError("'rewards' must be a dict of reward weights")

        observe = cfg.get("observe_rewards", [])
        if not isinstance(observe, list):
            raise ConfigValidationError("'observe_rewards' must be a list")

        rewards_config = _require(cfg, "rewards_config")

        return RewardsConfig(
            weights=rewards_weights,
            observe_only=observe,
            config=RewardParamsConfig.from_dict(rewards_config),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration - minimal."""

    enabled: bool
    eval_every_steps: int
    eval_data_size: int
    seed: int
    # New: text dump controls moved from env to YAML
    dump_conversations: bool
    dump_max_per_step: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "EvaluationConfig":
        return EvaluationConfig(
            enabled=_require(cfg, "enabled", "evaluation"),
            eval_every_steps=_require(cfg, "eval_every_steps", "evaluation"),
            eval_data_size=_require(cfg, "eval_data_size", "evaluation"),
            seed=_require(cfg, "seed", "evaluation"),
            dump_conversations=_require(cfg, "dump_conversations", "evaluation"),
            dump_max_per_step=_require(cfg, "dump_max_per_step", "evaluation"),
        )


@dataclass(frozen=True)
class LossConfig:
    """Loss configuration - immutable ratios from base."""

    teacher_loss_weight: float
    student_loss_weight: float
    caption_loss_weight: float
    grounding_loss_weight: float
    formatting_loss_weight: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LossConfig":
        return LossConfig(
            teacher_loss_weight=_require(cfg, "teacher_loss_weight", "loss"),
            student_loss_weight=_require(cfg, "student_loss_weight", "loss"),
            caption_loss_weight=_require(cfg, "caption_loss_weight", "loss"),
            grounding_loss_weight=_require(cfg, "grounding_loss_weight", "loss"),
            formatting_loss_weight=_require(cfg, "formatting_loss_weight", "loss"),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration - mostly immutable from base."""

    ddp_backend: str
    local_rank: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RuntimeConfig":
        return RuntimeConfig(
            ddp_backend=_require(cfg, "ddp_backend", "runtime"),
            local_rank=_require(cfg, "local_rank", "runtime"),
        )


@dataclass(frozen=True)
class RLConfig:
    """Root v2 config - all fields required."""

    paths: PathsConfig
    experiment: ExperimentConfig
    model: ModelConfig
    sampling: SamplingConfig
    generation: GenerationConfig
    grpo: GRPOConfig
    # normalization section is optional and ignored
    training: TrainingConfig
    optimizer: OptimizerConfig
    layer_freezing: LayerFreezingConfig
    logging: LoggingConfig
    checkpointing: CheckpointingConfig
    rewards: RewardsConfig
    evaluation: EvaluationConfig
    loss: LossConfig
    runtime: RuntimeConfig

    @staticmethod
    def from_yaml_dict(cfg: Dict[str, Any]) -> "RLConfig":
        """Load v2 config with strict validation."""

        # Detect new structure
        using_v2 = "paths" in cfg and isinstance(cfg["paths"], dict)

        if using_v2:
            # NEW: Direct mapping with strict validation
            return RLConfig(
                paths=PathsConfig.from_dict(_require(cfg, "paths")),
                experiment=ExperimentConfig.from_dict(_require(cfg, "experiment")),
                model=ModelConfig.from_dict(_require(cfg, "model")),
                sampling=SamplingConfig.from_dict(_require(cfg, "sampling")),
                generation=GenerationConfig.from_dict(_require(cfg, "generation")),
                grpo=GRPOConfig.from_dict(_require(cfg, "grpo")),
                # normalization section is optional and ignored
                training=TrainingConfig.from_dict(_require(cfg, "training")),
                optimizer=OptimizerConfig.from_dict(_require(cfg, "optimizer")),
                layer_freezing=LayerFreezingConfig.from_dict(
                    _require(cfg, "layer_config")
                ),
                logging=LoggingConfig.from_dict(_require(cfg, "logging")),
                checkpointing=CheckpointingConfig.from_dict(
                    _require(cfg, "checkpointing")
                ),
                rewards=RewardsConfig.from_dict(
                    cfg
                ),  # Handles rewards + rewards_config
                evaluation=EvaluationConfig.from_dict(_require(cfg, "evaluation")),
                loss=LossConfig.from_dict(_require(cfg, "loss")),
                runtime=RuntimeConfig.from_dict(_require(cfg, "runtime")),
            )
        else:
            # OLD: Legacy format not supported - force migration
            raise ConfigValidationError(
                "Legacy config format detected. Please migrate to v2 structure with "
                "'paths:', 'experiment:', 'sampling:', 'generation:' sections.\n"
                "See configs/dense_rl/debug.yaml for example."
            )
