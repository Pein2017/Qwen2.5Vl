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
    sample_k_per_rank: bool
    reward_average_window: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "SamplingConfig":
        return SamplingConfig(
            prompt_batch_size=_require(cfg, "prompt_batch_size", "sampling"),
            sample_k=_require(cfg, "sample_k", "sampling"),
            sample_k_per_rank=_require(cfg, "sample_k_per_rank", "sampling"),
            reward_average_window=_require(cfg, "reward_average_window", "sampling"),
        )


@dataclass(frozen=True)
class DynamicLengthConfig:
    """Dynamic length - all required."""

    enabled: bool
    estimator: str
    alpha: float
    eos_margin: int
    min_cap: int
    max_cap: int
    hard_cap: bool

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "DynamicLengthConfig":
        return DynamicLengthConfig(
            enabled=_require(cfg, "enabled", "dynamic_length"),
            estimator=_require(cfg, "estimator", "dynamic_length"),
            alpha=_require(cfg, "alpha", "dynamic_length"),
            eos_margin=_require(cfg, "eos_margin", "dynamic_length"),
            min_cap=_require(cfg, "min_cap", "dynamic_length"),
            max_cap=_require(cfg, "max_cap", "dynamic_length"),
            hard_cap=_require(cfg, "hard_cap", "dynamic_length"),
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Generation config - all required."""

    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float
    dynamic_length: DynamicLengthConfig

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "GenerationConfig":
        return GenerationConfig(
            max_new_tokens=_require(cfg, "max_new_tokens", "generation"),
            min_new_tokens=_require(cfg, "min_new_tokens", "generation"),
            temperature=_require(cfg, "temperature", "generation"),
            top_p=_require(cfg, "top_p", "generation"),
            dynamic_length=DynamicLengthConfig.from_dict(
                _require(cfg, "dynamic_length", "generation")
            ),
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
    """GRPO algorithm - all required except beta_anneal, max_advantage_magnitude, and steps_per_generation."""

    epsilon_low: float
    epsilon_high: float
    beta_start: float
    loss_type: str
    scale_rewards: bool
    mask_truncated_completions: bool
    beta_anneal: Optional[BetaAnnealConfig] = (
        None  # Truly optional (must come after required)
    )
    max_advantage_magnitude: Optional[float] = (
        None  # Truly optional (must come after required)
    )
    steps_per_generation: Optional[int] = (
        None  # None = auto (same as gradient_accumulation_steps)
    )

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "GRPOConfig":
        beta_anneal_dict = cfg.get("beta_anneal")
        steps_per_gen = cfg.get("steps_per_generation")

        # Validate steps_per_generation if provided
        if steps_per_gen is not None:
            steps_per_gen = int(steps_per_gen)
            if steps_per_gen < 1:
                raise ConfigValidationError(
                    f"grpo.steps_per_generation must be >= 1, got {steps_per_gen}"
                )

        return GRPOConfig(
            epsilon_low=_require(cfg, "epsilon_low", "grpo"),
            epsilon_high=_require(cfg, "epsilon_high", "grpo"),
            beta_start=_require(cfg, "beta_start", "grpo"),
            loss_type=_require(cfg, "loss_type", "grpo"),
            scale_rewards=_require(cfg, "scale_rewards", "grpo"),
            mask_truncated_completions=_require(
                cfg, "mask_truncated_completions", "grpo"
            ),
            beta_anneal=BetaAnnealConfig.from_dict(beta_anneal_dict)
            if beta_anneal_dict
            else None,
            max_advantage_magnitude=cfg.get("max_advantage_magnitude"),  # Optional
            steps_per_generation=steps_per_gen,  # Optional
        )


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization config - all required."""

    cross_rank_advantages: bool

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "NormalizationConfig":
        return NormalizationConfig(
            cross_rank_advantages=_require(
                cfg, "cross_rank_advantages", "normalization"
            ),
        )


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

        dataset_size = cfg.get("dataset_size", -1)
        if dataset_size < -1:
            raise ConfigValidationError(
                f"dataset_size must be >= -1, got {dataset_size}"
            )

        warmup = _require(cfg, "warmup_ratio", "training")
        if warmup < 0.0:
            raise ConfigValidationError(f"warmup_ratio must be >= 0.0, got {warmup}")

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
        )


@dataclass(frozen=True)
class LearningRatesConfig:
    """Learning rates - all required."""

    llm: float
    vision: float
    merger: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LearningRatesConfig":
        return LearningRatesConfig(
            llm=_require(cfg, "llm", "learning_rates"),
            vision=_require(cfg, "vision", "learning_rates"),
            merger=_require(cfg, "merger", "learning_rates"),
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
    """Length vs GT reward config - all required."""

    estimator: str
    lower: float
    upper: float
    gamma: float
    tail_numeric_weight: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "LengthVsGTConfig":
        return LengthVsGTConfig(
            estimator=_require(cfg, "estimator", "length_vs_gt"),
            lower=_require(cfg, "lower", "length_vs_gt"),
            upper=_require(cfg, "upper", "length_vs_gt"),
            gamma=_require(cfg, "gamma", "length_vs_gt"),
            tail_numeric_weight=_require(cfg, "tail_numeric_weight", "length_vs_gt"),
        )


@dataclass(frozen=True)
class RewardParamsConfig:
    """Reward-specific hyperparameters - all required."""

    clip_sigma: float
    tau_iou: float
    tau_quad: float
    tau_line: float
    length_vs_gt: LengthVsGTConfig
    # Optional per-reward params
    line_giou: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RewardParamsConfig":
        line_giou_cfg = cfg.get("line_giou")
        if line_giou_cfg is not None and not isinstance(line_giou_cfg, dict):
            raise ConfigValidationError(
                "rewards_config.line_giou must be a mapping when provided"
            )
        # If provided, validate known fields
        if isinstance(line_giou_cfg, dict) and "buffer_frac" in line_giou_cfg:
            bf = line_giou_cfg["buffer_frac"]
            try:
                _ = float(bf)
            except Exception:
                raise ConfigValidationError(
                    "rewards_config.line_giou.buffer_frac must be a float"
                )

        return RewardParamsConfig(
            clip_sigma=_require(cfg, "clip_sigma", "rewards_config"),
            tau_iou=_require(cfg, "tau_iou", "rewards_config"),
            tau_quad=_require(cfg, "tau_quad", "rewards_config"),
            tau_line=_require(cfg, "tau_line", "rewards_config"),
            length_vs_gt=LengthVsGTConfig.from_dict(
                _require(cfg, "length_vs_gt", "rewards_config")
            ),
            line_giou=line_giou_cfg,
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
    """Evaluation configuration - all required."""

    enabled: bool
    eval_every_steps: int
    rounds: int
    per_rank_samples: int
    save_samples: int
    log_text_snippets: bool
    seed: int

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "EvaluationConfig":
        return EvaluationConfig(
            enabled=_require(cfg, "enabled", "evaluation"),
            eval_every_steps=_require(cfg, "eval_every_steps", "evaluation"),
            rounds=_require(cfg, "rounds", "evaluation"),
            per_rank_samples=_require(cfg, "per_rank_samples", "evaluation"),
            save_samples=_require(cfg, "save_samples", "evaluation"),
            log_text_snippets=_require(cfg, "log_text_snippets", "evaluation"),
            seed=_require(cfg, "seed", "evaluation"),
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
    normalization: NormalizationConfig
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
                normalization=NormalizationConfig.from_dict(
                    _require(cfg, "normalization")
                ),
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
