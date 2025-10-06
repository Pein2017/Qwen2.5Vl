"""
Enhanced Configuration System for GRPO Training

This module provides comprehensive configuration management aligned with src_new training pipeline,
supporting layer freezing, differentiated learning rates, logging, and checkpointing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for layer freezing and unfreezing."""

    # Vision tower settings
    vision_freeze_patch_embed: bool = True
    vision_freeze_bottom_layers: bool = True
    vision_trainable_top_k_blocks: int = 4

    # LLM layer settings
    llm_freeze_bottom_layers: bool = False
    llm_trainable_top_k_blocks: int = -1  # -1 means all layers

    # Merger/projection settings
    merger_freeze: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LayerConfig":
        """Create LayerConfig from nested dictionary."""
        vision = data.get("vision_tower", {})
        llm = data.get("llm", {})
        merger = data.get("merger", {})

        return LayerConfig(
            vision_freeze_patch_embed=vision.get("freeze_patch_embed", True),
            vision_freeze_bottom_layers=vision.get("freeze_bottom_layers", True),
            vision_trainable_top_k_blocks=vision.get("trainable_top_k_blocks", 4),
            llm_freeze_bottom_layers=llm.get("freeze_bottom_layers", False),
            llm_trainable_top_k_blocks=llm.get("trainable_top_k_blocks", -1),
            merger_freeze=merger.get("freeze", False),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer and learning rates."""

    type: str = "adamw"

    # Layered learning rates (aligned with src_new)
    base_lr: float = 5.0e-6
    vision_lr: float = 1.0e-7
    merger_lr: float = 5.0e-5
    coordinate_lr: float = 1.0e-7

    # Adam parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-3
    max_grad_norm: float = 1.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OptimizerConfig":
        """Create OptimizerConfig from dictionary."""
        learning_rates = data.get("learning_rates", {})

        return OptimizerConfig(
            type=data.get("type", "adamw"),
            base_lr=learning_rates.get("base", 5.0e-6),
            vision_lr=learning_rates.get("vision", 1.0e-7),
            merger_lr=learning_rates.get("merger", 5.0e-5),
            coordinate_lr=learning_rates.get("coordinate", 1.0e-7),
            adam_beta1=data.get("adam_beta1", 0.9),
            adam_beta2=data.get("adam_beta2", 0.95),
            adam_epsilon=data.get("adam_epsilon", 1.0e-8),
            weight_decay=data.get("weight_decay", 1.0e-3),
            max_grad_norm=data.get("max_grad_norm", 1.0),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training parameters."""

    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # Memory optimization
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Precision
    bf16: bool = True
    fp16: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from dictionary."""
        return TrainingConfig(
            num_train_epochs=data.get("num_train_epochs", 3),
            max_steps=data.get("max_steps", -1),
            per_device_train_batch_size=data.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=data.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 4),
            lr_scheduler_type=data.get("lr_scheduler_type", "cosine"),
            warmup_ratio=data.get("warmup_ratio", 0.1),
            warmup_steps=data.get("warmup_steps", 0),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            dataloader_num_workers=data.get("dataloader_num_workers", 8),
            pin_memory=data.get("pin_memory", True),
            prefetch_factor=data.get("prefetch_factor", 4),
            bf16=data.get("bf16", True),
            fp16=data.get("fp16", False),
        )


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO-specific parameters."""

    sample_k: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    # GRPO algorithm parameters
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.01
    loss_type: str = "grpo"

    # Reward scaling
    scale_rewards: bool = True
    mask_truncated_completions: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GRPOConfig":
        """Create GRPOConfig from dictionary."""
        return GRPOConfig(
            sample_k=data.get("sample_k", 4),
            max_new_tokens=data.get("max_new_tokens", 128),
            temperature=data.get("temperature", 0.9),
            top_p=data.get("top_p", 1.0),
            repetition_penalty=data.get("repetition_penalty", 1.05),
            epsilon_low=data.get("epsilon_low", 0.2),
            epsilon_high=data.get("epsilon_high", 0.2),
            beta=data.get("beta", 0.01),
            loss_type=data.get("loss_type", "grpo"),
            scale_rewards=data.get("scale_rewards", True),
            mask_truncated_completions=data.get("mask_truncated_completions", True),
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for logging and monitoring."""

    logging_steps: int = 5
    log_level: str = "INFO"
    disable_tqdm: bool = False

    # TensorBoard integration
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: Optional[str] = None

    # Detailed metrics
    log_predictions: bool = True
    log_rewards_breakdown: bool = True
    log_on_each_node: bool = False

    # Wandb integration
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LoggingConfig":
        """Create LoggingConfig from dictionary."""
        wandb = data.get("wandb", {})

        return LoggingConfig(
            logging_steps=data.get("logging_steps", 5),
            log_level=data.get("log_level", "INFO"),
            disable_tqdm=data.get("disable_tqdm", False),
            report_to=data.get("report_to", ["tensorboard"]),
            logging_dir=data.get("logging_dir"),
            log_predictions=data.get("log_predictions", True),
            log_rewards_breakdown=data.get("log_rewards_breakdown", True),
            log_on_each_node=data.get("log_on_each_node", False),
            wandb_enabled=wandb.get("enabled", False),
            wandb_project=wandb.get("project"),
            wandb_entity=wandb.get("entity"),
            wandb_tags=wandb.get("tags", []),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpointing and evaluation."""

    # Save strategy
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_reward"
    greater_is_better: bool = True

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Save additional components
    save_tokenizer: bool = True
    save_processor: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CheckpointConfig":
        """Create CheckpointConfig from dictionary."""
        return CheckpointConfig(
            save_strategy=data.get("save_strategy", "steps"),
            save_steps=data.get("save_steps", 100),
            save_total_limit=data.get("save_total_limit", 3),
            evaluation_strategy=data.get("evaluation_strategy", "steps"),
            eval_steps=data.get("eval_steps", 50),
            load_best_model_at_end=data.get("load_best_model_at_end", True),
            metric_for_best_model=data.get("metric_for_best_model", "eval_reward"),
            greater_is_better=data.get("greater_is_better", True),
            resume_from_checkpoint=data.get("resume_from_checkpoint"),
            save_tokenizer=data.get("save_tokenizer", True),
            save_processor=data.get("save_processor", True),
        )


@dataclass(frozen=True)
class EnhancedRLConfig:
    """
    Enhanced RL configuration that encompasses all training aspects.

    This replaces the simple RLLoaderConfig with a comprehensive system
    aligned with src_new training pipeline.
    """

    # Core paths
    model_path: str
    train_data_path: str
    val_data_path: str
    data_root: str

    # Model configuration
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    max_pixels: int = 27525120

    # Structured configurations
    layer_config: LayerConfig = field(default_factory=LayerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)

    # Output configuration
    output_dir: str = "/data3/Qwen2.5-VL-main/outputs/rl_enhanced_grpo"
    run_name: str = "enhanced_grpo"
    overwrite_output_dir: bool = False

    # Runtime settings
    seed: int = 42
    local_rank: int = -1
    ddp_backend: str = "nccl"

    @staticmethod
    def from_yaml_dict(cfg: Dict[str, Any]) -> "EnhancedRLConfig":
        """Create EnhancedRLConfig from YAML dictionary."""
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a mapping (YAML dict)")

        # Validate required paths
        required_paths = ["model_path", "train_data_path", "val_data_path", "data_root"]
        for path_key in required_paths:
            if not cfg.get(path_key):
                raise ValueError(f"Missing required '{path_key}' in config")

        # Model configuration
        model_cfg = cfg.get("model", {})

        # Parse structured configurations
        layer_config = LayerConfig.from_dict(cfg.get("layer_config", {}))
        optimizer_config = OptimizerConfig.from_dict(cfg.get("optimizer", {}))
        training_config = TrainingConfig.from_dict(cfg.get("training", {}))
        grpo_config = GRPOConfig.from_dict(cfg.get("grpo", {}))
        logging_config = LoggingConfig.from_dict(cfg.get("logging", {}))
        checkpoint_config = CheckpointConfig.from_dict(cfg.get("checkpointing", {}))

        # Output configuration
        output_cfg = cfg.get("output", {})
        runtime_cfg = cfg.get("runtime", {})

        return EnhancedRLConfig(
            # Core paths
            model_path=cfg["model_path"],
            train_data_path=cfg["train_data_path"],
            val_data_path=cfg["val_data_path"],
            data_root=cfg["data_root"],
            # Model configuration
            attn_implementation=model_cfg.get("attn_implementation", "eager"),
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            use_cache=model_cfg.get("use_cache", True),
            max_pixels=model_cfg.get("max_pixels", 27525120),
            # Structured configurations
            layer_config=layer_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            grpo_config=grpo_config,
            logging_config=logging_config,
            checkpoint_config=checkpoint_config,
            # Reward weights
            reward_weights=cfg.get("rewards", {}),
            # Output configuration
            output_dir=output_cfg.get(
                "output_dir", "/data3/Qwen2.5-VL-main/outputs/rl_enhanced_grpo"
            ),
            run_name=output_cfg.get("run_name", "enhanced_grpo"),
            overwrite_output_dir=output_cfg.get("overwrite_output_dir", False),
            # Runtime settings
            seed=runtime_cfg.get("seed", 42),
            local_rank=runtime_cfg.get("local_rank", -1),
            ddp_backend=runtime_cfg.get("ddp_backend", "nccl"),
        )

    def to_hf_training_args(self) -> Dict[str, Any]:
        """
        Convert to HuggingFace TrainingArguments format.

        This method creates a dictionary compatible with HF TrainingArguments
        for seamless integration with TRL GRPOTrainer.
        """
        args = {
            # Basic training
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "overwrite_output_dir": self.overwrite_output_dir,
            "num_train_epochs": self.training_config.num_train_epochs,
            "max_steps": self.training_config.max_steps,
            # Batch sizes and gradients
            "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training_config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "gradient_checkpointing": self.training_config.gradient_checkpointing,
            "max_grad_norm": self.optimizer_config.max_grad_norm,
            # Optimizer and learning rate
            "learning_rate": self.optimizer_config.base_lr,
            "weight_decay": self.optimizer_config.weight_decay,
            "adam_beta1": self.optimizer_config.adam_beta1,
            "adam_beta2": self.optimizer_config.adam_beta2,
            "adam_epsilon": self.optimizer_config.adam_epsilon,
            # Learning rate scheduling
            "lr_scheduler_type": self.training_config.lr_scheduler_type,
            "warmup_ratio": self.training_config.warmup_ratio,
            "warmup_steps": self.training_config.warmup_steps,
            # Precision
            "bf16": self.training_config.bf16,
            "fp16": self.training_config.fp16,
            # Logging
            "logging_steps": self.logging_config.logging_steps,
            "report_to": self.logging_config.report_to,
            "logging_dir": self.logging_config.logging_dir,
            "disable_tqdm": self.logging_config.disable_tqdm,
            # Checkpointing
            "save_strategy": self.checkpoint_config.save_strategy,
            "save_steps": self.checkpoint_config.save_steps,
            "save_total_limit": self.checkpoint_config.save_total_limit,
            "evaluation_strategy": self.checkpoint_config.evaluation_strategy,
            "eval_steps": self.checkpoint_config.eval_steps,
            "load_best_model_at_end": self.checkpoint_config.load_best_model_at_end,
            "metric_for_best_model": self.checkpoint_config.metric_for_best_model,
            "greater_is_better": self.checkpoint_config.greater_is_better,
            # Data loading
            "dataloader_num_workers": self.training_config.dataloader_num_workers,
            "dataloader_pin_memory": self.training_config.pin_memory,
            "dataloader_prefetch_factor": self.training_config.prefetch_factor,
            "remove_unused_columns": False,  # Required for multimodal
            # Runtime
            "seed": self.seed,
            "local_rank": self.local_rank,
            "ddp_backend": self.ddp_backend,
        }

        # Add resume checkpoint if specified
        if self.checkpoint_config.resume_from_checkpoint:
            args["resume_from_checkpoint"] = (
                self.checkpoint_config.resume_from_checkpoint
            )

        return args

    def get_layered_learning_rates(self) -> Dict[str, float]:
        """Get learning rates for different model components."""
        return {
            "base": self.optimizer_config.base_lr,
            "vision": self.optimizer_config.vision_lr,
            "merger": self.optimizer_config.merger_lr,
            "coordinate": self.optimizer_config.coordinate_lr,
        }

    def setup_logging(self) -> None:
        """Set up logging based on configuration."""
        log_level = getattr(
            logging, self.logging_config.log_level.upper(), logging.INFO
        )
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Set specific logger levels
        logging.getLogger("src_new.rl").setLevel(log_level)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)


# Legacy compatibility
RLLoaderConfig = EnhancedRLConfig
