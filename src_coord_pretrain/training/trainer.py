from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, cast

from torch.utils.data import Subset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
)

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HasTokenizer(Protocol):
    """Protocol for objects exposing a HuggingFace tokenizer and save_pretrained method."""

    tokenizer: Any  # PreTrainedTokenizerBase, but relaxed for runtime variants

    def save_pretrained(self, save_directory: str) -> Any: ...


@dataclass(frozen=True)
class TrainConfig:
    config_path: str

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config YAML")
        p = _resolve_path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config YAML must parse to a dict")
        return cfg


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()


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
    proc = Qwen2VLProcessor.from_pretrained(str(model_path))
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

        if self._freeze_vision and hasattr(self.model, "visual"):
            for _, p in self.model.visual.named_parameters():
                p.requires_grad = False

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
                }
            )
        if llm_params_nodecay:
            param_groups.append(
                {"params": llm_params_nodecay, "lr": self._llm_lr, "weight_decay": 0.0}
            )
        if mlp_params_decay:
            param_groups.append(
                {
                    "params": mlp_params_decay,
                    "lr": self._mlp_lr,
                    "weight_decay": weight_decay,
                }
            )
        if mlp_params_nodecay:
            param_groups.append(
                {"params": mlp_params_nodecay, "lr": self._mlp_lr, "weight_decay": 0.0}
            )

        if not param_groups:
            raise RuntimeError(
                "No trainable parameters found after freezing and grouping."
            )

        from torch.optim import AdamW

        self.optimizer = AdamW(
            param_groups, lr=self._llm_lr, betas=(0.9, 0.999), eps=1e-8
        )

        if self.args.max_steps and self.args.max_steps > 0:
            num_training_steps = self.args.max_steps
        else:
            dl_len = len(self.get_train_dataloader())
            steps_per_epoch = math.ceil(
                dl_len / max(1, self.args.gradient_accumulation_steps)
            )
            num_training_steps = int(steps_per_epoch * self.args.num_train_epochs)
        self.create_scheduler(num_training_steps=num_training_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coord bootstrap training entry")
    parser.add_argument(
        "--config", type=str, required=True, help="Absolute path to YAML config"
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    # Required config keys (now YAML-driven like src_new)
    required_keys = [
        "model_path",
        "output_dir",
        "data_path",
        "max_coord_value",
        "coordinate_tokens_enabled",
        "coordinate_init_mode",
        # runtime/hparams
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "llm_lr",
        "mlp_lr",
        "max_epochs",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "val_ratio",
        "seed",
        "bf16",
        "fp16",
        "weight_decay",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")

    model_path = _resolve_path(cfg["model_path"])
    output_dir = _resolve_path(cfg["output_dir"])
    data_path = _resolve_path(cfg["data_path"])

    processor = _load_processor(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(str(model_path))

    max_coord_value = int(cfg["max_coord_value"])
    coord_meta = _validate_coord_tokens(processor, max_coord_value)

    ds_conf = DatasetConfig(
        data_path=str(data_path),
        max_coord_value=max_coord_value,
        use_apply_chat_template=True,
    )
    dataset = CoordBootstrapDataset(tokenizer=processor.tokenizer, config=ds_conf)
    collator = DataCollatorCoordBootstrap(
        tokenizer=processor.tokenizer, config=CollatorConfig()
    )

    train_subset, eval_subset = _build_subsets(
        dataset, val_ratio=float(cfg["val_ratio"]), seed=int(cfg["seed"])
    )

    deepspeed_cfg = cfg.get("deepspeed_config")
    deepspeed_cfg = str(_resolve_path(deepspeed_cfg)) if deepspeed_cfg else None

    args_train = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["max_epochs"]),
        max_steps=-1,
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        remove_unused_columns=False,
        bf16=bool(cfg["bf16"]),
        fp16=bool(cfg["fp16"]),
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        deepspeed=deepspeed_cfg,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    trainer = CustomLrTrainer(
        model=model,
        args=args_train,
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        data_collator=collator,
        llm_lr=float(cfg["llm_lr"]),
        mlp_lr=float(cfg["mlp_lr"]),
        freeze_vision=bool(cfg.get("freeze_vision", True)),
    )

    trainer.evaluate()
    trainer.train()

    final_metrics = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    with (output_dir / "coord_token_ids.json").open("w", encoding="utf-8") as f:
        json.dump(coord_meta, f, ensure_ascii=False, indent=2)

    coord_cfg = {
        "coordinate_tokens_enabled": True,
        "max_coord_value": max_coord_value,
        "coordinate_token_count": int(max_coord_value) + 1,
        "vocab_size_after": int(
            getattr(processor.tokenizer, "vocab_size", len(processor.tokenizer))
        ),
    }
    with (output_dir / "coordinate_config.json").open("w", encoding="utf-8") as f:
        json.dump(coord_cfg, f, ensure_ascii=False, indent=2)
    with (output_dir / "metrics-final.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
