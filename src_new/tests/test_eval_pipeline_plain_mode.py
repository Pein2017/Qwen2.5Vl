import os
import json
import tempfile
from pathlib import Path

import pytest
import torch

from transformers import Qwen2VLProcessor

from src_new.config.config import load_config, save_config
from src_new.data.dataset import Dataset
from src_new.data.collator import create_data_collator
from src_new.training.bbu_trainer import BBUTrainer


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32000):
        super().__init__()
        self.vocab_size = vocab_size
        # Minimal training_config required by BBUTrainer/TrainingStateManager
        self.training_config = type(
            "_Cfg",
            (),
            {
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "coordinate_tokens_enabled": False,
                "coord_aux_enabled": False,
                "span_include_im_end_in_labels": True,
                "debug_alignment": False,
            },
        )()

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels")
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = input_ids.shape
        # Build spans mask from student spans
        spans_list = kwargs.get("student_assistant_spans") or kwargs.get("assistant_spans") or []
        mask = torch.zeros((batch, seq_len - 1), dtype=torch.bool, device=input_ids.device)
        if isinstance(spans_list, list):
            for b in range(min(batch, len(spans_list))):
                row = spans_list[b]
                if not isinstance(row, list):
                    continue
                for st, ed in row:
                    st2 = max(0, min(seq_len - 1, int(st)))
                    ed2 = max(st2, min(seq_len, int(ed)))
                    if ed2 > st2:
                        mask[b, st2 : ed2 - 1] = True
        # Random logits
        logits = torch.randn(batch, seq_len, self.vocab_size, device=input_ids.device)
        loss = None
        if isinstance(labels, torch.Tensor):
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            per_tok = torch.nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(batch, seq_len - 1)
            denom = (mask & (shifted_labels != -100)).sum().clamp_min(1)
            loss = (per_tok * mask.float()).sum() / denom.float()
        return {"loss": loss, "logits": logits}


def _make_synthetic_sample(img_rel: str):
    # One line object, plain JSON mode should yield one assistant line
    return {
        "images": [img_rel],
        "objects": [
            {"line": [10, 20, 50, 60], "desc": "电线/分布散乱"},
        ],
    }


def _write_minimal_dataset(tmp_root: Path) -> tuple[str, str, str]:
    # Create tiny 1x1 white image to satisfy processor; Qwen2VLImageProcessor expects actual image
    from PIL import Image

    data_root = tmp_root / "data_root"
    images_dir = data_root / "images"
    data_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    img_path = images_dir / "img01.jpg"
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(img_path)

    train_path = data_root / "train.jsonl"
    val_path = data_root / "val.jsonl"
    teacher_pool_path = data_root / "teacher_pool.jsonl"

    sample = _make_synthetic_sample(str(Path("images") / img_path.name))

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(teacher_pool_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return str(train_path), str(val_path), str(teacher_pool_path)


def test_end_to_end_eval_nonzero_loss_plain_mode(tmp_path: Path):
    # 1) Prepare minimal dataset
    train_path, val_path, teacher_pool_path = _write_minimal_dataset(tmp_path)

    # 2) Load and override standard.yaml minimally
    repo_root = Path(__file__).resolve().parents[2]
    std_yaml = repo_root / "configs/phase_1/standard.yaml"
    cfg = load_config(str(std_yaml))

    # Override paths and reduce workload
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    cfg_dict.update(
        {
            "data_root": str(Path(train_path).parent),
            "train_data_path": train_path,
            "val_data_path": val_path,
            "teacher_pool_file": teacher_pool_path,
            "output_dir": str(tmp_path / "outputs"),
            "tb_dir": str(tmp_path / "tb"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_steps": 1,
            "save_steps": 1000,
            "save_total_limit": 1,
            "logging_steps": 1,
            "run_name": "pytest_plain_mode",
            # Force plain text JSON mode
            "plain_text_mode_enabled": True,
            # Coordinate tokens OFF; plain text JSON mode ON
            "coordinate_tokens_enabled": False,
            # Collator standard to keep batching simple
            "collator_type": "standard",
            # No augmentation
            "use_aug": False,
            "augmentation": None,
            "augmentation_schedule": [],
            "teacher_augmentation": None,
            # Small sequence to accelerate
            "max_total_length": 2048,
            # Single GPU assumed
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": False,
            # Faster CPU dataloader in test
            "dataloader_num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 1,
        }
    )

    # Persist a test-local config to reuse loader/validation path
    test_cfg_path = tmp_path / "test_standard_local.yaml"
    save_config(type(cfg)(**cfg_dict), str(test_cfg_path))
    cfg2 = load_config(str(test_cfg_path))

    # 3) Build processor, datasets, collator
    processor = Qwen2VLProcessor.from_pretrained(cfg2.model_path, trust_remote_code=True)

    train_ds = Dataset(
        data_path=cfg2.train_data_path,
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        teacher_pool_manager=None,
        config=cfg2,
    )
    val_ds = Dataset(
        data_path=cfg2.val_data_path,
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        teacher_pool_manager=None,
        config=cfg2,
    )

    # Initialize conversation processors on datasets
    train_ds.set_processor(processor)
    val_ds.set_processor(processor)

    # Assert augmentation disabled
    assert getattr(train_ds, "augmentation_pipeline", None) is None
    assert getattr(val_ds, "augmentation_pipeline", None) is None

    collator = create_data_collator(collator_type="standard", tokenizer=processor.tokenizer, config=cfg2)

    # Collate a batch
    batch = collator([val_ds[0]])

    # Spans should exist
    spans = batch.get("student_assistant_spans") or batch.get("assistant_spans")
    assert spans and isinstance(spans, list) and len(spans[0]) > 0

    # Ensure there are labeled tokens inside assistant spans (non -100)
    labels = batch["labels"]
    covered = 0
    for st, ed in spans[0]:
        covered += int((labels[0, st:ed] != -100).sum().item())
    assert covered > 0, "No labeled tokens inside assistant spans"

    # Validate group masks coverage/disjointness using plugin (plain JSON mode)
    from src_new.losses.token_grouping import TokenGroupingPlugin

    plugin = TokenGroupingPlugin(processor.tokenizer)
    gm = plugin.build_group_masks(
        labels=batch["labels"],
        teacher_spans=None,
        student_spans=spans,
        input_ids=batch["input_ids"],
    )
    # Assistant mask (shifted)
    assist = torch.zeros_like(batch["labels"], dtype=torch.bool)
    for st, ed in spans[0]:
        assist[0, st:ed] = True
    assist = assist[:, 1:]

    union = gm.student_caption | gm.student_grounding | gm.student_formatting
    # Coverage
    assert torch.equal(union, assist)
    # Disjointness
    assert not (gm.student_caption & gm.student_grounding).any()
    assert not (gm.student_caption & gm.student_formatting).any()
    assert not (gm.student_grounding & gm.student_formatting).any()
