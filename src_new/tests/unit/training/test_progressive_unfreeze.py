#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from src_new.config.config import load_config
from src_new.training.callbacks import ProgressiveUnfreezeCallback


class TestProgressiveUnfreezeConfig:
    def _make_temp_config(self, overrides: dict) -> str:
        # Minimal valid config using temp dirs
        base = {
            "model_path": "/tmp",  # will be overwritten to a temp dir below
            "model_size": "3B",
            "model_max_length": 32000,
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            "num_train_epochs": 4,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-6,
            "vision_lr": 5e-7,
            "merger_lr": 1e-5,
            "llm_lr": 5e-6,
            "adapter_lr": 0.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0001,
            "max_grad_norm": 0.5,
            "lr_scheduler_type": "cosine",
            "gradient_checkpointing": True,
            "bf16": True,
            "fp16": False,
            "use_flash_attention": True,
            "mixed_precision": "bf16",
            "data_root": "data",
            "max_total_length": 12000,
            "num_teacher_samples": 0,
            "collator_type": "packed",
            "teacher_ratio": 0.0,
            "language": "chinese",
            "output_dir": "output",
            "run_name": "unfreeze_test",
            "max_coord_value": 1024,
            "model_hidden_size": 2048,
            "coordinate_tokens_enabled": False,
            "coordinate_loss_weight": 0.0,
            "regular_loss_weight": 1.0,
            "coordinate_temperature": 0.7,
            "coordinate_init_mode": "fourier_ramp",
            "coord_aux_enabled": False,
            "coord_aux_tau": 1.2,
            "coord_aux_sigma_bins": 8.0,
            "coord_aux_window_bins": 32,
            "coord_aux_topk": 100,
            "coord_aux_lambda_kce": 0.5,
            "coord_aux_lambda_unlike": 0.05,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 1,
            "logging_steps": 1,
            "logging_dir": "logs",
            "report_to": "tensorboard",
            "disable_tqdm": True,
            "verbose": False,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "tb_dir": "tb",
            "teacher_loss_weight": 0.3,
            "student_loss_weight": 1.0,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "max_pixels": 401408,
            "training_prompt_style": True,
            "use_consistent_prompts": True,
            "skip_vocab_extension": False,
        }
        base.update(overrides)
        # Write to temp YAML with real paths
        tmp_root = tempfile.mkdtemp()
        model_dir = Path(tmp_root) / "model"
        model_dir.mkdir()
        data_dir = Path(tmp_root) / "data"
        data_dir.mkdir()
        base["model_path"] = str(model_dir)
        base["data_root"] = str(data_dir)
        base["train_data_path"] = str(data_dir / "train.jsonl")
        base["val_data_path"] = str(data_dir / "val.jsonl")
        base["teacher_pool_file"] = str(data_dir / "teacher_pool.jsonl")
        base["output_dir"] = str(Path(tmp_root) / "output")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(base, f)
            return f.name

    def test_progressive_unfreeze_validation_ok(self):
        cfg_path = self._make_temp_config(
            {
                "prog_unfreeze_enabled": True,
                "prog_unfreeze_epoch_stage0_end": 1,
                "prog_unfreeze_epoch_stage1_end": 3,
                "prog_unfreeze_top_k_layers": 2,
                "prog_unfreeze_coord_slice_only": True,
                "lr_merger": 1.0e-5,
                "lr_coord_slice": 3.0e-5,
                "lr_top_layers": 6.0e-6,
                "lr_full_model": 5.0e-6,
            }
        )
        cfg = load_config(cfg_path)
        assert cfg.prog_unfreeze_enabled is True
        assert cfg.prog_unfreeze_top_k_layers == 2

    def test_progressive_unfreeze_validation_errors(self):
        # stage0_end invalid
        with pytest.raises(ValueError):
            cfg_path = self._make_temp_config(
                {
                    "prog_unfreeze_enabled": True,
                    "prog_unfreeze_epoch_stage0_end": 0,
                    "prog_unfreeze_epoch_stage1_end": 2,
                    "prog_unfreeze_top_k_layers": 1,
                }
            )
            load_config(cfg_path)
        # stage1_end <= stage0_end
        with pytest.raises(ValueError):
            cfg_path = self._make_temp_config(
                {
                    "prog_unfreeze_enabled": True,
                    "prog_unfreeze_epoch_stage0_end": 2,
                    "prog_unfreeze_epoch_stage1_end": 2,
                    "prog_unfreeze_top_k_layers": 1,
                }
            )
            load_config(cfg_path)
        # top_k_layers < 1
        with pytest.raises(ValueError):
            cfg_path = self._make_temp_config(
                {
                    "prog_unfreeze_enabled": True,
                    "prog_unfreeze_epoch_stage0_end": 1,
                    "prog_unfreeze_epoch_stage1_end": 2,
                    "prog_unfreeze_top_k_layers": 0,
                }
            )
            load_config(cfg_path)


class DummyTrainer:
    class Args:
        num_train_epochs = 4
        gradient_accumulation_steps = 1

    def __init__(self):
        self.args = DummyTrainer.Args()
        self.state = type("S", (), {"epoch": 0})()

        # Build a minimal model stub with parameters and required APIs
        class MinimalEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(100, 8))

        class MinimalLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(100, 8))

        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Include a visual.merger param
                self.visual = nn.Module()
                self.visual.merger = nn.Linear(8, 8)
                # Embeddings and lm_head
                self._emb = MinimalEmb()
                self.lm_head = MinimalLMHead()
                # Add some decoder layer params with indices
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

            def get_input_embeddings(self):
                return self._emb

            def named_parameters(self, *args, **kwargs):
                # Yield a few named params with expected names
                for i, layer in enumerate(self.model.layers):
                    yield (f"model.layers.{i}.self_attn.q_proj.weight", layer.weight)
                # visual merger
                for n, p in self.visual.merger.named_parameters():
                    yield (f"visual.merger.{n}", p)
                # embeddings and lm_head
                yield ("embed_tokens.weight", self._emb.weight)
                yield ("lm_head.weight", self.lm_head.weight)

        self.model = MinimalModel()
        # Attach training_config with layer count
        self.model.training_config = type("C", (), {"model_num_layers": 4})()
        # Minimal coordinate processor with token range for masking path
        self.model.coordinate_processor = type(
            "CP", (), {"coordinate_token_range": (10, 20)}
        )()
        self._rebuild_called = 0

    def create_optimizer(self):
        pass

    def create_scheduler(self, *a, **k):
        pass

    def get_train_dataloader(self):
        class DL:
            def __len__(self):
                return 1

        return DL()


def test_callback_stage_transitions():
    tr = DummyTrainer()
    cb = ProgressiveUnfreezeCallback(
        stage0_end_epoch=1, stage1_end_epoch=3, top_k_layers=2, coord_slice_only=True
    )
    # Inject trainer ref
    cb._trainer_ref = tr
    # Begin training -> stage1
    cb.on_train_begin(args=None, state=tr.state, control=None, trainer=tr)
    assert cb._stage == 1
    # Stage 1: only merger + embeddings + lm_head require grad
    req_names = {n for n, p in tr.model.named_parameters() if p.requires_grad}
    assert any(n.startswith("visual.merger") for n in req_names)
    assert "embed_tokens.weight" in req_names
    assert "lm_head.weight" in req_names
    # Decoder layers should be frozen
    assert all(
        not any(nn.startswith(f"model.layers.{i}") for nn in req_names)
        for i in range(4)
    )
    # Epoch 1 start -> unfreeze top-K -> stage2
    tr.state.epoch = 1
    cb.on_epoch_begin(args=None, state=tr.state, control=None, trainer=tr)
    assert cb._stage == 2
    # Top 2 layers (indices 2,3) should be unfrozen; lower remain frozen
    req_names = {n for n, p in tr.model.named_parameters() if p.requires_grad}
    assert any(n.startswith("model.layers.2") for n in req_names)
    assert any(n.startswith("model.layers.3") for n in req_names)
    assert not any(n.startswith("model.layers.0") for n in req_names)
    assert not any(n.startswith("model.layers.1") for n in req_names)
    # Epoch 3 start -> full unfreeze -> stage3
    tr.state.epoch = 3
    cb.on_epoch_begin(args=None, state=tr.state, control=None, trainer=tr)
    assert cb._stage == 3
    # All layers should be trainable now
    req_names = {n for n, p in tr.model.named_parameters() if p.requires_grad}
    for i in range(4):
        assert any(n.startswith(f"model.layers.{i}") for n in req_names)
