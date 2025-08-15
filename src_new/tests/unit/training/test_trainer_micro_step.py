#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro training-step and checkpoint smoke test for BBUTrainer without external deps.
"""

import os
from unittest.mock import Mock

import torch
from transformers import TrainingArguments

from src_new.models.loss_manager import LossManager
from src_new.models.wrapper import DetectionModel
from src_new.processing.token_processor import TokenConfig, TokenProcessor
from src_new.training.bbu_trainer import BBUTrainer


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=151665, hidden=8):
        super().__init__()
        self.config = type("C", (), {"vocab_size": vocab_size, "hidden_size": hidden})()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size: int, **kwargs):
        old_in = self.embed
        old_out = self.lm_head
        hidden = old_in.embedding_dim
        new_in = torch.nn.Embedding(new_size, hidden)
        new_out = torch.nn.Linear(hidden, new_size, bias=False)
        with torch.no_grad():
            rows = min(old_in.num_embeddings, new_size)
            new_in.weight[:rows] = old_in.weight[:rows]
            new_out.weight[:rows, :] = old_out.weight[:rows, :]
        self.embed = new_in
        self.lm_head = new_out
        self.config.vocab_size = new_size

    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # Save a minimal state dict to mimic HF save
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        return

    def forward(self, input_ids=None, labels=None):
        if input_ids is None:
            raise ValueError("input_ids required")
        # Ensure shape [batch, seq]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        x = self.embed(input_ids)  # [b, s, h]
        logits = self.lm_head(x)  # [b, s, v]
        # Dummy loss to keep interface; DetectionModel's LossManager computes real components
        loss = None
        return type("O", (), {"loss": loss, "logits": logits})()


class DummyCfg:
    coordinate_tokens_enabled = True
    max_coord_value = 1024
    use_cache = False
    torch_dtype = "float32"
    attn_implementation = "eager"
    skip_vocab_extension = True
    best_checkpoint_metric = "eval_loss"
    best_checkpoint_greater_is_better = False

    # Loss manager required attributes
    coordinate_loss_weight = 0.05
    regular_loss_weight = 1.0
    teacher_loss_weight = 0.3
    student_loss_weight = 1.0
    coordinate_temperature = 1.0

    # Coordinate auxiliary loss attributes (disabled for testing)
    coord_aux_enabled = False
    coord_aux_tau = 1.2
    coord_aux_sigma_bins = 8.0
    coord_aux_window_bins = 32
    coord_aux_topk = 100
    coord_aux_lambda_kce = 0.5
    coord_aux_lambda_unlike = 0.05


def test_trainer_micro_step_and_checkpoint(tmp_path):
    # Ensure CPU-only execution to avoid CUDA/DataParallel issues in unit test
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Tokenizer and model setup
    class Tok:
        def __init__(self):
            self._v = {f"tok_{i}": i for i in range(151665)}

        def get_vocab(self):
            return dict(self._v)

        def add_special_tokens(self, d):
            toks = d.get("additional_special_tokens", [])
            for t in toks:
                if t not in self._v:
                    self._v[t] = len(self._v)
            return len(toks)

    tok = Tok()
    tp = TokenProcessor(
        TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=1024,
            coordinate_init_mode="fourier_ramp",
        )
    )
    # Extend vocabulary and model embeddings to keep ids in range
    tok = tp.extend_tokenizer_vocabulary(tok)

    model = TinyModel()
    tp.extend_model_embeddings(model, tok)

    wrapped = DetectionModel(
        base_model=model, config=DummyCfg(), tokenizer=tok, skip_expansion=True
    )

    # Minimal dataset with a single tiny sample
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            # one token + label (1D seq tensors to avoid extra batch dim stacking)
            return {
                "input_ids": torch.tensor([151667]),
                "labels": torch.tensor([151667]),
            }

    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        logging_steps=1,
        learning_rate=1e-4,
        disable_tqdm=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_safetensors=True,
        no_cuda=True,  # Force CPU for this unit test to avoid CUDA/DP side effects
    )

    # Patch Trainer wrapping to avoid Distributed wrappers in this unit test
    orig_wrap = BBUTrainer._wrap_model
    BBUTrainer._wrap_model = lambda self, model, training=True, dataloader=None: model

    try:
        trainer = BBUTrainer(
            model=wrapped,
            processing_class=None,
            training_args=args,
            train_dataset=DS(),
        )

        trainer.train()
    finally:
        # Restore original wrapping method
        BBUTrainer._wrap_model = orig_wrap

    # Check that checkpoint artifacts directory is created
    ckpt_dirs = [p for p in (tmp_path / "out").glob("checkpoint-*")]
    assert len(ckpt_dirs) >= 1

    # Ensure embedding shapes remain padded and valid
    assert wrapped.base_model.get_input_embeddings().weight.shape[0] % 128 == 0


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=151665, hidden_size=32):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = type(
            "C", (), {"vocab_size": vocab_size, "hidden_size": hidden_size}
        )()

    def forward(self, input_ids=None, labels=None):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return type("O", (), {"loss": None, "logits": logits})()


class DummyTrainer:
    def __init__(self, loss_manager: LossManager):
        self.loss_manager = loss_manager

    def training_step(self, model: torch.nn.Module, batch):
        out = model(**batch)
        # Minimal call to compute components to exercise path
        _ = self.loss_manager.compute_loss_components(
            logits=out.logits, labels=batch["labels"]
        )
        return out


def test_micro_step_trainer_flow(mock_config):
    # Basic sanity check for micro-step flow with small dummy model
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = {
        "<|line_start|>": 151665,
        "<|line_end|>": 151666,
    }
    loss_manager = LossManager(
        mock_config,
        TokenProcessor(
            TokenConfig(
                coordinate_tokens_enabled=False,
                max_coord_value=32,
                coordinate_init_mode="fourier_ramp",
            )
        ),
        tokenizer,
    )
    model = DummyModel(vocab_size=152704, hidden_size=32)
    trainer = DummyTrainer(loss_manager)
    batch = {
        "input_ids": torch.randint(0, 152704, (1, 16)),
        "labels": torch.randint(0, 152704, (1, 16)),
    }
    outputs = trainer.training_step(model, batch)
    assert outputs is not None


def test_training_state_manager_merges_diagnostics(mock_config):
    # Verify that diagnostics dicts are merged into logs automatically
    from unittest.mock import Mock

    from src_new.models.loss_manager import LossComponents
    from src_new.training.training_state_manager import TrainingStateManager

    # Minimal model and config
    model = Mock()
    manager = TrainingStateManager(config=mock_config, model=model)

    # Simulate accumulated components with diagnostics
    diagnostics = {
        "student_window_mass": 0.9,
        "student_gt_prob": 0.8,
    }
    comp = LossComponents(
        loss=torch.tensor(1.0),
        diagnostics={k: torch.tensor(v) for k, v in diagnostics.items()},
    )
    manager.accumulate_loss_components(comp)
    logs = manager.log_training_metrics(
        tr_loss=torch.tensor(1.0), grad_norm=None, model=model, start_time=0.0
    )

    assert "student_window_mass" in logs
    assert "student_gt_prob" in logs
    assert abs(logs["student_window_mass"] - diagnostics["student_window_mass"]) < 1e-6
    assert abs(logs["student_gt_prob"] - diagnostics["student_gt_prob"]) < 1e-6
