import sys
from types import SimpleNamespace

import pytest
import torch

import src_new.rl.runner as runner


class DummyEmb(torch.nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n, 1))
        self.num_embeddings = n

    def forward(self, x):
        return x


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = DummyEmb(7)
        # Put a parameter on CPU with bf16 dtype compatibility
        self.register_parameter("p", torch.nn.Parameter(torch.zeros(1)))
        self._input_embeddings = self.emb

    def get_input_embeddings(self):
        return self.emb


class DummyTokenizer:
    def __init__(self):
        self._vocab = {"a": 0, "b": 1}

    def get_vocab(self):
        return self._vocab


class DummyProcessor:
    pass


class DummyHFBundle(SimpleNamespace):
    pass


@pytest.fixture()
def tmp_yaml(tmp_path):
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
        model_path: /abs/model
        bf16: true
        model:
          attn_implementation: eager
          image_max_pixels: 401408
        loss:
          teacher_loss_weight: 0.5
          student_loss_weight: 1.0
          caption_loss_weight: 1.0
          grounding_loss_weight: 1.0
          formatting_loss_weight: 1.0
        per_device_train_batch_size: 1
        update_steps: 1
        learning_rate: 1e-5
        weight_decay: 0.0
        max_steps: 1
        warmup_steps: 0
        logging_steps: 1
        save_steps: 1
        seed: 0
        sample_k: 1
        max_new_tokens: 8
        temperature: 1.0
        top_p: 0.95
        repetition_penalty: 1.1
        """,
        encoding="utf-8",
    )
    return str(yml)


def test_main_load_mode_prints_json(tmp_yaml, monkeypatch, capsys):
    def fake_build_hf_components(*args, **kwargs):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        processor = DummyProcessor()
        return DummyHFBundle(model=model, tokenizer=tokenizer, processor=processor)

    monkeypatch.setattr(runner, "build_hf_components", fake_build_hf_components)

    argv = ["runner", "--config", tmp_yaml, "--mode", "load"]
    monkeypatch.setenv("SRC_RL_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("BBU_DEEPSPEED_ENABLED", "false")
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    out = capsys.readouterr().out.strip()
    assert out.startswith("{") and '"ok": true' in out and out.endswith("}")
