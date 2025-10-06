import os
from types import SimpleNamespace

import src_new.rl.runner as runner


class FakeTrainer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        processor,
        train_dataset,
        val_dataset,
        reward_functions,
        reward_names,
        reward_weights,
        enhanced_cfg,
        raw_config,
        output_dir,
    ):
        self.cfg = raw_config
        self.output_dir = output_dir
        self.closed = False

    def train(self):
        tb_dir = os.path.join(self.cfg.get("tb_dir"), self.cfg.get("run_name"))
        os.makedirs(tb_dir, exist_ok=True)
        # simulate TB scalar write
        with open(os.path.join(tb_dir, "events.fake"), "w", encoding="utf-8") as f:
            f.write("scalar:1\n")
        # simulate checkpoint write
        ckpt_dir = os.path.join(self.output_dir, "checkpoint-1")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"\x00\x01")

    def close(self):
        self.closed = True


class _PFM:
    def apply_phase(self, **kwargs):
        return SimpleNamespace(
            phase="phase_3",
            top_k_llm_layers=0,
            top_k_vision_blocks=0,
            patch_embed_frozen=True,
        )


def test_runner_train_debug_monkeypatched(tmp_path, monkeypatch):
    out = tmp_path / "out"
    tb = tmp_path / "tb"
    out.mkdir()
    tb.mkdir()

    def fake_build_datasets(_path):
        cfg = {
            "rewards": {"parse": 1.0},
            "output_dir": str(out),
            "tb_dir": str(tb),
            "run_name": "debug",
            "layer_config": {
                "llm_trainable_top_k_blocks": 0,
                "vision_trainable_top_k_blocks": 0,
                "vision_freeze_patch_embed": True,
            },
        }
        enhanced_cfg = SimpleNamespace(
            setup_logging=lambda: None,
            layer_config=SimpleNamespace(
                llm_trainable_top_k_blocks=0,
                vision_trainable_top_k_blocks=0,
                vision_freeze_patch_embed=True,
            ),
            output_dir=str(out),
        )
        return {
            "train": None,
            "val": None,
            "tokenizer": object(),
            "model": object(),
            "processor": None,
            "cfg": cfg,
            "enhanced_cfg": enhanced_cfg,
        }

    monkeypatch.setattr(runner, "build_datasets", fake_build_datasets)
    monkeypatch.setattr(runner, "PhaseFreezeManager", _PFM)
    monkeypatch.setattr(runner, "BBUGRPOTrainer", FakeTrainer)

    runner.train("ignored.yaml")

    # Validate outputs
    assert (tb / "debug" / "events.fake").exists()
    ckpts = list(out.glob("checkpoint-1"))
    assert ckpts and (ckpts[0] / "pytorch_model.bin").exists()
