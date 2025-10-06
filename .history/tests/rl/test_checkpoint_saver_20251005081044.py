import json
import os
from types import SimpleNamespace

from src_new.training.checkpoint_saver import BestCheckpointManager, CheckpointSaver


class DummyProcessingClass:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class DummyProcessor:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(
            os.path.join(path, "preprocessor_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"ok": True}, f)


class DummyModel:
    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.safetensors"), "wb") as f:
            f.write(b"\x00\x01")

    @property
    def generation_config(self):
        return None


def test_checkpoint_saver_ensures_aux_files(tmp_path):
    args = SimpleNamespace(should_save=True)
    saver = CheckpointSaver(args=args, checkpoint_manager=BestCheckpointManager())

    ckpt = tmp_path / "ckpt"
    model = DummyModel()
    tok = DummyProcessingClass()

    saver._save_inference_checkpoint(
        model=model, checkpoint_dir=str(ckpt), processing_class=tok, processor=None
    )

    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "tokenizer.json").exists()
    # Fallback preprocessor config should exist even when processor is None
    assert (ckpt / "preprocessor_config.json").exists()
