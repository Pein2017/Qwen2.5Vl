import os

import pytest

from src_new.rl.data.dataset import RLDenseJSONLDataset
from src_new.rl.runner import build_components
from src_new.rl.utils import create_builder


@pytest.mark.skipif(
    not os.path.exists("configs/dense_rl/debug.yaml"),
    reason="debug config not found",
)
def test_components_load_and_dataset_builds():
    cfg_path = "configs/dense_rl/debug.yaml"
    bundles = build_components(cfg_path)
    tok = bundles["tokenizer"]
    proc = bundles["processor"]
    raw_cfg = bundles["raw_config"]

    assert tok is not None and proc is not None
    train_path = raw_cfg.get("train_data_path")
    data_root = raw_cfg.get("data_root")
    assert isinstance(train_path, str) and isinstance(data_root, str)

    builder = create_builder(proc)
    ds = RLDenseJSONLDataset(
        train_path, ctx=type("C", (), {"builder": builder, "data_root": data_root})()
    )
    # Sanity: can read first sample
    sample = ds[0]
    assert "input_ids" in sample and "attention_mask" in sample and "meta" in sample
