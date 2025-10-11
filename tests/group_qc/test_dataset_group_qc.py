from pathlib import Path

import pytest
from PIL import Image

from src_post.data.dataset_group_qc import RLGroupQCDataset


def _save_rgb(fp: Path, size=(8, 8)):
    fp.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (127, 127, 127)).save(fp)


def test_pattern_b_with_mission_and_label_normalization(tmp_path: Path):
    root = tmp_path / "group_data" / "bbu_scene_2.0_order"
    mission = "挡风板安装检查"
    gid_pass = "QC-20231218-0025165"
    gid_fail = "QC-TEMP-20240913-0014643"
    _save_rgb(root / mission / "审核通过" / gid_pass / "a.jpeg")
    _save_rgb(root / mission / "审核不通过" / gid_fail / f"{gid_fail}_1.jpeg")

    ds = RLGroupQCDataset(jsonl_or_dir_path=str(root))
    assert len(ds) == 2
    labels = sorted([ds[i]["label"] for i in range(len(ds))])
    assert labels == ["fail", "pass"]
    # mission should be populated in meta for pattern B
    metas = [ds[i]["meta"] for i in range(len(ds))]
    assert all(isinstance(m, dict) for m in metas)
    assert set(mission for m in metas for mission in [m.get("mission")]) == {mission}


def test_pattern_a_without_mission(tmp_path: Path):
    root = tmp_path / "groupsA"
    gid = "QC-20221216-0000197"
    _save_rgb(root / "审核通过" / gid / "1.jpg")
    ds = RLGroupQCDataset(jsonl_or_dir_path=str(root))
    assert len(ds) == 1
    item = ds[0]
    assert item["label"] == "pass"
    assert "mission" not in item["meta"]


def test_unknown_label_dir_under_mission_failfast(tmp_path: Path):
    root = tmp_path / "groupsB"
    mission = "BBU线缆布放要求"
    gid = "QC-20221216-0000197"
    _save_rgb(root / mission / "未知标签" / gid / "x.png")
    with pytest.raises(ValueError):
        RLGroupQCDataset(jsonl_or_dir_path=str(root))
