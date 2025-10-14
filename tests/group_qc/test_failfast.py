import json
from pathlib import Path

import pytest

from src_post.data.dataset_group_qc import RLGroupQCDataset


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        RLGroupQCDataset(jsonl_or_dir_path="/non/existent/path/for/group_qc")


def test_jsonl_schema_missing_images_or_label(tmp_path: Path):
    p = tmp_path / "groups.jsonl"
    rows = [
        {"label": "pass", "meta": {"group_id": "g1"}},  # missing images
        {"images": ["/abs/p1.jpg"], "meta": {"group_id": "g2"}},  # missing label
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError):
        RLGroupQCDataset(jsonl_or_dir_path=str(p))
