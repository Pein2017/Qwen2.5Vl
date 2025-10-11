#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class GroupSample:
    image_paths: List[str]
    label: str
    meta: Dict[str, Any]


class RLGroupQCDataset(Dataset):
    """RL dataset where each item is a group (work order) of images and a group-level label.

    Supported sources:
      - JSONL file: each line {"images": [...], "label": "pass"|"fail", "meta": {...}}
      - Directory (two patterns):
        A) {root}/{审核通过|审核不通过}/{group_id}/*.(jpg|jpeg|png)
        B) {root}/{mission}/{审核通过|审核不通过}/{group_id}/*.(jpg|jpeg|png)
    """

    def __init__(self, jsonl_or_dir_path: str, preload: bool = False) -> None:
        self.source_path = str(jsonl_or_dir_path)
        self._samples: List[GroupSample] = []
        self._preloaded_images: Optional[List[List[Image.Image]]] = None

        p = Path(self.source_path)
        if p.is_dir():
            self._load_from_dir(p)
        else:
            self._load_jsonl(p)

        if preload:
            self._preload_images()

    def _normalize_label_name(self, name: str) -> Optional[str]:
        n = name.strip().lower()
        if n in {"审核通过", "通过", "pass", "passed"}:
            return "pass"
        if n in {"审核不通过", "不通过", "fail", "failed"}:
            return "fail"
        return None

    def _collect_groups_under_label_dir(
        self, label_dir: Path, label_norm: str, mission: Optional[str]
    ) -> int:
        """Collect groups under a normalized label directory. Returns number of groups found."""
        count = 0
        for group_dir in sorted([d for d in label_dir.iterdir() if d.is_dir()]):
            imgs: List[str] = []
            for f in sorted(group_dir.iterdir()):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in {".jpg", ".jpeg", ".png"}:
                    imgs.append(str(f.resolve()))
            if not imgs:
                continue
            meta: Dict[str, Any] = {
                "group_id": group_dir.name,
                "label_dir": label_dir.name,
            }
            if mission:
                meta["mission"] = mission
            self._samples.append(
                GroupSample(
                    image_paths=imgs,
                    label=label_norm,
                    meta=meta,
                )
            )
            count += 1
        return count

    def _load_from_dir(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        # Detect pattern A: root contains label dirs directly
        first_level_dirs = [d for d in root.iterdir() if d.is_dir()]
        if not first_level_dirs:
            raise ValueError(
                f"No subdirectories found under {root}. Expected at least label or mission directories"
            )

        # If any first-level dir is a label dir, treat as Pattern A
        any_label = any(
            self._normalize_label_name(d.name) is not None for d in first_level_dirs
        )
        if any_label:
            # Pattern A: root/{label}/{group_id}
            for ldir in sorted(first_level_dirs):
                label_norm = self._normalize_label_name(ldir.name)
                if label_norm is None:
                    # skip unrelated dirs at this level
                    continue
                self._collect_groups_under_label_dir(ldir, label_norm, mission=None)
        else:
            # Pattern B: root/{mission}/{label}/{group_id}
            total_groups = 0
            for mission_dir in sorted(first_level_dirs):
                if not mission_dir.is_dir():
                    continue
                mission_name = mission_dir.name
                label_dirs = [d for d in mission_dir.iterdir() if d.is_dir()]
                if not label_dirs:
                    continue
                for ldir in sorted(label_dirs):
                    label_norm = self._normalize_label_name(ldir.name)
                    if label_norm is None:
                        # Strict fail-fast for unknown label names under mission
                        raise ValueError(
                            f"Unknown label directory under mission '{mission_name}': {ldir.name}. Expected one of: 审核通过|审核不通过|通过|不通过|pass|fail"
                        )
                    total_groups += self._collect_groups_under_label_dir(
                        ldir, label_norm, mission=mission_name
                    )
            if total_groups == 0:
                raise ValueError(
                    f"No labeled groups found under {root}. Expected structure: {root}/<mission>/审核通过|审核不通过/<group_id>/*.jpeg"
                )

        if not self._samples:
            raise ValueError(
                f"No groups found under {root} with expected image extensions"
            )

    def _load_jsonl(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.source_path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                images = obj.get("images", [])
                if not isinstance(images, list) or len(images) == 0:
                    raise ValueError(
                        "Each JSONL row must have a non-empty 'images' list"
                    )
                label = obj.get("label", None)
                if not isinstance(label, str):
                    raise ValueError(
                        "Each JSONL row must have string 'label' (e.g., 'pass'|'fail')"
                    )
                meta = obj.get("meta", {}) or {}
                self._samples.append(
                    GroupSample(
                        image_paths=[str(Path(p).resolve()) for p in images],
                        label=label,
                        meta=meta,
                    )
                )

    def _preload_images(self) -> None:
        self._preloaded_images = []
        for sample in self._samples:
            imgs: List[Image.Image] = []
            for p in sample.image_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(img)
            self._preloaded_images.append(imgs)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._samples[idx]
        if self._preloaded_images is not None:
            images = self._preloaded_images[idx]
        else:
            images = [Image.open(p).convert("RGB") for p in sample.image_paths]
        return {
            "image_paths": sample.image_paths,
            "images": images,
            "label": sample.label,
            "meta": sample.meta,
            "num_images": len(sample.image_paths),
        }

    def get_label_indices(self) -> Dict[str, List[int]]:
        """Return mapping from normalized label ('pass'|'fail') to list of dataset indices.

        This method does not load any images and relies on labels gathered at init time.
        """
        mapping: Dict[str, List[int]] = {"pass": [], "fail": []}
        for i, s in enumerate(self._samples):
            lab = str(s.label).strip().lower()
            if lab not in mapping:
                mapping[lab] = []
            mapping[lab].append(i)
        return mapping
