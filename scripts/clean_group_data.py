#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ImageAnno:
    has_negative: bool
    negatives: List[str]
    sample_index: int
    source_path: str


def build_negative_lexicon(include_label_unreadable: bool, extended: bool) -> Set[str]:
    # Base negatives aligned with RL lexicon (src_post/rewards/lexicon.py)
    base: Set[str] = {
        "未拧紧",
        "露铜",
        "复接",
        "生锈",
        "不符合要求",
        "无保护措施",
        "弯曲半径不合理",
        "弯曲半径<4cm或者成环",
        "分布散乱",
        "安装方向错误",
    }
    # Teacher selector variant token
    base.add("弯曲半径不合理（弯曲半径<4cm或者成环）")
    # Shield conformity explicit negative (from mapping/templates semantics)
    if extended:
        base.add("这个BBU设备未按要求配备挡风板")
    if include_label_unreadable:
        base.add("标签/无法识别")
    return base


def normalize_text(s: str) -> str:
    # Collapse whitespace variants to simplify substring checks
    return re.sub(r"\s+", "", s.strip())


def load_all_samples_index(
    jsonl_path: Path,
    include_label_unreadable: bool,
    extended_negatives: bool,
) -> Tuple[Dict[str, ImageAnno], Set[str]]:
    """Index image_id(lower) -> ImageAnno. Also returns the negative lexicon used."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    neg_lex = build_negative_lexicon(include_label_unreadable, extended_negatives)

    index: Dict[str, ImageAnno] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            images = rec.get("images") or []
            if not isinstance(images, list):
                continue
            # Compute negatives
            meta = rec.get("meta") or {}
            has_negative = bool(meta.get("has_negative", False))
            negatives: List[str] = list(meta.get("negatives", [])) if isinstance(meta.get("negatives", []), list) else []
            if not has_negative:
                # Fallback: scan desc fields to be robust
                found: Set[str] = set()
                for obj in rec.get("objects", []) or []:
                    desc = normalize_text(str(obj.get("desc", "")))
                    for tok in neg_lex:
                        if tok in desc:
                            found.add(tok)
                if found:
                    has_negative = True
                    negatives = sorted(found)

            for p in images:
                if not isinstance(p, str):
                    continue
                image_id = os.path.basename(p).lower()
                # Keep the strongest signal if duplicated
                prev = index.get(image_id)
                if prev is None or (has_negative and not prev.has_negative):
                    index[image_id] = ImageAnno(
                        has_negative=has_negative,
                        negatives=list(negatives),
                        sample_index=i,
                        source_path=p,
                    )
    return index, neg_lex


def iter_pass_groups(group_root: Path) -> List[Path]:
    """Return list of pass group directories under group_root.

    Supports both forms:
      - group_root = <mission_dir> containing '审核通过/<group_id>'
      - group_root = higher root; we recursively find all '审核通过/<group_id>'
    """
    out: List[Path] = []
    if not group_root.exists():
        return out
    # Case 1: group_root itself has 审核通过
    direct_pass = group_root / "审核通过"
    if direct_pass.is_dir():
        for group_dir in direct_pass.iterdir():
            if group_dir.is_dir():
                out.append(group_dir)
        return out
    # Case 2: recursively search for 审核通过
    for pass_dir in group_root.glob("**/审核通过"):
        if pass_dir.is_dir():
            for group_dir in pass_dir.iterdir():
                if group_dir.is_dir():
                    out.append(group_dir)
    return out


def list_group_images(group_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    out: List[str] = []
    try:
        for p in group_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p.name)
    except Exception:
        pass
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_label_dir(group_dir: Path) -> Tuple[Optional[Path], Optional[Path], str, str]:
    """Return (pass_dir, fail_dir, mission, group_id) for a given group_dir path."""
    # group_dir: .../<mission>/审核通过/<group_id>
    parts = group_dir.parts
    if "审核通过" in parts:
        idx = parts.index("审核通过")
        mission = parts[idx - 1] if idx - 1 >= 0 else ""
        parent = group_dir.parent.parent if group_dir.parent.name == "审核通过" else group_dir.parent
        # Fail sibling could be under the same mission root
        fail_dir = group_dir.parent.parent / "审核不通过" if group_dir.parent.name == "审核通过" else None
        return group_dir.parent, fail_dir, mission, group_dir.name
    # Fallback: attempt to detect
    return None, None, "", group_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean pass groups conflicting with negative annotations")
    parser.add_argument("--all_samples", type=str, default="/data3/Qwen2.5-VL-main/data/ds_v2_full/all_samples.jsonl")
    parser.add_argument("--group_root", type=str, default="/data3/Qwen2.5-VL-main/group_data/bbu_scene_2.0_order")
    parser.add_argument("--output", type=str, default="/data3/Qwen2.5-VL-main/data/cleaning_reports/group_conflicts.jsonl")
    parser.add_argument("--include_label_unreadable", action="store_true", help="Treat 标签/无法识别 as negative")
    parser.add_argument("--strict", action="store_true", help="Use strict lexicon only (no extended negatives)")
    parser.add_argument("--move_conflicts", action="store_true", help="Move conflicting pass groups to 审核不通过")
    parser.add_argument("--dry_run", action="store_true", help="Dry-run mode (do not move)", default=True)
    args = parser.parse_args()

    all_samples = Path(args.all_samples)
    group_root = Path(args.group_root)
    output_path = Path(args.output)

    ensure_dir(output_path.parent)

    index, neg_lex = load_all_samples_index(
        all_samples,
        include_label_unreadable=bool(args.include_label_unreadable),
        extended_negatives=(not bool(args.strict)),
    )

    pass_groups = iter_pass_groups(group_root)

    total_groups = 0
    conflict_groups = 0
    total_overlap_images = 0
    total_offending_images = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for gdir in sorted(pass_groups):
            total_groups += 1
            files = list_group_images(gdir)
            if not files:
                continue
            overlap = []
            offending = []
            for fname in files:
                info = index.get(fname.lower())
                if info is not None:
                    overlap.append(fname)
                    if info.has_negative:
                        offending.append({
                            "image": fname,
                            "negatives": info.negatives,
                            "sample_index": info.sample_index,
                            "source_path": info.source_path,
                        })
            if offending:
                conflict_groups += 1
                total_overlap_images += len(overlap)
                total_offending_images += len(offending)
                pass_dir, fail_dir, mission, group_id = find_label_dir(gdir)
                rec = {
                    "group_dir": str(gdir),
                    "mission": mission,
                    "label_dir": "审核通过",
                    "num_images": len(files),
                    "overlap_images": len(overlap),
                    "offending_images": offending,
                    "neg_lexicon_size": len(neg_lex),
                }
                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                writer.flush()

                # Optional move
                if bool(args.move_conflicts) and not bool(args.dry_run):
                    try:
                        if fail_dir is None:
                            # Try to find sibling 审核不通过 under the same mission root
                            mission_root = gdir.parent.parent
                            fail_dir = mission_root / "审核不通过"
                        ensure_dir(fail_dir)
                        dest = fail_dir / group_id
                        if dest.exists():
                            # Avoid collision
                            k = 1
                            while (fail_dir / f"{group_id}_conflict{k}").exists():
                                k += 1
                            dest = fail_dir / f"{group_id}_conflict{k}"
                        shutil.move(str(gdir), str(dest))
                    except Exception as e:
                        err_rec = {"error": f"move_failed: {e}", "group_dir": str(gdir)}
                        writer.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                        writer.flush()

    summary = {
        "total_pass_groups": total_groups,
        "conflict_groups": conflict_groups,
        "overlap_images_total": total_overlap_images,
        "offending_images_total": total_offending_images,
        "output": str(output_path),
        "group_root": str(group_root),
        "all_samples": str(all_samples),
        "include_label_unreadable": bool(args.include_label_unreadable),
        "strict": bool(args.strict),
        "move_conflicts": bool(args.move_conflicts),
        "dry_run": bool(args.dry_run),
    }
    # Print concise summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
