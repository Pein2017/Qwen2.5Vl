#!/usr/bin/env python3
"""
Create a teacher pool by selecting samples that together cover all labels in label_hierarchy.json
while ensuring diversity across sparse, medium, and dense scenes.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def flatten_hierarchy(hierarchy_path: str) -> set[str]:
    with open(hierarchy_path, encoding="utf-8") as f:
        entries = json.load(f)
    labels: set[str] = set()
    for entry in entries:
        obj_type = entry.get("object_type")
        if obj_type:
            labels.add(obj_type)
        for prop in entry.get("property", []):
            labels.add(prop)
    return labels


def load_samples(data_path: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        samples = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def sample_labels(sample: dict, universe: set[str]) -> set[str]:
    # Handle two possible object formats: dict with 'ref'/ 'bbox', or list of dicts
    target = sample.get("target", sample)
    objs = target.get("objects", sample.get("objects", {}))
    desc_list: list[str] = []
    # Format A: intermediate JSONL with {'ref': [...], 'bbox': [...]} mapping
    if isinstance(objs, dict) and "ref" in objs:
        # each entry is a description string
        ref_list = objs.get("ref", [])
        desc_list.extend([d for d in ref_list if isinstance(d, str)])
    # Format B: list of dicts with 'desc' or 'description'
    elif isinstance(objs, list):
        # elements may be strings or dicts
        for item in objs:
            if isinstance(item, str):
                desc_list.append(item)
            elif isinstance(item, dict):
                d = item.get("desc") or item.get("description")
                if isinstance(d, str):
                    desc_list.append(d)
    # Otherwise skip
    labels: set[str] = set()
    for desc in desc_list:
        for label in universe:
            if label in desc:
                labels.add(label)
    return labels


def bucket_for_sample(sample: dict) -> str:
    target = sample.get("target", sample)
    objs = target.get("objects", sample.get("objects", {}))
    # Determine object count
    if isinstance(objs, dict) and "ref" in objs:
        n = len(objs.get("ref", []))
    elif isinstance(objs, list):
        n = len(objs)
    else:
        n = 0
    if n <= 3:
        return "sparse"
    elif n <= 10:
        return "medium"
    else:
        return "dense"


def get_bboxes_and_dims(sample: dict):
    target = sample.get("target", sample)
    objs = target.get("objects", sample.get("objects", {}))
    # extract bbox_2d coordinates
    if isinstance(objs, dict) and "bbox" in objs:
        bboxes = objs.get("bbox", [])
    elif isinstance(objs, list):
        bboxes = [
            item["bbox_2d"]
            for item in objs
            if isinstance(item, dict) and "bbox_2d" in item
        ]
    else:
        bboxes = []
    # get image dims
    width = sample.get("width")
    height = sample.get("height")
    return bboxes, width, height


def compute_spatial_bucket(bboxes, width, height):
    if not bboxes or not width or not height:
        return "unknown"
    centers = []
    for bbox_2d in bboxes:
        if len(bbox_2d) != 4:
            continue
        x1, y1, x2, y2 = bbox_2d
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        centers.append((cx, cy))
    if not centers:
        return "unknown"
    avg_x = sum(cx for cx, cy in centers) / len(centers)
    avg_y = sum(cy for cx, cy in centers) / len(centers)
    h = "left" if avg_x < 1 / 3 else ("center" if avg_x < 2 / 3 else "right")
    v = "top" if avg_y < 1 / 3 else ("middle" if avg_y < 2 / 3 else "bottom")
    return f"{v}-{h}"


def compute_size_bucket(bboxes, width, height):
    if not bboxes or not width or not height:
        return "unknown"
    total_area = width * height
    fracs = []
    for bbox_2d in bboxes:
        if len(bbox_2d) != 4:
            continue
        x1, y1, x2, y2 = bbox_2d
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        fracs.append((w * h) / total_area)
    if not fracs:
        return "unknown"
    avg_frac = sum(fracs) / len(fracs)
    return "small" if avg_frac < 0.05 else ("medium" if avg_frac < 0.2 else "large")


# -----------------------------------------------------------------------------
# Helper to convert intermediate sample to clean format
# -----------------------------------------------------------------------------


def convert_to_clean_format(intermediate_sample: dict) -> dict:
    """Convert intermediate JSONL sample to clean semantic format.

    The intermediate format stores objects as a mapping::
        {
          "ref":  [desc1, desc2, ...],
          "bbox": [[x1,y1,x2,y2], ...]
        }

    This helper converts it to the clean format expected by the training
    pipeline::
        {
          "images": ["path.jpg"],
          "objects": [{"bbox_2d": [..], "desc": "..."}, ...]
        }
    """

    if "images" not in intermediate_sample:
        raise KeyError("Sample missing required 'images' key")
    if "objects" not in intermediate_sample:
        raise KeyError("Sample missing required 'objects' key")

    images = intermediate_sample["images"]
    objects_data = intermediate_sample["objects"]

    # Extract description and bbox lists (fail fast on invalid structure)
    if not isinstance(objects_data, dict):
        raise TypeError("'objects' must be a dict with 'ref' and 'bbox' keys")

    ref_list = objects_data.get("ref")
    bbox_list = objects_data.get("bbox")

    if ref_list is None or bbox_list is None:
        raise KeyError("Objects data missing 'ref' or 'bbox' keys")
    if len(ref_list) != len(bbox_list):
        raise ValueError("ref and bbox list length mismatch")

    clean_objects: list[dict] = []
    for ref_desc, bbox in zip(ref_list, bbox_list):
        # Chinese descriptions are already compact; English verbose ones use ';'
        if ";" in ref_desc:
            # Very lightweight parse: keep segments before first ';'
            # Full parsing would require ResponseFormatter, but to minimise
            # dependencies we extract the object_type/property parts.
            parts = [seg.strip() for seg in ref_desc.split(";") if seg.strip()]
            if parts:
                ref_desc = parts[0].split(":", 1)[-1].strip()
        clean_objects.append({"bbox_2d": bbox, "desc": ref_desc})

    return {"images": images, "objects": clean_objects}


def main():
    parser = argparse.ArgumentParser(
        description="Build a teacher pool covering all labels and scene diversity"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSONL dataset file (e.g. train.jsonl)",
    )
    parser.add_argument(
        "--hierarchy", type=str, required=True, help="Path to label_hierarchy.json"
    )
    parser.add_argument(
        "--max_teachers",
        type=int,
        default=10,
        help="Maximum number of teacher samples to select",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_conversion/teacher_pool.jsonl",
        help="Output JSONL file to save selected teacher samples (clean format)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for final pool fill"
    )
    args = parser.parse_args()

    universe = flatten_hierarchy(args.hierarchy)
    samples = load_samples(args.data_path)

    # Compute per-sample metadata (labels, count, spatial, size)
    info_list = []
    for idx, sample in enumerate(samples):
        labels = sample_labels(sample, universe)
        count_bucket = bucket_for_sample(sample)
        bboxes, width, height = get_bboxes_and_dims(sample)
        spatial_bucket = compute_spatial_bucket(bboxes, width, height)
        size_bucket = compute_size_bucket(bboxes, width, height)
        info_list.append(
            {
                "idx": idx,
                "labels": labels,
                "count_bucket": count_bucket,
                "spatial_bucket": spatial_bucket,
                "size_bucket": size_bucket,
            }
        )

    # Multi-objective greedy selection across labels, count, spatial, and size
    need_labels = set(universe)
    need_counts = set(item["count_bucket"] for item in info_list)
    need_spatial = set(
        item["spatial_bucket"]
        for item in info_list
        if item["spatial_bucket"] != "unknown"
    )
    need_sizes = set(
        item["size_bucket"] for item in info_list if item["size_bucket"] != "unknown"
    )
    teacher_indices: list[int] = []
    while (need_labels or need_counts or need_spatial or need_sizes) and len(
        teacher_indices
    ) < args.max_teachers:
        best_score = -1
        best_info = None
        for info in info_list:
            if info["idx"] in teacher_indices:
                continue
            gain_labels = len(info["labels"] & need_labels)
            gain_count = 1 if info["count_bucket"] in need_counts else 0
            gain_spatial = 1 if info["spatial_bucket"] in need_spatial else 0
            gain_size = 1 if info["size_bucket"] in need_sizes else 0
            score = 10 * gain_labels + 3 * gain_count + 2 * gain_spatial + 1 * gain_size
            if score > best_score:
                best_score = score
                best_info = info
        if not best_info or best_score <= 0:
            break
        teacher_indices.append(best_info["idx"])
        need_labels -= best_info["labels"]
        need_counts.discard(best_info["count_bucket"])
        need_spatial.discard(best_info["spatial_bucket"])
        need_sizes.discard(best_info["size_bucket"])

    # Phase C: random fill up to max_teachers for additional scene variety
    remaining_infos = [info for info in info_list if info["idx"] not in teacher_indices]
    random.seed(args.seed)
    while len(teacher_indices) < args.max_teachers and remaining_infos:
        choice = random.choice(remaining_infos)
        teacher_indices.append(choice["idx"])
        remaining_infos = [
            info for info in remaining_infos if info["idx"] != choice["idx"]
        ]

    # Prepare teacher samples in clean format: detect format on the fly
    teacher_samples_clean: list[dict] = []
    for idx in teacher_indices:
        sample = samples[idx]
        # If objects field is already a list (clean format), use directly
        if isinstance(sample.get("objects"), list):
            teacher_samples_clean.append(sample)
        else:
            teacher_samples_clean.append(convert_to_clean_format(sample))

    # Write each teacher sample as a JSON line
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ts in teacher_samples_clean:
            f.write(json.dumps(ts, ensure_ascii=False) + "\n")

    print(
        f"Selected {len(teacher_samples_clean)} teacher samples; saved to {args.output}"
    )


if __name__ == "__main__":
    main()
