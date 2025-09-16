#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

from src_new_json.config.config import load_config
from src_new_json.data.dataset import Dataset
# transformers imports are optional for this test (we don't need a tokenizer/processor)
# from transformers import AutoTokenizer, Qwen2VLImageProcessor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--samples", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    # Minimal components (no HF downloads needed for pairing test)
    tokenizer = None
    image_processor = None

    # Datasets
    train_ds = Dataset(
        data_path=cfg.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=None,
        config=cfg,
    )
    val_ds = Dataset(
        data_path=cfg.val_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=None,
        config=cfg,
    )

    print(f"Loaded train={len(train_ds)} val={len(val_ds)}")

    # Run epochs and sample
    for e in range(args.epochs):
        train_ds.set_epoch(e)
        # Collect teacher assignment stats over a subset
        num_with_teacher = 0
        same_bucket = 0
        checked = 0
        for i in range(min(args.samples, len(train_ds))):
            spec = getattr(train_ds, "_episode_map", {})
            ep = spec.get(i) if isinstance(spec, dict) else None
            if ep and ep.get("context_idx") is not None:
                num_with_teacher += 1
                cidx = int(ep["context_idx"])  # type: ignore[index]
                # Bucket check
                from src_new_json.sampling.bucketed_sampling import BucketedSamplingEngine
                mt_i = BucketedSamplingEngine._major_type_of(train_ds.samples[i])
                mt_j = BucketedSamplingEngine._major_type_of(train_ds.samples[cidx])
                if mt_i == mt_j:
                    same_bucket += 1
            checked += 1
        ratio = (num_with_teacher / max(1, checked)) * 100.0
        same_bucket_pct = (same_bucket / max(1, num_with_teacher)) * 100.0 if num_with_teacher else 0.0
        print(f"Epoch {e}: paired={num_with_teacher}/{checked} ({ratio:.1f}%), same_bucket={same_bucket_pct:.1f}%")

        # Fetch a few structured samples (without processor) to ensure teacher injection works
        for j in range(min(3, len(train_ds))):
            structured = train_ds._create_structured_sample(train_ds.samples[j], j)
            has_teachers = len(structured.get("teacher_samples", [])) > 0
            print(f"  idx={j} has_teachers={has_teachers}")

    # Validation dataset should have no teachers
    val_ds.set_epoch(0)
    spec = getattr(val_ds, "_episode_map", {})
    has_any = any(v.get("context_idx") is not None for v in spec.values()) if isinstance(spec, dict) else False
    print(f"Val pairing present? {has_any} (expected False)")


if __name__ == "__main__":
    main()
