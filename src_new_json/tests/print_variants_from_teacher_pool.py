#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLProcessor

from src_new_json.losses.token_grouping import TokenGroupingPlugin
from src_new_json.processing.conversation_processor import ConversationProcessor
from src_new_json.processing.span_extraction import find_assistant_spans
from src_new_json.processing.special_tokens import IM_END, IM_START, IMAGE_PAD
from src_new_json.types import ConversationVariant
from src_new_json.utils.path_manager import create_path_manager


def _load_samples(jsonl_path: str, limit: int, offset: int = 0) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip malformed JSON at line {i}: {e}")
    return samples


def _load_images_for_sample(sample: Dict[str, Any], data_root: str) -> List[Image.Image]:
    pm = create_path_manager(data_root)
    imgs: List[Image.Image] = []
    for rel in sample.get("images", [])[:1]:  # single-image conversations in this dataset
        p = pm.resolve_path(rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    if not imgs:
        raise RuntimeError("Sample contains no images")
    return imgs


def _make_random_image(width: int = 28, height: int = 28) -> Image.Image:
    arr = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path() -> Path:
    return _repo_root() / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"


def _default_data_root() -> Path:
    return _repo_root() / "data/ds_v2_full"


@dataclass(frozen=True)
class Config:
    jsonl: str
    data_root: str
    model: str
    limit: int = 2
    offset: int = 0
    random_image: bool = False
    random_size: int = 28


CONFIG = Config(
    jsonl=str(_default_data_root() / "teacher_pool.jsonl"),
    data_root=str(_default_data_root()),
    model=str(_default_model_path()),
    limit=2,
    offset=0,
    random_image=False,
    random_size=28,
)


def main() -> None:
    model_path = Path(CONFIG.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please place Qwen2.5-VL-3B-Instruct processor there."
        )

    jsonl_path = Path(CONFIG.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    data_root = str(CONFIG.data_root)

    processor = Qwen2VLProcessor.from_pretrained(str(model_path))
    conv = ConversationProcessor(processor=processor)
    tokenizer = conv.processor.tokenizer

    samples = _load_samples(str(jsonl_path), limit=CONFIG.limit, offset=CONFIG.offset)
    if not samples:
        print("No samples loaded. Exiting.")
        return

    variants: List[Tuple[str, ConversationVariant]] = [
        ("DENSE_CAPTION", ConversationVariant.DENSE_CAPTION),
        ("COORDS_TO_DESC", ConversationVariant.COORDS_TO_DESC),
        ("DESC_TO_COORDS", ConversationVariant.DESC_TO_COORDS),
        ("SUMMARY", ConversationVariant.SUMMARY),
    ]

    for idx, s in enumerate(samples):
        print("=" * 120)
        print(f"Sample #{idx + CONFIG.offset} images={s.get('images', [])}")
        try:
            if CONFIG.random_image:
                imgs = [_make_random_image(CONFIG.random_size, CONFIG.random_size)]
                print(f"[INFO] Using random image of size {CONFIG.random_size}x{CONFIG.random_size}")
            else:
                imgs = _load_images_for_sample(s, data_root)
        except Exception as e:
            print(f"[ERROR] Failed to prepare image for sample {idx}: {e}")
            continue

        for name, variant in variants:
            try:
                outputs = conv.create_conversation(sample=s, images=imgs, variant=variant)
                text = outputs.get("decoded_text") or outputs.get("conversation_text", "")
            except Exception as e:
                print(f"--- {name} ---")
                print(f"[ERROR] Failed to build conversation: {type(e).__name__}: {e}")
                continue

            print(f"--- {name} ---")
            print(text)
            # Print token counts to help verify placeholders and structure
            im_start_count = text.count(IM_START)
            im_end_count = text.count(IM_END)
            img_pad_count = text.count(IMAGE_PAD)
            print(f"[COUNTS] {IM_START}={im_start_count}, {IM_END}={im_end_count}, {IMAGE_PAD}={img_pad_count}")

            # ===== Span + Grouped-token visualization (assistant-only) =====
            # Build 1D tensors
            input_ids_1d = outputs.get("input_ids")
            if not isinstance(input_ids_1d, torch.Tensor):
                print("[WARN] input_ids missing; skip spans/group masks")
                continue
            if input_ids_1d.dim() != 1:
                print("[WARN] Unexpected input_ids dim; skip spans/group masks")
                continue
            offsets = outputs.get("offset_mapping")
            if not isinstance(offsets, torch.Tensor):
                print("[WARN] offset_mapping missing; skip spans/group masks")
                continue

            # Extract assistant spans (single sample) with EOS inclusion
            spans = find_assistant_spans(
                full_text=text,
                offset_mapping=offsets,
                tokenizer=tokenizer,
                include_eos=True,
                has_teachers=False,
                input_ids_1d=input_ids_1d,
                num_teachers=0,
            )
            # Split to teacher/student span lists
            teacher_spans: List[Tuple[int, int]] = [(st, en) for (st, en, is_t) in spans if is_t]
            student_spans: List[Tuple[int, int]] = [(st, en) for (st, en, is_t) in spans if not is_t]

            # Build labels: -100 outside assistant spans
            labels_1d = input_ids_1d.clone()
            labels_1d[:] = -100
            for st, en in teacher_spans + student_spans:
                labels_1d[st:en] = input_ids_1d[st:en]

            # Prepare 2D tensors for grouping plugin
            labels_2d = labels_1d.unsqueeze(0)
            input_ids_2d = input_ids_1d.unsqueeze(0)
            teacher_spans_batched = [teacher_spans]
            student_spans_batched = [student_spans]

            # Group masks (caption/grounding/formatting), aligned to shifted CE
            plugin = TokenGroupingPlugin(tokenizer)
            variant_key = str(getattr(variant, "value", variant)).strip().lower()
            gm = plugin.build_group_masks(
                labels=labels_2d,
                teacher_spans=teacher_spans_batched if teacher_spans_batched[0] else None,
                student_spans=student_spans_batched if student_spans_batched[0] else None,
                input_ids=input_ids_2d,
                variant_key=variant_key,
            )

            # Helper: convert shifted masks to original token indices (add 1)
            def _idx_set(mask_2d: torch.Tensor) -> List[int]:
                if mask_2d is None:
                    return []
                arr = mask_2d[0].nonzero(as_tuple=False).view(-1).tolist()
                return [i + 1 for i in arr]

            s_cap_idx = set(_idx_set(gm.student_caption))
            s_grd_idx = set(_idx_set(gm.student_grounding))
            s_fmt_idx = set(_idx_set(gm.student_formatting))
            t_cap_idx = set(_idx_set(gm.teacher_caption))
            t_grd_idx = set(_idx_set(gm.teacher_grounding))
            t_fmt_idx = set(_idx_set(gm.teacher_formatting))

            # Print span ranges
            print(f"[ASSISTANT_SPANS] teacher={teacher_spans} student={student_spans}")

            # Unified token dump inside assistant regions only
            ids_list = [int(x) for x in input_ids_1d.tolist()]
            toks = tokenizer.convert_ids_to_tokens(ids_list)
            offs = offsets.tolist()

            def _in_any_span(i: int, spans_list: List[Tuple[int, int]]) -> bool:
                for st, en in spans_list:
                    if st <= i < en:
                        return True
                return False

            print("[TOKENS role=student|teacher, group=caption|grounding|formatting] (index id token text)")
            for i in range(len(ids_list)):
                is_t = _in_any_span(i, teacher_spans)
                is_s = _in_any_span(i, student_spans)
                if not (is_t or is_s):
                    continue
                sidx = i  # original index
                # Determine group from shifted masks (index-1)
                tag = "-"
                if is_t:
                    if sidx in t_cap_idx:
                        tag = "caption"
                    elif sidx in t_grd_idx:
                        tag = "grounding"
                    elif sidx in t_fmt_idx:
                        tag = "formatting"
                else:
                    if sidx in s_cap_idx:
                        tag = "caption"
                    elif sidx in s_grd_idx:
                        tag = "grounding"
                    elif sidx in s_fmt_idx:
                        tag = "formatting"
                role = "teacher" if is_t else "student"
                ch0, ch1 = offs[i]
                snippet = text[ch0:ch1].replace("\n", " ⏎ ")
                print(f"  i={i:>4} role={role:<7} group={tag:<10} id={ids_list[i]:>6} tok={toks[i]!s} text='{snippet}'")

            # Group counts summary
            def _cnt(x: set[int]) -> int:
                return len(x)
            print(
                "[GROUP_COUNTS] student(c,g,f)=(%d,%d,%d) | teacher(c,g,f)=(%d,%d,%d)" % (
                    _cnt(s_cap_idx), _cnt(s_grd_idx), _cnt(s_fmt_idx),
                    _cnt(t_cap_idx), _cnt(t_grd_idx), _cnt(t_fmt_idx),
                )
            )

    print("=" * 120)
    print("Done.")


if __name__ == "__main__":
    main()
