#!/usr/bin/env python3
"""
Analyze JSONL generation results for duplication within prompts.

Each JSONL row is expected to contain at least the following fields:
  - step: int
  - prompt_idx: int
  - completion_idx: int
  - prediction_text_raw: str
  - ground_truth_text: str (optional for this analysis)
  - gen_len_tokenizer: int (optional)
  - gt_len_tokenizer: int (optional)
  - terminated_with_eos: bool (optional)
  - truncated: bool (optional)

The script groups rows by (step, prompt_idx) and reports duplicate ratios
across completions for each prompt, both exact-string and canonicalized
(geometry blocks removed, whitespace normalized).

Usage:
  python scripts/analyze_gen_duplicates.py \
      --input /path/to/gen_text.jsonl \
      --top-k 15 \
      --output-csv /tmp/dup_report.csv

Exit code 0 on success; non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


GEOM_PATTERNS = [
    r"<\|quad_start\|>\s*\[.*?\]\s*<\|quad_end\|>",
    r"<\|box_start\|>\s*\[.*?\]\s*<\|box_end\|>",
    r"<\|line_start\|>\s*\[.*?\]\s*<\|line_end\|>",
]

# Re-add canonicalize_text (was accidentally removed)


def canonicalize_text(
    text: str,
    *,
    strip_geometry: bool = True,
    strip_im_end: bool = True,
    collapse_ws: bool = True,
) -> str:
    """Return a canonical form to compare completions ignoring geometry blocks.

    - Optionally removes geometry coordinate segments wrapped by known wrappers
    - Removes <|im_end|> token
    - Collapses whitespace and trims
    """
    if not isinstance(text, str):
        return ""
    s = text
    if strip_geometry:
        for pat in GEOM_PATTERNS:
            s = re.sub(pat, " [GEOM] ", s, flags=re.DOTALL)
        # collapse consecutive [GEOM]
        s = re.sub(r"(?:\[GEOM\]\s*)+", " [GEOM] ", s)
    if strip_im_end:
        s = s.replace("<|im_end|>", " ")
    if collapse_ws:
        s = re.sub(r"\s+", " ", s, flags=re.MULTILINE).strip()
    return s


# ===== New: geometry parsing and IoU analysis =====

_OBJ_GEOM_RE = re.compile(
    r"<\|object_ref_start\|>(?P<ref>.*?)<\|object_ref_end\|>\s*"
    r"(?:(?:<\|box_start\|>\s*\[(?P<box>[^\]]+)\]\s*<\|box_end\|>)|"
    r"(?:<\|quad_start\|>\s*\[(?P<quad>[^\]]+)\]\s*<\|quad_end\|>)|"
    r"(?:<\|line_start\|>\s*\[(?P<line>[^\]]+)\]\s*<\|line_end\|>))",
    flags=re.DOTALL,
)


def _parse_ints(csv_like: str) -> List[int]:
    try:
        return [
            int(x.strip()) for x in csv_like.replace("\n", " ").split(",") if x.strip()
        ]
    except Exception:
        out: List[int] = []
        cur = re.findall(r"-?\d+", csv_like)
        for t in cur:
            try:
                out.append(int(t))
            except Exception:
                pass
        return out


def parse_objects(text: str) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text:
        return objs
    for m in _OBJ_GEOM_RE.finditer(text):
        ref = str(m.group("ref")).strip()
        if m.group("box") is not None:
            coords = _parse_ints(m.group("box"))
            typ = "box"
        elif m.group("quad") is not None:
            coords = _parse_ints(m.group("quad"))
            typ = "quad"
        elif m.group("line") is not None:
            coords = _parse_ints(m.group("line"))
            typ = "line"
        else:
            coords = []
            typ = "unknown"
        objs.append({"ref": ref, "type": typ, "coords": coords})
    return objs


def _aabb_from_item(item: Dict[str, Any]):
    coords = item.get("coords") or []
    if not isinstance(coords, list) or len(coords) < 4:
        return None
    typ = str(item.get("type"))
    if typ == "box" and len(coords) >= 4:
        x1, y1, x2, y2 = coords[:4]
        # normalize ordering
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    elif typ == "quad" and len(coords) >= 8:
        xs = coords[0::2][:4]
        ys = coords[1::2][:4]
        return (min(xs), min(ys), max(xs), max(ys))
    else:
        # For lines or unknown, approximate by bounding box if possible
        xs = coords[0::2]
        ys = coords[1::2]
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
        return None


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def pair_and_score_geometry(
    pred_objs: List[Dict[str, Any]], gt_objs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # Group by ref string
    pred_by_ref: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gt_by_ref: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in pred_objs:
        pred_by_ref[str(o.get("ref", ""))].append(o)
    for o in gt_objs:
        gt_by_ref[str(o.get("ref", ""))].append(o)

    total_gt = sum(len(v) for v in gt_by_ref.values())
    total_pred = sum(len(v) for v in pred_by_ref.values())
    matched = 0
    ious: List[float] = []
    type_matches = 0

    # Greedy per-ref pairing by IoU using AABBs
    for ref, gt_list in gt_by_ref.items():
        preds = list(pred_by_ref.get(ref, []))
        used = [False] * len(preds)
        for g in gt_list:
            best_iou = 0.0
            best_j = -1
            g_box = _aabb_from_item(g)
            for j, p in enumerate(preds):
                if used[j]:
                    continue
                p_box = _aabb_from_item(p)
                if g_box is None or p_box is None:
                    cur = 0.0
                else:
                    cur = _box_iou(g_box, p_box)
                if cur > best_iou:
                    best_iou = cur
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
                matched += 1
                ious.append(best_iou)
                if str(gt_list[0].get("type")) == str(preds[best_j].get("type")):
                    type_matches += 1

    coverage = float(matched) / float(max(total_gt, 1))
    overgen = max(0, total_pred - total_gt) / float(max(total_gt, 1))
    avg_iou = float(mean(ious)) if ious else 0.0
    type_match_ratio = float(type_matches) / float(max(matched, 1))

    return {
        "num_gt": int(total_gt),
        "num_pred": int(total_pred),
        "matched": int(matched),
        "coverage": round(coverage, 4),
        "overgen_ratio": round(overgen, 4),
        "avg_iou": round(avg_iou, 4),
        "type_match_ratio": round(type_match_ratio, 4),
    }


def analyze_geometry(
    rows: Iterable[Dict[str, Any]],
    *,
    min_completions: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            key = (int(r.get("step")), int(r.get("prompt_idx")))
        except Exception:
            continue
        groups[key].append(r)

    per_group: List[Dict[str, Any]] = []
    covers: List[float] = []
    ious: List[float] = []
    types_ok: List[float] = []
    overgens: List[float] = []

    for (step, pidx), lst in groups.items():
        if len(lst) < min_completions:
            continue
        # Compare the first completion vs GT per prompt by default
        r0 = lst[0]
        pred = str(r0.get("prediction_text_raw", ""))
        gt = str(r0.get("ground_truth_text", ""))
        if not gt:
            continue
        pred_objs = parse_objects(pred)
        gt_objs = parse_objects(gt)
        metrics = pair_and_score_geometry(pred_objs, gt_objs)
        metrics.update({"step": step, "prompt_idx": pidx})
        per_group.append(metrics)
        covers.append(metrics["coverage"])
        ious.append(metrics["avg_iou"])
        types_ok.append(metrics["type_match_ratio"])
        overgens.append(metrics["overgen_ratio"])

    summary = {
        "groups": len(per_group),
        "avg_coverage": float(mean(covers)) if covers else 0.0,
        "avg_iou": float(mean(ious)) if ious else 0.0,
        "avg_type_match_ratio": float(mean(types_ok)) if types_ok else 0.0,
        "avg_overgen_ratio": float(mean(overgens)) if overgens else 0.0,
    }
    return per_group, summary


def write_geom_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    header = [
        "step",
        "prompt_idx",
        "num_gt",
        "num_pred",
        "matched",
        "coverage",
        "overgen_ratio",
        "avg_iou",
        "type_match_ratio",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [r.get(k, "") for k in header]
            vals = [str(v).replace("\n", " ").replace(",", " ") for v in vals]
            f.write(",".join(vals) + "\n")


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode failed at line {ln}", file=sys.stderr)
                continue


def quantiles(values: List[float], qs: List[float]) -> List[float]:
    if not values:
        return [float("nan") for _ in qs]
    arr = sorted(values)
    out = []
    for q in qs:
        if q <= 0:
            out.append(arr[0])
        elif q >= 1:
            out.append(arr[-1])
        else:
            idx = q * (len(arr) - 1)
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                out.append(arr[lo])
            else:
                w = idx - lo
                out.append(arr[lo] * (1 - w) + arr[hi] * w)
    return out


def analyze_groups(
    rows: Iterable[Dict[str, Any]],
    *,
    strip_geometry: bool,
    min_completions: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            key = (int(r.get("step")), int(r.get("prompt_idx")))
        except Exception:
            # skip malformed rows
            continue
        groups[key].append(r)

    per_group: List[Dict[str, Any]] = []
    dup_exact_rates: List[float] = []
    dup_canon_rates: List[float] = []
    term_fracs: List[float] = []
    trunc_fracs: List[float] = []
    gen_len_means: List[float] = []
    gen_len_ratios: List[float] = []

    for (step, pidx), lst in groups.items():
        if len(lst) < min_completions:
            continue
        texts = [str(x.get("prediction_text_raw", "")) for x in lst]
        canons = [canonicalize_text(t, strip_geometry=strip_geometry) for t in texts]
        uniq_exact = len(set(texts))
        uniq_canon = len(set(canons))
        n = len(lst)
        dup_exact = 1.0 - (uniq_exact / float(max(n, 1)))
        dup_canon = 1.0 - (uniq_canon / float(max(n, 1)))

        terminated = [bool(x.get("terminated_with_eos", False)) for x in lst]
        truncated = [bool(x.get("truncated", False)) for x in lst]
        term_frac = sum(terminated) / float(n)
        trunc_frac = sum(truncated) / float(n)

        glens = [int(x.get("gen_len_tokenizer", 0)) for x in lst]
        gtlens = [
            int(x.get("gt_len_tokenizer", 0))
            for x in lst
            if int(x.get("gt_len_tokenizer", 0)) > 0
        ]
        glen_mean = float(mean(glens)) if glens else 0.0
        glen_med = float(median(glens)) if glens else 0.0
        ratio_mean = (
            float(glen_mean / mean(gtlens)) if glens and gtlens else float("nan")
        )

        # Most frequent canonical completion (top mode)
        canon_counts = Counter(canons)
        top_canon, top_c_count = canon_counts.most_common(1)[0]

        per_group.append(
            {
                "step": step,
                "prompt_idx": pidx,
                "num_completions": n,
                "dup_exact": round(dup_exact, 4),
                "dup_canon": round(dup_canon, 4),
                "terminated_frac": round(term_frac, 4),
                "truncated_frac": round(trunc_frac, 4),
                "gen_len_mean": round(glen_mean, 2),
                "gen_len_median": round(glen_med, 2),
                "gen_len_vs_gt_mean": round(ratio_mean, 3)
                if not math.isnan(ratio_mean)
                else float("nan"),
                "top_canon_sample": top_canon[:200],
                "top_canon_count": int(top_c_count),
            }
        )

        dup_exact_rates.append(dup_exact)
        dup_canon_rates.append(dup_canon)
        term_fracs.append(term_frac)
        trunc_fracs.append(trunc_frac)
        if not math.isnan(ratio_mean):
            gen_len_ratios.append(ratio_mean)
        if glens:
            gen_len_means.append(glen_mean)

    summary = {
        "groups": len(per_group),
        "avg_dup_exact": float(mean(dup_exact_rates)) if dup_exact_rates else 0.0,
        "avg_dup_canon": float(mean(dup_canon_rates)) if dup_canon_rates else 0.0,
        "p90_dup_exact": float(quantiles(dup_exact_rates, [0.9])[0])
        if dup_exact_rates
        else 0.0,
        "p90_dup_canon": float(quantiles(dup_canon_rates, [0.9])[0])
        if dup_canon_rates
        else 0.0,
        "avg_terminated_frac": float(mean(term_fracs)) if term_fracs else 0.0,
        "avg_truncated_frac": float(mean(trunc_fracs)) if trunc_fracs else 0.0,
        "avg_gen_len": float(mean(gen_len_means)) if gen_len_means else 0.0,
        "avg_len_ratio": float(mean(gen_len_ratios))
        if gen_len_ratios
        else float("nan"),
    }

    return per_group, summary


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                "step,prompt_idx,num_completions,dup_exact,dup_canon,terminated_frac,truncated_frac,gen_len_mean,gen_len_median,gen_len_vs_gt_mean,top_canon_count,top_canon_sample\n"
            )
        return
    # Stable header order
    header = [
        "step",
        "prompt_idx",
        "num_completions",
        "dup_exact",
        "dup_canon",
        "terminated_frac",
        "truncated_frac",
        "gen_len_mean",
        "gen_len_median",
        "gen_len_vs_gt_mean",
        "top_canon_count",
        "top_canon_sample",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [r.get(k, "") for k in header]
            # Escape commas and newlines in sample
            vals = [str(v).replace("\n", " ").replace(",", " ") for v in vals]
            f.write(",".join(vals) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze generation duplicates per prompt"
    )
    parser.add_argument("--input", required=True, help="Path to gen_text.jsonl")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for per-group duplication report",
    )
    parser.add_argument(
        "--output-geom-csv",
        default=None,
        help="Optional CSV path for per-group geometry report",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of worst groups to print"
    )
    parser.add_argument(
        "--no-canon",
        action="store_true",
        help="Do not canonicalize (keep geometry blocks) for duplication calc",
    )
    parser.add_argument(
        "--no-geom",
        action="store_true",
        help="Disable geometry/IoU analysis",
    )
    parser.add_argument(
        "--min-completions",
        type=int,
        default=2,
        help="Skip groups with fewer completions",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERR] Input file not found: {args.input}", file=sys.stderr)
        return 2

    rows = list(load_jsonl(args.input))
    per_group, summary = analyze_groups(
        rows,
        strip_geometry=(not args.no_canon),
        min_completions=max(1, int(args.min_completions)),
    )

    # Sort by canonical duplication descending, then exact
    per_group_sorted = sorted(
        per_group,
        key=lambda r: (r["dup_canon"], r["dup_exact"], -r["num_completions"]),
        reverse=True,
    )

    print("==== Summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    top_k = max(0, int(args.top_k))
    if top_k > 0 and per_group_sorted:
        print("\n==== Worst groups (by dup_canon) ====")
        for r in per_group_sorted[:top_k]:
            print(
                f"step={r['step']} prompt_idx={r['prompt_idx']} n={r['num_completions']} "
                f"dup_canon={r['dup_canon']:.3f} dup_exact={r['dup_exact']:.3f} "
                f"term={r['terminated_frac']:.3f} trunc={r['truncated_frac']:.3f} "
                f"len_mean={r['gen_len_mean']:.1f} len_med={r['gen_len_median']:.1f} "
                f"len_vs_gt_mean={r['gen_len_vs_gt_mean']} "
                f"top_canon_count={r['top_canon_count']}\n  sample: {r['top_canon_sample']}\n"
            )

    if args.output_csv:
        try:
            write_csv(per_group_sorted, args.output_csv)
            print(f"[OK] Wrote per-group duplication CSV to {args.output_csv}")
        except Exception as exc:
            print(f"[WARN] Failed to write duplication CSV: {exc}", file=sys.stderr)

    # Optional geometry analysis
    has_gt = any(
        isinstance(r.get("ground_truth_text"), str) and r.get("ground_truth_text")
        for r in rows
    )
    if not args.no_geom and has_gt:
        geom_rows, geom_summary = analyze_geometry(
            rows, min_completions=max(1, int(args.min_completions))
        )
        print("\n==== Geometry Summary ====")
        for k, v in geom_summary.items():
            print(f"{k}: {v}")
        # Show worst coverage groups
        if top_k > 0 and geom_rows:
            geom_sorted = sorted(geom_rows, key=lambda r: (r["coverage"], r["avg_iou"]))
            print("\n==== Worst groups (by coverage, then IoU) ====")
            for r in geom_sorted[:top_k]:
                print(
                    f"step={r['step']} prompt_idx={r['prompt_idx']} "
                    f"gt={r['num_gt']} pred={r['num_pred']} matched={r['matched']} "
                    f"coverage={r['coverage']:.3f} overgen={r['overgen_ratio']:.3f} "
                    f"avg_iou={r['avg_iou']:.3f} type_match={r['type_match_ratio']:.3f}"
                )
        if args.output_geom_csv:
            try:
                write_geom_csv(geom_rows, args.output_geom_csv)
                print(f"[OK] Wrote per-group geometry CSV to {args.output_geom_csv}")
            except Exception as exc:
                print(f"[WARN] Failed to write geometry CSV: {exc}", file=sys.stderr)

    # Heuristics/warnings for likely causes of duplication
    try:
        if summary.get("avg_dup_canon", 0.0) >= 0.5:
            print(
                "\n[Hint] High duplication: consider increasing temperature/top_p and enabling repetition_penalty."
            )
        if summary.get("avg_terminated_frac", 1.0) < 0.7:
            print(
                "[Hint] Low EOS termination: reduce max_new_tokens, enable dynamic_length, and set mask_truncated_completions=true."
            )
        if (
            not math.isnan(summary.get("avg_len_ratio", float("nan")))
            and summary["avg_len_ratio"] > 1.5
        ):
            print(
                "[Hint] Generations much longer than GT: tighten dynamic caps (alpha≈1.1, eos_margin≈24)."
            )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
