from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.eval_utils")


def dump_samples(
    rows: Iterable[Dict[str, Any]],
    *,
    output_file: str | None,
    output_dir: str | None,
    step: int | None,
    limit: int | None,
) -> str | None:
    """Write sample rows to a JSONL path and return the path used.

    Preference order:
    1) Sidecar next to ``output_file`` (replace extension with ``.samples.jsonl``)
    2) ``output_dir/eval_samples.jsonl`` or ``output_dir/eval_samples/step-<step>.jsonl`` when ``step`` provided
    """
    try:
        rows_list: List[Dict[str, Any]] = list(rows)
        if not rows_list:
            return None
        if isinstance(limit, int) and limit > 0:
            rows_list = rows_list[:limit]

        # Sidecar path when an output_file is specified
        if isinstance(output_file, str) and len(output_file) > 0:
            base = output_file.rsplit(".", 1)[0]
            out_path = base + ".samples.jsonl"
        else:
            out_root = output_dir or "."
            os.makedirs(out_root, exist_ok=True)
            if isinstance(step, int) and step >= 0:
                sub = os.path.join(out_root, "eval_samples")
                os.makedirs(sub, exist_ok=True)
                out_path = os.path.join(sub, f"step-{step}.jsonl")
            else:
                out_path = os.path.join(out_root, "eval_samples.jsonl")

        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _LOGGER.info("Saved eval samples to %s", out_path)
        return out_path
    except Exception as exc:
        _LOGGER.warning("Failed to dump eval samples: %s", exc)
        return None


def objects_to_boxes(
    objs: List[Dict[str, Any]] | None,
) -> List[Tuple[int, int, int, int]]:
    """Convert mixed bbox/quad objects to axis-aligned boxes for fast metrics."""
    out: List[Tuple[int, int, int, int]] = []
    for o in objs or []:
        if not isinstance(o, dict):
            continue
        if "bbox_2d" in o and isinstance(o["bbox_2d"], list) and len(o["bbox_2d"]) >= 4:
            x1, y1, x2, y2 = o["bbox_2d"][:4]
            try:
                out.append((int(x1), int(y1), int(x2), int(y2)))
            except Exception:
                continue
        elif "quad" in o and isinstance(o["quad"], list) and len(o["quad"]) >= 8:
            q = o["quad"][:8]
            try:
                xs = [int(v) for v in q[0::2]]
                ys = [int(v) for v in q[1::2]]
                out.append((min(xs), min(ys), max(xs), max(ys)))
            except Exception:
                continue
    return out
