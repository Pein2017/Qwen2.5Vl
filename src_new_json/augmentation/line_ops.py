from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from src_new_json.config.augmentation_config import LineAugConfig


def _length(pts: List[Tuple[int, int]]) -> float:
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _resample_equidistant(pts: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    if len(pts) < 2 or n <= 2:
        return pts
    # Compute cumulative lengths
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(
            dists[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        )
    total = dists[-1]
    if total <= 1e-6:
        return [pts[0]] * n
    targets = [i * total / (n - 1) for i in range(n)]
    out: List[Tuple[int, int]] = []
    j = 0
    for t in targets:
        while j < len(dists) - 1 and dists[j + 1] < t:
            j += 1
        if j >= len(dists) - 1:
            out.append(pts[-1])
            continue
        # linear interpolate between pts[j] and pts[j+1]
        ratio = (
            0.0
            if dists[j + 1] == dists[j]
            else (t - dists[j]) / (dists[j + 1] - dists[j])
        )
        x = int(round(pts[j][0] + ratio * (pts[j + 1][0] - pts[j][0])))
        y = int(round(pts[j][1] + ratio * (pts[j + 1][1] - pts[j][1])))
        out.append((x, y))
    return out


def apply_line_ops(
    sample: Dict[str, Any], cfg: LineAugConfig, rng: random.Random
) -> Dict[str, Any]:
    objects = sample["objects"]
    out_objects: List[Dict[str, Any]] = []
    jit_min, jit_max = cfg.jitter_px_minmax
    # Clamp helpers for consistency with preprocessing pipeline
    width = int(sample["width"])
    height = int(sample["height"])

    def _clamp(x: int, y: int) -> Tuple[int, int]:
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    for o in objects:
        if "line" not in o:
            out_objects.append(o)
            continue
        l = o["line"]
        pts = [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
        if _length(pts) < cfg.min_length_px:
            out_objects.append(o)
            continue
        # jitter
        jittered: List[Tuple[int, int]] = []
        amp = rng.uniform(jit_min, jit_max)
        for x, y in pts:
            jx = int(round(x + rng.uniform(-amp, amp)))
            jy = int(round(y + rng.uniform(-amp, amp)))
            jx, jy = _clamp(jx, jy)
            jittered.append((jx, jy))
        # resample
        resampled = _resample_equidistant(jittered, cfg.resample_points)
        # clamp resampled too (numerical safety)
        resampled = [_clamp(x, y) for (x, y) in resampled]
        flat: List[int] = []
        for x, y in resampled:
            flat.extend([x, y])
        out_objects.append({"line": flat, "desc": o["desc"]})
    out = dict(sample)
    out["objects"] = out_objects
    return out
