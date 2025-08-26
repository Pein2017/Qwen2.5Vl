from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from src_new.augmentation.standardize import canonicalize_quad


def overlay_quads(
    image: Image.Image,
    objects: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for obj in objects:
        if "quad" in obj:
            q = obj["quad"]
            # Canonicalize ordering to clockwise starting from top-left
            pts = [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
            canon = canonicalize_quad(pts)
            cpts = [
                (canon[0], canon[1]),
                (canon[2], canon[3]),
                (canon[4], canon[5]),
                (canon[6], canon[7]),
                (canon[0], canon[1]),
            ]
            draw.line(cpts, fill=color, width=2)
        elif "bbox_2d" in obj:
            b = obj["bbox_2d"]
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
        elif "line" in obj:
            l = obj["line"]
            pts = [(l[i], l[i + 1]) for i in range(0, len(l), 2)]
            if len(pts) >= 2:
                draw.line(pts, fill=(255, 0, 0), width=2)
    return out
