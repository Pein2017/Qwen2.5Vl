from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter


def round_by_factor(number: int, factor: int) -> int:
	if factor <= 0:
		raise ValueError("factor must be positive")
	return int(round(number / factor) * factor)


def ceil_by_factor(number: int, factor: int) -> int:
	if factor <= 0:
		raise ValueError("factor must be positive")
	return int(math.ceil(number / factor) * factor)


def floor_by_factor(number: int, factor: int) -> int:
	if factor <= 0:
		raise ValueError("factor must be positive")
	return int(math.floor(number / factor) * factor)


@dataclass(frozen=True)
class CanvasTransform:
	"""Describes a rotation about the original image center and an optional translation.

	angle_rad: rotation angle in radians
	translate_xy: translation to apply after rotation (e.g., when expanding canvas)
	out_size_wh: (width, height) of the output canvas
	"""

	angle_rad: float
	translate_xy: Tuple[float, float]
	out_size_wh: Tuple[int, int]


def deg2rad(angle_deg: float) -> float:
	return angle_deg * math.pi / 180.0


def rotate_point_about_center(
	x: float, y: float, cx: float, cy: float, angle_rad: float
) -> Tuple[float, float]:
	cos_t = math.cos(angle_rad)
	sin_t = math.sin(angle_rad)
	dx = x - cx
	dy = y - cy
	rx = cos_t * dx - sin_t * dy
	ry = sin_t * dx + cos_t * dy
	return rx + cx, ry + cy


def compute_expanded_canvas_and_offset(
	width: int, height: int, angle_rad: float
) -> Tuple[int, int, float, float]:
	"""Compute expanded canvas size and translation offset after rotation.

	Uses floor/ceil bounds to better align with PIL.rotate(expand=True) behavior.
	Returns (out_w, out_h, offset_x, offset_y) where offset should be added post rotation.
	"""
	cx = (width - 1) / 2.0
	cy = (height - 1) / 2.0
	corners = (
		(0.0, 0.0),
		(width - 1.0, 0.0),
		(width - 1.0, height - 1.0),
		(0.0, height - 1.0),
	)
	rotated = [rotate_point_about_center(x, y, cx, cy, angle_rad) for (x, y) in corners]
	xs = [p[0] for p in rotated]
	ys = [p[1] for p in rotated]
	min_x = math.floor(min(xs))
	min_y = math.floor(min(ys))
	max_x = math.ceil(max(xs))
	max_y = math.ceil(max(ys))
	out_w = int(max_x - min_x + 1)
	out_h = int(max_y - min_y + 1)
	offset_x = -float(min_x)
	offset_y = -float(min_y)
	return out_w, out_h, offset_x, offset_y


def pil_interpolation(mode: str) -> int:
	mode_lower = mode.lower()
	if mode_lower == "nearest":
		return Image.NEAREST
	if mode_lower == "bilinear":
		return Image.BILINEAR
	if mode_lower == "bicubic":
		return Image.BICUBIC
	raise ValueError(f"Unsupported interpolation mode: {mode}")


def round_and_clamp_points(
	points_xy: Sequence[Tuple[float, float]], out_w: int, out_h: int
) -> List[Tuple[int, int]]:
	rounded: List[Tuple[int, int]] = []
	for x, y in points_xy:
		xi = int(round(x))
		yi = int(round(y))
		xi = max(0, min(out_w - 1, xi))
		yi = max(0, min(out_h - 1, yi))
		rounded.append((xi, yi))
	return rounded


def smart_resize_dimensions(
	width: int,
	height: int,
	factor: int,
	min_pixels: int,
	max_pixels: int,
	max_ratio: float,
) -> Tuple[int, int]:
	if width <= 0 or height <= 0:
		raise ValueError("Image dimensions must be positive")
	if factor <= 0:
		raise ValueError("factor must be positive")
	if min_pixels <= 0 or max_pixels <= 0:
		raise ValueError("Pixel thresholds must be positive")
	if min_pixels > max_pixels:
		raise ValueError("min_pixels must be <= max_pixels")
	if max_ratio <= 0:
		raise ValueError("max_ratio must be positive")
	short_edge = min(width, height)
	long_edge = max(width, height)
	if short_edge == 0:
		raise ValueError("Zero dimension encountered in smart resize")
	ratio = long_edge / short_edge
	if ratio > max_ratio:
		raise ValueError(
		f"Absolute aspect ratio must be <= {max_ratio}, got {ratio:.4f}"
		)
	adj_w = max(factor, round_by_factor(width, factor))
	adj_h = max(factor, round_by_factor(height, factor))
	current_pixels = adj_w * adj_h
	if current_pixels > max_pixels:
		beta = math.sqrt((width * height) / max_pixels)
		adj_w = max(factor, floor_by_factor(int(width / beta), factor))
		adj_h = max(factor, floor_by_factor(int(height / beta), factor))
		current_pixels = adj_w * adj_h
		if current_pixels > max_pixels:
			scale = math.sqrt(current_pixels / max_pixels)
			adj_w = max(factor, floor_by_factor(int(adj_w / scale), factor))
			adj_h = max(factor, floor_by_factor(int(adj_h / scale), factor))
	elif current_pixels < min_pixels:
		beta = math.sqrt(min_pixels / (width * height))
		adj_w = max(factor, ceil_by_factor(int(width * beta), factor))
		adj_h = max(factor, ceil_by_factor(int(height * beta), factor))
		current_pixels = adj_w * adj_h
		if current_pixels < min_pixels:
			scale = math.sqrt(min_pixels / max(current_pixels, 1))
			adj_w = max(factor, ceil_by_factor(int(adj_w * scale), factor))
			adj_h = max(factor, ceil_by_factor(int(adj_h * scale), factor))
	return adj_w, adj_h


# ---- Reusable geometry/object helpers ----

def polygon_area(pts: List[Tuple[int, int]]) -> float:
	area2 = 0.0
	for i in range(len(pts)):
		x1, y1 = pts[i]
		x2, y2 = pts[(i + 1) % len(pts)]
		area2 += x1 * y2 - x2 * y1
	return abs(area2) * 0.5


def point_in_quad(px: float, py: float, quad: List[Tuple[int, int]]) -> bool:
	# Split quad into two triangles for barycentric area check
	def tri_area(a, b, c):
		return abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0)

	whole = polygon_area(quad)
	a1 = tri_area((px, py), quad[0], quad[1])
	a2 = tri_area((px, py), quad[1], quad[2])
	a3 = tri_area((px, py), quad[2], quad[3])
	a4 = tri_area((px, py), quad[3], quad[0])
	return abs((a1 + a2 + a3 + a4) - whole) < 1e-3


def quad_contains_quad(inner: List[Tuple[int, int]], outer: List[Tuple[int, int]]) -> bool:
	for x, y in inner:
		if not point_in_quad(x, y, outer):
			return False
	return True


def quad_area(quad: List[Tuple[int, int]]) -> float:
	return polygon_area(quad)


def quad_bbox(quad: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
	xs = [p[0] for p in quad]
	ys = [p[1] for p in quad]
	return (min(xs), min(ys), max(xs), max(ys))


def bbox_from_quad_flat(q_flat: List[int]) -> Tuple[int, int, int, int]:
	xs = q_flat[0::2]
	ys = q_flat[1::2]
	return (min(xs), min(ys), max(xs), max(ys))


def iou_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	iw = max(0, ix2 - ix1)
	ih = max(0, iy2 - iy1)
	inter = iw * ih
	if inter == 0:
		return 0.0
	area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
	area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


def quad_inside_bounds(q: List[Tuple[int, int]], width: int, height: int) -> bool:
	for x, y in q:
		if not (0 <= x < width and 0 <= y < height):
			return False
	return True


def is_quad(o: Dict[str, Any]) -> bool:
	return "quad" in o


def is_bbox(o: Dict[str, Any]) -> bool:
	return "bbox_2d" in o


def is_line(o: Dict[str, Any]) -> bool:
	return "line" in o


def object_quad(o: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
	if "quad" in o:
		q = o["quad"]
		return [(int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (int(q[4]), int(q[5])), (int(q[6]), int(q[7]))]
	if "bbox_2d" in o:
		b = o["bbox_2d"]
		return [(int(b[0]), int(b[1])), (int(b[2]), int(b[1])), (int(b[2]), int(b[3])), (int(b[0]), int(b[3]))]
	return None


def line_points(o: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
	if "line" in o:
		l = o["line"]
		return [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
	return None


def obj_bbox(o: Dict[str, Any]) -> Tuple[int, int, int, int]:
	if "quad" in o:
		return bbox_from_quad_flat(o["quad"])  # type: ignore[arg-type]
	if "bbox_2d" in o:
		b = o["bbox_2d"]
		return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
	if "line" in o:
		l = o["line"]
		xs = l[0::2]
		ys = l[1::2]
		return (min(xs), min(ys), max(xs), max(ys))
	return (0, 0, 0, 0)


def rasterize_quads_to_mask(
	quads: List[List[Tuple[int, int]]], size: Tuple[int, int]
) -> Image.Image:
	mask = Image.new("L", size, 0)
	draw = ImageDraw.Draw(mask)
	for q in quads:
		draw.polygon(q, outline=255, fill=255)
	return mask


def build_occupancy_mask(
	sample: Dict[str, Any], width: int, height: int, down: int, margin: int
) -> Image.Image:
	quads: List[List[Tuple[int, int]]] = []
	for o in (sample["objects"] if (isinstance(sample, dict) and ("objects" in sample) and isinstance(sample["objects"], list)) else []):
		if "quad" in o:
			q = o["quad"]
			quads.append([(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])])
		elif "bbox_2d" in o:
			b = o["bbox_2d"]
			x1, y1, x2, y2 = b
			quads.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
		elif "line" in o:
			# approximate with small-width polygon around line; skip for occupancy simplicity
			continue
	if not quads:
		return Image.new("L", (max(1, width // down), max(1, height // down)), 0)

	base = rasterize_quads_to_mask(quads, (width, height))
	if margin > 0:
		base = base.filter(ImageFilter.MaxFilter(size=margin | 1))
	low = base.resize((max(1, width // down), max(1, height // down)), Image.NEAREST)
	return low
