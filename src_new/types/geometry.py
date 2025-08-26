from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Quad:
    coords: List[int]  # length 8


@dataclass(frozen=True)
class Line:
    coords: List[int]  # even length >= 4


def _is_int(value) -> bool:
    # bool is a subclass of int in Python; exclude it explicitly
    return isinstance(value, int) and not isinstance(value, bool)


def is_valid_box_coords(coords: List[int]) -> bool:
    if not isinstance(coords, list) or len(coords) != 4:
        return False
    if not all(_is_int(v) for v in coords):
        return False
    x1, y1, x2, y2 = coords
    return x1 < x2 and y1 < y2


def is_valid_quad_coords(coords: List[int]) -> bool:
    if not isinstance(coords, list) or len(coords) != 8:
        return False
    if not all(_is_int(v) for v in coords):
        return False
    return True


def is_valid_line_coords(coords: List[int]) -> bool:
    if not isinstance(coords, list) or len(coords) < 4 or (len(coords) % 2) != 0:
        return False
    if not all(_is_int(v) for v in coords):
        return False
    return True


__all__ = [
    "Box",
    "Quad",
    "Line",
    "is_valid_box_coords",
    "is_valid_quad_coords",
    "is_valid_line_coords",
]
