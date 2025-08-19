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


__all__ = ["Box", "Quad", "Line"]
