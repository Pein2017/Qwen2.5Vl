from typing import NamedTuple


class CoordTokenRange(NamedTuple):
    start_id: int
    end_exclusive: int


__all__ = [
    "CoordTokenRange",
]
