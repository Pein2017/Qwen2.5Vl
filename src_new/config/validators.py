"""Utility validators for configuration schema gating.

These helpers operate on nested dictionaries using dotted JSON-pointer style
paths (e.g., ``features.augmentation.enabled`` or
``features.augmentation_schedule[0].preset``). They return human readable
error messages so callers can aggregate them and raise a single exception
containing all issues.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence


_MISSING = object()


def _tokenize(path: str) -> List[Any]:
    """Split dotted/array paths into components.

    Supports ``foo.bar`` and ``foo[0].bar`` style addressing.
    """
    if not path:
        return []

    tokens: List[Any] = []
    parts = path.split(".")
    for part in parts:
        remainder = part
        while remainder:
            if "[" not in remainder:
                tokens.append(remainder)
                remainder = ""
                continue
            prefix, rest = remainder.split("[", 1)
            if prefix:
                tokens.append(prefix)
            if "]" not in rest:
                # Malformed; keep remainder as-is to surface downstream error
                tokens.append(rest)
                remainder = ""
                continue
            idx_str, remainder = rest.split("]", 1)
            idx_str = idx_str.strip()
            if idx_str.isdigit():
                tokens.append(int(idx_str))
            else:
                # Preserve non-integer indices verbatim to surface context later
                tokens.append(idx_str)
            # Remove optional leading dot when looping
            if remainder.startswith("."):
                remainder = remainder[1:]
    return tokens


def _resolve(data: Any, tokens: Sequence[Any]) -> Any:
    current = data
    for token in tokens:
        if isinstance(token, int):
            if isinstance(current, list) and 0 <= token < len(current):
                current = current[token]
            else:
                return _MISSING
        else:
            if isinstance(current, dict) and token in current:
                current = current[token]
            else:
                return _MISSING
    return current


def _get_value(data: Any, path: str) -> Any:
    return _resolve(data, _tokenize(path))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def requires_if(data: Any, path: str, expected_value: Any, required_fields: Iterable[str]) -> List[str]:
    """Ensure required fields exist when ``path`` equals ``expected_value``."""
    errors: List[str] = []
    current = _get_value(data, path)
    if current is _MISSING:
        return errors
    if current == expected_value:
        for field in required_fields:
            if _get_value(data, field) is _MISSING:
                errors.append(
                    f"{field} is required when {path} == {_format_value(expected_value)}"
                )
    return errors


def forbids_when(data: Any, path: str, expected_value: Any, forbidden_fields: Iterable[str]) -> List[str]:
    """Ensure certain fields are absent when ``path`` equals ``expected_value``."""
    errors: List[str] = []
    current = _get_value(data, path)
    if current is _MISSING:
        return errors
    if current == expected_value:
        for field in forbidden_fields:
            if _get_value(data, field) is not _MISSING:
                errors.append(
                    f"{field} must not be set when {path} == {_format_value(expected_value)}"
                )
    return errors


def exactly_one(data: Any, paths: Sequence[str]) -> List[str]:
    """Require exactly one of the provided dotted paths to be truthy."""
    present = []
    for path in paths:
        value = _get_value(data, path)
        if value is not _MISSING and bool(value):
            present.append(path)
    if len(present) == 1:
        return []
    if len(present) == 0:
        return [f"Exactly one of {', '.join(paths)} must be provided"]
    return [
        f"Exactly one of {', '.join(paths)} must be provided (found {', '.join(present)})"
    ]


def in_set(data: Any, path: str, allowed: Iterable[Any]) -> List[str]:
    """Ensure the value at ``path`` belongs to ``allowed``."""
    allowed_set = set(allowed)
    current = _get_value(data, path)
    if current is _MISSING:
        return []
    if current not in allowed_set:
        return [
            f"{path} must be one of {sorted(allowed_set)}, got {_format_value(current)}"
        ]
    return []


def positive_int(data: Any, path: str, allow_zero: bool = False) -> List[str]:
    """Validate that ``path`` points to a positive integer value."""
    value = _get_value(data, path)
    if value is _MISSING:
        return []
    if not isinstance(value, int):
        return [f"{path} must be an integer, got {_format_value(value)}"]
    if allow_zero:
        if value < 0:
            return [f"{path} must be >= 0, got {value}"]
    else:
        if value <= 0:
            return [f"{path} must be > 0, got {value}"]
    return []


def probability(data: Any, path: str) -> List[str]:
    """Validate that ``path`` points to a probability in ``[0, 1]``."""
    value = _get_value(data, path)
    if value is _MISSING:
        return []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{path} must be a numeric probability in [0, 1], got {_format_value(value)}"]
    if value < 0 or value > 1:
        return [f"{path} must be within [0, 1], got {value}"]
    return []


def non_negative(data: Any, path: str) -> List[str]:
    value = _get_value(data, path)
    if value is _MISSING:
        return []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{path} must be numeric, got {_format_value(value)}"]
    if value < 0:
        return [f"{path} must be >= 0, got {value}"]
    return []


def greater_than(data: Any, path: str, minimum: float, inclusive: bool = False) -> List[str]:
    value = _get_value(data, path)
    if value is _MISSING:
        return []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{path} must be numeric, got {_format_value(value)}"]
    if inclusive:
        if value < minimum:
            return [f"{path} must be >= {minimum}, got {value}"]
    else:
        if value <= minimum:
            return [f"{path} must be > {minimum}, got {value}"]
    return []


__all__ = [
    "requires_if",
    "forbids_when",
    "exactly_one",
    "in_set",
    "positive_int",
    "probability",
    "non_negative",
    "greater_than",
]
