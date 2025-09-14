# Shim for conversation templates: re-export legacy SYSTEM_PROMPT if present
try:
    from ..templates import get_system_prompt as _get_system_prompt  # type: ignore

    SYSTEM_PROMPT = _get_system_prompt(True)
except Exception:
    SYSTEM_PROMPT = None  # Fallback placeholder
