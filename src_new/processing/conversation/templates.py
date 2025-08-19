# Shim for conversation templates: re-export legacy SYSTEM_PROMPT if present
try:
    from ..templates import SYSTEM_PROMPT as SYSTEM_PROMPT  # type: ignore
except Exception:
    SYSTEM_PROMPT = None  # Fallback placeholder
