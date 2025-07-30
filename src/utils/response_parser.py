"""
Compatibility shim for response_parser.py

This file maintains backward compatibility by re-exporting the ResponseParser
class from the new data_utils module.
"""

# Import from new location
from .data_utils import ResponseParser


# Re-export for backward compatibility
__all__ = ["ResponseParser"]
