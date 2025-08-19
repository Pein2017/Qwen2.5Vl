# Development Tools

This directory contains development and debugging utilities for Qwen2.5-VL that are not part of the main runtime API.

## Available Tools

### Performance Monitor
- **Module**: `performance_monitor.py`
- **Purpose**: Track and measure performance metrics during model initialization
- **Usage**: Import from `src_new.tools` for development/debugging

### Checkpoint Validator
- **Module**: `checkpoint_validator.py`
- **Purpose**: Validate checkpoint structure and content for inference compatibility
- **Usage**: Import from `src_new.tools` or use CLI interface

## Command Line Interface

Use the CLI script for easy access to development tools:

```bash
# Validate a single checkpoint
python -m src_new.tools.cli validate-checkpoint /path/to/checkpoint

# Validate all checkpoints in a directory
python -m src_new.tools.cli validate-all-checkpoints /path/to/output_dir

# Show performance monitor status
python -m src_new.tools.cli performance-status

# Reset performance monitor
python -m src_new.tools.cli reset-performance
```

## Usage in Code

```python
# Import development tools (not for production use)
from src_new.tools import PerformanceMonitor, CheckpointValidator

# Performance monitoring
monitor = PerformanceMonitor()
with monitor.time_operation("my_operation"):
    # Your code here
    pass

# Checkpoint validation
validator = CheckpointValidator("/path/to/checkpoint")
is_valid, results = validator.validate_checkpoint()
```

## Design Philosophy

These tools follow the project's fail-fast philosophy:
- No silent fallbacks or broad try/except blocks
- Explicit validation with detailed error messages
- Clear separation from production runtime code

## Migration Notes

These utilities were moved from `src_new.utils` to maintain a clean public API. They are no longer exported from the main utils module to prevent accidental use in production code.
