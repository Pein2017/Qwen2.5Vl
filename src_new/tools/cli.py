#!/usr/bin/env python3
"""
Command-line interface for development tools.

This script provides easy access to development and debugging utilities
that are not part of the main runtime API.
"""

import argparse
import sys

from .checkpoint_validator import validate_all_checkpoints, validate_checkpoint
from .performance_monitor import get_performance_monitor, reset_performance_monitor


def validate_checkpoint_cmd(args):
    """Validate a single checkpoint."""
    is_valid, results = validate_checkpoint(args.checkpoint_path, print_report=True)
    return 0 if is_valid else 1


def validate_all_checkpoints_cmd(args):
    """Validate all checkpoints in a directory."""
    results = validate_all_checkpoints(args.output_dir)

    print(f"\n🔍 Validation Summary for {len(results)} checkpoints:")
    valid_count = 0
    for checkpoint_name, (is_valid, _) in results.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {checkpoint_name}")
        if is_valid:
            valid_count += 1

    print(f"\n📊 {valid_count}/{len(results)} checkpoints are valid")
    return 0 if valid_count == len(results) else 1


def performance_monitor_cmd(args):
    """Show performance monitor status."""
    monitor = get_performance_monitor()
    print(monitor.get_summary())
    return 0


def reset_performance_monitor_cmd(args):
    """Reset the performance monitor."""
    reset_performance_monitor()
    print("✅ Performance monitor reset")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Development tools for Qwen2.5-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Checkpoint validation commands
    validate_parser = subparsers.add_parser(
        "validate-checkpoint", help="Validate a single checkpoint"
    )
    validate_parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    validate_parser.set_defaults(func=validate_checkpoint_cmd)

    validate_all_parser = subparsers.add_parser(
        "validate-all-checkpoints", help="Validate all checkpoints in a directory"
    )
    validate_all_parser.add_argument(
        "output_dir", help="Directory containing checkpoints"
    )
    validate_all_parser.set_defaults(func=validate_all_checkpoints_cmd)

    # Performance monitoring commands
    perf_parser = subparsers.add_parser(
        "performance-status", help="Show performance monitor status"
    )
    perf_parser.set_defaults(func=performance_monitor_cmd)

    reset_perf_parser = subparsers.add_parser(
        "reset-performance", help="Reset performance monitor"
    )
    reset_perf_parser.set_defaults(func=reset_performance_monitor_cmd)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
