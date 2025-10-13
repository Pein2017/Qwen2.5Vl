#!/usr/bin/env python3
"""
RL feature checks script is deprecated.

The manual GRPO trainer has been removed in favor of TRL's GRPOTrainer.
Use:
  python -m src_new.rl.runner --config <yaml> --mode train

This script remains as a placeholder for backward compatibility.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "This script is deprecated. Use 'python -m src_new.rl.runner --config <yaml> --mode train' instead.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
