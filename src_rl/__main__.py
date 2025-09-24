#!/usr/bin/env python3
import argparse
from .runner import main as loader_main


def main():
    parser = argparse.ArgumentParser(description="src_rl entrypoint")
    parser.add_argument("--config", type=str, required=False, help="Path to RL YAML config for loader")
    args, unknown = parser.parse_known_args()
    if args.config:
        loader_main()
    else:
        print("Usage: python -m src_rl --config /abs/path/to/configs/rl/dense_grpo.yaml")


if __name__ == "__main__":
    main()
