#!/bin/bash

python ./src_coord_pretrain/scripts/generate_coord_bootstrap.py --output ./src_coord_pretrain/data/coord_bootstrap.jsonl --num_samples 50000 --ratio_identity 0.5 --ratio_arithmetic 0.2 --ratio_reverse 0.3 --max_coord 1024 --seed 42 --dedup