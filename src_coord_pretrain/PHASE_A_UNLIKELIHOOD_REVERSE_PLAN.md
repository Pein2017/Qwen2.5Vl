# Deprecated: Phase A + Unlikelihood + Reverse Mapping (Plan Only)

This document described a multi‑phase pipeline. The current implementation uses a simplified single‑phase training pipeline with a cosine scheduler, frozen vision tower, gradient mask on embeddings (only coordinate tokens update), and optional unlikelihood and reverse mapping.

See `src_coord_pretrain/README.md` and `src_coord_pretrain/training/trainer.py` for the updated design.
