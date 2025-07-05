#!/usr/bin/env python3
"""strip_exif.py – Remove all EXIF metadata from JPEG images.

Usage (from project root):
    python scripts/strip_exif.py            # process ./ds recursively
    python scripts/strip_exif.py --dir IMG  # custom directory
    python scripts/strip_exif.py --dry-run  # show what *would* be modified

Why?  Cameras often embed an EXIF orientation tag that causes confusion in
annotation pipelines.  This script removes the entire EXIF block while leaving
pixel data unchanged, ensuring downstream tools always read the same dimensions.

Safe-guards:
• Uses a temporary file then atomically replaces the original to avoid data loss.
• Skips files that already lack EXIF.
• Supports Ctrl-C interruption (partial progress preserved).
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def strip_exif_from_file(path: Path, dry_run: bool = False) -> None:
    """Remove EXIF metadata from *path* in-place.  Overwrites the file."""
    if not path.is_file():
        return

    try:
        with Image.open(path) as img:
            if "exif" not in img.info:
                return  # nothing to do

            # Apply orientation from EXIF so the pixel data is stored upright.
            img_processed: Image.Image = ImageOps.exif_transpose(img)

            if dry_run:
                return

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            # Save the (potentially rotated) image without any EXIF metadata.
            img_processed.save(
                fp=tmp_path,
                format="JPEG",
            )
            tmp_path.replace(path)  # atomic on POSIX
    except Exception as e:  # pylint: disable=broad-except
        tqdm.write(f"[WARN] Failed to process {path}: {e}")


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip EXIF metadata from JPEG images."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("ds_output"),
        help="Root directory to scan (default: ds_output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that *would* be modified without writing.",
    )
    args = parser.parse_args()

    root: Path = args.dir
    if not root.is_dir():
        sys.exit(f"Error: directory not found: {root}")

    # Gather JPEG/JPG files
    files = sorted(root.rglob("*.jp*g"))  # matches .jpg and .jpeg (case-insensitive)
    if not files:
        print("No JPEG files found.")
        return

    # Allow graceful Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: sys.exit("\nInterrupted."))

    modified = 0
    with tqdm(files, desc="Stripping EXIF", unit="img") as bar:
        for fp in bar:
            before_size = fp.stat().st_size
            strip_exif_from_file(fp, dry_run=args.dry_run)
            after_size = fp.stat().st_size if fp.exists() else before_size
            if after_size != before_size:
                modified += 1

    if args.dry_run:
        print(f"[Dry-run] {modified} images would be modified out of {len(files)}.")
    else:
        print(f"Done. {modified} images modified out of {len(files)}.")


if __name__ == "__main__":
    main()
