# Packed Collator Internals & Performance Notes

> **Purpose:** Capture the quirks and fixes surrounding `PackedDataCollator` that concatenates variable-length sequences into one long row.

---

## 1. Why packed?
Padding wastes memory when sequence lengths vary widely.  Concatenation reduces memory & FLOPs and plays nicely with Flash Attention 2.

## 2. Critical fix (2025-06-25)
Originally `cu_seqlens` was dropped before the model forward, causing O((ΣL)²) attention and token bleed between samples.  `_maybe_unpack_packed()` now:
1. Reconstructs per-sample segments.
2. Re-pads to local max length.
3. Rebuilds `attention_mask` & span indices.

Result: packed & standard collators now train identically while saving ~25 % memory.

## 3. Boundary safety
During span adjustment any token that crosses a sample boundary (given by `cu_seqlens`) is removed from both teacher and student span lists to prevent gradient leakage.

---

### Related source files
* `src/data.py` – `PackedDataCollator`
* `src/training/trainer.py` – `_maybe_unpack_packed` 