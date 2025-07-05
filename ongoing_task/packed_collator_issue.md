# Packed Collator Performance Divergence – Post-Mortem
> **Resolved (2025-06-25)** — Fix merged into `src/training/trainer.py::_maybe_unpack_packed`.  Document kept for historical context.

## 1  Symptoms
* **Training loss curves:** with `collator_type: packed` the student LM loss plateaued / diverged while the `standard` (padding) collator trained smoothly.
* **GPU utilisation:** packed runs showed 2–3× higher attention FLOPs and noticeably slower steps even though the batch tokens-per-step were identical.
* **Qualitative checks:** generated samples contained cross-sample bleed-through, indicating the model attended beyond the intended sequence.

## 2  Root Cause
* `PackedDataCollator` concatenates all samples into **one** long sequence and emits a prefix-sum vector `cu_seqlens`.
* In `BBUTrainer.compute_loss` we **dropped that vector** (`model_inputs.pop("cu_seqlens", None)`) before the forward pass.
* Without `cu_seqlens` the model fell back to full-sequence causal attention ⇒ O((∑L)²) instead of O(∑Lᵢ²) and, more importantly, tokens from different samples could attend to one another.

## 3  Fix
Inside `src/training/trainer.py`:
1. Added helper `_maybe_unpack_packed` that:
   * Detects presence of `cu_seqlens`.
   * Splits the flat tensors back into per-sample segments.
   * Re-pads to the local max length → standard `(B, S)` layout.
   * Re-builds `attention_mask`, `position_ids`, and adjusts teacher / student spans.
   * Removes `cu_seqlens` to avoid downstream confusion.
2. Invoked this helper at the very start of `compute_loss` so **all** subsequent logic (mask building, loss splitting, detection head, etc.) now works on a valid batch.

## 4  Outcome
* Packed and standard collators now produce **identical attention patterns** and comparable loss curves.
* Training speed improved (no more O((∑L)²) cost) and memory dropped.
* Code change local to `src/`, no external library patches required.

