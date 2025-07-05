# Operations, Diagnostics & Fail-Fast Philosophy

> **Purpose:** Summarise runtime checks, monitoring hooks, and common failure modes.

---

## 1. Special tokens (IDs are asserted at startup)
| Token | String | Purpose |
|-------|--------|---------|
| IM_START | `<|im_start|>` | Start of chat turn |
| IM_END   | `<|im_end|>`   | End of turn & sequence |
| VISION_START | `<|vision_start|>` | Vision prefix |
| VISION_END   | `<|vision_end|>`   | Vision suffix |
| IMAGE_PAD    | `<|image_pad|>`    | One image patch |

## 2. Fail-fast guards (executed **before** model forward)
* Vision/text length – #`<|image_pad|>` tokens must equal length of `pixel_values`.
* Special-token sync – IDs in tokenizer vs hard-coded constants.
* Image size – JPEGs must already respect min/max size from `vision_process.py`.
* **Data conversion integrity** – JSON structure validation and bbox scaling accuracy.
* **Path consistency** – All JSONL files reference correct `ds_output/` image paths.

## 3. Monitoring hooks
| Metric | Logged by | Location |
|--------|-----------|----------|
| `lm_loss`, `teacher_lm_loss`, `student_lm_loss` | `BBUTrainer` | `trainer.compute_loss()` |
| `bbox_*`, `objectness_loss`, `caption_loss` | Detection head | `detection_loss.py` |
| Gradient / weight norms | Trainer | `trainer._log_metrics()` |

## 4. Common pitfalls (and fixes)
| Symptom | Likely cause |
|---------|--------------|
| "BOX BOX BOX" string, no coords | Used `model.generate()` instead of detection pipeline. |
| Vision token mismatch assertion | Used un-rescaled images or EXIF rotation was applied. |
| Caption loss >> bbox loss | `detection_caption_weight` too high. |
| **'list' object has no attribute 'get'** | **NEW**: Using old JSON format without running `clean_raw_json.py` first. |
| **FileNotFoundError for images** | **NEW**: Images not copied to `ds_output/` directory during conversion. |
| **Bbox coordinates out of bounds** | **NEW**: Incorrect bbox scaling - check `smart_resize` parameters. |

## 5. Solved issues (historical)
* Chat template divergence – detailed in `historical_inference_issues.md`.
* Packed-collator boundary bleed – see `advanced/collator_notes.md`.

## 6. Collator-specific assertions
`PackedDataCollator` adds two extra safety checks:
1. **Boundary masking** – label at each sample boundary (`cu_seqlens[1:-1]`) is set to `-100` to prevent cross-sample supervision.
2. **Position-ID reset** – verifies that `position_ids[0, cu_seqlens[:-1]] == 0` so every packed sample restarts the rotary cache.

`StandardDataCollator` validates that its boolean `attention_mask` has exactly `seq_len` true values per row.

---

### Related source files
* `src/chat_processor.py`
* `src/training/trainer.py`
* `src/detection_loss.py` 