# 🐞 Debug Diary & HF-Transformers Integration Guide (Merged)

*Last updated: 2025-06-21 – distilled from everything inside `memory/`*

---
## 0. Why This File Exists
`memory/` exploded with ad-hoc logs (~200 KB). This single doc keeps the actionable lessons so the old files can be binned.

---
## 1. Greatest Hits — Bugs We Already Fixed
| ID | Symptom / Error | Root Cause | Fix Commit |
|----|-----------------|-----------|-----------|
| G-001 | `shape '[0, 4, -1]' is invalid for input of size 1280` during `model.generate()` | Passed vision tensors on every generation step → empty seq_len after first step | Use official `prepare_inputs_for_generation`: vision inputs **only** in pre-fill step. Fail-fast if `pixel_values ⊕ image_grid_thw` inconsistent. |
| G-002 | “BOX BOX BOX” no coords in inference | Called `model.generate()` instead of detection pipeline | Always call `predict_detection` for structured output. |
| M-001 | GPU memory climbs every batch (DeepSpeed) | ① Vision token explosion (896 tokens for 256×364 image); ② Gradients not cleared; ③ DeepSpeed partition cache | a) Filter samples >200 vision tokens or resize to ≤512 px; b) `model.zero_grad(set_to_none=True)`; c) `empty_partition_cache()` each step. |
| M-002 | Deadlock on `torch.cuda.synchronize()` multi-GPU | Manual sync calls + `torch.distributed.barrier()` in training loop | Remove explicit sync/barrier; rely on DeepSpeed scheduler. |
| M-003 | CUDA OOM right after epoch transition | DeepSpeed ZeRO-2 buffers/optimizer state not cleared between epochs | Enhanced zero2 config (bucket sizes, overlap_comm), call `model.empty_partition_cache()` at epoch end, aggressive `torch.cuda.empty_cache()` |
| P-001 | Bad mRoPE after updating Transformers 4.41 | Off-by-one in 2-D rotary patch | Re-implemented `apply_mrope_fix()` in `src/models/patches.py`. Tests pass. |
| P-002 | Vision tokens mismatch assertion | Using original `ds/*` images instead of `ds_rescaled/*` / EXIF rotation auto-applied | Enforce *pre-scaled* JPEGs + forbid `ImageOps.exif_transpose`. |
| P-003 | Pixel-values dim mismatch warnings | Validation assumed 4-D `[N,C,H,W]`; Qwen returns 2-D `[patches,dim]` | Accept both; debug log dims; raise if neither 2-D nor 4-D |
| P-004 | Rope-embedding CUDA index error | Home-grown `get_rope_index_25` diverged from upstream 3-D RoPE | Drop custom; feed `image_grid_thw` & let official `get_rope_index()` handle; ensure tensors on correct device |
| L-001 | Log files 1 GB+ | Logger defaulted to DEBUG everywhere | Centralised `logger_utils.py` with per-module levels; only DEBUG on crash. |

*(See commit history for SHA references.)*

---
## 2. Integration Checklist with 🤗 Transformers Trainer
1. **compute_loss Contract**
   • Return *single* scalar `total_loss`.  
   • Side-metrics must be plain `float` (not Tensor) and stored in `outputs` for Trainer’s logger.
2. **prepare_inputs_for_generation**
   • Vision inputs appear **once**. Use official hook to set them to `None` afterwards.  
   • Keep cache-position logic identical to upstream to avoid KV shape mismatches.
3. **Parameter Naming → LR Groups**
   • `visual.*`, `merger.*`, `detection_head.*`, `model.*`, `lm_head.*` – nothing else.  
   • Any rogue parameter triggers `KeyError` (fail-fast).
4. **Special Tokens Sync**
   • On start-up assert `tokenizer.convert_tokens_to_ids(tok) == EXPECTED_ID` for every entry in `SpecialTokens`.
5. **Config Hygiene**
   • All hyper-params come from YAML. 0 hard-coded defaults.  
   • Validation dataclass raises on missing keys.
6. **Logging**
   • Let `Trainer.save_state()` handle JSON logs; we only push floats.  
   • Heavy debug info only on exception.
7. **DeepSpeed Quirks (if enabled)**
   • Call `model.empty_partition_cache()` after each step.  
   • Avoid `torch.cuda.synchronize()`; use `zero3` comm overlap.

---
## 3. Common Pitfalls (Still Waiting to Bite)
1. **Vision Token Inflation** – Keep an eye on `pixel_values.shape[0]`; >500 likely means bad image or wrong merge_size.
2. **Mixed Precision NaNs** – If `caption_loss` spikes → reduce `detection_caption_weight` or enable `torch.autograd.detect_anomaly()` for 1 batch.
3. **Frozen Detection Head** – Remember `detection_freeze_epochs`; set to `0` unless you *want* a delay.
4. **Chinese Glyph ❓** – Pass `--font_path` when visualising; CI machines lack CJK fonts.
5. **mRoPE Future Updates** – Upstream might adopt a new naming; rerun unit test `tests/test_mrope.py` after each Transformers bump.

---
## 4. Quick Commands
• Inspect vision tokens per batch:
```bash
python - <<'PY'
from src.chat_processor import ChatProcessor; import json, sys
sample=json.loads(open('sample.json').read())
proc=ChatProcessor.from_pretrained('Qwen/everything')
print(proc.debug_count_vision_tokens(sample))
PY
```
• Check parameter group assignment:
```bash
python -m src.training.verify_param_groups checkpoint_dir
```
• Benchmark memory leak:
```bash
bash scripts/debug_memory.sh  | tee mem.log
```

---
## 5. House-Keeping Rules
- Delete temp scripts once PR merged.  
- Keep this file updated – it’s the *single source* of debug/integration truth.  
- After adding a new failure-mode, append a row to section 1 (ID prefix `N-###`).

Good luck & keep failing fast 🚀 

---
## 6. Reference Material Pointers (for archaeology)
Should you need the nitty-gritty commit diff or historical rationale, these raw logs remain (will be pruned later):
- `memory/rope-embeddings-solution.md` – deep dive on 3-D RoPE.
- `memory/pixel_values_fix.md` – 2-D vs 4-D pixel tensors.
- `memory/chat_processor_refactoring.md` – full refactor plan & tests.
- `memory/logger.md` – timeline of pipeline refactors & ultra-compact format.
(Everything else now captured above.) 