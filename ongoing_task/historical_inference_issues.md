# 📋 Historical Reference: Resolved Inference Issues

*Status: **RESOLVED** (2025-06-28) — Document retained for historical reference*

---

## 🚨 Problem Summary
When running vision-language inference with Qwen2.5-VL, the model produced JSON-like output where the overall structure often remained intact, but keys, values, and labels were truncated, misspelled, or randomly tokenized.

**Examples of corrupted output:**
- `[{"bBox_3d": [140,0],"label_1d":"机c柜空"…]`
- `[{"bbox 2D": … "labe 1":"螺丝连 接点 /BB U接地 端"}]`

**Key observation**: Pure text generation remained coherent, confirming the language head was intact. The corruption only appeared when vision features were fused.

---

## 🏆 Root Cause: Chat Template Divergence

**The Issue:**
1. **Training phase** – `Qwen2_5VLCollator` overwrites `tokenizer.chat_template` *in-memory* with a custom Jinja string before each call to `apply_chat_template`.
2. **Checkpoint saving** – The tokenizer was saved *before* this overwrite, so the checkpoint on disk still contains the **default** Qwen2.5-VL template.
3. **Inference phase** – The default template lacked the exact `<|image_pad|>` layout expected by the fine-tuned weights, causing position-ID drift and scrambled decoding.

**The mismatch caused:**
- Systematic position drift of visual embeddings
- Scattered JSON decoding
- Truncated, misspelled, or corrupted output

---

## ✅ Solution Applied

**Hot-fix deployed in demo script:**
```python
from src.reference.qwen2_5vl_collator import Qwen2_5VLCollator
_ = Qwen2_5VLCollator(processor)  # mutates processor.tokenizer.chat_template
```

**Result:** This one-liner guaranteed template parity and restored perfectly coherent JSON output:
```json
[{"bbox_2d": [308,126,532,540],"label": "bbu基带处理单元/华为"}, {"bbox_2d": [267,134,327,200],"label": "螺丝连接点/BBU安装螺丝"}]
```

---

## 🛠️ Permanent Fix (Implemented)

**Modified `src/inference.py` and `eval/infer_dataset.py`:**
- Always call `Qwen2_5VLCollator(processor)` immediately after loading the processor
- Ensures template parity without writing any new files
- Both standard and teacher-guided inference now use identical ChatProcessor

**Key code change:**
```python
# In InferenceEngine.__init__()
self.chat_processor = ChatProcessor(
    tokenizer=self.processor.tokenizer,
    image_processor=self.processor.image_processor,
    use_training_prompts=True,
    language=self.language,
)
```

---

## 📊 Verification Results

**Before fix:**
- Garbled JSON with truncated/misspelled keys
- Position-ID drift causing scattered visual embeddings
- Poor inference performance despite low training loss

**After fix:**
- Perfect JSON structure and coherent content
- Proper alignment of visual embeddings with text tokens
- Inference performance matching training expectations

---

## 💡 Key Lessons Learned

1. **Template consistency is critical** between training and inference
2. **In-memory modifications** during training must be preserved in saved checkpoints
3. **Vision-text alignment** is extremely sensitive to token position shifts
4. **Always verify prompt structure** matches between training and inference pipelines

---

**This issue resolution ensured that the inference pipeline truly matches the training pipeline, fixing the core problem that caused poor inference results when not using teacher guidance.** 