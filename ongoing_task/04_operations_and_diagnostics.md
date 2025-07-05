# 🔧 Operations, Tokenisation & Diagnostics (Merged)

*Last updated: 2025-06-21 – condenses **chat_template_check.md**, **diagnose_inference.md**, **input_data_form.md**, and **raw_data_format.md***

---
## 1. Chat Template & Special Tokens
| Token                 | String              | ID      | Purpose                              |
| --------------------- | ------------------- | ------- | ------------------------------------ |
| `IM_START`            | `<|im_start|>`      | 151644  | Start of every chat turn             |
| `IM_END` (=`eos`)     | `<|im_end|>`        | 151645  | End of turn **and** end-of-sequence  |
| `VISION_START`        | `<|vision_start|>`  | 151652  | Vision segment prefix                |
| `VISION_END`          | `<|vision_end|>`    | 151653  | Vision segment suffix                |
| `IMAGE_PAD`           | `<|image_pad|>`     | 151654  | Single image patch token             |

`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=…)` automatically wraps every turn:
```
<|im_start|>ROLE\nCONTENT<|im_end|>
```
No extra `<|endoftext|>` is ever appended.

### Vision Token Expansion
At preprocessing time each raw `<IMAGE>` placeholder is replaced by:
```
<|vision_start|>  +  <|image_pad|> × N  +  <|vision_end|>
```
`N = (H/merge) × (W/merge)` after the *smart-resize & patch-merge* pipeline – guaranteed to match the later `pixel_values` length.

---
## 2. Pre-/Post-Processing Pipeline
1. **ChatProcessor**
   - `_create_conversation_messages` builds roles & placeholders.
   - `_process_images_and_tokens` loads JPEGs from `ds_rescaled/`, expands vision tokens, records `image_grid_thw`.
   - `_tokenize_conversation` builds `input_ids`, `attention_mask`, and `labels`.
2. **Batch Collator**
   - Pads sequences, stacks `pixel_values`, passes `ground_truth_objects` untouched (for loss).
3. **Inference** (`src/inference.py`)
   - Same processor path but with `add_generation_prompt=True` → prompt ends on `<|im_start|>assistant\n`.
   - `predict_detection` then feeds through detection head for structured output.

---
## 3. Sanity Checks & Fail-Fast Guards
The following asserts trigger *before* the model forward, catching 99 % of user-side issues:
- **Vision/Text Length** – number of `<|image_pad|>` tokens in prompt == `pixel_values` patches.
- **Special-Token Sync** – runtime converts every string in `SpecialTokens.TOKEN_STRINGS` back to id and matches `SpecialTokens.TOKEN_IDS`.
- **Image Size** – raw PIL size must already respect `[min_pixels, max_pixels]` from `vision_process.py`. If not, raise — never rescale here.

---
## 4. Typical Symptoms & Root Causes
| Symptom                                  | Likely Cause & Fix                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| Repeated "BOX" text, no coordinates      | Called `model.generate` instead of `predict_detection`.                 |
| All boxes identical / at centre          | BBox bias incorrect – re-init via `DetectionHead.reinit_bbox_bias()`.   |
| Vision token mismatch assertion fails     | Mismatching `ds_rescaled/*` vs. old images; regenerate data.            |
| Chinese chars render as squares in viz   | Pass a font path with Chinese support (`--font_path` CLI arg).          |
| Caption loss ≫ bbox loss after epoch 1   | Likely `detection_caption_weight` too high; tune down.                  |

---
## 5. Minimum Working Example (MWE)
```python
from src.inference import Qwen25VLInference
from PIL import Image

predictor = Qwen25VLInference('outputs/exp42/checkpoint')
imgs = [Image.open('ds_rescaled/000123.jpeg')]
prompt = "请描述图中每一个目标的位置和状态。"
boxes, captions = predictor.predict_detection(imgs, prompt)
# → boxes: Tensor[N,4] in absolute px, captions: List[str]
```

---
## 6. Debug Tips
1. **Print a sample conversation** – `BBUTrainer` dumps the first train & eval sample (with vision tokens). Verify visually.
2. **Check grad-norms** – in TensorBoard ensure `gn/vision_adapter` > 0 after detectors unfreeze.
3. **Visualise outputs early** – run `scripts/vis_predictions.py` (make sure to `--font_path "NotoSansCJK-Regular.otf"`).
4. **Enable `torch.autograd.detect_anomaly()`** for one batch if NaNs appear.

---
Everything here aims to keep runtime robust and failures loud. When in doubt, *fail fast* and inspect assertions rather than silently patching data. 🚦 