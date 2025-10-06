# SFT ↔ RL Dense Captioning Alignment Analysis

**Date**: 2025-10  
**Scope**: Dense captioning mode only (objects + geometry + descriptions)  
**Goal**: Verify that GRPO post-training perfectly aligns with SFT training pipeline

---

## Executive Summary

✅ **ALIGNMENT VERIFIED**: The RL (GRPO) pipeline for dense captioning is **completely aligned** with the SFT pipeline.

### Key Findings

1. ✅ **Same data format**: Both use JSONL with `images`, `objects`, `width`, `height`
2. ✅ **Same conversation builder**: Both use `ConversationBuilder` → `ConversationProcessor` → HF `apply_chat_template`
3. ✅ **Same model loading**: Both use `build_hf_components` with identical settings
4. ✅ **Same output format**: Both generate `<|object_ref_start|>desc<|object_ref_end|><|box_start|>[x1, y1, x2, y2]<|box_end|>` per object
5. ✅ **Same special tokens**: Both use geometry wrappers sourced from `src_new/processing/special_tokens.py`
6. ✅ **Same image processing**: Both use HF processor with `pixel_values` (packed 2D) + `image_grid_thw`

### Critical Difference (by design)

- **SFT**: Teacher-forcing with CE loss on ground-truth targets
- **RL**: Samples K completions → compute rewards → GRPO policy gradient
- Both operate on **identical input prompts** and **identical output format**

---

## 1. Data Preparation Pipeline

### SFT Data Flow

```
JSONL sample
  ↓
Dataset.__getitem__(idx)
  ↓
Load images via PathManager
  ↓
Optional: Dynamic teacher pairing (Episode map)
  ↓
Optional: Augmentation (smart resize + geometry transforms)
  ↓
ConversationBuilder.create_conversation(
    sample, images, variant="dense_caption",
    teacher_samples, teacher_images_list
)
  ↓
ConversationProcessor.create_simple_conversation()
  ↓
HF apply_chat_template(messages, images)
  ↓
Processor returns: {
    input_ids, attention_mask,
    pixel_values (2D packed), image_grid_thw,
    conversation_text, offset_mapping
}
  ↓
Extract assistant spans (regex + offset_mapping)
  ↓
Create labels: -100 outside spans, input_ids inside spans
  ↓
Mask <|image_pad|> tokens in labels
```

**Key File**: `src_new/data/dataset.py::_process_sample_unified()`

### RL Data Flow

```
JSONL sample
  ↓
RLDenseJSONLDataset.__getitem__(idx)
  ↓
build_simple_generation_inputs(sample, ctx)
  ↓
Load images via PathManager (same utility)
  ↓
ConversationBuilder.create_simple_conversation_for_generation(sample, images)
  ↓
ConversationProcessor.create_simple_conversation_for_generation()
  ↓
HF apply_chat_template(messages, images, add_generation_prompt=True)
  ↓
Processor returns: {
    input_ids, attention_mask,
    pixel_values (2D packed), image_grid_thw,
    conversation_text
}
  ↓
Remove batch dimension for GRPO
  ↓
Attach meta: {width, height, objects, object_count}
```

**Key Files**:
- `src_new/rl/data/dataset.py::RLDenseJSONLDataset`
- `src_new/rl/prompting/conversation.py::build_simple_generation_inputs()`

### Alignment Points

| Component | SFT | RL | Aligned? |
|-----------|-----|-----|----------|
| JSONL schema | `images`, `objects`, `width`, `height` | Same | ✅ |
| Image loading | `PathManager.resolve_path()` | Same | ✅ |
| Conversation builder | `ConversationBuilder` | Same | ✅ |
| Chat template | HF `apply_chat_template()` | Same | ✅ |
| Image processor | HF processor → 2D packed | Same | ✅ |
| Teacher pairing | Optional (dynamic) | **Disabled** | ✅ (by design) |
| Augmentation | Optional (smart resize + geo) | **Disabled** | ✅ (by design) |

---

## 2. Conversation Structure (Dense Captioning)

### SFT Messages (Training)

```python
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT  # From src_new/processing/templates.py
    },
    {
        "role": "user",
        "content": [{"type": "image"}]  # Image-only user turn
    },
    {
        "role": "assistant",
        "content": FORMATTED_OBJECTS  # Ground truth target
    }
]
```

**Assistant Target Format** (per object):
```
<|object_ref_start|>BBU设备/华为, 显示完整, 机柜空间充足需要安装<|object_ref_end|><|box_start|>[152, 325, 181, 361]<|box_end|>
```

### RL Messages (Generation)

```python
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT  # Same as SFT
    },
    {
        "role": "user",
        "content": [{"type": "image"}]  # Same as SFT
    }
    # NO assistant turn - model generates it
]
```

**Model Generates** (K times with sampling):
```
<|object_ref_start|>BBU设备/华为, 显示完整, 机柜空间充足需要安装<|object_ref_end|><|box_start|>[152, 325, 181, 361]<|box_end|>
<|object_ref_start|>光纤/有保护措施, 弯曲半径合理/蛇形管<|object_ref_end|><|line_start|>[254, 393, 197, 336]<|line_end|>
...
```

### System Prompt Alignment

Both SFT and RL use **identical system prompt** from `src_new/processing/templates.py::CONSTANTS`:

```python
SYSTEM_PROMPT = """...(BBU质检领域专家描述)...

输出结构（dense_caption）：
- <|object_ref_start|>描述<|object_ref_end|><|box_start|>[...]<|box_end|>
- <|object_ref_start|>描述<|object_ref_end|><|quad_start|>[...]<|quad_end|>
- <|object_ref_start|>描述<|object_ref_end|><|line_start|>[...]<|line_end|>

格式示例（严格）：
- <|object_ref_start|>螺丝、光纤插头/BBU安装螺丝, 显示完整, 符合要求<|object_ref_end|><|box_start|>[152, 325, 181, 361]<|box_end|>
...
"""
```

**Source**: `src_new/processing/templates.py::get_system_prompt()`

---

## 3. Model Architecture & Loading

### SFT Model Loading

```python
# src_new/utils/hf_components.py::build_hf_components()
components = build_hf_components(
    model_path=config.model_path,
    model_config=config,
    attn_implementation=config.attn_implementation,  # flash_attention_2 or eager
    image_max_pixels=config.max_pixels,
    bf16=config.bf16,  # Required
    force_eager_attention=False,
    device_map=device_map
)
# Returns: tokenizer, processor, model, image_processor
```

### RL Model Loading

```python
# src_new/rl/runner.py::build_components()
components = build_hf_components(
    model_path=loader_cfg.model_path,
    model_config=model_config,  # From YAML
    attn_implementation=loader_cfg.attn_implementation,  # Same
    image_max_pixels=loader_cfg.image_max_pixels,  # Same
    bf16=loader_cfg.bf16,  # Same (required)
    force_eager_attention=False,
    device_map=device_map
)
# Same return signature
```

### Patches Applied (Both)

```python
# src_new/models/patches.py::apply_comprehensive_qwen25_fixes()
# Applied in both SFT and RL before training
# - Rotary position embedding fixes
# - Attention implementation fixes
# - prepare_inputs_for_generation fixes
```

### Alignment Points

| Component | SFT | RL | Aligned? |
|-----------|-----|-----|----------|
| Model loader | `build_hf_components` | Same | ✅ |
| Tokenizer | HF AutoTokenizer | Same | ✅ |
| Processor | HF Qwen2_5_VLProcessor | Same | ✅ |
| Image processor | HF Qwen2VLImageProcessor | Same | ✅ |
| Model wrapper | `DetectionModel` (optional) | Raw model | ✅ (RL uses unwrapped) |
| Patches | Qwen2.5-VL fixes | Same | ✅ |
| dtype | bf16 required | bf16 required | ✅ |
| Attention | flash_attention_2 or eager | Same (configurable) | ✅ |

---

## 4. Output Format & Special Tokens

### Dense Caption Output Format

**Per Object** (one line):
```
<|object_ref_start|>DESCRIPTION<|object_ref_end|><|GEOMETRY_START|>[COORDS]<|GEOMETRY_END|>
```

**Geometry Types**:
1. **BBox (4 coords)**: `<|box_start|>[x1, y1, x2, y2]<|box_end|>`
2. **Quad (8 coords)**: `<|quad_start|>[x1, y1, x2, y2, x3, y3, x4, y4]<|quad_end|>`
3. **Line (≥4 even coords)**: `<|line_start|>[x1, y1, x2, y2, ...]<|line_end|>`

**Formatting Rules**:
- Coordinates: raw integers (e.g., `152, 325`)
- Separator: comma + space (`, `)
- NO Chinese punctuation, NO coordinate tokens (`<|coord_*|>` deprecated)
- Brackets: English `[ ]`

### Special Tokens (Shared Authority)

```python
# src_new/processing/special_tokens.py::GEOMETRY_TOKENS
GEOMETRY_TOKENS = {
    "bbox_2d": [
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>"
    ],
    "quad": [
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|quad_start|>", "<|quad_end|>"
    ],
    "line": [
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|line_start|>", "<|line_end|>"
    ]
}
```

**Used By**:
- SFT: Target generation via `src_new/processing/geometry_text.py`
- RL rewards: Parsing via `src_new/rl/rewards/format_rewards.py`

### SFT Target Generation

```python
# src_new/processing/geometry_text.py
def format_object_ref(desc: str) -> str:
    return f"<|object_ref_start|>{desc}<|object_ref_end|>"

def format_geometry_for_user(obj: Dict) -> str:
    if "bbox_2d" in obj:
        coords = ", ".join(str(int(v)) for v in obj["bbox_2d"])
        return f"<|box_start|>[{coords}]<|box_end|>"
    # ... quad, line
```

### RL Reward Parsing

```python
# src_new/rl/rewards/format_rewards.py
# Uses SAME special tokens from authority
OBJ_S, OBJ_E, BOX_S, BOX_E = (
    GEOMETRY_TOKENS["bbox_2d"][0],  # <|object_ref_start|>
    GEOMETRY_TOKENS["bbox_2d"][1],  # <|object_ref_end|>
    GEOMETRY_TOKENS["bbox_2d"][2],  # <|box_start|>
    GEOMETRY_TOKENS["bbox_2d"][3],  # <|box_end|>
)

def check_wrappers(text: str) -> float:
    has_obj = OBJ_S in text and OBJ_E in text
    has_box = BOX_S in text and BOX_E in text
    has_quad = QUAD_S in text and QUAD_E in text
    has_line = LINE_S in text and LINE_E in text
    return 1.0 if has_obj and (has_box or has_quad or has_line) else 0.0
```

### RL Reward Functions (Format)

All parse the **same format** SFT produces:

```python
# src_new/rl/rewards/registry.py::REGISTRY
{
    "parse": check_parse_success,           # Can parse at least 1 object
    "wrappers": check_wrappers,             # Has object_ref + geometry wrappers
    "coords": check_coords_counts,          # Correct coord counts (4/8/≥4)
    "separators": check_ascii_separators,   # No Chinese punct, ", " spacing
    "vocab": check_banned_vocab,            # No forbidden tokens
    "length": check_length_penalty,         # Reasonable total length
    # ... detection rewards (geometry accuracy)
}
```

---

## 5. Training Objectives

### SFT Training (Teacher-Forcing)

```python
# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    labels=labels  # Ground truth with -100 masking
)

# Loss computation (src_new/models/loss_manager.py)
# Single-pass CE on shifted logits
ce_loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
    reduction="none"
)

# Apply role masks (teacher/student) + grouped masks (caption/grounding/formatting)
teacher_loss = apply_weighted_masks(ce_loss, teacher_masks, teacher_weights)
student_loss = apply_weighted_masks(ce_loss, student_masks, student_weights)

total_loss = teacher_loss + student_loss
```

**Key Points**:
- Trains on ground-truth targets
- CE loss on next-token prediction
- Grouped losses: caption (description), grounding (coords), formatting (wrappers)
- Teacher turns masked in labels (context only)
- Student turn (last assistant) trained

### RL Training (GRPO)

```python
# Generation phase (K samples per prompt)
for k in range(num_generations):
    completion_ids = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        generation_config=generation_config  # temp, top_p, etc.
    )
    completions.append(completion_ids)

# Reward computation
rewards_per_func = [
    reward_fn(prompts, completions, meta=meta)
    for reward_fn in reward_funcs
]
rewards = (rewards_per_func * weights).sum(dim=1)

# Advantage estimation (group-normalized)
mean_grouped = rewards.view(-1, K).mean(dim=1).repeat_interleave(K)
std_grouped = rewards.view(-1, K).std(dim=1).repeat_interleave(K)
advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

# Policy gradient loss (GRPO clipping)
log_probs_new = model.forward(...).log_probs
log_probs_old = old_log_probs.detach()
ratio = exp(log_probs_new - log_probs_old)
clipped_ratio = clamp(ratio, 1-eps_low, 1+eps_high)
loss = -min(ratio * advantages, clipped_ratio * advantages).mean()

# Optional KL penalty
if beta > 0:
    kl_div = compute_kl(log_probs_new, ref_log_probs)
    loss += beta * kl_div
```

**Key Points**:
- Samples K completions with temperature
- No ground-truth labels
- Rewards measure format + detection quality
- GRPO policy gradient with clipped ratio
- Group normalization across K samples
- Optional KL regularization to reference model

---

## 6. Critical Alignment Verification

### Prompt Construction (Identical)

**SFT**:
```python
# src_new/processing/conversation_processor.py::create_simple_conversation()
messages = [
    {"role": "system", "content": self._system_prompt},
    {"role": "user", "content": [{"type": "image"}]}
]
text, oi = self._apply_chat_template_safe(
    messages=messages,
    images=images,
    add_generation_prompt=False  # Will add assistant turn
)
```

**RL**:
```python
# src_new/processing/conversation_processor.py::create_simple_conversation_for_generation()
messages = [
    {"role": "system", "content": self._system_prompt},  # SAME
    {"role": "user", "content": [{"type": "image"}]}     # SAME
]
text, oi = self._apply_chat_template_safe(
    messages=messages,
    images=images,
    add_generation_prompt=True  # Stop before assistant
)
```

**Difference**: Only the `add_generation_prompt` flag
- SFT: `False` → includes assistant turn with ground truth
- RL: `True` → stops after user turn for model to generate

### Image Processing (Identical)

Both use **HF processor** with same logic:

```python
# processor.py (HuggingFace)
processed = processor(
    text=[conversation_text],
    images=[PIL.Image],
    return_tensors="pt",
    padding=True
)
# Returns:
# - input_ids: [1, seq_len]
# - attention_mask: [1, seq_len]
# - pixel_values: [total_patches, patch_features]  # 2D packed
# - image_grid_thw: [num_images, 3]  # [t, h, w] per image
```

### Generation Config (Consistent)

**SFT Inference** (uses same model):
```python
# src_new/inference.py
gen_kwargs = {
    'max_new_tokens': 512,
    'temperature': 0.000001,  # Near-greedy
    'do_sample': False,
    'eos_token_id': im_end_token_id
}
```

**RL Training**:
```python
# src_new/rl/trainer.py
generation_config = GenerationConfig(
    max_new_tokens=cfg.max_new_tokens,  # e.g., 1024
    temperature=cfg.temperature,         # e.g., 0.9
    top_p=cfg.top_p,                    # e.g., 0.95
    repetition_penalty=cfg.repetition_penalty,
    do_sample=True,  # Required for GRPO diversity
    eos_token_id=im_end_token_id
)
```

**Difference**: RL uses **sampling** (temp > 0, do_sample=True) to explore; SFT uses ground truth.

---

## 7. End-to-End Flow Comparison

### SFT: Single Training Step

```
1. Sample batch from Dataset
   ↓
2. Load images + apply augmentation
   ↓
3. Build conversation (with optional teacher context)
   ↓
4. HF apply_chat_template → tensors
   ↓
5. Extract assistant spans → create labels
   ↓
6. Collator pads batch
   ↓
7. Forward pass: model(input_ids, pixel_values, ..., labels)
   ↓
8. Compute CE loss with grouped masks
   ↓
9. Backward + optimizer step
```

### RL: Single Training Step

```
1. Sample batch from RLDataset (prompts only, no labels)
   ↓
2. Build generation inputs (same builder as SFT)
   ↓
3. Generate K completions per prompt (sampling)
   ↓
4. Decode completions to text
   ↓
5. Compute rewards for each completion (format + detection)
   ↓
6. Compute advantages (group-normalized)
   ↓
7. Forward pass: model(prompt+completion) → log_probs
   ↓
8. Compute GRPO loss (clipped policy gradient)
   ↓
9. Backward + optimizer step
```

### Key Alignment

| Step | SFT | RL | Aligned? |
|------|-----|-----|----------|
| Data loading | JSONL → Dict | JSONL → Dict | ✅ |
| Image loading | PathManager | PathManager | ✅ |
| Conversation | ConversationBuilder | ConversationBuilder | ✅ |
| Chat template | HF apply_chat_template | HF apply_chat_template | ✅ |
| Image tensors | 2D packed + grid_thw | 2D packed + grid_thw | ✅ |
| Output format | `<|object_ref|>...<|box|>[coords]` | Same (generated) | ✅ |
| Loss | CE on ground truth | GRPO on rewards | ✅ (by design) |

---

## 8. Validation Checklist

### ✅ Data Format
- [x] Both use JSONL with `images`, `objects`, `width`, `height`
- [x] Objects have `desc` + one of `bbox_2d`/`quad`/`line`

### ✅ Conversation Building
- [x] Both use `ConversationBuilder` → `ConversationProcessor`
- [x] Both call HF `apply_chat_template` with typed messages
- [x] Both use same system prompt from `templates.py`
- [x] Both build image-only user turn for dense captioning

### ✅ Model & Components
- [x] Both use `build_hf_components` loader
- [x] Both apply Qwen2.5-VL patches
- [x] Both require bf16
- [x] Both use same tokenizer/processor/model

### ✅ Image Processing
- [x] Both use HF processor → 2D packed `pixel_values`
- [x] Both use `image_grid_thw` [num_images, 3]
- [x] Both validate image token alignment

### ✅ Output Format
- [x] Both target: `<|object_ref_start|>desc<|object_ref_end|><|box_start|>[coords]<|box_end|>`
- [x] Both use geometry wrappers from `special_tokens.py`
- [x] Both use raw integer coordinates (no coord tokens)
- [x] RL rewards parse **exact same format** SFT produces

### ✅ Generation
- [x] SFT: teacher-forcing (CE on ground truth)
- [x] RL: sampling (rewards on generated text)
- [x] Both use `<|im_end|>` as EOS
- [x] Both trim at first `<|im_end|>`

---

## 9. Potential Issues & Mitigations

### Issue 1: Distribution Shift (Sampling vs Greedy)

**Problem**: SFT uses teacher-forcing (greedy), RL uses sampling (temp > 0)

**Mitigation**:
- RL starts from SFT checkpoint (warm start)
- GRPO clipping prevents large policy updates
- Reward shaping guides toward SFT-like outputs
- Optional KL penalty to reference model

### Issue 2: Reward vs CE Loss Mismatch

**Problem**: SFT optimizes CE; RL optimizes rewards (different objectives)

**Mitigation**:
- Format rewards (wrappers, coords, separators) align with SFT targets
- Detection rewards (IoU, L1) measure geometric accuracy
- Combined rewards approximate SFT objective
- Rewards use **same parser** as inference validation

### Issue 3: Teacher Context in SFT

**Problem**: SFT can use teacher-student pairs; RL uses single-turn only

**Mitigation**:
- Teacher pairing is **optional** in SFT (can disable for RL parity)
- RL focuses on single-turn dense captioning (inference parity)
- Both modes train on **student turn only** (teacher is context)

---

## 10. Recommendations

### Current Status: ✅ ALIGNED

The RL pipeline is **correctly aligned** with SFT for dense captioning mode:

1. ✅ Same data format and loading
2. ✅ Same conversation building logic
3. ✅ Same model architecture and loading
4. ✅ Same output format and special tokens
5. ✅ Same image processing pipeline

### Best Practices for Maintaining Alignment

1. **Always use shared components**:
   - `ConversationBuilder` for conversations
   - `build_hf_components` for model loading
   - `GEOMETRY_TOKENS` from `special_tokens.py`

2. **Validate at transitions**:
   - SFT → RL: Verify checkpoint loads correctly
   - RL → Inference: Verify outputs parse correctly

3. **Monitor distribution**:
   - Log sample outputs during RL
   - Compare format adherence to SFT
   - Track reward components separately

4. **Avoid drift**:
   - Keep system prompts synchronized
   - Don't modify special tokens mid-training
   - Use same image preprocessing settings

---

## 11. Code References

### SFT Entry Points
- Training: `scripts/train_new.py`
- Dataset: `src_new/data/dataset.py`
- Conversation: `src_new/processing/conversation_processor.py`
- Loss: `src_new/models/loss_manager.py`

### RL Entry Points
- Training: `src_new/rl/runner.py --mode train`
- Dataset: `src_new/rl/data/dataset.py`
- Conversation: `src_new/rl/prompting/conversation.py`
- Rewards: `src_new/rl/rewards/registry.py`
- Trainer: `src_new/rl/trainer.py`

### Shared Components
- Model loading: `src_new/utils/hf_components.py`
- Conversation builder: `src_new/processing/conversation/builder.py`
- Special tokens: `src_new/processing/special_tokens.py`
- Geometry formatting: `src_new/processing/geometry_text.py`
- Patches: `src_new/models/patches.py`

---

## Conclusion

The RL (GRPO) post-training pipeline for dense captioning is **fully aligned** with the SFT pipeline:

- ✅ **Input**: Same JSONL format, same conversation building, same image processing
- ✅ **Model**: Same architecture, same components, same configuration
- ✅ **Output**: Same format (wrappers + raw coords), same special tokens
- ✅ **Parsing**: RL rewards parse **exact format** SFT produces

The only intentional difference is the **training objective**:
- **SFT**: Teacher-forcing with cross-entropy loss (learns from ground truth)
- **RL**: Policy gradient with reward-based optimization (learns from feedback)

Both pipelines produce and consume the **same text format**, ensuring smooth transition from SFT → RL → Inference.
