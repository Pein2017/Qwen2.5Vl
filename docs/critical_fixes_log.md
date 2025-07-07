# 📋 Critical Fixes Log

This document tracks critical bug fixes applied to the codebase, covering both high-level architectural changes and specific implementation patches.

---

## 1. Architectural Fixes

This section covers fundamental fixes to the training and inference architecture that resolved major inconsistencies and performance issues.

### 1.1. Missing Student Loss Backpropagation

**Problem**: Teacher and student losses were computed but **NOT included in backpropagation**.
- `src/training/loss_manager.py:234-235`: `.item()` calls removed gradients from teacher/student losses.
- `src/training/loss_manager.py:132`: `total_loss` only included base LM loss, missing teacher/student components.

**Result**: 
- Teacher loss decreased (tracked for logging only).
- Student loss fluctuated (student never learned through backpropagation).
- Training "converged" (base language modeling worked).
- Inference was poor (student never properly trained).

**Fix Applied**:
- Removed `.item()` calls to preserve gradients: `teacher_loss_tensor`, `student_loss_tensor`.
- Added weighted teacher-student losses to backpropagation: 
  ```python
  total_loss = lm_loss + detection_loss + weighted_teacher_loss + weighted_student_loss
  ```
- Applied config weights: `teacher_loss_weight: 0.3`, `student_loss_weight: 1.0`.

### 1.2. Training-Inference Model Loading Mismatch

**Problem**: Training and inference used **completely different model loading paths**.
- Training: `Qwen25VLWithDetection` wrapper with detection capabilities.
- Inference: Plain `Qwen2_5_VLForConditionalGeneration` without detection.

**Result**: Different model architectures, tokenizer settings, and processing pipelines.

**Fix Applied**:
- Created a **unified model loader** (`src/models/model_loader.py`).
- Both training and inference MUST use an identical loading process.
- Only difference: `for_inference=True/False` (affects padding side only).
- **NO SILENT FALLBACKS** - all errors exposed immediately.

### 1.3. Data Collator Safety Issues

**Problem**: Refactoring changed safe dictionary access to direct key access.
- Changed from `instance.get("teacher_assistant_spans", [])` to `instance["teacher_assistant_spans"]`.
- Caused silent KeyError crashes corrupting training batches.

**Fix Applied**:
- Restored safe dictionary access with defaults in `src/data.py`.
- Prevents silent failures when span data is missing.

### 1.4. Detection Configuration Mismatch

**Problem**: Config had `detection_enabled: false` but dataset contained detection objects.
- Model trained for language modeling but evaluated on detection tasks.

**Fix Applied**:
- Enabled detection in `configs/base_flat_v2.yaml`.
- Added validation script to catch such mismatches.

### 1.5. Flash Attention Padding Incompatibility

**Problem**: Training used `padding_side='right'` but Qwen2.5-VL Flash Attention requires `padding_side='left'`.
- Error: "You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention"
- Training fails with Flash Attention enabled.

**Fix Applied**:
- **ModelFactory**: Added `tokenizer.padding_side = 'left'` in `src/core/model_factory.py`.
- **StandardDataCollator**: Updated to use LEFT padding.
- **position_ids handling**: Updated to match LEFT padding alignment.

---

## 2. Implementation Fixes

This section covers specific runtime bugs that blocked training or caused incorrect behavior.

### 2.1. Losses.py Shape & Parser Fixes

**Problem**: `shape '[0, 4, -1]' is invalid for input of size 1280` (masked_scatter). Caused by an `image_embeds` shape mismatch & a fragile response parser.

**Fix Applied**:
- **Shape Error in `extract_embeddings_from_model`**:
  ```python
  # NEW
  if image_embeds.dim() == 2:
      image_embeds_flat = image_embeds.view(-1)
  else:
      image_embeds_flat = image_embeds.reshape(-1)

  num_mask = image_mask.sum().item()
  assert len(image_embeds_flat) >= num_mask, "Not enough image features to scatter"
  inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_flat[:num_mask])
  ```
  This flattens 2-D embeds, validates length, and raises a clear error on mismatch.
- **ResponseParser Upgrade**:
    - Added `_parse_json_format()` to handle true JSON lists.
    - Retained backward compatibility with `_parse_unquoted_format()`.
    - Added fallbacks and standardized field names.

### 2.2. mRoPE Dimension Mismatch Fix

**Problem**: A 3-D rotary embedding bug in HuggingFace **Qwen2.5-VL** (`apply_multimodal_rotary_pos_emb`) caused a dimension mismatch (`split_with_sizes` expects 128 but got 288) when multiple images were batched.

**Fix Applied**:
- **Patched Function (`src/models/patches.py`)**:
  ```python
  def apply_multimodal_rotary_pos_emb_fixed(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
      # Remove erroneous doubling
      if len(mrope_section) > 6 and mrope_section[:len(mrope_section)//2] == mrope_section[len(mrope_section)//2:]:
          mrope_section = mrope_section[: len(mrope_section)//2]  # de-duplicate

      expected = sum(mrope_section)
      assert expected == cos.size(-1), f"mRoPE dim mismatch: {expected=} {cos.size(-1)=}"
      # ... (rest of the implementation)
  ```
  The fix removes the incorrect doubling of `mrope_section` and adds a strict dimension assertion.
- **Collator Alignment**: The `FlattenedDataCollator` is now the default, which avoids the per-sample duplication that triggered the bug.

### 2.3. Validation & Regression Tests Added

| Test | Script | What it covers |
|------|--------|----------------|
| Unit test – Loss shapes | `tests/test_losses.py` | Various image/text embed shapes, mask coverage. |
| Unit test – Parser | `tests/test_parser_formats.py` | JSON, unquoted, edge-case formats. |
| mRoPE sanity | `tests/test_mrope.py` | Confirms head_dim equality & successful forward pass. |
| End-to-end training smoke | `scripts/train.py --validate-only` | 1 epoch run with multi-image chat template. |

---

## 3. Data & Template Fixes

This section covers fixes related to the data processing pipeline and conversation templates, which resolved critical silent data corruption and training/validation inconsistencies.

### 3.1. Centralized BBox Coordinate Management

**Problem**: Bounding boxes were shifted and misaligned due to multiple, inconsistent scaling operations and unhandled EXIF orientation changes.
- EXIF rotation was applied to images, but not to the corresponding bounding box coordinates.
- Coordinates were scaled multiple times in different parts of the data pipeline.
- Dimension mismatches between JSON metadata and actual image dimensions were not handled correctly.

**Fix Applied**:
- A centralized `CoordinateManager` (`data_conversion/coordinate_manager.py`) was created to handle all bounding box transformations.
- This manager applies a consistent pipeline: EXIF orientation compensation, dimension mismatch rescaling, and smart resize scaling.
- All other modules were updated to use this central manager, eliminating redundant and conflicting logic.

### 3.2. Label Hierarchy Filtering
**Problem**: Valid objects were being silently dropped from the dataset because the `label_hierarchy.json` file was incomplete and did not account for variations in raw data labels.
- For example, `连接点（螺丝）` was present in the data but only `螺丝连接点` was in the hierarchy, causing the object to be filtered out.

**Fix Applied**:
- The `label_hierarchy.json` was updated to include all missing and variant object types and properties, using token-mapped terms to ensure consistency.
- This ensures that all valid objects from the raw data are correctly processed and included in the final training set.

### 3.3. Training/Validation Inconsistency

**Problem**: The model showed a large performance gap between training and validation due to inconsistencies in data distribution and prompt styles.
- **Data Distribution**: Training used a high ratio of "teacher" examples, while validation used none, creating a distribution mismatch.
- **Prompt Style**: Training prompts were verbose and context-rich, while validation prompts were concise, leading to a stylistic mismatch.

**Fix Applied**:
- The data loader was updated to use a consistent teacher ratio for both training and validation by default.
- The chat processor was updated to use a consistent prompt style for both modes by default.
- Configuration options were added to allow for explicitly testing zero-shot validation performance if desired.

### 3.4. Enhanced Conversation Templates

**Problem**: The conversation templates did not clearly differentiate between teacher examples and the final student task, nor did they provide context for the learning objective.

**Fix Applied**:
- System prompts were enhanced with a "learning mode" explanation.
- User messages were updated to clearly label examples, e.g., "📚 参考示例 1/2" for teacher examples and "🎯 现在请...检测以下目标图像" for the student task.
- Meta-learning instructions were added to guide the model on how to learn from the provided examples. 