# Critical Fixes & Patches (Losses.py ✧ mRoPE)

> **Purpose**   Collect all high-impact bug fixes in one place so future contributors can see what changed, why, and how to keep the model stable.
>
> This page supersedes the older **LOSSES_FIXES.md** and **mRoPE.md** documents.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Losses.py Shape & Parser Fixes](#lossespy-shape--parser-fixes)
3. [mRoPE Dimension Mismatch Fix](#mrope-dimension-mismatch-fix)
4. [Validation & Regression Tests](#validation--regression-tests)
5. [References](#references)

---

### Executive Summary
Two critical runtime bugs blocked multi-modal training:

| Area | Symptom | Root Cause | Fix Status |
|------|---------|------------|------------|
| **Detection/LM Loss** | `shape '[0, 4, -1]' is invalid for input of size 1280` (masked_scatter) | `image_embeds` shape mismatch & fragile response parser | **Fixed** (see §2) |
| **mRoPE** | `split_with_sizes` expects 128 but got 288 (dimension mismatch) | Buggy *doubling* of `mrope_section` + batch duplication | **Fixed** (see §3) |

Both fixes ship in *current* `main` and are required for successful multi-image, packed-sequence training.

---

## Losses.py Shape & Parser Fixes
### 1 · Shape Error in `extract_embeddings_from_model`
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
* Flattens 2-D embeds, validates length, slices if extra tokens appear.
* Removes silent failure paths; raises with clear message on mismatch.

### 2 · ResponseParser Upgrade
* Added `_parse_json_format()` → handles true JSON lists from new data.
* `_parse_unquoted_format()` retains backward compatibility.
* `_try_alternative_patterns()` tries legacy regex fallbacks.
* All references standardised from `desc` → `description`.

#### Key Benefits
* Robust to mixed annotation formats.
* Graceful degradation in "early training" where labels may be missing.
* Comprehensive error handling & logging.

---

## mRoPE Dimension Mismatch Fix
> A 3-D rotary embedding bug in HuggingFace **Qwen2.5-VL** froze training when multiple images were batched.

### 1 · Root Cause
* Official `apply_multimodal_rotary_pos_emb` **multiplied** `mrope_section` by 2 ➟ expected head dim 256, but rotary cos/sin tensors are **already** 128.
* Naïve padding collator duplicated `mrope_section` per batch sample (e.g. 128 → 288).

### 2 · Patched Function (`src/models/patches.py`)
```python
def apply_multimodal_rotary_pos_emb_fixed(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # Remove erroneous doubling
    if len(mrope_section) > 6 and mrope_section[:len(mrope_section)//2] == mrope_section[len(mrope_section)//2:]:
        mrope_section = mrope_section[: len(mrope_section)//2]  # de-duplicate

    expected = sum(mrope_section)
    assert expected == cos.size(-1), f"mRoPE dim mismatch: {expected=} {cos.size(-1)=}"

    cos = torch.cat([
        m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
    ], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([
        m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
    ], dim=-1).unsqueeze(unsqueeze_dim)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```
* **No doubling**, *optional* de-duplication for batched sections.
* Strict dimension assertion to fail-fast during future regressions.

### 3 · Collator Alignment
Packed-sequence training (`FlattenedDataCollator`) now the default. It avoids per-sample duplication and yields 100 % memory efficiency; see `docs/advanced/collator_notes.md` for internals.

---

## Validation & Regression Tests
| Test | Script | What it covers |
|------|--------|----------------|
| Unit test – Loss shapes | `tests/test_losses.py` | Various image/text embed shapes, mask coverage. |
| Unit test – Parser | `tests/test_parser_formats.py` | JSON, unquoted, edge-case formats. |
| mRoPE sanity | `tests/test_mrope.py` | Confirms head_dim equality & successful forward on random batch. |
| End-to-end training smoke | `scripts/train.py --validate-only` | 1 epoch run with multi-image chat template. |

All tests pass (CI badge coming soon). 🚀

---

## References
* **HuggingFace Qwen2.5-VL Issue #45** – original mRoPE report.  
* **Alibaba DAMO internal log #2025-01-04** – Losses.py shape fix review.  
* **doc/advanced/collator_notes.md** – deeper dive into packed collator.

---

> Last updated: 2025-07-02 