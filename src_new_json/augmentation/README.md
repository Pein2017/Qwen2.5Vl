# Augmentation Module (src_new_json/augmentation)

This module provides training‑ready data augmentation for our detection‑focused VL pipeline with type‑aware safeguards. The current implementation focuses on robust, deterministic image/line augmentations with OCR protection. Occlusion detection has been removed.

- Unified geometry invariants: quads are canonicalized to start at top‑left and proceed clockwise; coordinates are in bounds and non‑degenerate.
- Pixel–coordinate consistency: geometry changes happen only when pixels are transformed accordingly.
- HF‑first integration: augmentations run before conversation assembly; downstream tokenization/span logic is unchanged.

---

## Quick start (enable in YAML)

1) Enable with a preset in your training YAML:
```yaml
augmentation:
  preset: moderate   # off|conservative|moderate|aggressive
  rng_seed: 12345
  apply_to_teachers: false
  lines_policy: transform   # identity|drop_objects|error|transform
  debug_visualization: false
  debug_output_dir: null
```

2) Launch training:
```bash
bash scripts/run_new_train.sh
```

3) What it does now:
- ImageGeom: rotation about the image center (pixels + coordinates)
- Photometric: Albumentations image‑only pool; OCR‑safe when labels are present
- Line ops: jitter and equidistant resample for fiber/wire
- Criteria: occlusion detection (removed)

Tip: keep validation clean by disabling augmentation for val runs (separate config or toggle off).

---

## Preset‑based configuration

Presets map to a validated `AugmentationConfig`:
- off: augmentation disabled
- conservative: light photometric + small geometry + line jitter
- moderate: balanced default
- aggressive: stronger perturbations

Advanced users can still supply the explicit block (see Configuration) for fine‑grained control.

---

## What’s Included (v1)

- Image‑level geometry
  - Rotation about the image center with automatic canvas expansion.
    - Converts `bbox_2d` to `quad` after transform; `quad` preserved; `line` rotated as polyline.
    - Current implementation uses `image_geom.rotate_deg_range`; other fields are reserved for future.

- Photometric (image‑only)
  - Albumentations SomeOf pool: brightness/contrast, gamma, HSV, RGB shift, sharpen, plus JPEG/noise/blur/dropout when not OCR‑guarded.
  - OCR guard: if any object desc contains “标签” and `ocr.label_protect` is true, a safer pool is used automatically.

- Line ops (fiber/wire)
  - Per‑point jitter within bounds, then equidistant resample to a fixed number of points.
  - Minimum length gate avoids perturbing tiny segments.

- Criteria
  - Occlusion detection: removed.

- Type policies (placeholders for future object‑level ops)
  - Per‑type policy table is included and validated. It’s used for policy checks and forward‑compatibility, but object‑level move/copy/blur are not executed in v1.

---

## Canonicalization & Guarantees

- Quad ordering: top‑left first, then clockwise.
- Bounds: every vertex satisfies `0 ≤ x < width` and `0 ≤ y < height`.
- Non‑degenerate: polygons have positive area; tiny/invalid candidates are rejected.
- Pixel–coord consistency: geometry ops transform pixels and coordinates together.

---

## Configuration (YAML)

Representative explicit configuration block under `augmentation:`:
```yaml
augmentation:
  enabled: true
  rng_seed: 12345
  apply_to_teachers: false
  lines_policy: transform   # identity|drop_objects|error|transform

  # Image-level geometry (v1 uses rotation; other fields reserved)
  image_geom:
    rotate_deg_range: [-10, 10]
    translate_pct: 0.10
    scale_range: [0.9, 1.1]
    perspective_pct: 0.03
    crop_pct: 0.05
    multiscale_short_edges: [896, 1024, 1280]

  # Photometric RandAug (Albumentations)
  photometric:
    enabled: true
    apply_prob: 0.8
    num_ops: 2
    magnitude: 0.5
    ocr_safe_pool: true

  # Line jitter + resample (fiber/wire)
  lines:
    enabled: true
    jitter_px_minmax: [2, 6]
    resample_points: 32
    min_length_px: 24

  # OCR protection
  ocr:
    label_protect: true
    force_unreadable_on_strong_distortion: true

  # Criteria (occlusion detection removed)
```

Notes:
- `image_geom` currently applies rotation; other fields are validated but not yet active.
- Photometric uses Albumentations 1.3.x; OCR‑safe pool is applied when label objects exist and `ocr.label_protect` is on.
- Lines are never moved/copied; only jitter+resample is applied when enabled.

---

## Integration & Execution Order

Augmentations run inside `src_new_json/data/dataset.py` before conversation assembly:

1) Load images
2) Image‑level rotation (image + coordinates)
3) Albumentations photometric (image‑only) with OCR guard
4) Line ops (fiber/wire) jitter + resample
5) Conversation building and tokenization

Teacher samples are not augmented by default (`apply_to_teachers: false`).
Seeding is deterministic per worker and per sample (`rng_seed` mixed with worker id and sample index).

---

## Implementation Highlights

- Rotation
  - Expanded canvas computed to match PIL’s `expand=True`; coordinates rotated, rounded, clamped; `width/height` updated.
  - `bbox_2d` is converted to `quad` after rotation and canonicalized.

- Photometric
  - `albumentations.SomeOf` samples `num_ops` based on `magnitude`; safe pool auto‑selected when OCR‑guarded.

- Line ops
  - Jitter amplitude sampled in `[jitter_px_minmax]`, then equidistant resample to `resample_points`.

- Criteria
  - Occlusion detection: removed.

---

## Debugging & Visualization

A minimal helper is provided to overlay object geometry for quick inspection:
```python
from PIL import Image
from src_new_json.augmentation.viz_debug import overlay_quads

img = Image.open("/path/to/image.jpg")
objects = sample["objects"]  # after augmentation
vis = overlay_quads(img, objects)
vis.save("/tmp/aug_vis.jpg")
```

---

## Dependencies

- Albumentations (tested: 1.3.x)
- albucore (<0.1) and opencv‑python‑headless (~4.10) for 1.3.x

These are installed in the `ms` conda environment in our debug setup. Adjust versions as needed.

---

## Extending

- Image geometry: enable translate/scale/perspective/crop/multiscale in `image_ops`.
- Type‑aware object‑level ops (planned): cut/move with inpaint (copy‑paste disabled), masked object blur — see `src_new_json/augmentation/bbu_specialized_plan.md`.
- Photometric pool: add safe ops (e.g., CLAHE, tone curve) and tune ranges in `photometric.py`.

---

## Real dataset format & supported types

- Top-level keys per sample:
  - `images`: list of image paths (typically 1 per sample)
  - `objects`: list of labeled geometries
  - `width`, `height`: canvas dimensions in pixels
- Object schema (one geometry key per object + `desc`):
  - `quad`: 8 ints `[x1,y1, x2,y2, x3,y3, x4,y4]`
  - `bbox_2d`: 4 ints `[x1,y1, x2,y2]` (converted to `quad` after rotation)
  - `line`: even-length list of ints (≥4), flattened polyline
  - `desc`: Chinese label string with a type prefix and attributes/tokens
- Recognized type prefixes (via `object_registry`):
  - `BBU设备` → `bbu`
  - `挡风板` → `bbu_shield`
  - `螺丝、光纤插头` → `connect_point`
  - `标签` → `label`
  - `光纤` → `fiber`
  - `电线` → `wire`
- Common tokens in `desc`:
  - Completeness/compliance: e.g., `显示完整`, `只显示部分`, `符合要求`

Example object (abbrev):
```json
{"quad": [335,79,337,671,232,80,225,662], "desc": "BBU设备/中兴,显示完整,机柜空间充足需要安装"}
```

## Compatibility checklist (your data → pipeline)

- Image-level geometry (rotation):
  - Supports `quad`, `bbox_2d`, and `line`. `bbox_2d` is converted to `quad` and all quads are canonicalized after rotation.
  - Your quads that are not TL→CW ordered are canonicalized during the transform.
- Photometric (OCR-safe):
  - Your data includes label objects with `desc` starting `标签/...`; this triggers OCR guard when `ocr.label_protect` is true.
- Line ops (fiber/wire):
  - Your `光纤`/`电线` are provided as `line`; jitter + equidistant resample are applied when `lines.enabled` is true and length ≥ `min_length_px`.
- Criteria (occlusion detection):
  - Removed.
- Type policies:
  - Prefixes in your `desc` map to canonical types (`bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`, `wire`) for policy checks.
- Dimensions:
  - Your samples include `width`/`height` which are required by rotation steps.
- Note:
  - Quad canonicalization occurs during the geometry transform. If you disable `image_geom`, ensure upstream quads are canonicalized.

## Troubleshooting

- Empty objects after augmentation: fail‑fast; check upstream data and bounds.
- Labels become unreadable: with OCR guard on, safer photometric is applied automatically; otherwise tune `magnitude/num_ops`.
- Lines appear jagged: increase `resample_points` or reduce jitter amplitude.
