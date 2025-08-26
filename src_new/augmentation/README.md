# Augmentation Module (src_new/augmentation)

Robust, training‑ready data augmentation for detection‑focused VL fine‑tuning. The module is designed around three principles:

- Unified geometry invariants: all quads are canonicalized to start at top‑left and proceed clockwise; coordinates are within bounds and non‑degenerate.
- Pixel–coordinate consistency: geometry changes occur only when pixels are transformed accordingly (no coordinates‑only jitter that can corrupt supervision).
- HF‑first integration: augmentations run before conversation assembly; downstream tokenization and span logic remain unchanged.

---

## What’s Included

- Image‑level geometry
  - AngleRotate (arbitrary degrees): rotates images and corresponding coordinates together.
    - Config: angle sampling (fixed, range, set), `expand` canvas, interpolation, fill color.
- Photometric (image‑only)
  - Albumentations RandAug: safe pool of photometric ops (brightness/contrast, gamma, hue/sat/value, RGB shift, JPEG compression, Gaussian noise, sharpen, Gaussian/Motion blur, coarse dropout) sampled via `SomeOf`.
    - No geometry change; labels untouched.
- Object‑level (v1)
  - Local affine (rotation + translation) applied per object WITH pixels (when enabled).
    - Converts `bbox_2d` to `quad` after transform.
    - Rejects out‑of‑bounds and degenerate (tiny area) quads; optional non‑overlap constraint with IoU threshold and resampling.
    - Simple polygon mask paste with black fill at the source region to reveal displacement.
    - Lines are skipped in v1.

---

## Canonicalization & Guarantees

- Quad ordering: top‑left first, then clockwise (convex‑hull + clockwise enforcement).
- Bounds: every vertex satisfies `0 ≤ x < width` and `0 ≤ y < height`.
- Non‑degenerate: polygon area must be positive (tiny‑area candidates rejected).
- Pixel–coord consistency: per‑object affine runs only when `apply_pixels: true`. If false, it is skipped entirely; no coordinates‑only jitter.

---

## Configuration (YAML)

Below is a representative configuration block (add under your main YAML at `augmentation:`):

```yaml
augmentation:
  enabled: true
  rng_seed: 12345
  apply_to_teachers: false
  lines_policy: drop_objects   # identity|drop_objects|error

  # Image-level rotation
  op:
    sample_mode: uniform_range   # fixed|uniform_range|set
    angle_min_deg: -30
    angle_max_deg: 30
    angles_set_deg: null
    expand: true
    interpolation: bilinear      # nearest|bilinear|bicubic
    fill_color: [0, 0, 0]

  # Photometric RandAug (Albumentations)
  albumentations_rand:
    enabled: true
    apply_prob: 0.8
    num_ops: 2
    magnitude: 0.6               # 0..1, coarse strength scale
    safe_ops_only: true

  # Per-object local affine (v1)
  object_local_affine:
    enabled: true
    apply_pixels: true           # REQUIRED for v1; no coords-only jitter
    per_object_prob: 0.8
    max_rotation_deg: 10.0
    translate_px: 8
    avoid_overlap: true
    iou_thresh: 0.05
    max_resample: 10
```

Notes:
- AngleRotate is required (it’s the image‑level geometry backbone). Albumentations and local‑affine are optional.
- If you set `object_local_affine.enabled: true` but `apply_pixels: false`, the affine path is skipped (no geometry noise).

---

## Integration & Execution Order

Augmentations run inside `src_new/data/dataset.py` before conversation assembly:

1) Load images
2) Image‑level AngleRotate (image + coordinates)
3) Albumentations RandAug (image‑only photometric)
4) Per‑object Local Affine (pixels + coordinates) if `apply_pixels: true`
5) Conversation building and tokenization

Teacher samples are not augmented by default (configurable via `apply_to_teachers`).

Seeding: deterministic per worker and per sample (`rng_seed` mixed with worker id and sample index).

---

## Implementation Highlights

- AngleRotate (arbitrary degrees)
  - Computes rotated canvas size (when `expand: true`) and applies rotation to both image and object points.
  - `bbox_2d` is turned into 4 corners, rotated, then canonicalized to `quad`.

- Albumentations RandAug (safe pool)
  - Uses `albumentations.SomeOf` to sample `num_ops` photometric ops with probability `apply_prob` per image. No geometry changes.

- Object Local Affine (v1)
  - For each non‑line object (with probability `per_object_prob`):
    - Sample small rotation around object centroid and translation (in pixels).
    - Reject out‑of‑bounds vertices and degenerate area; optional non‑overlap using IoU threshold with resampling.
    - Update coordinates (quad), paste polygon‑masked pixels at the new location, and black‑fill the original region.
    - Lines are skipped in v1.

---

## Invariants After Any Augmentation

- Quads are canonicalized: top‑left → clockwise.
- Every vertex is an integer pixel within image bounds.
- Polygons are convex and have positive area (invalid candidates are rejected).
- No coordinates‑only jitter in the pipeline.

---

## Debugging & Visualization

- Real‑data visualization script: `src_new/tests/vis_aug_angle_rotate_real.py`
  - Produces a small grid showing baseline, strong AngleRotate, Albumentations, and object‑affine panels.
  - Outputs in `outputs/aug_vis/aug_vis_*.jpg`.

---

## Dependencies

- Albumentations (tested: 1.3.x)
- albucore (<0.1) and opencv‑python‑headless (~4.10) for 1.3.x

These are installed in the `ms` environment as part of the debug setup. You can pin or adjust versions as needed in your environment.

---

## Extending

- Albumentations pool: add/remove photometric ops (e.g., CLAHE, RandomToneCurve) and tune strength in `_build_albumentations_policy`.
- Object local affine:
  - Add z‑order and occlusion metadata when enabling overlaps.
  - Feather polygon edges for smooth blending.
  - Add multi‑image support if your samples include multiple images.

---

## Troubleshooting

- “Crossed” quads: rejected by convex‑hull canonicalizer; if this occurs before augmentation (raw data), validate upstream and fix in data conversion.
- No visible object movement: ensure `object_local_affine.apply_pixels: true`, and increase `max_rotation_deg`/`translate_px` for debug.
- Coordinates changing without pixels: not possible in v1; the pipeline skips affine when `apply_pixels` is false.
