## Final Object‑Aware Augmentation Refactor Plan (Merged A + pragmatic B)

### Why this plan
- Baseline: Plan A (fail‑fast, YAML‑explicit, deterministic, minimal churn)
- Borrowed from Plan B: thin object‑type resolver/registry and a backward‑compat factory (no heavy strategy/parallel layers)

### Goals
- Simple and disentangled: image‑level, line‑level, object‑level, and policy/validators are separate.
- Efficient: single pass, cached photometric policy, one occupancy mask, no per‑object rebuilds.
- Deterministic: seed = rng_seed ^ (worker_id<<16) ^ sample_index; no thread races.
- Fail‑fast: strict validation and conditional‑attribute guards aligned with `data_conversion/hierarchical_attribute_mapping.json`.

### Background (Data objects and attributes)
- Source specs:
  - `data_conversion/hierarchical_attribute_mapping.json` (authoritative hierarchy, Chinese labels, geometry constraints, hierarchy_template, conditional attributes)
  - `data_conversion/attribute_taxonomy.json` (aliases, supported geometries, content keys)
- Supported object types and geometries (must not be violated by augmentation):
  - `bbu`（BBU设备）: `quad`, `bbox_2d`
  - `bbu_shield`（挡风板）: `quad`, `bbox_2d`
  - `connect_point`（螺丝、光纤插头）: `quad`, `bbox_2d`
  - `label`（标签）: `quad`, `bbox_2d`
  - `fiber`（光纤）: `line`
  - `wire`（电线）: `line`
- Key hierarchy templates (desc 拼装规则，逗号分隔同级，斜杠分隔层级；可选项用方括号):
  - `bbu`: `BBU设备,brand,completeness,windshield_requirement/[windshield_conformity]/[special_text]`
  - `bbu_shield`: `挡风板,brand,completeness,obstruction,direction/[special_text]`
  - `connect_point`: `螺丝、光纤插头,type,completeness,compliance/[specific_issues]/[special_text]`
  - `label`: `标签/text_content`
  - `fiber`: `光纤,obstruction,protection,bend_radius/[protection_details]/[special_text]`
  - `wire`: `电线,obstruction,organization/[special_text]`
- Conditional attributes（必须由上游条件触发，否则不得出现）:
  - `connect_point.specific_issues` 仅当 `compliance=不符合要求`
  - `fiber.protection_details` 仅当 `protection=有保护措施`
  - `bbu.windshield_conformity` 仅当 `windshield_requirement=机柜空间充足需要安装`
- OCR/text:
  - `label.text_content` 为自由文本；若增强使其不可读，训练支路将标签置为“不能”。
- Obstruction tokens（遮挡一致性）:
  - 目前增广阶段仅做遮挡监控，不修改 `desc`（数据预处理阶段已剥离相关标记）。
- Content key bridge（常用字段，便于在代码中定位和更新）:
  - BBU: `bbu_brand`, `bbu_stituation`, `bbu_equipment`（及其一致性字段）
  - 挡风板: `bbu_shield_brand`, `bbu_shield_situation`, `bbu_shield_cover`, `bbu_shield_install_direction`
  - 连接点: `connect_point_type`, `connect_point_situation`, `connect_point_check`, `specific_issues`
  - 光纤: `fiber_cover`, `fiber_protection`, `fiber_bend_radius`, `protection_details`
  - 电线: `wire_cover`, `wire_organization`
  - 通用: `special_situation`
- Type resolver hints（降噪与匹配）:
  - 依据 `desc` 前缀中文标签匹配类型：`BBU设备/…` → `bbu`；`挡风板/…` → `bbu_shield`；`螺丝、光纤插头/…` → `connect_point`；`标签/…` → `label`；`光纤/…` → `fiber`；`电线/…` → `wire`
  - 可参考 `attribute_taxonomy.json` 中 `aliases` 作为宽松匹配的备选（仅用于宽容解析，不改变存储标准）。

---

### Architecture (lightweight modules)
- `augmentation/image_ops.py`: rotate, translate/scale, light perspective, safe crop, multi‑scale.
- `augmentation/photometric.py`: albumentations policy factory (general vs OCR‑safe).
- `augmentation/line_ops.py`: control‑point jitter, equidistant resample (N=32), bend‑radius estimation.
- `augmentation/object_ops.py`: cut/move (with pixels + inpaint), copy‑paste (alpha feather), object blur.
- `augmentation/policies.py`: type/attribute policy tables, OCR guard, thresholds.
- `augmentation/validators.py`: geometry bounds, conditional attributes, desc normalization.
- `augmentation/compose.py`: one small orchestrator to run stages.
- Borrow from B (thin only):
  - `augmentation/object_registry.py` (helper): resolve object type from `desc` (e.g., 前缀“BBU设备/…”, “挡风板/…”, “螺丝、光纤插头/…”, “标签/…”, “光纤/…”, “电线/…”)。
  - `augmentation/__init__.py` factory: legacy vs object‑aware pipeline switch (backward compatible).

Keep `augmentation/base.py` as a thin façade delegating to `compose.py` (for current call sites).

---

### Config surface (YAML‑first, explicit; no hidden defaults)
Extend `src_new_json/config/augmentation_config.py` with minimal new blocks (preserve existing names):
- Image geometry
  - `ImageGeomConfig`: `rotate_deg_range`, `translate_pct`, `scale_range`, `perspective_pct`, `crop_pct`, `multiscale_short_edges`.
- Photometric
  - `PhotometricConfig`: `enabled`, `apply_prob`, `num_ops`, `magnitude`, `ocr_safe_pool` (bool).
- Lines
  - `LineAugConfig`: `enabled`, `jitter_px_minmax`, `resample_points` (e.g., 32), `min_length_px`.
- Per‑type policy (object‑level)
  - `TypePolicyConfig` map keyed by {`bbu`,`bbu_shield`,`connect_point`,`label`,`fiber`,`wire`}:
    - `allow_move`, `allow_copy_paste`, `allow_blur`, `occluder_prob`, `inpaint_source`, `max_iou_with_existing`, `max_occ_fraction`, `occ_grid_downscale`, `occ_margin_px`, `same_plane_constraint` (bool), `copy_paste_attempts`, `alpha_feather_px`, `allowed_copy_types`.
- OCR
  - `OCRPolicyConfig`: `label_protect` (True), `force_unreadable_on_strong_distortion` (True).
- Criteria (reuse existing)
  - `OcclusionCriterionConfig` (bbox overlap 0.15–0.25; line width 2–3 px).

Disable `object_transform` wrapper by default in presets; use explicit blocks above.

Minimal YAML example (moderate):
```yaml
augmentation:
  enabled: true
  rng_seed: 12345
  apply_to_teachers: false
  lines_policy: transform

  image_geom:
    rotate_deg_range: [-8, 8]
    translate_pct: 0.10
    scale_range: [0.9, 1.1]
    perspective_pct: 0.03
    crop_pct: 0.05
    multiscale_short_edges: [896, 1024, 1280]

  photometric:
    enabled: true
    apply_prob: 0.8
    num_ops: 2
    magnitude: 0.5
    ocr_safe_pool: true

  lines:
    enabled: true
    jitter_px_minmax: [2, 6]
    resample_points: 32
    min_length_px: 24

  type_policies:
    label:
      allow_move: true
      allow_copy_paste: true
      allow_blur: false
      occluder_prob: 0.2
      inpaint_source: true
      max_iou_with_existing: 0.25
      max_occ_fraction: 0.15
      occ_grid_downscale: 8
      occ_margin_px: 6
      same_plane_constraint: true
      copy_paste_attempts: 20
      alpha_feather_px: 1.5
      allowed_copy_types: ["label"]
    connect_point:
      allow_move: true
      allow_copy_paste: true
      allow_blur: true
      occluder_prob: 0.2
      inpaint_source: true
      max_iou_with_existing: 0.25
      max_occ_fraction: 0.15
      occ_grid_downscale: 8
      occ_margin_px: 6
      same_plane_constraint: true
      copy_paste_attempts: 20
      alpha_feather_px: 1.5
      allowed_copy_types: ["connect_point"]
    bbu:
      allow_move: false
      allow_copy_paste: false
      allow_blur: false
      occluder_prob: 0.1
      inpaint_source: false
      max_iou_with_existing: 0.2
      max_occ_fraction: 0.15
      occ_grid_downscale: 8
      occ_margin_px: 6
      same_plane_constraint: true
    bbu_shield:
      allow_move: true
      allow_copy_paste: false
      allow_blur: false
      occluder_prob: 0.2
      inpaint_source: true
      max_iou_with_existing: 0.25
      max_occ_fraction: 0.15
      occ_grid_downscale: 8
      occ_margin_px: 6
      same_plane_constraint: true
    fiber:
      allow_move: false
      allow_copy_paste: false
      allow_blur: false
      occluder_prob: 0.15
      inpaint_source: false
      same_plane_constraint: true
    wire:
      allow_move: false
      allow_copy_paste: false
      allow_blur: false
      occluder_prob: 0.15
      inpaint_source: false
      same_plane_constraint: true

  ocr:
    label_protect: true
    force_unreadable_on_strong_distortion: true

  criteria:
    occlusion:
      enabled: true
      min_overlap_fraction_bbox: 0.2
      min_overlap_fraction_line: 0.2
      mask_downscale: 8
      line_width_px: 2
```

---

### Policy layer (tables, not code)
- `resolve_object_type(desc)` → {`bbu`,`bbu_shield`,`connect_point`,`label`,`fiber`,`wire`} via中文标签前缀匹配。
- Conditional attributes (from mapping):
  - `connect_point.specific_issues` only if `compliance=不符合要求`.
  - `fiber.protection_details` only if `protection=有保护措施`.
  - `bbu.windshield_conformity` only if `bbu_equipment=机柜空间充足需要安装`.
- OCR guard: if label undergoes strong blur/flare/jpeg that breaks readability, set `label_text_content=不能` in training branch.
- Hflip disabled globally (中文与方向语义)。

---

### Stage pipeline (compose)
1) `image_geom`: rotate (±3–10°), translate/scale, light perspective (2–4%), safe crop (≤5%), multi‑scale.
2) `photometric`: general pool; if any `label` exists and `ocr.label_protect`, use OCR‑safe pool.
3) `line_ops` (fiber/wire): control‑point jitter (±2–6 px), equidistant resample N=32, bend‑radius estimation + relabel。
4) `object_ops.move` (cut/move): for allowed types (`label`, `connect_point`, some `bbu_shield`) with inpaint and same‑plane constraint; IoU/occ gates。
5) `object_ops.copy_paste`: long‑tail only (`label`, `connect_point`), alpha feather 1–2 px, attempts/occ/IoU gates；tag `synthetic=true`。
6) `object_ops.blur`: skip `label` when OCR protect on。
7) `criteria.occlusion`: 仅监控遮挡重叠比例（不修改 `desc`）。
8) `validators`: geometry bounds, quad order, line length阈值, conditional attributes, OCR coherence。

Determinism: one RNG; no parallel per‑object threads.

---

### Type‑specific recipes (关键要点)
- `bbu`:
  - Logo可辨优先；用噪声/压缩/轻模糊，避免强遮挡；不做复制；仅轻视角/光照变化。
- `bbu_shield`:
  - 允许少量 cut/move（同平面、短距离）；方向语义敏感：禁 hflip；遮挡仅监控不改写 `desc`。
- `connect_point`:
  - cut/move（5–15%位移）+ inpaint；copy‑paste仅预热/长尾；`specific_issues`合成（未拧紧/露铜/复接/生锈）。
- `label`:
  - 轻几何/外观；强扰动时置 `text_content=不能`；字体/字号/膜面多样化；copy‑paste 受控。
- `fiber`/`wire`:
  - 只做控制点抖动+等距重采样；不整体平移/复制；弯曲半径程序化生成并重算。

---

### Photometric pools
- General‑safe: brightness/contrast/sat/hsv, gamma, rgb shift, sharpen, mild blur, JPEG 35–95, Gauss noise, coarse dropout, glare (局部)。
- OCR‑safe: same as上，但降低/禁用强模糊与强眩光；JPEG/噪声范围收紧。

---

### Validators & mapping guards
- Geometry: all coords in bounds; quads canonicalized TL→CW; line length ≥ `min_length_px`。
- Conditional attributes: enforce mapping JSON 规则；否则 fail‑fast（指明对象与字段）。
- Desc normalization: 遮挡触发自动替换；合成对象写入 `synthetic/moved_instance/occluded` 元数据。

---

### Presets (conservative/moderate/aggressive)
- Presets only fill explicit blocks; do not use `object_transform` wrapper.
- Conservative: ±15°→缩小到±8°；num_ops=1; magnitude≈0.4; object move off by default。
- Moderate: ±30°→缩小到±10°；num_ops=2; magnitude≈0.5; enable move for label/connect_point (低概率)。
- Aggressive: ±45°→缩小到±15°；num_ops=3; magnitude≈0.7; stronger but keep OCR‑safe when label present。

---

### Backward compatibility (borrowed from B)
- `augmentation/__init__.py`:
```python
def get_augmentation_pipeline(cfg) -> "AugmentationPipeline":
    if is_legacy_config(cfg):
        return LegacyAugmentationPipeline.from_config(cfg)
    return ObjectAwareAugmentationPipeline.from_config(cfg)
```
- Feature flag in YAML: `use_object_aware_augmentation: true|false` (default false at first rollout).

---

### Efficiency checklist
- Build albumentations compose once per worker & tier; reuse.
- Build occupancy mask once per sample (low‑res `occ_grid_downscale`).
- Minimize PIL↔numpy crossings; keep PIL for compositing; batch occluder masks.
- Short‑circuit ops when no eligible objects.

---

### Tests & acceptance
- Unit tests:
  - `test_line_ops.py`: jitter+resample invariants; bend_radius relabel.
  - `test_object_ops.py`: bounds, IoU/occ gates, inpaint, OCR skip.
  - `test_policies.py`: type resolve, conditional guards, OCR guard.
  - `test_compose.py`: stage order, determinism by seed.
- Acceptance:
  - No silent defaults; strict validation.
  - Lines never moved/copied; resampled to fixed N; bend_radius relabeled。
  - OCR‑safe by default; forced “不能” path covered。
  - Non‑degrading val accuracy vs current “moderate”; improved long‑尾 for `specific_issues`/不可读/小弯半径。

---

### Rollout (phased)
1) Phase 1: land module split + composer, parity with current presets (object move off).
2) Phase 2: enable `line_ops` for fiber/wire; validate.
3) Phase 3: enable cut/move for `connect_point`/`label` (+inpaint); copy‑paste off by default.
4) Phase 4: add occluders + OCR guard; tune thresholds.
5) Phase 5: optional copy‑paste for detection warm‑up only; tag `synthetic=true`; keep disabled in main SFT.

---

### Implementation checklist (files)
- New: `augmentation/{image_ops.py, photometric.py, line_ops.py, object_ops.py, policies.py, validators.py, compose.py, object_registry.py}`
- Edit: `augmentation/base.py` (delegate), `augmentation/presets.py` (explicit blocks, disable wrapper), `augmentation/README.md` (YAML examples), `config/augmentation_config.py` (new dataclasses), `augmentation/__init__.py` (factory)
- Assets (optional): `data/occluders/` RGBA cutouts

This merged plan is minimal, executable, and aligned with your mapping JSON and training goals. It keeps code simple and deterministic while adding just enough structure for per‑类型/属性 control and future growth.