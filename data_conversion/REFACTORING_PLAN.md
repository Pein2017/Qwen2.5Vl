### Data Conversion Refactor Plan (Non‑breaking)

Purpose: keep the codebase concise while leaving the core pipeline and CLI unchanged. Changes are additive and internal first; existing public entry points continue to work.

---

### Goals and constraints
- **Primary goals**:
  - **Extract common structure** used across `unified_processor.py`, `coordinate_manager.py`, `vision_process.py`, and helpers.
  - **Remove duplication** (EXIF handling, smart-resize scaling, occlusion stripping, object sorting, constants).
  - **Isolate concerns**: orchestration vs. geometry vs. image I/O vs. validation vs. taxonomy/teacher selection.
- **Constraints**:
  - **Core pipeline unchanged**: `unified_processor.UnifiedProcessor` behavior and outputs remain identical.
  - **CLI compatibility preserved**: `convert_dataset.sh` and `python data_conversion/unified_processor.py ...` keep the same arguments and defaults.
  - **Incremental**: introduce modules first, then redirect callers to them via thin wrappers.

---

### Current pain points (where redundancy exists)
- **EXIF handling duplicated**:
  - `vision_process.ImageProcessor.to_rgb` applies `ImageOps.exif_transpose()`.
  - `coordinate_manager.get_exif_transform_matrix()` also computes EXIF-oriented dimensions with `ImageOps.exif_transpose()`.
  - A legacy `strip_exif_orientation.py` exists as a stand‑alone tool.
- **Smart‑resize scaling is split**:
  - `vision_process.smart_resize()` is the source of truth.
  - `coordinate_manager` imports it in multiple places; keep that single origin.
- **Object sorting logic duplicated**:
  - Sorting in `unified_processor` (top‑to‑bottom, then left‑to‑right) and utilities in `core_modules.ObjectProcessor`.
- **Occlusion stripping logic scattered**:
  - `unified_processor._sanitize_description` implements token filtering; CLI flag is `--strip_occlusion` (mapped to `remove_occlusion_tokens` in config).
- **Object‑type constants spread**:
  - Allowed set defined in `config.DataConversionConfig._validate_parameters()`; also referenced across docs, README, and comments.
- **Teacher selection nested**:
  - `TeacherSelector` is nested inside `unified_processor.py` though it’s logically standalone and moderately large.

---

### Target architecture (additive, non‑breaking)
Introduce small, focused modules. Keep existing APIs working by delegating to these.

- `data_conversion/constants.py`
  - **Source of truth** for:
    - `OBJECT_TYPES = {"bbu","bbu_shield","label","fiber","wire","connect_point"}`
    - `DEFAULT_LABEL_HIERARCHY` (the current in‑code fallback in `unified_processor`).
  - Existing modules import from here; keep old fallbacks temporarily for safety.

- `data_conversion/utils/exif_utils.py`
  - Centralize EXIF logic:
    - `apply_exif_orientation(pil_image: Image.Image) -> Image.Image`
    - `get_exif_transform(path: Path) -> tuple[bool, int, int, int, int]`
  - `coordinate_manager.get_exif_transform_matrix()` and `vision_process.ImageProcessor.to_rgb()` delegate to this to avoid divergence.

- `data_conversion/utils/sanitizers.py`
  - `strip_occlusion_tokens(text: str) -> str` (Chinese: remove tokens containing "遮挡" across levels, preserving separators).
  - `unified_processor` delegates to this when `config.remove_occlusion_tokens` is true.

- `data_conversion/teacher_selector.py`
  - Move the nested `TeacherSelector` class out of `unified_processor.py` unchanged.
  - In `unified_processor.py`, `from .teacher_selector import TeacherSelector` to keep imports stable.

- `data_conversion/utils/sorting.py` (optional)
  - `sort_objects_tlbr(objects: list[dict]) -> list[dict]` using first coordinate pair, top‑to‑bottom then left‑to‑right.
  - `unified_processor` and `core_modules.ObjectProcessor` both delegate here to avoid drift.

- Keep these intact (only delegate internally):
  - `coordinate_manager.py` as the single point for geometry transforms and canonicalization.
  - `vision_process.py` as the single point for smart resize and image copying/resizing.
  - `validation_manager.py`, `flexible_taxonomy_processor.py`, `utils/file_ops.py` remain as is.

- CLI consolidation later (non‑blocking): `data_conversion/cli.py` provides subcommands `convert`, `validate`, `merge`, `strip-exif`. Existing scripts import the same parser shape; `convert_dataset.sh` remains unchanged, or later calls `python -m data_conversion.cli convert` behind the scenes.

---

### Non‑breaking migration plan (phased)
- **Phase 1: Add modules, wire through wrappers**
  - Add `constants.py` and reference it inside `config.DataConversionConfig` (keep current checks; prefer constants when present).
  - Add `utils/exif_utils.py`; update `coordinate_manager` and `vision_process` to call it internally (keep public methods unchanged).
  - Add `utils/sanitizers.py`; change `unified_processor` to import and call it, leaving `_sanitize_description()` as a thin wrapper or removing it if unused.
  - Extract `TeacherSelector` to `teacher_selector.py`; re‑export via import in `unified_processor`.
  - Optional: add `utils/sorting.py`; call it from `unified_processor` and `core_modules` with identical semantics.

- **Phase 2: Consolidate CLI (no behavior change)**
  - Introduce `cli.py` with a shared `argparse` builder to eliminate duplicate CLI definitions in `unified_processor.py`, `pipeline_manager.py`, `dataset_merger.py`, `strip_exif_orientation.py`.
  - Keep the current entry points; they import the shared parser to construct the same flags and defaults.

- **Phase 3: Deprecate redundant tools (wrappers)**
  - Mark `strip_exif_orientation.py` as a wrapper around `utils/exif_utils.py` (already noted as deprecated in header).
  - Optionally mark parts of `core_modules.py` as deprecated after `utils/sorting.py` replaces its sorting; keep wrappers for backward compatibility.

---

### Invariants to preserve
- **Outputs**: `train.jsonl`, `val.jsonl`, `teacher_pool.jsonl`, `all_samples.jsonl`, `label_vocabulary.json`, and processed `images/` remain byte‑for‑byte identical for the same inputs and configuration.
- **Geometry**: Canonical ordering, clamping, and transformations are unchanged; only implementation is centralized.
- **Sorting**: Object ordering is unchanged (top‑to‑bottom, then left‑to‑right using the first coordinate pair).
- **CLI**: Flags, defaults, and semantics unchanged (`--resize`, `--object_types`, `--val_ratio`, `--max_teachers`, `--seed`, `--strip_occlusion`, etc.).
- **Performance**: No regressions (>5%) on end‑to‑end runtime.

---

### Proposed APIs (thin, delegated)
- `utils/exif_utils.py`
```python
from pathlib import Path
from PIL import Image
from typing import Tuple

def apply_exif_orientation(img: Image.Image) -> Image.Image: ...

def get_exif_transform(path: Path) -> Tuple[bool, int, int, int, int]:
    """Return (is_transformed, orig_w, orig_h, new_w, new_h)."""
```

- `utils/sanitizers.py`
```python
def strip_occlusion_tokens(text: str) -> str: ...
```

- `utils/sorting.py`
```python
def sort_objects_tlbr(objects: list[dict]) -> list[dict]: ...
```

- `constants.py`
```python
OBJECT_TYPES = {"bbu", "bbu_shield", "label", "fiber", "wire", "connect_point"}
DEFAULT_LABEL_HIERARCHY = {
    "螺丝、光纤插头": ["BBU安装螺丝", "BBU端光纤插头"],
    "标签": [],
    "BBU设备": ["华为"],
    "光纤": [],
    "电线": [],
    "挡风板": ["华为"],
}
```

---

### Acceptance criteria
- Identical outputs across a regression suite:
  - Golden fixtures (subset of `ds_v2`) for multiple object‑type subsets, with and without `--resize`, with and without `--strip_occlusion`.
  - Snapshot the produced JSONL files and image dimensions.
- Unit tests for:
  - EXIF transforms (dimension detection; identity on images without EXIF orientation).
  - Smart‑resize boundary conditions (MIN/MAX pixels, factor alignment).
  - Occlusion sanitizer preserving separators and dropping only tokens containing "遮挡".
  - Object sorting stability for ties and edge coordinates.

---

### Rollout and rollback
- Roll out Phase 1 in a single PR; keep wrappers and re‑exports to avoid import breakage.
- If any regression is detected, flip feature flags back by reverting imports to the previous implementations (wrappers make this trivial).

---

### Notes on what will NOT change
- The end‑to‑end invocation and config flow described in the repository Top‑Level Rules remains the same:
  - Data conversion via `bash data_conversion/convert_dataset.sh`.
  - Training and inference entry points unaffected.
