# BBU Installation Quality Inspection Dataset

## Purpose & Scope
- Vision-language dataset for detecting BBU-related components and assessing installation quality in telecom cabinets.
- Downstream consumer of JSONL produced by `data_conversion/unified_processor.py`. This document is the concise, English-only spec.
- Supports a fourth SFT dialogue variant (optional): image → one-line Chinese summary. Summaries are deterministically distilled from per-object `desc` fields and used only for SFT summary training.

## Files
- Location: `data/ds_v2_full/`
  - `train.jsonl`, `val.jsonl`
  - `teacher_pool.jsonl`
  - `all_samples.jsonl`
- Optional summary SFT files (if generated):
  - `data/summary_sft/train.jsonl`, `data/summary_sft/val.jsonl` (see Summary Field below)

## Record Schema (JSONL per line)
- images: List[str] — processed image paths (usually 1 path per record).
- objects: List[Object]
  - Exactly one geometry field per object:
    - bbox_2d: [x1, y1, x2, y2]
    - quad: [x1, y1, x2, y2, x3, y3, x4, y4]
    - line: [x1, y1, x2, y2, ..., xn, yn]
  - desc: Hierarchical, slash- and comma-separated string built from the templates below. The literals in data are canonical and defined by the taxonomy/mapping; this document explains them in English. A final slash-level free-text "remarks" segment is allowed (see Description Construction).
- width, height: int — processed image size.
- meta: Dict — per-sample metadata (see Meta block).
- summary (optional, SFT-only): str — a single-line Chinese summary distilled from objects; used by the SFT "summary" variant as teacher-forcing target. Per-image only: must NOT include any group-level or pass/fail decisions; must contain no coordinates or special tokens.

## Geometry & Object Types
- Geometry formats
  - box_points: axis-aligned rectangle (two [x,y] pairs: top-left and bottom-right)
  - quad: 4-point polygon
  - line: polyline (fibers/wires)
- Allowed geometry per object type
  - bbu: quad, box_points
  - bbu_shield: quad, box_points
  - connect_point (screws, fiber connectors): quad, box_points
  - fiber: line
  - wire: line
  - label: quad, box_points

## Attributes by Object Type
All values in data are canonical (defined in the taxonomy). Below are their meanings in English.

- bbu
  - brand: {Huawei, ZTE, Ericsson}
  - visibility: {complete, partial}
  - windshield_requirement: {not_required, required}
  - windshield_conformity (only if required): {shield_present_per_requirement, shield_missing_or_nonconformant}
  - remarks (free text, optional) as final slash segment in desc

- bbu_shield
  - brand: {Huawei, ZTE}
  - visibility: {complete, partial}
  - obstruction: {clear, blocked}
  - install_direction: {correct, incorrect}
  - remarks (free text, optional) as final slash segment in desc

- connect_point (screws, fiber connectors)
  - type: {bbu_mount_screw, cabinet_ground_screw, busbar_ground_screw, fiber_connector_bbu_end, fiber_connector_odf_end}
  - visibility: {complete, partial}
  - compliance: {compliant, noncompliant}
  - specific_issues (only if noncompliant; multi-value): {loose, exposed_copper, double_connection, rusty}
  - remarks (free text, optional) as final slash segment in desc

- fiber
  - obstruction: {clear, blocked}
  - protection: {none, protected}
  - protection_details (only if protected): {snake_tube, armored, both}
  - bend_radius: {ok, violation}
  - remarks (free text, optional) as final slash segment in desc

- wire
  - obstruction: {clear, blocked}
  - organization: {neat, disorganized}
  - remarks (free text, optional) as final slash segment in desc

- label
  - text: free text (OCR-derived); may be absent if unreadable

## Description (desc) Construction
- Separators
  - comma ",": separate attributes within a level
  - slash "/": separate hierarchical levels
  - conditional attributes appear only when parent condition is met
  - free text is inserted as-is for labels and for a final optional "remarks" segment
- Object templates (English structure; optional segments in brackets)
  - bbu: "BBU/brand,visibility,windshield_requirement/[windshield_conformity]/[remarks]"
  - bbu_shield: "Shield/brand,visibility,obstruction,install_direction/[remarks]"
  - connect_point: "ConnectPoint/type,visibility,compliance/[specific_issues]/[remarks]"
  - fiber: "Fiber/obstruction,protection,bend_radius/[protection_details]/[remarks]"
  - wire: "Wire/obstruction,organization/[remarks]"
  - label: "Label/[text]"
- Slash-level semantics (positional contract)
  - Level-0 (prefix): object type literal (e.g., "BBU", "Shield", ...)
  - Level-1: comma-joined canonical attributes for the object (e.g., brand, visibility, ...)
  - Level-2: conditional segment only when the parent condition is met (e.g., bbu `windshield_conformity` when requirement=required; connect_point `specific_issues` when compliance=noncompliant; fiber `protection_details` when protected)
  - Level-Last: optional final "remarks" free-text segment. When present, it always occupies the last slash level after all canonical and conditional segments.
- Remarks (free text)
  - Purpose: carry special circumstances that affect QC interpretation (e.g., "cannot determine", "angle issue", "partially sleeved snake tube", "screw missing", "rectified", "space limited").
  - Placement: only as the final slash segment of desc (after all canonical attributes and any conditional segments). Consumers should detect remarks by position (last slash segment), not by keywords.
  - Sanitization: must not contain coordinates or special tokens (no `<|...|>`, `<`, `>`, `[`, `]`); keep short (recommended ≤ 40 Chinese characters or equivalent length).

## Meta (per record)
- tokens: List[str] — normalized vocabulary present in the sample
- object_types_present: List[str]
- geometry_set: List[str] — {bbox_2d, quad, line}
- brand: str — sample-level brand context if determinable; otherwise "unknown"
- object_count: int — number of objects
- small_box_frac: float — fraction of small objects
- line_points: int — total number of polyline points
- line_length_norm: float — normalized polyline length
- has_negative: bool — whether any negative/noncompliant attribute exists
- negatives: List[str] — negative attributes present
- difficulty: str — {easy, medium, hard}

## Validation Rules
- One geometry field per object; geometry must match the object type.
- desc must follow the template structure and separator rules.
- For bbu with windshield_requirement = required, include windshield_conformity.
- For fiber with protection = protected, include protection_details.
- For connect_point with compliance = noncompliant, specific_issues may include multiple values (comma-separated).
- Remarks (if present) must appear only as the final slash segment and pass sanitization (no special tokens/coordinates).
- Labels may omit text when unreadable.

## Modeling Notes
- Training: parse desc into structured labels (object type + attributes) for multi-task learning: detection + attribute classification + compliance. A final free-text remarks segment may be present and is treated as unstructured text.
- Optional SFT summary variant: a single-line Chinese summary per image distilled from objects (no coordinates/special tokens; label readability only as clear/unclear). This stabilizes Stage-A behavior before RL.
- Inference: reconstruct desc per object and compute image-level QC metrics (e.g., shield presence when required, fiber bend violations, connector compliance, wiring organization). Remarks may carry human notes such as "cannot determine" or "rectified"; do not force parsing.
