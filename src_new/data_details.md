# BBU Installation Quality Inspection Dataset

## Purpose & Scope
- Vision-language dataset for detecting BBU-related components and assessing installation quality in telecom cabinets.
- Downstream consumer of JSONL produced by `data_conversion/unified_processor.py`. This document is the concise, English-only spec.

## Files
- Location: `data/ds_v2_full/`
  - `train.jsonl`, `val.jsonl`
  - `teacher_pool.jsonl`
  - `all_samples.jsonl`

## Record Schema (JSONL per line)
- images: List[str] — processed image paths (usually 1 path per record).
- objects: List[Object]
  - Exactly one geometry field per object:
    - bbox_2d: [x1, y1, x2, y2]
    - quad: [x1, y1, x2, y2, x3, y3, x4, y4]
    - line: [x1, y1, x2, y2, ..., xn, yn]
  - desc: Hierarchical, slash- and comma-separated string built from the templates below. The literals in data are canonical and defined by the taxonomy/mapping; this document describes them in English.
- width, height: int — processed image size.
- meta: Dict — per-sample metadata (see Meta block).

## Geometry & Object Types
- Geometry formats
  - bbox_2d: axis-aligned rectangle
  - quad: 4-point polygon
  - line: polyline (fibers/wires)
- Allowed geometry per object type
  - bbu: quad, bbox_2d
  - bbu_shield: quad, bbox_2d
  - connect_point (screws, fiber connectors): quad, bbox_2d
  - fiber: line
  - wire: line
  - label: quad, bbox_2d

## Attributes by Object Type
All values in data are canonical (defined in the taxonomy). Below are their meanings in English.

- bbu
  - brand: {Huawei, ZTE, Ericsson}
  - visibility: {complete, partial}
  - windshield_requirement: {not_required, required}
  - windshield_conformity (only if required): {shield_present_per_requirement, shield_missing_or_nonconformant}

- bbu_shield
  - brand: {Huawei, ZTE}
  - visibility: {complete, partial}
  - obstruction: {clear, blocked}
  - install_direction: {correct, incorrect}

- connect_point (screws, fiber connectors)
  - type: {bbu_mount_screw, cabinet_ground_screw, busbar_ground_screw, fiber_connector_bbu_end, fiber_connector_odf_end}
  - visibility: {complete, partial}
  - compliance: {compliant, noncompliant}
  - specific_issues (only if noncompliant; multi-value): {loose, exposed_copper, double_connection, rusty}

- fiber
  - obstruction: {clear, blocked}
  - protection: {none, protected}
  - protection_details (only if protected): {snake_tube, armored, both}
  - bend_radius: {ok, violation}

- wire
  - obstruction: {clear, blocked}
  - organization: {neat, disorganized}

- label
  - text: free text (OCR-derived); may be absent if unreadable

## Description (desc) Construction
- Separators
  - comma ",": separate attributes within a level
  - slash "/": separate hierarchical levels
  - conditional attributes appear only when parent condition is met
  - free text is inserted as-is for labels/special notes
- Object templates (English structure)
  - bbu: "BBU/brand,visibility,windshield_requirement/[windshield_conformity]"
  - bbu_shield: "Shield/brand,visibility,obstruction,install_direction"
  - connect_point: "ConnectPoint/type,visibility,compliance/[specific_issues]"
  - fiber: "Fiber/obstruction,protection,bend_radius/[protection_details]"
  - wire: "Wire/obstruction,organization"
  - label: "Label/[text]"

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
- Labels may omit text when unreadable.

## Modeling Notes
- Training: parse desc into structured labels (object type + attributes) for multi-task learning: detection + attribute classification + compliance.
- Inference: reconstruct desc per object and compute image-level QC metrics (e.g., shield presence when required, fiber bend violations, connector compliance, wiring organization).
