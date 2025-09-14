# Design Document

## Overview
Diagnose and fix Stage‑A drift and zeroed metrics in `src_post/` GRPO runner. Symptoms:
- Stage‑A summaries are off‑domain (nonsense/unrelated), while Stage‑B replies are somewhat reasonable.
- Metrics like acc_best/any_hit near 0, resp_len large/variable; rewards clustered (std≈0), causing near‑zero losses.
- `simple_vl_infer.py` produces excellent single‑image outputs on the same checkpoint.

Hypothesis: Post‑inference pipeline deviates from SFT inference contracts: image encoding/typed chat, stopping, logits masking, or processor mismatch. Also potential TF path leakage (wrapper vs base), or extreme decode constraints.

## Steering Document Alignment

### Technical Standards
- HF‑first encoding via official `Qwen2VLProcessor` (typed messages) with EXIF‑aware resize.
- Fail‑fast invariants on image tokens vs `image_grid_thw`; single‑image Stage‑A.
- No hard‑coded IDs; avoid masking normal Chinese tokens.

### Project Structure
- Use existing subpackages: `prompting/`, `generation/`, `tf/`, `rewards/`, `runner.py`. Keep minimal changes and strong validation logs.

## Code Reuse Analysis

### Existing Components to Leverage
- `src_post/prompting/conversation.py`: Stage‑A/B builders aligned to SFT summary prompts.
- `src_post/generation/generation.py`: EXIF+smart resize, typed chat, diagnostics for `<|image_pad|>` and `image_grid_thw`.
- `src_post/tf/teacher_forcing.py`: routes forwards to base model.

### Integration Points
- `runner.py` calls Stage‑A context builder and Stage‑B TF/rewards.
- `GeometryCoordMaskLogitsProcessor` used during decoding only.

## Architecture

- Add strict runtime checks to match `simple_vl_infer.py`:
  1) Use the same `processor` and ensure `chat_template` is preserved.
  2) Verify `<|image_pad|>` count > 0 and `image_grid_thw.size(0) == 1` per Stage‑A call; raise with actionable message if not.
  3) Match generation config to `simple_vl_infer.py` (reduce penalties; smaller `no_repeat_ngram_size`).
  4) Ensure logits masking only bans geometry/coord tokens (no overlap with common CJK).
  5) Confirm `sanitize_stage_a` default off for initial diagnosis.
  6) Add per‑image debug JSON lines logging of decoded prompt and 1‑best Stage‑A output.
  7) Ensure `processor_path == checkpoint_path` tokenizer/processor set; do not mix different processors.

- Stage‑B TF:
  - Continue using base model for TF; cache probabilities once per prompt.

## Components and Interfaces

### DiagnosticsHooks
- Purpose: Centralized logging/guards for Stage‑A.
- Interfaces: `assert_image_alignment(enc, text) -> None`; `log_stage_a_prompt(enc, decoded_text)`
- Dependencies: `Qwen2VLProcessor`, `DetectionModel`

### Runner Tweaks
- Purpose: Wire diagnostics; adjust gen configs; dump per‑item Stage‑A outputs into results JSONL (new field `stage_a_lines`).
- Interfaces: internal to `runner.py`.

## Data Models
- Extend per‑item JSON record with `stage_a_lines: List[str]` and `stage_a_diag: {image_pad: int, image_grids: int}` for debugging.

## Error Handling
- If `<|image_pad|>` count == 0, raise ValueError with hint: “Check processor path; ensure typed image list length==1; verify chat_template exists; avoid using text‑only Stage‑A.”
- If `image_grid_thw.size(0) != 1`, raise ValueError with actual vs expected.

## Testing Strategy
- Unit: verify `GeometryCoordMaskLogitsProcessor` ban set excludes common CJK tokens.
- Integration: A/B compare `simple_vl_infer.py` vs `generation.build_stage_a_context_lines` on the same image—expect similar outputs.
- E2E: Run runner with `limit_groups=2`, log Stage‑A diags; assert `image_pad>0 && grids==1` and reasonable Stage‑A lines.
