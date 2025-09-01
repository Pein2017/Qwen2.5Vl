## Progressive Unfreeze Training Plan for Qwen2.5‑VL (src_new)

### Purpose
- Stabilize and speed up convergence by training in stages with controlled parameter updates.
- Precisely “shape” coordinate prediction while protecting backbone language/vision capacity early on.
- Enforce row‑wise control over token embedding and LM head updates limited to the coordinate token slice.

### Audience
- Engineers extending `src_new` training. Familiar with Qwen2.5‑VL integration and this repo’s conventions.


## Background and Current State
- Entry points
  - Training launcher: `/data3/Qwen2.5-VL-main/scripts/run_new_train.sh` (prod) and `/data3/Qwen2.5-VL-main/scripts/run_debug.sh` (debug)
  - Training script: `/data3/Qwen2.5-VL-main/scripts/train_new.py`
- Core modules (src_new)
  - Model wrapper: `/data3/Qwen2.5-VL-main/src_new/models/wrapper.py`
  - Loss manager: `/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py`
  - Coordinate losses: `/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py`
  - Config schema/validation: `/data3/Qwen2.5-VL-main/src_new/config/config.py`
  - Callbacks: `/data3/Qwen2.5-VL-main/src_new/training/callbacks.py`
- Current loss system (debug config)
  - LLM CE: teacher/student spans with next‑token alignment
  - Coordinate auxiliary losses (enabled): Kernelized‑KL (sparse window) + Unlikelihood(top‑K); Laplacian reg on coord embedding slice
  - YAML: `coord_aux_enabled: true` + knobs (tau, sigma_bins, window_bins, topk, lambdas)

Why progressive unfreeze
- Stage training reduces gradient shock and preserves pre‑trained capacity until adapters and coordinate subspace stabilize.
- Targeted unfreeze (top layers) yields most benefit with least disruption, before full joint training.


## Design Overview
### Stages (single launch)
- Stage 0 (Bootstrap: epochs [0, E0))
  - Train only:
    - `visual.merger` (vision→LLM aligner MLP)
    - Coordinate rows of `embed_tokens.weight` and `lm_head.weight`
  - Freeze:
    - All Qwen2.5‑VL decoder layers
    - Non‑coordinate rows in embeddings/LM head via gradient masks
- Stage 1 (Adapter unfreeze: epochs [E0, E1))
  - Keep Stage 0 items trainable
  - Unfreeze top K decoder layers (last K layers): self‑attn, MLP, norms
  - Keep non‑coord grad masks active
- Stage 2 (Joint: epochs [E1, end))
  - Remove masks; unfreeze all remaining modules
  - Optionally lower LR for stability; keep Laplacian reg on coord embeddings

Rationale
- Stage 0 adapts the “interfaces” (merger + small output/input subspace) to the coordinate task
- Stage 1 expands capacity locally near output to learn coordinate‑context coupling
- Stage 2 completes joint training once gradients are well behaved


## Codebase Integration Map (src_new)
- `models/wrapper.py`
  - Qwen2.5‑VL model wrapping, tokenizer/processor setup
  - Creates `LossManager`; now reads aux loss knobs from YAML
  - Will remain unchanged for unfreeze; callback will control requires_grad
- `training/callbacks.py`
  - Place to implement `ProgressiveUnfreezeCallback` (skeleton exists)
  - Responsibilities:
    - Compute coordinate token row range (via `model.coordinate_processor.coordinate_token_range`)
    - Apply grad hooks for row‑wise masking on embeddings/head
    - Freeze/unfreeze modules at stage boundaries
    - Rebuild optimizer/scheduler after transitions
- `scripts/train_new.py`
  - Register the callback (when enabled by YAML) after trainer creation
  - Optional: pass LR overrides to trainer’s optimizer grouping
- `config/config.py`
  - Add/validate progressive unfreeze keys (see below)


## Configuration (YAML; explicit, no getattr fallbacks)
Add to config YAML:
```yaml
# Progressive unfreeze
prog_unfreeze_enabled: true
prog_unfreeze_epoch_stage0_end: 1     # E0 (end of Stage 0)
prog_unfreeze_epoch_stage1_end: 3     # E1 (end of Stage 1)
prog_unfreeze_top_k_layers: 4         # K: last K decoder layers in Stage 1
prog_unfreeze_coord_slice_only: true  # apply row-wise coord slice masking

# LR overrides per stage/group (optional but recommended)
lr_merger: 1.0e-5
lr_coord_slice: 3.0e-5
lr_top_layers: 6.0e-6
lr_full_model: 5.0e-6
```
Validation to enforce in `Config.__post_init__`:
- `prog_unfreeze_enabled` is bool
- `prog_unfreeze_epoch_stage0_end >= 1`
- `prog_unfreeze_epoch_stage1_end > prog_unfreeze_epoch_stage0_end`
- `prog_unfreeze_top_k_layers >= 1`
- LRs positive


## Implementation Plan
### 1) Complete ProgressiveUnfreezeCallback (src_new/training/callbacks.py)
- Constructor
  - Accept: `freeze_vision_llm_epochs` (E0), `full_unfreeze_epoch` (E1), `coord_slice_only` (bool), `top_k_layers` (K), `lr_overrides` (dict)
- Helpers (some already stubbed)
  - `_get_base_model(trainer)`: return HF model under wrapper
  - `_get_coord_token_range(trainer)`: use `model.coordinate_processor.coordinate_token_range`; validate
  - `_find_embedding_and_lmhead(trainer)`: return embedding and lm_head parameters
  - `_apply_coord_slice_grad_masks(embed_param, lm_head_param, coord_start, coord_end_exclusive)`
    - Register hooks that zero gradients outside the row window:
      ```python
      def _mask_embed_grad(g): return g * row_mask.unsqueeze(1).to(g.dtype)
      def _mask_lm_head_grad(g):
          if g.dim() != 2: return g
          if lm_head_param.shape[0] == vocab_rows:  # [vocab, hidden]
              return g * row_mask.unsqueeze(1).to(g.dtype)
          elif lm_head_param.shape[1] == vocab_rows:  # [hidden, vocab]
              return g * row_mask.to(g.dtype)
          return g
      ```
  - `_clear_masks()`
  - `_rebuild_optimizer_and_scheduler(trainer)`: call `trainer.create_optimizer()` then `trainer.create_scheduler(remaining_steps)`
    - Compute `remaining_steps` from dataloader length and args
- Stage transitions
  - `_stage1_freeze_and_mask(trainer)` (called in `on_train_begin`)
    - Freeze all params: `p.requires_grad = False`
    - Unfreeze `visual.merger` params by name contains `"visual.merger"`
    - Unfreeze embeddings + lm_head params
    - If `coord_slice_only`: apply row masks using coord token range
    - `self._stage = 1`; rebuild optimizer/scheduler
  - `on_epoch_begin`
    - At epoch == E0: unfreeze top K decoder layers (last K):
      - For each selected layer: unfreeze `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`
      - Keep masks active on embeddings/head
      - Rebuild optimizer/scheduler
    - At epoch == E1: remove masks, unfreeze all remaining params, rebuild optimizer/scheduler, `self._stage = 2`
  - `on_train_end`: `_clear_masks()`

### 2) Register callback (scripts/train_new.py)
- After trainer creation, if `config.prog_unfreeze_enabled`:
  - Construct callback using YAML values (E0, E1, K, coord_slice_only, lr_overrides)
  - Register with trainer: `trainer.add_callback(ProgressiveUnfreezeCallback(...))`

### 3) Optimizer parameter groups (BBUTrainer)
- In `create_optimizer()` (inside the trainer), create param groups reflecting current stage:
  - Group 1: visual.merger → `lr_merger`
  - Group 2: coord slice of embeddings/head → `lr_coord_slice`
  - Group 3: top K layers (when active) → `lr_top_layers`
  - Group 4: remaining trainable params → `lr_full_model`
- Implementation pattern: filter parameters by `requires_grad is True` and name patterns; for coord slice, either rely on row masks or optionally scale per‑row grads in hook (keep masks simpler for now).


## Testing (TDD)
- Unit tests (fast, CPU)
  - Row mask hooks zero gradients on non‑coord rows (embeddings/head)
  - After Stage 0 init, only merger + coord rows have `requires_grad=True` or non‑zero grads
  - After Stage 1 transition, selected top K layers gain grads; masks persist
  - After Stage 2, masks removed; all params require grad
- Integration smoke
  - Tiny dataset; config with `prog_unfreeze_enabled: true`, `E0=1`, `E1=2`, `K=2`
  - Verify no NaN/Inf; logs show stage transitions


## Monitoring & Logging
- At stage change, log:
  - Stage number, epoch, number of trainable params per group
  - Optimizer groups with LR assignments
- Continue tracking loss components (teacher_llm, teacher_l1, student_*) via existing metrics


## Risk Management
- Potential risks
  - Over‑freezing yields underfitting: adjust E0/E1 shorter
  - Instability upon full unfreeze: reduce `lr_full_model`, keep Laplacian reg
  - Mask shape mismatch: validate coord range early; assert vocab rows match model dims
- Mitigations
  - Strict validations and fail‑fast with actionable messages
  - Keep small LR for newly unfrozen groups; rebuild optimizer on transitions


## Example YAML (debug)
```yaml
prog_unfreeze_enabled: true
prog_unfreeze_epoch_stage0_end: 1
prog_unfreeze_epoch_stage1_end: 3
prog_unfreeze_top_k_layers: 4
prog_unfreeze_coord_slice_only: true
lr_merger: 1.0e-5
lr_coord_slice: 3.0e-5
lr_top_layers: 6.0e-6
lr_full_model: 5.0e-6
```


## Implementation Checklist
- [ ] config/config.py: add fields + validation
- [ ] callbacks.py: implement `ProgressiveUnfreezeCallback` methods and helpers
- [ ] train_new.py: register callback when enabled
- [ ] BBUTrainer: parameter groups + optimizer rebuild support
- [ ] tests: unit (hooks, transitions) + integration smoke
- [ ] docs: update `src_new/README.md` to reference this plan


## Navigation Reference (src_new)
- `/data3/Qwen2.5-VL-main/src_new/models/wrapper.py` — wrapper and LossManager wiring
- `/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py` — CE, aux coord losses
- `/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py` — KCE, Unlikelihood, Laplacian utils
- `/data3/Qwen2.5-VL-main/src_new/training/callbacks.py` — place for `ProgressiveUnfreezeCallback`
- `/data3/Qwen2.5-VL-main/src_new/config/config.py` — YAML schema and validation
- `/data3/Qwen2.5-VL-main/scripts/train_new.py` — trainer creation, callback registration
- `/data3/Qwen2.5-VL-main/configs/bbu_v2/debug.yaml` — debug config (can enable progressive unfreeze here)


## Notes
- Keep all toggles and knobs in YAML only (no getattr fallbacks).
- Preserve TDD: write small failing tests for each hook/transition before coding logic.
- Keep absolute paths in scripts/commands to align with repo rules.
