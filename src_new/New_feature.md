### Lossless Coordinate-Logit Head — Design & Integration Plan

#### Goals and Constraints
- **Preserve language capacity**: Do not degrade text generation quality.
- **Lossless at init**: Initial behavior identical to baseline; head outputs zero so logits are unchanged.
- **Improve grounding**: Sharpen coordinate token predictions only (vocab slice `[coord_start:coord_end)`).
- **Generation-compatible**: Must work under `model.generate()` without caller changes.
- **Config- and test-driven**: All knobs in YAML; add unit/integration tests before code.

#### Do we need a router/gate module?
- **Short answer**: No, not initially. If the model already separates text vs coordinate modes well, a global residual head applied to the coord slice is sufficient.
- **Recommended**: Keep a single learnable scalar gate (initialized to 0.0) on the head’s output. This lets the model “opt in” to the head smoothly.
- **Optional later**: Token-wise gates or heuristics only if you observe interference (text degradation) in ablations.

#### High-Level Design
- **Residual coord head**: A small MLP maps last hidden state `[B,T,H]` → delta logits `[B,T,K]`, where `K = coord_end - coord_start`.
  - Example: `Linear(H → M) + GELU + Linear(M → K)`; `M` is `coord_head_hidden_dim`.
  - Apply a scalar gate `g ∈ [-1,1]` via `tanh(g)`; init all deltas to zero (zero-init last linear + gate=0.0).
- **Residual addition**: Add delta only to logits slice `[..., coord_start:coord_end]` of the base `lm_head` logits.
- **Zero-init guarantees lossless start**: Language logits and non-coordinate vocab remain untouched; coord slice is identical until training moves the gate/weights.

#### Configuration Additions (YAML → Config)
- `coord_head_enabled: bool`
- `coord_head_hidden_dim: int` (e.g., 512/768/1024)
- `coord_head_lr: float` (e.g., 1e-5)
- `coord_head_gate_init: float` (default 0.0; no in-code default per policy)
- Optional: `coord_head_dropout: float` (0.0–0.1), `coord_head_norm: bool` (LayerNorm before MLP)

#### Training Integration
- **Optimizer groups** (in `/data3/Qwen2.5-VL-main/src_new/training/bbu_trainer.py`):
  - Add group `coord_head` with `params=detection_model.coord_head.parameters()` and `lr=coord_head_lr`.
  - Keep existing groups: `vision`, `merger`, `coord_slice` (embed/lm_head coord range), `llm`.
- **Progressive unfreeze** (callback):
  - Freeze `vision + llm` for first 1–2 epochs (`freeze_vision_llm_epochs: 2`, `coord_slice_only: true`).
  - Train `coord_head + coord_slice + merger` early, then unfreeze all.
- **Loss wiring**: No change to `LossManager`. CE is over full vocab; coordinate losses (L1 or aux KCE/Unlikelihood) apply on the updated coord logits slice automatically.
- **Aux coord losses**: Keep `coord_aux_enabled: true` (they remain complementary to the head). Tune `tau`, `sigma_bins`, `window_bins`, `topk`, lambdas as per current best.

#### Inference / Generation Compatibility
- Implement inside `/data3/Qwen2.5-VL-main/src_new/models/wrapper.py` forward:
  - Call base model with `output_hidden_states=True` and preserve `use_cache`.
  - Compute `delta = coord_head(hidden_states[-1])` and add to the coord slice of logits.
  - Return logits as usual; `generate()` will automatically use the enhanced logits.
- Overhead: One small MLP on `[B,T,H]` per forward; negligible vs LLM.

#### Monitoring & Validation (add to TrainingStateManager and debug logs)
- **Head activity**: Track `||delta||` mean/std, gate value `tanh(g)` over steps.
- **Coord vs text CE**: Report CE on coord positions vs text positions (using existing span/coord masks).
- **Pixel-space metrics** (if available): L1(px) per axis and IoU for bbox/quad; average path error for lines.
- **No-regression checks**: Monitor perplexity on non-coordinate tokens; should remain stable.

#### Rollout Plan (Ablations)
1) Baseline (no head) + current best coord aux settings.
2) Head enabled (scalar gate, hidden=512), aux disabled → measure deltas.
3) Head + aux enabled → tune `coord_head_lr`, `coordinate_loss_weight` ramp, aux `tau/window/topk`.
4) Sensitivity: hidden dim 512 vs 1024; with/without dropout; 1 vs 2 warmup epochs of freeze.
5) Optional: token-wise gating (only if text degradation observed).

#### Risks & Mitigations
- **Text degradation**: Keep head scoped to coord slice; zero-init and small LR; progressive unfreeze; monitor text CE.
- **Overfitting coord slice**: Add mild weight decay; optionally dropout in head; early stop on val metrics.
- **Training instability**: Start with gate≈0 and low LR; increase gradually if head remains under-utilized.

#### Implementation Checklist
- Config (`/data3/Qwen2.5-VL-main/src_new/config/config.py`): add required fields, validation, no defaults.
- Wrapper (`/data3/Qwen2.5-VL-main/src_new/models/wrapper.py`):
  - Build `coord_head` when `coord_head_enabled`.
  - Zero-init last linear and set gate from config.
  - In forward, add residual on coord slice using `hidden_states[-1]`.
- Trainer (`/data3/Qwen2.5-VL-main/src_new/training/bbu_trainer.py`): add `coord_head` param group with `coord_head_lr`.
- Tests (`/data3/Qwen2.5-VL-main/src_new/tests/`):
  - Unit: shapes, zero-init is lossless (logits unchanged at step 0), residual only affects coord slice.
  - Integration: loss decreases on synthetic batch with coord labels; text CE unchanged.
  - Generation: `model.generate()` path runs and returns tokens; coord slice modified when gate≠0.
- Docs: Update `/data3/Qwen2.5-VL-main/src_new/UNIFIED_DOCUMENTATION.md` (Training Architecture & Loss System) to mention the optional coord head.

#### Suggested Initial Hyperparameters
- `coord_head_enabled: true`
- `coord_head_hidden_dim: 512`
- `coord_head_lr: 1e-5`
- `coord_head_gate_init: 0.0`
- Progressive unfreeze epochs: `2`
- Keep current `coord_aux_*` as per your best val; if coords still weak, ramp `coordinate_loss_weight` from `0.5 → 1.5` over first 1–2k steps.

#### Example YAML Snippet (add to your training config)
```yaml
coord_head_enabled: true
coord_head_hidden_dim: 512
coord_head_lr: 1.0e-5
coord_head_gate_init: 0.0
# Existing knobs (keep/tune as needed)
coord_aux_enabled: true
coord_aux_tau: 1.2
coord_aux_sigma_bins: 8
coord_aux_window_bins: 32
coord_aux_topk: 100
coord_aux_lambda_kce: 1.0
coord_aux_lambda_unlike: 0.05
```

#### Notes on Router/Gate Variants (for future)
- **Scalar gate (recommended)**: single parameter; lowest risk; already included.
- **Token-wise gate**: small sigmoid head over `hidden_states[-1]` to attenuate deltas where text is expected. Only consider if text metrics regress.
- **Training-time oracle gate**: multiply deltas by ground-truth `coord_mask` during training for efficiency; still keep scalar gate. In inference, fall back to scalar gate only.
