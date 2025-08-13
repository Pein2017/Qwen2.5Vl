# Coordinate Distance Training Roadmap (CE kept, No Soft-Expectation L1)

Scope: Keep Qwen2.5‑VL architecture unchanged. Maintain CE on coordinate tokens. Remove the soft‑expectation L1 path entirely. Add three components only:
- Kernelized‑KL on the coordinate slice
- Unlikelihood on non‑coordinate tokens at coordinate positions
- Laplacian regularizer on coordinate embeddings

All features plug into the existing span/mask pipeline and are numerically stable by construction.


## What stays vs. what changes

- Keep: Standard CE on all assistant tokens (including coordinate tokens) with the existing single‑pass CE in `LossManager`.
- Remove: Soft‑Expectation L1 path and any dependency on `expected_coord`.
- Add: Kernelized‑KL (sparse window), Unlikelihood (top‑K) on non‑coordinate tokens at coordinate positions, and Laplacian regularizer on the coordinate embedding slice.


## Integration points in this repo

- `/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py`
  - Reuse: `_compute_granular_teacher_student_loss(...)` and its next‑token shifting and span masks.
  - Add: new coordinate losses computed at positions where the LABEL is a coordinate token (you already derive `teacher_coord_mask_shifted` and `student_coord_mask_shifted`).
  - Compose: Keep CE unchanged; add weighted sums of: Kernelized‑KL and Unlikelihood to the existing coordinate loss slots. We will continue to store the final weighted coordinate loss in the existing fields `teacher_l1_loss` and `student_l1_loss` to preserve external interfaces.
  - Add: a small hook to compute Laplacian regularizer from an embedding accessor.

- `/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py`
  - Add stateless utilities: sparse kernel window builder, Kernelized‑KL, Unlikelihood (top‑K), Laplacian regularizer.
  - Do not modify the model or tokenizer; only math utilities.

- `/data3/Qwen2.5-VL-main/src_new/models/wrapper.py`
  - Provide an embedding accessor to `LossManager` that returns the coordinate embedding slice at runtime, using the already computed coordinate token range.


## Precise wiring in LossManager

Where: `LossManager._compute_granular_teacher_student_loss(...)` (already does next‑token shift, span masks, and makes `teacher_coord_mask_shifted`/`student_coord_mask_shifted`).

Steps:
1) Keep CE path unchanged (no masking of coordinates). You already compute per‑token CE once and average over masks.
2) Build a combined coordinate mask for shifted positions if you want a single computation, or compute per‑group separately:
   - `coord_mask_teacher = teacher_coord_mask_shifted`
   - `coord_mask_student = student_coord_mask_shifted`
3) Extract coordinate slice logits for the shifted logits: `coord_logits_full = shifted_logits[..., coord_start_id:coord_end_id]`.
4) For each mask (teacher then student):
   - Gather valid positions `(b,t)` where mask is True.
   - Derive targets y in bin units: `y = (shifted_labels[mask] - coord_start_id).clamp_(0, K)`.
   - Compute Kernelized‑KL on those positions (sparse window).
   - Compute Unlikelihood on those positions against non‑coordinate tokens using full‑vocab logits.
   - Average each loss over the count of coordinate positions in that group; add weighted sum to that group’s coordinate loss bucket.
5) If an embedding accessor is set, compute Laplacian regularizer on the coordinate embedding slice once per forward and add to total loss with a tiny weight.

Return the same `dict` keys as today but with `teacher_l1_loss`/`student_l1_loss` now representing the aggregate coordinate auxiliary loss (Kernelized‑KL + Unlikelihood), leaving CE untouched.


## New utilities (coordinate_loss.py)

All functions operate in float32 and clamp logits to avoid overflow/underflow. Return zeros when masks are empty. Use `torch.nan_to_num` as final guards.

- Sparse kernel target around GT bin
```python
def build_kernel_indices_and_q(y: torch.Tensor, K: int, sigma: float, window: int):
    # y: [N] integer ground-truth in [0, K]
    # Returns idxs: [N, W], q_vals: [N, W]
    # Clip per-sample window to [0, K], normalize q per row (Gaussian or Laplace kernel)
    # Implement using simple loops or vectorized ranges; ensure fixed width via padding and masks
    return idxs, q_vals
```

- Kernelized‑KL over sparse window
```python
def kernelized_kl_sparse(coord_logits: torch.Tensor,  # [N, K+1]
                         idxs: torch.Tensor,          # [N, W]
                         q_vals: torch.Tensor,        # [N, W]
                         tau: float,
                         eps: float = 1e-6) -> torch.Tensor:
    logits = (coord_logits.float() / tau).clamp(-50.0, 50.0)
    p_full = torch.softmax(logits, dim=-1)
    p_w = p_full.gather(-1, idxs)
    q_w = q_vals / (q_vals.sum(-1, keepdim=True) + eps)
    # KL(q||p) on the window
    kl = (q_w * (torch.log(q_w + eps) - torch.log(p_w + eps))).sum(-1)
    return torch.nan_to_num(kl.mean(), nan=0.0, posinf=1e6, neginf=1e6)
```

- Unlikelihood on non‑coordinate tokens at coordinate positions (top‑K)
```python
def unlikelihood_topk_text(logits_all: torch.Tensor,    # [B, T, V]
                           coord_mask: torch.Tensor,    # [B, T]
                           noncoord_vocab_mask: torch.BoolTensor,  # [V]
                           topk: int = 100,
                           eps: float = 1e-6) -> torch.Tensor:
    if coord_mask is None or coord_mask.sum().item() == 0:
        return logits_all.new_tensor(0.0)
    # Select non-coordinate sub-vocab
    logits_text = logits_all[..., noncoord_vocab_mask].float().clamp(-50.0, 50.0)
    probs_text = torch.softmax(logits_text, dim=-1)
    k = min(topk, probs_text.size(-1))
    top_vals, _ = torch.topk(probs_text, k=k, dim=-1)
    loss = -torch.log(1.0 - top_vals + eps)  # [B, T, k]
    denom = coord_mask.sum() * k + eps
    out = (loss * coord_mask.unsqueeze(-1).float()).sum() / denom
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)
```

- Laplacian regularizer on coordinate embedding slice
```python
def laplacian_regularizer(emb_slice: torch.Tensor, order: int = 1) -> torch.Tensor:
    # emb_slice: [K+1, d]
    if emb_slice is None or emb_slice.numel() == 0:
        return emb_slice.new_tensor(0.0)
    if order == 1:
        diff = emb_slice[1:] - emb_slice[:-1]
    else:
        diff = emb_slice[2:] - 2 * emb_slice[1:-1] + emb_slice[:-2]
    out = (diff.pow(2).sum(dim=-1)).mean()
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)
```


## LossManager changes (no CE masking, CE remains the primary signal)

Add fields (programmatic toggles; no YAML required initially):
- `self.coord_start_id, self.coord_end_id` (already available via `coordinate_loss_fn`)
- `self.coord_ids_vocab: BoolTensor[V]` and `self.noncoord_ids_vocab = ~coord_ids_vocab`
- `self.kernel_tau, self.kernel_sigma_bins, self.kernel_window_bins`
- `self.unlike_topk`
- `self.lambda_kce, self.lambda_unlike, self.lambda_lap1, self.lambda_lap2`
- `self._embedding_accessor: Optional[Callable[[], torch.Tensor]]`

During init, build vocab masks once using `token_processor.get_coordinate_token_range(tokenizer)` and the tokenizer’s `vocab_size`.

In `_compute_granular_teacher_student_loss(...)` after computing `per_token_loss` and masks:

```python
# 1) Coordinate positions (shifted)
coord_mask_teacher = teacher_mask_shifted & coord_label_mask[:, 1:]
coord_mask_student = student_mask_shifted & coord_label_mask[:, 1:]

# 2) Extract coordinate logits and targets at shifted positions
coord_logits_full = shifted_logits[..., coord_start:coord_end]  # [B, T-1, K+1]

# Helper to compute group losses (teacher or student)
def compute_coord_group_losses(group_mask):
    if not group_mask.any():
        return shifted_logits.new_tensor(0.0), shifted_logits.new_tensor(0.0)
    pos = group_mask.nonzero(as_tuple=False)  # [N, 2] (b, t)
    # coord logits slice at positions
    coord_logits = coord_logits_full[pos[:,0], pos[:,1], :]  # [N, K+1]
    # ground-truth y in bins
    y_ids = shifted_labels[pos[:,0], pos[:,1]]
    y = (y_ids - coord_start).clamp(0, K)
    # Kernelized-KL
    idxs, q_vals = build_kernel_indices_and_q(y, K=K, sigma=self.kernel_sigma_bins, window=self.kernel_window_bins)
    kce = kernelized_kl_sparse(coord_logits, idxs, q_vals, tau=self.kernel_tau)
    # Unlikelihood on non-coordinate tokens (full vocab) at those positions
    # Note: we apply coord mask to full logits via the same positions
    # Build a binary mask [B, T-1] -> then reuse group_mask for broadcasting inside unlikelihood
    unlike = unlikelihood_topk_text(shifted_logits, group_mask, self.noncoord_ids_vocab, topk=self.unlike_topk)
    return kce, unlike

kce_t, ul_t = compute_coord_group_losses(coord_mask_teacher)
kce_s, ul_s = compute_coord_group_losses(coord_mask_student)

teacher_coord_loss = self.lambda_kce * kce_t + self.lambda_unlike * ul_t
student_coord_loss = self.lambda_kce * kce_s + self.lambda_unlike * ul_s
```

- Assign these to the existing fields when building the return dict:
  - `teacher_l1_loss = teacher_coord_loss if teacher_coord_loss > 0 else None`
  - `student_l1_loss = student_coord_loss if student_coord_loss > 0 else None`
- Weighting outside remains the same (teacher/student × coordinate_loss_weight), preserving Trainer behavior.

Laplacian regularizer (once per batch):
```python
if self._embedding_accessor is not None:
    emb_slice = self._embedding_accessor()  # [K+1, d]
    lap1 = laplacian_regularizer(emb_slice, order=1)
    lap2 = laplacian_regularizer(emb_slice, order=2) if self.lambda_lap2 > 0 else shifted_logits.new_tensor(0.0)
    # Add to total_loss directly after component weighting
```

Note: The CE path is unchanged; we do not mask out coordinate targets from CE.


## Wrapper hook (embedding accessor)

In `DetectionModel` (wrapper): after initializing/ensuring `LossManager`, pass a callable:

```python
# inside wrapper after tokenizer is set and coordinate_token_range is known
start_id, end_id = self.coordinate_processor.coordinate_token_range
get_coord_emb = lambda: self.get_input_embeddings().weight[start_id:end_id]
self.loss_manager.set_embedding_accessor(get_coord_emb)
```

If tokens aren’t extended yet (`(0,0)`), the accessor should return None or a zero‑tensor; `LossManager` must guard against it.


## Numerical stability and safety rules

- Always compute custom losses in float32.
- Clamp logits before softmax/log operations: `clamp(-50, 50)`.
- Add `eps=1e-6` for every log or division; `torch.nan_to_num` on final scalars.
- Normalize over the number of coordinate positions (and over top‑K for Unlikelihood) to keep gradients scale‑invariant across batches.
- Return exact zeros when the relevant mask is empty (keep graph with `.new_tensor(0.0)`).


## Default knobs (programmatic, no YAML required yet)

- `kernel_tau = 1.2`, `kernel_sigma_bins = 8`, `kernel_window_bins = 32`
- `unlike_topk = 100`
- Weights: `lambda_kce = 0.3–0.7`, `lambda_unlike = 0.02–0.05`, `lambda_lap1 = 1e-4`, `lambda_lap2 = 1e-5`
- Keep your existing CE weights as is. Adjust only if you see gradient spikes.


## Tests to add (minimal)

- Unit (losses):
  - Kernelized‑KL decreases when coord logits peak near y; finite and stable when no coords.
  - Unlikelihood finite for empty and non‑empty masks; decreases when non‑coord mass at coord positions is suppressed.
  - Laplacian returns 0 for perfectly linear embeddings; small positive for slight curvature; finite always.
- Integration:
  - End‑to‑end training with features disabled remains identical.
  - With features enabled (tiny weights), training runs without Inf/NaN and logs finite losses.


## Rollout plan

1) Implement utilities in `coordinate_loss.py` (sparse kernel, KL, unlikelihood, Laplacian). Guard all numerics.
2) Extend `LossManager` to compute new coordinate losses at shifted coordinate positions; keep CE unchanged; reuse `teacher_l1_loss`/`student_l1_loss` fields to carry the new aggregate.
3) Add embedding accessor from `wrapper` and wire Laplacian; default tiny weights.
4) Add unit tests and a short smoke run; monitor loss scalars and ensure no NaN/Inf.


## Notes

- CE remains the primary driver on coordinates; Kernelized‑KL shapes the distribution around the correct bin; Unlikelihood suppresses text tokens at coordinate slots; Laplacian maintains a smooth coordinate embedding manifold. No soft‑expectation is used anywhere in this plan.
