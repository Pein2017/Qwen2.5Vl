# Reward Logging Fix Summary

## Issues Fixed

### 1. **Main reward metrics showing 0.0000±0.0000**

**Root Cause:**
- The `_optimizer_step` was being passed `buffer_data` (a single completion's data dict)
- This dict didn't contain the full `rewards` tensor needed for logging
- The `_log_step` was looking for `generation_result["rewards"]` which was `None` or empty

**Fix:**
- Reconstructed full `generation_result_for_log` from `GenerationBuffer` before optimizer step
- Flattened `rewards_list` and `advantages_list` from all prompts in buffer:
  ```python
  all_rewards = torch.cat([r.to(self._device) for r in gen_buffer.rewards_list], dim=0)
  all_advantages = torch.cat([a.to(self._device) for a in gen_buffer.advantages_list], dim=0)
  ```
- This ensures `reward` and `reward_std` metrics show actual values in console

**Valid Indicator:**
- `[PROMPT_BATCH] reward_avg=1.3865` was already showing correct rewards
- This comes from `PromptBatchTelemetry` which tracks rewards independently
- After the fix, both `reward=` and `reward_avg=` will show the same values

### 2. **Missing detailed per-reward component metrics in console**

**Root Cause:**
- `GenerationBuffer` was not storing `rewards_per_func` and `raw_rewards_per_func`
- Console formatter has `[REWARDS_NORM]` and `[REWARDS_RAW]` lines but data wasn't available

**Fix:**
1. **Extended `GenerationBuffer` dataclass:**
   ```python
   rewards_per_func_list: List[torch.Tensor]  # P prompts, each [K, num_rewards]
   raw_rewards_per_func_list: List[torch.Tensor]  # P prompts, each [K, num_rewards]
   ```

2. **Captured per-function rewards during generation:**
   ```python
   if "rewards_per_func" in single_gen:
       rewards_per_func_all.append(single_gen["rewards_per_func"].cpu())
   if "raw_rewards_per_func" in single_gen:
       raw_rewards_per_func_all.append(single_gen["raw_rewards_per_func"].cpu())
   ```

3. **Reconstructed and added to logging:**
   ```python
   all_rewards_per_func = torch.cat([r.to(self._device) for r in gen_buffer.rewards_per_func_list], dim=0)
   all_raw_rewards_per_func = torch.cat([r.to(self._device) for r in gen_buffer.raw_rewards_per_func_list], dim=0)
   generation_result_for_log["rewards_per_func"] = all_rewards_per_func
   generation_result_for_log["raw_rewards_per_func"] = all_raw_rewards_per_func
   ```

## Expected Console Output After Fix

```
[CORE] step=1 epoch=0.100 loss=0.8227 reward=1.2244±0.3456 raw_reward=0.9876±0.2345 lr=5.000e-06 grad_norm=0.210 eta=12.6min
[ADVANTAGES] adv_std=1.1293 adv_max=1.9181
[GENERATION] temperature=1.0000 beta=0.0000 term_ratio=0.000
[DYNAMIC_CAP] enabled=true
[REWARDS_NORM] wrappers=0.045 coords=0.043 separators=0.042 vocab=0.018 length_vs_gt=0.195 bbox_giou=0.320 quad_l1=0.095 line_l1=0.091 ordering=0.092 coverage=0.098 geometry_sanity=0.095 caption_f1=0.185 grounding_acc=0.093
[REWARDS_RAW] wrappers=0.98 coords=0.95 separators=0.92 vocab=0.85 length_vs_gt=0.88 bbox_giou=0.75 quad_l1=0.12 line_l1=0.15 ordering=0.89 coverage=0.91 geometry_sanity=0.94 caption_f1=0.78 grounding_acc=0.81
[PROMPT_BATCH] fill=1.00 reward_avg=1.2244 traj=4
```

## Files Modified

1. **`src_new/rl/generation_buffer.py`:**
   - Added `rewards_per_func_list` and `raw_rewards_per_func_list` fields

2. **`src_new/rl/grpo_trainer.py`:**
   - Added per-function reward list declarations in `_generate_buffer`
   - Extracted and stored per-function rewards from `single_gen`
   - Reconstructed full `generation_result_for_log` with all rewards before optimizer step
   - Removed duplicate `prompt_batch_metrics` computation from `_optimizer_step`

## Validation

Run your training and verify:
1. `reward=X.XXXX±Y.YYYY` shows non-zero values (matches `reward_avg=`)
2. `[REWARDS_NORM]` line appears with all reward component values
3. `[REWARDS_RAW]` line appears with all raw reward component values

All per-reward metrics are also logged to TensorBoard under `rewards/{name}/mean` and `raw_rewards/{name}/mean`.
