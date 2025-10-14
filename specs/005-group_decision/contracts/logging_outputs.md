# Contract — Logging Outputs

## Results JSONL (per-rank)
- group_index, group_id, mission, gt_label
- images: [{image_id}]
- k_b, k_a, used_minimal_prompt, pairwise_triggered
- stage_b:
  - best: {pred_label, reward, raw}
  - candidates: [{raw, reward, pred_label}]  # 全量 K_B
- stage_a:
  - [{image_index, candidates: [{text, reward, margin}]}]  # 每图 K_A 候选（诊断）
- 说明：为精简体积，已移除 item-level 诊断中的 `present_pass_tokens` 与 `present_fail_tokens` 字段。

## Metrics JSONL (rank0)
- loss, reward_best_mean/std, acc_best, acc_any, accuracy, fn_rate
- resp_len_mean, grad_norm_mean
- kl_b_mean, skip_updates_std0
- pairwise_trigger_rate, phase_a_entropy_mean
- reward_margin_best, flip_rate
- decision_ce_mean/ema, eta_hours
- sb_clip_low/high/region_mean, sb_entropy_mask_ratio (当启用)

## TensorBoard
- 同名 scalars；每 N 步记录；
- 文本/样例：可打印首个组的摘要与最佳原因行（隐私合规下）。
