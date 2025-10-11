# Contract — Logging Outputs

## Results JSONL (per-rank)
- group_index, mission, gt_label, pred_label, reward
- images: [{image_id, caption}], stage_b_raw
- k_b, k_a, used_minimal_prompt, pairwise_triggered
- (optional) candidates: [{raw, reward, pred_label}]

## Metrics JSONL (rank0)
- loss, reward_best_mean/std, acc_best, acc_any
- resp_len_mean, grad_norm_mean
- kl_b_mean, skip_updates_std0
- pairwise_trigger_rate, phase_a_entropy_mean
- reward_margin_best, flip_rate
- decision_ce_mean/ema, eta_hours
- sb_clip_low/high/region_mean, sb_entropy_mask_ratio (当启用)

## TensorBoard
- 同名 scalars；每 N 步记录；
- 文本/样例：可打印首个组的摘要与最佳原因行（隐私合规下）。
