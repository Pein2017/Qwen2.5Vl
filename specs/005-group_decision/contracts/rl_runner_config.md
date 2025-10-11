# Contract — RLRunnerConfig (src_post)

## Required Keys
- checkpoint, processor, output_dir, {train_data_dir|eval_data_dir}
- device, seed, limit_groups, log_step
- temperature, top_p, max_new_tokens_stage_a, max_new_tokens_stage_b
- K_B, K_A, K_set, adv_clip, length_norm
- train, epochs, batch_size, learning_rate, weight_decay, max_grad_norm, drop_last
- llm_top_k_block, vision_top_k_block, freeze_patch_embed, train_aligner, aligner_lr, llm_lr, vision_lr
- use_ref_kl, ref_checkpoint, lambda_kl_stage_b, lambda_kl_stage_a
- train_stage_a_mode, stage_a_weight, max_images_tf
- group_reward_mode, use_mission_checklist
- pairwise_credit_enabled, pairwise_pairs_per_group, pairwise_delta_threshold
- use_uncertainty_gate, uncertainty_gate_min_entropy
- tb_log_dir, run_name, save_step, save_limit

## Spec‑Aligned Defaults
- use_mission_checklist=true
- Stage‑B: K_B=2, temperature=0.7, top_p=0.95
- KL: use_ref_kl=true（ref=当前SFT）
- Stage‑A 门控：use_uncertainty_gate=true, entropy_threshold=1.2
- 并行：8×A100‑80G + Accelerate

## Fail‑Fast
- 所有路径必须存在；枚举越界直接报错；奖励名需在注册表中；日志/保存路径提前创建。

## Suggested Defaults (non-binding)
- Stage‑B 裁剪：`enable_clipped_grpo: false`（若开启需设置：`epsilon_low ∈ (0,1]`, `loss_type_stage_b ∈ {grpo,bnpo,dr_grpo}`）
- Stage‑B 熵掩码：`enable_entropy_mask_stage_b: false`（若开启二选一：`entropy_top_quantile_stage_b` 或 `entropy_min_threshold_stage_b`）
- 解码屏蔽（仅影响生成，不改梯度）：Stage‑A `mask_geometry_tokens: true`, `mask_coordinate_tokens: true`; Stage‑B 两项均 false
