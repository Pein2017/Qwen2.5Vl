# Phase 0 — Research

## Questions
- 最小提示 vs checklist：对 `group_margin`、解析成功率与原因文本聚焦度的影响？
- K_B=2 vs K_B=3：奖励方差、收敛速度与显存/时长性价比？
- KL 到参考策略：lambda_kl=0.02 是否足以防漂移？
- 条件式信用分配的门控阈值：熵阈=1.2 是否能减少“自信但错误”的更新？

## Protocol
- 固定硬件：8×A100‑80G；bf16；Accelerate；
- 固定模型：SFT checkpoint + 处理器；
- 固定采样：Stage‑B `K_B=2, temp=0.7, top_p=0.95`；
- 比较变量：`use_mission_checklist ∈ {false,true}`；`K_B ∈ {2,3}`；`lambda_kl ∈ {0,0.02}`；`entropy_threshold ∈ {1.2,1.5}`。

## Metrics
- 组级：accuracy、group_margin（均值/方差）、FN率；
- 稳定性：reward_std==0 占比、超长率、特殊标记泄露率；
- 文本质量：决策解析成功率、原因覆盖 checklist 关键词比例；
- 资源：每步时长、显存峰值。

## Expected Outcomes
- checklist 提示在原因聚焦与解析稳定性上更好，accuracy 不低于最小提示；
- K_B=2 足以产生稳定方差，K_B=3 性价比视数据而定；
- 小权重 KL 有助稳定，不显著损害探索；
- 熵阈=1.2 相比 1.5 更灵敏，能过滤部分“低熵且误导”的候选。
