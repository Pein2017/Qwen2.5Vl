## 动态对比配对（训练期）实施计划（仅 src_new，基于全量数据、按 epoch 动态）

### 目标与约束
- 目标：在训练阶段为部分样本构造“对比配对”两轮对话（最多 2 个样本，即等价 `num_teacher=1`），以更强的梯度信号覆盖关键概念（正/负属性、品牌差异、几何互补等），同时不改变核心损失与分组路径。
- 约束：
  - 评估与部署一致：Eval/Inference 必须是“单轮、无上下文”，严禁依赖其他样本；即 eval 强制配对比例为 0、pair 比例为 0。
  - 训练损失路径不改动：仍然只训练“最后一个助理回合（student）”，前置回合视为上下文（context/teacher，标签为 -100）。
  - 最多 2 个样本：每个 episode 最多包含 1 个对比上下文 + 1 个目标样本；上下文在前、目标在后，复用“最后一轮 = student”的既有实现。
  - 数据来源：摈弃固定 `teacher_pool` 文件，改为“以当前训练集全量样本”为候选池，并“按 epoch 动态重建索引+采样”。

### 术语（弱化“师生”）
- target（目标样本）：本轮真正用于产生梯度的样本，其助理回合必须位于“最后一轮”。
- context（对比上下文样本）：用于形成对比语境的样本，助理回合在前，标签计入损失（权重沿用 `teacher_loss_weight`）。

### 桶划分（默认：多数票）
- 对每个样本统计其 `objects[*].desc` 的类型前缀在桶中的出现次数：`bbu, bbu_shield, connect_point, fiber, wire, label`，无法映射则忽略。
- 取出现次数最多的桶作为主类型；若并列，按对象在样本中的出现顺序稳定打破（先出现者优先）。无可映射则落入 `misc`。

### 跨桶探索（可选）
- `dynamic_pair_cross_bucket_explore_prob: 0.0~0.1`（建议 0~0.05）：以该小概率，忽略同桶限制，在全体样本中随机抽取 teacher（仍排除 self）。
- 目的：防止长期同桶导致的过拟合，提供少量多样性；默认 0.0 关闭。

### 高层策略（极简随机版）
- 遍历训练集全部样本，保证每个样本至少作为 student（target）被训练一次（单轮或双轮）。
- 对于当前样本 i：以 `teacher_ratio` 的伯努利概率决定是否配对；
  - 若不配对：单轮。
  - 若配对：从“同主类型桶”均匀随机抽取 1 个 teacher j（可重复、允许多次被选中，排除 self）；若该桶为空则退化到全量随机（排除 self）。对话顺序固定为 `teacher(j) → student(i)`。
- 不使用 loss/业务打分；不设使用上限；不做去重；不改损失路径（最后一轮为 student）。

### 配置项（最小）
- `dynamic_pairing_enabled: true`
- `teacher_ratio: 0.0~1.0`（训练期生效，eval 强制 0）
- `dynamic_pair_target_assignment: current`（固定当前样本为 target）
- 其余 `candidate_pool_size/temperature/max_teacher_uses` 等不再使用（可保留占位但无效）。

### 数据结构与 Dataset 行为
- `EpisodeSpec(target_idx=i, context_idx=j)` 保存在 `_episode_map`，键为 `target_idx`，保证 `__getitem__(i)` 可取到其 teacher。
- `set_epoch(epoch)`：
  - 固定随机种子 `seed = base_seed + epoch`；
  - 构建按主类型的桶索引；
  - 逐样本伯努利决定是否抽 teacher 并填充 `_episode_map`；
  - eval 分支清空 `_episode_map` 强制单轮。
- `__getitem__`：
  - 若 `_episode_map[i]` 存在，则注入 `teacher_samples=[samples[j]]` 并走双轮；否则单轮。
- 移除任何 teacher_pool 回退与 loss‑based 权重路径。

### Eval/Inference（不变）
- Eval/Inference 均为单轮：`teacher_ratio=0.0`，不加载 teacher pool。

### 指标（简化）
- `pair_episodes`, `single_episodes`, `target_is_current_pct(=100%)`, `coverage_context_unique_pct`, `context_usage_histogram`。
- `avg_contrast_score` 固定为 0.0（占位）。

### 失败回退
- 同类型桶为空：退化到全量随机（排除 self）。
- 样本数极少（=1）：跳过配对，单轮。

### 变更点
- 代码：
  - `src_new/pairing/contrastive_pairing.py`：改为桶内均匀随机（可重复），固定 `current` 为 target；移除权重/上限/打分。
  - `src_new/data/dataset.py`：删除 `set_pairing_weights` 与对 `teacher_pool_manager` 的回退；仅使用 `_episode_map` 注入 teacher。
- 文档：删除 loss‑based 与业务打分描述，保留“桶内随机”方案。

