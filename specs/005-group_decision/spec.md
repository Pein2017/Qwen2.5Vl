# Feature Specification: AI质检（BBU）多图组级决策 — Stage‑A 摘要 + Stage‑B 汇总（GRPO 信用分配）

**Feature Branch**: `005-group_decision`  
**Created**: 2025-10-11  
**Status**: Draft  
**Constitution Version**: 4.0.0  
**Input**: User description: "我现在需要创立一个spec，专门来负责的AI质检项目来进行多张图片的审核。"

## Clarifications

### Session 2025-10-11

- Q: 初始训练日程与权重如何设定为默认基线？ → A: 先训练 Stage‑B，再协同训练
- Q: 默认的组级奖励配置选择哪种？ → A: 组合（任务感知）：在基础组合上加入 coverage/taxonomy/consistency
- Q: Stage‑B 提示默认是否包含 mission checklist？ → A: 始终包含 checklist
- Q: 训练资源与规模基线？ → A: 8卡A100‑80G + Accelerate
- Q: Stage‑B 采样与温度默认？ → A: K_B=2, temperature=0.7, top_p=0.95
- Q: KL 参考策略来源？ → A: 使用当前 SFT checkpoint 作为参考
- Q: 条件式 Stage‑A 的不确定性门控默认？ → A: 开启门控，熵阈=1.2
- Q: 结果衡量的主KPI？ → A: 准确率为主，边际为辅

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 端到端组级判定（仅有组级GT，GRPO训练）(Priority: P1)

作为质检研发者，我希望在只有工单级（多图）通过/不通过标签的前提下，使用已完成 SFT 的 Qwen2.5‑VL 模型先逐图生成一行中文摘要（Stage‑A），再对全组摘要进行一次汇总决策（Stage‑B），并用 GRPO 将组级奖励的相对优势传回 Stage‑B 回答与各图摘要，从而持续提升组级判定准确率。

**Why this priority**: 只依赖工单级标签即可优化端到端行为，是项目价值的核心闭环。

**Independent Test**: 使用 20–50 个小样本组，在单卡或双卡上运行 `src_post/runner.py`，观察：
- 结果 JSONL 是否按组输出：每图一行摘要、Stage‑B 两行决策文本；
- 奖励方差（std）> 0 且训练稳定（无 NaN/Inf）；
- 若仅训练数百步，组级 margin/准确率较初始有可见提升。

**Acceptance Scenarios**:
1. Given 已训练好的 SFT 检查点和同目录处理器，When 以 YAML 指定 `K_B≥2`、`group_reward_mode=margin_only|combined` 运行训练，Then `results.rank*.jsonl` 按组落盘，且 `reward_std>0` 的步数≥90%。
2. Given 仅有组级 pass/fail 标签，When 训练 1000 步，Then 验证集组级准确率或决策对数边际均值较起点提升（≥+3%acc 或 ≥+0.2 margin）。

---

### User Story 2 - 逐图摘要信用分配优化（Stage‑A 条件式）(Priority: P2)

作为质检研发者，我希望在没有逐图 GT 的情况下，通过条件式信用分配：对每张图采样多个候选摘要，替换入 Stage‑B 提示后重新计算组级奖励差分，把归一化优势回传到该图摘要 token 的对数似然，从而让摘要逐步聚焦对最终判定最关键的证据。

**Why this priority**: 决策质量的瓶颈往往在摘要层，信用分配是提升解释性与稳健性的关键手段。

**Independent Test**: 开启 `train_stage_a_mode=conditional`、`K_A≥2`，在小数据集上运行并检查：
- 日志含 `best_single_delta`，且非零占比>50%；
- `stage_a_top_m`、`use_uncertainty_gate` 开关对优势是否产生门控或聚焦作用。

**Acceptance Scenarios**:
1. Given 条件式模式，When 训练若干步，Then 日志出现 `best_single_delta` 与 `phase_a_entropy_mean`，且 `loss` 非零、无异常。
2. Given `pairwise_credit_enabled=true` 且单图 Δ 弱，When 触发回退，Then 日志含 `pairwise_triggered=true`，并对两图平均回传优势。

---

### User Story 3 - 决策输出一致性与最小偏置提示 (Priority: P3)

作为业务方，我需要 Stage‑B 输出严格两行模板（通过/不通过 + 原因），并在不依赖硬规则的前提下保持“最小提示偏置”，使模型主要从摘要中学习“证据→结论”的加权与一致性偏好。

**Why this priority**: 降低对 checklist 模板的过拟合，提升真实场景泛化。

**Independent Test**: 使用 `use_mission_checklist=false` 和 `true` 两种模式采样，解析决策文本结构是否始终合法；`decision_strict` 相关奖励能对格式违规给出负分。

**Acceptance Scenarios**:
1. Given 最小提示，When 采样 K_B 次，Then 所有候选均能被解析为 {label, reason}，并满足“通过一行/不通过两行”的格式约束。
2. Given 启用 `use_mission_checklist=true`，When 训练 500 步，Then 组级指标不低于最小提示的基线，且原因文本更聚焦 checklist 重点词。

---

### Edge Cases

- 数据不平衡：pass ≫ fail 或相反。需支持 `balance_pass_fail` 采样或离线重采样策略。
- 奖励方差≈0：同一提示 K 个候选奖励相同。应当跳过该头更新并记录 `skip_std0`，或进行有限次数重采样（`max_resample_times`）。
- 特殊标记泄露：摘要/决策出现 `<|...|>`、坐标或方括号。以 `cleanliness/formatting/special_penalty` 轻惩罚，并可在解码期屏蔽相关 token。
- 过长输出：Stage‑B 达到 `max_new_tokens` 无 `<|im_end|>`。按软性超长惩罚（`soft_overlong_penalty`）或缩短上限。
- 多卡一致性：DDP 下每 rank 独立采样，优势标准化应在组内（同一提示）进行；全局仅做标量求和与 ETA 估计。
- 图像占位/张量不一致：渲染文本 `<|image_pad|>` 数与 `image_grid_thw`、`pixel_values` 不匹配时 fail‑fast。
- 任务名/清单缺失或未知：`mission` 无法解析时直接失败并提示可用集合。
- 路径问题：组目录/JSONL 中图片路径不可达时立即报错并指明具体样本。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 Stage‑A 摘要生成**：
  - 每张图片仅输出一行中文摘要；禁止坐标/几何/特殊标记与引号；允许“无法确认/遮挡”等谨慎表述；
  - 使用 `Qwen2VLProcessor.apply_chat_template()` 的 typed image 输入；解码遇到换行或单 token 的 `。` 可提前停止；
  - 生成参数可独立配置（`temperature/top_p/max_new_tokens_stage_a` 等）。

- **FR-002 Stage‑B 决策与解析**：
  - 汇总多行摘要后输出严格两行模板：通过：`总评: 通过`；不通过：`总评: 不通过` + `原因: <关键依据>`；
  - 采用前缀约束构造（`build_decision_prefix_constraint`）确保首行格式；
  - 解析使用 `src_post/prompting/span_parser.py`，产出 `{label, reason}`，失败即丢弃该候选（奖励记 0）。

- **FR-003 奖励与 GRPO**：
  - 组级主信号为 `group_margin = log(p_pass+ε)-log(p_fail+ε)`，由一次 TF 概率得到；
  - `group_reward_mode ∈ {margin_only, combined}`；`combined` 可叠加 `decision_strict/cleanliness/coverage/...` 塑形；
  - 采样 `K_B≥2`，对同一提示计算 z‑score 优势并裁剪到 `[-adv_clip, +adv_clip]`；可选 KL 到参考策略；
  - 可启用“分段裁剪”与“熵掩码”在回答 token 维度稳定更新（可选）。
  - 默认（任务感知组合）：`group_reward_mode=combined`，包含
    `group_margin, decision_strict, cleanliness, special_penalty, rep_penalty, quote_penalty, coverage, taxonomy, consistency`；
    建议权重示例（可微调）：`1.0, 3.0, 0.2, -0.2, -0.7, -0.4, 0.8, 0.2, 0.3`。
  - 默认采样参数（Stage‑B）：`K_B=2, temperature=0.7, top_p=0.95`；
  - 默认 KL 参考：`use_ref_kl=true`，`ref_checkpoint=当前 SFT checkpoint`。

- **FR-004 信用分配（Stage‑A）**：
  - 条件式：固定其他摘要，仅替换第 i 张的候选集合并重算组级奖励差分；优势传到该图摘要 token 的 TF 对数似然；
  - 可选配对回退：当 `best_single_delta < 阈值` 且组级预测错误时，抽取 1–2 对图组合替换并平均回传优势；
  - 可选不确定性门控：基于候选摘要的 TF‑logits 熵阈值，仅对“高熵且提升边际”的候选给优势或放大优势；
  - 预算控制：`K_A/K_set/max_images_tf/stage_a_top_m` 控制成本与聚焦度；
  - 默认门控：`use_uncertainty_gate=true`，`uncertainty_gate_min_entropy=1.2`。

- **FR-005 配置与校验（fail‑fast）**：
  - 使用 `src_post/config.RLRunnerConfig` 严格校验必填键（路径、采样、训练、冻结、奖励、门控、保存等）；
  - `use_ref_kl=true` 时必须给出 `ref_checkpoint`；`group_reward_mode/train_stage_a_mode` 越界直接失败；
  - 路径（checkpoint/processor/dataset）与 mission 清单均需即时校验并抛出可读错误。

- **FR-006 训练与优化**：
  - 选择性解冻：aligner/merger 必训；可按 `llm_top_k_block/vision_top_k_block` 仅放开末端少量层；
  - 三组学习率：`aligner_lr/llm_lr/vision_lr`；支持余弦+warmup；梯度裁剪与梯度累积；
  - 可按步保存检查点并滚动保留 `save_limit` 个目录。
  - 默认训练日程（基线）：
    - 阶段1（Stage‑B 预热）：`train_stage_a_mode=off`，`train_stage_b=true`，`stage_b_weight=1.0`，可选小权重 KL 到参考（`lambda_kl_stage_b≈0.02`）。
    - 阶段2（协同训练）：`train_stage_a_mode=conditional`，`stage_a_weight=1.0`，`stage_b_weight=1.0`，`K_A∈{2,3}`，可保留小 KL 稳定风格。

- **FR-007 生成与约束**：
  - Stage‑B 允许最小偏置提示或带 mission checklist 的提示；
  - 默认：`use_mission_checklist=true`；
  - 运行期可选择屏蔽几何/坐标 token（仅解码期，不影响梯度）。

- **FR-008 日志与产物**：
  - 结果：`results.rank{R}.jsonl`，含每图摘要、候选决策及奖励、最佳候选；
  - 指标：`metrics.rank0.jsonl` + 可选 TensorBoard（`tb_log_dir/run_name`）；
  - 关键统计：`reward_best_mean/std/acc_best/any_hit/resp_len/kl_b/skip_std0/pairwise_trigger_rate/decision_ce_ema/eta_hours` 等。

- **FR-009 并行与多卡**：
  - 默认硬件与并行：`8×A100‑80G`，优先使用 HuggingFace Accelerate 启动多卡训练；
  - 兼容 `torch.distributed.run` / DDP；`WORLD_SIZE=8` 时按 rank 切分样本索引；
  - 各 rank 独立采样候选与计算优势；teacher‑forcing/KL 在 DDP 包裹下做全归约。

- **FR-010 数据输入**：
  - 目录布局：`审核通过|审核不通过/{group_id}/*.jpg`；或 JSONL（每行含 `images: [..], label: pass|fail`）。

### Key Entities *(include if feature involves data)*

- **Group（工单）**：一个审核任务的多张图片集合；属性：`images[], label(pass|fail), meta{group_id}`。
- **Image（图片）**：输入图片本体；在 Stage‑A 产生一行中文摘要。
- **SummaryLine（摘要）**：每图一行的中文短句；无坐标/特殊标记；可能含谨慎表述；作为 Stage‑B 上下文。
- **StageBDecision（决策）**：两行文本产物；解析为 `{label, reason}`。
- **Reward/Advantage（奖励/优势）**：对候选回答/摘要集合的标量得分与 z‑score 优势，用于 GRPO 更新。
- **RLRunnerConfig**：训练配置载体；包含路径、采样、冻结、奖励、门控、日志与保存等必填键。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 组级准确率/边际提升（KPI优先级）**：主KPI=组级准确率；辅KPI=决策对数边际。
  - 在固定验证集上，训练 1 个 epoch 或等价步数后，较初始 SFT：
  - 组级准确率提升 ≥ +3% 或
  - 决策对数边际均值提升 ≥ +0.2。
- **SC-002 训练稳定性**：全程无 NaN/Inf；`reward_std==0` 的步占比 ≤ 10%；无死锁；显存峰值稳定在可接受范围内。
- **SC-003 文本规范度（Stage‑A/B）**：
  - 摘要清洁度（无 `<|...|>`/方括号/坐标）≥ 0.98；
  - 决策格式严格解析成功率 ≥ 0.99。
- **SC-004 产物完备性**：
  - 生成 `results.rank*.jsonl`、`metrics.rank0.jsonl`、TensorBoard 目录，以及 `output_dir/checkpoints/grpo_final/`；
  - 重运行可复现（同随机种子，指标方差在合理范围）。
- **SC-005 可操作性**：
  - 单条命令可启动训练：`bash scripts/run_group_qc_rl.sh /abs/config.yaml`；
  - 必填 YAML 缺失将 fail‑fast 并给出可读错误与“合法键”列表。
