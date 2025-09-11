## 群组质检后训练（GRPO）从零到一教程

本教程面向熟悉 SFT（有监督微调）但对 RL 尤其是 GRPO 不熟悉的同学，详细讲解 `src_post/` 模块如何把一个已经在 `src_new/` 里做过 SFT 的 Qwen2.5‑VL 继续做“群组级（工单级）质检”的强化学习后训练。

- 两个阶段：
  - Stage‑A（单图摘要）：对每张图生成一行简洁中文摘要（用于 Stage‑B 的上下文）。
  - Stage‑B（群组决策）：汇总所有摘要，输出“总评: 通过/不通过”，不通过时给出简短“原因”。
- 方法：GRPO（Group Relative Policy Optimization）。不需要价值头（value head），只靠同一提示下采样多条回答的相对排名信号来更新策略。

---

## 0. 快速开始（5 分钟）

```bash
# 从仓库根目录
conda activate ms
bash scripts/run_group_qc_rl.sh
```

配置文件（默认 `configs/rl/group_qc_grpo.yaml`）的关键字段：
```yaml
checkpoint: /abs/path/to/sft_checkpoint
processor:  /abs/path/to/processor
train_data_dir: /abs/path/to/train_groups
output_dir: /abs/path/to/output_post/grpo

# 采样与生成
K_B: 4          # Stage‑B 每个群组采样条数
K_A: 3          # Stage‑A（若开启）每个图的候选摘要数
train_stage_a_mode: conditional   # 可选 off/conditional/joint
stage_a_weight: 1.0

# KL 正则
use_ref_kl: true
lambda_kl_stage_b: 0.02
lambda_kl_stage_a: 0.02

# 奖励组合（含惩罚）
reward_fns: label_match,formatting,cleanliness,rep_penalty,quote_penalty,special_penalty
reward_weights: 1.0,0.5,0.7,-0.7,-0.4,-1.0
```

数据目录（推荐）：
```text
/abs/path/to/data/
  审核通过/
    <group_id>/image_*.jpg
  审核不通过/
    <group_id>/image_*.jpg
```
标签会被正则化为 `pass|fail`。

---

## 1. 总览与与 SFT 的差异

- SFT：用参考文本最小化交叉熵；训练时 teacher‑forcing，目标是“拟合参考”。
- 本模块（RL, GRPO）：
  - 仍然 teacher‑forcing，但监督信号来自“同一提示的多条采样回答之间的相对好坏”（z‑score 优势），而不是固定参考答案。
  - 不需要价值头；无需估计每个 token 的回报，适合短文本决策、主观噪声较大的场景。
  - 通常只微调 LM 头 + 若干顶层语言层（vision 侧保持不变）。

训练目标（概念）：
- Stage‑B：对同一群组的 K_B 条回答，奖励越高的回答，其 log 概率被“推高”；反之被“压低”。
- Stage‑A（可选）：用群组决策的奖励对“哪种单图摘要更有用”进行条件信用分配，让摘要风格也逐渐贴近任务偏好。

---

## 2. 目录与关键文件

- `src_post/grpo_runner.py`：主入口（数据迭代、对话构造、生成、奖励、GRPO 更新、KL 正则、优化器等）。
- `src_post/conversation.py`：Stage‑A/Stage‑B 提示构造与任务提示（mission hints）。
- `src_post/dataset_group_qc.py`：群组级数据集（目录或 JSONL）。
- `src_post/rewards/`：奖励函数注册与具体实现（匹配、格式、洁净度、重复惩罚、引号惩罚、特殊标记惩罚、任务一致性、taxonomy）。
- `src_post/span_parser.py`：Stage‑B 输出解析器（正则提取“总评/原因”）。
- `src_post/logits_processors.py`：推理期的几何/坐标 token 掩蔽（可选）。
- `src_post/utils/text.py`：格式/提示覆盖度等评分工具。

---

## 3. 数据与加载

数据两种来源：
- 目录树：`train_data_dir/审核通过|审核不通过/<group_id>/*.jpg`（`dataset_group_qc.py` 自动正则化为 `pass|fail`）。
- JSONL：每行形如 `{ "images": [..], "label": "pass|fail", "meta": {...} }`。

返回样本字典：
```python
{
  "image_paths": List[str],
  "images": List[PIL.Image],
  "label": "pass" | "fail",
  "meta": Dict[str, Any],
  "num_images": int,
}
```

---

## 4. 对话与编码（Stage‑A/Stage‑B）

- 使用官方 `Qwen2VLProcessor` 的 `apply_chat_template`。Stage‑A 的 user 消息包含 typed 内容 `[{'type':'image'}, {'type':'text', ...}]`。
- 我们沿用 SFT 阶段的视觉预处理（`smart_resize + EXIF 校正`）以保持视觉 token 对齐一致性。
- Stage‑A 采用“贪心”生成一行摘要作为 Stage‑B 的上下文；如果训练 Stage‑A，则另行“采样”候选摘要用于 GRPO。
- Stage‑A 生成有自定义停止条件：遇到换行或 `。`（若是单 token 编码）即停止。

伪代码（对话与编码）：
```python
# Stage‑A（单图）
messages_a = build_stage_a_messages(mission)
text_a = processor.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
enc_a = processor(text=[text_a], images=[img_proc], return_tensors="pt")

# Stage‑B（纯文本）
messages_b = build_stage_b_messages(summary_lines=context_lines, checklist_lines=checklist)
text_b = processor.apply_chat_template(messages_b, tokenize=False, add_generation_prompt=True)
enc_b = processor(text=[text_b], images=None, return_tensors="pt")
```

---

## 5. 前向与目标（数学与细节）

记：
- 提示 token 序列为 \(x\)（含系统与用户 prompt，Stage‑B 不带图像）。
- 采样得到的回答为 \(y^{(k)} = (y^{(k)}_1, \dots, y^{(k)}_{T_k})\)。
- 模型策略为 \(\pi_\theta\)，参考模型为 \(\pi_{\mathrm{ref}}\)（可与初始策略相同并冻结）。

### 5.1 Teacher‑Forcing 对数似然

我们在“提示+回答”的拼接上做一次前向，构造“只在回答位置”的 mask，取这些位置的逐 token 对数概率并求和：
$$
\log p_\theta\big(y^{(k)}\mid x\big)
= \sum_{t=1}^{T_k} \log \pi_\theta\big( y^{(k)}_t\,\big|\, x, y^{(k)}_{<t} \big).
$$

代码要点：
- 将 `full_ids = [prompt_ids, resp_ids]` 拼接；`logits[:, :-1, :]` 与 `targets = full_ids[:, 1:]` 对齐。
- 用 `mask[:, prompt_len-1:] = True` 仅选择“回答对应的 next‑token 目标位置”。
- 若 `length_norm=True`，则再除以回答 token 数，抑制长回答优势。

### 5.2 GRPO 优势（z‑score）

对同一提示采样 \(K\) 条回答，得到奖励 \(r_1, \dots, r_K\)。
$$
A_k = \mathrm{clip}\Bigg( \frac{r_k - \overline{r}}{\mathrm{std}(r)+\varepsilon},\; -a,\; a \Bigg),
$$
其中 \(\overline{r}\) 是均值，`std` 为无偏差或有偏差方差的标准差（实现中用总体标准差），其中 \(a\) 为 `adv_clip`。若 \(\mathrm{std}\approx 0\) 则令 \(A_k=0\)（跳过更新）。

### 5.3 Stage‑B 目标（群组决策）

$$
\mathcal{L}_\text{B} 
:= \sum_{k=1}^{K_B} \Big( -\,\operatorname{sg}(A_k)\cdot \log p_\theta(y^{(k)}\mid x) \Big)
\; + \; \lambda^{B}_{\mathrm{KL}}\,\mathrm{KL}_\text{resp}(\pi_\theta\,\Vert\,\pi_{\text{ref}}),
$$
其中响应段 KL 的实现是在“回答位置”取平均：
$$
\mathrm{KL}_\text{resp} = \frac{1}{T}\sum_{t\in \mathrm{resp}} \sum_{v} p_\theta(v\mid z_t)\,\Big(\log p_\theta(v\mid z_t) - \log p_{\mathrm{ref}}(v\mid z_t)\Big).
$$

### 5.4 Stage‑A 目标（可选）

- 条件式（conditional）：对第 \(i\) 张图固定其他摘要，采样 \(K_A\) 个候选摘要 \(s_i^{(k)}\)。把 \(s_i^{(k)}\) 填入上下文，重建 Stage‑B 提示，得到群组奖励 \(r_i^{(k)}\) 并做 z‑score：
  $$
  \mathcal{L}^{(i)}_\text{A} 
  = \sum_{k=1}^{K_A} \Big( -\,\operatorname{sg}(A_i^{(k)})\cdot \log p_\theta\big(s_i^{(k)}\mid x_i\big) \Big)
  + \lambda^{A}_{\mathrm{KL}}\,\mathrm{KL}_{\mathrm{resp}}(\pi_\theta\,\Vert\,\pi_{\text{ref}}).
  $$
  这里 \(x_i\) 是单图 Stage‑A 的提示；`log p` 只在该摘要的 token 上求和。

- 联合式（joint）：一次性为整组图采样 \(K_{\text{set}}\) 组摘要集合 \(S^{(s)}=\{s_1^{(s)},...,s_n^{(s)}\}\)，对每组集合跑一次 Stage‑B 奖励并 z‑score，loss 为“该组内所有摘要的 log 概率之和”乘以对应优势：
  $$
  \mathcal{L}_\text{A} = \sum_{s=1}^{K_{\mathrm{set}}} \Big( -\,\operatorname{sg}(A^{(s)}) \cdot \sum_{i=1}^{n} \log p_\theta\big(s_i^{(s)}\mid x_i\big) \Big) + \mathrm{KL}\,\text{项}.
  $$

### 5.5 总损失

$$
\mathcal{L} = \mathcal{L}_\text{B} + \omega_\text{A}\,\mathcal{L}_\text{A},\quad \omega_\text{A}=\mathrm{stage\_a\_weight}.
$$
优化器使用 AdamW，分组设置对 LM 头、最后若干层、可选 projector/aligner 使用不同学习率，梯度裁剪 `max_grad_norm`。

---

## 6. 奖励函数（`src_post/rewards/`）

- `label_match`：二值匹配（预测 `pass|fail` 与 GT 是否一致）。
- `formatting`：`utils/text.compute_formatting_score`，衡量摘要对“任务提示（hints）”的覆盖与简洁性，范围 [0,1]。
- `cleanliness`：非法标记/坐标/过多拉丁字符等的洁净度惩罚（越干净分越高）。
- 负向惩罚（需配负权重）：
  - `rep_penalty`：重复词/跑字惩罚 ∈[0,1]；
  - `quote_penalty`：出现引号记 1；
  - `special_penalty`：出现 `<|...|>` 记 1。
- `consistency`：Stage‑B `原因` 与摘要/检查项的字符集合重叠一致性，归一化 [0,1]。
- `taxonomy`：可选，基于外部词汇表的命中评分（若文件缺失，默认为 0）。

最终奖励按权重加权后再做“绝对权重和归一化”：
$$
R = \frac{\sum_j w_j r_j}{\sum_j |w_j| + \epsilon}.
$$

---

## 7. 前向过程（端到端伪代码）

```python
# 输入：一条群组样本（多张图，标签 pass|fail）
# 输出：一次更新（或仅评估）

# 1) Stage‑A 构造上下文（贪心，每图一行）
context_lines = []
for img in images:
    enc_a = encode_stage_a(img, mission)
    line = generate_one_line(enc_a, greedy=True, stop_on_newline=True)
    line = sanitize(line) if sanitize_stage_a else line
    context_lines.append(line)

# 2) Stage‑B 采样 K_B 条回答并打分 → 优势
enc_b = encode_stage_b(context_lines, checklist)
replies, rewards = [], []
for _ in range(K_B):
    y = sample_response(enc_b, cfg_b)
    replies.append(y)
    parsed = parse_decision(y)
    r = compose_reward(gt_label, parsed.label, context_lines, parsed.reason)
    rewards.append(r)
A = zscore_and_clip(rewards, adv_clip)

# 3) Stage‑B teacher‑forcing 求和 logp + 可选 KL
loss_b = 0.0
for k in range(K_B):
    logp_k = tf_sum_logprob_over_response(model, enc_b, replies[k], length_norm)
    loss_b += -(stop_grad(A[k]) * logp_k)
if use_ref_kl and lambda_kl_b > 0:
    loss_b += lambda_kl_b * mean_kl_to_ref_over_response(model, ref_model, enc_b, replies)

# 4) Stage‑A GRPO（可选 conditional 或 joint）
loss_a = 0.0
if train_stage_a_mode == 'conditional':
    for i, img in enumerate(images):
        cand, rewards_i = [], []
        enc_ai = encode_stage_a(img, mission)
        for _ in range(K_A):
            s = sample_one_line(enc_ai, cfg_a, stop_on_newline=True)
            variant = context_lines.copy(); variant[i] = sanitize(s) if sanitize_stage_a else s
            r = eval_stage_b_reward(variant, checklist, gt_label)
            cand.append(s); rewards_i.append(r)
        A_i = zscore_and_clip(rewards_i, adv_clip)
        for k, s in enumerate(cand):
            logp = tf_sum_logprob_over_response(model, enc_ai, s, length_norm)
            loss_a += -(stop_grad(A_i[k]) * logp)
    if use_ref_kl and lambda_kl_a > 0:
        loss_a += lambda_kl_a * mean_kl_stage_a_over_candidates(...)
elif train_stage_a_mode == 'joint':
    sets, set_rewards = [], []
    for s in range(K_set):
        set_lines, set_tok_ids = [], []
        for img in images:
            enc_ai = encode_stage_a(img, mission)
            s_i = sample_one_line(enc_ai, cfg_a, stop_on_newline=True)
            set_lines.append(s_i); set_tok_ids.append(tokens(s_i))
        r_set = eval_stage_b_reward(set_lines, checklist, gt_label)
        sets.append(set_tok_ids); set_rewards.append(r_set)
    A_set = zscore_and_clip(set_rewards, adv_clip)
    for s, tok_ids_per_img in enumerate(sets):
        logp_sum = 0.0
        for i, img in enumerate(images):
            enc_ai = encode_stage_a(img, mission)
            logp_sum += tf_sum_logprob_over_response(model, enc_ai, tok_ids_per_img[i], length_norm)
        loss_a += -(stop_grad(A_set[s]) * logp_sum)
    if use_ref_kl and lambda_kl_a > 0:
        loss_a += lambda_kl_a * mean_kl_stage_a_over_sets(...)

# 5) 合并与更新
loss = loss_b + stage_a_weight * loss_a
optimizer.zero_grad(); loss.backward()
clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

---

## 8. 生成与解码细节

- 生成配置：
  - Stage‑A 默认 `repetition_penalty=1.6, no_repeat_ngram_size=24, max_new_tokens<=64`；
  - Stage‑B 默认 `repetition_penalty=1.3, no_repeat_ngram_size=16, max_new_tokens<=200`；
  - 均支持 `temperature, top_p`。
- 停止条件（Stage‑A）：若换行或 `。` 被编码为单个 token，会在首次生成到这些 token 时停止。
- 可选掩蔽：`mask_geometry_tokens / mask_coordinate_tokens` 在推理期屏蔽对应 token。
- 文本清洗（可选）：`sanitize_stage_a=true` 会去除 `<|...|>`、坐标样式、引号等，去重分隔片段并做长度裁剪（80 字）。

---

## 9. 可训练参数与优化器

- 冻结全模型 → 解冻 LM 头（`train_lm_head`）与“最后 N 层语言层”（`train_last_n_layers`）。
- 可选额外解冻 aligner/merger/projector（`train_aligner`），自动在常见路径下探测模块并单独设置学习率。
- 优化器 AdamW：按 param group 设置 `lr_lm_head / lr_last_layers / lr_aligner`。未配置则回退到 `learning_rate`。

---

## 10. 多卡与结果输出

- 多卡通过环境变量 `RANK/WORLD_SIZE/LOCAL_RANK` 自动分片：只取 `idx % world_size == rank` 的样本。
- 每个 rank 写 `results.rank{RANK}.jsonl`：包含 `group_index, gt_label, pred_label, reward, images[captions], stage_b_raw`。
- 训练结束保存权重到 `output_dir/checkpoints/grpo_final/`（同目录下尝试保存 processor）。

---

## 11. 配置项一览（与实现对齐）

- 采样与优势：`K_B, K_A, K_set, adv_clip, length_norm`。
- KL：`use_ref_kl, ref_checkpoint, lambda_kl_stage_b, lambda_kl_stage_a`。
- 生成：`temperature, top_p, max_new_tokens_stage_a, max_new_tokens_stage_b`。
- 掩蔽与清洗：`mask_geometry_tokens, mask_coordinate_tokens, sanitize_stage_a`。
- 训练：`num_updates, batch_size, learning_rate, weight_decay, max_grad_norm, train_lm_head, train_last_n_layers, train_aligner, lr_*`。
- Stage‑A 模式：`train_stage_a_mode in {off, conditional, joint}, stage_a_weight`。
- 奖励：`reward_fns, reward_weights`（长度会自动对齐，多余截断，不足补 1.0）。

---

## 12. GRPO vs PPO（直觉与取舍）

- PPO：需要价值头、clip 比例、优势估计等，工程开销与对尺度敏感度更高；样本效率更好。
- GRPO：取消价值头，依赖同一提示的“相对排序”信号（z‑score）。特别适合短回答、主观噪声场景（只关心相对好坏）。
- 本任务中：Stage‑B 短文本决策非常适配 GRPO；Stage‑A 用条件式/集合式进一步把群组信号下沉到单图摘要上。

---

## 13. 常见问题与调参建议

- 摘要重复/口水：
  - 提高 Stage‑A `repetition_penalty` 或 `no_repeat_ngram_size`；
  - 提高 `rep_penalty` 的负权重（如 `-0.7 → -1.0`）。
- 出现 `<|...|>` 或坐标痕迹：
  - 开启 `sanitize_stage_a` 与 `special_penalty` 负权重；
  - 评测时可启用 `logits_processors` 层面的掩蔽。
- 优势方差过小（std≈0）：
  - 跳过该步（实现已处理），或适度增大 `K_B/K_A`。
- 模型漂移：
  - 适当提高 KL 系数、调小学习率、先只训练少量顶层。
- 标签识别错误：
  - 强化 `label_match` 权重，并检查 `span_parser.py` 的正则是否覆盖你的格式变体。

---

## 14. 扩展与二次开发

- 自定义奖励：在 `src_post/rewards/` 新增函数，并注册到 `REGISTRY`（`rewards/__init__.py`）。
- 自定义任务提示：在 `conversation.py` 的 `MISSION_STAGE_A_HINTS` 中增改条目，或动态构造 checklist。
- 数据格式：可从目录树切到 JSONL 以携带更丰富的 `meta`。
- 解码约束：`logits_processors.py` 可扩展更多“推理期”约束，不影响训练梯度。

---

## 15. 与代码实现的关键对齐点

- Teacher‑forcing 仅在“回答位置”累计 log 概率（`prompt_len-1:`）。
- Stage‑A 上下文用于构造 Stage‑B 提示，默认用“贪心最佳”摘要；训练 Stage‑A 时另行“采样候选摘要”。
- KL 仅在回答位置计算；Stage‑A/Stage‑B 分别有各自的 `lambda_kl`。
- 参数组解冻顺序：全冻结 → LM 头 → 最后 N 层 → 可选 aligner/projector。

---

## 16. 参考入口与运行方式

- 入口：
  - 训练/评测：`python -m src_post.grpo_runner --config /abs/path/to/group_qc_grpo.yaml`
  - 脚本：`bash scripts/run_group_qc_rl.sh`
- 环境：
  - 使用 `ms` conda 环境：`conda activate ms`；
  - 直连 Python：`/root/miniconda3/envs/ms/bin/python`。

---

## 17. 迷你算例（直觉演示）

- 同一群组采样 3 条决策：`r = [0.2, 1.0, 0.8]` → z‑score 约 `A ≈ [-1.1, +0.8, +0.3]`；
  - Loss：`L_B = -A1*logp(y1) - A2*logp(y2) - A3*logp(y3) + λ KL`，因此推高 y2,y3、压低 y1。
- 某图的 3 条摘要候选：`r_i = [0.9, 0.1, 0.7]` → `A_i ≈ [+0.8, -1.0, +0.2]`；
  - `L_A` 在该图的摘要 token 上做相同加权。

---

若你已熟悉 SFT，本教程给出的前向/目标及伪代码足以复刻 `src_post/` 的关键路径。推荐结合源码文件名搜索对应实现以加深理解。祝训练顺利！
