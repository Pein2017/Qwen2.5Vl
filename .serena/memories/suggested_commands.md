# 项目关键命令

## 环境激活
```bash
conda activate ms
cd /data3/Qwen2.5-VL-main
```

## src_rl/ RL 后训练
```bash
# 组件加载测试
/root/miniconda3/envs/ms/bin/python -m src_rl.runner --config configs/rl/dense_grpo.yaml --mode load

# 开始 GRPO 训练
/root/miniconda3/envs/ms/bin/python -m src_rl.runner --config configs/rl/dense_grpo.yaml --mode train

# 离线评估
/root/miniconda3/envs/ms/bin/python -m src_rl.eval --config configs/rl/dense_grpo.yaml --input_file data/ds_v2_full/val.jsonl --data_root data/ds_v2_full

# 运行单元测试
pytest src_rl/rewards/test_format_rewards.py -v
pytest src_rl/rewards/test_detection_rewards.py -v
pytest src_rl/tools/test_parity_check.py -v
```

## src_new/ SFT 训练  
```bash
# 标准训练
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config configs/phase_3/standard.yaml --log_level INFO

# 推理
/root/miniconda3/envs/ms/bin/python -m src_new.inference --model_path /path/to/checkpoint --data_root /path/to/data --dataset val --output_file results.jsonl
```

## 数据转换
```bash
bash data_conversion/convert_dataset.sh
```

## 测试命令
```bash
pytest tests/ -v
pytest src_rl/ -v
```

## GPU 和环境
- 使用 `/root/miniconda3/envs/ms/bin/python` 确保环境一致
- 支持单 GPU 和多 GPU 训练
- bf16 自动检测和降级