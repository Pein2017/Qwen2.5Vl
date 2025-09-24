# Qwen2.5-VL Project Overview

## Purpose
多模态AI质量检测系统，专门用于BBU设备检测和描述，基于Qwen2.5-VL视觉语言模型。

## 主要架构
- **src_new/** (推荐): 组合式架构，测试覆盖全面，生产就绪
- **src/** (遗留): 成熟代码库，复杂继承结构
- **src_rl/** (RL后训练): GRPO强化学习后训练模块
- **src_post/** (后训练): 其他后训练方法

## 技术栈
- Python 3.12+
- PyTorch
- Transformers (HuggingFace)
- TRL (Transformers Reinforcement Learning)
- Qwen2.5-VL模型

## 核心组件
- 数据转换：data_conversion/
- 配置管理：configs/
- 脚本工具：scripts/
- 测试：tests/, tests_post/