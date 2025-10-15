# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个从零构建的微型Transformer GPT模型项目，仅使用numpy等基础运算库，不依赖任何深度学习框架。项目实现了完整的Transformer架构，包括多头注意力、前馈网络、层归一化等核心组件。

## 开发环境配置

### 依赖要求
- Python 3.7+
- numpy (唯一依赖)

### 安装依赖
```bash
pip install numpy
```

## 常用命令

### 测试和验证
```bash
# 运行完整测试套件
python -m pytest tests/

# 快速训练测试
python examples/main.py --quick

# 完整训练演示
python examples/main.py
```

### 单组件测试
各个模块都包含独立的测试函数，可以通过以下方式运行：
```bash
# 测试单个模块
python -c "from gpt_without_libs.models.layers.embedding import test_embedding; test_embedding()"
python -c "from gpt_without_libs.models.layers.attention import test_multihead_attention; test_multihead_attention()"
python -c "from gpt_without_libs.models.core.mini_gpt import test_mini_gpt; test_mini_gpt()"
```

## 核心架构

### 模型组件结构
1. **嵌入层** (`embedding.py`): 词嵌入实现，支持前向和反向传播
2. **位置编码** (`positional_encoding.py`): 正弦/余弦位置编码，固定不参与训练
3. **多头注意力** (`attention.py`): 自注意力机制，支持掩码和dropout
4. **前馈网络** (`feedforward.py`): 双层MLP + GELU激活函数
5. **层归一化** (`layer_norm.py`): 可学习的gamma和beta参数
6. **Transformer块** (`transformer_block.py`): 完整的解码器块，包含残差连接
7. **主模型** (`mini_gpt.py`): 完整的GPT架构实现

### 训练系统
- **分词器** (`tokenizer.py`): 字符级分词器，支持特殊token
- **训练器** (`training.py`): 完整的训练循环实现，包含交叉熵损失、余弦退火学习率调度、Top-K采样等

### 关键设计特点
- **纯numpy实现**: 无深度学习框架依赖
- **完整的自动微分**: 手动实现所有组件的反向传播
- **稳定的数值计算**: 防止梯度爆炸/消失
- **模块化设计**: 每个组件独立，便于测试和学习

## 模型配置

### 默认模型参数
- 词汇表大小: 1000-5000 (可配置)
- 嵌入维度: 256
- 注意力头数: 8
- Transformer层数: 6
- 前馈网络维度: 1024
- 最大序列长度: 512
- 总参数数: ~524万

### 轻量级配置 (快速测试)
- 嵌入维度: 64
- 注意力头数: 2
- Transformer层数: 2
- 前馈网络维度: 128
- 最大序列长度: 32

## 训练特性

- **损失函数**: 交叉熵
- **优化器**: SGD (手动实现)
- **学习率调度**: 余弦退火
- **正则化**: Dropout (0.1)
- **数值稳定性**: 梯度裁剪、稳定的softmax

## 开发规范

### 代码结构
- 每个组件都是独立的模块
- 包含前向传播 `forward()` 和反向传播 `backward()` 方法
- 参数更新通过 `update()` 方法实现
- 所有模块都有对应的测试函数

### 注意事项
1. 项目编译运行较快，每次更改后都应该重新构建测试
2. 纯numpy实现，训练速度相对较慢
3. 仅支持字符级分词
4. 在进行大型更改前，建议备份关键组件
5. 每次更改项目后，更新 VERSIONS.md 记录版本信息

### 文件命名规范
- 模型组件: `component_name.py` (如 `attention.py`)
- 训练相关: `training.py`, `main.py`
- 测试文件: `test_all.py`
- 文档: `README.md`, `VERSIONS.md`

## 测试策略

项目包含完整的测试套件，验证所有组件功能：
- 组件级别的单元测试
- 端到端的训练测试
- 数值稳定性验证
- 梯度计算正确性检查

## 常见问题

### 训练相关
- 如果损失不下降，检查学习率设置
- 如果出现数值溢出，检查梯度裁剪设置
- 快速测试使用 `--quick` 参数

### 模型保存/加载
- 模型保存为 `.pkl` 格式
- 包含所有可训练参数
- 加载时自动验证模型配置匹配