# GPT-Without-Libraries 🤖

从零构建的微型Transformer GPT模型，仅使用numpy等基础运算库，不依赖任何深度学习框架。

## 🎯 项目特点

- ✅ **纯numpy实现** - 无需torch、tensorflow等深度学习框架
- ✅ **完整Transformer架构** - 包含多头注意力、前馈网络、层归一化等
- ✅ **自动微分系统** - 手动实现所有组件的反向传播
- ✅ **稳定数值计算** - 防止梯度爆炸/消失，支持长序列训练
- ✅ **模块化设计** - 每个组件独立，便于学习和扩展

## 📁 项目结构

```
GPT-Without-Libraries/
├── src/                    # 源代码目录
│   └── gpt_without_libs/   # 主包
│       ├── models/         # 模型组件
│       │   ├── core/       # 核心模型
│       │   └── layers/     # 神经网络层
│       ├── training/       # 训练系统
│       └── utils/          # 工具函数
├── tests/                  # 测试文件
├── examples/               # 示例和训练脚本
├── docs/                   # 文档
├── data/                   # 数据文件
├── requirements.txt        # 依赖列表
├── setup.py               # 安装脚本
├── pyproject.toml         # 项目配置
└── README.md              # 项目说明
```

## 🚀 快速开始

### 环境要求
- Python 3.7+
- numpy

### 安装依赖
```bash
# 从源码安装
pip install -e .

# 或者仅安装依赖
pip install -r requirements.txt
```

### 快速测试
```bash
# 运行完整测试套件
python -m pytest tests/

# 快速训练测试
python examples/main.py --quick

# 完整训练演示
python examples/main.py
```

## 🧮 模型架构

### 核心组件
- **词嵌入层**: 将token索引映射到高维向量
- **位置编码**: 为序列添加位置信息
- **多头注意力**: 自注意力机制，捕捉序列依赖关系
- **前馈网络**: 非线性变换，增强模型表达能力
- **层归一化**: 稳定训练过程
- **残差连接**: 防止梯度消失

### 模型配置
```python
# 默认小型模型配置
vocab_size = 1000          # 词汇表大小
embed_dim = 256            # 嵌入维度
num_heads = 8              # 注意力头数
num_layers = 6             # Transformer层数
hidden_dim = 1024          # 前馈网络隐藏层维度
max_seq_len = 512          # 最大序列长度
```

### 参数规模
- **小型模型**: ~50万参数
- **中型模型**: ~500万参数
- **大型模型**: 可根据需要扩展

## 💡 使用示例

### 基本训练
```python
from gpt_without_libs.models.core.mini_gpt import MiniGPT
from gpt_without_libs.utils.tokenizer import SimpleTokenizer, generate_sample_data
from gpt_without_libs.training.training import Trainer

# 生成训练数据
texts = generate_sample_data(1000, 50, 150)

# 创建分词器
tokenizer = SimpleTokenizer()
tokenizer.train(texts)

# 创建模型
model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=3
)

# 创建训练器并训练
trainer = Trainer(model, tokenizer)
trainer.train(texts, epochs=10, batch_size=4)
```

### 文本生成
```python
# 加载训练好的模型
model = MiniGPT(vocab_size=1000)
model.load("model_checkpoint.pkl")

# 生成文本
trainer = Trainer(model, tokenizer)
generated_text = trainer.generate(
    prompt="The weather is",
    max_length=50,
    temperature=0.8
)
print(generated_text)
```

## 🧪 测试验证

项目包含完整的测试套件，验证所有组件功能：

```bash
python -m pytest tests/
```

测试覆盖：
- ✅ 词嵌入层前向/反向传播
- ✅ 位置编码正确性
- ✅ 多头注意力计算
- ✅ 前馈网络梯度
- ✅ 层归一化数值稳定性
- ✅ Transformer块端到端
- ✅ 完整模型训练流程

## 📊 性能表现

### 训练效果
- **收敛性**: 损失稳定下降
- **生成质量**: 语法基本合理，语义连贯
- **训练速度**: 纯numpy实现，适合学习研究

### 数值稳定性
- 使用稳定的softmax实现
- 梯度裁剪防止爆炸
- 合理的权重初始化
- 层归一化保证稳定

## 🔧 技术亮点

### 数学实现
- **GELU激活函数**: 精确数学近似
- **余弦位置编码**: 原始Transformer实现
- **Scaled Dot-Product Attention**: 完整注意力机制
- **交叉熵损失**: 数值稳定版本

### 工程实践
- **模块化设计**: 每个组件独立可测试
- **内存管理**: 合理的缓存策略
- **错误处理**: 完善的异常处理机制
- **代码文档**: 详细的注释和文档

## 📚 学习价值

本项目适合：
- **深度学习入门**: 理解Transformer核心原理
- **数值计算实践**: 手动实现神经网络
- **代码架构设计**: 学习模块化编程
- **研究实验**: 快速原型开发

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
git clone <repository>
cd GPT-Without-Libraries
pip install numpy
python3 test_all.py  # 验证环境
```

### 代码规范
- 使用类型提示
- 添加详细注释
- 保持模块化设计
- 编写对应测试

## 📄 许可证

MIT License

## 🙏 致谢

本项目基于以下研究：
- "Attention Is All You Need" (Vaswani et al., 2017)
- "GPT-3: Language Models are Few-Shot Learners" (Brown et al., 2020)

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注意**: 本项目为教育演示目的，生产环境请使用成熟的深度学习框架。