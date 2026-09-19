# GPT Without Libraries

从零实现的小型 decoder-only Transformer。前向传播、手写反向传播、AdamW、数据准备、checkpoint 和生成只使用 Python 标准库与 NumPy；NVIDIA GPU 可选用 CuPy。不依赖 PyTorch、TensorFlow、JAX 或 Transformers。

这是学习和小规模实验项目。仓库不包含预训练权重或历史 30 万条训练数据；随附的 24 条原创对话仅用于跑通流程，不能据此宣称模型具备可靠问答或推理能力。

## 快速开始（CPU）

Python 3.11+，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python prepare_data.py \
  --input examples/tiny_dialogues.jsonl \
  --out-dir data/tiny --seq-len 32 --val-frac 0.2 \
  --vocab-size 512 --assistant-loss-only

python train.py \
  --data-dir data/tiny --checkpoint-dir data/tiny_checkpoints \
  --steps 100 --batch-size 4 --d-model 32 --n-layers 2 --n-heads 4 \
  --lr 0.003 --min-lr 0.0003 --warmup-steps 10 \
  --eval-every 50 --eval-iters 5 --save-every 50 --device cpu

python sample.py \
  --data-dir data/tiny --checkpoint data/tiny_checkpoints/best.npz \
  --prompt $'用户：你好。\n助手：' --max-new-tokens 60 \
  --temperature 0 --device cpu
```

训练从 `meta.json` 读取上下文长度；采样从 checkpoint 读取完整模型配置。生成包含原始 prompt；`--temperature 0` 为贪心解码，正数为概率采样，`--top-k 0` 关闭 top-k，`--seed` 控制可复现采样。建议实际对话 prompt 使用 `用户：问题\n助手：` 的格式（其中 `\n` 为真实换行）。

默认 CPU，CUDA 环境请自行安装与驱动匹配的 CuPy 发行包，并给训练和采样加 `--device cuda`。没有 CuPy 时明确报错，不会悄悄回退到 CPU。GPU 路径尚未在本次修复中实机验证。

## 自己的数据

每行一条完整对话，UTF-8 JSONL：

```json
{"conversations":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，很高兴和你交流。"}]}
```

```bash
python prepare_data.py --input dataset.jsonl --out-dir data/processed \
  --seq-len 128 --val-frac 0.05 --vocab-size 6000 --assistant-loss-only
```

- 空行和空对话跳过；格式错误会报告文件和行号。`--max-docs` 限制有效对话数量。
- 按完整对话划分；相同格式化内容的重复样本归为一组，避免重复课程样本跨集合泄漏。`--val-frac` 是独立对话组的比例，重复次数不同会使 token 比例偏离该值。
- 只用训练集建立词表。验证集的未见字符会编码为 `<unk>`；这不能视为模型已经学会这些字符。
- 每个集合必须至少有 `seq_len + 1` 个 token，并能产生有监督信号的窗口。数据太少时明确报错，添加数据或降低 `--seq-len`。
- `--assistant-loss-only` 仅监督回答及回答末尾的 EOS；用户文本和角色前缀作为上下文。每条对话必须含非空助手回答。
- `--min-mask-frac` 限制训练窗口内的监督比例。无合格窗口时会报错，不会把无监督窗口当作有效训练。验证仅排除零监督窗口，不使用训练比例阈值。
- 可用 `--tokenizer-type subword` 试验贪心子词编码；默认字符级更简单。微调时用 `--tokenizer-in 原数据目录/tokenizer.json` 保持 token ID 一致。

输出 `train.bin`、`val.bin`、对应 `*_mask.bin`、`tokenizer.json` 和 `meta.json`。token 为 uint16 或 uint32，mask 为 uint8。新格式总是加载 mask，即使全 token 训练也不监督跨文档的 BOS。

训练窗口仍会跨越同一集合内的文档，attention 不做文档隔离；这属于简单连续流训练，不是按文档独立的 packing。重复内容分组仅防止完全相同的格式化对话泄漏，不检测语义相近问题或等价算式。

## 保存、续训与微调

`best.npz` 保存最低采样验证损失对应的状态；`latest.npz` 按 `--save-every` 保存，并在正常结束或 `--stop-after` 停止时保存。意外中断时使用最近一次已完成的 checkpoint。

新版 checkpoint 包含：

- 权重、模型结构、tokenizer 指纹；加载时检查形状、有限数值及词表身份。
- AdamW 一阶/二阶矩、优化器步数、训练步数和最佳验证损失。
- batch 随机数状态、学习率日程参数和数据指纹。

写入采用临时文件加原子替换，避免中途写坏已有 checkpoint。保存不使用 pickle。

例如先运行到第 50 步，仍按总共 100 步安排学习率：

```bash
python train.py --data-dir data/tiny --checkpoint-dir data/resume_demo \
  --steps 100 --stop-after 50 --batch-size 4 \
  --d-model 32 --n-layers 2 --n-heads 4 \
  --lr 0.003 --min-lr 0.0003 --warmup-steps 10 \
  --eval-every 50 --eval-iters 5 --save-every 50

python train.py --data-dir data/tiny --checkpoint-dir data/resume_demo \
  --resume data/resume_demo/latest.npz --steps 100 --batch-size 4 \
  --lr 0.003 --min-lr 0.0003 --warmup-steps 10 \
  --eval-every 50 --eval-iters 5 --save-every 50
```

`--steps` 是整个日程的总步数，不是额外训练步数。续训要求相同数据、设备、训练日程和验证配置；不匹配会拒绝恢复。验证/预览使用独立 RNG，不影响训练采样。CPU 的连续运行与恢复运行已做逐数组一致性测试；不同硬件或 NumPy/CuPy 版本不保证逐位一致。

改变数据或日程应使用 `--init-from` 微调：加载模型结构及权重，从新的 AdamW 状态开始。

```bash
python train.py --data-dir data/tiny --checkpoint-dir data/finetune \
  --init-from data/tiny_checkpoints/best.npz --steps 100 \
  --batch-size 4 --lr 0.0001 --min-lr 0.00001
```

旧 checkpoint 若带结构元数据仍可采样或 `--init-from`；没有 tokenizer 指纹会警告，必须使用当时的原始 tokenizer。旧权重无法恢复不存在的优化器状态。只保存权重且没有结构元数据的文件，可通过 Python `GPT.load(..., vocab_size=..., seq_len=..., d_model=..., n_layers=..., n_heads=...)` 显式加载。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试包括全部参数的中心差分梯度、多头因果性、稳定交叉熵、loss mask、AdamW 参考计算、最短 batch、重复数据隔离、tokenizer 校验、小模型过拟合、命令行端到端及精确恢复。临时数据由测试自动清理；GitHub Actions 在 Python 3.11/3.12 与 NumPy 1.x/2.x 上运行。

本次结果与限制见 [验证报告](docs/VALIDATION.md)。旧 README 中的训练路线和输出保留在 [历史训练记录](docs/HISTORICAL_TRAINING.md)，其中的权重、指标和 GPU 吞吐量本次未复现。

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `gpt_from_scratch/model.py` | Pre-LN Transformer、GELU、因果注意力、共享 embedding、手写梯度 |
| `gpt_from_scratch/optim.py` | AdamW 及优化器状态 |
| `gpt_from_scratch/data.py` | 对话读取、分组划分、有监督窗口采样 |
| `gpt_from_scratch/tokenizer.py` | 字符/子词编码、对话格式、词表指纹 |
| `gpt_from_scratch/checkpoint.py` | 完整训练状态保存与恢复 |
| `gpt_from_scratch/sampling.py` | 共享生成逻辑 |
| `prepare_data.py`, `train.py`, `sample.py` | 命令行入口 |
| `build_curriculum.py`, `build_math_drill.py`, `filter_dataset.py` | 可选课程/算术数据与启发式过滤 |

过滤脚本的短语规则只是历史实验启发式，不代表高质量数据标准；例如直接排除“我无法”可能删除合理回答。应结合实际任务审核数据。当前没有 KV cache、混合精度、梯度累积或分布式训练，数据和有效窗口索引驻留内存，不适合直接扩展到大规模训练。

## License

[MIT](LICENSE)，Copyright (c) 2026 Lao Chou。更新记录见 [VERSIONS.md](VERSIONS.md)。
