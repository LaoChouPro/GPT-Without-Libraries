# GPT Without Libraries

一个从零实现的迷你 GPT 训练项目。模型、反向传播、优化器、数据管线和采样逻辑都直接用 Python + NumPy/CuPy 编写，不依赖 PyTorch、TensorFlow、JAX、Keras、Transformers 等深度学习框架。

这个项目的目标不是做一个大模型包装器，而是把一个 decoder-only Transformer 从参数初始化、前向传播、手写 backward、AdamW 更新、数据编码、loss mask、checkpoint、采样全部打通，并在本地中文对话数据上训练出一个可观察的小语言模型。

## 当前状态

项目已经完成了从零训练到课程微调的基本闭环：

- 先在 `dataset.jsonl` 的 30 万条原始对话样本上训练中文对话底座。
- 发现原始数据中存在大量拒答模板、实时信息模板、身份污染和翻译噪声后，加入课程数据和过滤数据进行纠偏。
- 当前主模型可以稳定完成一部分短问答，例如四大发明、北京春季出游、机器学习解释、学习计划等。
- 冷门知识、抽象类比、文言文、外语混合输入和复杂推理仍然不可靠。

推荐 checkpoint：

```text
data/checkpoints_stage2/best.npz
```

稍大参数实验分支：

```text
data/checkpoints_large_stage2/best.npz
```

## 设计原则

本项目尽量保持实现透明，避免把关键训练逻辑藏在框架内部。

- 只依赖 `numpy` 或 `cupy` 做张量计算。
- 模型参数是普通字典里的数组。
- 每个算子显式保存 cache，并在 backward 中手写梯度。
- 训练脚本直接调用 `forward -> backward -> AdamW.step`。
- checkpoint 使用 `.npz` 保存，可以直接用 NumPy 打开检查权重。
- 数据格式保持简单，方便替换为自己的 JSONL 对话数据。

## 模型结构

核心实现位于：

```text
gpt_from_scratch/model.py
```

整体结构是标准 decoder-only Transformer：

```text
token ids
  -> token embedding
  -> position embedding
  -> Transformer block x N
       -> LayerNorm
       -> causal self-attention
       -> residual connection
       -> LayerNorm
       -> MLP: Linear -> GELU -> Linear
       -> residual connection
  -> final LayerNorm
  -> tied output projection
  -> logits
```

当前主模型配置：

| item | value |
| --- | ---: |
| vocab size | 8000 |
| context length | 192 |
| d_model | 192 |
| layers | 4 |
| attention heads | 6 |
| head dim | 32 |
| FFN hidden size | 768 |
| parameters | about 3.35M |

稍大实验模型配置：

| item | value |
| --- | ---: |
| vocab size | 8000 |
| context length | 192 |
| d_model | 256 |
| layers | 6 |
| attention heads | 8 |
| head dim | 32 |
| FFN hidden size | 1024 |
| parameters | about 6.84M |

### Attention

attention 使用 causal mask，保证第 `t` 个 token 只能看到 `0..t` 的历史上下文：

```python
scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
mask = xp.triu(xp.ones((tsz, tsz), dtype=bool), 1)
scores = xp.where(mask[None, None, :, :], -1e9, scores)
att = softmax(scores, axis=-1)
y = att @ v
```

Q、K、V 由一个合并矩阵产生，再按最后一维切分：

```python
qkv = linear(ln1, qkv_w, qkv_b)
q, k, v = xp.split(qkv, 3, axis=-1)
```

### Pre-LN block

每个 block 使用 pre-layernorm：

```text
h = h + attention(layernorm(h))
h = h + mlp(layernorm(h))
```

小模型训练时，pre-LN 比 post-LN 更容易稳定，尤其是在没有框架自动混合精度、没有复杂初始化策略的情况下。

### 权重共享

输出层没有单独的 `lm_head` 参数，而是复用 token embedding：

```python
logits = h @ self.params["tok_emb"].T
```

这样可以减少参数量，也让小模型更快收敛。

## 手写反向传播

项目没有 autograd。下面这些 backward 都是手写的：

- `linear_backward`
- `layernorm_backward`
- `gelu_backward`
- attention backward
- residual 梯度合流
- token embedding 梯度累加
- position embedding 梯度累加
- tied output projection 对 embedding 的梯度

attention backward 中显式计算：

```text
datt = dy @ v.T
dv   = att.T @ dy
ds   = softmax_backward(datt)
dq   = ds @ k
dk   = ds.T @ q
```

embedding 梯度使用 `xp.add.at` 累加重复 token 的贡献：

```python
xp.add.at(grads["tok_emb"], flat_idx, flat_dh)
```

这部分实现牺牲了一些性能，但非常适合检查梯度流向和理解训练过程。

## 优化器

优化器位于：

```text
gpt_from_scratch/optim.py
```

实现了一个简洁版 AdamW：

- `beta1 = 0.9`
- `beta2 = 0.95`
- bias correction
- decoupled weight decay
- embedding 和一维参数跳过 weight decay

训练脚本中还包含：

- warmup
- cosine learning rate decay
- global grad norm clipping
- periodic eval
- best/latest checkpoint 保存

## Tokenizer

Tokenizer 位于：

```text
gpt_from_scratch/tokenizer.py
```

当前主线使用字符级 tokenizer，词表大小 8000。中文小数据场景下，字符级 tokenizer 更稳，尤其是在没有完整 BPE/Unigram 训练器的情况下。

项目中也实现了一个简单 subword tokenizer，但当前实验结果不如字符级稳定。

特殊 token：

```text
<pad>
<unk>
<bos>
<eos>
```

对话会被格式化为：

```text
用户：...
助手：...
```

## Assistant-only loss

训练数据中既有用户输入，也有助手回答。为了让模型更集中学习“回答”，数据准备阶段可以生成 loss mask：

```bash
python3 prepare_data.py \
  --input dataset.jsonl \
  --out-dir data/char_300k \
  --max-docs 300000 \
  --vocab-size 8000 \
  --seq-len 192 \
  --assistant-loss-only
```

开启后：

- 用户部分只作为上下文。
- 助手部分参与 cross entropy。
- `<eos>` 也参与训练，帮助模型学会停止。

训练时还会优先采样助手 token 比例足够高的窗口：

```bash
--min-mask-frac 0.25
```

这能避免 batch 里大部分 token 都是不计 loss 的提示文本。

## 数据管线

原始数据格式为 JSONL：

```json
{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

数据准备脚本会输出：

```text
train.bin
val.bin
train_mask.bin
val_mask.bin
tokenizer.json
meta.json
```

token 文件使用 `uint16` 或 `uint32` 存储，取决于词表大小。

## 训练路线

实际效果比较好的路线不是直接把原始数据喂到底，而是分阶段训练。

### 1. 原始 30 万条预训练

```bash
python3 prepare_data.py \
  --input dataset.jsonl \
  --out-dir data/char_300k \
  --max-docs 300000 \
  --vocab-size 8000 \
  --seq-len 192 \
  --assistant-loss-only

python3 train.py \
  --data-dir data/char_300k \
  --checkpoint-dir data/checkpoints_char300k \
  --steps 12000 \
  --batch-size 16 \
  --seq-len 192 \
  --d-model 192 \
  --n-layers 4 \
  --n-heads 6 \
  --lr 8e-5 \
  --min-lr 8e-6 \
  --warmup-steps 300 \
  --weight-decay 0.01 \
  --eval-every 1000 \
  --eval-iters 30 \
  --save-every 1000 \
  --min-mask-frac 0.25
```

这一阶段学到中文分布、对话格式和基本句子结构，但原始数据噪声会让小模型频繁输出拒答模板或无意义套话。

### 2. 课程数据纠偏

课程数据由 `build_curriculum.py` 构建，包含：

- 高质量短问答样本
- 自动生成的简单算术样本
- 从原始数据中过滤出的较干净样本
- 主题样本加权，例如四大发明、北京春天、机器学习解释等

```bash
python3 build_curriculum.py \
  --input dataset.jsonl \
  --output data/curriculum_stage2.jsonl \
  --max-source 300000 \
  --max-general 10000 \
  --max-topic 3000 \
  --seed-repeat 300
```

再使用原 tokenizer 编码，保证 checkpoint 兼容：

```bash
python3 prepare_data.py \
  --input data/curriculum_stage2.jsonl \
  --out-dir data/curriculum_stage2 \
  --max-docs 1000000 \
  --tokenizer-in data/char_300k/tokenizer.json \
  --seq-len 192 \
  --val-frac 0.02 \
  --assistant-loss-only
```

微调：

```bash
python3 train.py \
  --data-dir data/curriculum_stage2 \
  --checkpoint-dir data/checkpoints_stage2 \
  --init-from data/checkpoints_focused/best.npz \
  --steps 2500 \
  --batch-size 16 \
  --seq-len 192 \
  --d-model 192 \
  --n-layers 4 \
  --n-heads 6 \
  --lr 2e-5 \
  --min-lr 3e-6 \
  --warmup-steps 100 \
  --weight-decay 0.001 \
  --eval-every 500 \
  --eval-iters 40 \
  --save-every 500 \
  --min-mask-frac 0.40
```

### 3. 稍大模型实验

```bash
python3 train.py \
  --data-dir data/curriculum_focused \
  --checkpoint-dir data/checkpoints_large_focused \
  --steps 1800 \
  --batch-size 12 \
  --seq-len 192 \
  --d-model 256 \
  --n-layers 6 \
  --n-heads 8 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --warmup-steps 100 \
  --weight-decay 0.001 \
  --eval-every 300 \
  --eval-iters 40 \
  --save-every 300 \
  --min-mask-frac 0.45
```

稍大模型更容易拟合课程任务，但混入通用样本后仍然需要更干净的数据和更长训练。

## 采样

```bash
python3 sample.py \
  --data-dir data/curriculum_stage2 \
  --checkpoint data/checkpoints_stage2/best.npz \
  --prompt $'用户：北京春季去哪赏花？\n助手：' \
  --max-new-tokens 140 \
  --temperature 0.35 \
  --top-k 12
```

示例输出：

```text
用户：北京春季去哪赏花？
助手：北京春天适合去玉渊潭看樱花、去颐和园和北海公园踏青、去北京植物园赏桃花，也可以去奥林匹克森林公园散步骑行。出门前注意查看风沙和花粉情况。
```

另一个示例：

```text
用户：请用一段话介绍中国古代四大发明。
助手：中国古代四大发明是造纸术、印刷术、火药和指南针。造纸术和印刷术推动了知识传播，火药改变了军事技术，指南针促进了航海和远距离交流。它们对中国和世界文明的发展都有深远影响。
```

失败示例：

```text
用户：23加19等于多少？
助手：20加10等于120。
```

这类失败很重要。它说明当前模型并没有真正学会可靠算术，只是在小范围内记住了一些模式。

## 项目结构

```text
.
├── gpt_from_scratch/
│   ├── model.py          # Transformer, forward, backward, checkpoint
│   ├── optim.py          # AdamW
│   ├── tokenizer.py      # char/subword tokenizer and conversation formatting
│   └── __init__.py
├── prepare_data.py       # JSONL -> token bin + loss mask
├── train.py              # training loop
├── sample.py             # autoregressive sampling
├── filter_dataset.py     # simple dataset filtering
├── build_curriculum.py   # curriculum dataset builder
├── build_math_drill.py   # arithmetic drill dataset builder
└── README.md
```

## 环境

推荐环境：

```text
Python 3.11+
NumPy
CuPy with CUDA support
NVIDIA GPU
```

CPU 也能运行，但训练速度会慢很多。

安装依赖示例：

```bash
python3 -m pip install numpy cupy-cuda12x
```

如果没有 CUDA，可以只安装 NumPy，并在训练/采样时使用：

```bash
--device cpu
```

## 训练性能参考

在 RTX 5070 Ti 上，3.35M 参数模型、`seq_len=192`、`batch_size=16` 的训练速度大约在：

```text
150k - 165k tokens/s
```

6.84M 参数模型、`seq_len=192`、`batch_size=12` 的训练速度大约在：

```text
85k - 95k tokens/s
```

实际速度取决于 CuPy 版本、CUDA 版本、batch size 和 eval 频率。

## 已知限制

这个项目目前仍然是实验性质的小模型训练系统。

- 上下文长度只有 192。
- 字符级 tokenizer 会让长文本更占上下文。
- attention mask 每次 forward 都会重建，还有优化空间。
- 没有 dropout、KV cache、mixed precision、gradient accumulation。
- 没有成熟的 dataset packing 和 checkpoint resume optimizer state。
- 手写 backward 更容易调试，但性能不如成熟框架。
- 小模型很容易被数据污染带偏。
- 当前 checkpoint 对课程内短问答表现较好，对冷门知识和复杂推理不可靠。

## 为什么不直接用大框架

这个项目的重点是可见性。用成熟框架写一个小 GPT 很快，但很多关键细节会被 autograd、module abstraction 和 optimizer wrapper 隐藏起来。

这里保留了训练语言模型最核心的部件：

- tensor shape 怎么流动
- attention 的梯度怎么回传
- tied embedding 的梯度怎么合并
- loss mask 怎么影响 dlogits
- 小数据污染如何改变生成分布
- curriculum fine-tuning 如何纠正小模型行为

适合用来学习、调试和做小规模训练实验。

## License

No license has been selected yet. Add a license before redistributing or using this code in another project.
