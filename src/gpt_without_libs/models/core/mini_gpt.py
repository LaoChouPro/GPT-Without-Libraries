"""
微型GPT模型主类
完整的Transformer语言模型
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict

from ..layers.embedding import Embedding
from ..layers.positional_encoding import PositionalEncoding
from ..layers.transformer_block import TransformerBlock


class MiniGPT:
    """微型GPT模型"""

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 6, hidden_dim: int = 1024, max_seq_len: int = 512,
                 dropout: float = 0.1):
        """
        初始化微型GPT模型

        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            hidden_dim: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: Dropout率
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # 词嵌入层
        self.embedding = Embedding(vocab_size, embed_dim)

        # 位置编码层
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # Transformer层
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ]

        # 输出投影层（词汇表大小映射）
        self.output_projection = np.random.randn(embed_dim, vocab_size) * 0.02

        # 用于反向传播的中间变量
        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入token索引 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] (可选)

        Returns:
            输出logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # 词嵌入
        embed_output = self.embedding.forward(x)  # [batch_size, seq_len, embed_dim]

        # 位置编码
        pos_output = self.pos_encoding.forward(embed_output)  # [batch_size, seq_len, embed_dim]

        # 通过所有Transformer层
        hidden = pos_output
        for i, transformer_block in enumerate(self.transformer_blocks):
            hidden = transformer_block.forward(hidden, mask)  # [batch_size, seq_len, embed_dim]

        # 输出投影
        logits = np.matmul(hidden, self.output_projection)  # [batch_size, seq_len, vocab_size]

        # 保存中间变量用于反向传播
        self.cache = {
            'x': x,
            'embed_output': embed_output,
            'pos_output': pos_output,
            'hidden': hidden,
            'logits': logits
        }

        return logits

    def generate_mask(self, seq_len: int) -> np.ndarray:
        """
        生成因果注意力掩码

        Args:
            seq_len: 序列长度

        Returns:
            注意力掩码 [seq_len, seq_len]
        """
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return mask

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_logits: 输出梯度 [batch_size, seq_len, vocab_size]

        Returns:
            输入梯度 [batch_size, seq_len]
        """
        if not self.cache:
            raise ValueError("需要先调用forward")

        # 获取缓存变量
        x = self.cache['x']
        embed_output = self.cache['embed_output']
        pos_output = self.cache['pos_output']
        hidden = self.cache['hidden']
        logits = self.cache['logits']

        batch_size, seq_len, vocab_size = grad_logits.shape

        # 输出投影的梯度
        grad_output_projection = np.matmul(hidden.transpose(0, 2, 1), grad_logits).sum(axis=0)
        grad_hidden = np.matmul(grad_logits, self.output_projection.T)

        # Transformer层的反向传播
        grad_pos_output = grad_hidden
        for transformer_block in reversed(self.transformer_blocks):
            grad_pos_output = transformer_block.backward(grad_pos_output)

        # 位置编码的梯度
        grad_embed_output = self.pos_encoding.backward(grad_pos_output)

        # 词嵌入的梯度
        grad_embedding_weights = self.embedding.backward(grad_embed_output, x)

        # 保存梯度用于参数更新
        self.grad_cache = {
            'grad_output_projection': grad_output_projection,
            'grad_embedding_weights': grad_embedding_weights
        }

        # 输入x是离散的token索引，不需要梯度
        return np.zeros_like(x)

    def update(self, learning_rate: float):
        """更新所有参数"""
        if not hasattr(self, 'grad_cache'):
            raise ValueError("需要先调用backward")

        # 更新输出投影层
        self.output_projection -= learning_rate * self.grad_cache['grad_output_projection']

        # 更新词嵌入层
        self.embedding.update(self.grad_cache['grad_embedding_weights'], learning_rate)

        # 更新所有Transformer层
        for transformer_block in self.transformer_blocks:
            transformer_block.update(learning_rate)

    def save(self, filepath: str):
        """
        保存模型参数

        Args:
            filepath: 保存路径
        """
        model_state = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'embedding_weights': self.embedding.weights,
            'output_projection': self.output_projection
        }

        # 保存Transformer层的参数
        for i, transformer_block in enumerate(self.transformer_blocks):
            model_state[f'transformer_{i}_attention_W_q'] = transformer_block.attention.W_q
            model_state[f'transformer_{i}_attention_W_k'] = transformer_block.attention.W_k
            model_state[f'transformer_{i}_attention_W_v'] = transformer_block.attention.W_v
            model_state[f'transformer_{i}_attention_W_o'] = transformer_block.attention.W_o
            model_state[f'transformer_{i}_ff_W1'] = transformer_block.feedforward.W1
            model_state[f'transformer_{i}_ff_b1'] = transformer_block.feedforward.b1
            model_state[f'transformer_{i}_ff_W2'] = transformer_block.feedforward.W2
            model_state[f'transformer_{i}_ff_b2'] = transformer_block.feedforward.b2
            model_state[f'transformer_{i}_norm1_gamma'] = transformer_block.norm1.gamma
            model_state[f'transformer_{i}_norm1_beta'] = transformer_block.norm1.beta
            model_state[f'transformer_{i}_norm2_gamma'] = transformer_block.norm2.gamma
            model_state[f'transformer_{i}_norm2_beta'] = transformer_block.norm2.beta

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """
        加载模型参数

        Args:
            filepath: 加载路径
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # 检查模型参数是否匹配
        assert model_state['vocab_size'] == self.vocab_size
        assert model_state['embed_dim'] == self.embed_dim
        assert model_state['num_heads'] == self.num_heads
        assert model_state['num_layers'] == self.num_layers
        assert model_state['hidden_dim'] == self.hidden_dim

        # 加载词嵌入层参数
        self.embedding.weights = model_state['embedding_weights']

        # 加载输出投影层参数
        self.output_projection = model_state['output_projection']

        # 加载Transformer层参数
        for i, transformer_block in enumerate(self.transformer_blocks):
            transformer_block.attention.W_q = model_state[f'transformer_{i}_attention_W_q']
            transformer_block.attention.W_k = model_state[f'transformer_{i}_attention_W_k']
            transformer_block.attention.W_v = model_state[f'transformer_{i}_attention_W_v']
            transformer_block.attention.W_o = model_state[f'transformer_{i}_attention_W_o']
            transformer_block.feedforward.W1 = model_state[f'transformer_{i}_ff_W1']
            transformer_block.feedforward.b1 = model_state[f'transformer_{i}_ff_b1']
            transformer_block.feedforward.W2 = model_state[f'transformer_{i}_ff_W2']
            transformer_block.feedforward.b2 = model_state[f'transformer_{i}_ff_b2']
            transformer_block.norm1.gamma = model_state[f'transformer_{i}_norm1_gamma']
            transformer_block.norm1.beta = model_state[f'transformer_{i}_norm1_beta']
            transformer_block.norm2.gamma = model_state[f'transformer_{i}_norm2_gamma']
            transformer_block.norm2.beta = model_state[f'transformer_{i}_norm2_beta']

        print(f"模型已从 {filepath} 加载")

    def count_parameters(self) -> int:
        """
        计算模型参数总数

        Returns:
            参数总数
        """
        total_params = 0

        # 词嵌入层参数
        total_params += self.embedding.weights.size

        # 位置编码参数（固定，不算入可训练参数）

        # 输出投影层参数
        total_params += self.output_projection.size

        # Transformer层参数
        for transformer_block in self.transformer_blocks:
            # 注意力层参数
            total_params += transformer_block.attention.W_q.size
            total_params += transformer_block.attention.W_k.size
            total_params += transformer_block.attention.W_v.size
            total_params += transformer_block.attention.W_o.size

            # 前馈网络参数
            total_params += transformer_block.feedforward.W1.size
            total_params += transformer_block.feedforward.b1.size
            total_params += transformer_block.feedforward.W2.size
            total_params += transformer_block.feedforward.b2.size

            # 层归一化参数
            total_params += transformer_block.norm1.gamma.size
            total_params += transformer_block.norm1.beta.size
            total_params += transformer_block.norm2.gamma.size
            total_params += transformer_block.norm2.beta.size

        return total_params


def test_mini_gpt():
    """测试微型GPT模型"""
    print("测试微型GPT模型...")
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    hidden_dim = 256
    batch_size = 2
    seq_len = 10

    # 创建模型
    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)

    # 创建测试输入
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model.forward(x)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"输出形状错误: {logits.shape}"

    # 反向传播
    grad_logits = np.random.randn(batch_size, seq_len, vocab_size)
    grad_x = model.backward(grad_logits)
    assert grad_x.shape == (batch_size, seq_len), f"输入梯度形状错误: {grad_x.shape}"

    # 更新参数
    model.update(0.001)

    # 计算参数数量
    num_params = model.count_parameters()
    print(f"模型参数总数: {num_params:,}")

    # 测试保存和加载
    model.save("test_model.pkl")
    new_model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)
    new_model.load("test_model.pkl")

    # 清理测试文件
    if os.path.exists("test_model.pkl"):
        os.remove("test_model.pkl")

    print("微型GPT模型测试通过！")


if __name__ == "__main__":
    test_mini_gpt()