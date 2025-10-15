"""
完整的Transformer块模块
包含自注意力、前馈网络、残差连接和层归一化
"""

import numpy as np
from typing import Tuple, Optional

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock:
    """Transformer解码器块"""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        """
        初始化Transformer块

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 注意力层
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 前馈网络
        self.feedforward = FeedForward(embed_dim, hidden_dim, dropout)

        # 层归一化
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        # 用于反向传播的中间变量
        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] (可选)

        Returns:
            输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"嵌入维度不匹配: {embed_dim} vs {self.embed_dim}"

        # 保存原始输入用于残差连接
        x_orig = x

        # 第一个子层：自注意力 + 残差连接 + 层归一化
        # LayerNorm before attention (Pre-LN)
        x_norm1 = self.norm1.forward(x)
        attn_output = self.attention.forward(x_norm1, mask)

        # 残差连接
        x = x_orig + attn_output

        # 保存中间变量
        x_after_attn = x

        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        # LayerNorm before feedforward (Pre-LN)
        x_norm2 = self.norm2.forward(x)
        ff_output = self.feedforward.forward(x_norm2)

        # 残差连接
        x = x_after_attn + ff_output

        # 保存中间变量用于反向传播
        self.cache = {
            'x_orig': x_orig,
            'x_norm1': x_norm1,
            'attn_output': attn_output,
            'x_after_attn': x_after_attn,
            'x_norm2': x_norm2,
            'ff_output': ff_output,
            'output': x
        }

        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 输出梯度 [batch_size, seq_len, embed_dim]

        Returns:
            输入梯度 [batch_size, seq_len, embed_dim]
        """
        if not self.cache:
            raise ValueError("需要先调用forward")

        # 获取缓存变量
        x_orig = self.cache['x_orig']
        x_norm1 = self.cache['x_norm1']
        attn_output = self.cache['attn_output']
        x_after_attn = self.cache['x_after_attn']
        x_norm2 = self.cache['x_norm2']
        ff_output = self.cache['ff_output']
        output = self.cache['output']

        # 第二个子层的反向传播
        # 残差连接的梯度
        grad_ff_output = grad_output
        grad_x_after_attn = grad_output

        # 前馈网络的梯度
        grad_x_norm2, grad_W1_ff, grad_b1_ff, grad_W2_ff, grad_b2_ff = self.feedforward.backward(grad_ff_output)

        # 层归一化2的梯度
        grad_x_norm2 += grad_x_after_attn  # 残差连接
        grad_x_after_attn_2, grad_gamma2, grad_beta2 = self.norm2.backward(grad_x_norm2)

        # 第一个子层的反向传播
        # 残差连接的梯度
        grad_attn_output = grad_x_after_attn_2
        grad_x_orig_2 = grad_x_after_attn_2

        # 注意力层的梯度
        grad_x_norm1, grad_W_q, grad_W_k, grad_W_v, grad_W_o = self.attention.backward(grad_attn_output)

        # 层归一化1的梯度
        grad_x_norm1 += grad_x_orig_2  # 残差连接
        grad_x, grad_gamma1, grad_beta1 = self.norm1.backward(grad_x_norm1)

        # 保存各层的梯度用于参数更新
        self.grad_cache = {
            'grad_W_q': grad_W_q,
            'grad_W_k': grad_W_k,
            'grad_W_v': grad_W_v,
            'grad_W_o': grad_W_o,
            'grad_W1_ff': grad_W1_ff,
            'grad_b1_ff': grad_b1_ff,
            'grad_W2_ff': grad_W2_ff,
            'grad_b2_ff': grad_b2_ff,
            'grad_gamma1': grad_gamma1,
            'grad_beta1': grad_beta1,
            'grad_gamma2': grad_gamma2,
            'grad_beta2': grad_beta2
        }

        return grad_x

    def update(self, learning_rate: float):
        """更新所有参数"""
        if not hasattr(self, 'grad_cache'):
            raise ValueError("需要先调用backward")

        # 更新注意力层参数
        self.attention.update(
            self.grad_cache['grad_W_q'],
            self.grad_cache['grad_W_k'],
            self.grad_cache['grad_W_v'],
            self.grad_cache['grad_W_o'],
            learning_rate
        )

        # 更新前馈网络参数
        self.feedforward.update(
            self.grad_cache['grad_W1_ff'],
            self.grad_cache['grad_b1_ff'],
            self.grad_cache['grad_W2_ff'],
            self.grad_cache['grad_b2_ff'],
            learning_rate
        )

        # 更新层归一化参数
        self.norm1.update(self.grad_cache['grad_gamma1'], self.grad_cache['grad_beta1'], learning_rate)
        self.norm2.update(self.grad_cache['grad_gamma2'], self.grad_cache['grad_beta2'], learning_rate)


def test_transformer_block():
    """测试Transformer块"""
    print("测试Transformer块...")
    embed_dim = 64
    num_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 10

    # 创建Transformer块
    transformer_block = TransformerBlock(embed_dim, num_heads, hidden_dim)

    # 创建测试输入
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = transformer_block.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 验证残差连接：输出应该与输入有一定的相似性
    assert not np.allclose(output, x), "残差连接应该改变输出"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_x = transformer_block.backward(grad_output)
    assert grad_x.shape == (batch_size, seq_len, embed_dim), f"输入梯度形状错误: {grad_x.shape}"

    # 更新参数
    transformer_block.update(0.001)

    print("Transformer块测试通过！")


if __name__ == "__main__":
    test_transformer_block()