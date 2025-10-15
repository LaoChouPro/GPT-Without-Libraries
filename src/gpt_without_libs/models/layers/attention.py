"""
多头注意力机制模块
"""

import numpy as np
import math
from typing import Tuple, Optional


class MultiHeadAttention:
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        assert embed_dim % num_heads == 0, "嵌入维度必须能被注意力头数整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 权重矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.02

        # 用于反向传播的中间变量
        self.cache = {}

    def _split_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """
        将输入分割成多个头

        Args:
            x: 输入 [batch_size, seq_len, embed_dim]
            batch_size: 批次大小
            seq_len: 序列长度

        Returns:
            分割后的输入 [batch_size, num_heads, seq_len, head_dim]
        """
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """
        将多个头合并

        Args:
            x: 多头输入 [batch_size, num_heads, seq_len, head_dim]
            batch_size: 批次大小
            seq_len: 序列长度

        Returns:
            合并后的输出 [batch_size, seq_len, embed_dim]
        """
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] (可选)

        Returns:
            注意力输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"嵌入维度不匹配: {embed_dim} vs {self.embed_dim}"

        # 计算Q, K, V
        Q = np.matmul(x, self.W_q)  # [batch_size, seq_len, embed_dim]
        K = np.matmul(x, self.W_k)  # [batch_size, seq_len, embed_dim]
        V = np.matmul(x, self.W_v)  # [batch_size, seq_len, embed_dim]

        # 分割成多个头
        Q = self._split_heads(Q, batch_size, seq_len)  # [batch_size, num_heads, seq_len, head_dim]
        K = self._split_heads(K, batch_size, seq_len)  # [batch_size, num_heads, seq_len, head_dim]
        V = self._split_heads(V, batch_size, seq_len)  # [batch_size, num_heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]

        # 应用掩码
        if mask is not None:
            if mask.ndim == 2:  # [seq_len, seq_len]
                mask = mask.reshape(1, 1, seq_len, seq_len)  # [1, 1, seq_len, seq_len]
                mask = np.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))  # [batch_size, 1, seq_len, seq_len]
            elif mask.ndim == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.reshape(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = np.where(mask == 0, -1e9, attention_scores)

        # 计算注意力权重
        attention_weights = self._softmax(attention_scores, axis=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 应用dropout
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape) / (1 - self.dropout)
            attention_weights *= dropout_mask

        # 计算注意力输出
        attention_output = np.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并多头
        attention_output = self._combine_heads(attention_output, batch_size, seq_len)  # [batch_size, seq_len, embed_dim]

        # 最终线性变换
        output = np.matmul(attention_output, self.W_o)  # [batch_size, seq_len, embed_dim]

        # 保存中间变量用于反向传播
        self.cache = {
            'x': x,
            'Q': Q,
            'K': K,
            'V': V,
            'attention_weights': attention_weights,
            'attention_output': attention_output
        }

        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """稳定的softmax实现"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播

        Args:
            grad_output: 输出梯度 [batch_size, seq_len, embed_dim]

        Returns:
            输入梯度和权重梯度
        """
        if not self.cache:
            raise ValueError("需要先调用forward")

        batch_size, seq_len, embed_dim = grad_output.shape
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attention_weights = self.cache['attention_weights']
        attention_output = self.cache['attention_output']

        # 对W_o的梯度
        grad_W_o = np.matmul(x.transpose(0, 2, 1), grad_output).sum(axis=0)
        grad_attention_output = np.matmul(grad_output, self.W_o.T)

        # 分割梯度
        grad_attention_output = self._split_heads(grad_attention_output, batch_size, seq_len)

        # 对V的梯度
        grad_V = np.matmul(attention_weights.transpose(0, 1, 3, 2), grad_attention_output)

        # 对注意力权重的梯度
        grad_attention_weights = np.matmul(grad_attention_output, V.transpose(0, 1, 3, 2))

        # 对注意力分数的梯度 (通过softmax)
        grad_attention_scores = attention_weights * (grad_attention_weights -
                                                    np.sum(attention_weights * grad_attention_weights, axis=-1, keepdims=True))

        # 对Q和K的梯度
        grad_Q = np.matmul(grad_attention_scores, K)
        grad_K = np.matmul(grad_attention_scores.transpose(0, 1, 3, 2), Q)

        # 合并多头梯度
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # 对W_q, W_k, W_v的梯度
        grad_W_q = np.matmul(x.transpose(0, 2, 1), grad_Q).sum(axis=0)
        grad_W_k = np.matmul(x.transpose(0, 2, 1), grad_K).sum(axis=0)
        grad_W_v = np.matmul(x.transpose(0, 2, 1), grad_V).sum(axis=0)

        # 对输入x的梯度
        grad_x = (np.matmul(grad_Q, self.W_q.T) +
                 np.matmul(grad_K, self.W_k.T) +
                 np.matmul(grad_V, self.W_v.T))

        return grad_x, grad_W_q, grad_W_k, grad_W_v, grad_W_o

    def update(self, grad_W_q: np.ndarray, grad_W_k: np.ndarray, grad_W_v: np.ndarray,
               grad_W_o: np.ndarray, learning_rate: float):
        """更新权重"""
        self.W_q -= learning_rate * grad_W_q
        self.W_k -= learning_rate * grad_W_k
        self.W_v -= learning_rate * grad_W_v
        self.W_o -= learning_rate * grad_W_o


def test_multihead_attention():
    """测试多头注意力机制"""
    print("测试多头注意力机制...")
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10

    # 创建多头注意力层
    attention = MultiHeadAttention(embed_dim, num_heads)

    # 创建测试输入
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = attention.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_x, grad_W_q, grad_W_k, grad_W_v, grad_W_o = attention.backward(grad_output)

    assert grad_x.shape == (batch_size, seq_len, embed_dim), f"输入梯度形状错误: {grad_x.shape}"
    assert grad_W_q.shape == (embed_dim, embed_dim), f"W_q梯度形状错误: {grad_W_q.shape}"
    assert grad_W_k.shape == (embed_dim, embed_dim), f"W_k梯度形状错误: {grad_W_k.shape}"
    assert grad_W_v.shape == (embed_dim, embed_dim), f"W_v梯度形状错误: {grad_W_v.shape}"
    assert grad_W_o.shape == (embed_dim, embed_dim), f"W_o梯度形状错误: {grad_W_o.shape}"

    # 更新权重
    attention.update(grad_W_q, grad_W_k, grad_W_v, grad_W_o, 0.001)

    print("多头注意力机制测试通过！")


if __name__ == "__main__":
    test_multihead_attention()