"""
从零构建的微型Transformer GPT模型
仅使用numpy等基础运算库，不依赖任何深度学习框架
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
import re


class Embedding:
    """词嵌入层"""

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        初始化词嵌入层

        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # 使用Xavier初始化
        self.weights = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / (vocab_size + embed_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入token索引 [batch_size, seq_len]

        Returns:
            嵌入向量 [batch_size, seq_len, embed_dim]
        """
        return self.weights[x]

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 输出梯度 [batch_size, seq_len, embed_dim]
            x: 输入token索引 [batch_size, seq_len]

        Returns:
            权重梯度 [vocab_size, embed_dim]
        """
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, x, grad_output)
        return grad_weights

    def update(self, grad_weights: np.ndarray, learning_rate: float):
        """更新权重"""
        self.weights -= learning_rate * grad_weights


def test_embedding():
    """测试词嵌入层"""
    print("测试词嵌入层...")
    vocab_size = 1000
    embed_dim = 128
    batch_size = 2
    seq_len = 10

    # 创建嵌入层
    embedding = Embedding(vocab_size, embed_dim)

    # 创建测试输入
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    output = embedding.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_weights = embedding.backward(grad_output, x)
    assert grad_weights.shape == (vocab_size, embed_dim), f"梯度形状错误: {grad_weights.shape}"

    # 更新权重
    embedding.update(grad_weights, 0.001)

    print("词嵌入层测试通过！")


if __name__ == "__main__":
    test_embedding()