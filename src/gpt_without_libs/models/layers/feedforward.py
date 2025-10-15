"""
前馈神经网络模块
"""

import numpy as np
from typing import Tuple


class FeedForward:
    """前馈神经网络"""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        初始化前馈网络

        Args:
            embed_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 权重和偏置
        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

        # 用于反向传播的中间变量
        self.cache = {}

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """
        GELU激活函数

        Args:
            x: 输入

        Returns:
            GELU激活后的输出
        """
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        GELU导数

        Args:
            x: 输入

        Returns:
            GELU导数
        """
        # GELU的近似导数
        tanh_out = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        sec_sq = 1.0 - tanh_out**2
        return 0.5 * (1.0 + tanh_out + x * sec_sq * np.sqrt(2.0 / np.pi) * (1.0 + 0.134145 * x**2))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入 [batch_size, seq_len, embed_dim]

        Returns:
            输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"嵌入维度不匹配: {embed_dim} vs {self.embed_dim}"

        # 第一个线性层
        linear1_out = np.matmul(x, self.W1) + self.b1  # [batch_size, seq_len, hidden_dim]

        # GELU激活
        gelu_out = self._gelu(linear1_out)  # [batch_size, seq_len, hidden_dim]

        # Dropout
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, gelu_out.shape) / (1 - self.dropout)
            gelu_out *= dropout_mask

        # 第二个线性层
        output = np.matmul(gelu_out, self.W2) + self.b2  # [batch_size, seq_len, embed_dim]

        # 保存中间变量用于反向传播
        self.cache = {
            'x': x,
            'linear1_out': linear1_out,
            'gelu_out': gelu_out
        }

        return output

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

        x = self.cache['x']
        linear1_out = self.cache['linear1_out']
        gelu_out = self.cache['gelu_out']

        batch_size, seq_len, embed_dim = grad_output.shape

        # 对W2和b2的梯度
        grad_W2 = np.matmul(gelu_out.transpose(0, 2, 1), grad_output).sum(axis=0)
        grad_b2 = np.sum(grad_output, axis=(0, 1))

        # 对gelu_out的梯度
        grad_gelu_out = np.matmul(grad_output, self.W2.T)

        # GELU的梯度
        grad_linear1_out = grad_gelu_out * self._gelu_derivative(linear1_out)

        # 对W1和b1的梯度
        grad_W1 = np.matmul(x.transpose(0, 2, 1), grad_linear1_out).sum(axis=0)
        grad_b1 = np.sum(grad_linear1_out, axis=(0, 1))

        # 对输入x的梯度
        grad_x = np.matmul(grad_linear1_out, self.W1.T)

        return grad_x, grad_W1, grad_b1, grad_W2, grad_b2

    def update(self, grad_W1: np.ndarray, grad_b1: np.ndarray,
               grad_W2: np.ndarray, grad_b2: np.ndarray, learning_rate: float):
        """更新权重"""
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2


def test_feedforward():
    """测试前馈神经网络"""
    print("测试前馈神经网络...")
    embed_dim = 64
    hidden_dim = 256
    batch_size = 2
    seq_len = 10

    # 创建前馈网络
    ff = FeedForward(embed_dim, hidden_dim)

    # 创建测试输入
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = ff.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_x, grad_W1, grad_b1, grad_W2, grad_b2 = ff.backward(grad_output)

    assert grad_x.shape == (batch_size, seq_len, embed_dim), f"输入梯度形状错误: {grad_x.shape}"
    assert grad_W1.shape == (embed_dim, hidden_dim), f"W1梯度形状错误: {grad_W1.shape}"
    assert grad_b1.shape == (hidden_dim,), f"b1梯度形状错误: {grad_b1.shape}"
    assert grad_W2.shape == (hidden_dim, embed_dim), f"W2梯度形状错误: {grad_W2.shape}"
    assert grad_b2.shape == (embed_dim,), f"b2梯度形状错误: {grad_b2.shape}"

    # 更新权重
    ff.update(grad_W1, grad_b1, grad_W2, grad_b2, 0.001)

    print("前馈神经网络测试通过！")


if __name__ == "__main__":
    test_feedforward()