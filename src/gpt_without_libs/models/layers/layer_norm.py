"""
层归一化模块
"""

import numpy as np
from typing import Tuple


class LayerNorm:
    """层归一化"""

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        """
        初始化层归一化

        Args:
            embed_dim: 嵌入维度
            eps: 数值稳定性参数
        """
        self.embed_dim = embed_dim
        self.eps = eps

        # 可学习参数
        self.gamma = np.ones(embed_dim)  # 缩放参数
        self.beta = np.zeros(embed_dim)  # 偏移参数

        # 用于反向传播的中间变量
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入 [batch_size, seq_len, embed_dim]

        Returns:
            归一化后的输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"嵌入维度不匹配: {embed_dim} vs {self.embed_dim}"

        # 计算均值和方差
        mean = np.mean(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        var = np.var(x, axis=-1, keepdims=True)    # [batch_size, seq_len, 1]

        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和偏移
        output = self.gamma * x_normalized + self.beta

        # 保存中间变量用于反向传播
        self.cache = {
            'x': x,
            'x_normalized': x_normalized,
            'mean': mean,
            'var': var
        }

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播

        Args:
            grad_output: 输出梯度 [batch_size, seq_len, embed_dim]

        Returns:
            输入梯度和参数梯度
        """
        if not self.cache:
            raise ValueError("需要先调用forward")

        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        mean = self.cache['mean']
        var = self.cache['var']

        batch_size, seq_len, embed_dim = x.shape

        # 对gamma和beta的梯度
        grad_gamma = np.sum(grad_output * x_normalized, axis=(0, 1))
        grad_beta = np.sum(grad_output, axis=(0, 1))

        # 对x_normalized的梯度
        grad_x_normalized = grad_output * self.gamma

        # 对x的梯度 (层归一化的反向传播)
        N = embed_dim
        x_centered = x - mean
        std_inv = 1.0 / np.sqrt(var + self.eps)

        grad_x = (1.0 / N) * std_inv * (
            N * grad_x_normalized -
            np.sum(grad_x_normalized, axis=-1, keepdims=True) -
            x_centered * std_inv**2 * np.sum(grad_x_normalized * x_centered, axis=-1, keepdims=True)
        )

        return grad_x, grad_gamma, grad_beta

    def update(self, grad_gamma: np.ndarray, grad_beta: np.ndarray, learning_rate: float):
        """更新参数"""
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta


def test_layer_norm():
    """测试层归一化"""
    print("测试层归一化...")
    embed_dim = 64
    batch_size = 2
    seq_len = 10

    # 创建层归一化
    layer_norm = LayerNorm(embed_dim)

    # 创建测试输入
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = layer_norm.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 验证归一化效果：均值接近0，方差接近1
    mean_check = np.mean(output, axis=-1)
    var_check = np.var(output, axis=-1)
    assert np.allclose(mean_check, 0, atol=1e-5), "均值不为0"
    assert np.allclose(var_check, 1, atol=1e-5), "方差不为1"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_x, grad_gamma, grad_beta = layer_norm.backward(grad_output)

    assert grad_x.shape == (batch_size, seq_len, embed_dim), f"输入梯度形状错误: {grad_x.shape}"
    assert grad_gamma.shape == (embed_dim,), f"gamma梯度形状错误: {grad_gamma.shape}"
    assert grad_beta.shape == (embed_dim,), f"beta梯度形状错误: {grad_beta.shape}"

    # 更新参数
    layer_norm.update(grad_gamma, grad_beta, 0.001)

    print("层归一化测试通过！")


if __name__ == "__main__":
    test_layer_norm()