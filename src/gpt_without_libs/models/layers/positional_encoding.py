"""
位置编码模块
为输入序列添加位置信息
"""

import numpy as np
import math
from typing import Optional


class PositionalEncoding:
    """位置编码层"""

    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        """
        初始化位置编码

        Args:
            embed_dim: 嵌入维度
            max_seq_len: 最大序列长度
        """
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.encoding = self._create_positional_encoding()

    def _create_positional_encoding(self) -> np.ndarray:
        """
        创建位置编码矩阵

        Returns:
            位置编码矩阵 [max_seq_len, embed_dim]
        """
        # 创建位置编码矩阵
        encoding = np.zeros((self.max_seq_len, self.embed_dim))

        # 位置索引
        position = np.arange(0, self.max_seq_len, dtype=np.float32).reshape(-1, 1)

        # 除数项
        div_term = np.exp(np.arange(0, self.embed_dim, 2, dtype=np.float32) *
                         -(math.log(10000.0) / self.embed_dim))

        # 应用sin和cos函数
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播，添加位置编码

        Args:
            x: 输入嵌入 [batch_size, seq_len, embed_dim]

        Returns:
            添加位置编码后的输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len <= self.max_seq_len, f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}"
        assert embed_dim == self.embed_dim, f"嵌入维度不匹配: {embed_dim} vs {self.embed_dim}"

        # 添加位置编码
        pos_encoding = self.encoding[:seq_len, :].reshape(1, seq_len, embed_dim)
        return x + pos_encoding

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 输出梯度 [batch_size, seq_len, embed_dim]

        Returns:
            输入梯度 [batch_size, seq_len, embed_dim]
        """
        # 位置编码是固定的，所以梯度直接传递
        return grad_output


def test_positional_encoding():
    """测试位置编码"""
    print("测试位置编码...")
    embed_dim = 128
    max_seq_len = 512
    batch_size = 2
    seq_len = 10

    # 创建位置编码层
    pos_enc = PositionalEncoding(embed_dim, max_seq_len)

    # 创建测试输入
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = pos_enc.forward(x)
    assert output.shape == (batch_size, seq_len, embed_dim), f"输出形状错误: {output.shape}"

    # 验证位置编码确实改变了输入
    assert not np.allclose(output, x), "位置编码没有改变输入"

    # 测试位置编码的唯一性
    pos1 = pos_enc.encoding[0, :4]  # 第一个位置的前4个维度
    pos2 = pos_enc.encoding[1, :4]  # 第二个位置的前4个维度
    assert not np.allclose(pos1, pos2), "不同位置的位置编码应该不同"

    # 反向传播
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    grad_input = pos_enc.backward(grad_output)
    assert grad_input.shape == (batch_size, seq_len, embed_dim), f"梯度形状错误: {grad_input.shape}"
    assert np.allclose(grad_input, grad_output), "位置编码的梯度应该等于输出梯度"

    print("位置编码测试通过！")


if __name__ == "__main__":
    test_positional_encoding()