"""
完整的模型测试脚本
测试所有组件的功能
"""

import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入所有模块
from gpt_without_libs.models.layers.embedding import test_embedding
from gpt_without_libs.models.layers.positional_encoding import test_positional_encoding
from gpt_without_libs.models.layers.attention import test_multihead_attention
from gpt_without_libs.models.layers.feedforward import test_feedforward
from gpt_without_libs.models.layers.layer_norm import test_layer_norm
from gpt_without_libs.models.layers.transformer_block import test_transformer_block
from gpt_without_libs.models.core.mini_gpt import MiniGPT, test_mini_gpt


def test_all_components():
    """测试所有组件"""
    print("=" * 50)
    print("开始测试所有组件...")
    print("=" * 50)

    try:
        # 测试基础组件
        test_embedding()
        test_positional_encoding()
        test_layer_norm()
        test_multihead_attention()
        test_feedforward()
        test_transformer_block()
        test_mini_gpt()

        print("=" * 50)
        print("所有组件测试通过！✅")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward_pass():
    """测试完整模型的前向传播"""
    print("\n测试完整模型前向传播...")

    from gpt_without_libs.models.core.mini_gpt import MiniGPT

    # 创建一个小的模型进行测试
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    hidden_dim = 64

    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)

    # 测试输入
    batch_size = 1
    seq_len = 20
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # 生成掩码
    mask = model.generate_mask(seq_len)

    # 前向传播
    logits = model.forward(x, mask)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"预期输出形状: ({batch_size}, {seq_len}, {vocab_size})")

    assert logits.shape == (batch_size, seq_len, vocab_size), "输出形状不匹配"

    # 检查输出是否合理（不是NaN或Inf）
    assert not np.any(np.isnan(logits)), "输出包含NaN"
    assert not np.any(np.isinf(logits)), "输出包含Inf"

    print("前向传播测试通过！✅")
    return True


def test_model_backward_pass():
    """测试完整模型的反向传播"""
    print("\n测试完整模型反向传播...")

    from gpt_without_libs.models.core.mini_gpt import MiniGPT

    # 创建模型
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    hidden_dim = 64

    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)

    # 测试数据
    batch_size = 1
    seq_len = 10
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model.forward(x)

    # 模拟损失梯度（简单的梯度）
    grad_logits = np.random.randn(*logits.shape) * 0.1

    # 反向传播
    grad_x = model.backward(grad_logits)

    print(f"输入梯度形状: {grad_x.shape}")
    print(f"预期输入梯度形状: {x.shape}")

    assert grad_x.shape == x.shape, "输入梯度形状不匹配"

    print("反向传播测试通过！✅")
    return True


def test_model_training_step():
    """测试模型的完整训练步骤"""
    print("\n测试模型训练步骤...")

    from gpt_without_libs.models.core.mini_gpt import MiniGPT

    # 创建模型
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    hidden_dim = 64

    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)

    # 模拟训练数据
    batch_size = 2
    seq_len = 8
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    y = np.random.randint(0, vocab_size, (batch_size, seq_len))

    learning_rate = 0.001

    # 保存初始参数
    initial_weights = model.output_projection.copy()

    # 前向传播
    logits = model.forward(x)

    # 计算简单的交叉熵损失梯度
    # 这里简化处理，实际训练中需要更复杂的损失计算
    loss_grad = np.zeros_like(logits)
    for b in range(batch_size):
        for t in range(seq_len):
            loss_grad[b, t, y[b, t]] = -1.0 / vocab_size

    # 反向传播
    model.backward(loss_grad)

    # 更新参数
    model.update(learning_rate)

    # 检查参数是否发生了变化
    weight_change = np.abs(model.output_projection - initial_weights).mean()
    print(f"权重平均变化: {weight_change:.6f}")

    assert weight_change > 1e-8, "权重没有更新"

    print("训练步骤测试通过！✅")
    return True


def main():
    """主测试函数"""
    print("开始运行完整的模型测试...")

    # 运行所有测试
    success = True

    try:
        success &= test_all_components()
        success &= test_model_forward_pass()
        success &= test_model_backward_pass()
        success &= test_model_training_step()

        if success:
            print("\n🎉 所有测试通过！模型功能正常。")
            print("\n模型参数统计:")
            from gpt_without_libs.models.core.mini_gpt import MiniGPT
            model = MiniGPT(1000, 256, 8, 6, 1024)
            num_params = model.count_parameters()
            print(f"完整模型参数数量: {num_params:,}")
        else:
            print("\n❌ 部分测试失败，请检查代码。")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()