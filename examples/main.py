"""
主训练脚本
完整的模型训练演示
"""

import numpy as np
import time
import os
import sys
from typing import List

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpt_without_libs.models.core.mini_gpt import MiniGPT
from gpt_without_libs.utils.tokenizer import SimpleTokenizer, generate_sample_data
from gpt_without_libs.training.training import Trainer


def main():
    """主训练函数"""
    print("=" * 60)
    print("微型GPT模型训练演示")
    print("=" * 60)

    # 1. 生成训练数据
    print("\n1. 生成训练数据...")
    num_samples = 200
    texts = generate_sample_data(num_samples, min_length=30, max_length=100)
    print(f"生成 {len(texts)} 个样本")

    # 2. 创建分词器
    print("\n2. 训练分词器...")
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 3. 创建模型
    print("\n3. 创建模型...")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        hidden_dim=256,
        max_seq_len=64,
        dropout=0.1
    )
    print(f"模型参数数量: {model.count_parameters():,}")

    # 4. 创建训练器
    print("\n4. 初始化训练器...")
    trainer = Trainer(model, tokenizer)

    # 5. 开始训练
    print("\n5. 开始训练...")
    print("=" * 40)

    training_config = {
        'epochs': 5,
        'batch_size': 4,
        'max_length': 32,
        'learning_rate': 0.001,
        'save_dir': 'checkpoints',
        'save_every': 50
    }

    print(f"训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    start_time = time.time()

    try:
        # 执行训练
        losses = trainer.train(
            texts=texts,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            max_length=training_config['max_length'],
            learning_rate=training_config['learning_rate'],
            save_dir=training_config['save_dir'],
            save_every=training_config['save_every']
        )

        training_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {training_time:.2f}秒")

        # 6. 测试文本生成
        print("\n6. 测试文本生成...")
        print("=" * 40)

        test_prompts = [
            "The weather is",
            "I like to",
            "Computer is",
            "Good morning"
        ]

        for prompt in test_prompts:
            generated = trainer.generate(
                prompt=prompt,
                max_length=20,
                temperature=0.8,
                top_k=10
            )
            print(f"提示: '{prompt}'")
            print(f"生成: '{generated}'")
            print("-" * 30)

        # 7. 保存最终模型和分词器
        print("\n7. 保存最终模型...")
        model.save("mini_gpt_final.pkl")
        tokenizer.save("tokenizer_final.pkl")

        print("\n" + "=" * 60)
        print("🎉 训练演示完成!")
        print("✅ 模型已保存: mini_gpt_final.pkl")
        print("✅ 分词器已保存: tokenizer_final.pkl")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试模式"""
    print("快速测试模式...")

    # 生成少量数据
    texts = generate_sample_data(50, 20, 50)

    # 创建分词器和模型
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        hidden_dim=128,
        max_seq_len=32
    )

    trainer = Trainer(model, tokenizer)

    # 快速训练几步
    print("执行快速训练...")
    for step in range(5):
        input_ids, target_ids = trainer.prepare_batch(texts, batch_size=2, max_length=16)
        loss = trainer.train_step(input_ids, target_ids, learning_rate=0.001)
        print(f"步骤 {step + 1}, 损失: {loss:.4f}")

    # 测试生成
    prompt = "Hello"
    generated = trainer.generate(prompt, max_length=15, temperature=0.7)
    print(f"提示: '{prompt}' -> 生成: '{generated}'")

    print("快速测试完成! ✅")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()