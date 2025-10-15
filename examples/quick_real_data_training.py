"""
使用真实data.txt的快速训练版本
适合快速验证和测试
"""

import numpy as np
import sys
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import time
import os
from typing import List
import re

from gpt_without_libs.models.core.mini_gpt import MiniGPT
from improved_dataset import WordTokenizer
from improved_training import ImprovedTrainer


class QuickRealDataProcessor:
    """快速真实数据处理器"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_and_preprocess(self) -> List[str]:
        """快速加载和预处理数据"""
        print(f"📂 快速加载数据文件: {self.filepath}")

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"✅ 文件加载成功")

            # 简化处理：按段落分割
            paragraphs = content.split('\n\n')
            processed_texts = []

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                # 清理
                paragraph = paragraph.replace('**', '').replace('*', '')
                paragraph = paragraph.replace('—', '-')
                paragraph = re.sub(r'\s+', ' ', paragraph)

                # 过滤长度合适的段落
                if 50 <= len(paragraph) <= 300:
                    processed_texts.append(paragraph)

            print(f"✅ 处理完成，获得 {len(processed_texts)} 个高质量段落")
            return processed_texts

        except Exception as e:
            print(f"❌ 文件加载失败: {e}")
            return []


def quick_real_data_main():
    """快速真实数据训练主函数"""
    print("=" * 80)
    print("🚀 快速使用真实data.txt数据集训练GPT模型")
    print("=" * 80)

    # 1. 加载真实数据
    print("\n📂 加载真实数据集...")
    data_processor = QuickRealDataProcessor('data.txt')
    training_texts = data_processor.load_and_preprocess()

    if not training_texts:
        print("❌ 数据加载失败")
        return

    # 数据统计
    total_chars = sum(len(text) for text in training_texts)
    total_words = sum(len(text.split()) for text in training_texts)
    print(f"📊 数据统计:")
    print(f"  样本数: {len(training_texts)}")
    print(f"  总字符: {total_chars:,}")
    print(f"  总单词: {total_words:,}")

    # 显示样本
    print(f"\n📝 训练样本示例:")
    for i, text in enumerate(training_texts[:3]):
        print(f"{i+1}. {text[:150]}...")

    # 2. 创建分词器
    print(f"\n🔤 创建词级分词器...")
    tokenizer = WordTokenizer(min_freq=1)  # 降低最小频率以保留更多词汇
    tokenizer.train(training_texts)
    print(f"✅ 词汇表大小: {tokenizer.vocab_size}")

    # 3. 创建适配的模型
    vocab_size = tokenizer.vocab_size
    embed_dim = 128  # 适中的嵌入维度
    num_heads = 4    # 适中的注意力头数（必须能整除embed_dim）
    num_layers = 4   # 适中的层数
    hidden_dim = embed_dim * 2

    print(f"\n🧠 创建适配模型:")
    print(f"  词汇表: {vocab_size}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头: {num_heads}")
    print(f"  层数: {num_layers}")

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_len=128,
        dropout=0.1
    )

    print(f"✅ 模型参数: {model.count_parameters():,}")

    # 4. 创建训练器
    trainer = ImprovedTrainer(model, tokenizer)

    # 5. 快速训练配置
    print(f"\n🎯 开始快速训练...")
    print("=" * 50)

    training_config = {
        'epochs': 8,      # 较少的训练轮数
        'batch_size': 4,  # 较小的批次大小
        'max_length': 64, # 适中的序列长度
        'learning_rate': 0.001,
        'save_dir': 'quick_real_checkpoints'
    }

    print(f"快速训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    start_time = time.time()

    try:
        # 执行训练
        losses = trainer.train(
            texts=training_texts,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            max_length=training_config['max_length'],
            learning_rate=training_config['learning_rate'],
            save_dir=training_config['save_dir'],
            save_every=5
        )

        training_time = time.time() - start_time
        print(f"\n🎉 快速训练完成! 耗时: {training_time:.2f}秒")
        print(f"最终损失: {losses[-1]:.4f}")

        # 6. 测试真实数据训练效果
        print(f"\n🎭 测试真实数据训练效果:")
        print("=" * 50)

        test_prompts = [
            "Education is",
            "Technology has",
            "Students need",
            "Learning helps",
            "In the future",
            "Programming teaches",
            "Knowledge and"
        ]

        for prompt in test_prompts:
            print(f"\n📝 '{prompt}'")

            for temp in [0.6, 0.9]:
                generated = trainer.generate(prompt, max_length=30, temperature=temp, top_k=8)
                print(f"  T={temp}: '{generated}'")
            print("-" * 30)

        # 7. 保存模型
        print(f"\n💾 保存快速训练模型...")
        os.makedirs('quick_real_checkpoints', exist_ok=True)
        model.save("quick_real_checkpoints/quick_real_model.pkl")
        tokenizer.save("quick_real_checkpoints/quick_real_tokenizer.pkl")

        print(f"\n" + "=" * 80)
        print(f"🎊 真实数据快速训练完成!")
        print(f"✅ 模型保存: quick_real_checkpoints/quick_real_model.pkl")
        print(f"✅ 分词器: quick_real_checkpoints/quick_real_tokenizer.pkl")
        print(f"✅ 基于真实英文文本，质量显著提升!")
        print("=" * 80)

    except KeyboardInterrupt:
        print(f"\n⏹️ 训练被中断")
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")


if __name__ == "__main__":
    quick_real_data_main()