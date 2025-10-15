"""
使用修复数据集的训练脚本
解决句点过度使用的问题
"""

import numpy as np
import sys
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import time
import os
from typing import List

from gpt_without_libs.models.core.mini_gpt import MiniGPT
from fixed_dataset import FixedEnglishTextDataset
from improved_dataset import WordTokenizer
from improved_training import ImprovedTrainer, softmax


def fixed_main():
    """使用修复数据集的主训练函数"""
    print("=" * 80)
    print("🔧 使用修复数据集的GPT模型训练 - 解决句点过度使用问题")
    print("=" * 80)

    # 1. 生成修复的训练数据
    print("\n📚 生成修复的训练数据...")
    dataset_generator = FixedEnglishTextDataset()

    # 生成多样化的数据
    simple_texts = [dataset_generator.generate_balanced_sentence() for _ in range(400)]
    paragraph_texts = [dataset_generator.generate_balanced_paragraph() for _ in range(300)]
    conversation_texts = [dataset_generator._generate_conversation() for _ in range(200)]
    mixed_texts = dataset_generator.generate_fixed_dataset(100)

    # 合并所有数据
    all_texts = simple_texts + paragraph_texts + conversation_texts + mixed_texts
    print(f"总共生成 {len(all_texts)} 个修复的训练样本")

    # 显示训练样本和标点统计
    print("\\n📝 修复后的训练样本示例:")
    for i in range(5):
        print(f"{i+1}. {all_texts[i]}")

    # 统计标点符号使用
    dot_count = sum(text.count('.') for text in all_texts)
    exclamation_count = sum(text.count('!') for text in all_texts)
    question_count = sum(text.count('?') for text in all_texts)
    comma_count = sum(text.count(',') for text in all_texts)
    total_punc = dot_count + exclamation_count + question_count + comma_count

    print(f"\\n📊 标点符号分布:")
    print(f"句点 (.): {dot_count} 次 ({dot_count/total_punc:.1%})")
    print(f"感叹号 (!): {exclamation_count} 次 ({exclamation_count/total_punc:.1%})")
    print(f"问号 (?): {question_count} 次 ({question_count/total_punc:.1%})")
    print(f"逗号 (,): {comma_count} 次 ({comma_count/total_punc:.1%})")

    # 2. 创建词级分词器
    print("\\n🔤 创建词级分词器...")
    tokenizer = WordTokenizer(min_freq=2)
    tokenizer.train(all_texts)

    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"词汇表示例: {list(tokenizer.word_to_id.keys())[:15]}...")

    # 3. 创建模型
    print("\\n🧠 创建模型...")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        max_seq_len=128,
        dropout=0.1
    )
    print(f"模型参数数量: {model.count_parameters():,}")

    # 4. 创建训练器
    print("\\n🏃‍♂️ 初始化训练器...")
    trainer = ImprovedTrainer(model, tokenizer)

    # 5. 开始训练
    print("\\n🎯 开始修复数据集训练...")
    print("=" * 60)

    training_config = {
        'epochs': 10,
        'batch_size': 8,
        'max_length': 64,
        'learning_rate': 0.0005,
        'save_dir': 'fixed_checkpoints',
        'save_every': 20
    }

    print(f"修复训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    start_time = time.time()

    try:
        # 执行训练
        losses = trainer.train(
            texts=all_texts,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            max_length=training_config['max_length'],
            learning_rate=training_config['learning_rate'],
            save_dir=training_config['save_dir'],
            save_every=training_config['save_every']
        )

        training_time = time.time() - start_time
        print(f"\\n🎉 修复训练完成! 总耗时: {training_time:.2f}秒")
        print(f"最终损失: {losses[-1]:.4f}")

        # 6. 测试修复后的生成效果
        print("\\n🔍 测试修复后的文本生成效果...")
        print("=" * 60)

        test_prompts = [
            "The weather is",
            "I like to",
            "Technology helps",
            "Students learn",
            "What do you",
            "She works as",
            "The best way",
            "My friends"
        ]

        for prompt in test_prompts:
            print(f"\\n📝 提示: '{prompt}'")

            # 使用不同参数生成
            generated1 = trainer.generate(prompt, max_length=25, temperature=0.7, top_k=8)
            generated2 = trainer.generate(prompt, max_length=25, temperature=1.0, top_k=12)

            print(f"  🌡️ T=0.7: '{generated1}'")
            print(f"  🌡️ T=1.0: '{generated2}'")

            # 统计标点使用
            dots = generated1.count('.')
            total = generated1.count('.') + generated1.count('!') + generated1.count('?') + generated1.count(',')
            if total > 0:
                dot_ratio = dots / total
                print(f"  📊 句点占比: {dot_ratio:.1%}")

            print("-" * 50)

        # 7. 保存修复的模型
        print("\\n💾 保存修复的模型...")
        os.makedirs('fixed_checkpoints', exist_ok=True)
        model.save("fixed_checkpoints/fixed_model.pkl")
        tokenizer.save("fixed_checkpoints/fixed_tokenizer.pkl")

        print("\\n" + "=" * 80)
        print("🎊 修复训练演示完成!")
        print("✅ 修复模型已保存: fixed_checkpoints/fixed_model.pkl")
        print("✅ 修复分词器已保存: fixed_checkpoints/fixed_tokenizer.pkl")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fixed_main()