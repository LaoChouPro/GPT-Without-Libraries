"""
使用真实data.txt数据集的训练脚本
使用高质量的英文文本进行训练
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
from improved_training import ImprovedTrainer, softmax


class RealDataProcessor:
    """真实数据处理器"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_and_preprocess(self) -> List[str]:
        """加载和预处理数据"""
        print(f"📂 加载数据文件: {self.filepath}")

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"✅ 文件加载成功，总字符数: {len(content)}")

            # 清理和分割文本
            processed_texts = self._process_text(content)

            print(f"✅ 处理完成，获得 {len(processed_texts)} 个训练样本")
            return processed_texts

        except Exception as e:
            print(f"❌ 文件加载失败: {e}")
            return []

    def _process_text(self, content: str) -> List[str]:
        """处理文本内容"""
        # 替换特殊字符
        content = content.replace('**', '')  # 移除粗体标记
        content = content.replace('*', '')   # 移除斜体标记
        content = content.replace('—', '-')  # 统一破折号
        content = content.replace('"', "'")  # 统一引号

        # 按句子分割（改进的句子分割）
        sentences = self._split_sentences(content)

        # 过滤和清理句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤太短的句子
                # 清理多余空格
                sentence = re.sub(r'\s+', ' ', sentence)
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """改进的句子分割"""
        # 使用正则表达式分割句子，保留标点符号
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)

        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def create_training_samples(self, sentences: List[str], max_length: int = 100) -> List[str]:
        """创建训练样本"""
        training_samples = []

        # 1. 单句样本
        for sentence in sentences:
            if len(sentence) <= max_length:
                training_samples.append(sentence)

        # 2. 组合句样本（2-3句组合）
        for i in range(len(sentences) - 1):
            combined = sentences[i] + ' ' + sentences[i + 1]
            if len(combined) <= max_length:
                training_samples.append(combined)

        # 3. 段落样本（4-6句组合）
        for i in range(len(sentences) - 3):
            paragraph = ' '.join(sentences[i:i+4])
            if len(paragraph) <= max_length:
                training_samples.append(paragraph)

        print(f"✅ 创建了 {len(training_samples)} 个训练样本")
        return training_samples


def analyze_data_statistics(texts: List[str]):
    """分析数据统计信息"""
    print("\n📊 数据统计分析:")
    print("=" * 40)

    # 基本统计
    total_chars = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)
    avg_length = total_chars / len(texts) if texts else 0
    avg_words = total_words / len(texts) if texts else 0

    print(f"样本数量: {len(texts)}")
    print(f"总字符数: {total_chars:,}")
    print(f"总单词数: {total_words:,}")
    print(f"平均长度: {avg_length:.1f} 字符")
    print(f"平均单词数: {avg_words:.1f} 个")

    # 标点符号统计
    dots = sum(text.count('.') for text in texts)
    exclamation = sum(text.count('!') for text in texts)
    question = sum(text.count('?') for text in texts)
    comma = sum(text.count(',') for text in texts)
    total_punc = dots + exclamation + question + comma

    print(f"\n标点符号分布:")
    if total_punc > 0:
        print(f"句点 (.): {dots} 次 ({dots/total_punc:.1%})")
        print(f"感叹号 (!): {exclamation} 次 ({exclamation/total_punc:.1%})")
        print(f"问号 (?): {question} 次 ({question/total_punc:.1%})")
        print(f"逗号 (,): {comma} 次 ({comma/total_punc:.1%})")

    # 词汇复杂度分析
    all_words = ' '.join(texts).lower().split()
    unique_words = len(set(all_words))
    avg_word_length = sum(len(word) for word in all_words) / len(all_words) if all_words else 0

    print(f"\n词汇分析:")
    print(f"总词数: {len(all_words):,}")
    print(f"独特词数: {unique_words:,}")
    print(f"词汇多样性: {unique_words/len(all_words):.2%}")
    print(f"平均词长: {avg_word_length:.1f} 字符")


def real_data_main():
    """使用真实数据集的主训练函数"""
    print("=" * 80)
    print("📚 使用真实data.txt数据集训练GPT模型")
    print("=" * 80)

    # 1. 加载和处理真实数据
    print("\n📂 加载真实数据集...")
    data_processor = RealDataProcessor('data.txt')
    raw_sentences = data_processor.load_and_preprocess()

    if not raw_sentences:
        print("❌ 数据加载失败，退出训练")
        return

    # 2. 创建训练样本
    print("\n🔧 创建训练样本...")
    training_texts = data_processor.create_training_samples(raw_sentences, max_length=150)

    # 3. 数据统计分析
    analyze_data_statistics(training_texts)

    # 4. 显示样本示例
    print("\n📝 训练样本示例:")
    for i, text in enumerate(training_texts[:5]):
        print(f"{i+1}. {text[:100]}...")

    # 5. 创建词级分词器
    print("\n🔤 创建词级分词器...")
    tokenizer = WordTokenizer(min_freq=2)
    tokenizer.train(training_texts)

    print(f"✅ 词汇表大小: {tokenizer.vocab_size}")
    print(f"词汇表示例: {list(tokenizer.word_to_id.keys())[:15]}...")

    # 6. 创建模型（根据词汇表大小调整）
    vocab_size = tokenizer.vocab_size
    embed_dim = min(256, max(128, vocab_size // 4))  # 根据词汇表大小动态调整
    num_heads = 8
    num_layers = 6
    hidden_dim = embed_dim * 4

    print(f"\n🧠 创建模型 (基于真实数据优化)...")
    print(f"词汇表大小: {vocab_size}")
    print(f"嵌入维度: {embed_dim}")
    print(f"注意力头数: {num_heads}")
    print(f"Transformer层数: {num_layers}")
    print(f"隐藏层维度: {hidden_dim}")

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_len=256,  # 增加序列长度以处理更长文本
        dropout=0.1
    )

    print(f"模型参数数量: {model.count_parameters():,}")

    # 7. 创建训练器
    print("\n🏃‍♂️ 初始化训练器...")
    trainer = ImprovedTrainer(model, tokenizer)

    # 8. 开始训练
    print("\n🎯 开始真实数据集训练...")
    print("=" * 60)

    training_config = {
        'epochs': 20,  # 增加训练轮数
        'batch_size': 8,
        'max_length': 128,  # 较长的序列长度
        'learning_rate': 0.0003,  # 稍低的学习率
        'save_dir': 'real_data_checkpoints',
        'save_every': 15
    }

    print(f"真实数据训练配置:")
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
            save_every=training_config['save_every']
        )

        training_time = time.time() - start_time
        print(f"\n🎉 真实数据训练完成! 总耗时: {training_time:.2f}秒")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"损失下降: {losses[0] - losses[-1]:.4f}")

        # 9. 全面测试真实数据训练效果
        print("\n🎭 测试真实数据训练效果...")
        print("=" * 60)

        test_prompts = [
            "Education is",
            "Technology has",
            "Lifelong learning",
            "In the classroom",
            "Students need",
            "Programming teaches",
            "The future of",
            "We must learn",
            "Knowledge and",
            "In conclusion"
        ]

        for prompt in test_prompts:
            print(f"\n📝 提示: '{prompt}'")

            # 使用不同参数生成
            generated1 = trainer.generate(prompt, max_length=40, temperature=0.6, top_k=8)
            generated2 = trainer.generate(prompt, max_length=40, temperature=0.9, top_k=12)
            generated3 = trainer.generate(prompt, max_length=40, temperature=1.2, top_k=15)

            print(f"  🌡️ T=0.6: '{generated1}'")
            print(f"  🌡️ T=0.9: '{generated2}'")
            print(f"  🌡️ T=1.2: '{generated3}'")

            # 统计标点使用
            dots = generated1.count('.')
            total = generated1.count('.') + generated1.count('!') + generated1.count('?') + generated1.count(',')
            if total > 0:
                dot_ratio = dots / total
                print(f"  📊 句点占比: {dot_ratio:.1%}")

            print("-" * 50)

        # 10. 保存真实数据训练的模型
        print("\n💾 保存真实数据训练的模型...")
        os.makedirs('real_data_checkpoints', exist_ok=True)
        model.save("real_data_checkpoints/real_data_model.pkl")
        tokenizer.save("real_data_checkpoints/real_data_tokenizer.pkl")

        print("\n" + "=" * 80)
        print("🎊 真实数据训练演示完成!")
        print("✅ 真实数据模型已保存: real_data_checkpoints/real_data_model.pkl")
        print("✅ 真实数据分词器已保存: real_data_checkpoints/real_data_tokenizer.pkl")
        print("✅ 训练基于真实的英文文本，质量更高!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    real_data_main()