"""
改进的训练脚本
使用更好的数据和更长的训练来提升语言能力
"""

import numpy as np
import sys
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import time
import os
from typing import List

from gpt_without_libs.models.core.mini_gpt import MiniGPT
from improved_dataset import EnglishTextDataset, WordTokenizer
from gpt_without_libs.training.training import Trainer, cross_entropy_loss


class ImprovedTrainer(Trainer):
    """改进的训练器"""

    def __init__(self, model: MiniGPT, tokenizer: WordTokenizer):
        super().__init__(model, tokenizer)

    def prepare_batch(self, texts: List[str], batch_size: int, max_length: int) -> tuple:
        """准备训练批次，使用词级分词器"""
        # 随机选择batch_size个文本
        selected_texts = np.random.choice(texts, batch_size, replace=True)

        # 编码文本
        sequences = []
        for text in selected_texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > max_length + 1:
                encoded = encoded[:max_length + 1]
            sequences.append(encoded)

        # 填充序列
        padded = self.tokenizer.pad_sequences(sequences, max_length=max_length + 1, padding_side='right')

        # 分离输入和目标
        input_ids = padded[:, :-1]  # [batch_size, max_length]
        target_ids = padded[:, 1:]  # [batch_size, max_length]

        return input_ids, target_ids

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, learning_rate: float) -> float:
        """执行一步训练"""
        # 生成因果掩码
        batch_size, seq_len = input_ids.shape
        mask = self.model.generate_mask(seq_len)

        # 前向传播
        logits = self.model.forward(input_ids, mask)  # [batch_size, seq_len, vocab_size]

        # 计算损失和梯度
        loss, grad_logits = cross_entropy_loss(logits, target_ids, ignore_index=self.tokenizer.pad_id)

        # 反向传播
        self.model.backward(grad_logits)

        # 更新参数
        self.model.update(learning_rate)

        return loss

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None) -> str:
        """改进的文本生成"""
        # 编码提示
        input_ids = np.array([self.tokenizer.encode(prompt, add_special_tokens=True)])
        generated_ids = input_ids[0].tolist()

        self.model.cache = {}  # 清除缓存

        for _ in range(max_length):
            # 准备输入
            current_input = np.array([generated_ids[-self.model.max_seq_len:]])

            # 生成掩码
            seq_len = current_input.shape[1]
            mask = self.model.generate_mask(seq_len)

            # 前向传播
            with np.errstate(all='ignore'):  # 忽略数值警告
                logits = self.model.forward(current_input, mask)

            # 获取最后一个token的logits
            next_token_logits = logits[0, -1, :]

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K采样
            if top_k is not None:
                top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                top_k_logits = next_token_logits[top_k_indices]
                top_k_probs = softmax(top_k_logits)
                next_token_idx = np.random.choice(top_k_indices, p=top_k_probs)
            # Top-P采样
            elif top_p is not None:
                sorted_indices = np.argsort(next_token_logits)[::-1]
                sorted_logits = next_token_logits[sorted_indices]
                sorted_probs = softmax(sorted_logits)
                cumulative_probs = np.cumsum(sorted_probs)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
                probs = softmax(next_token_logits)
                next_token_idx = np.random.choice(len(probs), p=probs)
            else:
                # 直接从所有token中采样
                probs = softmax(next_token_logits)
                next_token_idx = np.random.choice(len(probs), p=probs)

            # 添加到生成序列
            generated_ids.append(int(next_token_idx))

            # 如果生成了结束token，停止生成
            if next_token_idx == self.tokenizer.end_id:
                break

        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """稳定的softmax函数"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def improved_main():
    """改进的主训练函数"""
    print("=" * 80)
    print("🚀 改进的GPT模型训练 - 提升语言能力")
    print("=" * 80)

    # 1. 生成高质量的训练数据
    print("\n📚 生成高质量训练数据...")
    dataset_generator = EnglishTextDataset()

    # 生成多样化的数据
    general_texts = dataset_generator.generate_dataset(
        num_samples=300,
        min_paragraphs=2,
        max_paragraphs=4,
        topics=["general"]
    )

    tech_texts = dataset_generator.generate_dataset(
        num_samples=200,
        min_paragraphs=1,
        max_paragraphs=3,
        topics=["technology"]
    )

    edu_texts = dataset_generator.generate_dataset(
        num_samples=200,
        min_paragraphs=1,
        max_paragraphs=3,
        topics=["education"]
    )

    life_texts = dataset_generator.generate_dataset(
        num_samples=200,
        min_paragraphs=1,
        max_paragraphs=3,
        topics=["life"]
    )

    stories = dataset_generator.get_story_dataset(100)

    # 合并所有数据
    all_texts = general_texts + tech_texts + edu_texts + life_texts + stories
    print(f"总共生成 {len(all_texts)} 个训练样本")

    # 显示一些训练样本
    print("\n📝 训练样本示例:")
    for i in range(3):
        print(f"{i+1}. {all_texts[i][:100]}...")

    # 2. 创建词级分词器
    print("\n🔤 创建词级分词器...")
    tokenizer = WordTokenizer(min_freq=2)
    tokenizer.train(all_texts)

    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"词汇表示例: {list(tokenizer.word_to_id.keys())[:15]}...")

    # 3. 创建更大的模型
    print("\n🧠 创建改进的模型...")
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

    # 4. 创建改进的训练器
    print("\n🏃‍♂️ 初始化改进的训练器...")
    trainer = ImprovedTrainer(model, tokenizer)

    # 5. 开始长时间训练
    print("\n🎯 开始长时间训练...")
    print("=" * 60)

    training_config = {
        'epochs': 15,  # 增加训练轮数
        'batch_size': 8,  # 适中的批次大小
        'max_length': 64,  # 适中的序列长度
        'learning_rate': 0.0005,  # 稍低的学习率
        'save_dir': 'improved_checkpoints',
        'save_every': 20
    }

    print(f"改进的训练配置:")
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
        print(f"\n🎉 训练完成! 总耗时: {training_time:.2f}秒")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"损失下降: {losses[0] - losses[-1]:.4f}")

        # 6. 全面测试文本生成
        print("\n🎭 测试改进的文本生成能力...")
        print("=" * 60)

        test_prompts = [
            "The weather is",
            "I like to",
            "Technology has",
            "Education is",
            "In the future",
            "Once upon a time",
            "The best way to",
            "Many people believe",
            "When I was young",
            "Computers help us"
        ]

        for prompt in test_prompts:
            print(f"\n📝 提示: '{prompt}'")

            # 使用不同参数生成
            generated1 = trainer.generate(prompt, max_length=30, temperature=0.7, top_k=8)
            generated2 = trainer.generate(prompt, max_length=30, temperature=1.0, top_k=12)
            generated3 = trainer.generate(prompt, max_length=30, temperature=0.5, top_p=0.9)

            print(f"  🌡️ T=0.7, Top-K=8:  '{generated1}'")
            print(f"  🌡️ T=1.0, Top-K=12: '{generated2}'")
            print(f"  🌡️ T=0.5, Top-P=0.9: '{generated3}'")
            print("-" * 50)

        # 7. 保存最终模型
        print("\n💾 保存最终模型...")
        os.makedirs('improved_checkpoints', exist_ok=True)
        model.save("improved_checkpoints/improved_model.pkl")
        tokenizer.save("improved_checkpoints/improved_tokenizer.pkl")

        print("\n" + "=" * 80)
        print("🎊 改进训练演示完成!")
        print("✅ 改进模型已保存: improved_checkpoints/improved_model.pkl")
        print("✅ 改进分词器已保存: improved_checkpoints/improved_tokenizer.pkl")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    improved_main()