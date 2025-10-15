"""
训练脚本
包含损失函数、优化器和训练循环
"""

import numpy as np
import time
import os
from typing import List, Tuple, Optional

from ..models.core.mini_gpt import MiniGPT
from ..utils.tokenizer import SimpleTokenizer


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """稳定的softmax函数"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -100) -> Tuple[float, np.ndarray]:
    """
    计算交叉熵损失和梯度

    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        targets: 目标token [batch_size, seq_len]
        ignore_index: 忽略的token索引

    Returns:
        (损失值, 对logits的梯度)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # 计算softmax概率
    probs = softmax(logits, axis=-1)  # [batch_size, seq_len, vocab_size]

    # 计算损失
    loss = 0.0
    valid_tokens = 0

    for b in range(batch_size):
        for t in range(seq_len):
            target = targets[b, t]
            if target != ignore_index:
                # 避免log(0)
                prob = probs[b, t, target]
                prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
                loss -= np.log(prob)
                valid_tokens += 1

    if valid_tokens > 0:
        loss /= valid_tokens

    # 计算梯度
    grad_logits = probs.copy()  # [batch_size, seq_len, vocab_size]

    for b in range(batch_size):
        for t in range(seq_len):
            target = targets[b, t]
            if target != ignore_index:
                grad_logits[b, t, target] -= 1.0
                grad_logits[b, t, :] /= valid_tokens
            else:
                grad_logits[b, t, :] = 0.0

    return loss, grad_logits


def cosine_annealing_schedule(step: int, total_steps: int, max_lr: float = 0.001, min_lr: float = 1e-5) -> float:
    """
    余弦退火学习率调度

    Args:
        step: 当前步数
        total_steps: 总步数
        max_lr: 最大学习率
        min_lr: 最小学习率

    Returns:
        当前学习率
    """
    cosine_factor = 0.5 * (1 + np.cos(np.pi * step / total_steps))
    return min_lr + (max_lr - min_lr) * cosine_factor


class Trainer:
    """训练器"""

    def __init__(self, model: MiniGPT, tokenizer: SimpleTokenizer):
        """
        初始化训练器

        Args:
            model: 模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_history = []

    def prepare_batch(self, texts: List[str], batch_size: int, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练批次

        Args:
            texts: 文本列表
            batch_size: 批次大小
            max_length: 最大长度

        Returns:
            (输入序列, 目标序列)
        """
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
        """
        执行一步训练

        Args:
            input_ids: 输入token
            target_ids: 目标token
            learning_rate: 学习率

        Returns:
            损失值
        """
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

    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 4,
              max_length: int = 128, learning_rate: float = 0.001,
              save_dir: str = "checkpoints", save_every: int = 100) -> List[float]:
        """
        训练模型

        Args:
            texts: 训练文本
            epochs: 训练轮数
            batch_size: 批次大小
            max_length: 最大序列长度
            learning_rate: 学习率
            save_dir: 保存目录
            save_every: 保存间隔

        Returns:
            损失历史
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        print(f"开始训练，总样本数: {len(texts)}")
        print(f"模型参数数量: {self.model.count_parameters():,}")

        total_steps = epochs * (len(texts) // batch_size)
        current_step = 0

        for epoch in range(epochs):
            epoch_losses = []
            epoch_start_time = time.time()

            # 计算每个epoch的批次数
            num_batches = len(texts) // batch_size

            for batch_idx in range(num_batches):
                # 调度学习率
                current_lr = cosine_annealing_schedule(current_step, total_steps, learning_rate, learning_rate * 0.1)

                # 准备批次数据
                input_ids, target_ids = self.prepare_batch(texts, batch_size, max_length)

                # 执行训练步骤
                loss = self.train_step(input_ids, target_ids, current_lr)
                epoch_losses.append(loss)

                # 记录损失
                self.training_history.append(loss)

                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss:.4f}, LR: {current_lr:.6f}")

                # 保存检查点
                if current_step % save_every == 0 and current_step > 0:
                    checkpoint_path = os.path.join(save_dir, f"model_step_{current_step}.pkl")
                    self.model.save(checkpoint_path)

                current_step += 1

            # 每个epoch结束时保存模型
            epoch_loss = np.mean(epoch_losses)
            epoch_time = time.time() - epoch_start_time

            print(f"Epoch {epoch + 1} 完成! 平均损失: {epoch_loss:.4f}, 耗时: {epoch_time:.2f}s")

            # 保存epoch模型
            epoch_checkpoint = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pkl")
            self.model.save(epoch_checkpoint)

        # 保存最终模型
        final_model_path = os.path.join(save_dir, "model_final.pkl")
        self.model.save(final_model_path)

        print(f"训练完成! 最终模型已保存到: {final_model_path}")
        return self.training_history

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> str:
        """
        生成文本

        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样

        Returns:
            生成的文本
        """
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


def test_training():
    """测试训练功能"""
    print("测试训练功能...")

    from tokenizer import generate_sample_data

    # 生成样本数据
    texts = generate_sample_data(50, 20, 80)

    # 创建分词器
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)

    # 创建小型模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        max_seq_len=64
    )

    # 创建训练器
    trainer = Trainer(model, tokenizer)

    print("开始简单训练测试...")

    # 执行几步训练
    losses = []
    for step in range(10):
        input_ids, target_ids = trainer.prepare_batch(texts, batch_size=2, max_length=32)
        loss = trainer.train_step(input_ids, target_ids, learning_rate=0.001)
        losses.append(loss)
        print(f"步骤 {step + 1}, 损失: {loss:.4f}")

    print(f"平均损失: {np.mean(losses):.4f}")

    # 测试文本生成
    prompt = "The weather is"
    generated = trainer.generate(prompt, max_length=30, temperature=0.8)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")

    print("训练功能测试通过！✅")


if __name__ == "__main__":
    test_training()