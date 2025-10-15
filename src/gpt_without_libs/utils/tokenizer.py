"""
简单的分词器
基于字符级别的分词，适合小型语料
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import pickle
import os


class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        初始化分词器

        Args:
            vocab: 词汇表，如果为None则从语料中学习
        """
        if vocab is not None:
            self.vocab = vocab
            self.vocab_size = len(vocab)
            self.char_to_id = {char: i for i, char in enumerate(vocab)}
            self.id_to_char = {i: char for i, char in enumerate(vocab)}
        else:
            self.vocab = []
            self.vocab_size = 0
            self.char_to_id = {}
            self.id_to_char = {}

        # 特殊token
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.start_token = "<s>"
        self.end_token = "</s>"

        self.pad_id = None
        self.unk_id = None
        self.start_id = None
        self.end_id = None

        self._add_special_tokens()

    def _add_special_tokens(self):
        """添加特殊token"""
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]

        for token in special_tokens:
            if token not in self.char_to_id:
                self.char_to_id[token] = self.vocab_size
                self.id_to_char[self.vocab_size] = token
                self.vocab.append(token)
                self.vocab_size += 1

        # 设置特殊token的ID
        self.pad_id = self.char_to_id[self.pad_token]
        self.unk_id = self.char_to_id[self.unk_token]
        self.start_id = self.char_to_id[self.start_token]
        self.end_id = self.char_to_id[self.end_token]

    def train(self, texts: List[str], min_freq: int = 1):
        """
        从文本中训练分词器

        Args:
            texts: 训练文本列表
            min_freq: 最小词频阈值
        """
        # 统计字符频率
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 添加高频字符到词汇表
        for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq and char not in self.char_to_id:
                self.char_to_id[char] = self.vocab_size
                self.id_to_char[self.vocab_size] = char
                self.vocab.append(char)
                self.vocab_size += 1

        print(f"训练完成，词汇表大小: {self.vocab_size}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        编码文本为token序列

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token

        Returns:
            token序列
        """
        if add_special_tokens:
            tokens = [self.start_id]
        else:
            tokens = []

        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))

        if add_special_tokens:
            tokens.append(self.end_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        解码token序列为文本

        Args:
            token_ids: token序列
            skip_special_tokens: 是否跳过特殊token

        Returns:
            解码后的文本
        """
        text = ""
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_id, self.start_id, self.end_id]:
                continue
            char = self.id_to_char.get(token_id, self.unk_token)
            text += char
        return text

    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None,
                     padding_side: str = 'right', truncation: bool = True) -> np.ndarray:
        """
        填充序列到相同长度

        Args:
            sequences: token序列列表
            max_length: 最大长度，如果为None则使用最长序列的长度
            padding_side: 填充位置 ('left' 或 'right')
            truncation: 是否截断超过最大长度的序列

        Returns:
            填充后的序列数组 [num_sequences, max_length]
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        for seq in sequences:
            if truncation and len(seq) > max_length:
                seq = seq[:max_length]
            elif len(seq) < max_length:
                pad_length = max_length - len(seq)
                if padding_side == 'right':
                    seq = seq + [self.pad_id] * pad_length
                else:
                    seq = [self.pad_id] * pad_length + seq
            padded_sequences.append(seq)

        return np.array(padded_sequences, dtype=np.int32)

    def save(self, filepath: str):
        """
        保存分词器

        Args:
            filepath: 保存路径
        """
        tokenizer_state = {
            'vocab': self.vocab,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'start_token': self.start_token,
            'end_token': self.end_token
        }

        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_state, f)

        print(f"分词器已保存到: {filepath}")

    def load(self, filepath: str):
        """
        加载分词器

        Args:
            filepath: 加载路径
        """
        with open(filepath, 'rb') as f:
            tokenizer_state = pickle.load(f)

        self.vocab = tokenizer_state['vocab']
        self.char_to_id = tokenizer_state['char_to_id']
        self.id_to_char = tokenizer_state['id_to_char']
        self.vocab_size = tokenizer_state['vocab_size']
        self.pad_token = tokenizer_state['pad_token']
        self.unk_token = tokenizer_state['unk_token']
        self.start_token = tokenizer_state['start_token']
        self.end_token = tokenizer_state['end_token']

        self.pad_id = self.char_to_id[self.pad_token]
        self.unk_id = self.char_to_id[self.unk_token]
        self.start_id = self.char_to_id[self.start_token]
        self.end_id = self.char_to_id[self.end_token]

        print(f"分词器已从 {filepath} 加载")


def generate_sample_data(num_samples: int = 1000, min_length: int = 50, max_length: int = 200) -> List[str]:
    """
    生成简单的英文样本数据

    Args:
        num_samples: 样本数量
        min_length: 最小长度
        max_length: 最大长度

    Returns:
        样本文本列表
    """
    # 简单的英文词汇库
    words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "can", "cannot",
        "this", "that", "these", "those", "it", "they", "he", "she", "we", "you", "i",
        "good", "bad", "big", "small", "hot", "cold", "new", "old", "young", "fast", "slow",
        "go", "come", "run", "walk", "sit", "stand", "eat", "drink", "sleep", "work", "play",
        "book", "paper", "pen", "computer", "phone", "car", "house", "tree", "flower", "water",
        "hello", "goodbye", "thank", "please", "sorry", "yes", "no", "maybe", "well", "now"
    ]

    sentences = []
    for _ in range(num_samples):
        # 生成随机长度的句子
        sentence_length = np.random.randint(min_length, max_length + 1)

        # 随机选择单词组成句子
        sentence_words = []
        for _ in range(sentence_length):
            word = np.random.choice(words)
            sentence_words.append(word)

        # 组成句子并添加标点
        sentence = " ".join(sentence_words) + "."
        sentences.append(sentence.capitalize())

    return sentences


def test_tokenizer():
    """测试分词器"""
    print("测试分词器...")

    # 生成样本数据
    texts = generate_sample_data(100, 10, 50)

    # 创建并训练分词器
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)

    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 测试编码和解码
    test_text = "Hello world, this is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded[:10]}...")  # 只显示前10个
    print(f"解码: {decoded}")

    # 测试序列填充
    sequences = [tokenizer.encode(text) for text in texts[:5]]
    padded = tokenizer.pad_sequences(sequences, max_length=20)

    print(f"填充前序列形状: {[len(seq) for seq in sequences]}")
    print(f"填充后形状: {padded.shape}")

    # 测试保存和加载
    tokenizer.save("test_tokenizer.pkl")
    new_tokenizer = SimpleTokenizer()
    new_tokenizer.load("test_tokenizer.pkl")

    # 清理测试文件
    if os.path.exists("test_tokenizer.pkl"):
        os.remove("test_tokenizer.pkl")

    print("分词器测试通过！✅")


if __name__ == "__main__":
    test_tokenizer()