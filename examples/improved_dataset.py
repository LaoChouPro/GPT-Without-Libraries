"""
改进的英文语料数据集
包含更真实的英文文本和更好的分词策略
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import random


class EnglishTextDataset:
    """英文文本数据集生成器"""

    def __init__(self):
        # 常用英文单词库
        self.articles = ['the', 'a', 'an']
        self.pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']
        self.nouns = [
            'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world',
            'life', 'hand', 'part', 'child', 'eye', 'woman', 'place', 'work',
            'week', 'case', 'point', 'government', 'company', 'number', 'group',
            'problem', 'fact', 'home', 'water', 'room', 'mother', 'area', 'money',
            'story', 'book', 'school', 'student', 'teacher', 'computer', 'house',
            'car', 'city', 'country', 'university', 'family', 'friend', 'team'
        ]
        self.verbs = [
            'be', 'have', 'do', 'say', 'go', 'get', 'make', 'know', 'think', 'take',
            'see', 'come', 'want', 'look', 'use', 'find', 'give', 'tell', 'work',
            'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean',
            'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start', 'show',
            'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'happen',
            'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue'
        ]
        self.adjectives = [
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
            'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
            'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'happy', 'sad',
            'beautiful', 'ugly', 'fast', 'slow', 'hot', 'cold', 'easy', 'difficult',
            'interesting', 'boring', 'exciting', 'relaxing', 'clean', 'dirty', 'expensive',
            'cheap', 'modern', 'traditional', 'popular', 'famous', 'unknown'
        ]
        self.prepositions = [
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
            'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out',
            'against', 'during', 'without', 'before', 'under', 'around', 'near'
        ]
        self.conjunctions = ['and', 'but', 'or', 'because', 'so', 'yet', 'for', 'nor']
        self.adverbs = [
            'up', 'so', 'out', 'just', 'now', 'how', 'then', 'more', 'also', 'here',
            'well', 'only', 'very', 'even', 'back', 'there', 'down', 'still', 'in',
            'as', 'too', 'when', 'never', 'really', 'most', 'again', 'always',
            'often', 'sometimes', 'usually', 'quickly', 'slowly', 'carefully',
            'happily', 'sadly', 'easily', 'difficultly', 'beautifully'
        ]

        # 句子模板
        self.sentence_patterns = [
            # 主谓宾
            ['article', 'noun', 'verb', 'article', 'noun'],
            # 主系表
            ['article', 'noun', 'verb', 'adjective'],
            # 主谓介宾
            ['article', 'noun', 'verb', 'preposition', 'article', 'noun'],
            # 主谓状
            ['article', 'noun', 'verb', 'adverb'],
            # 复合句
            ['pronoun', 'verb', 'article', 'noun', 'conjunction', 'pronoun', 'verb', 'adjective'],
            # 疑问句
            ['auxiliary', 'pronoun', 'verb', 'article', 'noun', 'question'],
            # 否定句
            ['pronoun', 'auxiliary', 'not', 'verb', 'article', 'noun'],
        ]

        # 预定义的常用短语和句子
        self.common_phrases = [
            "the weather is nice today",
            "i like to read books",
            "she works at the university",
            "they are playing in the garden",
            "he is a good student",
            "we have a beautiful house",
            "the children are happy",
            "my family lives in the city",
            "i enjoy studying computer science",
            "the company is growing fast",
            "she speaks three languages",
            "they travel around the world",
            "he plays football every weekend",
            "we need to solve this problem",
            "the teacher explains the lesson",
            "students work hard for exams",
            "technology changes our lives",
            "music makes me feel happy",
            "the sun rises in the east",
            "birds sing in the morning",
            "i drink coffee every morning",
            "she writes interesting stories",
            "they build modern houses",
            "the car needs more fuel",
            "we celebrate birthdays together",
            "nature is very beautiful",
            "computers help people work",
            "friends support each other",
            "learning new skills is important",
            "the internet connects everyone",
            "exercise keeps you healthy",
            "books contain valuable knowledge",
            "time passes quickly",
            "money cannot buy happiness",
            "education opens many doors",
            "teamwork achieves great results",
            "honesty is the best policy",
            "patience is a virtue",
            "practice makes perfect",
            "health is wealth",
            "knowledge is power"
        ]

    def generate_sentence(self, pattern: List[str] = None, length_limit: int = 15) -> str:
        """根据模式生成句子"""
        if pattern is None:
            pattern = random.choice(self.sentence_patterns)

        words = []
        for word_type in pattern:
            if word_type == 'article':
                word = random.choice(self.articles)
            elif word_type == 'pronoun':
                word = random.choice(self.pronouns)
            elif word_type == 'noun':
                word = random.choice(self.nouns)
            elif word_type == 'verb':
                word = random.choice(self.verbs)
            elif word_type == 'adjective':
                word = random.choice(self.adjectives)
            elif word_type == 'preposition':
                word = random.choice(self.prepositions)
            elif word_type == 'conjunction':
                word = random.choice(self.conjunctions)
            elif word_type == 'adverb':
                word = random.choice(self.adverbs)
            elif word_type == 'auxiliary':
                word = random.choice(['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'can', 'could'])
            elif word_type == 'question':
                word = random.choice(['?', '.', '!'])
            else:
                word = random.choice(self.nouns)

            words.append(word)

            # 避免句子过长
            if len(words) >= length_limit:
                break

        sentence = ' '.join(words)

        # 确保句子以适当的标点结尾
        if not sentence.endswith(('.', '!', '?')):
            sentence += random.choice(['.', '.', '!'])

        return sentence

    def generate_paragraph(self, num_sentences: int = 5, topic: str = "general") -> str:
        """生成段落"""
        sentences = []

        # 如果有主题，先写一个主题句
        if topic != "general":
            topic_sentences = {
                "technology": ["Technology has changed our world in many ways.",
                              "Computers and smartphones are everywhere today.",
                              "The internet connects people globally."],
                "education": ["Education is very important for personal development.",
                             "Students learn many useful skills at school.",
                             "Teachers help students achieve their goals."],
                "life": ["Life is full of challenges and opportunities.",
                        "People work hard to achieve their dreams.",
                        "Family and friends provide emotional support."]
            }
            if topic in topic_sentences:
                sentences.append(random.choice(topic_sentences[topic]))

        # 生成其余句子
        for _ in range(num_sentences - len(sentences)):
            # 30%概率使用预定义短语
            if random.random() < 0.3:
                sentences.append(random.choice(self.common_phrases))
            else:
                sentences.append(self.generate_sentence())

        return ' '.join(sentences)

    def generate_dataset(self, num_samples: int = 1000,
                        min_paragraphs: int = 1,
                        max_paragraphs: int = 3,
                        topics: List[str] = None) -> List[str]:
        """生成数据集"""
        if topics is None:
            topics = ["general", "technology", "education", "life"]

        dataset = []

        for i in range(num_samples):
            # 随机选择段落数
            num_paragraphs = random.randint(min_paragraphs, max_paragraphs)

            # 随机选择主题
            topic = random.choice(topics)

            # 生成段落
            paragraph = self.generate_paragraph(num_paragraphs, topic)

            dataset.append(paragraph)

            # 每100个样本打印一次进度
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本")

        return dataset

    def get_story_dataset(self, num_stories: int = 100) -> List[str]:
        """生成故事数据集"""
        stories = []

        story_templates = [
            "Once upon a time, there was a {adjective} {noun} who lived in a {adjective} {noun}. Every day, the {noun} would {verb} to the {noun}. One day, something {adjective} happened.",
            "In a {adjective} land far away, a {adjective} {noun} discovered a {adjective} {noun}. The {noun} decided to {verb} and find the {adjective} {noun}. After many {adjective} adventures, the {noun} finally {verb}.",
            "The {adjective} {noun} woke up early in the morning. Today was a {adjective} day for {verb}-ing. The {noun} went to the {noun} and met a {adjective} {noun}. Together, they {verb} to the {adjective} {noun}."
        ]

        for i in range(num_stories):
            template = random.choice(story_templates)

            # 替换模板中的占位符
            story = template
            story = story.replace("{noun}", random.choice(self.nouns))
            story = story.replace("{verb}", random.choice(self.verbs))
            story = story.replace("{adjective}", random.choice(self.adjectives))

            # 添加更多细节
            story += f" The {random.choice(self.nouns)} was very {random.choice(self.adjectives)}."
            story += f" They {random.choice(self.verbs)} {random.choice(self.adverbs)}."
            story += " In the end, everything turned out well."

            stories.append(story)

            if (i + 1) % 20 == 0:
                print(f"已生成 {i + 1}/{num_stories} 个故事")

        return stories


class WordTokenizer:
    """词级别分词器"""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.vocab = []
        self.vocab_size = 0
        self.word_to_id = {}
        self.id_to_word = {}

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
            if token not in self.word_to_id:
                self.word_to_id[token] = self.vocab_size
                self.id_to_word[self.vocab_size] = token
                self.vocab.append(token)
                self.vocab_size += 1

        self.pad_id = self.word_to_id[self.pad_token]
        self.unk_id = self.word_to_id[self.unk_token]
        self.start_id = self.word_to_id[self.start_token]
        self.end_id = self.word_to_id[self.end_token]

    def preprocess_text(self, text: str) -> List[str]:
        """预处理文本，分词"""
        # 转换为小写
        text = text.lower()

        # 替换标点符号
        text = text.replace('.', ' .')
        text = text.replace(',', ' ,')
        text = text.replace('!', ' !')
        text = text.replace('?', ' ?')
        text = text.replace(';', ' ;')
        text = text.replace(':', ' :')

        # 分割单词
        words = text.split()

        # 过滤空字符串
        words = [word for word in words if word.strip()]

        return words

    def train(self, texts: List[str]):
        """训练分词器"""
        # 统计词频
        word_freq = {}

        for text in texts:
            words = self.preprocess_text(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # 添加高频词到词汇表
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= self.min_freq and word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab.append(word)
                self.vocab_size += 1

        print(f"词级分词器训练完成，词汇表大小: {self.vocab_size}")
        print(f"总词数: {len(word_freq)}, 保留词数: {self.vocab_size - 4}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        words = self.preprocess_text(text)

        tokens = []
        if add_special_tokens:
            tokens.append(self.start_id)

        for word in words:
            tokens.append(self.word_to_id.get(word, self.unk_id))

        if add_special_tokens:
            tokens.append(self.end_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token序列"""
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_id, self.start_id, self.end_id]:
                continue
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)

        return ' '.join(words)

    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None,
                     padding_side: str = 'right', truncation: bool = True) -> np.ndarray:
        """填充序列"""
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
        """保存分词器"""
        tokenizer_state = {
            'vocab': self.vocab,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'start_token': self.start_token,
            'end_token': self.end_token
        }

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_state, f)

        print(f"词级分词器已保存到: {filepath}")

    def load(self, filepath: str):
        """加载分词器"""
        import pickle
        with open(filepath, 'rb') as f:
            tokenizer_state = pickle.load(f)

        self.vocab = tokenizer_state['vocab']
        self.word_to_id = tokenizer_state['word_to_id']
        self.id_to_word = tokenizer_state['id_to_word']
        self.vocab_size = tokenizer_state['vocab_size']
        self.min_freq = tokenizer_state['min_freq']
        self.pad_token = tokenizer_state['pad_token']
        self.unk_token = tokenizer_state['unk_token']
        self.start_token = tokenizer_state['start_token']
        self.end_token = tokenizer_state['end_token']

        self.pad_id = self.word_to_id[self.pad_token]
        self.unk_id = self.word_to_id[self.unk_token]
        self.start_id = self.word_to_id[self.start_token]
        self.end_id = self.word_to_id[self.end_token]

        print(f"词级分词器已从 {filepath} 加载")


def test_improved_dataset():
    """测试改进的数据集"""
    print("🧪 测试改进的英文数据集...")

    # 创建数据生成器
    dataset_generator = EnglishTextDataset()

    # 生成小样本测试
    print("\\n生成的句子样本:")
    for i in range(5):
        sentence = dataset_generator.generate_sentence()
        print(f"{i+1}. {sentence}")

    print("\\n生成的段落样本:")
    paragraph = dataset_generator.generate_paragraph(3, "technology")
    print(f"技术主题段落: {paragraph}")

    print("\\n生成的故事样本:")
    story = dataset_generator.get_story_dataset(1)[0]
    print(f"故事: {story}")

    # 测试词级分词器
    print("\\n📚 测试词级分词器...")

    # 生成一些文本
    texts = [
        "The weather is nice today.",
        "I like to read books and learn new things.",
        "She works at the university as a teacher.",
        "Technology has changed our lives completely.",
        "Education is very important for everyone."
    ]

    tokenizer = WordTokenizer(min_freq=1)
    tokenizer.train(texts)

    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"词汇表内容: {list(tokenizer.word_to_id.keys())[:20]}...")

    # 测试编码解码
    test_text = "The weather is nice today."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")

    # 测试序列填充
    sequences = [tokenizer.encode(text) for text in texts]
    padded = tokenizer.pad_sequences(sequences, max_length=15)
    print(f"\\n填充后形状: {padded.shape}")

    print("\\n✅ 改进数据集测试完成!")


if __name__ == "__main__":
    test_improved_dataset()