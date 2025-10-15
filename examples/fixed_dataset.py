"""
修复的数据集生成器
解决句点过度使用的问题
"""

import numpy as np
import random
from typing import List


class FixedEnglishTextDataset:
    """修复的英文文本数据集生成器"""

    def __init__(self):
        # 基础词汇库（从原版本继承）
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

        # 标点符号分布（平衡各种标点的使用）
        self.punctuation_weights = {
            '.': 0.3,    # 降低句点的权重
            '!': 0.2,    # 增加感叹号
            '?': 0.15,   # 增加问号
            ',': 0.25,   # 增加逗号
            '': 0.1      # 10%的句子不添加标点
        }

        # 预定义的更自然的句子
        self.natural_sentences = [
            "the weather is very nice today",
            "i like reading books in my free time",
            "she works as a teacher at the local school",
            "they enjoy playing sports on weekends",
            "he is studying computer science at university",
            "we often go to the park for a walk",
            "the children are playing in the garden",
            "my family lives in a small town",
            "technology has changed our daily lives",
            "education is very important for success",
            "the internet helps people communicate",
            "students work hard to achieve good grades",
            "the company is growing very fast",
            "she speaks three different languages",
            "they travel around the world every year",
            "he plays football with his friends",
            "the sun rises early in the morning",
            "birds sing beautifully in the trees",
            "i drink coffee every morning",
            "she writes interesting stories for children",
            "the car needs more fuel for the journey",
            "we celebrate important holidays together",
            "nature provides many beautiful scenes",
            "computers help people work efficiently",
            "friends always support each other",
            "learning new skills opens many doors",
            "the internet connects everyone globally",
            "regular exercise keeps you healthy",
            "books contain valuable knowledge",
            "time passes by very quickly",
            "money cannot buy true happiness",
            "teamwork achieves great results",
            "honesty is always the best policy"
        ]

    def generate_balanced_sentence(self, max_length: int = 12) -> str:
        """生成平衡的句子，减少句点偏好"""
        # 30%概率使用预定义的自然句子
        if random.random() < 0.3:
            sentence = random.choice(self.natural_sentences)
        else:
            # 生成随机句子
            sentence = self._generate_random_sentence(max_length)

        # 平衡地添加标点符号
        punctuation = self._choose_punctuation()
        if punctuation:
            sentence += punctuation

        return sentence

    def _generate_random_sentence(self, max_length: int) -> str:
        """生成随机句子"""
        sentence_patterns = [
            ['article', 'noun', 'verb', 'article', 'adjective', 'noun'],
            ['pronoun', 'verb', 'article', 'noun', 'preposition', 'article', 'noun'],
            ['article', 'adjective', 'noun', 'verb', 'adverb'],
            ['pronoun', 'auxiliary', 'verb', 'article', 'noun', 'conjunction', 'verb', 'adjective'],
            ['article', 'noun', 'verb', 'preposition', 'noun'],
        ]

        pattern = random.choice(sentence_patterns)
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
            else:
                word = random.choice(self.nouns)

            words.append(word)
            if len(words) >= max_length:
                break

        return ' '.join(words)

    def _choose_punctuation(self) -> str:
        """根据权重选择标点符号"""
        punctuations = list(self.punctuation_weights.keys())
        weights = list(self.punctuation_weights.values())
        return random.choices(punctuations, weights=weights)[0]

    def generate_balanced_paragraph(self, num_sentences: int = 3) -> str:
        """生成平衡的段落"""
        sentences = []

        for i in range(num_sentences):
            if i == 0:
                # 第一句话
                sentence = self.generate_balanced_sentence()
            elif i < num_sentences - 1:
                # 中间的话，可能用逗号连接
                if random.random() < 0.3:
                    # 连接到前一句话
                    connector = random.choice(['and', 'but', 'so', 'because'])
                    sentence = connector + ' ' + self.generate_balanced_sentence(max_length=8) + ','
                else:
                    sentence = self.generate_balanced_sentence()
            else:
                # 最后一句话
                sentence = self.generate_balanced_sentence()

            sentences.append(sentence)

        # 组合成段落
        if len(sentences) == 1:
            return sentences[0]
        else:
            return ' '.join(sentences[:-1]) + ' and ' + sentences[-1]

    def generate_fixed_dataset(self, num_samples: int = 1000) -> List[str]:
        """生成修复的数据集"""
        dataset = []

        for i in range(num_samples):
            # 生成不同类型的内容
            content_type = random.choice(['simple', 'paragraph', 'conversation'])

            if content_type == 'simple':
                # 简单句子
                text = self.generate_balanced_sentence()
            elif content_type == 'paragraph':
                # 段落
                text = self.generate_balanced_paragraph(random.randint(2, 4))
            else:
                # 对话形式
                text = self._generate_conversation()

            dataset.append(text)

            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个修复样本")

        return dataset

    def _generate_conversation(self) -> str:
        """生成简单的对话"""
        speakers = ['i', 'you', 'he', 'she']
        questions = [
            "how are you today",
            "what do you think about this",
            "where do you want to go",
            "when will you arrive",
            "why did you choose that"
        ]

        speaker = random.choice(speakers)
        question = random.choice(questions)

        # 问题和回答
        if '?' not in question:
            question += '?'

        answers = [
            "i am doing great",
            "that sounds interesting",
            "the weather is nice",
            "i think it is a good idea",
            "everything is going well"
        ]

        answer = random.choice(answers)

        return f"{speaker} asked {question} {answer}."


def test_fixed_dataset():
    """测试修复的数据集"""
    print("🧪 测试修复的数据集...")

    generator = FixedEnglishTextDataset()

    print("\\n修复后的句子样本:")
    for i in range(10):
        sentence = generator.generate_balanced_sentence()
        print(f"{i+1}. {sentence}")

    print("\\n修复后的段落样本:")
    for i in range(3):
        paragraph = generator.generate_balanced_paragraph()
        print(f"{i+1}. {paragraph}")

    print("\\n对话样本:")
    for i in range(3):
        conversation = generator._generate_conversation()
        print(f"{i+1}. {conversation}")

    # 统计标点符号使用
    print("\\n标点符号使用统计:")
    test_data = generator.generate_fixed_dataset(100)

    dot_count = sum(text.count('.') for text in test_data)
    exclamation_count = sum(text.count('!') for text in test_data)
    question_count = sum(text.count('?') for text in test_data)
    comma_count = sum(text.count(',') for text in test_data)

    print(f"句点 (.): {dot_count} 次")
    print(f"感叹号 (!): {exclamation_count} 次")
    print(f"问号 (?): {question_count} 次")
    print(f"逗号 (,): {comma_count} 次")
    print(f"句点占比: {dot_count/(dot_count+exclamation_count+question_count+comma_count+1):.2%}")

    print("\\n✅ 修复数据集测试完成!")


if __name__ == "__main__":
    test_fixed_dataset()