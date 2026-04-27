import argparse
import json
import random
from collections import Counter


BAD_ASSISTANT_PHRASES = (
    "请提供更多",
    "请提供具体",
    "请提供您",
    "请提供您想要",
    "无法提供实时",
    "无法实时",
    "无法获取实时",
    "无法访问",
    "我无法",
    "作为一个AI",
    "作为AI",
    "抱歉",
    "没有提供",
    "需要更多上下文",
    "无法确定",
    "可能是一个特定",
    "这个词语是一个",
    "这个句子通常",
    "我是通义千问",
    "通义千问",
    "阿里云开发",
    "我是ChatGPT",
    "OpenAI开发",
    "由OpenAI",
    "Qwen",
)

BAD_USER_PHRASES = (
    "翻译成英文",
    "翻译为英文",
    "翻译成英语",
    "翻译为英语",
    "通义千问",
    "Qwen",
    "阿里云开发",
)

TOPIC_TERMS = (
    "四大发明",
    "造纸术",
    "指南针",
    "火药",
    "印刷术",
    "北京",
    "春天",
    "自我介绍",
    "你好",
    "计划",
    "比较",
    "解释",
    "数学",
)


def conv(user, assistant):
    return {"conversations": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


SEED_EXAMPLES = [
    conv(
        "请用一段话介绍中国古代四大发明。",
        "中国古代四大发明是造纸术、印刷术、火药和指南针。造纸术和印刷术推动了知识传播，火药改变了军事技术，指南针促进了航海和远距离交流。它们对中国和世界文明的发展都有深远影响。",
    ),
    conv(
        "中国古代四大发明是什么？",
        "中国古代四大发明指造纸术、印刷术、火药、指南针。它们分别促进了书写保存、知识传播、军事技术和航海导航的发展，是中国古代科技的重要代表。",
    ),
    conv(
        "四大发明包括哪些？",
        "四大发明包括造纸术、印刷术、火药和指南针。简单说，纸让文字更容易保存，印刷让书籍更容易传播，火药影响了战争形态，指南针帮助人类远航。",
    ),
    conv(
        "北京春天适合去哪里玩？",
        "北京春天适合去玉渊潭看樱花、去颐和园和北海公园踏青、去北京植物园赏桃花，也可以去奥林匹克森林公园散步骑行。出门前注意查看风沙和花粉情况。",
    ),
    conv(
        "请描述一下北京的春天。",
        "北京的春天通常在3月至5月，天气逐渐转暖，玉渊潭、颐和园、北海公园和北京植物园会陆续开花。这个季节适合踏青、赏花和骑行，但也要注意大风、沙尘和昼夜温差。",
    ),
    conv(
        "你好，请介绍一下你自己。",
        "你好！我是一个从零实现的小型中文助手，可以根据你的问题进行简洁回答、解释概念、整理要点和给出计划。我的能力有限，但会尽量给出清楚、直接、有用的答案。",
    ),
    conv(
        "你能做什么？",
        "我可以回答常识问题、解释概念、做简单推理、写简短文本、整理列表和给出行动计划。如果问题需要最新实时信息，我会说明限制，并尽量基于已有信息回答。",
    ),
    conv(
        "如果你不知道答案应该怎么办？",
        "如果我不确定答案，应该直接说明不确定，并给出可以验证的方向，而不是编造细节。能推理的部分可以说明推理过程，不能确认的部分要明确标注。",
    ),
    conv(
        "请把“坚持训练，持续改进”改写得更自然。",
        "可以改写为：持续训练模型，并根据结果不断改进系统。",
    ),
    conv(
        "请用三点说明学习编程的方法。",
        "第一，先掌握变量、函数、数据结构等基础概念。第二，多写小项目，把知识用起来。第三，经常阅读和调试代码，从错误中总结规律。",
    ),
    conv(
        "请解释什么是机器学习。",
        "机器学习是一类让计算机从数据中学习规律的方法。它不是把所有规则都手写出来，而是通过训练样本调整模型参数，使模型能对新输入做预测或生成结果。",
    ),
    conv(
        "Transformer 的注意力机制在做什么？",
        "注意力机制会让每个位置根据相关性去读取其他位置的信息。对语言模型来说，它可以让当前词参考前文中更重要的词，从而生成更连贯的下一个词。",
    ),
    conv(
        "请比较苹果和橙子。",
        "苹果和橙子都是常见水果，都含有水分和维生素。苹果口感通常更脆，甜酸比较均衡；橙子汁水更多，酸甜明显，维生素C含量较突出。",
    ),
    conv(
        "17加28等于多少？",
        "17加28等于45。",
    ),
    conv(
        "如果一本书每天读20页，7天能读多少页？",
        "每天读20页，7天就是20乘以7，所以一共能读140页。",
    ),
    conv(
        "请给我一个周末学习计划。",
        "周六上午复习基础知识，下午做练习项目，晚上整理笔记。周日上午回顾错题和薄弱点，下午完成一个小作品，晚上总结下周要继续改进的内容。",
    ),
]


def add_generated_examples(full_arithmetic=False):
    examples = []
    if full_arithmetic:
        for a in range(0, 61):
            for b in range(0, 61):
                examples.append(conv(f"{a}加{b}等于多少？", f"{a}加{b}等于{a + b}。"))
        for a in range(2, 21):
            for b in range(2, 21):
                examples.append(conv(f"{a}乘以{b}等于多少？", f"{a}乘以{b}等于{a * b}。"))
        return examples
    for a in range(3, 20):
        for b in range(2, 15):
            if len(examples) >= 180:
                return examples
            examples.append(conv(f"{a}加{b}等于多少？", f"{a}加{b}等于{a + b}。"))
            examples.append(conv(f"{a}乘以{b}等于多少？", f"{a}乘以{b}等于{a * b}。"))
    return examples


def assistant_text(obj):
    return "\n".join(
        str(msg.get("content", ""))
        for msg in obj.get("conversations", [])
        if msg.get("role") == "assistant"
    )


def is_single_turn(obj):
    convs = obj.get("conversations", [])
    return (
        len(convs) == 2
        and convs[0].get("role") == "user"
        and convs[1].get("role") == "assistant"
        and str(convs[0].get("content", "")).strip()
        and str(convs[1].get("content", "")).strip()
    )


def too_repetitive(text):
    if len(text) < 80:
        return False
    chars = Counter(ch for ch in text if not ch.isspace())
    if chars and chars.most_common(1)[0][1] / max(1, sum(chars.values())) > 0.22:
        return True
    pairs = Counter(text[i : i + 2] for i in range(len(text) - 1))
    return bool(pairs and pairs.most_common(1)[0][1] > max(12, len(text) // 12))


def is_good(obj, min_answer_chars, max_answer_chars, max_user_chars):
    if not is_single_turn(obj):
        return False
    user = str(obj["conversations"][0]["content"]).strip()
    answer = str(obj["conversations"][1]["content"]).strip()
    if not (4 <= len(user) <= max_user_chars):
        return False
    if not (min_answer_chars <= len(answer) <= max_answer_chars):
        return False
    if any(phrase in user for phrase in BAD_USER_PHRASES):
        return False
    if "天气" in user and any(phrase in user for phrase in ("今天", "最近", "实时", "周末", "本周")):
        return False
    if any(phrase in answer for phrase in BAD_ASSISTANT_PHRASES):
        return False
    if too_repetitive(answer):
        return False
    return True


def score_topic(obj):
    text = str(obj["conversations"][0]["content"]) + "\n" + assistant_text(obj)
    return sum(1 for term in TOPIC_TERMS if term in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.jsonl")
    parser.add_argument("--output", default="data/curriculum_short.jsonl")
    parser.add_argument("--max-source", type=int, default=500000)
    parser.add_argument("--max-general", type=int, default=30000)
    parser.add_argument("--max-topic", type=int, default=4000)
    parser.add_argument("--seed-repeat", type=int, default=80)
    parser.add_argument("--min-answer-chars", type=int, default=16)
    parser.add_argument("--max-answer-chars", type=int, default=360)
    parser.add_argument("--max-user-chars", type=int, default=160)
    parser.add_argument("--shuffle-seed", type=int, default=20260427)
    parser.add_argument("--full-arithmetic", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.shuffle_seed)
    seeds = (SEED_EXAMPLES + add_generated_examples(args.full_arithmetic)) * args.seed_repeat
    rng.shuffle(seeds)

    topic = []
    general = []
    seen = 0
    kept_candidates = 0
    with open(args.input, "r", encoding="utf-8") as src:
        for line in src:
            if args.max_source is not None and seen >= args.max_source:
                break
            seen += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not is_good(obj, args.min_answer_chars, args.max_answer_chars, args.max_user_chars):
                continue
            kept_candidates += 1
            if score_topic(obj) > 0 and len(topic) < args.max_topic:
                topic.append(obj)
            elif len(general) < args.max_general:
                general.append(obj)
            if len(topic) >= args.max_topic and len(general) >= args.max_general:
                break

    combined = seeds + topic + general
    rng.shuffle(combined)
    with open(args.output, "w", encoding="utf-8") as dst:
        for obj in combined:
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "seen": seen,
                "kept_candidates": kept_candidates,
                "seed_examples": len(seeds),
                "topic_examples": len(topic),
                "general_examples": len(general),
                "total": len(combined),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
