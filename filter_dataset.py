import argparse
import json


BAD_PHRASES = (
    "请提供具体",
    "请提供更多",
    "无法提供实时",
    "无法获取实时",
    "作为一个AI",
    "作为AI",
    "我无法",
    "抱歉",
    "没有提供",
    "需要更多上下文",
)


def assistant_text(obj):
    return "\n".join(
        str(msg.get("content", ""))
        for msg in obj.get("conversations", [])
        if msg.get("role") == "assistant"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.jsonl")
    parser.add_argument("--output", default="data/filtered.jsonl")
    parser.add_argument("--max-in", type=int, default=None)
    parser.add_argument("--max-out", type=int, default=200000)
    parser.add_argument("--min-answer-chars", type=int, default=40)
    args = parser.parse_args()

    kept = 0
    seen = 0
    with open(args.input, "r", encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            if args.max_in is not None and seen >= args.max_in:
                break
            seen += 1
            obj = json.loads(line)
            answer = assistant_text(obj)
            if len(answer) < args.min_answer_chars:
                continue
            if any(phrase in answer for phrase in BAD_PHRASES):
                continue
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_out is not None and kept >= args.max_out:
                break
    print(json.dumps({"seen": seen, "kept": kept, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()

