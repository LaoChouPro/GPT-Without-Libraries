import argparse
import json
import random


def conv(user, assistant):
    return {"conversations": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/math_drill.jsonl")
    parser.add_argument("--max-add", type=int, default=99)
    parser.add_argument("--max-mul", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    examples = []
    for _ in range(args.repeat):
        for a in range(args.max_add + 1):
            for b in range(args.max_add + 1):
                s = a + b
                examples.append(conv(f"{a}加{b}等于多少？", f"{s}。"))
                examples.append(conv(f"{a}+{b}等于多少？", f"{s}。"))
        for a in range(2, args.max_mul + 1):
            for b in range(2, args.max_mul + 1):
                p = a * b
                examples.append(conv(f"{a}乘以{b}等于多少？", f"{p}。"))
                examples.append(conv(f"{a}x{b}等于多少？", f"{p}。"))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    with open(args.output, "w", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(examples), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
