import argparse
from pathlib import Path

import numpy as np

from gpt_from_scratch.checkpoint import check_tokenizer
from gpt_from_scratch.model import GPT
from gpt_from_scratch.sampling import generate
from gpt_from_scratch.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="data/checkpoints/latest.npz")
    parser.add_argument("--prompt", default="用户：你好，请介绍一下你自己。\n助手：")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    tok = load_tokenizer(Path(args.data_dir) / "tokenizer.json")
    check_tokenizer(args.checkpoint, tok)
    model = GPT.load(args.checkpoint, vocab_size=tok.vocab_size, device=args.device)
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k,
                   np.random.default_rng(args.seed)))


if __name__ == "__main__":
    main()
