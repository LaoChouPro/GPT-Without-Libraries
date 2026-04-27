import argparse
import json
from pathlib import Path

import numpy as np

from gpt_from_scratch.model import GPT, softmax
from gpt_from_scratch.tokenizer import load_tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
    xp = model.xp
    ids = tokenizer.encode(prompt, add_bos=True)
    for _ in range(max_new_tokens):
        ctx = ids[-model.seq_len :]
        x = xp.asarray(np.asarray(ctx, dtype=np.int64)[None, :])
        logits, _ = model.forward(x)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k:
            top_idx = xp.argsort(next_logits)[-top_k:]
            filtered = xp.full_like(next_logits, -1e9)
            filtered[top_idx] = next_logits[top_idx]
            next_logits = filtered
        probs = softmax(next_logits, axis=-1)
        next_id = int(xp.random.choice(xp.arange(model.vocab_size), size=(), p=probs).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="data/checkpoints/latest.npz")
    parser.add_argument("--prompt", default="用户：你好，请介绍一下你自己。\n助手：")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(data_dir / "tokenizer.json")
    ckpt = np.load(args.checkpoint)
    model = GPT.load(
        args.checkpoint,
        vocab_size=meta["vocab_size"],
        seq_len=int(ckpt["extra.seq_len"]),
        d_model=int(ckpt["extra.d_model"]),
        n_layers=int(ckpt["extra.n_layers"]),
        n_heads=int(ckpt["extra.n_heads"]),
        d_ff=int(ckpt["extra.d_ff"]),
        device=args.device,
    )
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
