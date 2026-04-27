import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from gpt_from_scratch.model import GPT, softmax
from gpt_from_scratch.optim import AdamW
from gpt_from_scratch.tokenizer import load_tokenizer


def load_tokens(path, dtype):
    return np.fromfile(path, dtype=np.dtype(dtype))


def get_batch(data, batch_size, seq_len, xp, mask_data=None, min_mask_frac=0.15):
    if mask_data is None:
        ix = np.random.randint(0, len(data) - seq_len - 1, size=(batch_size,))
    else:
        ix = []
        min_mask = max(1, int(seq_len * min_mask_frac))
        attempts = 0
        while len(ix) < batch_size and attempts < batch_size * 100:
            i = int(np.random.randint(0, len(data) - seq_len - 1))
            if int(mask_data[i + 1 : i + seq_len + 1].sum()) >= min_mask:
                ix.append(i)
            attempts += 1
        while len(ix) < batch_size:
            ix.append(int(np.random.randint(0, len(data) - seq_len - 1)))
        ix = np.asarray(ix, dtype=np.int64)
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.int64)
    if mask_data is None:
        return xp.asarray(x), xp.asarray(y), None
    m = np.stack([mask_data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.float32)
    return xp.asarray(x), xp.asarray(y), xp.asarray(m)


def clip_grad_norm(grads, max_norm):
    xp = next(iter(grads.values())).__array_namespace__() if hasattr(next(iter(grads.values())), "__array_namespace__") else None
    total = None
    for g in grads.values():
        s = (g * g).sum()
        total = s if total is None else total + s
    norm = float(total.item() ** 0.5)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads.values():
            g *= scale
    return norm


def evaluate(model, val_data, batch_size, seq_len, iters, val_mask=None):
    losses = []
    for _ in range(iters):
        xb, yb, mb = get_batch(val_data, batch_size, seq_len, model.xp, val_mask)
        loss, _, _ = model.forward(xb, yb, mb)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=40):
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
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--min-mask-frac", type=float, default=0.15)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    train_data = load_tokens(data_dir / "train.bin", meta["dtype"])
    val_data = load_tokens(data_dir / "val.bin", meta["dtype"])
    train_mask = val_mask = None
    if (data_dir / "train_mask.bin").exists() and meta.get("assistant_loss_only"):
        train_mask = load_tokens(data_dir / "train_mask.bin", "uint8")
        val_mask = load_tokens(data_dir / "val_mask.bin", "uint8")

    if args.init_from:
        model = GPT.load(
            args.init_from,
            vocab_size=meta["vocab_size"],
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            device=args.device,
        )
    else:
        model = GPT(
            vocab_size=meta["vocab_size"],
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            seed=args.seed,
            device=args.device,
        )
    opt = AdamW(model.params, lr=args.lr, weight_decay=args.weight_decay)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"params={model.num_params():,} train_tokens={len(train_data):,} val_tokens={len(val_data):,}")
    t0 = time.time()
    best_val = float("inf")
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            lr = args.min_lr + (args.lr - args.min_lr) * cosine
        opt.lr = lr
        xb, yb, mb = get_batch(train_data, args.batch_size, args.seq_len, model.xp, train_mask, args.min_mask_frac)
        loss, dlogits, cache = model.forward(xb, yb, mb)
        grads = model.backward(dlogits, cache)
        grad_norm = clip_grad_norm(grads, 1.0)
        opt.step(grads)

        if step == 1 or step % 10 == 0:
            elapsed = max(time.time() - t0, 1e-9)
            toks = step * args.batch_size * args.seq_len
            print(f"step {step:5d} loss {float(loss.item()):.4f} lr {lr:.2e} grad {grad_norm:.3f} tok/s {toks/elapsed:.0f}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_data, args.batch_size, args.seq_len, args.eval_iters, val_mask)
            ppl = math.exp(min(20.0, val_loss))
            print(f"eval step {step:5d} val_loss {val_loss:.4f} ppl {ppl:.2f}", flush=True)
            prompt = "用户：请用一段话介绍中国古代四大发明。\n助手："
            print(generate(model, tokenizer, prompt, max_new_tokens=120), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                model.save(
                    ckpt_dir / "best.npz",
                    {
                        "step": step,
                        "vocab_size": meta["vocab_size"],
                        "seq_len": args.seq_len,
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "d_ff": args.d_ff or 4 * args.d_model,
                        "val_loss": val_loss,
                    },
                )
                print(f"saved {ckpt_dir / 'best.npz'}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            model.save(
                ckpt_dir / "latest.npz",
                {
                    "step": step,
                    "vocab_size": meta["vocab_size"],
                    "seq_len": args.seq_len,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "d_ff": args.d_ff or 4 * args.d_model,
                },
            )
            print(f"saved {ckpt_dir / 'latest.npz'}", flush=True)


if __name__ == "__main__":
    main()
