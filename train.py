import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from gpt_from_scratch.checkpoint import check_tokenizer, load_training, save_training
from gpt_from_scratch.data import BatchSampler, load_tokens
from gpt_from_scratch.model import GPT
from gpt_from_scratch.optim import AdamW
from gpt_from_scratch.sampling import generate
from gpt_from_scratch.tokenizer import load_tokenizer, tokenizer_fingerprint


def get_batch(data, batch_size, seq_len, xp, mask_data=None, min_mask_frac=0.15, rng=None):
    return BatchSampler(data, seq_len, mask_data, min_mask_frac).batch(batch_size, xp, rng)


def clip_grad_norm(grads, max_norm):
    if not math.isfinite(max_norm) or max_norm <= 0 or not grads:
        raise ValueError("max_norm must be positive and gradients nonempty")
    # Sum in float64 to avoid overflowing the norm of otherwise finite float32 gradients.
    total = sum(float((g.astype("float64") ** 2).sum().item()) for g in grads.values())
    norm = math.sqrt(total)
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite gradients; optimizer update cancelled")
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / (norm + 1e-12)
    return norm


def evaluate(model, val_data, batch_size, seq_len, iters, val_mask=None, seed=1337):
    if iters < 1:
        raise ValueError("eval iters must be positive")
    sampler = BatchSampler(val_data, seq_len, val_mask)
    rng = np.random.default_rng(seed)
    loss_sum, tokens = 0.0, 0.0
    for _ in range(iters):
        xb, yb, mb = sampler.batch(batch_size, model.xp, rng)
        loss, _, _ = model.forward(xb, yb, mb)
        count = yb.size if mb is None else float(mb.sum().item())
        loss_sum += float(loss.item()) * count
        tokens += count
    return loss_sum / tokens


def learning_rate(step, steps, lr, min_lr, warmup_steps):
    if step <= warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, steps - warmup_steps)
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    parser = argparse.ArgumentParser(description="Train a NumPy/CuPy GPT; --steps is the total schedule length.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop-after", type=int, help="stop and save at this absolute step without changing the LR schedule")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--init-from", help="load weights with a new optimizer and training schedule")
    initial.add_argument("--resume", help="restore weights, optimizer, batch RNG and the original schedule")
    parser.add_argument("--min-mask-frac", type=float, default=0.15)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "eval_every", "eval_iters", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0 or args.sample_tokens < 0 or args.seed < 0:
        parser.error("warmup-steps, sample-tokens and seed must be nonnegative")
    if not all(math.isfinite(x) for x in (args.lr, args.min_lr, args.weight_decay, args.min_mask_frac)):
        parser.error("learning rates, weight decay and mask fraction must be finite")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.min_mask_frac <= 1:
        parser.error("require 0 <= min-lr <= lr, lr > 0, weight-decay >= 0 and 0 <= min-mask-frac <= 1")
    stop = args.steps if args.stop_after is None else args.stop_after
    if not 1 <= stop <= args.steps:
        parser.error("stop-after must be between 1 and steps")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    fingerprint = tokenizer_fingerprint(tokenizer)
    if meta["vocab_size"] != tokenizer.vocab_size or meta.get("tokenizer_fingerprint", fingerprint) != fingerprint:
        raise ValueError("dataset metadata does not match tokenizer")
    train_data = load_tokens(data_dir / "train.bin", meta["dtype"])
    val_data = load_tokens(data_dir / "val.bin", meta["dtype"])
    train_mask = val_mask = None
    if meta.get("has_loss_mask") or meta.get("assistant_loss_only"):
        train_mask = load_tokens(data_dir / "train_mask.bin", "uint8")
        val_mask = load_tokens(data_dir / "val_mask.bin", "uint8")
    for name, data in (("train", train_data), ("val", val_data)):
        if not len(data) or int(data.max()) >= tokenizer.vocab_size or int(data.min()) < 0:
            raise ValueError(f"{name}: empty data or token IDs outside vocabulary")
        if meta.get(f"{name}_tokens", len(data)) != len(data):
            raise ValueError(f"{name}: token count does not match metadata")
    digest = hashlib.sha256()
    for data in (train_data, val_data, train_mask, val_mask):
        if data is None:
            digest.update(b"none")
        else:
            digest.update(str((data.shape, data.dtype.str)).encode())
            digest.update(memoryview(data))
    training_config = {k: getattr(args, k) for k in
                       ("steps", "batch_size", "lr", "min_lr", "warmup_steps", "weight_decay",
                        "min_mask_frac", "seed", "eval_iters", "eval_every")}
    training_config["data_sha256"] = digest.hexdigest()
    training_config["device"] = args.device
    requested = {k: getattr(args, k) for k in ("seq_len", "d_model", "n_layers", "n_heads", "d_ff")}
    if args.resume:
        model, opt, start, best_val, rng = load_training(args.resume, tokenizer, training_config, args.device)
        for name, value in requested.items():
            if value is not None and value != model.config[name]:
                raise ValueError(f"resume {name} does not match checkpoint")
    else:
        if args.init_from:
            check_tokenizer(args.init_from, tokenizer)
            model = GPT.load(args.init_from, vocab_size=tokenizer.vocab_size, device=args.device, **requested)
        else:
            defaults = dict(seq_len=meta.get("seq_len", 128), d_model=128, n_layers=3, n_heads=4, d_ff=None)
            dimensions = {k: defaults[k] if v is None else v for k, v in requested.items()}
            model = GPT(vocab_size=tokenizer.vocab_size, seed=args.seed, device=args.device, **dimensions)
        opt = AdamW(model.params, lr=args.lr, weight_decay=args.weight_decay)
        start, best_val, rng = 0, float("inf"), np.random.default_rng(args.seed)
    if stop <= start:
        parser.error(f"checkpoint is already at step {start}; stop-after/steps must be greater")
    sampler = BatchSampler(train_data, model.seq_len, train_mask, args.min_mask_frac)
    BatchSampler(val_data, model.seq_len, val_mask)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"params={model.num_params():,} train_tokens={len(train_data):,} val_tokens={len(val_data):,}")
    t0 = time.monotonic()
    for step in range(start + 1, stop + 1):
        opt.lr = learning_rate(step, args.steps, args.lr, args.min_lr, args.warmup_steps)
        xb, yb, mb = sampler.batch(args.batch_size, model.xp, rng)
        loss, dlogits, cache = model.forward(xb, yb, mb)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("non-finite loss; optimizer update cancelled")
        grads = model.backward(dlogits, cache)
        grad_norm = clip_grad_norm(grads, 1.0)
        opt.step(grads)
        if step == start + 1 or step % 10 == 0:
            elapsed = max(time.monotonic() - t0, 1e-9)
            toks = (step - start) * args.batch_size * model.seq_len
            print(f"step {step:5d} loss {float(loss.item()):.4f} lr {opt.lr:.2e} grad {grad_norm:.3f} tok/s {toks/elapsed:.0f}", flush=True)
        # A temporary stop does not add evaluation events or change best_val.
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_data, args.batch_size, model.seq_len, args.eval_iters, val_mask, args.seed)
            if not math.isfinite(val_loss):
                raise FloatingPointError("non-finite validation loss")
            print(f"eval step {step:5d} val_loss {val_loss:.4f} ppl {math.exp(min(20.0, val_loss)):.2f}", flush=True)
            if args.sample_tokens:
                print(generate(model, tokenizer, "用户：你好。\n助手：", args.sample_tokens,
                               rng=np.random.default_rng(args.seed)), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_training(ckpt_dir / "best.npz", model, opt, step, best_val, rng, tokenizer, training_config)
                print(f"saved {ckpt_dir / 'best.npz'}", flush=True)
        if step % args.save_every == 0 or step == stop:
            save_training(ckpt_dir / "latest.npz", model, opt, step, best_val, rng, tokenizer, training_config)
            print(f"saved {ckpt_dir / 'latest.npz'}", flush=True)


if __name__ == "__main__":
    main()
