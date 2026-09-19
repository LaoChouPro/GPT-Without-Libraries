"""Shared autoregressive generation for the CLI and training previews."""
import math

import numpy as np

from .model import softmax, to_numpy


def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=40, rng=None):
    if max_new_tokens < 0 or top_k < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens/top_k/temperature must be nonnegative and temperature finite")
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match model")
    rng = np.random.default_rng() if rng is None else rng
    xp = model.xp
    ids = tokenizer.encode(prompt, add_bos=True)
    generated = []
    for _ in range(max_new_tokens):
        x = xp.asarray(ids[-model.seq_len:], dtype=xp.int64)[None, :]
        logits, _ = model.forward(x)
        next_logits = logits[0, -1].copy()
        # These are structural/unknown tokens, never visible assistant output.
        next_logits[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = -xp.inf
        if temperature == 0:
            next_id = int(xp.argmax(next_logits).item())
        else:
            next_logits /= temperature
            if 0 < top_k < model.vocab_size:
                top_idx = xp.argsort(next_logits)[-top_k:]
                filtered = xp.full_like(next_logits, -xp.inf)
                filtered[top_idx] = next_logits[top_idx]
                next_logits = filtered
            probs = to_numpy(softmax(next_logits)).astype(np.float64)
            if not np.isfinite(probs).all() or probs.sum() <= 0:
                raise FloatingPointError("generation produced invalid probabilities")
            probs /= probs.sum()
            next_id = int(rng.choice(model.vocab_size, p=probs))
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)
    # Preserve prompt characters even when the tokenizer cannot encode them.
    return prompt + tokenizer.decode(generated)
