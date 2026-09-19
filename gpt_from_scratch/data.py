"""JSONL validation, document-level splitting and supervised batch sampling."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tokenizer import format_conversation_segments


def read_documents(path, max_docs=None):
    documents = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if max_docs is not None and len(documents) >= max_docs:
                break
            try:
                segments = format_conversation_segments(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if segments:
                documents.append(segments)
    if not documents:
        raise ValueError("No usable conversations found")
    return documents


def split_documents(documents, val_frac=0.1, seed=1337):
    """Keep identical rendered conversations in one split, including repeats."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    groups = {}
    for doc in documents:
        # Use rendered text: different role labels can render identically.
        key = hashlib.sha256("".join(text for _, text in doc).encode()).hexdigest()
        groups.setdefault(key, []).append(doc)
    if len(groups) < 2:
        raise ValueError("At least two distinct conversations are needed for train/validation")
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_frac)))
    val = [doc for key in keys[:n_val] for doc in groups[key]]
    train = [doc for key in keys[n_val:] for doc in groups[key]]
    return train, val


def encode_documents(documents, tokenizer, assistant_only=False):
    ids, masks = [], []
    for segments in documents:
        if assistant_only and not any(kind == "assistant" for kind, _ in segments):
            raise ValueError("assistant-only training requires an assistant answer in every conversation")
        ids.append(tokenizer.bos_id)
        masks.append(0)  # Do not train prediction of a new document's BOS.
        for kind, text in segments:
            encoded = tokenizer.encode(text)
            ids.extend(encoded)
            masks.extend([int(not assistant_only or kind == "assistant")] * len(encoded))
        ids.append(tokenizer.eos_id)
        masks.append(int(not assistant_only or segments[-1][0] == "assistant"))
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    return np.asarray(ids, dtype=dtype), np.asarray(masks, dtype=np.uint8)


def load_tokens(path, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu" or Path(path).stat().st_size % dtype.itemsize:
        raise ValueError(f"invalid token file or dtype: {path}")
    return np.fromfile(path, dtype=dtype)


class BatchSampler:
    """Precompute valid starts once; never fall back to unsupervised windows."""
    def __init__(self, data, seq_len, mask_data=None, min_mask_frac=0.0):
        if seq_len < 1 or data.ndim != 1 or len(data) < seq_len + 1:
            raise ValueError(f"need at least seq_len + 1 ({seq_len + 1}) tokens")
        if not math.isfinite(min_mask_frac) or not 0 <= min_mask_frac <= 1:
            raise ValueError("min_mask_frac must be between 0 and 1")
        self.data, self.seq_len, self.mask = data, seq_len, mask_data
        self.starts = None
        if mask_data is not None:
            if mask_data.shape != data.shape or not np.isin(mask_data, [0, 1]).all():
                raise ValueError("mask must contain one binary value per token")
            prefix = np.concatenate(([0], np.cumsum(mask_data, dtype=np.int64)))
            counts = prefix[seq_len + 1:] - prefix[1:len(data) - seq_len + 1]
            minimum = max(1, math.ceil(seq_len * min_mask_frac))
            self.starts = np.flatnonzero(counts >= minimum)
            if not self.starts.size:
                raise ValueError("no windows meet min_mask_frac; lower it or add assistant answers")

    def batch(self, batch_size, xp=np, rng=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng() if rng is None else rng
        if self.starts is None:
            starts = rng.integers(len(self.data) - self.seq_len, size=batch_size)
        else:
            starts = rng.choice(self.starts, size=batch_size)
        offsets = starts[:, None] + np.arange(self.seq_len)
        x = xp.asarray(self.data[offsets], dtype=xp.int64)
        y = xp.asarray(self.data[offsets + 1], dtype=xp.int64)
        mask = None if self.mask is None else xp.asarray(self.mask[offsets + 1], dtype=xp.float32)
        return x, y, mask
