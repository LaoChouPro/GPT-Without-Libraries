import argparse
import json
from pathlib import Path

import numpy as np

from gpt_from_scratch.tokenizer import CharTokenizer, SubwordTokenizer, format_conversation, format_conversation_segments, load_tokenizer


def iter_texts(path, max_docs=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = format_conversation(obj)
            if text.strip():
                yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-docs", type=int, default=50000)
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--tokenizer-in", default=None)
    parser.add_argument("--tokenizer-type", choices=["char", "subword"], default="char")
    parser.add_argument("--subword-max-ngram", type=int, default=6)
    parser.add_argument("--subword-train-chars", type=int, default=8000000)
    parser.add_argument("--subword-char-vocab", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--assistant-loss-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = list(iter_texts(args.input, args.max_docs))
    if not texts:
        raise RuntimeError("No usable texts found.")

    if args.tokenizer_in:
        tokenizer = load_tokenizer(args.tokenizer_in)
    elif args.tokenizer_type == "subword":
        tokenizer = SubwordTokenizer.build(
            texts,
            args.vocab_size,
            max_ngram=args.subword_max_ngram,
            train_chars=args.subword_train_chars,
            char_vocab=args.subword_char_vocab,
        )
    else:
        tokenizer = CharTokenizer.build(texts, args.vocab_size)
    tokenizer.save(out_dir / "tokenizer.json")

    ids = []
    masks = []
    if args.assistant_loss_only:
        with open(args.input, "r", encoding="utf-8") as f:
            for doc_i, line in enumerate(f):
                if args.max_docs is not None and doc_i >= args.max_docs:
                    break
                obj = json.loads(line)
                ids.append(tokenizer.bos_id)
                masks.append(0)
                for kind, segment in format_conversation_segments(obj):
                    seg_ids = tokenizer.encode(segment)
                    ids.extend(seg_ids)
                    masks.extend([1 if kind == "assistant" else 0] * len(seg_ids))
                ids.append(tokenizer.eos_id)
                masks.append(1)
    else:
        for text in texts:
            encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
            ids.extend(encoded)
            masks.extend([1] * len(encoded))

    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    data = np.asarray(ids, dtype=dtype)
    split = max(args.seq_len + 2, int(len(data) * (1.0 - args.val_frac)))
    split = min(split, len(data) - args.seq_len - 2)
    train = data[:split]
    val = data[split:]
    mask_data = np.asarray(masks, dtype=np.uint8)
    train_mask = mask_data[:split]
    val_mask = mask_data[split:]

    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    train_mask.tofile(out_dir / "train_mask.bin")
    val_mask.tofile(out_dir / "val_mask.bin")
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(np.dtype(dtype)),
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "max_docs": args.max_docs,
        "seq_len": args.seq_len,
        "assistant_loss_only": bool(args.assistant_loss_only),
        "tokenizer_type": args.tokenizer_type if not args.tokenizer_in else "loaded",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
