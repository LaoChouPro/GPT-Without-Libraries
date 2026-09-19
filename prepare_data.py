import argparse
import json
from pathlib import Path

from gpt_from_scratch.data import read_documents, split_documents, encode_documents, BatchSampler
from gpt_from_scratch.tokenizer import CharTokenizer, SubwordTokenizer, load_tokenizer, tokenizer_fingerprint


def main():
    parser = argparse.ArgumentParser(description="Split complete conversations before training the tokenizer.")
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--assistant-loss-only", action="store_true")
    args = parser.parse_args()
    if args.seq_len < 1 or args.max_docs < 1 or args.vocab_size < 4:
        parser.error("seq-len/max-docs must be positive and vocab-size at least 4")
    documents = read_documents(args.input, args.max_docs)
    train_docs, val_docs = split_documents(documents, args.val_frac, args.seed)
    texts = ["".join(text for _, text in doc) for doc in train_docs]
    if args.tokenizer_in:
        tokenizer = load_tokenizer(args.tokenizer_in)
    elif args.tokenizer_type == "subword":
        tokenizer = SubwordTokenizer.build(texts, args.vocab_size, max_ngram=args.subword_max_ngram,
                                          train_chars=args.subword_train_chars, char_vocab=args.subword_char_vocab)
    else:
        tokenizer = CharTokenizer.build(texts, args.vocab_size)
    encoded = {}
    for name, docs in (("train", train_docs), ("val", val_docs)):
        tokens, mask = encode_documents(docs, tokenizer, args.assistant_loss_only)
        try:
            BatchSampler(tokens, args.seq_len, mask)
        except ValueError as exc:
            raise ValueError(f"{name} split: {exc}; add data or reduce --seq-len") from exc
        encoded[name] = tokens, mask
    # Validate both splits before touching existing output files.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    for name, (tokens, mask) in encoded.items():
        tokens.tofile(out_dir / f"{name}.bin")
        mask.tofile(out_dir / f"{name}_mask.bin")
    meta = {
        "format_version": 2, "vocab_size": tokenizer.vocab_size,
        "dtype": str(encoded["train"][0].dtype),
        "train_tokens": len(encoded["train"][0]), "val_tokens": len(encoded["val"][0]),
        "train_docs": len(train_docs), "val_docs": len(val_docs), "max_docs": args.max_docs,
        "seq_len": args.seq_len, "assistant_loss_only": args.assistant_loss_only,
        "has_loss_mask": True, "tokenizer_type": "subword" if isinstance(tokenizer, SubwordTokenizer) else "char",
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "split": "shuffled identical-conversation groups", "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
