import hashlib
import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def validate_vocabulary(stoi):
    if not isinstance(stoi, dict) or not all(t in stoi for t in SPECIAL_TOKENS):
        raise ValueError("vocabulary must include all four special tokens")
    if any(not isinstance(t, str) or not t for t in stoi):
        raise ValueError("vocabulary tokens must be nonempty strings")
    if any(type(i) is not int for i in stoi.values()) or set(stoi.values()) != set(range(len(stoi))):
        raise ValueError("vocabulary IDs must be unique contiguous integers from zero")


def tokenizer_fingerprint(tokenizer):
    payload = {"type": "subword" if isinstance(tokenizer, SubwordTokenizer) else "char",
               "stoi": tokenizer.stoi}
    if isinstance(tokenizer, SubwordTokenizer):
        payload["max_token_len"] = tokenizer.max_token_len
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CharTokenizer:
    def __init__(self, stoi):
        validate_vocabulary(stoi)
        self.stoi = dict(stoi)
        self.itos = [None] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @classmethod
    def build(cls, texts, vocab_size):
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must be at least 4")
        counter = Counter()
        for text in texts:
            counter.update(text)
        stoi = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for ch, _ in counter.most_common(max(0, vocab_size - len(stoi))):
            if ch not in stoi:
                stoi[ch] = len(stoi)
        return cls(stoi)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["stoi"])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"type": "char", "stoi": self.stoi}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        out = []
        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return "".join(out)


class SubwordTokenizer:
    def __init__(self, stoi, max_token_len=None):
        validate_vocabulary(stoi)
        self.stoi = dict(stoi)
        self.itos = [None] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        longest = max((len(t) for t in self.stoi if t not in SPECIAL_TOKENS), default=1)
        self.max_token_len = longest if max_token_len is None else max_token_len
        if type(self.max_token_len) is not int or self.max_token_len < longest:
            raise ValueError("max_token_len must cover every ordinary vocabulary token")

    @classmethod
    def build(cls, texts, vocab_size, max_ngram=6, train_chars=8_000_000, char_vocab=None, min_freq=3):
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must be at least 4")
        if max_ngram < 1 or train_chars < 1 or min_freq < 1 or (char_vocab is not None and char_vocab < 1):
            raise ValueError("subword training limits must be positive")
        if vocab_size == len(SPECIAL_TOKENS):
            return cls({token: i for i, token in enumerate(SPECIAL_TOKENS)}, max_ngram)
        char_counter = Counter()
        ngram_counter = Counter()
        seen_chars = 0
        for text in texts:
            if seen_chars >= train_chars:
                break
            if seen_chars + len(text) > train_chars:
                text = text[: train_chars - seen_chars]
            seen_chars += len(text)
            char_counter.update(text)
            text_len = len(text)
            for n in range(2, max_ngram + 1):
                if text_len < n:
                    break
                for i in range(text_len - n + 1):
                    token = text[i : i + n]
                    if "\n" in token:
                        continue
                    ngram_counter[token] += 1

        stoi = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        char_limit = char_vocab or max(512, min(vocab_size // 2, vocab_size - len(stoi)))
        base_chars = set()
        for ch, _ in char_counter.most_common(char_limit):
            if ch not in stoi:
                stoi[ch] = len(stoi)
                base_chars.add(ch)
            if len(stoi) >= vocab_size:
                return cls(stoi, max_ngram)

        scored = []
        for token, freq in ngram_counter.items():
            if freq < min_freq:
                continue
            if any(ch not in base_chars for ch in token):
                continue
            # Longer repeated chunks reduce context length more than pairs.
            scored.append((freq * (len(token) - 1), freq, len(token), token))
        scored.sort(reverse=True)
        for _, _, _, token in scored:
            if token not in stoi:
                stoi[token] = len(stoi)
                if len(stoi) >= vocab_size:
                    break
        return cls(stoi, max_ngram)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["stoi"], data.get("max_token_len"))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {"type": "subword", "stoi": self.stoi, "max_token_len": self.max_token_len},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        i = 0
        while i < len(text):
            found = None
            max_j = min(len(text), i + self.max_token_len)
            for j in range(max_j, i, -1):
                token = text[i:j]
                idx = self.stoi.get(token)
                if idx is not None and token not in SPECIAL_TOKENS:
                    found = (idx, j)
                    break
            if found is None:
                ids.append(self.unk_id)
                i += 1
            else:
                idx, i = found
                ids.append(idx)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        out = []
        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return "".join(out)


def load_tokenizer(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tok_type = data.get("type", "char")
    if tok_type == "subword":
        return SubwordTokenizer(data["stoi"], data.get("max_token_len"))
    if tok_type == "char":
        return CharTokenizer(data["stoi"])
    raise ValueError(f"unknown tokenizer type: {tok_type}")


def format_conversation(obj):
    return "".join(segment for _, segment in format_conversation_segments(obj))


def format_conversation_segments(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("conversations"), list):
        raise ValueError("each record must contain a conversations list")
    segments = []
    for msg in obj["conversations"]:
        if not isinstance(msg, dict):
            raise ValueError("conversation messages must be objects")
        role, content = msg.get("role"), msg.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
            raise ValueError("messages require a nonempty role and string content")
        content = content.strip()
        if not content:
            continue
        if role == "user":
            segments.append(("prompt", f"用户：{content}\n"))
        elif role == "assistant":
            segments.append(("prompt", "助手："))
            segments.append(("assistant", f"{content}\n"))
        else:
            segments.append(("prompt", f"{role}：{content}\n"))
    return segments
