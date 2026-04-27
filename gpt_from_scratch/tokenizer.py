import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class CharTokenizer:
    def __init__(self, stoi):
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
        self.stoi = dict(stoi)
        self.itos = [None] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.max_token_len = max_token_len or max(len(t) for t in self.stoi if t not in SPECIAL_TOKENS)

    @classmethod
    def build(cls, texts, vocab_size, max_ngram=6, train_chars=8_000_000, char_vocab=None, min_freq=3):
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
                if idx is not None:
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
    parts = []
    for msg in obj.get("conversations", []):
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"用户：{content}\n")
        elif role == "assistant":
            parts.append(f"助手：{content}\n")
        else:
            parts.append(f"{role}：{content}\n")
    return "".join(parts).strip() + "\n"


def format_conversation_segments(obj):
    segments = []
    for msg in obj.get("conversations", []):
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
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
