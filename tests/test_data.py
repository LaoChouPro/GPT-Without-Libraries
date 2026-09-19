import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpt_from_scratch.data import BatchSampler, read_documents, split_documents, encode_documents
from gpt_from_scratch.tokenizer import (CharTokenizer, SubwordTokenizer, SPECIAL_TOKENS,
                                      format_conversation_segments, load_tokenizer, tokenizer_fingerprint)


def conversation(question, answer):
    return {"conversations": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


class DataTests(unittest.TestCase):
    def test_minimal_batch_and_last_valid_start(self):
        data = np.arange(5)
        x, y, _ = BatchSampler(data, 4).batch(2)
        np.testing.assert_array_equal(x, [[0, 1, 2, 3]] * 2)
        np.testing.assert_array_equal(y, [[1, 2, 3, 4]] * 2)
        x, _, _ = BatchSampler(np.arange(6), 4).batch(100, rng=np.random.default_rng(1))
        self.assertEqual(set(x[:, 0]), {0, 1})

    def test_mask_selects_only_supervised_target_windows(self):
        data = np.arange(8)
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        x, y, m = BatchSampler(data, 3, mask, .3).batch(10)
        np.testing.assert_array_equal(x, [[4, 5, 6]] * 10)
        np.testing.assert_array_equal(y, [[5, 6, 7]] * 10)
        np.testing.assert_array_equal(m, [[0, 0, 1]] * 10)
        with self.assertRaisesRegex(ValueError, "no windows"):
            BatchSampler(data, 3, mask, .5)
        with self.assertRaises(ValueError):
            BatchSampler(data, 3, np.zeros(8))
        with self.assertRaises(ValueError):
            BatchSampler(data, 3, np.ones(7))

    def test_duplicate_documents_do_not_cross_splits(self):
        docs = [format_conversation_segments(conversation(str(i), str(i * 2))) for i in range(12)] * 5
        train, val = split_documents(docs, .25, 2)
        render = lambda ds: {"".join(s for _, s in d) for d in ds}
        self.assertFalse(render(train) & render(val))
        self.assertEqual(len(train) + len(val), len(docs))
        self.assertEqual((train, val), split_documents(docs, .25, 2))
        with self.assertRaises(ValueError):
            split_documents(docs[:1])

    def test_assistant_mask_and_eos(self):
        doc = format_conversation_segments(conversation("你好", "世界"))
        tok = CharTokenizer.build(["".join(s for _, s in doc)], 50)
        ids, mask = encode_documents([doc], tok, True)
        supervised = tok.decode(ids[mask == 1])
        self.assertEqual(supervised, "世界\n")
        self.assertEqual(mask[0], 0)
        self.assertEqual((ids[-1], mask[-1]), (tok.eos_id, 1))
        prompt_only = format_conversation_segments({"conversations": [{"role": "user", "content": "hi"}]})
        with self.assertRaisesRegex(ValueError, "assistant answer"):
            encode_documents([prompt_only], tok, True)
        _, mask = encode_documents([doc + prompt_only], tok, True)
        self.assertEqual(mask[-1], 0)

    def test_jsonl_blanks_and_bad_line_location(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "input.jsonl"
            path.write_text("\n" + json.dumps(conversation("a", "b")) + "\n\n", encoding="utf-8")
            self.assertEqual(len(read_documents(path)), 1)
            path.write_text("\n{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input.jsonl:2"):
                read_documents(path)

    def test_tokenizer_roundtrip_and_fingerprint(self):
        text = "你好，世界！hello hello\n"
        for cls in (CharTokenizer, SubwordTokenizer):
            tok = cls.build([text] * 4, 80)
            self.assertEqual(tok.decode(tok.encode(text, True, True)), text)
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "tok.json"
                tok.save(path)
                restored = load_tokenizer(path)
                self.assertEqual(restored.encode(text), tok.encode(text))
                self.assertEqual(tokenizer_fingerprint(restored), tokenizer_fingerprint(tok))
        tok = CharTokenizer.build(["ab"], 6)
        other = CharTokenizer(dict(tok.stoi, a=tok.stoi["b"], b=tok.stoi["a"]))
        self.assertNotEqual(tokenizer_fingerprint(tok), tokenizer_fingerprint(other))

    def test_invalid_vocabulary_and_small_subword(self):
        for cls in (CharTokenizer, SubwordTokenizer):
            with self.assertRaises(ValueError):
                cls.build(["abc"], 3)
            tok = cls.build([], 4)
            self.assertEqual(tok.vocab_size, 4)
            with self.assertRaises(ValueError):
                cls({t: 0 for t in SPECIAL_TOKENS})
        tok = SubwordTokenizer.build(["abc"], 4)
        self.assertEqual(tok.vocab_size, 4)
        tok = SubwordTokenizer.build(["<eos>hi"], 20)
        # User text must not inject the structural EOS token.
        self.assertNotIn(tok.eos_id, tok.encode("<eos>"))


if __name__ == "__main__":
    unittest.main()
