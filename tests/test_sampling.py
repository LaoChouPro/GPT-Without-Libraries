import unittest

import numpy as np

from gpt_from_scratch.sampling import generate
from gpt_from_scratch.tokenizer import CharTokenizer


class FakeModel:
    xp = np
    seq_len = 3

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.vocab_size = len(logits)
        self.contexts = []

    def forward(self, x):
        self.contexts.append(x.copy())
        return np.broadcast_to(self.logits, (*x.shape, self.vocab_size)), {}


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer.build(["ab"], 6)

    def test_greedy_suppresses_specials_and_crops_context(self):
        model = FakeModel([100, 90, 80, -1, 5, 0])
        text = generate(model, self.tok, "未知🙂", 5, 0)
        self.assertEqual(text, "未知🙂aaaaa")
        self.assertTrue(all(x.shape[1] <= 3 for x in model.contexts))

    def test_eos_stops_immediately(self):
        model = FakeModel([0, 0, 0, 10, 1, 1])
        self.assertEqual(generate(model, self.tok, "a", 20, 0), "a")
        self.assertEqual(len(model.contexts), 1)

    def test_top_k_and_seed(self):
        model = FakeModel([100, 90, 80, -50, 3, 2])
        self.assertEqual(generate(model, self.tok, "", 10, .8, 1), "a" * 10)
        a = generate(model, self.tok, "", 20, .8, 0, np.random.default_rng(8))
        b = generate(model, self.tok, "", 20, .8, 100, np.random.default_rng(8))
        self.assertEqual(a, b)
        for temp, topk in ((-1, 1), (np.nan, 1), (1, -1)):
            with self.assertRaises(ValueError):
                generate(model, self.tok, "", 1, temp, topk)


if __name__ == "__main__":
    unittest.main()
