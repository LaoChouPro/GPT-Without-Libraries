import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from gpt_from_scratch.model import GPT, cross_entropy, get_xp
from gpt_from_scratch.optim import AdamW
from train import clip_grad_norm, evaluate


class ModelTests(unittest.TestCase):
    def model(self):
        return GPT(7, 3, d_model=4, n_layers=2, n_heads=2, d_ff=5, device="cpu", seed=7)

    def test_all_parameter_gradients_finite_difference(self):
        model = self.model()
        model.params = {name: value.astype(np.float64) for name, value in model.params.items()}
        # Repeated token IDs exercise scatter-add plus tied output embeddings.
        x = np.array([[1, 1, 2], [3, 1, 3]])
        y = np.array([[2, 3, 4], [1, 4, 2]])
        mask = np.array([[0., 1., .5], [1., 0., 1.]])
        _, dl, cache = model.forward(x, y, mask)
        analytic = model.backward(dl, cache)
        eps = 1e-5
        for name, param in model.params.items():
            numerical = np.zeros_like(param)
            for index in np.ndindex(param.shape):
                original = param[index]
                param[index] = original + eps
                plus = float(model.forward(x, y, mask)[0])
                param[index] = original - eps
                minus = float(model.forward(x, y, mask)[0])
                param[index] = original
                numerical[index] = (plus - minus) / (2 * eps)
            with self.subTest(parameter=name):
                np.testing.assert_allclose(analytic[name], numerical, rtol=3e-4, atol=2e-7)

    def test_causal_attention_cannot_see_future(self):
        model = self.model()
        first = model.forward(np.array([[1, 2, 3]]))[0]
        second = model.forward(np.array([[1, 2, 6]]))[0]
        np.testing.assert_array_equal(first[:, :2], second[:, :2])
        prefix = model.forward(np.array([[1, 2]]))[0]
        np.testing.assert_allclose(first[:, :2], prefix, atol=1e-7)

    def test_extreme_cross_entropy_and_gradient(self):
        logits = np.array([[[1000., -1000.]]])
        loss, grad = cross_entropy(logits, np.array([[1]]))
        self.assertEqual(float(loss), 2000.)
        np.testing.assert_array_equal(grad, [[[1., -1.]]])
        loss2, _ = cross_entropy(logits + 10000, np.array([[1]]))
        self.assertEqual(float(loss2), float(loss))

    def test_masked_targets_do_not_change_loss_or_gradient(self):
        logits = np.array([[[1., 3.], [2., -1.]]])
        mask = np.array([[0., 1.]])
        a = cross_entropy(logits, np.array([[0, 1]]), mask)
        b = cross_entropy(logits, np.array([[1, 1]]), mask)
        self.assertEqual(float(a[0]), float(b[0]))
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[1][0, 0], [0, 0])

    def test_invalid_inputs_fail_early(self):
        for kw in ({"n_heads": 0}, {"n_layers": 0}, {"d_ff": 0}, {"d_model": 7, "n_heads": 2}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                GPT(8, 3, device="cpu", **kw)
        with self.assertRaises(ValueError):
            get_xp("gpu")
        model = self.model()
        for idx in (np.array([[-1]]), np.array([[7]]), np.array([[1.]]), np.empty((1, 0), dtype=int)):
            with self.assertRaises(ValueError):
                model.forward(idx)
        for mask in (np.zeros((1, 2)), np.array([[1., -1.]]), np.array([[1., np.nan]])):
            with self.assertRaises(ValueError):
                cross_entropy(np.zeros((1, 2, 7)), np.array([[0, 1]]), mask)
        with self.assertRaises(ValueError):
            cross_entropy(np.zeros((1, 2, 7)), np.array([[0, 7]]))

    def test_small_training_reduces_loss(self):
        model = self.model()
        opt = AdamW(model.params, lr=.015, weight_decay=0)
        x, y = np.array([[1, 2, 3]]), np.array([[2, 3, 4]])
        initial = float(model.forward(x, y)[0])
        for _ in range(80):
            loss, dl, cache = model.forward(x, y)
            grads = model.backward(dl, cache)
            clip_grad_norm(grads, 1.)
            opt.step(grads)
        final = float(model.forward(x, y)[0])
        self.assertLess(final, initial * .15)

    def test_checkpoint_roundtrip_and_architecture_validation(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "nested/model.npz"
            model.save(path)
            loaded = GPT.load(path, device="cpu")
            for key in model.params:
                np.testing.assert_array_equal(model.params[key], loaded.params[key])
            # Heads can be wrong while every weight shape is still identical.
            with self.assertRaisesRegex(ValueError, "n_heads"):
                GPT.load(path, n_heads=1, device="cpu")
            with np.load(path) as data:
                arrays = dict(data)
            arrays["ln_f_b"] = np.array([0.], dtype=np.float32)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "shape"):
                GPT.load(path, device="cpu")

    def test_atomic_checkpoint_failure_preserves_existing_file(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.npz"
            model.save(path)
            original = path.read_bytes()
            with patch("gpt_from_scratch.model.np.savez", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    model.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_evaluation_is_deterministic_and_does_not_consume_training_rng(self):
        model = self.model()
        data = np.tile(np.array([1, 2, 3, 4]), 5)
        mask = np.tile(np.array([0, 1, 1, 1]), 5)
        rng = np.random.default_rng(11)
        before = rng.bit_generator.state
        a = evaluate(model, data, 2, 3, 3, mask)
        b = evaluate(model, data, 2, 3, 3, mask)
        self.assertEqual(a, b)
        self.assertEqual(before, rng.bit_generator.state)

    def test_adamw_matches_independent_reference(self):
        params = {"w": np.array([[1., -2.]]), "tok_emb": np.array([[1., -2.]]), "bias": np.array([1., -2.])}
        opt = AdamW(params, lr=.01, betas=(.8, .9), weight_decay=.1)
        expected = {k: v.copy() for k, v in params.items()}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(p) for k, p in params.items()}
        for step in range(1, 4):
            grads = {k: np.full_like(p, .2 * step) for k, p in params.items()}
            for k in expected:
                m[k] = .8 * m[k] + .2 * grads[k]
                v[k] = .9 * v[k] + .1 * grads[k] ** 2
                decay = .999 if k == "w" else 1
                expected[k] = decay * expected[k] - .01 * (m[k] / (1 - .8 ** step)) / (np.sqrt(v[k] / (1 - .9 ** step)) + 1e-8)
            opt.step(grads)
        for k in params:
            np.testing.assert_allclose(params[k], expected[k], rtol=1e-12)

    def test_clip_rejects_nonfinite_and_handles_large_values(self):
        grads = {"w": np.array([3e20, 4e20], dtype=np.float32)}
        norm = clip_grad_norm(grads, 1.)
        self.assertTrue(np.isfinite(norm))
        np.testing.assert_allclose(grads["w"], [.6, .8], rtol=1e-6)
        with self.assertRaises(FloatingPointError):
            clip_grad_norm({"w": np.array([np.nan])}, 1.)


if __name__ == "__main__":
    unittest.main()
