"""Optional real-device parity check; skipped when CUDA is unavailable."""
import unittest

import numpy as np

from gpt_from_scratch.model import GPT, cp


def cuda_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(cuda_available(), "requires CuPy and a working CUDA device")
class CUDATests(unittest.TestCase):
    def test_cpu_cuda_forward_backward_and_update(self):
        from gpt_from_scratch.optim import AdamW
        cpu = GPT(9, 4, d_model=8, n_layers=2, n_heads=2, device="cpu")
        gpu = GPT(9, 4, d_model=8, n_layers=2, n_heads=2, device="cuda")
        x = np.array([[1, 2, 1, 3], [2, 3, 4, 2]])
        y = np.array([[2, 1, 3, 4], [3, 4, 2, 1]])
        mask = np.array([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
        a, ad, ac = cpu.forward(x, y, mask)
        b, bd, bc = gpu.forward(cp.asarray(x), cp.asarray(y), cp.asarray(mask))
        np.testing.assert_allclose(a, cp.asnumpy(b), rtol=2e-5, atol=2e-6)
        ag, bg = cpu.backward(ad, ac), gpu.backward(bd, bc)
        for key in ag:
            np.testing.assert_allclose(ag[key], cp.asnumpy(bg[key]), rtol=3e-4, atol=2e-6, err_msg=key)
        AdamW(cpu.params).step(ag)
        AdamW(gpu.params).step(bg)
        for key in cpu.params:
            np.testing.assert_allclose(cpu.params[key], cp.asnumpy(gpu.params[key]), rtol=3e-4, atol=3e-5, err_msg=key)


if __name__ == "__main__":
    unittest.main()
