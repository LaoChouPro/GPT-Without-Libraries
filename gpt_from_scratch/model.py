import math
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


def get_xp(device="cuda"):
    if device == "cuda":
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        return cp
    if device == "cpu":
        return np
    raise ValueError(f"unknown device: {device}")


def to_numpy(value):
    return cp.asnumpy(value) if cp is not None and isinstance(value, cp.ndarray) else np.asarray(value)


def atomic_savez(path, arrays):
    """Never replace a usable checkpoint with a partially written archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temporary = f.name
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def cross_entropy(logits, targets, loss_mask=None):
    """Stable log-sum-exp loss and its exact logit gradient."""
    xp = np_or_cp(logits)
    if targets.shape != logits.shape[:-1] or targets.dtype.kind not in "iu":
        raise ValueError("targets must be integer token IDs with shape logits.shape[:-1]")
    if bool(xp.any((targets < 0) | (targets >= logits.shape[-1])).item()):
        raise ValueError("target token ID outside vocabulary")
    shifted = logits - xp.max(logits, axis=-1, keepdims=True)
    exp_logits = xp.exp(shifted)
    sums = xp.sum(exp_logits, axis=-1, keepdims=True)
    flat = shifted.reshape(-1, logits.shape[-1])
    rows = xp.arange(targets.size)
    token_losses = xp.log(sums.reshape(-1)) - flat[rows, targets.reshape(-1)]
    mask = xp.ones(targets.size, dtype=logits.dtype)
    if loss_mask is not None:
        if loss_mask.shape != targets.shape:
            raise ValueError("loss_mask must have the same shape as targets")
        mask = xp.asarray(loss_mask, dtype=logits.dtype).reshape(-1)
        if not bool(xp.all(xp.isfinite(mask) & (mask >= 0)).item()):
            raise ValueError("loss_mask must be finite and nonnegative")
    denom = xp.sum(mask)
    if float(denom.item()) <= 0:
        raise ValueError("loss_mask contains no supervised tokens")
    loss = xp.sum(token_losses * mask) / denom
    grad = (exp_logits / sums).reshape(-1, logits.shape[-1])
    grad[rows, targets.reshape(-1)] -= 1
    grad *= mask[:, None] / denom
    return loss, grad.reshape(logits.shape)


def gelu(x):
    return 0.5 * x * (1.0 + np_or_cp(x).tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def gelu_backward(x):
    xp = np_or_cp(x)
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x**3)
    t = xp.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x**2)


def np_or_cp(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np


def softmax(x, axis=-1):
    xp = np_or_cp(x)
    x = x - xp.max(x, axis=axis, keepdims=True)
    ex = xp.exp(x)
    return ex / xp.sum(ex, axis=axis, keepdims=True)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    xp = np_or_cp(x)
    mean = xp.mean(x, axis=-1, keepdims=True)
    var = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv = 1.0 / xp.sqrt(var + eps)
    xhat = (x - mean) * inv
    out = xhat * gamma + beta
    return out, (xhat, inv, gamma)


def layernorm_backward(dout, cache):
    xp = np_or_cp(dout)
    xhat, inv, gamma = cache
    n = dout.shape[-1]
    dxhat = dout * gamma
    dx = (1.0 / n) * inv * (
        n * dxhat - xp.sum(dxhat, axis=-1, keepdims=True) - xhat * xp.sum(dxhat * xhat, axis=-1, keepdims=True)
    )
    dgamma = xp.sum(dout * xhat, axis=tuple(range(dout.ndim - 1)))
    dbeta = xp.sum(dout, axis=tuple(range(dout.ndim - 1)))
    return dx, dgamma, dbeta


def linear_forward(x, w, b):
    out = x.reshape(-1, x.shape[-1]) @ w
    out = out.reshape(*x.shape[:-1], w.shape[1])
    if b is not None:
        out = out + b
    return out, (x, w)


def linear_backward(dout, cache):
    xp = np_or_cp(dout)
    x, w = cache
    xr = x.reshape(-1, x.shape[-1])
    dor = dout.reshape(-1, dout.shape[-1])
    dx = (dor @ w.T).reshape(x.shape)
    dw = xr.T @ dor
    db = xp.sum(dor, axis=0)
    return dx, dw, db


class GPT:
    def __init__(self, vocab_size, seq_len, d_model=128, n_layers=3, n_heads=4, d_ff=None, seed=1337, device="cuda"):
        dimensions = dict(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model,
                          n_layers=n_layers, n_heads=n_heads, d_ff=d_ff if d_ff is not None else 4 * d_model)
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.xp = get_xp(device)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff or 4 * d_model
        self.device = device
        self.params = {}
        self.config = {name: int(value) for name, value in dimensions.items()}
        self.causal_mask = self.xp.triu(self.xp.ones((seq_len, seq_len), dtype=bool), 1)
        rng = np.random.default_rng(seed)

        def randn(shape, scale):
            return self.xp.asarray(rng.normal(0.0, scale, size=shape).astype(np.float32))

        self.params["tok_emb"] = randn((vocab_size, d_model), 0.02)
        self.params["pos_emb"] = randn((seq_len, d_model), 0.01)
        for l in range(n_layers):
            prefix = f"blocks.{l}."
            self.params[prefix + "ln1_g"] = self.xp.ones((d_model,), dtype=self.xp.float32)
            self.params[prefix + "ln1_b"] = self.xp.zeros((d_model,), dtype=self.xp.float32)
            self.params[prefix + "qkv_w"] = randn((d_model, 3 * d_model), 0.02 / math.sqrt(2 * n_layers))
            self.params[prefix + "qkv_b"] = self.xp.zeros((3 * d_model,), dtype=self.xp.float32)
            self.params[prefix + "proj_w"] = randn((d_model, d_model), 0.02 / math.sqrt(2 * n_layers))
            self.params[prefix + "proj_b"] = self.xp.zeros((d_model,), dtype=self.xp.float32)
            self.params[prefix + "ln2_g"] = self.xp.ones((d_model,), dtype=self.xp.float32)
            self.params[prefix + "ln2_b"] = self.xp.zeros((d_model,), dtype=self.xp.float32)
            self.params[prefix + "fc_w"] = randn((d_model, self.d_ff), 0.02)
            self.params[prefix + "fc_b"] = self.xp.zeros((self.d_ff,), dtype=self.xp.float32)
            self.params[prefix + "ff_w"] = randn((self.d_ff, d_model), 0.02 / math.sqrt(2 * n_layers))
            self.params[prefix + "ff_b"] = self.xp.zeros((d_model,), dtype=self.xp.float32)
        self.params["ln_f_g"] = self.xp.ones((d_model,), dtype=self.xp.float32)
        self.params["ln_f_b"] = self.xp.zeros((d_model,), dtype=self.xp.float32)

    def num_params(self):
        return int(sum(p.size for p in self.params.values()))

    def forward(self, idx, targets=None, loss_mask=None):
        xp = self.xp
        if idx.ndim != 2 or min(idx.shape) == 0 or idx.dtype.kind not in "iu":
            raise ValueError("idx must be a nonempty 2D integer array")
        if bool(xp.any((idx < 0) | (idx >= self.vocab_size)).item()):
            raise ValueError("input token ID outside vocabulary")
        bsz, tsz = idx.shape
        if tsz > self.seq_len:
            raise ValueError(f"sequence length {tsz} exceeds model limit {self.seq_len}")
        caches = {"idx": idx, "blocks": []}
        h = self.params["tok_emb"][idx] + self.params["pos_emb"][xp.arange(tsz)][None, :, :]
        caches["embed_h_shape"] = h.shape

        for l in range(self.n_layers):
            prefix = f"blocks.{l}."
            block_cache = {}
            ln1, block_cache["ln1"] = layernorm_forward(h, self.params[prefix + "ln1_g"], self.params[prefix + "ln1_b"])
            qkv, block_cache["qkv_linear"] = linear_forward(ln1, self.params[prefix + "qkv_w"], self.params[prefix + "qkv_b"])
            q, k, v = xp.split(qkv, 3, axis=-1)
            q = q.reshape(bsz, tsz, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, tsz, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, tsz, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:tsz, :tsz]
            scores = xp.where(mask[None, None, :, :], -xp.inf, scores)
            att = softmax(scores, axis=-1)
            y = att @ v
            y_merge = y.transpose(0, 2, 1, 3).reshape(bsz, tsz, self.d_model)
            att_out, block_cache["proj_linear"] = linear_forward(y_merge, self.params[prefix + "proj_w"], self.params[prefix + "proj_b"])
            h = h + att_out
            block_cache.update({"q": q, "k": k, "v": v, "att": att, "y_merge_shape": y_merge.shape})

            ln2, block_cache["ln2"] = layernorm_forward(h, self.params[prefix + "ln2_g"], self.params[prefix + "ln2_b"])
            fc, block_cache["fc_linear"] = linear_forward(ln2, self.params[prefix + "fc_w"], self.params[prefix + "fc_b"])
            act = gelu(fc)
            ff, block_cache["ff_linear"] = linear_forward(act, self.params[prefix + "ff_w"], self.params[prefix + "ff_b"])
            h = h + ff
            block_cache["fc_pre_act"] = fc
            caches["blocks"].append(block_cache)

        h, caches["ln_f"] = layernorm_forward(h, self.params["ln_f_g"], self.params["ln_f_b"])
        logits = h @ self.params["tok_emb"].T
        caches["final_h"] = h
        if targets is None:
            return logits, caches

        loss, dlogits = cross_entropy(logits, targets, loss_mask)
        return loss, dlogits, caches

    def backward(self, dlogits, caches):
        xp = self.xp
        grads = {k: xp.zeros_like(v) for k, v in self.params.items()}
        h_final = caches["final_h"]
        grads["tok_emb"] += dlogits.reshape(-1, self.vocab_size).T @ h_final.reshape(-1, self.d_model)
        dh = dlogits @ self.params["tok_emb"]
        dh, grads["ln_f_g"], grads["ln_f_b"] = layernorm_backward(dh, caches["ln_f"])

        for l in reversed(range(self.n_layers)):
            prefix = f"blocks.{l}."
            bc = caches["blocks"][l]

            dff = dh
            dh = dh.copy()
            dact, grads[prefix + "ff_w"], grads[prefix + "ff_b"] = linear_backward(dff, bc["ff_linear"])
            dfc = dact * gelu_backward(bc["fc_pre_act"])
            dln2, grads[prefix + "fc_w"], grads[prefix + "fc_b"] = linear_backward(dfc, bc["fc_linear"])
            dres2, grads[prefix + "ln2_g"], grads[prefix + "ln2_b"] = layernorm_backward(dln2, bc["ln2"])
            dh += dres2

            datto = dh
            dh = dh.copy()
            dy_merge, grads[prefix + "proj_w"], grads[prefix + "proj_b"] = linear_backward(datto, bc["proj_linear"])
            bsz, tsz, _ = dy_merge.shape
            dy = dy_merge.reshape(bsz, tsz, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            q, k, v, att = bc["q"], bc["k"], bc["v"], bc["att"]
            datt = dy @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ dy
            ds = att * (datt - xp.sum(datt * att, axis=-1, keepdims=True))
            ds /= math.sqrt(self.head_dim)
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q
            dq = dq.transpose(0, 2, 1, 3).reshape(bsz, tsz, self.d_model)
            dk = dk.transpose(0, 2, 1, 3).reshape(bsz, tsz, self.d_model)
            dv = dv.transpose(0, 2, 1, 3).reshape(bsz, tsz, self.d_model)
            dqkv = xp.concatenate([dq, dk, dv], axis=-1)
            dln1, grads[prefix + "qkv_w"], grads[prefix + "qkv_b"] = linear_backward(dqkv, bc["qkv_linear"])
            dres1, grads[prefix + "ln1_g"], grads[prefix + "ln1_b"] = layernorm_backward(dln1, bc["ln1"])
            dh += dres1

        idx = caches["idx"]
        flat_idx = idx.reshape(-1)
        flat_dh = dh.reshape(-1, self.d_model)
        xp.add.at(grads["tok_emb"], flat_idx, flat_dh)
        dpos = xp.sum(dh, axis=0)
        grads["pos_emb"][: dpos.shape[0]] += dpos
        return grads

    def save(self, path, extra=None):
        arrays = {k: to_numpy(v) for k, v in self.params.items()}
        metadata = dict(extra or {})
        for k, v in self.config.items():
            if k in metadata and metadata[k] != v:
                raise ValueError(f"checkpoint metadata disagrees with model: {k}")
            metadata[k] = v
        arrays.update({"extra." + k: np.asarray(v) for k, v in metadata.items()})
        atomic_savez(path, arrays)

    @classmethod
    def load(cls, path, vocab_size=None, seq_len=None, d_model=None, n_layers=None, n_heads=None, d_ff=None, device="cuda"):
        requested = dict(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model,
                         n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
        with np.load(path, allow_pickle=False) as data:
            config = {}
            for k, value in requested.items():
                stored = int(data["extra." + k]) if "extra." + k in data else None
                if value is not None and stored is not None and value != stored:
                    raise ValueError(f"checkpoint {k}={stored}, requested {value}")
                config[k] = stored if value is None else value
            if config["d_ff"] is None and config["d_model"] is not None:
                config["d_ff"] = 4 * config["d_model"]
            if any(v is None for v in config.values()):
                raise ValueError("legacy checkpoint lacks model configuration; supply dimensions explicitly")
            model = cls(**config, device=device)
            for k, param in model.params.items():
                if k not in data or data[k].shape != param.shape:
                    raise ValueError(f"checkpoint parameter missing or wrong shape: {k}")
                if data[k].dtype.kind != "f" or not np.isfinite(data[k]).all():
                    raise ValueError(f"checkpoint parameter is not finite floating-point data: {k}")
                param[...] = model.xp.asarray(data[k])
        return model
