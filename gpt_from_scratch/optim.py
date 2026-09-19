import math

import numpy as np

from .model import np_or_cp, to_numpy


class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        if (not all(math.isfinite(x) for x in (lr, *betas, eps, weight_decay))
                or lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas)):
            raise ValueError("invalid AdamW hyperparameters")
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: None for k in params}
        self.v = {k: None for k in params}

    def step(self, grads):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for name, p in self.params.items():
            g = grads.get(name)
            if g is None:
                continue
            if self.m[name] is None:
                self.m[name] = g * 0
                self.v[name] = g * 0
            self.m[name] = b1 * self.m[name] + (1.0 - b1) * g
            self.v[name] = b2 * self.v[name] + (1.0 - b2) * (g * g)
            mhat = self.m[name] / (1.0 - b1**self.t)
            vhat = self.v[name] / (1.0 - b2**self.t)
            if self.weight_decay and p.ndim >= 2 and not name.endswith("emb"):
                p *= 1.0 - self.lr * self.weight_decay
            p -= self.lr * mhat / (vhat**0.5 + self.eps)

    def state_dict(self):
        state = {"optim.t": np.asarray(self.t), "optim.lr": np.asarray(self.lr),
                 "optim.beta1": np.asarray(self.beta1), "optim.beta2": np.asarray(self.beta2),
                 "optim.eps": np.asarray(self.eps), "optim.weight_decay": np.asarray(self.weight_decay)}
        for name in self.params:
            if self.m[name] is not None:
                state["optim.m." + name] = to_numpy(self.m[name])
                state["optim.v." + name] = to_numpy(self.v[name])
        return state

    def load_state_dict(self, state):
        lr, b1, b2, eps, wd = (float(state["optim." + key])
                               for key in ("lr", "beta1", "beta2", "eps", "weight_decay"))
        candidate = AdamW(self.params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
        candidate.t = int(state["optim.t"])
        if candidate.t < 0:
            raise ValueError("optimizer step must be nonnegative")
        for name, param in self.params.items():
            for kind in ("m", "v"):
                key = f"optim.{kind}.{name}"
                if key not in state:
                    if candidate.t:
                        raise ValueError(f"missing optimizer moment: {key}")
                    continue
                value = state[key]
                if value.shape != param.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                    raise ValueError(f"invalid optimizer moment: {key}")
                if kind == "v" and (value < 0).any():
                    raise ValueError(f"negative optimizer second moment: {key}")
                getattr(candidate, kind)[name] = np_or_cp(param).asarray(value).copy()
        self.__dict__.update(candidate.__dict__)
