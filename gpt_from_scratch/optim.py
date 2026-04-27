class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
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
