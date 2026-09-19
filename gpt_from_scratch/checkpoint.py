"""Versioned training checkpoints; NumPy arrays and JSON only, no pickle."""
import json
import math
import warnings

import numpy as np

from .model import GPT, atomic_savez, to_numpy
from .optim import AdamW
from .tokenizer import tokenizer_fingerprint


def check_tokenizer(path, tokenizer):
    with np.load(path, allow_pickle=False) as data:
        if "extra.tokenizer_fingerprint" not in data:
            warnings.warn("Legacy checkpoint has no tokenizer fingerprint; use its original tokenizer.", stacklevel=2)
        elif str(data["extra.tokenizer_fingerprint"].item()) != tokenizer_fingerprint(tokenizer):
            raise ValueError("checkpoint/tokenizer mismatch (token IDs differ); use the original tokenizer")


def save_training(path, model, optimizer, step, best_val, rng, tokenizer, training_config):
    if step != optimizer.t:
        raise ValueError("training step does not match optimizer step")
    arrays = {k: to_numpy(v) for k, v in model.params.items()}
    metadata = dict(model.config, step=step, best_val=best_val, checkpoint_version=2,
                    tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
                    rng_state=json.dumps(rng.bit_generator.state),
                    training_config=json.dumps(training_config, sort_keys=True))
    arrays.update({"extra." + k: np.asarray(v) for k, v in metadata.items()})
    arrays.update(optimizer.state_dict())
    atomic_savez(path, arrays)


def load_training(path, tokenizer, training_config, device="cpu"):
    check_tokenizer(path, tokenizer)
    model = GPT.load(path, vocab_size=tokenizer.vocab_size, device=device)
    with np.load(path, allow_pickle=False) as data:
        if "extra.checkpoint_version" not in data or int(data["extra.checkpoint_version"]) != 2:
            raise ValueError("checkpoint has no resumable training state; use --init-from for fine-tuning")
        stored_config = json.loads(str(data["extra.training_config"].item()))
        if stored_config != training_config:
            changed = sorted(k for k in stored_config.keys() | training_config.keys()
                             if stored_config.get(k) != training_config.get(k))
            raise ValueError(f"resume requires the original training configuration/data; changed: {changed}")
        optimizer = AdamW(model.params)
        optimizer.load_state_dict(data)
        step, best_val = int(data["extra.step"]), float(data["extra.best_val"])
        if step != optimizer.t or step < 0 or math.isnan(best_val):
            raise ValueError("invalid training step or validation state")
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(data["extra.rng_state"].item()))
    return model, optimizer, step, best_val, rng
