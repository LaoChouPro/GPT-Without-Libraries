"""GPT Without Libraries - A pure NumPy implementation of Transformer GPT"""

__version__ = "1.0.1"
__author__ = "GPT Without Libraries Team"
__description__ = "A pure NumPy implementation of Transformer GPT model without deep learning frameworks"

from .models.core.mini_gpt import MiniGPT
from .utils.tokenizer import SimpleTokenizer
from .training.training import Trainer

__all__ = [
    "MiniGPT",
    "SimpleTokenizer",
    "Trainer"
]