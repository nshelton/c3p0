import numpy as np
import torch
from dataclasses import dataclass
from transformer_lens import HookedTransformer


def pick_device():
    # CPU by default: TransformerLens warns MPS can be silently incorrect, and
    # interp-scale models run instantly on CPU. Pass device="mps" to opt in.
    return "cpu"


_models = {}


def load(name="gpt2-small", device=None):
    device = device or pick_device()
    key = (name, device)
    if key not in _models:
        _models[key] = HookedTransformer.from_pretrained(name, device=device)
    return _models[key]


@dataclass
class Capture:
    prompt: str
    tokens: list        # str tokens, aligned to seq dim
    attn: np.ndarray    # [n_layers, n_heads, seq, seq]
    logits: np.ndarray  # [seq, vocab]
    cache: object       # raw ActivationCache, for anything else


def run(model, prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    attn = np.stack([
        cache["pattern", l][0].cpu().numpy()
        for l in range(model.cfg.n_layers)
    ])
    return Capture(
        prompt=prompt,
        tokens=model.to_str_tokens(prompt),
        attn=attn,
        logits=logits[0].cpu().numpy(),
        cache=cache,
    )


def act(capture, name, layer=None):
    key = name if layer is None else (name, layer)
    return capture.cache[key][0].cpu().numpy()
