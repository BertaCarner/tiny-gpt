from dataclasses import asdict
from pathlib import Path

import torch as t

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT


def save_checkpoint(model: GPT, path: str | Path, **metadata) -> None:
    """Save the weights together with the config, so the model can be rebuilt without knowing its size."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t.save({"config": asdict(model.cfg), "model": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> GPT:
    checkpoint = t.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
