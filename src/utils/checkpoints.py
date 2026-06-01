from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    return checkpoint.get("metadata", {})
