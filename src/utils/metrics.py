from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@torch.no_grad()
def accuracy(logits: Tensor, target: Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        k = min(k, logits.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        values.append((correct_k * (100.0 / target.numel())).item())
    return values


def write_metrics(path: str | Path, metrics: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
