from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from src.datasets import get_num_classes
from src.models import StudentBaselineModel, StudentDistillationModel, build_teacher
from src.training.common import get_device, results_path
from src.utils.config import load_config
from src.utils.metrics import write_metrics


def _count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _flops(model: nn.Module, example: torch.Tensor) -> float | None:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(example,), verbose=False)
        return float(macs * 2 / 1e9)
    except Exception:
        return None


@torch.no_grad()
def _latency_ms(model: nn.Module, example: torch.Tensor, repeats: int = 30) -> float:
    device = example.device
    model.eval()
    for _ in range(5):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def run(config: dict, kind: str, target: str | None = None) -> dict:
    device = get_device()
    num_classes = get_num_classes(config["dataset"]["name"])
    image_size = int(config["dataset"].get("image_size", 224))
    example = torch.randn(1, 3, image_size, image_size, device=device)

    if kind == "teacher":
        model = build_teacher(
            config["teacher"]["name"], num_classes, pretrained=False).to(device)
    elif kind == "baseline":
        model = StudentBaselineModel(num_classes, pretrained=False).to(device)
    else:
        if target is None:
            raise ValueError(
                "--target is required for distilled student costs")
        teacher = build_teacher(
            config["teacher"]["name"], num_classes, pretrained=False)
        model = StudentDistillationModel(
            target, teacher.spec, pretrained=False).to(device)

    metrics = {
        "kind": kind,
        "target": target,
        "params": _count_params(model),
        "gflops": _flops(model, example),
        "latency_ms_batch1": _latency_ms(model, example),
    }
    suffix = f"_{target}" if target else ""
    filename = f"{config['dataset']['name']}_{config['teacher']['name']}_{kind}{suffix}_costs.json"
    write_metrics(results_path(config, filename), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--kind", choices=["teacher", "baseline", "distilled"], required=True)
    parser.add_argument("--target", choices=["pregap", "postgap"])
    args = parser.parse_args()
    run(load_config(args.config), args.kind, args.target)


if __name__ == "__main__":
    main()
