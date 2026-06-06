from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import StudentBaselineModel, StudentDistillationModel, build_teacher
from src.training.common import eval_results_path, evaluate_classifier, get_device, seed_everything
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.losses import align_spatial_size
from src.utils.metrics import AverageMeter, accuracy, write_metrics


class _DistilledInference(torch.nn.Module):
    """Wraps student + frozen teacher head for end-to-end GFLOPs measurement."""

    def __init__(self, student: StudentDistillationModel, teacher: torch.nn.Module, target: str) -> None:
        super().__init__()
        self.student = student
        self.target = target
        self.pool = teacher.pool
        self.classifier = teacher.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.student(x)
        if self.target == "pregap":
            pooled = torch.flatten(self.pool(representation), 1)
            return self.classifier(pooled)
        return self.classifier(representation)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _flops(model: torch.nn.Module, example: torch.Tensor) -> float | None:
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(example,), verbose=False)
        return float(macs * 2 / 1e9)
    except Exception:
        return None


@torch.no_grad()
def _latency_ms(model: torch.nn.Module, example: torch.Tensor, repeats: int = 30) -> float:
    model.eval()
    for _ in range(5):
        model(example)
    if example.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(example)
    if example.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _compute_costs(model: torch.nn.Module, device: torch.device, image_size: int) -> dict:
    example = torch.randn(1, 3, image_size, image_size, device=device)
    return {
        "params": _count_params(model),
        "gflops": _flops(model, example),
        "latency_ms_batch1": _latency_ms(model, example),
    }


@torch.no_grad()
def evaluate_distilled(student, teacher, loader, device, amp: bool, target: str) -> dict:
    student.eval()
    teacher.eval()
    losses = AverageMeter()
    mses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, labels in tqdm(loader, desc="distilled eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            teacher_map, teacher_pooled = teacher.forward_features(images)
            student_repr = student(images)
            if target == "pregap":
                logits = teacher.classify_feature_map(student_repr)
                teacher_repr = teacher_map
            else:
                logits = teacher.classify_pooled(student_repr)
                teacher_repr = teacher_pooled
            ce = F.cross_entropy(logits, labels)
            mse = F.mse_loss(align_spatial_size(student_repr, teacher_repr), teacher_repr)
        acc1, acc5 = accuracy(logits, labels)
        batch = labels.size(0)
        losses.update(ce.item(), batch)
        mses.update(mse.item(), batch)
        top1.update(acc1, batch)
        top5.update(acc5, batch)
    return {"loss": losses.avg, "mse": mses.avg, "top1": top1.avg, "top5": top5.avg}


def _baseline_checkpoint_path(config: dict, student_name: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    return base / "students" / f"{dataset}_{student_name}_baseline.pt"


def _distill_checkpoint_path(config: dict, student_name: str, target: str, loss: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / "students" / f"{dataset}_{teacher}_{student_name}_{target}_{loss}.pt"


def run(
    config: dict,
    kind: str,
    student_name: str | None = None,
    target: str | None = None,
    loss: str | None = None,
    checkpoint: str | None = None,
    teacher_checkpoint: str | None = None,
    skip_costs: bool = False,
) -> dict:
    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device()
    data = build_dataloaders(config)
    amp = bool(config["training"].get("amp", True))
    image_size = int(config["dataset"].get("image_size", 224))
    dataset = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]

    if kind == "teacher":
        model = build_teacher(teacher_name, data.num_classes, pretrained=True).to(device)
        ckpt = checkpoint or str(
            Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
            / "teachers" / f"{dataset}_{teacher_name}_classifier.pt"
        )
        load_checkpoint(ckpt, model, map_location=device)
        metrics = evaluate_classifier(model, data.test, device, amp, "teacher eval")
        if not skip_costs:
            metrics["costs"] = _compute_costs(model, device, image_size)
        filename = f"{dataset}_{teacher_name}_teacher_eval.json"

    elif kind == "baseline":
        if student_name is None:
            raise ValueError("--student is required for baseline evaluation")
        model = StudentBaselineModel(student_name, data.num_classes).to(device)
        ckpt = checkpoint or str(_baseline_checkpoint_path(config, student_name))
        load_checkpoint(ckpt, model, map_location=device)
        metrics = evaluate_classifier(model, data.test, device, amp, "baseline eval")
        if not skip_costs:
            metrics["costs"] = _compute_costs(model, device, image_size)
        filename = f"{dataset}_{student_name}_baseline_eval.json"

    else:  # distilled
        if student_name is None or target is None or loss is None:
            raise ValueError("--student, --target, and --loss are required for distilled evaluation")
        teacher = build_teacher(teacher_name, data.num_classes, pretrained=True).to(device)
        t_ckpt = teacher_checkpoint or str(
            Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
            / "teachers" / f"{dataset}_{teacher_name}_classifier.pt"
        )
        load_checkpoint(t_ckpt, teacher, map_location=device)
        student = StudentDistillationModel(student_name, target, teacher.spec).to(device)
        s_ckpt = checkpoint or str(_distill_checkpoint_path(config, student_name, target, loss))
        load_checkpoint(s_ckpt, student, map_location=device)
        metrics = evaluate_distilled(student, teacher, data.test, device, amp, target)
        if not skip_costs:
            inference_model = _DistilledInference(student, teacher, target)
            metrics["costs"] = _compute_costs(inference_model, device, image_size)
        filename = f"{dataset}_{teacher_name}_{student_name}_{target}_{loss}_eval.json"

    write_metrics(eval_results_path(config, filename), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=["teacher", "baseline", "distilled"], required=True)
    parser.add_argument("--student", choices=["student_s", "student_m", "student_l"])
    parser.add_argument("--target", choices=["pregap", "postgap"])
    parser.add_argument("--loss", choices=["mse", "mse_ce"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--skip-costs", action="store_true", help="Skip params/GFLOPs/latency computation")
    args = parser.parse_args()
    run(load_config(args.config), args.kind, args.student, args.target,
        args.loss, args.checkpoint, args.teacher_checkpoint, args.skip_costs)


if __name__ == "__main__":
    main()
