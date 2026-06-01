from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import StudentBaselineModel, StudentDistillationModel, build_teacher
from src.training.common import checkpoint_path, evaluate_classifier, get_device, results_path, seed_everything
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.metrics import AverageMeter, accuracy, write_metrics


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
            student_representation = student(images)
            if target == "pregap":
                logits = teacher.classify_feature_map(student_representation)
                teacher_representation = teacher_map
            else:
                logits = teacher.classify_pooled(student_representation)
                teacher_representation = teacher_pooled
            ce = F.cross_entropy(logits, labels)
            mse = F.mse_loss(student_representation, teacher_representation)
        acc1, acc5 = accuracy(logits, labels)
        batch = labels.size(0)
        losses.update(ce.item(), batch)
        mses.update(mse.item(), batch)
        top1.update(acc1, batch)
        top5.update(acc5, batch)
    return {"loss": losses.avg, "mse": mses.avg, "top1": top1.avg, "top5": top5.avg}


def run(config: dict, kind: str, target: str | None, loss: str | None, checkpoint: str | None, teacher_checkpoint: str | None) -> dict:
    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device()
    data = build_dataloaders(config)
    amp = bool(config["training"].get("amp", True))

    if kind == "teacher":
        model = build_teacher(
            config["teacher"]["name"], data.num_classes, pretrained=True).to(device)
        load_checkpoint(checkpoint or checkpoint_path(
            config, "teachers", "classifier"), model, map_location=device)
        metrics = evaluate_classifier(
            model, data.test, device, amp, "teacher eval")
        filename = f"{config['dataset']['name']}_{config['teacher']['name']}_teacher_eval.json"
    elif kind == "baseline":
        model = StudentBaselineModel(
            data.num_classes, pretrained=True).to(device)
        load_checkpoint(checkpoint or checkpoint_path(
            config, "students", "mobilenet_baseline"), model, map_location=device)
        metrics = evaluate_classifier(
            model, data.test, device, amp, "baseline eval")
        filename = f"{config['dataset']['name']}_mobilenet_baseline_eval.json"
    else:
        if target is None or loss is None:
            raise ValueError(
                "--target and --loss are required for distilled evaluation")
        teacher = build_teacher(
            config["teacher"]["name"], data.num_classes, pretrained=True).to(device)
        load_checkpoint(teacher_checkpoint or checkpoint_path(
            config, "teachers", "classifier"), teacher, map_location=device)
        student = StudentDistillationModel(
            target, teacher.spec, pretrained=True).to(device)
        suffix = f"mobilenet_{target}_{loss}"
        load_checkpoint(checkpoint or checkpoint_path(
            config, "students", suffix), student, map_location=device)
        metrics = evaluate_distilled(
            student, teacher, data.test, device, amp, target)
        filename = f"{config['dataset']['name']}_{config['teacher']['name']}_{suffix}_eval.json"

    write_metrics(results_path(config, filename), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--kind", choices=["teacher", "baseline", "distilled"], required=True)
    parser.add_argument("--target", choices=["pregap", "postgap"])
    parser.add_argument("--loss", choices=["mse", "mse_ce"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint")
    args = parser.parse_args()
    run(load_config(args.config), args.kind, args.target,
        args.loss, args.checkpoint, args.teacher_checkpoint)


if __name__ == "__main__":
    main()
