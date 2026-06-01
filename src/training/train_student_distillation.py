from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import StudentDistillationModel, build_teacher
from src.training.common import checkpoint_path, get_device, make_optimizer, results_path, seed_everything
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.losses import distillation_loss
from src.utils.metrics import AverageMeter, accuracy, write_metrics


def _student_logits(teacher, target: str, student_representation):
    if target == "pregap":
        return teacher.classify_feature_map(student_representation)
    return teacher.classify_pooled(student_representation)


def _teacher_representation(teacher, target: str, images):
    feature_map, pooled = teacher.forward_features(images)
    return feature_map if target == "pregap" else pooled


def _run_epoch(student, teacher, loader, optimizer, device, amp, target: str, beta: float, alpha: float, train: bool, desc: str) -> dict:
    student.train(train)
    teacher.eval()
    losses = AverageMeter()
    mses = AverageMeter()
    ces = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and device.type == "cuda")
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_representation = _teacher_representation(
                    teacher, target, images)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                student_representation = student(images)
                logits = _student_logits(
                    teacher, target, student_representation) if beta > 0 else None
                loss, mse, ce = distillation_loss(
                    student_representation, teacher_representation, logits, labels, alpha, beta)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = labels.size(0)
            losses.update(loss.item(), batch)
            mses.update(mse.item(), batch)
            ces.update(ce.item(), batch)
            if logits is not None:
                acc1, acc5 = accuracy(logits.detach(), labels)
                top1.update(acc1, batch)
                top5.update(acc5, batch)
    return {"loss": losses.avg, "mse": mses.avg, "ce": ces.avg, "top1": top1.avg, "top5": top5.avg}


def run(config: dict, target: str, use_ce: bool, teacher_checkpoint: str | None = None) -> dict:
    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device()
    data = build_dataloaders(config)
    teacher = build_teacher(
        config["teacher"]["name"], data.num_classes, pretrained=True).to(device)
    teacher.freeze_encoder()
    load_checkpoint(teacher_checkpoint or checkpoint_path(
        config, "teachers", "classifier"), teacher, map_location=device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = StudentDistillationModel(
        target=target, teacher_spec=teacher.spec, pretrained=True).to(device)
    optimizer = make_optimizer(student, config)
    amp = bool(config["training"].get("amp", True))
    epochs = int(config["training"].get("epochs", 10))
    alpha = float(config["training"].get("alpha", 1.0))
    beta = float(config["training"].get("beta", 1.0)) if use_ce else 0.0
    suffix = f"mobilenet_{target}_{'mse_ce' if use_ce else 'mse'}"
    best_path = checkpoint_path(config, "students", suffix)
    history = []
    best_score = -float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(student, teacher, data.train, optimizer, device,
                                   amp, target, beta, alpha, True, f"distill train {epoch}/{epochs}")
        val_metrics = _run_epoch(student, teacher, data.val, optimizer, device,
                                 amp, target, beta, alpha, False, f"distill val {epoch}/{epochs}")
        history.append(
            {"epoch": epoch, "train": train_metrics, "val": val_metrics})
        score = val_metrics["top1"] if use_ce else -val_metrics["mse"]
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, student, {
                            "kind": "distilled_student", "epoch": epoch, "target": target, "use_ce": use_ce, "val": val_metrics})

    load_checkpoint(best_path, student, map_location=device)
    test_metrics = _run_epoch(student, teacher, data.test, optimizer,
                              device, amp, target, beta, alpha, False, "distill test")
    metrics = {"history": history, "test": test_metrics,
               "checkpoint": str(best_path)}
    filename = f"{config['dataset']['name']}_{config['teacher']['name']}_{suffix}.json"
    write_metrics(results_path(config, filename), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target", choices=["pregap", "postgap"], required=True)
    parser.add_argument("--loss", choices=["mse", "mse_ce"], required=True)
    parser.add_argument("--teacher-checkpoint")
    args = parser.parse_args()
    run(load_config(args.config), target=args.target, use_ce=args.loss ==
        "mse_ce", teacher_checkpoint=args.teacher_checkpoint)


if __name__ == "__main__":
    main()
