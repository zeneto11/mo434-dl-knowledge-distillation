from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import StudentDistillationModel, build_teacher
from src.training.common import (
    get_device,
    make_optimizer,
    make_scheduler,
    results_path,
    seed_everything,
    setup_run_logger,
    training_log_path,
)
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.losses import distillation_loss
from src.utils.metrics import AverageMeter, accuracy, write_metrics


def _distill_checkpoint_path(config: dict, student_name: str, target: str, loss_str: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / "students" / f"{dataset}_{teacher}_{student_name}_{target}_{loss_str}.pt"


def _student_logits(teacher, target: str, student_repr):
    if target == "pregap":
        return teacher.classify_feature_map(student_repr)
    return teacher.classify_pooled(student_repr)


def _teacher_representation(teacher, target: str, images):
    feature_map, pooled = teacher.forward_features(images)
    return feature_map if target == "pregap" else pooled


def _run_epoch(student, teacher, loader, optimizer, device, amp, target: str,
               beta: float, alpha: float, train: bool, desc: str) -> dict:
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
                teacher_repr = _teacher_representation(teacher, target, images)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                student_repr = student(images)
                logits = _student_logits(
                    teacher, target, student_repr) if beta > 0 else None
                loss, mse, ce = distillation_loss(
                    student_repr, teacher_repr, logits, labels, alpha, beta)
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


def run(config: dict, student_name: str, target: str, use_ce: bool,
        teacher_checkpoint: str | None = None) -> dict:
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = get_device()
    data = build_dataloaders(config)

    dataset = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]
    loss_str = "mse_ce" if use_ce else "mse"

    setup_run_logger(training_log_path(
        config, f"{dataset}_{teacher_name}_{student_name}_{target}_{loss_str}"))
    logger.info(
        f"student={student_name} teacher={teacher_name} dataset={dataset} "
        f"target={target} loss={loss_str} device={device} seed={seed}"
    )

    teacher = build_teacher(
        teacher_name, data.num_classes, pretrained=True).to(device)
    teacher.freeze_encoder()
    ckpt = teacher_checkpoint or str(
        Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
        / "teachers"
        / f"{dataset}_{teacher_name}_classifier.pt"
    )
    load_checkpoint(ckpt, teacher, map_location=device)
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentDistillationModel(
        student_name=student_name, target=target, teacher_spec=teacher.spec).to(device)
    optimizer = make_optimizer(student, config)
    amp = bool(config["training"].get("amp", True))
    epochs = int(config["training"].get("epochs", 30))
    alpha = float(config["training"].get("alpha", 1.0))
    beta = float(config["training"].get("beta", 1.0)) if use_ce else 0.0
    scheduler = make_scheduler(optimizer, config, epochs)

    best_path = _distill_checkpoint_path(
        config, student_name, target, loss_str)
    history = []
    best_score = -float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_m = _run_epoch(student, teacher, data.train, optimizer, device,
                             amp, target, beta, alpha, True, f"distill train {epoch}/{epochs}")
        val_m = _run_epoch(student, teacher, data.val, optimizer, device,
                           amp, target, beta, alpha, False, f"distill val {epoch}/{epochs}")
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        logger.info(
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_m['loss']:.4f} train_top1={train_m['top1']:.2f} "
            f"val_loss={val_m['loss']:.4f} val_top1={val_m['top1']:.2f} val_top5={val_m['top5']:.2f} "
            f"val_mse={val_m['mse']:.4f} val_ce={val_m['ce']:.4f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s"
        )
        score = val_m["top1"] if use_ce else -val_m["mse"]
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, student, {
                "kind": "distilled_student", "student": student_name,
                "epoch": epoch, "target": target, "use_ce": use_ce, "val": val_m,
            })
        if scheduler:
            scheduler.step()

    metrics = {"history": history, "checkpoint": str(best_path)}
    write_metrics(
        results_path(config, f"{dataset}_{teacher_name}_{student_name}_{target}_{loss_str}.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--student", choices=["student_s", "student_m", "student_l"], required=True)
    parser.add_argument(
        "--target", choices=["pregap", "postgap"], required=True)
    parser.add_argument("--loss", choices=["mse", "mse_ce"], required=True)
    parser.add_argument("--teacher-checkpoint")
    args = parser.parse_args()
    run(load_config(args.config), student_name=args.student, target=args.target,
        use_ce=args.loss == "mse_ce", teacher_checkpoint=args.teacher_checkpoint)


if __name__ == "__main__":
    main()
