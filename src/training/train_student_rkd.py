from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import StudentRKDModel, build_teacher
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
from src.utils.losses import RKDLoss
from src.utils.metrics import AverageMeter, accuracy, write_metrics


def _rkd_checkpoint_path(config: dict, student_name: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / "students" / f"{dataset}_{teacher}_{student_name}_rkd.pt"


def _run_epoch(student, teacher, loader, optimizer, device, amp, rkd_loss: RKDLoss,
               ce_weight: float, train: bool, desc: str) -> dict:
    student.train(train)
    teacher.eval()
    losses = AverageMeter()
    ces = AverageMeter()
    dists = AverageMeter()
    angles = AverageMeter()
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
                _, teacher_embedding = teacher.forward_features(images)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                student_embedding, logits = student.forward_with_embedding(
                    images)
                ce = F.cross_entropy(logits, labels)
                # Relational terms operate in float32 for numerically stable distances.
                relational, distance, angle = rkd_loss(
                    student_embedding.float(), teacher_embedding.float())
                loss = ce_weight * ce + relational
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = labels.size(0)
            losses.update(loss.item(), batch)
            ces.update(ce.item(), batch)
            dists.update(distance.item(), batch)
            angles.update(angle.item(), batch)
            acc1, acc5 = accuracy(logits.detach(), labels)
            top1.update(acc1, batch)
            top5.update(acc5, batch)
    return {
        "loss": losses.avg, "ce": ces.avg, "distance": dists.avg, "angle": angles.avg,
        "top1": top1.avg, "top5": top5.avg,
    }


def run(config: dict, student_name: str, teacher_checkpoint: str | None = None) -> dict:
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = get_device()
    data = build_dataloaders(config)

    dataset = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]

    rkd_cfg = config.get("rkd", {}) or {}
    distance_weight = float(rkd_cfg.get("distance_weight", 25.0))
    angle_weight = float(rkd_cfg.get("angle_weight", 50.0))
    ce_weight = float(rkd_cfg.get("ce_weight", 1.0))

    setup_run_logger(training_log_path(
        config, f"{dataset}_{teacher_name}_{student_name}_rkd"))
    logger.info(
        f"student={student_name} teacher={teacher_name} dataset={dataset} loss=rkd "
        f"distance_weight={distance_weight} angle_weight={angle_weight} ce_weight={ce_weight} "
        f"device={device} seed={seed}"
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

    student = StudentRKDModel(student_name, data.num_classes).to(device)
    optimizer = make_optimizer(student, config)
    amp = bool(config["training"].get("amp", True))
    epochs = int(config["training"].get("epochs", 30))
    scheduler = make_scheduler(optimizer, config, epochs)
    rkd_loss = RKDLoss(distance_weight=distance_weight,
                       angle_weight=angle_weight)

    best_path = _rkd_checkpoint_path(config, student_name)
    history = []
    best_top1 = -float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_m = _run_epoch(student, teacher, data.train, optimizer, device, amp,
                             rkd_loss, ce_weight, True, f"rkd train {epoch}/{epochs}")
        val_m = _run_epoch(student, teacher, data.val, optimizer, device, amp,
                           rkd_loss, ce_weight, False, f"rkd val {epoch}/{epochs}")
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        logger.info(
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_m['loss']:.4f} train_top1={train_m['top1']:.2f} "
            f"val_loss={val_m['loss']:.4f} val_top1={val_m['top1']:.2f} val_top5={val_m['top5']:.2f} "
            f"val_dist={val_m['distance']:.4f} val_angle={val_m['angle']:.4f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s"
        )
        if val_m["top1"] > best_top1:
            best_top1 = val_m["top1"]
            save_checkpoint(best_path, student, {
                "kind": "rkd_student", "student": student_name, "teacher": teacher_name,
                "epoch": epoch, "val": val_m,
            })
        if scheduler:
            scheduler.step()

    metrics = {"history": history, "checkpoint": str(best_path)}
    write_metrics(
        results_path(config, f"{dataset}_{teacher_name}_{student_name}_rkd.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--student", choices=["student_s", "student_m", "student_l"], required=True)
    parser.add_argument("--teacher-checkpoint")
    args = parser.parse_args()
    run(load_config(args.config), student_name=args.student,
        teacher_checkpoint=args.teacher_checkpoint)


if __name__ == "__main__":
    main()
