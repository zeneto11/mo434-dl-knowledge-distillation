from __future__ import annotations

import argparse
import time

import torch
from loguru import logger

from pathlib import Path

from src.datasets import build_dataloaders
from src.models import build_teacher
from src.training.common import (
    evaluate_classifier,
    get_device,
    make_optimizer,
    make_scheduler,
    results_path,
    seed_everything,
    setup_run_logger,
    train_classifier_epoch,
    training_log_path,
)
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.metrics import write_metrics


def run(config: dict) -> dict:
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = get_device()
    data = build_dataloaders(config)

    dataset = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]
    amp = bool(config["training"].get("amp", True))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))

    stage1_epochs = int(config["teacher"].get("stage1_epochs", 30))
    stage2_epochs = int(config["teacher"].get("stage2_epochs", 15))
    stage2_lr = float(config["teacher"].get("stage2_lr", 5e-5))

    setup_run_logger(training_log_path(
        config, f"{dataset}_{teacher_name}_teacher"))
    logger.info(
        f"teacher={teacher_name} dataset={dataset} device={device} seed={seed} num_classes={data.num_classes}")
    logger.info(
        f"stage1_epochs={stage1_epochs} stage2_epochs={stage2_epochs} stage2_lr={stage2_lr:.2e}")

    model = build_teacher(teacher_name, data.num_classes,
                          pretrained=True).to(device)
    best_top1 = -1.0
    history = []
    best_path = str(
        Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
        / "teachers"
        / f"{dataset}_{teacher_name}_classifier.pt"
    )

    # Stage 1: freeze encoder, train classifier head only
    model.freeze_encoder()
    optimizer1 = make_optimizer(model, config)
    scheduler1 = make_scheduler(optimizer1, config, stage1_epochs)
    logger.info(f"stage=1 lr={config['training'].get('lr', 0.001):.2e}")

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.perf_counter()
        train_m = train_classifier_epoch(
            model, data.train, optimizer1, device, amp, f"teacher s1 {epoch}/{stage1_epochs}")
        val_m = evaluate_classifier(
            model, data.val, device, amp, f"teacher val {epoch}/{stage1_epochs}")
        elapsed = time.perf_counter() - t0
        lr = optimizer1.param_groups[0]["lr"]
        history.append({"stage": 1, "epoch": epoch,
                       "train": train_m, "val": val_m})
        logger.info(
            f"stage=1 epoch={epoch}/{stage1_epochs} "
            f"train_loss={train_m['loss']:.4f} train_top1={train_m['top1']:.2f} "
            f"val_loss={val_m['loss']:.4f} val_top1={val_m['top1']:.2f} val_top5={val_m['top5']:.2f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s"
        )
        if val_m["top1"] > best_top1:
            best_top1 = val_m["top1"]
            save_checkpoint(best_path, model, {
                            "kind": "teacher", "stage": 1, "epoch": epoch, "val": val_m})
        if scheduler1:
            scheduler1.step()

    # Stage 2: unfreeze last block, fine-tune with smaller LR
    if stage2_epochs > 0:
        model.unfreeze_last_block()
        optimizer2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=stage2_lr,
            weight_decay=weight_decay,
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=stage2_epochs, eta_min=1e-7)
        logger.info(f"stage=2 lr={stage2_lr:.2e} (last block unfrozen)")

        for epoch in range(1, stage2_epochs + 1):
            t0 = time.perf_counter()
            train_m = train_classifier_epoch(
                model, data.train, optimizer2, device, amp, f"teacher s2 {epoch}/{stage2_epochs}")
            val_m = evaluate_classifier(
                model, data.val, device, amp, f"teacher val s2 {epoch}/{stage2_epochs}")
            elapsed = time.perf_counter() - t0
            lr = optimizer2.param_groups[0]["lr"]
            history.append({"stage": 2, "epoch": epoch,
                           "train": train_m, "val": val_m})
            logger.info(
                f"stage=2 epoch={epoch}/{stage2_epochs} "
                f"train_loss={train_m['loss']:.4f} train_top1={train_m['top1']:.2f} "
                f"val_loss={val_m['loss']:.4f} val_top1={val_m['top1']:.2f} val_top5={val_m['top5']:.2f} "
                f"lr={lr:.2e} elapsed={elapsed:.1f}s"
            )
            if val_m["top1"] > best_top1:
                best_top1 = val_m["top1"]
                save_checkpoint(best_path, model, {
                                "kind": "teacher", "stage": 2, "epoch": epoch, "val": val_m})
            scheduler2.step()

    load_checkpoint(best_path, model, map_location=device)
    test_m = evaluate_classifier(model, data.test, device, amp, "teacher test")
    logger.info(
        f"test_top1={test_m['top1']:.2f} test_top5={test_m['top5']:.2f}")

    metrics = {"history": history, "test": test_m, "checkpoint": best_path}
    write_metrics(results_path(
        config, f"{dataset}_{teacher_name}_teacher.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
