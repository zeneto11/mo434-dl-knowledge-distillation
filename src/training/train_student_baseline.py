from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from src.datasets import build_dataloaders
from src.models import StudentBaselineModel
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


def _baseline_checkpoint_path(config: dict, student_name: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    return base / "students" / f"{dataset}_{student_name}_baseline.pt"


def run(config: dict, student_name: str) -> dict:
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = get_device()
    data = build_dataloaders(config)

    dataset = config["dataset"]["name"]
    setup_run_logger(training_log_path(
        config, f"{dataset}_{student_name}_baseline"))
    logger.info(
        f"student={student_name} dataset={dataset} device={device} seed={seed} num_classes={data.num_classes}")

    model = StudentBaselineModel(student_name, data.num_classes).to(device)
    optimizer = make_optimizer(model, config)
    amp = bool(config["training"].get("amp", True))
    epochs = int(config["training"].get("epochs", 30))
    scheduler = make_scheduler(optimizer, config, epochs)
    best_top1 = -1.0
    history = []
    best_path = _baseline_checkpoint_path(config, student_name)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_m = train_classifier_epoch(
            model, data.train, optimizer, device, amp, f"baseline train {epoch}/{epochs}")
        val_m = evaluate_classifier(
            model, data.val, device, amp, f"baseline val {epoch}/{epochs}")
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        logger.info(
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_m['loss']:.4f} train_top1={train_m['top1']:.2f} "
            f"val_loss={val_m['loss']:.4f} val_top1={val_m['top1']:.2f} val_top5={val_m['top5']:.2f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s"
        )
        if val_m["top1"] > best_top1:
            best_top1 = val_m["top1"]
            save_checkpoint(best_path, model, {
                "kind": "baseline", "student": student_name, "epoch": epoch, "val": val_m})
        if scheduler:
            scheduler.step()

    load_checkpoint(best_path, model, map_location=device)
    test_m = evaluate_classifier(
        model, data.test, device, amp, "baseline test")
    logger.info(
        f"test_top1={test_m['top1']:.2f} test_top5={test_m['top5']:.2f}")

    metrics = {"history": history, "test": test_m,
               "checkpoint": str(best_path)}
    write_metrics(results_path(
        config, f"{dataset}_{student_name}_baseline.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--student", choices=["student_s", "student_m", "student_l"], required=True)
    args = parser.parse_args()
    run(load_config(args.config), student_name=args.student)


if __name__ == "__main__":
    main()
