from __future__ import annotations

import argparse

from src.datasets import build_dataloaders
from src.models import build_teacher
from src.training.common import (
    checkpoint_path,
    evaluate_classifier,
    get_device,
    make_optimizer,
    results_path,
    seed_everything,
    train_classifier_epoch,
)
from src.utils.checkpoints import save_checkpoint
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.metrics import write_metrics


def run(config: dict) -> dict:
    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device()
    data = build_dataloaders(config)
    model = build_teacher(config["teacher"]["name"],
                          data.num_classes, pretrained=True).to(device)
    model.freeze_encoder()
    optimizer = make_optimizer(model, config)
    amp = bool(config["training"].get("amp", True))
    epochs = int(config["training"].get("epochs", 10))
    best_top1 = -1.0
    history = []
    best_path = checkpoint_path(config, "teachers", "classifier")

    for epoch in range(1, epochs + 1):
        train_metrics = train_classifier_epoch(
            model, data.train, optimizer, device, amp, f"teacher train {epoch}/{epochs}")
        val_metrics = evaluate_classifier(
            model, data.val, device, amp, f"teacher val {epoch}/{epochs}")
        history.append(
            {"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(best_path, model, {
                            "kind": "teacher", "epoch": epoch, "val": val_metrics})

    load_checkpoint(best_path, model, map_location=device)
    test_metrics = evaluate_classifier(
        model, data.test, device, amp, "teacher test")
    metrics = {"history": history, "test": test_metrics,
               "checkpoint": str(best_path)}
    write_metrics(results_path(
        config, f"{config['dataset']['name']}_{config['teacher']['name']}_teacher.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
