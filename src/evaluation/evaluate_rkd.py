from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import build_dataloaders
from src.evaluation.evaluate import _compute_costs
from src.models import StudentRKDModel
from src.training.common import eval_results_path, evaluate_classifier, get_device, seed_everything
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.metrics import write_metrics


def _rkd_checkpoint_path(config: dict, student_name: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / "students" / f"{dataset}_{teacher}_{student_name}_rkd.pt"


def run(
    config: dict,
    student_name: str,
    checkpoint: str | None = None,
    skip_costs: bool = False,
) -> dict:
    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device()
    data = build_dataloaders(config)
    amp = bool(config["training"].get("amp", True))
    image_size = int(config["dataset"].get("image_size", 224))
    dataset = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]

    model = StudentRKDModel(student_name, data.num_classes).to(device)
    ckpt = checkpoint or str(_rkd_checkpoint_path(config, student_name))
    load_checkpoint(ckpt, model, map_location=device)
    metrics = evaluate_classifier(model, data.test, device, amp, "rkd eval")
    if not skip_costs:
        metrics["costs"] = _compute_costs(model, device, image_size)

    filename = f"{dataset}_{teacher_name}_{student_name}_rkd_eval.json"
    write_metrics(eval_results_path(config, filename), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--student", choices=["student_s", "student_m", "student_l"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--skip-costs", action="store_true",
                        help="Skip params/GFLOPs/latency computation")
    args = parser.parse_args()
    run(load_config(args.config), args.student, args.checkpoint, args.skip_costs)


if __name__ == "__main__":
    main()
