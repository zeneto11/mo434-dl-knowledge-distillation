from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from src.datasets.factory import IMAGENET_MEAN, IMAGENET_STD, _normalize_transform, get_num_classes
from src.models import StudentDistillationModel, build_teacher
from src.training.common import get_device, seed_everything
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.losses import align_spatial_size


def _load_test_dataset(config: dict):
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"]
    root = Path(dataset_cfg.get("root", "data"))
    image_size = int(dataset_cfg.get("image_size", 224))
    transform = _normalize_transform(image_size, train=False)
    if name == "aircraft":
        return datasets.FGVCAircraft(
            root=str(root), split="test", annotation_level="variant",
            transform=transform, download=True,
        )
    if name == "food101":
        return datasets.Food101(root=str(root), split="test", transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {name}")


def _unnormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def _activation_heatmap(feature_map: torch.Tensor, size: int) -> np.ndarray:
    """Channel-averaged activation magnitude, upsampled to `size x size` and min-max normalized."""
    heat = feature_map.mean(dim=1, keepdim=True)
    heat = F.interpolate(heat, size=(size, size), mode="bilinear", align_corners=False)
    heat = heat.squeeze().cpu().numpy()
    heat = heat - heat.min()
    denom = heat.max()
    return heat / denom if denom > 1e-8 else heat


def _distill_checkpoint_path(config: dict, student_name: str, target: str, loss: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / "students" / f"{dataset}_{teacher}_{student_name}_{target}_{loss}.pt"


def _select_examples(
    teacher, student, test_dataset, classes, device,
    num_images: int, pool_size: int, num_failures: int, rng: np.random.Generator,
) -> list[dict]:
    """Sample a pool of test images and pick a mix of correct and student-misclassified examples."""
    pool_indices = rng.choice(len(test_dataset), size=min(pool_size, len(test_dataset)), replace=False)

    candidates = []
    with torch.no_grad():
        for idx in pool_indices:
            image, label = test_dataset[int(idx)]
            batch = image.unsqueeze(0).to(device)

            teacher_map, _ = teacher.forward_features(batch)
            teacher_logits = teacher.classify_feature_map(teacher_map)

            student_map = student(batch)
            student_map_aligned = align_spatial_size(student_map, teacher_map)
            student_logits = teacher.classify_feature_map(student_map_aligned)

            teacher_pred = int(teacher_logits.argmax(dim=1).item())
            student_pred = int(student_logits.argmax(dim=1).item())
            mse = F.mse_loss(student_map_aligned, teacher_map).item()

            candidates.append({
                "image": _unnormalize(image),
                "teacher_heat": _activation_heatmap(teacher_map, image.shape[-1]),
                "student_heat": _activation_heatmap(student_map_aligned, image.shape[-1]),
                "label": classes[label] if classes else str(label),
                "teacher_pred": classes[teacher_pred] if classes else str(teacher_pred),
                "student_pred": classes[student_pred] if classes else str(student_pred),
                "teacher_correct": teacher_pred == label,
                "student_correct": student_pred == label,
                "mse": mse,
            })

    failures = [c for c in candidates if c["teacher_correct"] and not c["student_correct"]]
    successes = [c for c in candidates if c["teacher_correct"] and c["student_correct"]]

    chosen = failures[:num_failures]
    remaining = num_images - len(chosen)
    chosen += successes[:remaining]
    if len(chosen) < num_images:
        rest = [c for c in candidates if c not in chosen]
        chosen += rest[: num_images - len(chosen)]
    return chosen[:num_images]


def run(config: dict, student_name: str, target: str, loss: str,
        num_images: int, num_failures: int, pool_size: int, seed: int, output: str) -> None:
    seed_everything(seed)
    device = get_device()
    dataset_name = config["dataset"]["name"]
    teacher_name = config["teacher"]["name"]
    num_classes = get_num_classes(dataset_name)

    teacher = build_teacher(teacher_name, num_classes, pretrained=True).to(device).eval()
    teacher_ckpt = (
        Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
        / "teachers" / f"{dataset_name}_{teacher_name}_classifier.pt"
    )
    load_checkpoint(teacher_ckpt, teacher, map_location=device)
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentDistillationModel(student_name, target, teacher.spec).to(device).eval()
    student_ckpt = _distill_checkpoint_path(config, student_name, target, loss)
    load_checkpoint(student_ckpt, student, map_location=device)

    test_dataset = _load_test_dataset(config)
    classes = getattr(test_dataset, "classes", None)
    rng = np.random.default_rng(seed)

    rows = _select_examples(
        teacher, student, test_dataset, classes, device,
        num_images, pool_size, num_failures, rng,
    )

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.8, 2.5 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Input image", "Teacher pre-GAP activation", "Distilled student activation"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, data in enumerate(rows):
        axes[row, 0].imshow(data["image"])
        axes[row, 0].axis("off")

        axes[row, 1].imshow(data["image"])
        axes[row, 1].imshow(data["teacher_heat"], cmap="jet", alpha=0.45)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(data["image"])
        axes[row, 2].imshow(data["student_heat"], cmap="jet", alpha=0.45)
        axes[row, 2].axis("off")

        teacher_color = "green" if data["teacher_correct"] else "red"
        student_color = "green" if data["student_correct"] else "red"
        label_text = (
            f"True: {data['label']}\n"
            f"Teacher: {data['teacher_pred']}\n"
            f"Student: {data['student_pred']}\n"
            f"MSE: {data['mse']:.3f}"
        )
        axes[row, 0].text(
            -0.08, 0.5, label_text, transform=axes[row, 0].transAxes,
            fontsize=7, ha="right", va="center",
        )
        axes[row, 1].text(
            0.5, -0.08, "correct" if data["teacher_correct"] else "wrong",
            transform=axes[row, 1].transAxes, fontsize=7, ha="center", va="top", color=teacher_color,
        )
        axes[row, 2].text(
            0.5, -0.08, "correct" if data["student_correct"] else "wrong",
            transform=axes[row, 2].transAxes, fontsize=7, ha="center", va="top", color=student_color,
        )

    plt.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {n} examples ({sum(1 for r in rows if not r['student_correct'])} student failures) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student", choices=["student_s", "student_m", "student_l"], default="student_l")
    parser.add_argument("--target", choices=["pregap", "postgap"], default="pregap")
    parser.add_argument("--loss", choices=["mse", "mse_ce"], default="mse_ce")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--num-failures", type=int, default=2,
                         help="Max number of student-misclassified (teacher-correct) examples to include")
    parser.add_argument("--pool-size", type=int, default=40,
                         help="Number of random test images to scan when looking for failure cases")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/figures/report/feature_map_comparison.png")
    args = parser.parse_args()
    run(
        load_config(args.config), args.student, args.target, args.loss,
        args.num_images, args.num_failures, args.pool_size, args.seed, args.output,
    )


if __name__ == "__main__":
    main()
