"""
Launcher for the full knowledge-distillation experiment matrix.

Generates all training and evaluation commands for:
  4 teacher models
  6 student CE baselines    (3 students × 2 datasets)
  48 distilled students     (2 datasets × 2 teachers × 3 students × 2 targets × 2 losses)
  + corresponding evaluate commands

Usage examples:
  # Print all commands (dry-run):
  python -m src.run_experiments --dry-run

  # Run only missing experiments:
  python -m src.run_experiments --skip-existing

  # Run only aircraft + student_s + pregap:
  python -m src.run_experiments --datasets aircraft --students student_s --targets pregap

  # Save manifest of planned runs:
  python -m src.run_experiments --dry-run --manifest results/manifests/planned.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger


CONFIGS: dict[tuple[str, str], str] = {
    ("aircraft", "resnet50"):    "configs/aircraft_resnet50.yaml",
    ("aircraft", "convnext_tiny"): "configs/aircraft_convnext_tiny.yaml",
    ("food101",  "resnet50"):    "configs/food101_resnet50.yaml",
    ("food101",  "convnext_tiny"): "configs/food101_convnext_tiny.yaml",
}
ALL_DATASETS = ["aircraft", "food101"]
ALL_TEACHERS = ["resnet50", "convnext_tiny"]
ALL_STUDENTS = ["student_s", "student_m", "student_l"]
ALL_TARGETS  = ["pregap", "postgap"]
ALL_LOSSES   = ["mse", "mse_ce"]


# ── checkpoint existence helpers ──────────────────────────────────────────────

def _teacher_ckpt(dataset: str, teacher: str) -> Path:
    return Path("checkpoints") / "teachers" / f"{dataset}_{teacher}_classifier.pt"


def _baseline_ckpt(dataset: str, student: str) -> Path:
    return Path("checkpoints") / "students" / f"{dataset}_{student}_baseline.pt"


def _distill_ckpt(dataset: str, teacher: str, student: str, target: str, loss: str) -> Path:
    return Path("checkpoints") / "students" / f"{dataset}_{teacher}_{student}_{target}_{loss}.pt"


def _teacher_eval(dataset: str, teacher: str) -> Path:
    return Path("results") / "evaluation" / f"{dataset}_{teacher}_teacher_eval.json"


def _baseline_eval(dataset: str, student: str) -> Path:
    return Path("results") / "evaluation" / f"{dataset}_{student}_baseline_eval.json"


def _distill_eval(dataset: str, teacher: str, student: str, target: str, loss: str) -> Path:
    return Path("results") / "evaluation" / f"{dataset}_{teacher}_{student}_{target}_{loss}_eval.json"


# ── command builders ──────────────────────────────────────────────────────────

def _cmd(*parts: str) -> list[str]:
    return [sys.executable, "-m"] + list(parts)


def build_commands(
    datasets: list[str],
    teachers: list[str],
    students: list[str],
    targets: list[str],
    losses: list[str],
    skip_existing: bool,
    phase: str,
) -> list[dict]:
    """Return list of {name, cmd, output_path} dicts for all matching experiments."""
    runs: list[dict] = []

    if phase in ("all", "teacher"):
        for dataset in datasets:
            for teacher in teachers:
                config = CONFIGS[(dataset, teacher)]
                out = _teacher_ckpt(dataset, teacher)
                if skip_existing and out.exists():
                    continue
                runs.append({
                    "name": f"teacher_{dataset}_{teacher}",
                    "phase": "teacher",
                    "cmd": _cmd("src.training.train_teacher_classifier", "--config", config),
                    "checkpoint": str(out),
                })

    if phase in ("all", "baseline"):
        for dataset in datasets:
            config = CONFIGS[(dataset, teachers[0])]
            for student in students:
                out = _baseline_ckpt(dataset, student)
                if skip_existing and out.exists():
                    continue
                runs.append({
                    "name": f"baseline_{dataset}_{student}",
                    "phase": "baseline",
                    "cmd": _cmd("src.training.train_student_baseline",
                                "--config", config, "--student", student),
                    "checkpoint": str(out),
                })

    if phase in ("all", "distill"):
        for dataset in datasets:
            for teacher in teachers:
                config = CONFIGS[(dataset, teacher)]
                for student in students:
                    for target in targets:
                        for loss in losses:
                            out = _distill_ckpt(dataset, teacher, student, target, loss)
                            if skip_existing and out.exists():
                                continue
                            runs.append({
                                "name": f"distill_{dataset}_{teacher}_{student}_{target}_{loss}",
                                "phase": "distill",
                                "cmd": _cmd("src.training.train_student_distillation",
                                            "--config", config, "--student", student,
                                            "--target", target, "--loss", loss),
                                "checkpoint": str(out),
                            })

    if phase in ("all", "evaluate"):
        for dataset in datasets:
            for teacher in teachers:
                config = CONFIGS[(dataset, teacher)]
                out = _teacher_eval(dataset, teacher)
                if skip_existing and out.exists():
                    continue
                runs.append({
                    "name": f"eval_teacher_{dataset}_{teacher}",
                    "phase": "evaluate",
                    "cmd": _cmd("src.evaluation.evaluate",
                                "--config", config, "--kind", "teacher"),
                    "checkpoint": str(out),
                })
        for dataset in datasets:
            config = CONFIGS[(dataset, teachers[0])]
            for student in students:
                out = _baseline_eval(dataset, student)
                if skip_existing and out.exists():
                    continue
                runs.append({
                    "name": f"eval_baseline_{dataset}_{student}",
                    "phase": "evaluate",
                    "cmd": _cmd("src.evaluation.evaluate",
                                "--config", config, "--kind", "baseline", "--student", student),
                    "checkpoint": str(out),
                })
        for dataset in datasets:
            for teacher in teachers:
                config = CONFIGS[(dataset, teacher)]
                for student in students:
                    for target in targets:
                        for loss in losses:
                            out = _distill_eval(dataset, teacher, student, target, loss)
                            if skip_existing and out.exists():
                                continue
                            runs.append({
                                "name": f"eval_distill_{dataset}_{teacher}_{student}_{target}_{loss}",
                                "phase": "evaluate",
                                "cmd": _cmd("src.evaluation.evaluate",
                                            "--config", config, "--kind", "distilled",
                                            "--student", student, "--target", target, "--loss", loss),
                                "checkpoint": str(out),
                            })

    return runs


def _save_manifest(runs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"total": len(runs), "runs": runs}, f, indent=2)
    logger.info(f"Manifest saved to {path} ({len(runs)} runs)")


def run_all(runs: list[dict], dry_run: bool) -> None:
    total = len(runs)
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Scheduled {total} run(s)")
    for idx, run in enumerate(runs, 1):
        cmd_str = " ".join(run["cmd"])
        logger.info(f"[{idx}/{total}] {run['name']}")
        logger.info(f"  cmd: {cmd_str}")
        if dry_run:
            continue
        result = subprocess.run(run["cmd"], check=False)
        if result.returncode != 0:
            logger.error(f"FAILED (exit {result.returncode}): {run['name']}")
        else:
            logger.success(f"OK: {run['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the full KD experiment matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all commands without executing them")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose output checkpoint/eval file already exists")
    parser.add_argument("--phase",
                        choices=["all", "teacher", "baseline", "distill", "evaluate"],
                        default="all", help="Which experiment phase to run")
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS)
    parser.add_argument("--teachers", nargs="+", choices=ALL_TEACHERS, default=ALL_TEACHERS)
    parser.add_argument("--students", nargs="+", choices=ALL_STUDENTS, default=ALL_STUDENTS)
    parser.add_argument("--targets", nargs="+", choices=ALL_TARGETS, default=ALL_TARGETS)
    parser.add_argument("--losses", nargs="+", choices=ALL_LOSSES, default=ALL_LOSSES)
    parser.add_argument("--manifest", default=None,
                        help="Path to save the run manifest JSON (e.g. results/manifests/planned.json)")
    parser.add_argument("--log-file", default=None,
                        help="Path to write launcher log (default: stdout only)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(args.log_file,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   level="INFO", colorize=False)

    runs = build_commands(
        datasets=args.datasets,
        teachers=args.teachers,
        students=args.students,
        targets=args.targets,
        losses=args.losses,
        skip_existing=args.skip_existing,
        phase=args.phase,
    )

    if args.manifest:
        _save_manifest(runs, Path(args.manifest))

    run_all(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
