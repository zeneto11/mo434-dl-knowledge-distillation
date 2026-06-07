from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TEACHERS = ("resnet50", "convnext_tiny")
STUDENTS = ("student_s", "student_m", "student_l")
TARGETS = ("pregap", "postgap")
LOSSES = ("mse_ce", "mse")
TARGET_DESCRIPTIONS = {
    "pregap": "pre-GAP feature map with convolutional predictor",
    "postgap": "post-GAP pooled vector with MLP predictor",
}
STUDENT_DESCRIPTIONS = {
    "student_s": "Student-S (Conv32→Conv64→Conv128)",
    "student_m": "Student-M (Conv32→Conv64→Conv128→Conv256)",
    "student_l": "Student-L (Conv32→Conv64→Conv128→Conv256→Conv512)",
}

# ── Plotting conventions ──────────────────────────────────────────────────────
STUDENT_COLORS = {
    "student_s": "#1f77b4",
    "student_m": "#ff7f0e",
    "student_l": "#2ca02c",
}
STUDENT_LABELS_SHORT = {
    "student_s": "Student-S",
    "student_m": "Student-M",
    "student_l": "Student-L",
}
TEACHER_LABELS_SHORT = {
    "resnet50": "ResNet50",
    "convnext_tiny": "ConvNeXt-Tiny",
}
TEACHER_COLOR = "#9467bd"
BASELINE_ALPHA = 0.50
# Circle = pre-GAP (spatial), square = post-GAP (pooled)
TARGET_OVERLAYS = {"pregap": "3", "postgap": "4"}
TEACHER_MARKERS = {"resnet50": "o", "convnext_tiny": "s"}

# Black edge = CE supervision included, no edge = MSE-only
LOSS_EDGE = {"mse_ce": "black", "mse": "none"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _flatten_metrics(data: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in data.items() if key != "costs"}
    costs = data.get("costs") or {}
    if isinstance(costs, dict):
        for key, value in costs.items():
            row[f"cost_{key}"] = value
    return row


def collect_results(eval_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    teacher_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<teacher>{'|'.join(TEACHERS)})_teacher_eval\.json$"
    )
    baseline_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<student>{'|'.join(STUDENTS)})_baseline_eval\.json$"
    )
    distilled_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<teacher>{'|'.join(TEACHERS)})_(?P<student>{'|'.join(STUDENTS)})_"
        rf"(?P<target>{'|'.join(TARGETS)})_(?P<loss>{'|'.join(LOSSES)})_eval\.json$"
    )

    for path in sorted(eval_dir.glob("*_eval.json")):
        metrics = _flatten_metrics(_load_json(path))
        row: dict[str, Any] = {"file": path.name}

        if match := distilled_pattern.match(path.name):
            row.update(match.groupdict())
            row["model_type"] = "distilled"
            row["model"] = f"{row['student']} distilled"
        elif match := teacher_pattern.match(path.name):
            row.update(match.groupdict())
            row["model_type"] = "teacher"
            row["model"] = row["teacher"]
            row["student"] = None
            row["target"] = None
            row["loss_name"] = None
        elif match := baseline_pattern.match(path.name):
            row.update(match.groupdict())
            row["model_type"] = "baseline"
            row["model"] = f"{row['student']} baseline"
            row["teacher"] = None
            row["target"] = None
            row["loss_name"] = "ce"
        else:
            continue

        if "loss" in row and row["model_type"] == "distilled":
            row["loss_name"] = row.pop("loss")
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No evaluation JSON files found in {eval_dir}")
    return pd.DataFrame(rows)


# ── Derived metrics ───────────────────────────────────────────────────────────

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for column in ["baseline_top1", "teacher_top1", "top1_minus_baseline", "top1_minus_teacher"]:
        if column not in enriched.columns:
            enriched[column] = pd.NA
    for column in ["params_ratio_vs_teacher", "gflops_ratio_vs_teacher",
                   "params_saved_vs_teacher_pct", "gflops_saved_vs_teacher_pct", "top1_per_gflop"]:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    baseline_top1: dict[tuple, float] = {}
    for _, row in enriched[enriched["model_type"] == "baseline"].dropna(subset=["top1"]).iterrows():
        dataset = row.get("dataset")
        student = row.get("student")
        if dataset and student:
            baseline_top1[(dataset, student)] = row["top1"]

    teacher_lookup = (
        enriched[enriched["model_type"] == "teacher"]
        .dropna(subset=["top1"])
        .set_index(["dataset", "teacher"])["top1"]
        .to_dict()
    )
    teacher_params: dict = {}
    teacher_gflops: dict = {}
    if "cost_params" in enriched.columns:
        teacher_params = (
            enriched[enriched["model_type"] == "teacher"]
            .dropna(subset=["cost_params"])
            .set_index(["dataset", "teacher"])["cost_params"]
            .to_dict()
        )
    if "cost_gflops" in enriched.columns:
        teacher_gflops = (
            enriched[enriched["model_type"] == "teacher"]
            .dropna(subset=["cost_gflops"])
            .set_index(["dataset", "teacher"])["cost_gflops"]
            .to_dict()
        )

    for idx, row in enriched.iterrows():
        dataset = row.get("dataset")
        teacher = row.get("teacher")
        student = row.get("student")
        top1 = row.get("top1")
        if pd.notna(top1) and dataset and student and (dataset, student) in baseline_top1:
            enriched.at[idx, "baseline_top1"] = baseline_top1[(
                dataset, student)]
            enriched.at[idx, "top1_minus_baseline"] = top1 - \
                baseline_top1[(dataset, student)]
        if pd.notna(top1) and teacher and (dataset, teacher) in teacher_lookup:
            enriched.at[idx, "teacher_top1"] = teacher_lookup[(
                dataset, teacher)]
            enriched.at[idx, "top1_minus_teacher"] = top1 - \
                teacher_lookup[(dataset, teacher)]
        if teacher and (dataset, teacher) in teacher_params and pd.notna(row.get("cost_params", pd.NA)):
            ratio = row["cost_params"] / teacher_params[(dataset, teacher)]
            enriched.at[idx, "params_ratio_vs_teacher"] = ratio
            enriched.at[idx, "params_saved_vs_teacher_pct"] = (
                1.0 - ratio) * 100.0
        if teacher and (dataset, teacher) in teacher_gflops and pd.notna(row.get("cost_gflops", pd.NA)):
            ratio = row["cost_gflops"] / teacher_gflops[(dataset, teacher)]
            enriched.at[idx, "gflops_ratio_vs_teacher"] = ratio
            enriched.at[idx, "gflops_saved_vs_teacher_pct"] = (
                1.0 - ratio) * 100.0
        if pd.notna(top1) and pd.notna(row.get("cost_gflops", pd.NA)) and row["cost_gflops"] != 0:
            enriched.at[idx, "top1_per_gflop"] = top1 / row["cost_gflops"]

    return enriched


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _format_number(value: Any, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], precision: int = 3) -> str:
    available = [c for c in columns if c in df.columns]
    if not available or df.empty:
        return "_No data available._"
    table = df[available].copy()
    for column in table.columns:
        table[column] = table[column].map(
            lambda v: _format_number(v, precision))
    headers = [str(c) for c in table.columns]
    rows = table.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows))
              for i in range(len(headers))]
    header = "| " + " | ".join(headers[i].ljust(widths[i])
                               for i in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[i]
                                  for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i])
                          for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _best_row(df: pd.DataFrame, metric: str = "top1") -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None
    return valid.loc[valid[metric].idxmax()]


def _summary_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(by, dropna=False)["top1"].agg(
        ["mean", "max", "min", "count"]).reset_index()
    return grouped.sort_values(["mean", "max"], ascending=False)


def _best_config_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for value, group in df.groupby(by, dropna=False):
        best = _best_row(group)
        if best is None:
            continue
        rows.append({
            by: value,
            "best_teacher": best.get("teacher"),
            "best_student": best.get("student"),
            "best_target": best.get("target"),
            "best_loss": best.get("loss_name"),
            "best_top1": best.get("top1"),
            "best_top5": best.get("top5"),
            "best_minus_baseline": best.get("top1_minus_baseline"),
            "best_minus_teacher": best.get("top1_minus_teacher"),
        })
    return pd.DataFrame(rows).sort_values("best_top1", ascending=False) if rows else pd.DataFrame()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.lower())


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Non-dominated rows: sorted by x ascending, keeps those that improve y."""
    sub = df.dropna(subset=[x_col, y_col]).sort_values(
        x_col).reset_index(drop=True)
    frontier: list[int] = []
    best_y = -float("inf")
    for i, row in sub.iterrows():
        if row[y_col] > best_y:
            best_y = row[y_col]
            frontier.append(i)
    return sub.loc[frontier]


def _cost_legend_handles() -> list:
    """Shared legend for cost_vs_top1 plots."""
    handles: list = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=TEACHER_COLOR,
               markersize=14, label="Teacher"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#555555",
               markersize=9, alpha=BASELINE_ALPHA, label="CE baseline"),
    ]
    for s in STUDENTS:
        handles.append(Patch(facecolor=STUDENT_COLORS[s], alpha=0.9,
                             label=STUDENT_LABELS_SHORT[s]))
    handles += [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#666666",
               markersize=10, label="ResNet50"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="#666666",
               markersize=9, label="ConvNeXt-Tiny"),
        Line2D([0], [0], marker="3", color="black",
               linestyle="None", markersize=10,
               label="pre-GAP target"),
        Line2D([0], [0], marker="4", color="black",
               linestyle="None", markersize=10,
               label="post-GAP target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markeredgecolor="black", markeredgewidth=0.9,
               markersize=9, label="MSE+CE (black edge)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markersize=10, label="MSE (no edge)"),
        Line2D([0], [0], color="#333333", linewidth=1.2, linestyle="--",
               alpha=0.45, label="Pareto frontier"),
    ]
    return handles


# ── Per-dataset plots ─────────────────────────────────────────────────────────

def _plot_cost_vs_top1(dataset: str, dataset_df: pd.DataFrame, figures_dir: Path) -> str | None:
    """Scatter: GFLOPs vs Top-1. Color=student, marker=target, edge=loss, Pareto frontier."""
    if "cost_gflops" not in dataset_df.columns or not dataset_df["cost_gflops"].notna().any():
        return None

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Teachers
    for _, row in dataset_df[dataset_df["model_type"] == "teacher"].dropna(
        subset=["cost_gflops", "top1"]
    ).iterrows():
        ax.scatter(row["cost_gflops"], row["top1"], color=TEACHER_COLOR,
                   marker="*", s=300, zorder=6, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{TEACHER_LABELS_SHORT.get(str(row.get('teacher')), str(row.get('teacher')))}"
            f"\n({row['top1']:.1f}%)",
            (row["cost_gflops"], row["top1"]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=8, color=TEACHER_COLOR, fontweight="bold",
        )

    # CE baselines (diamond, semi-transparent, colored by student)
    for _, row in dataset_df[dataset_df["model_type"] == "baseline"].dropna(
        subset=["cost_gflops", "top1"]
    ).iterrows():
        student = str(row.get("student", ""))
        color = STUDENT_COLORS.get(student, "#7f7f7f")
        ax.scatter(row["cost_gflops"], row["top1"], color=color,
                   marker="D", s=50, zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(
            f"{STUDENT_LABELS_SHORT.get(student, student)}\nbaseline",
            (row["cost_gflops"], row["top1"]),
            xytext=(5, -13), textcoords="offset points",
            fontsize=7, color=color, alpha=0.85,
        )

    # Distilled: color=student, marker=teacher, cross=target, edge=loss
    for _, row in dataset_df[dataset_df["model_type"] == "distilled"].dropna(
        subset=["cost_gflops", "top1"]
    ).iterrows():
        student = str(row.get("student", ""))
        teacher = str(row.get("teacher", ""))
        target = str(row.get("target", "pregap"))
        loss = str(row.get("loss_name", ""))

        x = row["cost_gflops"]
        y = row["top1"]

        # Scatter teacher indicator
        ax.scatter(
            x, y,
            color=STUDENT_COLORS.get(student, "#333333"),
            marker=TEACHER_MARKERS.get(teacher, "o"),
            s=90,
            alpha=0.88,
            edgecolors=LOSS_EDGE.get(loss, "none"),
            linewidths=0.9,
            zorder=3,
        )

        # Overlay target indicator
        ax.scatter(
            x, y,
            marker=TARGET_OVERLAYS.get(target, "3"),
            color="black",
            s=70,
            linewidths=1.2,
            zorder=4,
        )

    # Pareto frontier across all models
    pareto = _pareto_frontier(dataset_df.dropna(subset=["cost_gflops", "top1"]),
                              "cost_gflops", "top1")
    if len(pareto) >= 2:
        # step-style so the staircase shape is visible
        xs = pareto["cost_gflops"].tolist()
        ys = pareto["top1"].tolist()
        ax.step(xs + [xs[-1] * 1.05], ys + [ys[-1]], where="post",
                color="#333333", linewidth=1.3, linestyle="--", alpha=0.45, zorder=2)

    ax.legend(handles=_cost_legend_handles(), loc="lower right",
              fontsize=8, framealpha=0.92, edgecolor="#cccccc")
    ax.set_title(f"{dataset}: Accuracy vs Compute (GFLOPs)", fontsize=13)
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.grid(alpha=0.22)
    plt.tight_layout()

    path = figures_dir / f"{_safe_name(dataset)}_cost_vs_top1.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _plot_param_vs_top1(dataset: str, dataset_df: pd.DataFrame, figures_dir: Path) -> str | None:
    """Scatter: Parameters vs Top-1. Color=student, marker=target, edge=loss, Pareto frontier."""
    if "cost_params" not in dataset_df.columns or not dataset_df["cost_params"].notna().any():
        return None

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Teachers
    for _, row in dataset_df[dataset_df["model_type"] == "teacher"].dropna(
        subset=["cost_params", "top1"]
    ).iterrows():
        ax.scatter(row["cost_params"], row["top1"], color=TEACHER_COLOR,
                   marker="*", s=300, zorder=6, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{TEACHER_LABELS_SHORT.get(str(row.get('teacher')), str(row.get('teacher')))}"
            f"\n({row['top1']:.1f}%)",
            (row["cost_params"], row["top1"]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=8, color=TEACHER_COLOR, fontweight="bold",
        )

    # CE baselines (diamond, semi-transparent, colored by student)
    for _, row in dataset_df[dataset_df["model_type"] == "baseline"].dropna(
        subset=["cost_params", "top1"]
    ).iterrows():
        student = str(row.get("student", ""))
        color = STUDENT_COLORS.get(student, "#7f7f7f")
        ax.scatter(row["cost_params"], row["top1"], color=color,
                   marker="D", s=50, zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(
            f"{STUDENT_LABELS_SHORT.get(student, student)}\nbaseline",
            (row["cost_params"], row["top1"]),
            xytext=(5, -13), textcoords="offset points",
            fontsize=7, color=color, alpha=0.85,
        )

    # Distilled: color=student, marker=teacher, cross=target, edge=loss
    for _, row in dataset_df[dataset_df["model_type"] == "distilled"].dropna(
        subset=["cost_params", "top1"]
    ).iterrows():
        student = str(row.get("student", ""))
        teacher = str(row.get("teacher", ""))
        target = str(row.get("target", "pregap"))
        loss = str(row.get("loss_name", ""))

        x = row["cost_params"]
        y = row["top1"]

        # Scatter teacher indicator
        ax.scatter(
            x, y,
            color=STUDENT_COLORS.get(student, "#333333"),
            marker=TEACHER_MARKERS.get(teacher, "o"),
            s=90,
            alpha=0.88,
            edgecolors=LOSS_EDGE.get(loss, "none"),
            linewidths=0.9,
            zorder=3,
        )

        # Overlay target indicator
        ax.scatter(
            x, y,
            marker=TARGET_OVERLAYS.get(target, "3"),
            color="black",
            s=70,
            linewidths=1.2,
            zorder=4,
        )

    # Pareto frontier across all models
    pareto = _pareto_frontier(dataset_df.dropna(subset=["cost_params", "top1"]),
                              "cost_params", "top1")
    if len(pareto) >= 2:
        # step-style so the staircase shape is visible
        xs = pareto["cost_params"].tolist()
        ys = pareto["top1"].tolist()
        ax.step(xs + [xs[-1] * 1.05], ys + [ys[-1]], where="post",
                color="#333333", linewidth=1.3, linestyle="--", alpha=0.45, zorder=2)

    ax.legend(handles=_cost_legend_handles(), loc="lower right",
              fontsize=8, framealpha=0.92, edgecolor="#cccccc")
    ax.set_title(f"{dataset}: Accuracy vs Compute (Params)", fontsize=13)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.grid(alpha=0.22)
    plt.tight_layout()

    path = figures_dir / f"{_safe_name(dataset)}_param_vs_top1.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _plot_pregap_vs_postgap(dataset: str, distilled: pd.DataFrame, figures_dir: Path) -> str | None:
    """Slope graph: for each (teacher, student, loss) pair, show postgap → pregap Top-1 shift."""
    plot_df = distilled.dropna(subset=["top1", "target", "student", "teacher"])
    if plot_df.empty:
        return None

    teachers_present = [t for t in TEACHERS if t in plot_df["teacher"].values]
    if not teachers_present:
        return None

    fig, axes = plt.subplots(1, len(teachers_present),
                             figsize=(6 * len(teachers_present), 5), sharey=True)
    if len(teachers_present) == 1:
        axes = [axes]

    for ax, teacher in zip(axes, teachers_present):
        teacher_df = plot_df[plot_df["teacher"] == teacher]

        for student in STUDENTS:
            for loss in LOSSES:
                sub = teacher_df[
                    (teacher_df["student"] == student) &
                    (teacher_df["loss_name"] == loss)
                ]
                post_rows = sub[sub["target"] == "postgap"]
                pre_rows = sub[sub["target"] == "pregap"]
                if post_rows.empty or pre_rows.empty:
                    continue

                y_post = float(post_rows.iloc[0]["top1"])
                y_pre = float(pre_rows.iloc[0]["top1"])
                color = STUDENT_COLORS.get(student, "#333333")
                ls = "-" if loss == "mse_ce" else "--"
                lw = 2.2 if loss == "mse_ce" else 1.5
                edge = "black" if loss == "mse_ce" else "none"

                ax.plot([0, 1], [y_post, y_pre], color=color,
                        linewidth=lw, linestyle=ls, alpha=0.88, zorder=3)
                teacher_marker = TEACHER_MARKERS.get(teacher, "o")
                for x, y in ((0, y_post), (1, y_pre)):
                    target = "postgap" if x == 0 else "pregap"

                    # Base marker = teacher
                    ax.scatter(
                        x, y,
                        color=color,
                        marker=teacher_marker,
                        s=90,
                        zorder=4,
                        edgecolors=edge,
                        linewidths=0.8,
                    )

                    # Overlay = target
                    ax.scatter(
                        x, y,
                        marker=TARGET_OVERLAYS[target],
                        color="black",
                        s=65,
                        zorder=5,
                        linewidths=1.0,
                    )

                # Annotate the delta on the connecting line midpoint
                delta = y_pre - y_post
                ax.annotate(
                    f"{delta:+.1f}pp",
                    xy=(0.5, (y_post + y_pre) / 2),
                    fontsize=7, ha="center", va="bottom",
                    color=color, alpha=0.85,
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["post-GAP", "pre-GAP"], fontsize=10)
        ax.set_title(TEACHER_LABELS_SHORT.get(teacher, teacher), fontsize=11)
        ax.set_xlim(-0.2, 1.2)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Top-1 accuracy (%)")
    fig.suptitle(f"{dataset}: Distillation target comparison — post-GAP vs pre-GAP",
                 fontsize=12, y=1.01)

    handles = []
    for s in STUDENTS:
        handles.append(Line2D([0], [0], color=STUDENT_COLORS[s], lw=2.2,
                              label=STUDENT_LABELS_SHORT[s]))
    handles += [
        Line2D([0], [0], color="#555555", lw=2.2,
               linestyle="-", label="MSE+CE"),
        Line2D([0], [0], color="#555555", lw=1.5,
               linestyle="--", label="MSE only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    path = figures_dir / f"{_safe_name(dataset)}_pregap_vs_postgap.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(path)


def _plot_gain_matrix(
    dataset: str, distilled: pd.DataFrame, figures_dir: Path
) -> str | None:
    """Heatmap: distillation gain over CE baseline — rows=student, cols=teacher×target.

    Each cell shows the best Top-1 gain (max over loss functions). Green = beats baseline,
    red = below baseline.
    """
    if distilled.empty or "top1_minus_baseline" not in distilled.columns:
        return None

    pairs = [(t, tgt) for t in TEACHERS for tgt in TARGETS]
    col_labels = [
        f"{TEACHER_LABELS_SHORT.get(t, t)}\n{tgt}"
        for t, tgt in pairs
    ]
    row_labels = [STUDENT_LABELS_SHORT.get(s, s) for s in STUDENTS]

    matrix = np.full((len(STUDENTS), len(pairs)), np.nan)
    for i, student in enumerate(STUDENTS):
        for j, (teacher, target) in enumerate(pairs):
            sub = distilled[
                (distilled["student"] == student) &
                (distilled["teacher"] == teacher) &
                (distilled["target"] == target)
            ].dropna(subset=["top1_minus_baseline"])
            if not sub.empty:
                matrix[i, j] = float(sub["top1_minus_baseline"].max())

    if np.all(np.isnan(matrix)):
        return None

    vmax = float(np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(STUDENTS)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(len(STUDENTS)):
        for j in range(len(pairs)):
            v = matrix[i, j]
            if not np.isnan(v):
                text_color = "black" if abs(v) < vmax * 0.55 else "white"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Top-1 gain vs CE baseline (pp)")
    ax.set_title(
        f"{dataset}: Distillation gain over CE baseline  "
        f"(best loss per cell; pp = percentage points)",
        fontsize=11,
    )
    plt.tight_layout()

    path = figures_dir / f"{_safe_name(dataset)}_gain_matrix.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _make_dataset_plots(dataset: str, dataset_df: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    plots: dict[str, str] = {}
    distilled = dataset_df[dataset_df["model_type"] == "distilled"].copy()

    path = _plot_cost_vs_top1(dataset, dataset_df, figures_dir)
    if path:
        plots["cost"] = path

    path = _plot_param_vs_top1(dataset, dataset_df, figures_dir)
    if path:
        plots["param"] = path

    path = _plot_pregap_vs_postgap(dataset, distilled, figures_dir)
    if path:
        plots["pregap_vs_postgap"] = path

    path = _plot_gain_matrix(dataset, distilled, figures_dir)
    if path:
        plots["gain_matrix"] = path

    return plots


# ── Global plots ──────────────────────────────────────────────────────────────

def _make_global_plots(df: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    plots: dict[str, str] = {}
    distilled = df[df["model_type"] == "distilled"].copy()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Horizontal bar: top-12 improvement over baseline (informative as-is)
    if not distilled.empty and "top1_minus_baseline" in distilled.columns:
        plot_df = distilled.dropna(subset=["top1_minus_baseline"]).copy()
        if not plot_df.empty:
            abbrev = {
                "resnet50": "RN50", "convnext_tiny": "CvN",
                "student_s": "S-S", "student_m": "S-M", "student_l": "S-L",
                "pregap": "pre", "postgap": "post",
                "mse_ce": "MSE+CE", "mse": "MSE",
            }
            plot_df["label"] = plot_df.apply(
                lambda r: (
                    f"{r['dataset']}/"
                    f"{abbrev.get(str(r.get('teacher', '')), str(r.get('teacher', '')))}/"
                    f"{abbrev.get(str(r.get('student', '')), str(r.get('student', '')))}/"
                    f"{abbrev.get(str(r.get('target', '')), str(r.get('target', '')))}/"
                    f"{abbrev.get(str(r.get('loss_name', '')), str(r.get('loss_name', '')))}"
                ),
                axis=1,
            )
            plot_df = plot_df.sort_values(
                "top1_minus_baseline", ascending=False).head(14)
            colors = ["#54A24B" if v >=
                      0 else "#E45756" for v in plot_df["top1_minus_baseline"]]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(plot_df["label"],
                    plot_df["top1_minus_baseline"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(
                "Top distilled students: improvement over matching CE baseline")
            ax.set_xlabel("Top-1 minus student CE baseline (pp)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            path = figures_dir / "global_improvement_over_baseline.png"
            plt.savefig(path, dpi=180)
            plt.close()
            plots["baseline_delta"] = str(path)

    # Global cost scatter: color=student, shape=target, dataset by fill alpha
    if "cost_gflops" in df.columns and df["cost_gflops"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        all_pts = df.dropna(subset=["cost_gflops", "top1"])
        dataset_names = sorted(all_pts["dataset"].dropna().unique())
        # Use different alpha levels to separate datasets visually
        dataset_alpha = {name: 0.95 if i == 0 else 0.45
                         for i, name in enumerate(dataset_names)}
        dataset_edge = {name: "black" if i == 0 else "none"
                        for i, name in enumerate(dataset_names)}

        for _, row in all_pts.iterrows():
            model_type = str(row.get("model_type", ""))
            student = str(row.get("student") or "")
            target = str(row.get("target") or "pregap")
            loss = str(row.get("loss_name") or "")
            dataset_name = str(row.get("dataset", ""))
            alpha = dataset_alpha.get(dataset_name, 0.7)

            if model_type == "teacher":
                ax.scatter(row["cost_gflops"], row["top1"], color=TEACHER_COLOR,
                           marker="*", s=280, zorder=6, alpha=alpha,
                           edgecolors="white", linewidths=0.5)
            elif model_type == "baseline":
                color = STUDENT_COLORS.get(student, "#7f7f7f")
                ax.scatter(row["cost_gflops"], row["top1"], color=color,
                           marker="D", s=50, zorder=5,
                           edgecolors=dataset_edge.get(dataset_name, "none"),
                           linewidths=0.6)
            else:
                color = STUDENT_COLORS.get(student, "#333333")
                teacher = str(row.get("teacher") or "")
                edgecolor = LOSS_EDGE.get(loss, "none")

                # Base marker = teacher
                ax.scatter(
                    row["cost_gflops"],
                    row["top1"],
                    color=color,
                    marker=TEACHER_MARKERS.get(teacher, "o"),
                    s=75,
                    zorder=3,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidths=0.8,
                )

                # Overlay = target
                ax.scatter(
                    row["cost_gflops"],
                    row["top1"],
                    marker=TARGET_OVERLAYS.get(target, "3"),
                    color="black",
                    s=55,
                    zorder=4,
                    alpha=alpha,
                    linewidths=1.0,
                )

        # Pareto per dataset
        for dataset_name in dataset_names:
            sub = all_pts[all_pts["dataset"] == dataset_name]
            pareto = _pareto_frontier(sub, "cost_gflops", "top1")
            if len(pareto) >= 2:
                xs = pareto["cost_gflops"].tolist()
                ys = pareto["top1"].tolist()
                ax.step(xs + [xs[-1] * 1.05], ys + [ys[-1]], where="post",
                        color="#333333", linewidth=1.0, linestyle="--", alpha=0.35, zorder=2)

        handles = _cost_legend_handles()
        # Append dataset legend entries
        for name in dataset_names:
            a = dataset_alpha.get(name, 0.7)
            handles.append(
                Patch(facecolor="#555555", alpha=a,
                      label=f"{name} ({'solid' if a > 0.6 else 'faded'})")
            )
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  framealpha=0.92, edgecolor="#cccccc")
        ax.set_title(
            "Global overview: Accuracy (Top-1) vs Compute", fontsize=13)
        ax.set_xlabel("GFLOPs")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(alpha=0.22)
        plt.tight_layout()
        path = figures_dir / "global_cost_vs_top1.png"
        plt.savefig(path, dpi=180)
        plt.close()
        plots["global_cost"] = str(path)

    # Global cost scatter (top5): color=student, shape=target, dataset by fill alpha
    if "cost_gflops" in df.columns and df["cost_gflops"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        all_pts = df.dropna(subset=["cost_gflops", "top5"])
        dataset_names = sorted(all_pts["dataset"].dropna().unique())
        # Use different alpha levels to separate datasets visually
        dataset_alpha = {name: 0.95 if i == 0 else 0.45
                         for i, name in enumerate(dataset_names)}
        dataset_edge = {name: "black" if i == 0 else "none"
                        for i, name in enumerate(dataset_names)}

        for _, row in all_pts.iterrows():
            model_type = str(row.get("model_type", ""))
            student = str(row.get("student") or "")
            target = str(row.get("target") or "pregap")
            loss = str(row.get("loss_name") or "")
            dataset_name = str(row.get("dataset", ""))
            alpha = dataset_alpha.get(dataset_name, 0.7)

            if model_type == "teacher":
                ax.scatter(row["cost_gflops"], row["top5"], color=TEACHER_COLOR,
                           marker="*", s=280, zorder=6, alpha=alpha,
                           edgecolors="white", linewidths=0.5)
            elif model_type == "baseline":
                color = STUDENT_COLORS.get(student, "#7f7f7f")
                ax.scatter(row["cost_gflops"], row["top5"], color=color,
                           marker="D", s=50, zorder=5,
                           edgecolors=dataset_edge.get(dataset_name, "none"),
                           linewidths=0.6)
            else:
                color = STUDENT_COLORS.get(student, "#333333")
                teacher = str(row.get("teacher") or "")
                edgecolor = LOSS_EDGE.get(loss, "none")

                # Base marker = teacher
                ax.scatter(
                    row["cost_gflops"],
                    row["top5"],
                    color=color,
                    marker=TEACHER_MARKERS.get(teacher, "o"),
                    s=75,
                    zorder=3,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidths=0.8,
                )

                # Overlay = target
                ax.scatter(
                    row["cost_gflops"],
                    row["top5"],
                    marker=TARGET_OVERLAYS.get(target, "3"),
                    color="black",
                    s=55,
                    zorder=4,
                    alpha=alpha,
                    linewidths=1.0,
                )

        # Pareto per dataset
        for dataset_name in dataset_names:
            sub = all_pts[all_pts["dataset"] == dataset_name]
            pareto = _pareto_frontier(sub, "cost_gflops", "top5")
            if len(pareto) >= 2:
                xs = pareto["cost_gflops"].tolist()
                ys = pareto["top5"].tolist()
                ax.step(xs + [xs[-1] * 1.05], ys + [ys[-1]], where="post",
                        color="#333333", linewidth=1.0, linestyle="--", alpha=0.35, zorder=2)

        handles = _cost_legend_handles()
        # Append dataset legend entries
        for name in dataset_names:
            a = dataset_alpha.get(name, 0.7)
            handles.append(
                Patch(facecolor="#555555", alpha=a,
                      label=f"{name} ({'solid' if a > 0.6 else 'faded'})")
            )
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  framealpha=0.92, edgecolor="#cccccc")
        ax.set_title(
            "Global overview: Accuracy (Top-5) vs Compute", fontsize=13)
        ax.set_xlabel("GFLOPs")
        ax.set_ylabel("Top-5 accuracy (%)")
        ax.grid(alpha=0.22)
        plt.tight_layout()
        path = figures_dir / "global_cost_vs_top5.png"
        plt.savefig(path, dpi=180)
        plt.close()
        plots["global_cost_top5"] = str(path)

    return plots


# ── Report assembly ───────────────────────────────────────────────────────────

def _relative_path(path: str, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _add_plot(lines: list[str], title: str, path: str | None, output: Path) -> None:
    if not path:
        return
    lines.extend(
        ["", f"![{title}]({_relative_path(path, output.parent)})", ""])


def _dataset_section(
    dataset: str, dataset_df: pd.DataFrame, output: Path, figures_dir: Path
) -> list[str]:
    distilled = dataset_df[dataset_df["model_type"] == "distilled"].copy()
    teachers_df = dataset_df[dataset_df["model_type"] == "teacher"].copy()
    baselines_df = dataset_df[dataset_df["model_type"] == "baseline"].copy()
    plots = _make_dataset_plots(dataset, dataset_df, figures_dir)

    result_columns = [
        "model_type", "teacher", "student", "target", "loss_name",
        "top1", "top5", "loss", "mse",
        "top1_minus_baseline", "top1_minus_teacher",
        "cost_params", "cost_gflops",
        "params_saved_vs_teacher_pct", "gflops_saved_vs_teacher_pct",
    ]
    lines = [f"## Dataset: {dataset}", ""]

    best_student = _best_row(distilled)
    best_baseline = _best_row(baselines_df)
    best_teacher = _best_row(teachers_df)
    if best_baseline is not None:
        lines.append(
            f"- Best student CE baseline: `{best_baseline.get('student')}` with "
            f"{_format_number(best_baseline['top1'])}% top-1."
        )
    if best_teacher is not None:
        lines.append(
            f"- Best teacher classifier: `{best_teacher['teacher']}` with "
            f"{_format_number(best_teacher['top1'])}% top-1."
        )
    if best_student is not None:
        lines.append(
            f"- Best distilled student: `{best_student.get('teacher')}` + "
            f"`{best_student.get('student')}` + `{best_student.get('target')}` + "
            f"`{best_student.get('loss_name')}` with {_format_number(best_student['top1'])}% top-1 "
            f"({_format_number(best_student.get('top1_minus_baseline'))} pp vs matching baseline, "
            f"{_format_number(best_student.get('top1_minus_teacher'))} pp vs its teacher)."
        )
    lines.append("")

    # Gain matrix gives a fast overview before the tables
    _add_plot(
        lines,
        f"{dataset} distillation gain over CE baseline",
        plots.get("gain_matrix"),
        output,
    )

    lines.extend([
        "### Evaluation Results", "",
        markdown_table(
            dataset_df.sort_values(
                ["model_type", "teacher", "student", "target", "loss_name"], na_position="last"),
            result_columns,
        ),
        "",
    ])

    # Q1: Which teacher transfers best?
    lines.extend(["### Question 1: Which Teacher Transfers Best?", ""])
    teacher_summary = _summary_by(distilled, "teacher")
    if not teacher_summary.empty:
        best = teacher_summary.iloc[0]
        lines.append(
            f"**Answer:** `{best['teacher']}` transfers best on this dataset by mean distilled Top-1 "
            f"({_format_number(best['mean'])}%)."
        )
    else:
        lines.append(
            "**Answer:** no distilled teacher comparison is available.")
    lines.extend(["", markdown_table(teacher_summary, [
                 "teacher", "mean", "max", "min", "count"]), ""])
    best_by_teacher = _best_config_by(distilled, "teacher")
    lines.extend([
        "Best configuration found for each teacher:", "",
        markdown_table(best_by_teacher, ["teacher", "best_student", "best_target", "best_loss",
                                         "best_top1", "best_top5", "best_minus_baseline", "best_minus_teacher"]),
    ])
    lines.append("")

    # Q2: What should the student predict?
    lines.extend(["### Question 2: What Should The Student Predict?", ""])
    target_summary = _summary_by(distilled, "target")
    if not target_summary.empty:
        best = target_summary.iloc[0]
        description = TARGET_DESCRIPTIONS.get(best["target"], best["target"])
        lines.append(
            f"**Answer:** `{best['target']}` performs best on this dataset by mean Top-1; "
            f"this corresponds to {description}."
        )
    else:
        lines.append("**Answer:** no target comparison is available.")
    lines.extend(["", markdown_table(target_summary, [
                 "target", "mean", "max", "min", "count"]), ""])
    best_by_target = _best_config_by(distilled, "target")
    lines.extend([
        "Best configuration found for each target:", "",
        markdown_table(best_by_target, ["target", "best_teacher", "best_student", "best_loss",
                                        "best_top1", "best_top5", "best_minus_baseline", "best_minus_teacher"]),
    ])
    # Slope graph shows the pregap vs postgap shift for every config
    _add_plot(lines, f"{dataset} pre-GAP vs post-GAP",
              plots.get("pregap_vs_postgap"), output)

    # Q3: What is the best student encoder + predictor architecture?
    lines.extend(
        ["### Question 3: What Is The Best Student Architecture?", ""])
    student_summary = _summary_by(distilled, "student")
    if not student_summary.empty:
        best = student_summary.iloc[0]
        desc = STUDENT_DESCRIPTIONS.get(best["student"], best["student"])
        lines.append(
            f"**Answer:** `{best['student']}` ({desc}) achieves the best mean distilled Top-1 "
            f"({_format_number(best['mean'])}%) on this dataset."
        )
    else:
        lines.append(
            "**Answer:** no student architecture comparison is available.")
    lines.extend(["", markdown_table(student_summary, [
                 "student", "mean", "max", "min", "count"]), ""])
    best_by_student = _best_config_by(distilled, "student")
    lines.extend([
        "Best configuration found for each student architecture:", "",
        markdown_table(best_by_student, ["student", "best_teacher", "best_target", "best_loss",
                                         "best_top1", "best_top5", "best_minus_baseline", "best_minus_teacher"]),
    ])
    architecture_table = distilled.sort_values(
        ["top1", "gflops_saved_vs_teacher_pct"], ascending=[False, False])
    lines.extend([
        "",
        "Architecture ranking with cost savings relative to the matching teacher:", "",
        markdown_table(architecture_table, [
            "teacher", "student", "target", "loss_name", "top1", "top5",
            "cost_params", "cost_gflops",
            "params_saved_vs_teacher_pct", "gflops_saved_vs_teacher_pct", "top1_per_gflop",
        ]),
    ])
    # Accuracy-vs-compute scatter is the primary visual for Q3
    _add_plot(lines, f"{dataset} accuracy vs compute (GFLOPs)",
              plots.get("cost"), output)
    _add_plot(lines, f"{dataset} accuracy vs compute (Params)",
              plots.get("param"), output)

    # Q4: What loss function should we use?
    lines.extend(["### Question 4: What Loss Function Should We Use?", ""])
    loss_summary = _summary_by(distilled, "loss_name")
    if not loss_summary.empty:
        best = loss_summary.iloc[0]
        lines.append(
            f"**Answer:** `{best['loss_name']}` performs best on this dataset by mean distilled "
            f"Top-1 ({_format_number(best['mean'])}%)."
        )
    else:
        lines.append("**Answer:** no loss comparison is available.")
    lines.extend(["", markdown_table(loss_summary, [
                 "loss_name", "mean", "max", "min", "count"]), ""])
    best_by_loss = _best_config_by(distilled, "loss_name")
    lines.extend([
        "Best configuration found for each loss:", "",
        markdown_table(best_by_loss, ["loss_name", "best_teacher", "best_student", "best_target",
                                      "best_top1", "best_top5", "best_minus_baseline", "best_minus_teacher"]),
        "",
    ])

    return lines


def generate_report(df: pd.DataFrame, output: Path, figures_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = add_derived_metrics(df)
    distilled = df[df["model_type"] == "distilled"].copy()
    datasets = sorted(df["dataset"].dropna().unique())
    global_plots = _make_global_plots(df, figures_dir)

    lines: list[str] = [
        "# Knowledge Distillation Evaluation Report",
        "",
        "This report is generated from JSON files in `results/evaluation`.",
        "",
        "Raw accuracies are analyzed separately per dataset. The global section only uses "
        "normalized comparisons such as improvement over the dataset baseline, improvement "
        "over the matching teacher, and cost ratios.",
        "",
        "**Plot conventions:**",
        "   - colour = student architecture (blue S-S, orange S-M, green S-L);",
        "   - overlay = distillation target (pre-GAP, post-GAP);",
        "   - edge = loss (black edge MSE+CE, no edge MSE-only);",
        "   - shape:  ",
        "       ★ = teacher;  ",
        "       ◆ = CE baseline;  ",
        "       ● = ResNet-50;  ",
        "       ■ = ConvNeXt-Tiny; ",
        "",
        "## Executive Summary By Dataset",
        "",
    ]

    for dataset in datasets:
        dataset_distilled = df[(df["dataset"] == dataset) & (
            df["model_type"] == "distilled")]
        best_student = _best_row(dataset_distilled)
        if best_student is None:
            lines.append(
                f"- `{dataset}`: no distilled student result is available.")
            continue
        lines.append(
            f"- `{dataset}`: best student is `{best_student.get('teacher')}` + "
            f"`{best_student.get('student')}` + `{best_student.get('target')}` + "
            f"`{best_student.get('loss_name')}` at {_format_number(best_student['top1'])}% top-1 "
            f"({_format_number(best_student.get('top1_minus_baseline'))} pp vs matching baseline; "
            f"{_format_number(best_student.get('gflops_saved_vs_teacher_pct'))}% GFLOPs saved vs teacher)."
        )

    lines.extend(["", "## Dataset-Specific Analysis", ""])
    for dataset in datasets:
        lines.extend(_dataset_section(
            dataset, df[df["dataset"] == dataset].copy(), output, figures_dir))
        lines.append("")

    lines.extend(["## Global Overview With Normalized Comparisons", ""])
    if not distilled.empty:
        best_per_dataset = (
            distilled.loc[distilled.groupby("dataset")["top1"].idxmax()]
            .sort_values("dataset")
        )
        lines.extend([
            "Best distilled student per dataset:", "",
            markdown_table(best_per_dataset, [
                "dataset", "teacher", "student", "target", "loss_name",
                "top1", "top1_minus_baseline", "top1_minus_teacher",
                "cost_params", "cost_gflops",
                "params_saved_vs_teacher_pct", "gflops_saved_vs_teacher_pct",
            ]),
            "",
        ])

        for group_col, label in [("teacher", "teacher"), ("target", "target"),
                                 ("student", "student architecture"), ("loss_name", "loss")]:
            norm = (
                distilled.groupby(group_col)[
                    ["top1_minus_baseline", "top1_minus_teacher",
                     "gflops_saved_vs_teacher_pct", "params_saved_vs_teacher_pct"]
                ]
                .mean()
                .reset_index()
                .sort_values("top1_minus_baseline", ascending=False)
            )
            lines.extend([
                f"Global {label} overview using normalized deltas:", "",
                markdown_table(norm, [group_col, "top1_minus_baseline", "top1_minus_teacher",
                                      "gflops_saved_vs_teacher_pct", "params_saved_vs_teacher_pct"]),
                "",
            ])

        _add_plot(lines, "Global improvement over baseline",
                  global_plots.get("baseline_delta"), output)
        _add_plot(lines, "Global accuracy vs compute",
                  global_plots.get("global_cost"), output)
        _add_plot(lines, "Global accuracy vs compute (top-5)",
                  global_plots.get("global_cost_top5"), output)
    else:
        lines.append("_No distilled student results available._")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="results/evaluation")
    parser.add_argument("--output", default="results/reports/report.md")
    parser.add_argument("--figures-dir", default="results/figures/report")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output = Path(args.output)
    figures_dir = Path(args.figures_dir)
    df = collect_results(eval_dir)
    generate_report(df, output, figures_dir)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
