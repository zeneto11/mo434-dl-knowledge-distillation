from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import yaml


TEACHERS = ("resnet50", "convnext_tiny")
STUDENTS = ("student_s", "student_m", "student_l")
TARGETS = ("pregap", "postgap")
LOSSES = ("mse_ce", "mse")  # keep mse_ce before mse in regexes

METRICS = ("loss", "ce", "mse", "top1", "top5")

TEACHER_LABELS = {
    "resnet50": "ResNet50",
    "convnext_tiny": "ConvNeXt-Tiny",
}
STUDENT_LABELS = {
    "student_s": "Student-S",
    "student_m": "Student-M",
    "student_l": "Student-L",
}
TARGET_LABELS = {
    "pregap": "pre-GAP",
    "postgap": "post-GAP",
}
LOSS_LABELS = {
    "mse": "MSE",
    "mse_ce": "MSE+CE",
    "ce": "CE",
}

STUDENT_COLORS = {
    "student_s": "#1f77b4",
    "student_m": "#ff7f0e",
    "student_l": "#2ca02c",
}
TEACHER_COLORS = {
    "resnet50": "#9467bd",
    "convnext_tiny": "#8c564b",
}
LOSS_MARKERS = {
    "mse_ce": "o",
    "mse": "s",
}
REFERENCE_LINE_ALPHA = 0.55
REFERENCE_LABEL_FONTSIZE = 6.5


# --------------------------------------------------------------------------- #
# Loading and parsing
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _regex_union(values: Iterable[str]) -> str:
    escaped = sorted((re.escape(v) for v in values), key=len, reverse=True)
    return "|".join(escaped)


def parse_training_filename(path: Path) -> dict[str, Any] | None:
    name = path.name
    teacher_re = _regex_union(TEACHERS)
    student_re = _regex_union(STUDENTS)
    target_re = _regex_union(TARGETS)
    loss_re = _regex_union(LOSSES)

    teacher_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<teacher>{teacher_re})_teacher\.json$")
    baseline_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<student>{student_re})_baseline\.json$")
    distilled_pattern = re.compile(
        rf"^(?P<dataset>.+?)_(?P<teacher>{teacher_re})_(?P<student>{student_re})_"
        rf"(?P<target>{target_re})_(?P<loss_name>{loss_re})\.json$"
    )

    if match := distilled_pattern.match(name):
        row = match.groupdict()
        row.update({"model_type": "distilled",
                   "model": f"{row['student']} distilled"})
        return row

    if match := teacher_pattern.match(name):
        row = match.groupdict()
        row.update(
            {
                "model_type": "teacher",
                "model": row["teacher"],
                "student": None,
                "target": None,
                "loss_name": None,
            }
        )
        return row

    if match := baseline_pattern.match(name):
        row = match.groupdict()
        row.update(
            {
                "model_type": "baseline",
                "model": f"{row['student']} baseline",
                "teacher": None,
                "target": None,
                "loss_name": "ce",
            }
        )
        return row

    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _metric_from_epoch(epoch: dict[str, Any], split: str, metric: str) -> float | None:
    split_data = epoch.get(split)
    if not isinstance(split_data, dict):
        return None
    return _as_float(split_data.get(metric))


def _flatten_history(path: Path, data: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    history = data.get("history")
    if not isinstance(history, list):
        raise ValueError(f"Expected a list field named 'history' in {path}")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {
            "file": path.name,
            "checkpoint": data.get("checkpoint"),
            "epoch_index": index,            # continuous, never resets
            "epoch": item.get("epoch", index),  # local (resets per stage)
            "stage": item.get("stage"),
            **meta,
        }
        for split in ("train", "val"):
            for metric in METRICS:
                row[f"{split}_{metric}"] = _metric_from_epoch(
                    item, split, metric)
        rows.append(row)
    return rows


def collect_training_results(train_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for path in sorted(train_dir.glob("*.json")):
        meta = parse_training_filename(path)
        if meta is None:
            continue

        data = _load_json(path)
        rows = _flatten_history(path, data, meta)
        if not rows:
            continue

        per_run_epochs = pd.DataFrame(rows)
        epoch_rows.extend(rows)
        run_rows.append(_summarize_run(path, data, meta, per_run_epochs))

    if not epoch_rows:
        raise FileNotFoundError(
            f"No project training JSON files found in {train_dir}")

    runs = add_derived_metrics(pd.DataFrame(run_rows))
    epochs = pd.DataFrame(epoch_rows)
    return runs, epochs


# --------------------------------------------------------------------------- #
# Per-run summarisation
# --------------------------------------------------------------------------- #
def _min_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    return None if valid.empty else float(valid.min())


def _max_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    return None if valid.empty else float(valid.max())


def _idx_best(epoch_df: pd.DataFrame, metric: str, higher_is_better: bool) -> int | None:
    if metric not in epoch_df.columns:
        return None
    valid = epoch_df.dropna(subset=[metric])
    if valid.empty:
        return None
    return int(valid[metric].idxmax() if higher_is_better else valid[metric].idxmin())


def _selection_rule(meta: dict[str, Any], epoch_df: pd.DataFrame) -> tuple[str, bool, str]:
    model_type = meta.get("model_type")
    loss_name = meta.get("loss_name")

    if model_type == "distilled" and loss_name == "mse" and epoch_df["val_mse"].notna().any():
        return "val_mse", False, "lowest validation MSE for MSE-only distillation"
    if epoch_df["val_top1"].notna().any():
        return "val_top1", True, "highest validation Top-1"
    if epoch_df["val_loss"].notna().any():
        return "val_loss", False, "lowest validation loss fallback"
    if epoch_df["val_mse"].notna().any():
        return "val_mse", False, "lowest validation MSE fallback"
    return "epoch_index", True, "last available epoch fallback"


def _stage_list(epoch_df: pd.DataFrame) -> str:
    if "stage" not in epoch_df.columns:
        return ""
    vals = [v for v in epoch_df["stage"].dropna().unique().tolist()]
    return ",".join(str(v) for v in vals)


def _stage_transition_epoch(epoch_df: pd.DataFrame) -> float | None:
    """
    Boundary between training stages in continuous epoch-index space.

    Teacher logs reset the local `epoch` field when stage 2 starts, so stage
    boundaries and staged plots MUST use `epoch_index`, not `epoch`.
    """
    if "stage" not in epoch_df.columns or epoch_df["stage"].dropna().nunique() < 2:
        return None

    ordered = epoch_df.sort_values("epoch_index").reset_index(drop=True)
    prev_stage = ordered.loc[0, "stage"]
    prev_index = float(ordered.loc[0, "epoch_index"])

    for idx in range(1, len(ordered)):
        stage = ordered.loc[idx, "stage"]
        current_index = float(ordered.loc[idx, "epoch_index"])
        if pd.notna(stage) and stage != prev_stage:
            return (prev_index + current_index) / 2.0
        prev_stage = stage
        prev_index = current_index

    return None


def _diff(a: Any, b: Any) -> float | None:
    aa = _as_float(a)
    bb = _as_float(b)
    if aa is None or bb is None:
        return None
    return aa - bb


def _pct_increase(final_value: Any, best_value: Any) -> float | None:
    final = _as_float(final_value)
    best = _as_float(best_value)
    if final is None or best is None or abs(best) < 1e-12:
        return None
    return (final - best) / abs(best) * 100.0


def _classification_metric_available(meta: dict[str, Any], epoch_df: pd.DataFrame) -> bool:
    top1 = epoch_df.get("val_top1", pd.Series(dtype=float)).dropna()
    if top1.empty:
        return False
    if meta.get("model_type") == "distilled" and meta.get("loss_name") == "mse":
        return bool(top1.abs().sum() > 0)
    return True


def _summarize_run(path: Path, data: dict[str, Any], meta: dict[str, Any], epoch_df: pd.DataFrame) -> dict[str, Any]:
    selection_metric, higher_is_better, selection_reason = _selection_rule(
        meta, epoch_df)
    best_idx = _idx_best(epoch_df, selection_metric, higher_is_better)
    best = epoch_df.loc[best_idx] if best_idx is not None else epoch_df.iloc[-1]
    first = epoch_df.iloc[0]
    final = epoch_df.iloc[-1]

    row: dict[str, Any] = {
        "file": path.name,
        "checkpoint": data.get("checkpoint"),
        **meta,
        "epochs": int(len(epoch_df)),
        "stages": _stage_list(epoch_df),
        "stage_transition_epoch": _stage_transition_epoch(epoch_df),
        "selection_metric": selection_metric,
        "selection_reason": selection_reason,
        # LOCAL best epoch (resets per stage)
        "best_epoch": best.get("epoch"),
        "best_stage": best.get("stage"),
        # CONTINUOUS best epoch
        "best_epoch_index": int(best.get("epoch_index")) if pd.notna(best.get("epoch_index")) else None,
        "best_stage_int": int(best.get("stage")) if pd.notna(best.get("stage")) else None,
    }

    for prefix, source in (("first", first), ("best", best), ("final", final)):
        for split in ("train", "val"):
            for metric in METRICS:
                row[f"{prefix}_{split}_{metric}"] = source.get(
                    f"{split}_{metric}")

    test = data.get("test") or {}
    if isinstance(test, dict):
        for metric in METRICS:
            row[f"test_{metric}"] = _as_float(test.get(metric))

    for metric in METRICS:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        row[f"max_val_{metric}"] = _max_valid(
            epoch_df[val_col]) if val_col in epoch_df else None
        row[f"min_val_{metric}"] = _min_valid(
            epoch_df[val_col]) if val_col in epoch_df else None

    row["val_top1_final_drop"] = _diff(
        row.get("max_val_top1"), row.get("final_val_top1"))
    row["val_loss_reduction"] = _diff(
        row.get("first_val_loss"), row.get("final_val_loss"))
    row["train_loss_reduction"] = _diff(
        row.get("first_train_loss"), row.get("final_train_loss"))
    row["val_mse_increase_from_min"] = _diff(
        row.get("final_val_mse"), row.get("min_val_mse"))

    for prefix in ("best", "final"):
        row[f"{prefix}_gap_top1_train_minus_val"] = _diff(
            row.get(f"{prefix}_train_top1"), row.get(f"{prefix}_val_top1"))
        row[f"{prefix}_gap_loss_val_minus_train"] = _diff(
            row.get(f"{prefix}_val_loss"), row.get(f"{prefix}_train_loss"))
        row[f"{prefix}_gap_mse_val_minus_train"] = _diff(
            row.get(f"{prefix}_val_mse"), row.get(f"{prefix}_train_mse"))

    row["classification_metric_available"] = _classification_metric_available(
        meta, epoch_df)

    return row


# --------------------------------------------------------------------------- #
# Diagnostics -- trimmed to critical warnings only
# --------------------------------------------------------------------------- #
def diagnose_run(row: pd.Series) -> tuple[list[str], str]:
    """Return (critical_warnings, concern_level).

    Only genuinely actionable problems are surfaced. Everything informational
    (e.g. MSE-only runs having no Top-1) is intentionally dropped.
    """
    warnings: list[str] = []

    model_type = row.get("model_type")
    loss_name = row.get("loss_name")
    classification_available = bool(row.get("classification_metric_available"))

    top1_drop = _as_float(row.get("val_top1_final_drop")) or 0.0
    top1_gap = _as_float(row.get("final_gap_top1_train_minus_val")) or 0.0
    loss_increase_pct = _as_float(
        row.get("val_loss_increase_from_min_pct")) or 0.0
    mse_increase_pct = _as_float(
        row.get("val_mse_increase_from_min_pct")) or 0.0
    train_loss_reduction = _as_float(row.get("train_loss_reduction"))

    severe = 0

    if classification_available:
        if top1_drop >= 2.0:
            warnings.append(
                f"val Top-1 fell {top1_drop:.1f} pp from its peak (overfitting / late-epoch decay)")
        if top1_gap >= 15.0:
            warnings.append(
                f"train-val Top-1 gap {top1_gap:.1f} pp (overfitting)")
            severe += 1

    if loss_increase_pct >= 15.0:
        warnings.append(
            f"val loss regressed {loss_increase_pct:.0f}% above its best")
        severe += 1

    if model_type == "distilled" and pd.notna(row.get("final_val_mse")) and mse_increase_pct >= 10.0:
        warnings.append(
            f"val MSE regressed {mse_increase_pct:.0f}% above its best")
        severe += 1

    if train_loss_reduction is not None and train_loss_reduction <= 0:
        warnings.append("train loss did not decrease (optimization failure)")
        severe += 1

    if severe >= 2:
        concern = "high"
    elif warnings:
        concern = "medium"
    else:
        concern = "low"

    return warnings, concern


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["val_mse_increase_from_min_pct"] = enriched.apply(
        lambda r: _pct_increase(r.get("final_val_mse"), r.get("min_val_mse")), axis=1
    )
    enriched["val_loss_increase_from_min_pct"] = enriched.apply(
        lambda r: _pct_increase(r.get("final_val_loss"), r.get("min_val_loss")), axis=1
    )

    all_warnings: list[str] = []
    all_concern: list[str] = []
    for _, row in enriched.iterrows():
        warnings, concern = diagnose_run(row)
        all_warnings.append(" | ".join(warnings))
        all_concern.append(concern)

    enriched["warnings"] = all_warnings
    enriched["concern_level"] = all_concern

    # gains relative to matching baseline / teacher
    baseline_top1 = (
        enriched[enriched["model_type"] == "baseline"]
        .dropna(subset=["best_val_top1"])
        .set_index(["dataset", "student"])["best_val_top1"]
        .to_dict()
        if "best_val_top1" in enriched.columns else {}
    )
    enriched["vs_baseline_top1"] = None
    for idx, row in enriched.iterrows():
        key = (row.get("dataset"), row.get("student"))
        # Only meaningful when the run actually has classification Top-1.
        if (pd.notna(row.get("best_val_top1")) and key in baseline_top1
                and bool(row.get("classification_metric_available"))):
            enriched.at[idx, "vs_baseline_top1"] = row["best_val_top1"] - \
                baseline_top1[key]

    return enriched


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def _format_number(value: Any, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if precision == 0:
            return str(int(round(value)))
        return f"{value:.{precision}f}"
    return str(value)


# Friendly short headers so the tables stay narrow and readable.
COLUMN_HEADERS = {
    "teacher": "Teacher",
    "student": "Student",
    "target": "Target",
    "loss_name": "Loss",
    "epochs": "Ep",
    "stages": "Stages",
    "best_epoch_index": "Best@",
    "best_stage": "BestStg",
    "best_stage_int": "Stg",
    "selection_metric": "SelBy",
    "best_val_top1": "Best T1",
    "final_val_top1": "Fin T1",
    "val_top1_final_drop": "T1 drop",
    "best_val_mse": "Best MSE",
    "final_val_mse": "Fin MSE",
    "val_mse_increase_from_min_pct": "MSE reg%",
    "final_gap_top1_train_minus_val": "T1 gap",
    "vs_baseline_top1": "vs base",
    "best_val_loss": "Best L",
    "final_val_loss": "Fin L",
    "concern_level": "Concern",
    "mean": "Mean",
    "best": "Best",
    "worst": "Worst",
    "count": "N",
}

# precision overrides per column
COLUMN_PRECISION = {
    "best_val_mse": 4,
    "final_val_mse": 4,
    "mean": 3,
    "best": 3,
    "worst": 3,
    "best_epoch_index": 0,
    "best_stage_int": 0,
    "epochs": 0,
    "count": 0,
    "val_mse_increase_from_min_pct": 1,
}


def markdown_table(df: pd.DataFrame, columns: list[str], default_precision: int = 2,
                   max_rows: int | None = None, mse_precision: bool = False) -> str:
    available = [c for c in columns if c in df.columns]
    if not available or df.empty:
        return "_No data available._"

    table = df[available].copy()
    if max_rows is not None:
        table = table.head(max_rows)

    for column in table.columns:
        prec = COLUMN_PRECISION.get(column, default_precision)
        table[column] = table[column].map(
            lambda v, p=prec: _format_number(v, p))

    headers = [COLUMN_HEADERS.get(c, str(c)) for c in table.columns]
    rows = table.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]

    header = "| " + " | ".join(headers[i].ljust(widths[i])
                               for i in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[i]
                                  for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i])
                              for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _sort_existing(df: pd.DataFrame, columns: list[str], ascending: bool | list[bool] = True) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    if isinstance(ascending, list):
        ascending = ascending[: len(cols)]
    return df.sort_values(cols, ascending=ascending, na_position="last")


def _best_row(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None
    return valid.loc[valid[metric].idxmax() if higher_is_better else valid[metric].idxmin()]


def _classification_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "classification_metric_available" not in df.columns:
        return pd.DataFrame()
    return df[(df["classification_metric_available"] == True) & df["best_val_top1"].notna()].copy()


def _distilled_mse_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[(df["model_type"] == "distilled") & df["best_val_mse"].notna()].copy()


def _relative_path(path: str | Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).lower()).strip("_")


def _compact_run_name(row: pd.Series | None) -> str:
    if row is None:
        return ""
    if row.get("model_type") == "teacher":
        return f"{row.get('teacher')}"
    if row.get("model_type") == "baseline":
        return f"{row.get('student')} baseline"
    return "/".join(
        str(x) for x in [row.get("teacher"), row.get("student"),
                         row.get("target"), row.get("loss_name")]
        if pd.notna(x)
    )


_ABBREV = {
    "resnet50": "RN50", "convnext_tiny": "CvN",
    "student_s": "S-S", "student_m": "S-M", "student_l": "S-L",
    "pregap": "pre", "postgap": "post",
    "mse": "MSE", "mse_ce": "MSE+CE", "ce": "CE",
}


def _abbrev_run_name(row: pd.Series | None) -> str:
    """Compact run label for narrow tables, e.g. 'CvN/S-L/post/MSE+CE'."""
    if row is None:
        return ""
    if row.get("model_type") == "teacher":
        return _ABBREV.get(str(row.get("teacher")), str(row.get("teacher")))
    if row.get("model_type") == "baseline":
        return f"{_ABBREV.get(str(row.get('student')), row.get('student'))} base"
    parts = [row.get("teacher"), row.get("student"),
             row.get("target"), row.get("loss_name")]
    return "/".join(_ABBREV.get(str(x), str(x)) for x in parts if pd.notna(x))


def _add_plot(lines: list[str], title: str, path: str | None, output: Path) -> None:
    if path:
        lines.extend(
            ["", f"![{title}]({_relative_path(path, output.parent)})", ""])


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _curve_for_file(epochs: pd.DataFrame, file_name: str, metric: str) -> pd.DataFrame:
    """One metric curve for one run, sorted by continuous epoch_index."""
    if epochs.empty or metric not in epochs.columns:
        return pd.DataFrame()
    return (
        epochs[epochs["file"] == file_name]
        .dropna(subset=[metric])
        .sort_values("epoch_index")
        .copy()
    )


def _draw_stage_boundary(ax: plt.Axes, run_epochs: pd.DataFrame) -> None:
    boundary = int(_stage_transition_epoch(run_epochs))
    if boundary is None:
        return
    ax.axvline(boundary, color="#444444", linestyle="--",
               linewidth=1.0, alpha=0.65)
    ymin, ymax = ax.get_ylim()
    ax.text(boundary, ymax, " stage 2", rotation=90, va="top", ha="left",
            fontsize=8, color="#444444")


def _plot_two_stage_curves(dataset: str, runs: pd.DataFrame, epochs: pd.DataFrame,
                           figures_dir: Path, *, model_type: str, group_col: str,
                           color_map: dict[str, str], label_map: dict[str, str],
                           title: str, mark_stages: bool) -> str | None:
    group_runs = runs[(runs["dataset"] == dataset) &
                      (runs["model_type"] == model_type)].copy()
    if group_runs.empty:
        return None

    fig, (ax_top1, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.8))
    any_curve = False
    boundary_epochs: list[pd.DataFrame] = []

    for _, run in _sort_existing(group_runs, [group_col]).iterrows():
        key = str(run.get(group_col))
        color = color_map.get(key, "#333333")
        label = label_map.get(key, key)

        run_epochs = (epochs[epochs["file"] == run["file"]]
                      .sort_values("epoch_index").copy())
        if mark_stages:
            boundary_epochs.append(run_epochs)

        for ax, metric in ((ax_top1, "top1"), (ax_loss, "loss")):
            train_col, val_col = f"train_{metric}", f"val_{metric}"
            sub = run_epochs.dropna(subset=[train_col, val_col], how="all")
            if sub.empty:
                continue
            if sub[train_col].notna().any():
                ax.plot(sub["epoch_index"], sub[train_col], color=color,
                        linestyle=":", marker="o", markersize=3,
                        linewidth=1.5, label=f"{label} train")
            if sub[val_col].notna().any():
                ax.plot(sub["epoch_index"], sub[val_col], color=color,
                        linestyle="-", marker="o", markersize=3,
                        markeredgecolor="white", markeredgewidth=0.5,
                        linewidth=2.0, label=f"{label} val")
            any_curve = True

    if not any_curve:
        plt.close(fig)
        return None

    # Draw stage boundaries AFTER axes are populated
    if mark_stages:
        for re_df in boundary_epochs:
            _draw_stage_boundary(ax_top1, re_df)
            _draw_stage_boundary(ax_loss, re_df)

    ax_top1.set_title(f"{dataset}: {title} Top-1")
    ax_top1.set_xlabel("Epochs")
    ax_top1.set_ylabel("Top-1 (%)")
    ax_top1.grid(alpha=0.25)
    ax_top1.legend(fontsize=8)

    ax_loss.set_title(f"{dataset}: {title} loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=8)

    if mark_stages:
        fig.suptitle(
            "Dashed line = transition from frozen-head training to last-block fine-tuning",
            fontsize=10)
    plt.tight_layout()

    suffix = "teacher" if model_type == "teacher" else "baseline"
    path = figures_dir / f"{_safe_name(dataset)}_{suffix}_training_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _legend_handles_for_distilled_plot() -> list[Line2D]:
    handles: list[Line2D] = []
    for student, color in STUDENT_COLORS.items():
        handles.append(Line2D([0], [0], color=color, lw=2,
                       label=STUDENT_LABELS.get(student, student)))
    for loss_name, marker in LOSS_MARKERS.items():
        handles.append(Line2D([0], [0], color="#222222", marker=marker,
                       lw=0, label=LOSS_LABELS.get(loss_name, loss_name)))
    handles.append(Line2D([0], [0], color="black",
                   lw=2.0, linestyle="-", label="Validation"))
    handles.append(Line2D([0], [0], color="black",
                   lw=2.0, linestyle=":", label="Training"))
    return handles


def _annotated_reference_line(ax: plt.Axes, y: float, *, color: str, label: str,
                              linewidth: float, alpha: float = REFERENCE_LINE_ALPHA) -> None:
    ax.axhline(y, color=color, linewidth=linewidth,
               alpha=alpha, linestyle="-", zorder=0)
    ax.text(0.01, y, label, transform=ax.get_yaxis_transform(), color=color,
            fontsize=REFERENCE_LABEL_FONTSIZE, ha="left", va="bottom", alpha=0.95,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.65, "pad": 0.6},
            zorder=3)


def _draw_reference_curves(ax: plt.Axes, dataset: str, teacher: str,
                           runs: pd.DataFrame) -> None:
    teacher_rows = runs[(runs["dataset"] == dataset) &
                        (runs["model_type"] == "teacher") &
                        (runs["teacher"] == teacher)]
    for _, row in teacher_rows.iterrows():
        best_top1 = _as_float(row.get("best_val_top1"))
        if best_top1 is None:
            continue
        _annotated_reference_line(
            ax, best_top1, color=TEACHER_COLORS.get(teacher, "#333333"),
            linewidth=2.0, label=f"{TEACHER_LABELS.get(teacher, teacher)} teacher")

    baseline_rows = runs[(runs["dataset"] == dataset) &
                         (runs["model_type"] == "baseline")]
    for _, row in baseline_rows.iterrows():
        best_top1 = _as_float(row.get("best_val_top1"))
        if best_top1 is None:
            continue
        student = str(row.get("student"))
        _annotated_reference_line(
            ax, best_top1, color=STUDENT_COLORS.get(student, "#333333"),
            linewidth=1, label=f"{STUDENT_LABELS.get(student, student)} baseline")


def _plot_distilled_by_teacher(dataset: str, runs: pd.DataFrame, epochs: pd.DataFrame,
                               figures_dir: Path, *, metric: str, ylabel: str,
                               include_references: bool) -> str | None:
    dataset_runs = runs[runs["dataset"] == dataset].copy()
    distilled = dataset_runs[dataset_runs["model_type"] == "distilled"].copy()
    if distilled.empty or metric not in epochs.columns:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharey=True)
    panel_map = {
        ("convnext_tiny", "pregap"): axes[0, 0],
        ("convnext_tiny", "postgap"): axes[0, 1],
        ("resnet50", "pregap"): axes[1, 0],
        ("resnet50", "postgap"): axes[1, 1],
    }
    any_curve = False

    for (teacher, target), ax in panel_map.items():
        teacher_runs = distilled[(distilled["teacher"] == teacher) &
                                 (distilled["target"] == target)].copy()
        if include_references:
            _draw_reference_curves(ax, dataset, teacher, runs)

        for _, run in _sort_existing(teacher_runs, ["student", "loss_name"]).iterrows():
            curve = _curve_for_file(epochs, run["file"], metric)
            if curve.empty:
                continue
            if (metric == "val_top1" and run.get("loss_name") == "mse"
                    and not bool(run.get("classification_metric_available"))):
                continue

            student = str(run.get("student"))
            color = STUDENT_COLORS.get(student, "#333333")
            marker = LOSS_MARKERS.get(run["loss_name"], "o")

            ax.plot(curve["epoch_index"], curve[metric], color=color,
                    linestyle="-", linewidth=2.0, marker=marker, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5, alpha=0.95)

            train_metric = metric.replace("val_", "train_")
            if train_metric in curve.columns:
                tc = curve.dropna(subset=[train_metric])
                if not tc.empty:
                    ax.plot(tc["epoch_index"], tc[train_metric], color=color,
                            linestyle=":", linewidth=1.5, marker=marker,
                            markersize=3, alpha=0.9)
            any_curve = True

        ax.set_title(
            f"{TEACHER_LABELS.get(teacher, teacher)} — {TARGET_LABELS.get(target, target)}")
        ax.set_xlabel("Epochs")
        ax.grid(alpha=0.25)

    if not any_curve:
        plt.close(fig)
        return None

    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)

    fig.legend(handles=_legend_handles_for_distilled_plot(),
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.10, 1, 0.92))

    path = figures_dir / \
        f"{_safe_name(dataset)}_{metric}_distilled_by_teacher.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _make_dataset_plots(dataset: str, runs: pd.DataFrame, epochs: pd.DataFrame,
                        figures_dir: Path) -> dict[str, str]:
    plots: dict[str, str] = {}

    teacher_plot = _plot_two_stage_curves(
        dataset, runs, epochs, figures_dir, model_type="teacher",
        group_col="teacher", color_map=TEACHER_COLORS, label_map=TEACHER_LABELS,
        title="teacher", mark_stages=True)
    if teacher_plot:
        plots["teacher_training"] = teacher_plot

    baseline_plot = _plot_two_stage_curves(
        dataset, runs, epochs, figures_dir, model_type="baseline",
        group_col="student", color_map=STUDENT_COLORS, label_map=STUDENT_LABELS,
        title="CE student-baseline", mark_stages=False)
    if baseline_plot:
        plots["baseline_training"] = baseline_plot

    top1_plot = _plot_distilled_by_teacher(
        dataset, runs, epochs, figures_dir, metric="val_top1",
        ylabel="Validation Top-1 (%)", include_references=True)
    if top1_plot:
        plots["distilled_top1_by_teacher"] = top1_plot

    mse_plot = _plot_distilled_by_teacher(
        dataset, runs, epochs, figures_dir, metric="val_mse",
        ylabel="Validation MSE", include_references=False)
    if mse_plot:
        plots["distilled_mse_by_teacher"] = mse_plot

    return plots


# --------------------------------------------------------------------------- #
# Summary tables
# --------------------------------------------------------------------------- #
def _executive_summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(runs["dataset"].dropna().unique()):
        d = runs[runs["dataset"] == dataset].copy()
        distilled = d[d["model_type"] == "distilled"]

        best_teacher = _best_row(
            d[d["model_type"] == "teacher"], "best_val_top1", True)
        best_baseline = _best_row(
            d[d["model_type"] == "baseline"], "best_val_top1", True)
        best_cls = _best_row(_classification_df(
            distilled), "best_val_top1", True)
        best_mse = _best_row(_distilled_mse_df(
            distilled), "best_val_mse", False)

        rows.append({
            "dataset": dataset,
            "best_teacher": _abbrev_run_name(best_teacher),
            "teacher_val_top1": best_teacher.get("best_val_top1") if best_teacher is not None else None,
            "best_baseline": _abbrev_run_name(best_baseline),
            "baseline_val_top1": best_baseline.get("best_val_top1") if best_baseline is not None else None,
            "best_distilled_run": _abbrev_run_name(best_cls),
            "distilled_val_top1": best_cls.get("best_val_top1") if best_cls is not None else None,
            "lowest_mse_run": _abbrev_run_name(best_mse),
            "best_val_mse": best_mse.get("best_val_mse") if best_mse is not None else None,
            "concern_high": int((d["concern_level"] == "high").sum()),
        })
    return pd.DataFrame(rows)


EXEC_HEADERS = {
    "dataset": "Dataset",
    "best_teacher": "Best teacher",
    "teacher_val_top1": "T.T1",
    "best_baseline": "Best baseline",
    "baseline_val_top1": "B.T1",
    "best_distilled_run": "Best distilled (T1)",
    "distilled_val_top1": "D.T1",
    "lowest_mse_run": "Lowest MSE run",
    "best_val_mse": "MSE",
    "concern_high": "High-concern",
}


def _slim_global_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One combined table: mean best val Top-1 (classification runs) and mean
    best val MSE (distilled runs) for every factor value across both datasets."""
    distilled = runs[runs["model_type"] == "distilled"].copy()
    cls = _classification_df(distilled)
    mse = _distilled_mse_df(distilled)

    rows: list[dict[str, Any]] = []
    factors = [("teacher", "Teacher"), ("target", "Target"),
               ("student", "Student"), ("loss_name", "Loss")]
    for col, factor_label in factors:
        values = sorted(set(distilled[col].dropna().unique()))
        for v in values:
            cls_vals = cls[cls[col] == v]["best_val_top1"].dropna()
            mse_vals = mse[mse[col] == v]["best_val_mse"].dropna()
            label_map = {"teacher": TEACHER_LABELS, "student": STUDENT_LABELS,
                         "target": TARGET_LABELS, "loss_name": LOSS_LABELS}.get(col, {})
            rows.append({
                "factor": factor_label,
                "value": label_map.get(v, v),
                "mean_top1": float(cls_vals.mean()) if not cls_vals.empty else None,
                "best_top1": float(cls_vals.max()) if not cls_vals.empty else None,
                "mean_mse": float(mse_vals.mean()) if not mse_vals.empty else None,
                "best_mse": float(mse_vals.min()) if not mse_vals.empty else None,
                "n": int(max(len(cls_vals), len(mse_vals))),
            })
    return pd.DataFrame(rows)


GLOBAL_HEADERS = {
    "factor": "Factor", "value": "Value", "mean_top1": "Mean T1",
    "best_top1": "Best T1", "mean_mse": "Mean MSE", "best_mse": "Best MSE", "n": "N",
}


def _render_named_table(df: pd.DataFrame, headers: dict[str, str],
                        precision: dict[str, int]) -> str:
    if df.empty:
        return "_No data available._"
    cols = list(df.columns)
    table = df.copy()
    for c in cols:
        p = precision.get(c, 2)
        table[c] = table[c].map(lambda v, pp=p: _format_number(v, pp))
    head = [headers.get(c, c) for c in cols]
    rows = table.astype(str).values.tolist()
    widths = [max(len(head[i]), *(len(r[i]) for r in rows))
              for i in range(len(head))]
    header = "| " + " | ".join(head[i].ljust(widths[i])
                               for i in range(len(head))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(head))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i])
                              for i in range(len(head))) + " |" for r in rows]
    return "\n".join([header, sep, *body])


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _critical_warnings_block(lines: list[str], df: pd.DataFrame, max_items: int = 8) -> None:
    flagged = df[df["concern_level"].isin(["high", "medium"])].copy()
    flagged = flagged[flagged["warnings"].str.len() > 0]
    if flagged.empty:
        lines.append("No critical training problems were detected.")
        lines.append("")
        return

    # If one warning pattern dominates many runs, summarise it once instead of repeating it per run.
    pattern_counts = flagged["warnings"].value_counts()
    widespread = pattern_counts[pattern_counts >= max_items]
    for pattern, count in widespread.items():
        lines.append(f"- _{count} runs_: {pattern}")
    remaining = flagged[~flagged["warnings"].isin(widespread.index)]

    order = {"high": 0, "medium": 1}
    remaining = remaining.assign(
        _o=remaining["concern_level"].map(order).fillna(2)).sort_values("_o")

    shown = remaining.head(max_items)
    for _, row in shown.iterrows():
        lines.append(
            f"- **{_compact_run_name(row)}** (`{row.get('concern_level')}`): {row.get('warnings')}")
    extra = len(remaining) - len(shown)
    if extra > 0:
        lines.append(
            f"- _…and {extra} more flagged run(s); see the table above._")
    lines.append("")


def _teacher_section(runs: pd.DataFrame, epochs: pd.DataFrame, output: Path,
                     figures_dir: Path) -> list[str]:
    lines = ["## Teacher Training", ""]
    lines.append(
        "Teacher adaptation is two-stage: the dataset head is trained with the "
        "encoder frozen, then the last encoder block is fine-tuned. The dashed "
        "line in the plots marks that transition. **`Best@` is the continuous "
        "epoch index**, so a stage-2 best shows its true position (e.g. 45), not "
        "the local stage-2 counter (e.g. 15).")
    lines.append("")

    for dataset in sorted(runs["dataset"].dropna().unique()):
        d = runs[(runs["dataset"] == dataset) & (
            runs["model_type"] == "teacher")].copy()
        if d.empty:
            continue
        plots = _make_dataset_plots(dataset, runs, epochs, figures_dir)
        lines.extend([f"### {dataset}", ""])
        lines.append(markdown_table(
            _sort_existing(d, ["teacher"]),
            ["teacher", "epochs", "best_epoch_index", "best_stage_int",
             "best_val_top1", "final_val_top1", "val_top1_final_drop",
             "best_val_loss", "final_val_loss", "concern_level"]))
        _add_plot(lines, f"{dataset} teacher training curves",
                  plots.get("teacher_training"), output)
    return lines


def _baseline_section(runs: pd.DataFrame, epochs: pd.DataFrame, output: Path,
                      figures_dir: Path) -> list[str]:
    lines = ["## Student Baselines (CE)", ""]
    lines.append(
        "Supervised CE students are the reference distillation must beat. "
        "`vs base` columns elsewhere are measured against these.")
    lines.append("")

    for dataset in sorted(runs["dataset"].dropna().unique()):
        d = runs[(runs["dataset"] == dataset) & (
            runs["model_type"] == "baseline")].copy()
        if d.empty:
            continue
        plots = _make_dataset_plots(dataset, runs, epochs, figures_dir)
        lines.extend([f"### {dataset}", ""])
        lines.append(markdown_table(
            _sort_existing(d, ["student"]),
            ["student", "epochs", "best_epoch_index", "best_val_top1",
             "final_val_top1", "val_top1_final_drop", "best_val_loss",
             "final_val_loss", "concern_level"]))
        _add_plot(lines, f"{dataset} baseline training curves",
                  plots.get("baseline_training"), output)
    return lines


def _dataset_section(dataset: str, runs: pd.DataFrame, epochs: pd.DataFrame,
                     output: Path, figures_dir: Path) -> list[str]:
    d = runs[runs["dataset"] == dataset].copy()
    distilled = d[d["model_type"] == "distilled"].copy()
    classification = _classification_df(distilled)
    mse_runs = _distilled_mse_df(distilled)
    plots = _make_dataset_plots(dataset, runs, epochs, figures_dir)

    lines = [f"## Distillation — {dataset}", ""]

    best_cls = _best_row(classification, "best_val_top1", True)
    best_mse = _best_row(mse_runs, "best_val_mse", False)
    if best_cls is not None:
        lines.append(
            f"- Best classification run: `{_compact_run_name(best_cls)}` — "
            f"{_format_number(best_cls.get('best_val_top1'))}% val Top-1.")
    if best_mse is not None:
        lines.append(
            f"- Lowest-MSE run: `{_compact_run_name(best_mse)}` — "
            f"{_format_number(best_mse.get('best_val_mse'), 4)} val MSE.")
    lines.append("")

    # Big combined table
    lines.extend(["### All distilled runs", ""])
    lines.append(markdown_table(
        _sort_existing(
            distilled, ["teacher", "student", "target", "loss_name"]),
        ["teacher", "student", "target", "loss_name", "epochs", "best_epoch_index",
         "best_val_top1", "final_val_top1", "val_top1_final_drop",
         "best_val_mse", "final_val_mse", "val_mse_increase_from_min_pct",
         "final_gap_top1_train_minus_val", "vs_baseline_top1", "concern_level"]))
    lines.append("")

    # Critical warnings (replaces the old flag summary + run-cards machinery)
    lines.extend(["### Critical warnings", ""])
    _critical_warnings_block(lines, distilled)

    # Plots
    lines.extend(["### Training curves", ""])
    lines.append(
        "Top-1 (4 panels, teacher × target). Grey lines are teacher and CE-baseline "
        "references; coloured lines are distilled students. MSE-only runs with "
        "placeholder zero Top-1 are omitted here and shown in the MSE plot.")
    _add_plot(lines, f"{dataset} distilled val Top-1 by teacher",
              plots.get("distilled_top1_by_teacher"), output)
    lines.append(
        "Validation MSE for all distilled runs (lower = better matching).")
    _add_plot(lines, f"{dataset} distilled val MSE by teacher",
              plots.get("distilled_mse_by_teacher"), output)

    return lines


def _experiment_config_section() -> list[str]:
    config_dir = Path("configs")

    configs = {}

    for config_file in sorted(config_dir.glob("*.yaml")):
        with config_file.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        key = f"{cfg['dataset']['name']}_{cfg['teacher']['name']}"

        configs[key] = {
            "image_size": cfg["dataset"]["image_size"],
            "batch_size": cfg["dataset"]["batch_size"],
            "num_workers": cfg["dataset"]["num_workers"],
            "stage1_epochs": cfg["teacher"]["stage1_epochs"],
            "stage2_epochs": cfg["teacher"]["stage2_epochs"],
            "stage2_lr": cfg["teacher"]["stage2_lr"],
            "epochs": cfg["training"]["epochs"],
            "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "scheduler": cfg["training"]["scheduler"],
            "alpha": cfg["training"]["alpha"],
            "beta": cfg["training"]["beta"],
            "amp": cfg["training"]["amp"],
            "seed": cfg["training"]["seed"],
        }

    lines = [
        "## Experiment Configuration",
        "",
        "Training hyperparameters loaded directly from `configs/*.yaml`.",
        "",
    ]

    columns = list(configs.keys())

    header = "| Parameter | " + " | ".join(columns) + " |"
    separator = "|-----------|" + "|".join(["---"] * len(columns)) + "|"

    lines.append(header)
    lines.append(separator)

    parameters = list(next(iter(configs.values())).keys())

    for param in parameters:
        row = [param]
        for cfg in configs.values():
            row.append(str(cfg[param]))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    return lines


def generate_report(runs: pd.DataFrame, epochs: pd.DataFrame, output: Path,
                    figures_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(runs["dataset"].dropna().unique())

    lines: list[str] = [
        "# Knowledge Distillation — Training Report",
        "",
        "Generated from JSON logs in `results/training_logs`. This report inspects "
        "*training dynamics and checkpoint quality* — final test metrics, parameter "
        "counts, GFLOPs and latency live in the separate evaluation report.",
        "",
        "**Checkpoint selection:** teachers, CE baselines and MSE+CE students are "
        "selected by highest validation Top-1; MSE-only students by lowest "
        "validation MSE (Top-1 can be a zero placeholder during training and is not "
        "meaningful for them).",
        "",
    ]
    lines.extend(_experiment_config_section())

    lines.extend([
        "## Executive Summary",
        "",
        _render_named_table(
            _executive_summary_table(runs),
            EXEC_HEADERS,
            {"teacher_val_top1": 2, "baseline_val_top1": 2, "distilled_val_top1": 2,
             "best_val_mse": 4, "concern_high": 0}),
        "",
    ])

    lines.extend(_teacher_section(runs, epochs, output, figures_dir))
    lines.append("")
    lines.extend(_baseline_section(runs, epochs, output, figures_dir))
    lines.append("")

    for dataset in datasets:
        lines.extend(_dataset_section(
            dataset, runs, epochs, output, figures_dir))
        lines.append("")

    # Slim global summary -- one combined table instead of 8 repeated ones.
    lines.extend([
        "## Global Factor Summary",
        "",
        "Averaged across both datasets. Top-1 columns use classification-bearing "
        "runs (CE / MSE+CE); MSE columns use all distilled runs with MSE logs.",
        "",
        _render_named_table(_slim_global_table(runs), GLOBAL_HEADERS,
                            {"mean_top1": 2, "best_top1": 2, "mean_mse": 4,
                             "best_mse": 4, "n": 0}),
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a concise training-diagnostics report from project training JSON logs.")
    parser.add_argument("--train-dir", default="results/training_logs")
    parser.add_argument(
        "--output", default="results/reports/training_report.md")
    parser.add_argument(
        "--figures-dir", default="results/figures/training_report")
    parser.add_argument(
        "--summary-csv", default="results/reports/training_summary.csv")
    parser.add_argument(
        "--epoch-csv", default="results/reports/training_epochs.csv")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output = Path(args.output)
    figures_dir = Path(args.figures_dir)
    summary_csv = Path(args.summary_csv)
    epoch_csv = Path(args.epoch_csv)

    runs, epochs = collect_training_results(train_dir)
    generate_report(runs, epochs, output, figures_dir)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    epoch_csv.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summary_csv, index=False)
    epochs.to_csv(epoch_csv, index=False)

    print(f"Wrote {output}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {epoch_csv}")


if __name__ == "__main__":
    main()
