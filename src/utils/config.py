from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config


def config_stem(config: dict[str, Any]) -> str:
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return f"{dataset}_{teacher}"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
