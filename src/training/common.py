from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.utils.metrics import AverageMeter, accuracy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(config: dict, kind: str, suffix: str) -> Path:
    base = Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
    dataset = config["dataset"]["name"]
    teacher = config["teacher"]["name"]
    return base / kind / f"{dataset}_{teacher}_{suffix}.pt"


def results_path(config: dict, filename: str) -> Path:
    base = Path(config["outputs"].get("results_dir", "results"))
    return base / "logs" / filename


def make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    params = [parameter for parameter in model.parameters()
              if parameter.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def train_classifier_epoch(model: nn.Module, loader, optimizer, device, amp: bool, desc: str) -> dict:
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and device.type == "cuda")
    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1, acc5 = accuracy(logits.detach(), labels)
        batch = labels.size(0)
        losses.update(loss.item(), batch)
        top1.update(acc1, batch)
        top5.update(acc5, batch)
    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader, device, amp: bool, desc: str = "eval") -> dict:
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        acc1, acc5 = accuracy(logits, labels)
        batch = labels.size(0)
        losses.update(loss.item(), batch)
        top1.update(acc1, batch)
        top5.update(acc5, batch)
    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}
