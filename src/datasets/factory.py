from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DATASET_CLASSES = {
    "aircraft": 100,
    "food101": 101,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int


def get_num_classes(dataset_name: str) -> int:
    try:
        return DATASET_CLASSES[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def _normalize_transform(image_size: int, train: bool) -> transforms.Compose:
    interpolation = transforms.InterpolationMode.BICUBIC
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def _aircraft(root: Path, split: str, image_size: int, limit: int | None) -> Dataset:
    dataset = datasets.FGVCAircraft(
        root=str(root),
        split=split,
        annotation_level="variant",
        transform=_normalize_transform(image_size, train=split == "train"),
        download=True,
    )
    return _limit(dataset, limit)


def _food101(root: Path, split: str, image_size: int, limit: int | None) -> Dataset:
    torchvision_split = "train" if split in {"train", "val"} else "test"
    dataset = datasets.Food101(
        root=str(root),
        split=torchvision_split,
        transform=_normalize_transform(image_size, train=split == "train"),
        download=True,
    )
    if split == "val":
        return _limit(Subset(dataset, range(0, len(dataset), 10)), limit)
    if split == "train":
        train_indices = [idx for idx in range(len(dataset)) if idx % 10 != 0]
        return _limit(Subset(dataset, train_indices), limit)
    return _limit(dataset, limit)


def build_dataloaders(config: dict) -> DataBundle:
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"]
    root = Path(dataset_cfg.get("root", "data"))
    image_size = int(dataset_cfg.get("image_size", 224))
    batch_size = int(dataset_cfg.get("batch_size", 32))
    num_workers = int(dataset_cfg.get("num_workers", 4))

    if name == "aircraft":
        train_ds = _aircraft(root, "train", image_size, dataset_cfg.get("train_limit"))
        val_ds = _aircraft(root, "val", image_size, dataset_cfg.get("val_limit"))
        test_ds = _aircraft(root, "test", image_size, dataset_cfg.get("test_limit"))
    elif name == "food101":
        train_ds = _food101(root, "train", image_size, dataset_cfg.get("train_limit"))
        val_ds = _food101(root, "val", image_size, dataset_cfg.get("val_limit"))
        test_ds = _food101(root, "test", image_size, dataset_cfg.get("test_limit"))
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    return DataBundle(
        train=DataLoader(train_ds, shuffle=True, **loader_kwargs),
        val=DataLoader(val_ds, shuffle=False, **loader_kwargs),
        test=DataLoader(test_ds, shuffle=False, **loader_kwargs),
        num_classes=get_num_classes(name),
    )
