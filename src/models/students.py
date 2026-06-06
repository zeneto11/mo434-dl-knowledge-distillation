from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.predictors import ConvFeaturePredictor, MLPPooledPredictor
from src.models.teachers import TeacherSpec


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


STUDENT_CHANNELS: dict[str, list[int]] = {
    "student_s": [32, 64, 128],
    "student_m": [32, 64, 128, 256],
    "student_l": [32, 64, 128, 256, 512],
}


class StudentEncoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers.append(_conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_channels = channels[-1]

    def forward_feature_map(self, x: Tensor) -> Tensor:
        return self.features(x)

    def forward_pooled(self, x: Tensor) -> Tensor:
        return torch.flatten(self.gap(self.forward_feature_map(x)), 1)


def build_student_encoder(name: str) -> StudentEncoder:
    if name not in STUDENT_CHANNELS:
        raise ValueError(f"Unknown student: {name!r}. Choose from {sorted(STUDENT_CHANNELS)}")
    return StudentEncoder(STUDENT_CHANNELS[name])


class StudentDistillationModel(nn.Module):
    def __init__(self, student_name: str, target: str, teacher_spec: TeacherSpec) -> None:
        super().__init__()
        if target not in {"pregap", "postgap"}:
            raise ValueError("target must be 'pregap' or 'postgap'")
        self.target = target
        self.encoder = build_student_encoder(student_name)
        if target == "pregap":
            self.predictor = ConvFeaturePredictor(
                self.encoder.feature_channels, teacher_spec.feature_channels)
        else:
            self.predictor = MLPPooledPredictor(
                self.encoder.feature_channels, teacher_spec.feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.target == "pregap":
            return self.predictor(self.encoder.forward_feature_map(x))
        return self.predictor(self.encoder.forward_pooled(x))


class StudentBaselineModel(nn.Module):
    def __init__(self, student_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = build_student_encoder(student_name)
        self.classifier = nn.Linear(self.encoder.feature_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder.forward_pooled(x))
