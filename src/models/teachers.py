from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet50,
)


@dataclass(frozen=True)
class TeacherSpec:
    feature_dim: int
    feature_channels: int


class TeacherClassifier(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        if name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.pool = model.avgpool
            feature_dim = model.fc.in_features
            feature_channels = feature_dim
        elif name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_tiny(weights=weights)
            self.encoder = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = model.classifier[2].in_features
            feature_channels = feature_dim
        else:
            raise ValueError(f"Unsupported teacher: {name}")

        self.spec = TeacherSpec(feature_dim=feature_dim, feature_channels=feature_channels)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.encoder(x)
        pooled = torch.flatten(self.pool(feature_map), 1)
        return feature_map, pooled

    def classify_feature_map(self, feature_map: Tensor) -> Tensor:
        pooled = torch.flatten(self.pool(feature_map), 1)
        return self.classifier(pooled)

    def classify_pooled(self, pooled: Tensor) -> Tensor:
        return self.classifier(pooled)

    def forward(self, x: Tensor) -> Tensor:
        _, pooled = self.forward_features(x)
        return self.classifier(pooled)


def build_teacher(name: str, num_classes: int, pretrained: bool = True) -> TeacherClassifier:
    return TeacherClassifier(name=name, num_classes=num_classes, pretrained=pretrained)
