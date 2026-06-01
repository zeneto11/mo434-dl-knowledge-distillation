from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from src.models.predictors import ConvFeaturePredictor, MLPPooledPredictor
from src.models.teachers import TeacherSpec


class MobileNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_channels = 960

    def forward_feature_map(self, x: Tensor) -> Tensor:
        return self.features(x)

    def forward_pooled(self, x: Tensor) -> Tensor:
        return torch.flatten(self.pool(self.forward_feature_map(x)), 1)


class StudentDistillationModel(nn.Module):
    def __init__(self, target: str, teacher_spec: TeacherSpec, pretrained: bool = True) -> None:
        super().__init__()
        if target not in {"pregap", "postgap"}:
            raise ValueError("target must be 'pregap' or 'postgap'")
        self.target = target
        self.encoder = MobileNetEncoder(pretrained=pretrained)
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
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = mobilenet_v3_large(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
