from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def align_spatial_size(student_representation: Tensor, teacher_representation: Tensor) -> Tensor:
    if (
        student_representation.ndim == 4
        and teacher_representation.ndim == 4
        and student_representation.shape[-2:] != teacher_representation.shape[-2:]
    ):
        return F.adaptive_avg_pool2d(student_representation, teacher_representation.shape[-2:])
    return student_representation


def distillation_loss(
    student_representation: Tensor,
    teacher_representation: Tensor,
    logits: Tensor | None,
    labels: Tensor,
    alpha: float,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    student_representation = align_spatial_size(student_representation, teacher_representation)
    mse = F.mse_loss(student_representation, teacher_representation)
    ce = F.cross_entropy(logits, labels) if logits is not None and beta > 0 else mse.new_tensor(0.0)
    return alpha * mse + beta * ce, mse.detach(), ce.detach()


def _pairwise_distances(embeddings: Tensor, eps: float = 1e-12) -> Tensor:
    squared_norm = embeddings.pow(2).sum(dim=1)
    products = embeddings @ embeddings.t()
    distances = (squared_norm.unsqueeze(1) + squared_norm.unsqueeze(0) - 2 * products).clamp(min=eps)
    distances = distances.sqrt()
    distances = distances.clone()
    distances[range(len(embeddings)), range(len(embeddings))] = 0.0
    return distances


class RKDLoss(nn.Module):
    """Relational Knowledge Distillation (Park et al., 2019).

    Transfers mutual relations between examples instead of individual outputs.
    The distance-wise potential matches normalized pairwise distances and the
    angle-wise potential matches the cosine of angles formed by example triplets.
    Both potentials are computed independently inside each embedding space, so
    the teacher and student embeddings may have different dimensionality and no
    projection layer is required.
    """

    def __init__(self, distance_weight: float = 25.0, angle_weight: float = 50.0) -> None:
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def distance_loss(self, student: Tensor, teacher: Tensor) -> Tensor:
        with torch.no_grad():
            teacher_distances = _pairwise_distances(teacher)
            teacher_mean = teacher_distances[teacher_distances > 0].mean()
            teacher_distances = teacher_distances / (teacher_mean + 1e-12)
        student_distances = _pairwise_distances(student)
        student_mean = student_distances[student_distances > 0].mean()
        student_distances = student_distances / (student_mean + 1e-12)
        return F.smooth_l1_loss(student_distances, teacher_distances)

    def angle_loss(self, student: Tensor, teacher: Tensor) -> Tensor:
        with torch.no_grad():
            teacher_edges = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            teacher_normed = F.normalize(teacher_edges, p=2, dim=2)
            teacher_angles = torch.bmm(teacher_normed, teacher_normed.transpose(1, 2)).view(-1)
        student_edges = student.unsqueeze(0) - student.unsqueeze(1)
        student_normed = F.normalize(student_edges, p=2, dim=2)
        student_angles = torch.bmm(student_normed, student_normed.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(student_angles, teacher_angles)

    def forward(self, student: Tensor, teacher: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distance = self.distance_loss(student, teacher)
        angle = self.angle_loss(student, teacher)
        loss = self.distance_weight * distance + self.angle_weight * angle
        return loss, distance.detach(), angle.detach()
