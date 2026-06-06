from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


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
