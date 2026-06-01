from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def distillation_loss(
    student_representation: Tensor,
    teacher_representation: Tensor,
    logits: Tensor | None,
    labels: Tensor,
    alpha: float,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mse = F.mse_loss(student_representation, teacher_representation)
    ce = F.cross_entropy(logits, labels) if logits is not None and beta > 0 else mse.new_tensor(0.0)
    return alpha * mse + beta * ce, mse.detach(), ce.detach()
