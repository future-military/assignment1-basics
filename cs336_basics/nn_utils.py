import torch
from torch import Tensor
from collections.abc import Iterable


def softmax(x: Tensor, dim: int) -> Tensor:
    max_value = x.max(
        dim=dim,
        keepdim=True,
    ).values

    shifted = x - max_value
    exponentials = torch.exp(shifted)

    denominator = exponentials.sum(
        dim=dim,
        keepdim=True,
    )

    return exponentials / denominator

def cross_entropy(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    max_values = inputs.max(dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_values

    log_sum_exp = torch.log(
        torch.exp(shifted_inputs).sum(dim=-1)
    )

    target_logits = shifted_inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits
    return losses.mean()

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not parameters:
        return

    total_norm_squared = sum(
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in parameters
    )
    total_norm = torch.sqrt(total_norm_squared)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        with torch.no_grad():
            for parameter in parameters:
                parameter.grad.mul_(scale)