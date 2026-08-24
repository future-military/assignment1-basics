import math
import torch
from torch import Tensor

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it <= cosine_cycle_iters:
        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_learning_rate
            + cosine_value
            * (max_learning_rate - min_learning_rate)
        )

    return min_learning_rate

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)

                state["step"] += 1

                step = state["step"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                first_moment.mul_(beta1).add_(
                    gradient,
                    alpha=1 - beta1,
                )

                second_moment.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                adjusted_lr = (
                    lr
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                parameter.addcdiv_(
                    first_moment,
                    second_moment.sqrt().add_(eps),
                    value=-adjusted_lr,
                )

                parameter.add_(
                    parameter,
                    alpha=-lr * weight_decay,
                )

        return loss
