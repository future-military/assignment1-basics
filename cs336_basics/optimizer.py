"""AdamW optimizer, cosine LR schedule with warmup, and gradient clipping
(info.pdf sec. 4.3-4.5)."""

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW (info.pdf sec. 4.3, Algorithm 1): Adam with decoupled weight
    decay applied directly to the parameters each step."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data -= lr * weight_decay * p.data
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def get_lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing schedule with linear warmup (info.pdf sec. 4.4)."""
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return lr_min + 0.5 * (1 + math.cos(progress * math.pi)) * (lr_max - lr_min)
    return lr_min


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clips the combined gradient of `params` to L2-norm `max_l2_norm`
    in place (info.pdf sec. 4.5)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
