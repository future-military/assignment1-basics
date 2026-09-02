"""Cross-entropy loss, batch sampling, and checkpointing (info.pdf sec.
4.1, 5.1, 5.2)."""

import os
import typing

import numpy as np
import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy loss over all batch-like dimensions (info.pdf
    sec. 4.1): l_i = -log softmax(o_i)[x_{i+1}], computed via the
    logsumexp identity so exp/log cancel and large logits stay stable."""
    logits = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples `batch_size` random length-`context_length` windows from the
    token stream `x`, paired with the next-token targets (info.pdf sec.
    5.1)."""
    max_start = len(x) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + 1 + context_length] for s in starts])
    inputs_t = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets_t = torch.from_numpy(targets.astype(np.int64)).to(device)
    return inputs_t, targets_t


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO,
) -> None:
    """Dumps model/optimizer state and the iteration number (info.pdf sec.
    5.2), enough to resume training exactly."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Inverse of `save_checkpoint`: restores model/optimizer state in
    place and returns the saved iteration number."""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
