import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    model_config: dict[str, object] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    if model_config is not None:
        checkpoint["model_config"] = model_config

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    
    model_device = next(model.parameters()).device
    checkpoint = torch.load(
        src,
        map_location=model_device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]

def load_checkpoint_config(
        src,
) -> dict[str, object] | None:
    checkpoint = torch.load(
        src,
        map_location="cpu",
    )
    return checkpoint.get("model_config")