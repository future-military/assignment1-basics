import os
import random
from typing import BinaryIO, IO

import numpy as np
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
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )

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

    torch_rng_state = checkpoint.get(
        "torch_rng_state"
    )
    if torch_rng_state is not None:
        torch.set_rng_state(
            torch_rng_state.cpu()
        )

    numpy_rng_state = checkpoint.get(
        "numpy_rng_state"
    )
    if numpy_rng_state is not None:
        np.random.set_state(
            numpy_rng_state
        )

    python_rng_state = checkpoint.get(
        "python_rng_state"
    )
    if python_rng_state is not None:
        random.setstate(
            python_rng_state
        )

    cuda_rng_state_all = checkpoint.get(
        "cuda_rng_state_all"
    )
    if (
        torch.cuda.is_available()
        and cuda_rng_state_all is not None
    ):
        torch.cuda.set_rng_state_all(
            [
                state.cpu()
                for state in cuda_rng_state_all
            ]
        )

    return checkpoint["iteration"]


def load_checkpoint_config(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, object] | None:
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
    )

    return checkpoint.get("model_config")