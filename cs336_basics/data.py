import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    starting_indices = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    inputs = np.stack(
        [
            dataset[start:start + context_length]
            for start in starting_indices
        ]
    )

    targets = np.stack(
        [
            dataset[start + 1:start + context_length + 1]
            for start in starting_indices
        ]
    )

    inputs_tensor = torch.tensor(
        inputs,
        dtype=torch.long,
        device=device,
    )
    targets_tensor = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )

    return inputs_tensor, targets_tensor