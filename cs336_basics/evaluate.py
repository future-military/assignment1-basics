import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from cs336_basics.config import ModelConfig
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Transformer checkpoints on fixed "
            "validation batches."
        )
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        required=True,
        help=(
            "Checkpoint to evaluate. Repeat this option "
            "to evaluate multiple checkpoints."
        ),
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def validate_args(
    checkpoint_paths: list[Path],
    validation_path: Path,
    batch_size: int,
    eval_batches: int,
) -> None:
    if batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0."
        )

    if eval_batches <= 0:
        raise ValueError(
            "--eval-batches must be greater than 0."
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}"
        )

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_data: np.ndarray,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_config = checkpoint.get("model_config")
    if saved_config is None:
        raise ValueError(
            "Checkpoint has no model configuration: "
            f"{checkpoint_path}"
        )

    model_config = ModelConfig(
        **saved_config
    )

    model = TransformerLM(
        **model_config.to_kwargs(),
        device=DEVICE,
        dtype=torch.float32,
    )

    model.load_state_dict(
        checkpoint["model"]
    )
    completed_steps = int(
        checkpoint["iteration"]
    )

    del checkpoint

    model.eval()

    # Reset after model construction and checkpoint loading.
    # Every checkpoint therefore receives identical batches.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    losses = []

    with torch.inference_mode():
        for _ in range(eval_batches):
            inputs, targets = get_batch(
                dataset=validation_data,
                batch_size=batch_size,
                context_length=(
                    model_config.context_length
                ),
                device=DEVICE,
            )

            logits = model(inputs)

            flat_logits = logits.reshape(
                -1,
                logits.shape[-1],
            )
            flat_targets = targets.reshape(-1)

            loss = cross_entropy(
                flat_logits,
                flat_targets,
            )
            losses.append(loss.item())

    mean_loss = float(np.mean(losses))

    if len(losses) > 1:
        standard_error = float(
            np.std(
                losses,
                ddof=1,
            )
            / math.sqrt(len(losses))
        )
    else:
        standard_error = 0.0

    confidence_radius = (
        1.96 * standard_error
    )
    perplexity = math.exp(mean_loss)

    tokens_evaluated = (
        batch_size
        * model_config.context_length
        * eval_batches
    )

    result = {
        "checkpoint": checkpoint_path.name,
        "step": completed_steps,
        "validation_loss": mean_loss,
        "perplexity": perplexity,
        "standard_error": standard_error,
        "ci95_lower": (
            mean_loss - confidence_radius
        ),
        "ci95_upper": (
            mean_loss + confidence_radius
        ),
        "tokens_evaluated": tokens_evaluated,
    }

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    args = parse_args()

    validate_args(
        checkpoint_paths=args.checkpoint_path,
        validation_path=args.validation_path,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
    )

    validation_data = np.load(
        args.validation_path,
        mmap_mode="r",
    )

    print(f"Evaluation device: {DEVICE}")
    print(f"Evaluation seed: {args.seed}")
    print(
        "Evaluation batches:",
        args.eval_batches,
    )

    results = []

    for checkpoint_path in args.checkpoint_path:
        print(
            f"\nEvaluating {checkpoint_path}..."
        )

        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            validation_data=validation_data,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )
        results.append(result)

        print(
            "Step:",
            result["step"],
        )
        print(
            "Validation loss:",
            f"{result['validation_loss']:.6f}",
        )
        print(
            "Perplexity:",
            f"{result['perplexity']:.6f}",
        )
        print(
            "95% loss interval:",
            f"[{result['ci95_lower']:.6f}, "
            f"{result['ci95_upper']:.6f}]",
        )

    if args.output_path is not None:
        args.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "checkpoint",
            "step",
            "validation_loss",
            "perplexity",
            "standard_error",
            "ci95_lower",
            "ci95_upper",
            "tokens_evaluated",
        ]

        with open(
            args.output_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(results)

        print(
            f"\nResults saved to "
            f"{args.output_path}"
        )


if __name__ == "__main__":
    main()