import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "cs336-assignment1-evaluation"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_MOUNT = "/mnt/cs336"

DEFAULT_CHECKPOINTS = (
    "l4_baseline_5000.pt,"
    "lr6e4_5000.pt,"
    "lr6e4_5000.best.pt,"
    "lr1e3_5000.pt,"
    "lr1e3_5000.best.pt"
)

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=False,
)

image = (
    modal.Image.debian_slim(
        python_version="3.13",
    )
    .uv_sync()
    .add_local_python_source(
        "cs336_basics"
    )
)


def parse_checkpoint_names(
    checkpoint_names: str,
) -> list[str]:
    names = [
        name.strip()
        for name in checkpoint_names.split(",")
        if name.strip()
    ]

    if not names:
        raise ValueError(
            "At least one checkpoint is required."
        )

    for name in names:
        if (
            Path(name).name != name
            or not name.endswith(".pt")
        ):
            raise ValueError(
                "Checkpoint names must be plain "
                f".pt filenames: {name}"
            )

    return names


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={
        VOLUME_MOUNT: volume,
    },
)
def evaluate_checkpoints(
    checkpoint_names: str,
    output_name: str,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> None:
    import torch

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    if eval_batches <= 0:
        raise ValueError(
            "eval_batches must be greater than 0."
        )

    if (
        Path(output_name).name != output_name
        or not output_name.endswith(".csv")
    ):
        raise ValueError(
            "output_name must be a plain CSV filename."
        )

    names = parse_checkpoint_names(
        checkpoint_names
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Modal function did not receive a CUDA GPU."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    volume_root = Path(VOLUME_MOUNT)

    remote_validation_path = (
        volume_root
        / "data"
        / "tinystories_10k_validation_ids.npy"
    )

    local_data_directory = Path(
        "/tmp/cs336_evaluation"
    )
    local_data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_validation_path = (
        local_data_directory
        / "tinystories_10k_validation_ids.npy"
    )

    print(
        "Copying validation data to local disk..."
    )
    shutil.copyfile(
        remote_validation_path,
        local_validation_path,
    )

    checkpoint_paths = []

    for name in names:
        checkpoint_path = (
            volume_root
            / "checkpoints"
            / name
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint_paths.append(
            checkpoint_path
        )

    output_path = (
        volume_root
        / "logs"
        / output_name
    )

    command = [
        "python",
        "-u",
        "-m",
        "cs336_basics.evaluate",
    ]

    for checkpoint_path in checkpoint_paths:
        command.extend(
            [
                "--checkpoint-path",
                str(checkpoint_path),
            ]
        )

    command.extend(
        [
            "--validation-path",
            str(local_validation_path),
            "--batch-size",
            str(batch_size),
            "--eval-batches",
            str(eval_batches),
            "--seed",
            str(seed),
            "--output-path",
            str(output_path),
        ]
    )

    print("Evaluating checkpoints:")
    for checkpoint_path in checkpoint_paths:
        print(f"  {checkpoint_path.name}")

    subprocess.run(
        command,
        check=True,
    )

    volume.commit()

    print("Evaluation completed.")
    print(f"Results: {output_path}")


@app.local_entrypoint()
def main(
    checkpoint_names: str = DEFAULT_CHECKPOINTS,
    output_name: str = "final_evaluation.csv",
    batch_size: int = 32,
    eval_batches: int = 100,
    seed: int = 2026,
) -> None:
    evaluate_checkpoints.remote(
        checkpoint_names=checkpoint_names,
        output_name=output_name,
        batch_size=batch_size,
        eval_batches=eval_batches,
        seed=seed,
    )