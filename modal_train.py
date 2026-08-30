import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "cs336-assignment1"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_MOUNT = "/mnt/cs336"

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
    .add_local_python_source("cs336_basics")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=8192,
    timeout=2*60*60,
    volumes={
        VOLUME_MOUNT: volume,
    },
)
def train_baseline() -> None:
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Modal function did not receive a CUDA GPU."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        "GiB",
    )

    volume_root = Path(VOLUME_MOUNT)

    remote_train_path = (
        volume_root
        / "data"
        / "tinystories_10k_train_ids.npy"
    )
    remote_validation_path = (
        volume_root
        / "data"
        / "tinystories_10k_validation_ids.npy"
    )

    local_data_directory = Path("/tmp/cs336_data")
    local_data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_train_path = (
        local_data_directory
        / "tinystories_10k_train_ids.npy"
    )
    local_validation_path = (
        local_data_directory
        / "tinystories_10k_validation_ids.npy"
    )

    print("Copying training data to local disk...")
    shutil.copyfile(
        remote_train_path,
        local_train_path,
    )

    print("Copying validation data to local disk...")
    shutil.copyfile(
        remote_validation_path,
        local_validation_path,
    )

    checkpoint_path = (
        volume_root
        / "checkpoints"
        / "l4_baseline_5000.pt"
    )
    log_path = (
        volume_root
        / "logs"
        / "l4_baseline_5000.csv"
    )

    command = [
        "python",
        "-m",
        "cs336_basics.train",
        "--model-preset",
        "tinystories",
        "--train-path",
        str(local_train_path),
        "--validation-path",
        str(local_validation_path),
        "--max-steps",
        "5000",
        "--batch-size",
        "32",
        "--gradient-accumulation-steps",
        "1",
        "--max-learning-rate",
        "0.0003",
        "--min-learning-rate",
        "0.00003",
        "--warmup-steps",
        "500",
        "--cosine-cycle-steps",
        "5000",
        "--eval-interval",
        "100",
        "--checkpoint-interval",
        "500",
        "--checkpoint-path",
        str(checkpoint_path),
        "--log-path",
        str(log_path),
        "--resume",
    ]

    print("Starting L4 baseline training...")

    subprocess.run(
        command,
        check=True,
    )

    volume.commit()

    print("Baseline training completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CSV log: {log_path}")


@app.local_entrypoint()
def main() -> None:
    train_baseline.remote()