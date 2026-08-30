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


def validate_experiment_config(
    run_name: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    cosine_cycle_steps: int,
    eval_interval: int,
    eval_batches: int,
    checkpoint_interval: int,
) -> None:
    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )

    if (
        not run_name
        or any(
            character not in allowed_characters
            for character in run_name
        )
    ):
        raise ValueError(
            "run_name may contain only letters, numbers, "
            "hyphens, and underscores."
        )

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be "
            "greater than 0."
        )

    if max_learning_rate <= 0:
        raise ValueError(
            "max_learning_rate must be greater than 0."
        )

    if min_learning_rate <= 0:
        raise ValueError(
            "min_learning_rate must be greater than 0."
        )

    if min_learning_rate > max_learning_rate:
        raise ValueError(
            "min_learning_rate must not exceed "
            "max_learning_rate."
        )

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be non-negative."
        )

    if cosine_cycle_steps <= warmup_steps:
        raise ValueError(
            "cosine_cycle_steps must be greater than "
            "warmup_steps."
        )

    if eval_interval <= 0:
        raise ValueError(
            "eval_interval must be greater than 0."
        )

    if eval_batches <= 0:
        raise ValueError(
            "eval_batches must be greater than 0."
        )

    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be greater than 0."
        )


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={
        VOLUME_MOUNT: volume,
    },
)
def train_experiment(
    run_name: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    cosine_cycle_steps: int,
    eval_interval: int,
    eval_batches: int,
    checkpoint_interval: int,
    resume: bool,
) -> None:
    import torch

    validate_experiment_config(
        run_name=run_name,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        cosine_cycle_steps=cosine_cycle_steps,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        checkpoint_interval=checkpoint_interval,
    )

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
        / f"{run_name}.pt"
    )
    log_path = (
        volume_root
        / "logs"
        / f"{run_name}.csv"
    )

    checkpoint_exists = checkpoint_path.exists()
    effective_resume = (
        resume
        or checkpoint_exists
    )

    if resume and not checkpoint_exists:
        raise FileNotFoundError(
            "Cannot resume because checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    if checkpoint_exists and not resume:
        print(
            "Existing checkpoint detected; "
            "automatically resuming."
        )

    effective_batch_size = (
        batch_size
        * gradient_accumulation_steps
    )
    tokens_per_step = effective_batch_size * 256
    total_tokens = max_steps * tokens_per_step

    print(f"Run name: {run_name}")
    print(f"Maximum steps: {max_steps}")
    print(f"Micro batch size: {batch_size}")
    print(
        "Gradient accumulation steps:",
        gradient_accumulation_steps,
    )
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Planned tokens: {total_tokens:,}")
    print(f"Maximum learning rate: {max_learning_rate}")
    print(f"Minimum learning rate: {min_learning_rate}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Evaluation batches: {eval_batches}")
    print(f"Requested resume: {resume}")
    print(f"Effective resume: {effective_resume}")

    command = [
        "python",
        "-u",
        "-m",
        "cs336_basics.train",
        "--model-preset",
        "tinystories",
        "--train-path",
        str(local_train_path),
        "--validation-path",
        str(local_validation_path),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--max-learning-rate",
        str(max_learning_rate),
        "--min-learning-rate",
        str(min_learning_rate),
        "--warmup-steps",
        str(warmup_steps),
        "--cosine-cycle-steps",
        str(cosine_cycle_steps),
        "--eval-interval",
        str(eval_interval),
        "--eval-batches",
        str(eval_batches),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--checkpoint-path",
        str(checkpoint_path),
        "--log-path",
        str(log_path),
    ]

    if effective_resume:
        command.append("--resume")

    print("Starting Modal training experiment...")

    subprocess.run(
        command,
        check=True,
    )

    volume.commit()

    print("Training experiment completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Best checkpoint:",
        checkpoint_path.with_name(
            f"{checkpoint_path.stem}.best"
            f"{checkpoint_path.suffix}"
        ),
    )
    print(f"CSV log: {log_path}")


@app.local_entrypoint()
def main(
    run_name: str = "experiment",
    max_steps: int = 5000,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    max_learning_rate: float = 0.0003,
    min_learning_rate: float = 0.00003,
    warmup_steps: int = 500,
    cosine_cycle_steps: int = 5000,
    eval_interval: int = 500,
    eval_batches: int = 20,
    checkpoint_interval: int = 500,
    resume: bool = False,
) -> None:
    validate_experiment_config(
        run_name=run_name,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        cosine_cycle_steps=cosine_cycle_steps,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        checkpoint_interval=checkpoint_interval,
    )

    train_experiment.remote(
        run_name=run_name,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        cosine_cycle_steps=cosine_cycle_steps,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        checkpoint_interval=checkpoint_interval,
        resume=resume,
    )