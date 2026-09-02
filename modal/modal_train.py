"""Run cs336_basics training on a Modal GPU instead of local CPU/MPS.

Setup (one-time, on your machine):
    pip install modal
    modal setup                     # opens a browser to link your Modal account

Usage:
    # 1. Upload the *real* full-size tokenized corpus to a Modal Volume.
    modal run modal_train.py::upload_data

    # 2. Launch GPU training (checkpoints/logs land in the same Volume).
    modal run modal_train.py --total-steps 20000 --warmup-steps 1000

    # 3. Pull results back down.
    modal volume get cs336-data checkpoints/model.pt ./checkpoints/model.pt
    modal volume get cs336-data train_log.csv ./train_log_gpu.csv
"""

import modal

app = modal.App("cs336-basics-train")

# Full-size tokenized corpus lives here (NOT test/data, which only has a
# ~1.27M-token debug slice).
LOCAL_DATA_DIR = "/Users/brian/Other/LLM/assignment1-basics/data"
LOCAL_CODE_DIR = "/Users/brian/Other/test"  # contains the cs336_basics package

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "einx>=0.4",
        "jaxtyping>=0.3",
        "tqdm>=4.67",
    )
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path="/root/cs336_basics",
        ignore=["data", "checkpoints", "__pycache__", ".git", ".venv-modal"],
    )
)


@app.function(volumes={VOLUME_PATH: volume})
def upload_data():
    """Copies train.npy/valid.npy already staged into the volume via `modal volume put`.
    Kept as a no-op placeholder; actual upload is done from the CLI (see README below)."""
    import os

    print("Files currently in volume:", os.listdir(VOLUME_PATH))


@app.function(gpu="A10G", image=image, volumes={VOLUME_PATH: volume}, timeout=6 * 60 * 60)
def train(
    total_steps: int = 20000,
    warmup_steps: int = 1000,
    batch_size: int = 64,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    eval_every: int = 200,
    checkpoint_every: int = 2000,
):
    import os
    import shutil
    import subprocess
    import sys

    # Copy the dataset from the network-backed Volume onto the container's
    # local disk first. train.py mmaps the .npy and does thousands of random
    # small reads per batch (get_batch samples random windows) -- against
    # the Volume's network filesystem that turns into a network round-trip
    # per read and stalls training almost completely. Local disk has none
    # of that overhead.
    local_data_dir = "/root/data"
    os.makedirs(local_data_dir, exist_ok=True)
    print("copying dataset from volume to local disk...")
    shutil.copy(f"{VOLUME_PATH}/train.npy", f"{local_data_dir}/train.npy")
    shutil.copy(f"{VOLUME_PATH}/valid.npy", f"{local_data_dir}/valid.npy")
    print("copy done.")

    local_checkpoint_dir = "/root/checkpoints"
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    local_checkpoint_path = f"{local_checkpoint_dir}/model.pt"
    local_log_path = "/root/train_log.csv"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "cs336_basics.train",
        "--train-data", f"{local_data_dir}/train.npy",
        "--valid-data", f"{local_data_dir}/valid.npy",
        "--vocab-size", "10000",
        "--context-length", str(context_length),
        "--d-model", str(d_model),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--d-ff", str(d_ff),
        "--batch-size", str(batch_size),
        "--total-steps", str(total_steps),
        "--warmup-steps", str(warmup_steps),
        "--lr-max", str(lr_max),
        "--lr-min", str(lr_min),
        "--device", "cuda",
        "--eval-every", str(eval_every),
        "--checkpoint-every", str(checkpoint_every),
        "--checkpoint-path", local_checkpoint_path,
        "--log-path", local_log_path,
    ]
    print("running:", " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(cmd, check=True, cwd="/root", env=env)

    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)
    shutil.copy(local_checkpoint_path, f"{VOLUME_PATH}/checkpoints/model.pt")
    shutil.copy(local_log_path, f"{VOLUME_PATH}/train_log.csv")
    volume.commit()


@app.local_entrypoint()
def spawn_train(
    total_steps: int = 20000,
    warmup_steps: int = 1000,
    batch_size: int = 64,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    eval_every: int = 200,
    checkpoint_every: int = 2000,
):
    """Fire-and-forget: spawns the remote call and exits immediately.
    The run is NOT tied to this (or any) local process afterwards -- use
    the printed call ID to check on or await it later, from any machine.
    """
    call = train.spawn(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        lr_max=lr_max,
        lr_min=lr_min,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
    )
    print(f"CALL_ID={call.object_id}")


@app.local_entrypoint()
def main(
    total_steps: int = 20000,
    warmup_steps: int = 1000,
    batch_size: int = 64,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    eval_every: int = 200,
    checkpoint_every: int = 2000,
):
    train.remote(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        lr_max=lr_max,
        lr_min=lr_min,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
    )
