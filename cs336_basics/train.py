import  numpy as np
import torch
from pathlib import Path
import argparse
import csv
import time

from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.nn_utils import (
    softmax,
    cross_entropy,
    gradient_clipping,
)
from cs336_basics.model import (
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    scaled_dot_product_attention,
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
)
from cs336_basics.data import get_batch
from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule
)
from cs336_basics.serialization import (
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.config import SMOKE_MODEL_CONFIG

#configurations

MODEL_CONFIG = SMOKE_MODEL_CONFIG

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "train_ids.npy"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation_ids.npy"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "smoke_model.pt"
LOG_PATH = (PROJECT_ROOT / "logs" / "smoke_training.csv")

MAX_STEPS = 35
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 5
COSINE_CYCLE_STEPS = 30
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 10
EVAL_BATCHES = 5

LOG_FIELDNAMES = [
    "step",
    "train_loss",
    "validation_loss",
    "learning_rate",
    "step_seconds",
    "elapsed_seconds",
    "tokens_seen",
    "tokens_per_second",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CS336 Transformer language model."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--cosine-cycle-steps",
        type=int,
        default=COSINE_CYCLE_STEPS,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--max-learning-rate",
        type=float,
        default=MAX_LR,
    )

    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=MIN_LR,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_STEPS,
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_PATH,
    )

    return parser.parse_args()

def write_csv_log(
        log_path: Path,
        row: dict[str, object],
        overwrite: bool = False,
) -> None:
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        log_path.exists()
        and log_path.stat().st_size > 0
    )

    mode = "w" if overwrite else "a"

    with open(
        log_path,
        mode,
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDNAMES,
        )

        if overwrite or not file_exists:
            writer.writeheader()

        writer.writerow(row)

def validate(model: TransformerLM, validation_data: np.ndarray, batch_size: int) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            inputs, targets = get_batch(
                dataset=validation_data,
                batch_size=BATCH_SIZE,
                context_length=MODEL_CONFIG.context_length,
                device=DEVICE,
            )
            logits = model(inputs)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            loss = cross_entropy(flat_logits, flat_targets)
            losses.append(loss.item())


    average_loss = sum(losses) / len(losses)
    model.train()

    return average_loss

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_data = np.load(TRAIN_PATH, mmap_mode="r")
    validation_data = np.load(VALIDATION_PATH, mmap_mode="r")

    model = TransformerLM(
        **MODEL_CONFIG.to_kwargs(),
        device=DEVICE,
        dtype=torch.float32,

    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = AdamW(model.parameters(), lr=args.max_learning_rate, weight_decay=WEIGHT_DECAY, betas = (0.9, 0.95))

    start_step = 0

    if args.resume and args.checkpoint_path.exists():
        start_step = load_checkpoint(
            src=args.checkpoint_path,
            model=model,
            optimizer=optimizer,
        )
        print(f"Resuming training from step {start_step}")
    else:
        print("Starting training from scratch")

    model.train()

    training_start_time = time.perf_counter()
    tokens_per_step = (
        args.batch_size
        * MODEL_CONFIG.context_length
    )

    completed_steps = start_step

    for step in range(start_step, args.max_steps):
        step_start_time = time.perf_counter()
        learning_rate = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.cosine_cycle_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=MODEL_CONFIG.context_length,
            device=DEVICE,
        )
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = cross_entropy(flat_logits, flat_targets)
        loss.backward()
        gradient_clipping(parameters = model.parameters(), max_l2_norm = MAX_GRAD_NORM)
        optimizer.step()

        step_seconds = (
            time.perf_counter()
            - step_start_time
        )
        tokens_per_second = (
            tokens_per_step / step_seconds
        )

        completed_steps = step + 1

        if completed_steps % args.checkpoint_interval == 0:
            args.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                model_config=MODEL_CONFIG.to_kwargs(),
                out=args.checkpoint_path,
            )
            print(f"Checkpoint saved at step {completed_steps}")

        validation_loss = None
        if completed_steps % args.eval_interval == 0:
            validation_loss = validate(model, validation_data, args.batch_size)
            print(f"Step {step + 1}: Training Loss = {loss.item():.4f}, Validation Loss = {validation_loss:.4f}")
        elapsed_seconds = (
            time.perf_counter()
            - training_start_time
        )

        tokens_seen = (
            (step + 1)
            * tokens_per_step
        )

        write_csv_log(
            log_path=args.log_path,
            row={
                "step": step + 1,
                "train_loss": loss.item(),
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "step_seconds": step_seconds,
                "elapsed_seconds": elapsed_seconds,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_per_second,
            },
            overwrite=(
                start_step == 0
                and step == 0
            ),
        )

        print(step + 1, learning_rate, loss.item())
    checkpoint_dir = args.checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=completed_steps,
        model_config=MODEL_CONFIG.to_kwargs(),
        out=args.checkpoint_path,
    )

if __name__ == "__main__":
    main()


