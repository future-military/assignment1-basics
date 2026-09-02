"""Training loop (info.pdf sec. 5.3): ties together the data loader, the
TransformerLM, AdamW, the cosine LR schedule, gradient clipping, and
checkpointing. Logs step, wall-clock time, and train/val loss to console
and a CSV file (info.pdf sec. 7.1)."""

import argparse
import csv
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.train_utils import cross_entropy, get_batch, load_checkpoint, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train-data", default="data/train.npy")
    p.add_argument("--valid-data", default="data/valid.npy")
    p.add_argument("--vocab-size", type=int, default=10000)
    # Model
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=704)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # Optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--lr-max", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Housekeeping
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint-path", default="checkpoints/model.pt")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--log-path", default="train_log.csv")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, num_batches):
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}")

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.valid_data, mmap_mode="r")
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"resumed from step {start_step}")

    import os

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
    log_file = open(args.log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow(["step", "wall_time_s", "train_loss", "val_loss", "lr"])

    start_time = time.time()
    for step in range(start_step, args.total_steps):
        lr = get_lr_cosine_schedule(
            step, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        val_loss = None
        if step % args.eval_every == 0 or step == args.total_steps - 1:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length, device, args.eval_batches
            )
            elapsed = time.time() - start_time
            print(
                f"step {step:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                f"| lr {lr:.2e} | {elapsed:.1f}s"
            )
            log_writer.writerow([step, f"{elapsed:.2f}", f"{loss.item():.4f}", f"{val_loss:.4f}", f"{lr:.6e}"])
            log_file.flush()

        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)

    save_checkpoint(model, optimizer, args.total_steps, args.checkpoint_path)
    log_file.close()
    print(f"done. final checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
