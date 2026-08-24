import  numpy as np
import torch
from pathlib import Path

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

#configurations
VOCAB_SIZE = 1000
CONTEXT_LENGTH = 64
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 256
ROPE_THETA = 10000.0
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "train_ids.npy"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation_ids.npy"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "smoke_model.pt"

MAX_STEPS = 30
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 5
COSINE_CYCLE_STEPS = 30
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 10
EVAL_BATCHES = 5

def validate(model: TransformerLM, validation_data: np.ndarray) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            inputs, targets = get_batch(
                dataset=validation_data,
                batch_size=BATCH_SIZE,
                context_length=CONTEXT_LENGTH,
                device=DEVICE,
            )
            logits = model(inputs)
            flat_logits = logits.reshape(-1, VOCAB_SIZE)
            flat_targets = targets.reshape(-1,)
            loss = cross_entropy(flat_logits, flat_targets)
            losses.append(loss.item())


    average_loss = sum(losses) / len(losses)
    model.train()
    return average_loss

def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_data = np.load(TRAIN_PATH, mmap_mode="r")
    validation_data = np.load(VALIDATION_PATH, mmap_mode="r")

    model = TransformerLM(

        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device=DEVICE,
        dtype=torch.float32,

    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, betas = (0.9, 0.95))
    model.train()

    for step in range(MAX_STEPS):
        learning_rate = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_iters=WARMUP_STEPS,
            cosine_cycle_iters=COSINE_CYCLE_STEPS,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            device=DEVICE,
        )
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        flat_logits = logits.reshape(-1, VOCAB_SIZE)
        flat_targets = targets.reshape(-1,)
        loss = cross_entropy(flat_logits, flat_targets)
        loss.backward()
        gradient_clipping(parameters = model.parameters(), max_l2_norm = MAX_GRAD_NORM)
        optimizer.step()

        if (step + 1) % EVAL_INTERVAL == 0:
            validation_loss = validate(model, validation_data)
            print(f"Step {step + 1}: Training Loss = {loss.item():.4f}, Validation Loss = {validation_loss:.4f}")

        print(step + 1, learning_rate, loss.item())
    checkpoint_dir = CHECKPOINT_PATH.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=MAX_STEPS,
        out=CHECKPOINT_PATH,
    )

if __name__ == "__main__":
    main()


