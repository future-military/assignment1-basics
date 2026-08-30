import pickle
from pathlib import Path
import torch
import argparse

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import (load_checkpoint, load_checkpoint_config)
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.config import (ModelConfig, SMOKE_MODEL_CONFIG)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "smoke_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=TOKENIZER_PATH,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    return parser.parse_args()

def load_tokenizer(tokenizer_path: Path,) -> Tokenizer:
    with open(tokenizer_path, "rb") as file:
        tokenizer_data = pickle.load(file)

    return Tokenizer(
        vocab=tokenizer_data["vocab"],
        merges=tokenizer_data["merges"],
        special_tokens=tokenizer_data["special_tokens"],
    )

def load_model(checkpoint_path: Path
    ) -> tuple[TransformerLM, ModelConfig]:
    saved_config = load_checkpoint_config(checkpoint_path)

    if saved_config is None:
        model_config = SMOKE_MODEL_CONFIG
        print("Checkpoint has no model config; "
              "using legacy smoke config")
    else:
        model_config = ModelConfig(**saved_config)

    model = TransformerLM(
        **model_config.to_kwargs(),
        device=DEVICE,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    completed_steps = load_checkpoint(
        src=checkpoint_path,
        model=model,
        optimizer=optimizer,
    )

    model.eval()
    print(f"Loaded checkpoint at step {completed_steps}")

    return model, model_config

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,

) -> str:

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token.")

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )
    end_token_bytes = "<|endoftext|>".encode("utf-8")

    end_token_id = next(
        (
            token_id
            for token_id, token_bytes in tokenizer.vocab.items()
            if token_bytes == end_token_bytes
        ),
        None,
    )

    for _ in range(max_new_tokens):
        context = generated[:, -context_length:]

        logits = model(context)

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        if top_k is not None:
            effective_top_k = min(
                top_k,
                next_token_logits.shape[-1],
            )

            cutoff = torch.topk(
                next_token_logits,
                k=effective_top_k,
                dim=-1,
            ).values[:, -1:]

            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < cutoff,
                float("-inf"),
            )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
        )
        if (
            end_token_id is not None
            and next_token_id.item() == end_token_id
        ):
            break
        generated = torch.cat(
            [generated, next_token_id],
            dim=-1,
        )


    generated_ids = generated[0].tolist()
    return tokenizer.decode(generated_ids)

def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be greater than 0."
        )

    if args.top_k is not None and args.top_k <= 0:
        raise ValueError(
            "--top-k must be greater than 0."
        )

    if args.max_new_tokens < 0:
        raise ValueError(
            "--max-new-tokens must be non-negative."
    )

    torch.manual_seed(42)

    tokenizer = load_tokenizer(args.tokenizer_path)
    model, model_config = load_model(args.checkpoint_path)
    if len(tokenizer.vocab) != model_config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match "
            "the model vocabulary size: "
            f"{len(tokenizer.vocab)} != "
            f"{model_config.vocab_size}"
        )

    print(
        f"Tokenizer vocabulary size: "
        f"{len(tokenizer.vocab)}"
    )


    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=model_config.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()