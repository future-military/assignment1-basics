from pathlib import Path
import pickle

import numpy as np

from .tokenizer import Tokenizer, train_bpe


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "tinystories_sample_5M.txt"
)

DATA_DIRECTORY = PROJECT_ROOT / "data"
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts"

SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 1000


def main() -> None:
    DATA_DIRECTORY.mkdir(exist_ok=True)
    ARTIFACT_DIRECTORY.mkdir(exist_ok=True)

    print("1. Training BPE tokenizer...")

    vocab, merges = train_bpe(
        input_path=SOURCE_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer_path = ARTIFACT_DIRECTORY / "tokenizer.pkl"

    with open(tokenizer_path, "wb") as file:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": SPECIAL_TOKENS,
            },
            file,
        )

    print(f"2. Tokenizer saved: {tokenizer_path}")
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Number of merges: {len(merges)}")

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=SPECIAL_TOKENS,
    )

    print("3. Reading and splitting text...")

    text = SOURCE_PATH.read_text(encoding="utf-8")
    approximate_split = int(len(text) * 0.9)


    split_index = text.rfind(
        SPECIAL_TOKENS[0],
        0,
        approximate_split,
    )

    if split_index == -1:
        split_index = approximate_split

    train_text = text[:split_index]
    validation_text = text[split_index:]

    print("4. Encoding training text...")
    train_ids = np.asarray(
        tokenizer.encode(train_text),
        dtype=np.uint16,
    )

    print("5. Encoding validation text...")
    validation_ids = np.asarray(
        tokenizer.encode(validation_text),
        dtype=np.uint16,
    )

    train_path = DATA_DIRECTORY / "train_ids.npy"
    validation_path = DATA_DIRECTORY / "validation_ids.npy"

    np.save(train_path, train_ids)
    np.save(validation_path, validation_ids)

    print("6. Data preparation complete")
    print(f"   Train tokens: {len(train_ids):,}")
    print(f"   Validation tokens: {len(validation_ids):,}")
    print(f"   Train file: {train_path}")
    print(f"   Validation file: {validation_path}")


if __name__ == "__main__":
    main()