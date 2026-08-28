import argparse
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.config import SMOKE_MODEL_CONFIG
from cs336_basics.tokenizer import (
    Tokenizer,
    train_bpe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "tinystories_sample_5M.txt"
)

DEFAULT_TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer.pkl"
)

DEFAULT_TRAIN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "train_ids.npy"
)

DEFAULT_VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_ids.npy"
)

SPECIAL_TOKENS = ["<|endoftext|>"]
ENCODING_CHUNK_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-text-path",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
    )

    parser.add_argument(
        "--validation-text-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=SMOKE_MODEL_CONFIG.vocab_size,
    )

    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
    )

    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_PATH,
    )

    parser.add_argument(
        "--validation-output-path",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT_PATH,
    )

    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
    )

    return parser.parse_args()


def choose_token_dtype(
    vocab_size: int,
) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)

    return np.dtype(np.uint32)


def save_tokenizer(
    tokenizer_path: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    tokenizer_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(tokenizer_path, "wb") as file:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": SPECIAL_TOKENS,
            },
            file,
        )


def count_file_tokens(
    tokenizer: Tokenizer,
    input_path: Path,
) -> int:
    with open(
        input_path,
        encoding="utf-8",
    ) as file:
        return sum(
            1
            for _ in tokenizer.encode_iterable(file)
        )


def encode_file_to_npy(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    dtype: np.dtype,
) -> int:
    print(f"Counting tokens in {input_path}...")

    token_count = count_file_tokens(
        tokenizer,
        input_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(token_count,),
    )

    offset = 0
    chunk: list[int] = []

    print(f"Encoding {input_path}...")

    with open(
        input_path,
        encoding="utf-8",
    ) as file:
        for token_id in tokenizer.encode_iterable(file):
            chunk.append(token_id)

            if len(chunk) >= ENCODING_CHUNK_SIZE:
                chunk_array = np.asarray(
                    chunk,
                    dtype=dtype,
                )

                next_offset = (
                    offset + len(chunk_array)
                )

                output_array[
                    offset:next_offset
                ] = chunk_array

                offset = next_offset
                chunk.clear()

    if chunk:
        chunk_array = np.asarray(
            chunk,
            dtype=dtype,
        )

        next_offset = offset + len(chunk_array)

        output_array[
            offset:next_offset
        ] = chunk_array

        offset = next_offset

    output_array.flush()

    if offset != token_count:
        raise RuntimeError(
            "Encoded token count changed between passes: "
            f"expected {token_count}, wrote {offset}"
        )

    return token_count


def split_sample_text(
    source_path: Path,
    split_ratio: float,
) -> tuple[str, str]:
    text = source_path.read_text(
        encoding="utf-8"
    )

    approximate_split = int(
        len(text) * split_ratio
    )

    split_index = text.rfind(
        SPECIAL_TOKENS[0],
        0,
        approximate_split,
    )

    if split_index == -1:
        split_index = approximate_split

    return (
        text[:split_index],
        text[split_index:],
    )


def encode_text_to_npy(
    tokenizer: Tokenizer,
    text: str,
    output_path: Path,
    dtype: np.dtype,
) -> int:
    token_ids = np.asarray(
        tokenizer.encode(text),
        dtype=dtype,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(output_path, token_ids)

    return len(token_ids)


def main() -> None:
    args = parse_args()

    if args.vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError(
            "--vocab-size is too small for byte tokens "
            "and special tokens."
        )

    if not 0.0 < args.split_ratio < 1.0:
        raise ValueError(
            "--split-ratio must be between 0 and 1."
        )

    print("1. Training BPE tokenizer...")

    vocab, merges = train_bpe(
        input_path=args.train_text_path,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )

    save_tokenizer(
        tokenizer_path=args.tokenizer_path,
        vocab=vocab,
        merges=merges,
    )

    print(f"2. Tokenizer saved: {args.tokenizer_path}")
    print(f"   Vocabulary size: {len(vocab):,}")
    print(f"   Number of merges: {len(merges):,}")

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=SPECIAL_TOKENS,
    )

    dtype = choose_token_dtype(args.vocab_size)

    if args.validation_text_path is None:
        print("3. Splitting sample corpus...")

        train_text, validation_text = split_sample_text(
            source_path=args.train_text_path,
            split_ratio=args.split_ratio,
        )

        print("4. Encoding training split...")

        train_token_count = encode_text_to_npy(
            tokenizer=tokenizer,
            text=train_text,
            output_path=args.train_output_path,
            dtype=dtype,
        )

        print("5. Encoding validation split...")

        validation_token_count = encode_text_to_npy(
            tokenizer=tokenizer,
            text=validation_text,
            output_path=args.validation_output_path,
            dtype=dtype,
        )
    else:
        print("3. Encoding separate train/validation files...")

        train_token_count = encode_file_to_npy(
            tokenizer=tokenizer,
            input_path=args.train_text_path,
            output_path=args.train_output_path,
            dtype=dtype,
        )

        validation_token_count = encode_file_to_npy(
            tokenizer=tokenizer,
            input_path=args.validation_text_path,
            output_path=args.validation_output_path,
            dtype=dtype,
        )

    print("6. Data preparation complete")
    print(f"   Token dtype: {dtype}")
    print(f"   Train tokens: {train_token_count:,}")
    print(
        f"   Validation tokens: "
        f"{validation_token_count:,}"
    )
    print(f"   Train file: {args.train_output_path}")
    print(
        f"   Validation file: "
        f"{args.validation_output_path}"
    )


if __name__ == "__main__":
    main()