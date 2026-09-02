"""Tokenizes data/train.txt and data/valid.txt with the already-trained BPE
tokenizer (scratch_vocab.json/scratch_merges.txt), saving token IDs as
uint16 numpy arrays (info.pdf sec. 2.7(d)) for memory-mapped loading during
training (info.pdf sec. 5.1)."""

import argparse
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]


def tokenize_file(tok: Tokenizer, in_path: str, out_path: str) -> None:
    ids: list[int] = []
    with open(in_path, encoding="utf-8") as f:
        for chunk_id in tok.encode_iterable(f):
            ids.append(chunk_id)
    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < len(tok.vocab), "token id exceeds uint16-safe vocab size"
    np.save(out_path, arr)
    print(f"{in_path} -> {out_path}: {len(arr):,} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="scratch_vocab.json")
    parser.add_argument("--merges", default="scratch_merges.txt")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    tok = Tokenizer.from_files(args.vocab, args.merges, SPECIAL_TOKENS)


    for split in ("TinyStoriesV2-GPT4-train", "TinyStoriesV2-GPT4-valid"):
        in_path = os.path.join(args.data_dir, f"{split}.txt")
        out_path = os.path.join(args.data_dir, f"{split[-5:]}.npy")
        tokenize_file(tok, in_path, out_path)


if __name__ == "__main__":
    main()
