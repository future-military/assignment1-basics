import os
import regex as re
from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_PATTERN = re.compile(PAT)
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = dict(enumerate(BYTE_TOKENS))
    merges = []
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    with open(input_path, "r", encoding = "utf-8") as file:
        text = file.read()

    pre_token_counts = {}



    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        delimiter = "|".join(escaped_tokens)
        segments = re.split(delimiter, text)

    else:
        segments = [text]

    for segment in segments:
        for match in TOKEN_PATTERN.finditer(segment):
            pre_token = match.group(0)
            encoded = pre_token.encode("utf-8")
            byte_tuple = tuple(BYTE_TOKENS[byte_value] for byte_value in encoded)
            pre_token_counts[byte_tuple] = pre_token_counts.get(byte_tuple, 0) + 1

    while len(vocab) < vocab_size:
        pair_counts = {}

        for token_tuple, count in pre_token_counts.items():
            if len(token_tuple) ==1:
                continue
            for pair in zip(token_tuple, token_tuple[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if not pair_counts:
            break

        best_pair = max(pair_counts, key = lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]
        new_pre_token_counts = {}

        for token_tuple, count in pre_token_counts.items():
            
            new_tokens = []
            index = 0

            while index < len(token_tuple):
                if index +1 <len(token_tuple) and (token_tuple[index], token_tuple[index+1]) ==  best_pair:
                    new_tokens.append(merged_token)
                    index += 2
                else:
                    new_tokens.append(token_tuple[index])
                    index +=1
            new_token_tuple = tuple(new_tokens)
            new_pre_token_counts[new_token_tuple] = new_pre_token_counts.get(new_token_tuple, 0) + count

        merges.append(best_pair)
        vocab[len(vocab)] = merged_token
        pre_token_counts = new_pre_token_counts
    

    return vocab, merges

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = special_bytes
                self.bytes_to_id[special_bytes] = new_id

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
    def decode(self, ids: list[int]) -> str:
        byte_pieces = (
            self.vocab[token_id] for token_id in ids
        )

        combined_bytes = b"".join(byte_pieces)
        return combined_bytes.decode(
            "utf-8",
            errors = "replace"
        )
    def _encode_pre_token(self, pre_token: str) -> list[int]:
        encoded = pre_token.encode("utf-8")

        tokens = [
            BYTE_TOKENS[byte_value]
            for byte_value in encoded
        ]

        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            for index in range(len(tokens) - 1):
                pair = (tokens[index], tokens[index + 1])

                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]

                    if rank < best_rank:
                        best_pair = pair
                        best_rank = rank

            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            new_tokens = []
            index = 0

            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    new_tokens.append(merged_token)
                    index += 2
                else:
                    new_tokens.append(tokens[index])
                    index += 1

            tokens = new_tokens

        return [
                self.bytes_to_id[token]
                for token in tokens
        ]

    def encode(self, text: str) -> list[int]:
        token_ids = []
        special_token_set = set(self.special_tokens)

        if self.special_tokens:
            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            escaped_tokens = [
                re.escape(token)
                for token in sorted_special_tokens
            ]

            delimiter = "(" + "|".join(escaped_tokens) + ")"
            segments = re.split(delimiter, text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            if segment in special_token_set:
                special_bytes = segment.encode("utf-8")
                token_ids.append(
                    self.bytes_to_id[special_bytes]
                )
                continue

            for match in TOKEN_PATTERN.finditer(segment):
                pre_token = match.group(0)

                token_ids.extend(
                    self._encode_pre_token(pre_token)
                )

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
