import json
from collections.abc import Iterable, Iterator

from pretokenizer import TOKEN_PATTERN, BYTE_TOKENS
import regex as re


def _bytes_to_unicode() -> dict[int, str]:
    """
    The reversible byte<->printable-unicode-char mapping from the GPT-2
    tokenizer. Used only for serializing vocab/merges to human-inspectable
    JSON/text files (info.pdf p.9's "serialize ... for further inspection")
    - raw bytes aren't valid JSON/text, but every byte value needs a
      distinct, round-trippable representation, including the ones that
      aren't printable on their own.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _bytes_to_str(b: bytes) -> str:
    return "".join(_BYTE_ENCODER[byte] for byte in b)


def _str_to_bytes(s: str) -> bytes:
    return bytes(_BYTE_DECODER[ch] for ch in s)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Builds the bytes -> id inverse of `vocab` (for encode) and a
        (bytes, bytes) -> rank map from `merges`, so encode can find the
        earliest-learned applicable merge (info.pdf p.6-7). Any special
        token without an existing vocab entry gets one appended.
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._token_to_id: dict[bytes, int] = {tb: tid for tid, tb in self.vocab.items()}

        next_id = max(self.vocab, default=-1) + 1
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self._token_to_id:
                self.vocab[next_id] = tok_bytes
                self._token_to_id[tok_bytes] = next_id
                next_id += 1

        # Longest-first so one special token that's a prefix of another
        # (e.g. "<|endoftext|>" vs "<|endoftext|><|endoftext|>") isn't
        # shadowed by the shorter match.
        self._sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_split_pattern = (
            "(" + "|".join(re.escape(tok) for tok in self._sorted_specials) + ")"
            if self._sorted_specials
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Loads a serialized vocab/merges pair - a GPT-2-style JSON vocab of
        {unicode-mapped token string: id}, and a merges text file of
        "tok1 tok2" lines in the same mapping - and constructs a Tokenizer
        from them.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        vocab = {tid: _str_to_bytes(tok_str) for tok_str, tid in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                left, right = line.split(" ")
                merges.append((_str_to_bytes(left), _str_to_bytes(right)))

        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        """
        Maps each id to its token bytes, concatenates them, and decodes as
        UTF-8 with errors="replace" - an arbitrary/partial id sequence
        isn't guaranteed to concatenate into valid UTF-8.
        """
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        """
        Starts from single-byte tokens, then repeatedly applies whichever
        adjacent pair has the lowest merge rank (earliest-learned merge -
        info.pdf p.6-7) until no adjacent pair has one, merging every
        occurrence of that pair each round. Maps the final tokens to ids.
        """
        tokens: list[bytes] = [BYTE_TOKENS[byte] for byte in pre_token.encode("utf-8")]

        while len(tokens) > 1:
            pairs = list(zip(tokens, tokens[1:]))
            best_pair = min(
                (p for p in pairs if p in self.merge_ranks),
                key=lambda p: self.merge_ranks[p],
                default=None,
            )
            if best_pair is None:
                break

            a, b = best_pair
            merged = a + b
            new_tokens: list[bytes] = []
            j = 0
            while j < len(tokens):
                if j + 1 < len(tokens) and tokens[j] == a and tokens[j + 1] == b:
                    new_tokens.append(merged)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

        return [self._token_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """
        Splits `text` on special tokens (keeping them, via a capturing
        group, so they can be emitted as their own id); each non-special
        segment is scanned with TOKEN_PATTERN and each match encoded via
        `_encode_pre_token`.
        """
        if self._special_split_pattern:
            segments = re.split(self._special_split_pattern, text)
        else:
            segments = [text]

        special_set = set(self.special_tokens)
        ids: list[int] = []
        for segment in segments:
            if not segment:
                continue
            if segment in special_set:
                ids.append(self._token_to_id[segment.encode("utf-8")])
                continue
            for match in TOKEN_PATTERN.finditer(segment):
                ids.extend(self._encode_pre_token(match.group()))

        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """Lazily yields ids for each string in `iterable`, one at a time -
        never materializes one big list, so arbitrarily large input
        (e.g. a file handle) can be tokenized in constant memory."""
        for text in iterable:
            yield from self.encode(text)
