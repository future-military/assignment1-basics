import os
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

# Pre-tokenization regex from the GPT-2 tokenizer (info.pdf p.6).
# A single byte value is represented as a length-1 `bytes` object, since
# Python has no standalone "byte" type.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_PATTERN = re.compile(PAT)
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Uniformly-spaced initial guesses; each gets nudged forward below to
    # the nearest split_special_token occurrence so no chunk boundary can
    # land inside a pre-token or a special-token span.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # No occurrence before EOF: fall back to the file end rather
            # than leaving this boundary at its (mid-token) guess.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Returns:
        Mapping from pre-token (as a tuple of single-byte bytes objects,
        e.g. (b't', b'e', b'x', b't')) to its frequency count in `text`.

    Splits `text` on `special_tokens` first, so no pre-token spans a
    special-token boundary (info.pdf p.8), then runs TOKEN_PATTERN over
    each resulting segment via re.finditer.
    """
    counts: Counter[tuple[bytes, ...]] = Counter()

    if special_tokens:
        # Longest-first so that one special token which is a prefix of
        # another (e.g. "<|endoftext|>" vs "<|endoftext|><|endoftext|>")
        # doesn't get matched short.
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(tok) for tok in sorted_specials)
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    # Cache str -> pre-token tuple per call: natural text repeats the same
    # words constantly, so this avoids redundant UTF-8 encode + per-byte
    # tuple construction for repeats within a segment/chunk.
    cache: dict[str, tuple[bytes, ...]] = {}

    for segment in segments:
        for match in TOKEN_PATTERN.finditer(segment):
            word = match.group()
            pre_token = cache.get(word)
            if pre_token is None:
                pre_token = tuple(BYTE_TOKENS[b] for b in word.encode("utf-8"))
                cache[word] = pre_token
            counts[pre_token] += 1

    return counts


def _pretokenize_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """Worker entry point: read one [start, end) byte range and pretokenize it.

    Must be a module-level function (not a closure/lambda) so it can be
    pickled and sent to a worker process by ProcessPoolExecutor.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")
    return pretokenize(text, special_tokens)


def pretokenize_file(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenize a (potentially large) file in parallel.

    Splits the file into `num_processes` chunks along boundaries that are
    guaranteed to fall on occurrences of `special_tokens[0]` (so no chunk
    boundary can land in the middle of a pre-token or a special-token
    span), dispatches one `pretokenize` call per chunk to a worker
    process pool, and sums the resulting counts.

    Requires `special_tokens` to be non-empty, since chunk boundaries are
    anchored on the first special token (info.pdf p.8 recommends using
    the document delimiter, e.g. "<|endoftext|>", for this).
    """
    if not special_tokens:
        raise ValueError("pretokenize_file needs at least one special token to chunk on")

    num_processes = num_processes or os.cpu_count() or 1
    split_token = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    total: Counter[tuple[bytes, ...]] = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        futures = [
            pool.submit(_pretokenize_chunk, input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        for future in futures:
            total.update(future.result())

    return dict(total)
