import heapq
import os
from collections import Counter, defaultdict

from pretokenizer import BYTE_TOKENS, pretokenize, pretokenize_file


def _load_word_freqs(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None,
) -> dict[tuple[bytes, ...], int]:
    if special_tokens:
        return pretokenize_file(str(input_path), special_tokens, num_processes)
    with open(input_path, encoding="utf-8") as f:
        return pretokenize(f.read(), special_tokens)


def _init_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    vocab: dict[int, bytes] = {i: BYTE_TOKENS[i] for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab, next_id


class _ReverseTieBreak:
    """Wraps a pair so that, on equal heap priority (equal count), heapq
    (a min-heap) pops the lexicographically GREATEST pair first - matching
    the "prefer lexicographically greater pair" tie-break rule."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "_ReverseTieBreak") -> bool:
        return self.pair > other.pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer (info.pdf p.6-9).

    Pair selection each round uses a lazy-deletion max-heap for O(log P)
    amortized cost per merge (vs. an O(P) linear scan): entries go stale
    whenever a pair's count changes, and staleness is detected on pop by
    comparing against the live count in `pair_counts`, so we just discard
    stale entries instead of trying to remove them from the heap.

    Returns:
        vocab: dict[int, bytes] mapping token ID -> token bytes. Starts as
            the 256 byte values, then the special tokens, then one entry
            per merge, in order of creation.
        merges: list[tuple[bytes, bytes]] the sequence of merges applied,
            in order of creation.
    """
    num_processes = kwargs.get("num_processes")

    vocab, next_id = _init_vocab(special_tokens)
    word_freqs = _load_word_freqs(input_path, special_tokens, num_processes)

    items = list(word_freqs.items())
    words: list[list[bytes]] = [list(pre_token) for pre_token, _ in items]
    word_counts: list[int] = [count for _, count in items]

    # Incrementally-maintained pair statistics: total frequency of each
    # adjacent symbol pair across all words, and, per pair, how many times
    # it currently occurs in each word that contains it (so a merge only
    # touches affected words, not the whole corpus - info.pdf p.8
    # "Optimizing the merging step"). This needs to be an occurrence
    # *count* per word rather than a plain set of word indices: the same
    # pair value can occur more than once in one word, and a boundary
    # update that removes one occurrence must not evict the word entirely
    # while another occurrence of that pair still survives elsewhere in it.
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_index: dict[tuple[bytes, bytes], Counter[int]] = defaultdict(Counter)

    for i, w in enumerate(words):
        c = word_counts[i]
        for pair in zip(w, w[1:]):
            pair_counts[pair] += c
            pair_index[pair][i] += 1

    heap: list[tuple[int, _ReverseTieBreak, tuple[bytes, bytes]]] = [
        (-count, _ReverseTieBreak(pair), pair) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    def push(pair: tuple[bytes, bytes]) -> None:
        count = pair_counts.get(pair)
        if count:
            heapq.heappush(heap, (-count, _ReverseTieBreak(pair), pair))

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        best_pair = None
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -neg_count:
                best_pair = pair
                break
            # else: stale entry (count changed since it was pushed) - skip
        if best_pair is None:
            break

        a, b = best_pair
        merged = a + b
        vocab[next_id] = merged
        next_id += 1
        merges.append(best_pair)

        # `best_pair`'s aggregate count is fully consumed by this merge:
        # every occurrence, across every affected word, is about to be
        # merged away, so there is nothing left to decrement it by
        # per-occurrence below (and no entry to re-delete).
        pair_counts.pop(best_pair, None)

        affected = list(pair_index.pop(best_pair, ()))
        for i in affected:
            _merge_word(i, a, b, merged, best_pair, words, word_counts, pair_counts, pair_index, push)

    return vocab, merges


def _merge_word(
    i: int,
    a: bytes,
    b: bytes,
    merged: bytes,
    best_pair: tuple[bytes, bytes],
    words: list[list[bytes]],
    word_counts: list[int],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_index: dict[tuple[bytes, bytes], Counter[int]],
    push,
) -> None:
    """
    Merge every non-overlapping occurrence of (a, b) in words[i], updating
    pair_counts/pair_index for only the pairs that actually change (the
    boundary pairs touching each merge point) rather than recomputing
    every pair in the word from scratch.

    Uses a doubly-linked list (index-based prev/next arrays) over the
    word's symbols instead of rebuilding a plain list. This matters for
    correctness, not just speed: if the word has back-to-back occurrences
    of the pair (e.g. "aaaa" merging (a, a)), the boundary symbol next to
    one merge point may itself be consumed by the very next merge in the
    same pass. A linked list always reflects the current, live adjacency,
    so the transient pair added by the first merge and removed by the
    second nets out correctly; reading ahead into the original static
    list would get this wrong.
    """
    w = words[i]
    c = word_counts[i]
    n = len(w)

    val: list[bytes] = list(w)
    nxt: list[int | None] = list(range(1, n)) + [None]
    prv: list[int | None] = [None] + list(range(0, n - 1))

    def bump(pair: tuple[bytes, bytes], sign: int) -> None:
        # best_pair's aggregate count was already fully removed up front
        # (see caller) - a per-occurrence touch of it here would either
        # double-remove it or resurrect a stale/deleted entry.
        if pair == best_pair:
            return
        pair_counts[pair] += sign * c
        occ = pair_index[pair]
        occ[i] += sign
        if occ[i] <= 0:
            del occ[i]
            if not occ:
                del pair_index[pair]
        if sign < 0 and pair_counts[pair] <= 0:
            del pair_counts[pair]
        # Refresh the heap with the pair's current live count, whether it
        # went up or down - a stale (pre-update) entry left unrefreshed
        # after a decrement would get discarded as stale on pop with no
        # live-value entry left behind to replace it, silently dropping
        # the pair from future selection.
        push(pair)

    k = 0
    while k is not None:
        k2 = nxt[k]
        if k2 is not None and val[k] == a and val[k2] == b:
            p, q = prv[k], nxt[k2]

            if p is not None:
                bump((val[p], a), -1)
            if q is not None:
                bump((b, val[q]), -1)

            val[k] = merged
            nxt[k] = q
            if q is not None:
                prv[q] = k

            if p is not None:
                bump((val[p], merged), 1)
            if q is not None:
                bump((merged, val[q]), 1)

            # val[k] is now `merged`, which can never equal `a` (it's
            # strictly longer), so position k cannot start another (a, b)
            # match - safe to continue scanning from q without re-checking k.
            k = q
        else:
            k = k2

    words[i] = [val[k] for k in _walk(nxt, n)]


def _walk(nxt: list[int | None], n: int):
    k = 0 if n else None
    while k is not None:
        yield k
        k = nxt[k]
