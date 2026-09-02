"""Text generation: temperature scaling and top-p (nucleus) sampling
(info.pdf sec. 6)."""

import torch

from cs336_basics.model import TransformerLM, softmax


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zeroes out all but the smallest set of highest-probability tokens
    whose cumulative probability reaches `top_p`, then renormalizes
    (info.pdf sec. 6, eq. 24)."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the first token that crosses top_p too, so V(p) never comes up empty.
    cutoff = (cumulative - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    end_of_text_id: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
) -> list[int]:
    """Autoregressively samples up to `max_new_tokens` tokens continuing
    `prompt_ids`, stopping early on `end_of_text_id` (info.pdf sec. 6)."""
    model.eval()
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = ids[-model.context_length :]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]  # (vocab_size,)

        probs = softmax(logits / temperature, dim=-1)
        if top_p is not None:
            probs = _apply_top_p(probs, top_p)

        next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if next_id == end_of_text_id:
            break

    return ids
