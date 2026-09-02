"""Transformer language model (info.pdf sec. 3): pre-norm decoder-only
Transformer with RMSNorm, SwiGLU feed-forward, and causal multi-head
self-attention with RoPE."""

import math

import torch
import torch.nn as nn


def _trunc_normal(tensor: torch.Tensor, std: float) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)


class Linear(nn.Module):
    """y = W x, no bias (info.pdf sec. 3.3.2)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        _trunc_normal(self.weight, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token-ID -> d_model vector lookup (info.pdf sec. 3.3.3)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _trunc_normal(self.weight, std=1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (info.pdf sec. 3.4.1, eq. 4). Upcasts to
    float32 internally so squaring the input doesn't overflow in fp16/bf16."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight.to(torch.float32)
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network: FFN(x) = W2(SiLU(W1 x) * W3 x)
    (info.pdf sec. 3.4.2, eq. 7)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)
        return self.w2(silu * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (info.pdf sec. 3.4.3): rotates pairs of query/key dims by an
    angle proportional to sequence position. Precomputes cos/sin for every
    position up to max_seq_len since the rotation doesn't depend on the
    input values, only on position and dimension."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        half = d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() * 2 / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, half)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]  # (..., seq_len, half)
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rotated_x1
        out[..., 1::2] = rotated_x2
        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax (info.pdf sec. 3.4.4, eq. 10): subtract
    the max along `dim` before exponentiating."""
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V (info.pdf sec.
    3.4.4, eq. 11). `mask[i, j] == True` means query i may attend to key j;
    False positions get -inf pre-softmax so they contribute 0 probability."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ v


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and RoPE applied to Q/K
    (info.pdf sec. 3.4.5, eq. 12-14)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch, seq_len, d_model = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(*batch, seq_len, self.num_heads, self.d_head)
            return t.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # RoPE rotation is identical across heads, so the head dim acts
            # as just another batch-like dimension (info.pdf sec. 3.4.5).
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        out = out.transpose(-3, -2).contiguous().view(*batch, seq_len, d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (info.pdf sec. 3.4, 3.5, Figure 2, eq. 15):
    y = x + Attn(RMSNorm(x)); z = y + FFN(RMSNorm(y))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Full Transformer language model (info.pdf sec. 3.1, 3.5, Figure 1):
    token embedding -> num_layers Transformer blocks -> final RMSNorm ->
    LM head, producing next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        d_head = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_head, context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
