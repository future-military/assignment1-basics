import math
import torch
from torch import Tensor, nn

from .nn_utils import softmax

class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        weight = torch.empty(
            d_out,
            d_in,
            device=device,
            dtype=dtype,
        )

        std = math.sqrt(
            2 / (d_in + d_out)
        )

        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        weight = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )

        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype

        x_float = x.to(torch.float32)

        rms = torch.sqrt(
            x_float.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        normalized = x_float / rms
        output = normalized * self.weight

        return output.to(original_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.w1 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )

        self.w2 = Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.w3 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x)
        value = self.w3(x)

        silu_gate = gate * torch.sigmoid(gate)
        hidden = silu_gate * value

        return self.w2(hidden)

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = softmax(
        scores,
        dim=-1,
    )

    return attention_weights @ V

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        inverse_frequencies = theta ** (
            -dimension_indices / d_k
        )

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    ) -> Tensor:
        cos = self.cos_cache[token_positions].to(x.dtype)
        sin = self.sin_cache[token_positions].to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        ).flatten(-2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        # (..., sequence, d_model)
        # → (..., sequence, heads, head_dim)
        # → (..., heads, sequence, head_dim)
        return x.unflatten(
            -1,
            (self.num_heads, self.head_dim),
        ).transpose(-3, -2)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        sequence_length = x.shape[-2]

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )
            else:
                token_positions = token_positions.to(x.device)

            # Q/K에는 head 차원이 있지만 positions에는 없으므로
            # 크기 1짜리 head 축을 넣어 broadcasting한다.
            rope_positions = token_positions.unsqueeze(-2)

            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )

        attended = scaled_dot_product_attention(
            Q,
            K,
            V,
            causal_mask,
        )

        # (..., heads, sequence, head_dim)
        # → (..., sequence, heads, head_dim)
        # → (..., sequence, d_model)
        combined = attended.transpose(-3, -2).flatten(-2)

        return self.output_proj(combined)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        # Pre-norm attention + residual connection
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )

        # Pre-norm feed-forward + residual connection
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            d_in=d_model,
            d_out=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_indices: Tensor) -> Tensor:
        sequence_length = in_indices.shape[-1]

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        token_positions = torch.arange(
            sequence_length,
            device=in_indices.device,
        )

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits