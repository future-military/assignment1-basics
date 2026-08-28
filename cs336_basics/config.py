from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    rope_theta: float

    def to_kwargs(self) -> dict[str, int | float]:
        return asdict(self)


SMOKE_MODEL_CONFIG = ModelConfig(
    vocab_size=1000,
    context_length=64,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=256,
    rope_theta=10000.0,
)