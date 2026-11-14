from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GPTConfig:
    d_model: int = 768
    d_vocab: int = 50257
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    # "standard": GPT-2's, added to the token embeddings at the start of the residual stream.
    # "shortformer": kept out of the residual stream and added to the input of every attention layer's queries
    # and keys only (Press et al., 2021), so values carry only token information. Makes induction heads form
    # far more easily in small models.
    positional_embedding: str = "standard"


TINY = GPTConfig(d_model=256, d_head=64, d_mlp=1024, n_heads=4, n_layers=4, n_ctx=256)

PRESETS: dict[str, GPTConfig] = {
    # Same shape as OpenAI's GPT-2 small (124M parameters), so its weights load directly.
    "gpt2-small": GPTConfig(),
    # Small model for training on TinyStories on a single consumer GPU.
    "tiny": TINY,
    # The same, with only the positional embeddings changed.
    "tiny-shortformer": replace(TINY, positional_embedding="shortformer"),
}
