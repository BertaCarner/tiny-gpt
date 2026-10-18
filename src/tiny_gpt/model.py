"""A GPT-2 style decoder-only transformer, written from scratch.

Every weight is an explicit `nn.Parameter` with a named shape, and every matrix
multiplication is an `einops.einsum` that spells out its dimensions. The layout
follows the GPT-2 architecture exactly (pre-LayerNorm, learned positional
embeddings, GELU MLP), so OpenAI's pretrained weights can be loaded into it.
"""

import math

import einops
import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn

from tiny_gpt.config import GPTConfig


def gelu_new(x: Tensor) -> Tensor:
    """The tanh approximation of GELU that GPT-2 was trained with."""
    return 0.5 * x * (1.0 + t.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class LayerNorm(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(-1, keepdim=True)
        var = residual.var(-1, keepdim=True, unbiased=False)
        normalized = (residual - mean) / (var + self.cfg.layer_norm_eps).sqrt()
        return normalized * self.w + self.b


class Embed(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "posn d_model -> batch posn d_model", batch=batch)


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        for W in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.normal_(W, std=cfg.init_range)
        # Not persistent: it's a constant, so it stays out of saved checkpoints.
        self.register_buffer("IGNORE", t.tensor(float("-inf")), persistent=False)

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Queries, keys and values for every head
        q = einops.einsum(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
        ) + self.b_Q
        k = einops.einsum(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
        ) + self.b_K
        v = einops.einsum(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
        ) + self.b_V

        # Scores between every query and every key, then scale, mask and softmax
        attn_scores = einops.einsum(
            q, k,
            "batch posn_Q n_heads d_head, batch posn_K n_heads d_head -> batch n_heads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Weighted sum of value vectors
        z = einops.einsum(
            v, attn_pattern,
            "batch posn_K n_heads d_head, batch n_heads posn_Q posn_K -> batch posn_Q n_heads d_head",
        )

        # Project back to d_model, summing over heads, and add the output bias
        return einops.einsum(
            z, self.W_O,
            "batch posn_Q n_heads d_head, n_heads d_head d_model -> batch posn_Q d_model",
        ) + self.b_O

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """Stops each position from attending to positions after it."""
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        future = t.triu(all_ones, diagonal=1).bool()
        return attn_scores.masked_fill(future, self.IGNORE)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        nn.init.normal_(self.W_out, std=cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(
            normalized_resid_mid, self.W_in,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
        ) + self.b_in
        post = gelu_new(pre)
        return einops.einsum(
            post, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model",
        ) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        return self.mlp(self.ln2(resid_mid)) + resid_mid


class Unembed(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=cfg.init_range)
        # GPT-2 has no unembedding bias; kept (frozen at zero) so the layer is general.
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(self, normalized_resid_final: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return einops.einsum(
            normalized_resid_final, self.W_U,
            "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
        ) + self.b_U


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        return self.unembed(self.ln_final(residual))


def next_token_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn_minus_1"]:
    """Log-probability the model gave to each actual next token."""
    log_probs = logits.log_softmax(dim=-1)
    return log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
