from dataclasses import replace

import pytest
import torch as t

from tiny_gpt.config import PRESETS, GPTConfig
from tiny_gpt.model import GPT, KVCache
from tiny_gpt.sample import generate

CFG = GPTConfig(d_model=32, d_head=8, d_mlp=64, n_heads=4, n_layers=2, n_ctx=16, d_vocab=100,
                positional_embedding="shortformer")


@pytest.fixture
def model() -> GPT:
    t.manual_seed(0)
    return GPT(CFG).eval()


@t.inference_mode()
def test_matches_transformer_lens_shortformer(model):
    """Same weights in TransformerLens's own shortformer implementation give the same logits."""
    transformer_lens = pytest.importorskip("transformer_lens")
    tl_cfg = transformer_lens.HookedTransformerConfig(
        d_model=CFG.d_model, d_head=CFG.d_head, d_mlp=CFG.d_mlp, n_heads=CFG.n_heads, n_layers=CFG.n_layers,
        n_ctx=CFG.n_ctx, d_vocab=CFG.d_vocab, act_fn="gelu_new", normalization_type="LN",
        positional_embedding_type="shortformer",
    )
    tl_model = transformer_lens.HookedTransformer(tl_cfg).eval()
    tl_model.load_state_dict(model.state_dict(), strict=False)
    tokens = t.randint(0, CFG.d_vocab, (2, 12))
    t.testing.assert_close(model(tokens), tl_model(tokens), atol=1e-5, rtol=1e-5)


@t.inference_mode()
def test_positions_are_the_only_difference_from_standard(model):
    standard = GPT(replace(CFG, positional_embedding="standard")).eval()
    standard.load_state_dict(model.state_dict())
    tokens = t.randint(0, CFG.d_vocab, (2, 12))
    assert not t.allclose(model(tokens), standard(tokens))
    # With every positional embedding zero, the two variants compute exactly the same thing.
    model.pos_embed.W_pos.zero_()
    standard.pos_embed.W_pos.zero_()
    t.testing.assert_close(model(tokens), standard(tokens))


@t.inference_mode()
def test_cached_forward_matches_full_forward(model):
    tokens = t.randint(0, CFG.d_vocab, (2, 12))
    cache = KVCache.empty(CFG)
    actual = t.cat([model(piece, cache) for piece in (tokens[:, :5], tokens[:, 5:6], tokens[:, 6:])], dim=1)
    t.testing.assert_close(actual, model(tokens), atol=1e-5, rtol=1e-5)


def test_generation_with_cache_matches_without(model):
    prompt = t.randint(0, CFG.d_vocab, (10,))
    kwargs = dict(max_new_tokens=20, temperature=0.0)
    with_cache = generate(model, prompt, use_cache=True, **kwargs)
    assert with_cache.tolist() == generate(model, prompt, use_cache=False, **kwargs).tolist()


def test_shortformer_preset_differs_from_tiny_only_in_positions():
    assert replace(PRESETS["tiny-shortformer"], positional_embedding="standard") == PRESETS["tiny"]


def test_unknown_positional_embedding_raises():
    with pytest.raises(ValueError, match="positional_embedding"):
        GPT(replace(CFG, positional_embedding="rotary"))
