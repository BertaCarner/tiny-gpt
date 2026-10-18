import pytest
import torch as t

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT, KVCache
from tiny_gpt.sample import generate

CFG = GPTConfig(d_model=32, d_head=8, d_mlp=64, n_heads=4, n_layers=2, n_ctx=16, d_vocab=100)


@pytest.fixture
def model() -> GPT:
    t.manual_seed(0)
    return GPT(CFG).eval()


@t.inference_mode()
def test_cached_forward_matches_full_forward(model):
    """Feeding a sequence in pieces through a cache gives the same logits as one full pass."""
    tokens = t.randint(0, CFG.d_vocab, (2, 12))
    expected = model(tokens)

    cache = KVCache.empty(CFG)
    pieces = [tokens[:, :5], tokens[:, 5:8], tokens[:, 8:9], tokens[:, 9:12]]
    actual = t.cat([model(piece, cache) for piece in pieces], dim=1)

    t.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert cache.seq_len == 12


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_generation_with_cache_matches_without(model, temperature):
    # 10 prompt tokens + 20 new tokens goes past the 16-token context window, so this also covers the cache rebuild.
    prompt = t.randint(0, CFG.d_vocab, (10,))
    kwargs = dict(max_new_tokens=20, temperature=temperature, seed=0)
    with_cache = generate(model, prompt, use_cache=True, **kwargs)
    without_cache = generate(model, prompt, use_cache=False, **kwargs)
    assert with_cache.tolist() == without_cache.tolist()


@t.inference_mode()
def test_select_keeps_chosen_batch_rows(model):
    tokens = t.randint(0, CFG.d_vocab, (3, 6))
    cache = KVCache.empty(CFG)
    model(tokens, cache)
    selected = cache.select(t.tensor([2, 2, 0]))
    next_tokens = t.randint(0, CFG.d_vocab, (3, 1))
    expected = model(t.cat([tokens[[2, 2, 0]], next_tokens], dim=1))[:, -1]
    t.testing.assert_close(model(next_tokens, selected)[:, -1], expected, atol=1e-5, rtol=1e-5)


def test_forward_past_context_window_raises(model):
    cache = KVCache.empty(CFG)
    model(t.randint(0, CFG.d_vocab, (1, 16)), cache)
    with pytest.raises(ValueError, match="context window"):
        model(t.randint(0, CFG.d_vocab, (1, 1)), cache)
