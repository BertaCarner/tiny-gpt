import torch as t
import torch.nn.functional as F

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT, LayerNorm, gelu_new
from tiny_gpt.sample import sample_next_token

SMALL = GPTConfig(d_model=32, d_head=8, d_mlp=64, n_heads=4, n_layers=2, n_ctx=16, d_vocab=100)


def test_output_shape():
    model = GPT(SMALL)
    tokens = t.randint(0, SMALL.d_vocab, (3, 10))
    assert model(tokens).shape == (3, 10, SMALL.d_vocab)


def test_causal_masking_hides_future_tokens():
    """Changing the last token must not change the logits at any earlier position."""
    t.manual_seed(0)
    model = GPT(SMALL).eval()
    tokens = t.randint(0, SMALL.d_vocab, (1, 10))
    changed = tokens.clone()
    changed[0, -1] = (tokens[0, -1] + 1) % SMALL.d_vocab
    t.testing.assert_close(model(tokens)[:, :-1], model(changed)[:, :-1])


def test_layer_norm_matches_pytorch():
    ln = LayerNorm(SMALL)
    ln.w.data, ln.b.data = t.randn(SMALL.d_model), t.randn(SMALL.d_model)
    x = t.randn(2, 5, SMALL.d_model)
    expected = F.layer_norm(x, (SMALL.d_model,), ln.w, ln.b, eps=SMALL.layer_norm_eps)
    t.testing.assert_close(ln(x), expected)


def test_gelu_new_matches_pytorch_tanh_approximation():
    x = t.linspace(-6, 6, 200)
    t.testing.assert_close(gelu_new(x), F.gelu(x, approximate="tanh"))


def test_attention_mask_buffer_is_not_saved_in_checkpoints():
    assert not any("IGNORE" in key for key in GPT(SMALL).state_dict())


def test_greedy_sampling_picks_argmax():
    logits = t.tensor([0.1, 3.0, 0.5])
    assert sample_next_token(logits, temperature=0.0) == 1


def test_top_k_only_samples_from_k_most_likely():
    logits = t.tensor([5.0, 4.0, 0.0, -1.0, -2.0])
    g = t.Generator().manual_seed(0)
    samples = {sample_next_token(logits, top_k=2, generator=g) for _ in range(200)}
    assert samples <= {0, 1}


def test_top_p_keeps_smallest_set_covering_p():
    # Probabilities ~ [0.64, 0.24, 0.09, 0.03]: top_p=0.8 needs the first two tokens.
    logits = t.tensor([0.64, 0.24, 0.09, 0.03]).log()
    g = t.Generator().manual_seed(0)
    samples = {sample_next_token(logits, top_p=0.8, generator=g) for _ in range(300)}
    assert samples == {0, 1}
