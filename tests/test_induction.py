import torch as t

from tiny_gpt.config import GPTConfig
from tiny_gpt.induction import (
    common_tokens,
    induction_scores,
    previous_token_scores,
    random_repeated_tokens,
    repeated_training_rows,
    run_with_patterns,
)
from tiny_gpt.model import GPT

CFG = GPTConfig(d_model=32, d_head=8, d_mlp=64, n_heads=4, n_layers=2, n_ctx=16, d_vocab=100)


def test_random_repeated_tokens_is_bos_then_two_copies():
    vocab = t.tensor([5, 6, 7])
    tokens = random_repeated_tokens(vocab, seq_len=6, batch_size=4, bos=0)
    assert tokens.shape == (4, 13)
    assert (tokens[:, 0] == 0).all()
    assert t.equal(tokens[:, 1:7], tokens[:, 7:])
    assert t.isin(tokens[:, 1:], vocab).all()


def test_repeated_training_rows_repeat_with_varied_periods():
    vocab = t.arange(50, 100)
    rows = repeated_training_rows(vocab, n_rows=20, n_ctx=64, bos=0, generator=t.Generator().manual_seed(0))
    assert rows.shape == (20, 64)
    assert (rows[:, 0] == 0).all() and t.isin(rows[:, 1:], vocab).all()
    periods = set()
    for row in rows[:, 1:]:
        # The smallest shift that maps the row onto itself; random tokens could make it smaller than the
        # period actually drawn only by an unlikely coincidence.
        period = next(p for p in range(1, 64) if t.equal(row[p:], row[:-p]))
        assert 10 <= period <= 31
        periods.add(period)
    assert len(periods) > 5  # not one fixed offset the model could learn by position alone


def test_common_tokens_ranks_by_frequency():
    tokens = t.tensor([3, 3, 3, 1, 1, 2])
    assert common_tokens(tokens, d_vocab=10, k=2).tolist() == [3, 1]
    assert len(common_tokens(tokens, d_vocab=10)) == 3  # only 3 distinct tokens appear


def test_scores_on_hand_made_attention_patterns():
    """Head 0 does exactly what an induction head should, head 1 attends to the previous token, head 2 to BOS."""
    seq_len = 5
    n = 1 + 2 * seq_len
    pattern = t.zeros(1, 1, 3, n, n)
    pattern[..., 2, :, 0] = 1.0
    for q in range(n):
        pattern[0, 0, 1, q, max(q - 1, 0)] = 1.0
        # X[i] of the second copy (position seq_len + 1 + i) looks at X[i+1] of the first copy (position i + 2).
        pattern[0, 0, 0, q, q - (seq_len - 1) if q > seq_len else 0] = 1.0

    ind = induction_scores(pattern, seq_len)[0]
    prev = previous_token_scores(pattern)[0]
    t.testing.assert_close(ind, t.tensor([1.0, 0.0, 0.0]))
    assert prev[1] > 0.9 and prev[0] < 0.5 and prev[2] < 0.2


def test_run_with_patterns_reads_every_layer_and_cleans_up():
    t.manual_seed(0)
    model = GPT(CFG).eval()
    tokens = t.randint(0, CFG.d_vocab, (2, 10))
    logits, pattern = run_with_patterns(model, tokens)

    assert pattern.shape == (2, CFG.n_layers, CFG.n_heads, 10, 10)
    t.testing.assert_close(pattern.sum(-1), t.ones(2, CFG.n_layers, CFG.n_heads, 10))
    assert (pattern.triu(diagonal=1) == 0).all()  # causal: nothing attends to the future
    t.testing.assert_close(logits, model(tokens))
    assert all(len(block.attn.hook_pattern._forward_hooks) == 0 for block in model.blocks)
