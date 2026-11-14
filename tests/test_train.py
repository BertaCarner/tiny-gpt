import json

import numpy as np
import torch as t

from tiny_gpt.checkpoint import load_checkpoint
from tiny_gpt.config import GPTConfig
from tiny_gpt.data import pack_sequences
from tiny_gpt.train import TrainArgs, evaluate, learning_rate, train

CFG = GPTConfig(d_model=32, d_head=8, d_mlp=64, n_heads=4, n_layers=1, n_ctx=16, d_vocab=10)


def counting_sequences(n: int) -> t.Tensor:
    """Each row counts upwards mod 10 from a random start, so every next token is predictable."""
    starts = t.randint(0, 10, (n, 1), generator=t.Generator().manual_seed(0))
    return ((starts + t.arange(CFG.n_ctx)) % 10).int()


def test_training_learns_a_predictable_pattern(tmp_path):
    tokens = counting_sequences(256)
    args = TrainArgs(batch_size=16, max_steps=150, lr=3e-3, warmup_steps=10, eval_every=150, eval_batches=2,
                     out_dir=str(tmp_path))
    model = train(CFG, args, tokens, tokens[:32])
    assert evaluate(model, tokens[:32], 16, None)["eval_accuracy"] > 0.95


def test_checkpoint_round_trip(tmp_path):
    tokens = counting_sequences(64)
    args = TrainArgs(batch_size=8, max_steps=5, eval_every=5, eval_batches=1, out_dir=str(tmp_path),
                     snapshot_steps="2,5")
    model = train(CFG, args, tokens, tokens[:8]).cpu().eval()
    reloaded = load_checkpoint(tmp_path / "model.pt")
    t.testing.assert_close(reloaded(tokens[:4].long()), model(tokens[:4].long()))
    final_snapshot = load_checkpoint(tmp_path / "snapshots" / "step_000005.pt")
    t.testing.assert_close(final_snapshot(tokens[:4].long()), model(tokens[:4].long()))
    assert (tmp_path / "snapshots" / "step_000002.pt").exists()


def test_induction_metrics_are_logged_during_training(tmp_path):
    tokens = counting_sequences(64)
    args = TrainArgs(batch_size=8, max_steps=6, eval_every=6, eval_batches=1, out_dir=str(tmp_path),
                     snapshot_steps="", induction_every=2, induction_seqs=4)
    train(CFG, args, tokens, tokens[:8])
    rows = [json.loads(line) for line in (tmp_path / "induction.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4, 6]
    assert {"induction/random_loss_second", "induction/text_loss_second", "induction_score/0.3"} <= rows[0].keys()


def test_training_with_repeated_rows_runs(tmp_path):
    tokens = counting_sequences(64)
    args = TrainArgs(batch_size=8, max_steps=3, eval_every=3, eval_batches=1, out_dir=str(tmp_path),
                     snapshot_steps="", repeat_fraction=0.25)
    train(CFG, args, tokens, tokens[:8])


def test_learning_rate_warms_up_then_decays_to_min():
    args = TrainArgs(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert learning_rate(0, args) == 1e-4
    assert abs(learning_rate(10, args) - 1e-3) < 1e-12
    assert abs(learning_rate(100, args) - 1e-4) < 1e-12
    assert all(learning_rate(s, args) >= learning_rate(s + 1, args) for s in range(10, 100))


def test_pack_sequences_separates_documents_and_drops_remainder():
    packed = pack_sequences([[1, 2, 3], [4, 5], [6, 7, 8, 9]], separator=0, seq_len=4)
    # Stream: 0 1 2 3 | 0 4 5 0 | 6 7 8 9  (12 tokens = 3 full rows)
    np.testing.assert_array_equal(packed, [[0, 1, 2, 3], [0, 4, 5, 0], [6, 7, 8, 9]])
    assert pack_sequences([[1, 2]], separator=0, seq_len=4).shape == (0, 4)
