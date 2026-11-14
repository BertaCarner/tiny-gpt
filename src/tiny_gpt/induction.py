"""Measure induction heads: attention heads that find the earlier copy of the current token and predict what came next.

Both tests show the model a sequence twice in a row, `[BOS] X X`. A model with induction heads predicts the second
copy of X far better than the first, because it can look back at what followed each token last time:
- random tokens (from the tokens common in the training data), where the first copy is unpredictable, so any
  improvement on the second copy can only come from copying;
- a passage of real held-out text, the more natural version of the same test.

During the second copy, an induction head at the query for token X[i] attends to X[i+1] in the first copy
(the token after the previous occurrence). A previous-token head attends to the position just before the query;
an induction head in a later layer needs one to know "what came before each earlier token".
"""

import torch as t
from jaxtyping import Float, Int
from torch import Tensor

from tiny_gpt.model import GPT, next_token_log_probs


def common_tokens(tokens: Int[Tensor, "..."], d_vocab: int, k: int = 2000) -> Int[Tensor, "k"]:
    """The `k` most frequent token ids in `tokens` (fewer if fewer distinct tokens appear)."""
    counts = t.bincount(tokens.flatten().long(), minlength=d_vocab)
    k = min(k, int((counts > 0).sum()))
    return counts.topk(k).indices


def repeat_with_bos(x: Int[Tensor, "batch seq_len"], bos: int) -> Int[Tensor, "batch 1+2*seq_len"]:
    return t.cat([t.full((len(x), 1), bos, dtype=x.dtype, device=x.device), x, x], dim=1)


def random_repeated_tokens(
    vocab: Int[Tensor, "k"], seq_len: int, batch_size: int, bos: int, seed: int = 0
) -> Int[Tensor, "batch 1+2*seq_len"]:
    generator = t.Generator().manual_seed(seed)
    idx = t.randint(0, len(vocab), (batch_size, seq_len), generator=generator)
    return repeat_with_bos(vocab.cpu()[idx], bos)


def repeated_training_rows(
    vocab: Int[Tensor, "k"], n_rows: int, n_ctx: int, bos: int, generator: t.Generator, min_period: int = 10
) -> Int[Tensor, "n_rows n_ctx"]:
    """Training rows `[BOS] X X X ...`: random common tokens repeated to fill the context.

    Each row repeats with its own random period, so the repeats can only be predicted by copying by content
    (find the earlier copy of the current token, predict what followed it), not by a fixed position offset.
    """
    max_period = (n_ctx - 1) // 2
    min_period = min(min_period, max_period)  # tiny test models have short contexts
    rows = []
    for _ in range(n_rows):
        period = int(t.randint(min_period, max_period + 1, (1,), generator=generator))
        x = vocab.cpu()[t.randint(0, len(vocab), (period,), generator=generator)]
        rows.append(t.cat([t.tensor([bos]), x.repeat(n_ctx // period + 1)])[:n_ctx])
    return t.stack(rows)


def induction_scores(
    pattern: Float[Tensor, "batch layer head posn_Q posn_K"], seq_len: int
) -> Float[Tensor, "layer head"]:
    """Average attention from each token of the second copy to the token after its first occurrence."""
    # Token X[i] of the second copy is at position seq_len + 1 + i, and X[i+1] of the first copy at i + 2,
    # so they are on the diagonal seq_len - 1 below the main one. Its first two entries are queries in the first
    # copy (which has no earlier occurrence to look at), so they are skipped.
    return pattern.diagonal(-(seq_len - 1), dim1=-2, dim2=-1)[..., 2:].mean((0, -1))


def previous_token_scores(pattern: Float[Tensor, "batch layer head posn_Q posn_K"]) -> Float[Tensor, "layer head"]:
    """Average attention from each position to the one just before it."""
    return pattern.diagonal(-1, dim1=-2, dim2=-1).mean((0, -1))


@t.inference_mode()
def run_with_patterns(
    model: GPT, tokens: Int[Tensor, "batch posn"]
) -> tuple[Float[Tensor, "batch posn d_vocab"], Float[Tensor, "batch layer head posn_Q posn_K"]]:
    """Logits, plus every layer's attention pattern read through the `hook_pattern` points."""
    patterns = []
    hooks = [
        block.attn.hook_pattern.register_forward_hook(lambda module, inputs, output: patterns.append(output))
        for block in model.blocks
    ]
    try:
        logits = model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return logits, t.stack(patterns, dim=1)


def repeated_sequence_metrics(
    model: GPT, tokens: Int[Tensor, "batch 1+2*seq_len"]
) -> tuple[dict[str, float], Float[Tensor, "layer head"], Float[Tensor, "layer head"]]:
    """Loss on the first and second copy, and induction and previous-token scores for every head."""
    seq_len = (tokens.shape[1] - 1) // 2
    logits, pattern = run_with_patterns(model, tokens)
    # log_probs[:, j] is for predicting tokens[:, j + 1]. The first token of each copy is skipped: after BOS
    # nothing can predict it, and the second copy's first token can't be copied yet (nothing has repeated).
    log_probs = next_token_log_probs(logits.float(), tokens)
    losses = {
        "loss_first": -log_probs[:, 1:seq_len].mean().item(),
        "loss_second": -log_probs[:, seq_len + 1 :].mean().item(),
    }
    return losses, induction_scores(pattern, seq_len), previous_token_scores(pattern)


def induction_metrics(
    model: GPT, random_tokens: Int[Tensor, "batch posn"], text_tokens: Int[Tensor, "batch posn"]
) -> dict[str, float]:
    """Everything logged during training, as flat W&B-style keys."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    metrics = {}
    random_losses, ind, prev = repeated_sequence_metrics(model, random_tokens.to(device).long())
    text_losses, _, _ = repeated_sequence_metrics(model, text_tokens.to(device).long())
    for name, value in random_losses.items():
        metrics[f"induction/random_{name}"] = value
    for name, value in text_losses.items():
        metrics[f"induction/text_{name}"] = value
    metrics["induction/max_induction_score"] = ind.max().item()
    metrics["induction/max_prev_token_score"] = prev.max().item()
    for layer in range(ind.shape[0]):
        for head in range(ind.shape[1]):
            metrics[f"induction_score/{layer}.{head}"] = ind[layer, head].item()
            metrics[f"prev_token_score/{layer}.{head}"] = prev[layer, head].item()
    model.train(was_training)
    return metrics
