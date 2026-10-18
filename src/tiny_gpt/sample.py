"""Text generation: greedy, temperature, top-k and top-p (nucleus) sampling."""

import argparse

import torch as t
from jaxtyping import Float, Int
from torch import Tensor

from tiny_gpt.model import GPT, KVCache


def sample_next_token(
    logits: Float[Tensor, "d_vocab"],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: t.Generator | None = None,
) -> int:
    if temperature == 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k is not None:
        kth_largest = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth_largest, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        # Drop a token once the tokens ranked above it already cover top_p of the probability.
        drop_sorted = (cumulative - sorted_logits.softmax(-1)) >= top_p
        logits = logits.masked_fill(drop_sorted.scatter(0, sorted_idx, drop_sorted), float("-inf"))
    probs = logits.softmax(-1)
    return int(t.multinomial(probs, num_samples=1, generator=generator))


@t.inference_mode()
def generate(
    model: GPT,
    tokens: Int[Tensor, "posn"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_token: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> Int[Tensor, "new_posn"]:
    """Extend `tokens` one token at a time.

    With `use_cache`, the prompt is processed once and each later step only runs the newest token through
    the model. Without it, each step reruns the whole sequence. Both give identical outputs.
    """
    generator = None
    if seed is not None:
        generator = t.Generator(device=tokens.device).manual_seed(seed)
    model.eval()
    n_ctx = model.cfg.n_ctx
    cache = KVCache.empty(model.cfg) if use_cache else None
    new_input = tokens
    for _ in range(max_new_tokens):
        if cache is None:
            logits = model(tokens[-n_ctx:].unsqueeze(0))[0, -1]
        else:
            if cache.seq_len + len(new_input) > n_ctx:
                # Context window full: positions can't shift inside a cache, so rebuild it over the latest n_ctx tokens.
                cache, new_input = KVCache.empty(model.cfg), tokens[-n_ctx:]
            logits = model(new_input.unsqueeze(0), cache)[0, -1]
        next_token = sample_next_token(logits, temperature, top_k, top_p, generator)
        tokens = t.cat([tokens, t.tensor([next_token], device=tokens.device)])
        new_input = tokens[-1:]
        if next_token == stop_token:
            break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained checkpoint, or with OpenAI's GPT-2.")
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", default=None, help="Path to a model.pt from tiny-gpt-train (default: GPT-2)")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if args.checkpoint:
        from tiny_gpt.checkpoint import load_checkpoint

        model = load_checkpoint(args.checkpoint)
    else:
        from tiny_gpt.pretrained import load_pretrained_gpt2

        model = load_pretrained_gpt2()
    tokens = t.tensor(tokenizer.encode(args.prompt))
    out = generate(
        model, tokens, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
        stop_token=tokenizer.eos_token_id, seed=args.seed,
    )
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
