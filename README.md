# tiny-gpt

GPT-2 implemented from scratch in PyTorch: every layer written by hand, and **verified against OpenAI's pretrained weights**. Loaded with GPT-2 small's weights, this implementation produces the same logits as Hugging Face's reference model (to within 1e-4) and the same greedy generations.

```console
$ tiny-gpt-sample "The best thing about learning machine learning is" --seed 0 --max-new-tokens 30
The best thing about learning machine learning is that when it's done correctly, you can
easily see the the trends that are happening and see what your machine learning model predicts.
```

## What's implemented

| Component | File | Notes |
|---|---|---|
| Token + learned positional embeddings | [`model.py`](src/tiny_gpt/model.py) | |
| Multi-head causal self-attention | [`model.py`](src/tiny_gpt/model.py) | Per-head weight tensors, every contraction written as a named `einsum` |
| LayerNorm, GELU MLP, pre-LN residual blocks | [`model.py`](src/tiny_gpt/model.py) | Matches GPT-2's architecture exactly |
| Loading OpenAI's GPT-2 weights | [`pretrained.py`](src/tiny_gpt/pretrained.py) | Unpacks Hugging Face's fused `c_attn` QKV matrix into per-head shapes |
| Greedy, temperature, top-k, top-p sampling | [`sample.py`](src/tiny_gpt/sample.py) | |
| KV cache | [`model.py`](src/tiny_gpt/model.py), [`sample.py`](src/tiny_gpt/sample.py) | Each new token is processed once; rebuilt when generation outgrows the context window |
| TinyStories tokenization and sequence packing | [`data.py`](src/tiny_gpt/data.py) | Stories joined with `<\|endoftext\|>`, cut into fixed-length rows |
| Training loop | [`train.py`](src/tiny_gpt/train.py) | AdamW, warmup + cosine LR schedule, gradient clipping, mixed precision, W&B logging |

No `nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention`: the point is to show every operation explicitly.

## Verification

The test suite ([`tests/`](tests)) runs on every push via GitHub Actions:

- **Parity with GPT-2:** logits match `transformers.GPT2LMHeadModel` on real text, and greedy generation produces identical tokens, with and without the KV cache.
- **KV cache:** feeding a sequence through the cache in pieces gives the same logits as one full pass, and generation with the cache matches generation without it token for token, including past the end of the context window.
- **Causality:** changing a future token never changes the logits at earlier positions.
- **Components:** LayerNorm and GELU match PyTorch's built-in implementations; top-k and top-p only ever sample from the allowed set of tokens.
- **Training:** the training loop learns a simple deterministic sequence to >95% accuracy, checkpoints reload to identical outputs, and the learning-rate schedule and data packing are checked directly.

## KV cache benchmark

Without a cache, generating each token reruns the model over the whole sequence so far, so every token costs more than the one before. The cache stores each layer's keys and values, so each step only processes the newest token and the speed stays flat:

![Generation time with and without the KV cache](docs/kv_cache_benchmark.png)

| Tokens generated | Without cache | With KV cache | Speedup |
|---:|---:|---:|---:|
| 32 | 6.96 s (4.6 tok/s) | 2.02 s (15.9 tok/s) | 3.4x |
| 64 | 15.27 s (4.2 tok/s) | 3.84 s (16.7 tok/s) | 4.0x |
| 128 | 45.92 s (2.8 tok/s) | 7.75 s (16.5 tok/s) | 5.9x |
| 256 | 129.47 s (2.0 tok/s) | 16.60 s (15.4 tok/s) | 7.8x |

GPT-2 small (124M), 32-token prompt, greedy decoding, on a laptop CPU (Intel Core, 4 threads). Reproduce with `python benchmarks/kv_cache.py`; raw numbers in [`docs/kv_cache_benchmark.json`](docs/kv_cache_benchmark.json).

## Quick start

```bash
pip install -e ".[dev]"
pytest                      # downloads GPT-2 small (~500 MB) on first run
tiny-gpt-sample "Once upon a time" --temperature 0.8 --top-p 0.95
```

### Training

```bash
pip install -e ".[train]"
tiny-gpt-train --preset tiny --max-steps 5000 --wandb-project tiny-gpt
tiny-gpt-sample "Once upon a time" --checkpoint checkpoints/model.pt
```

To train on a free GPU, open [`notebooks/train_on_kaggle.ipynb`](notebooks/train_on_kaggle.ipynb) on Kaggle.

### As a library

```python
import torch as t
from tiny_gpt import GPT, PRESETS

model = GPT(PRESETS["tiny"])                    # randomly initialised, ready to train
logits = model(t.randint(0, 50257, (2, 64)))    # (batch, position, vocab)
```

## Roadmap

- [x] From-scratch GPT-2, verified against OpenAI's weights
- [x] Sampling: greedy, temperature, top-k, top-p
- [ ] Train a small model on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with loss curves and sample outputs
- [x] KV cache, with a generation-speed benchmark
- [ ] Beam search
- [ ] Scaling experiment: how loss changes with model width and depth
- [ ] Trained model on the Hugging Face Hub, with a Gradio demo

## Acknowledgements

The model code started from the "Transformer from Scratch" exercises in [ARENA](https://github.com/callummcdougall/ARENA_3.0), a curriculum for ML alignment research engineering. This repository extends it into a standalone, tested package.
