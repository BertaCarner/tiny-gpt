"""Benchmark generation speed with and without the KV cache.

    python benchmarks/kv_cache.py --preset gpt2-small

Uses random weights (speed doesn't depend on the weights) and greedy decoding
with no stop token, so every run generates exactly the requested number of tokens.
Writes a Markdown table to stdout, and JSON + a PNG chart to --out-dir.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch as t

from tiny_gpt import GPT, PRESETS
from tiny_gpt.sample import generate

SERIES = {  # (label, colour): categorical slots 1 and 2 of the chart palette
    True: ("With KV cache", "#2a78d6"),
    False: ("Without cache", "#eb6834"),
}


def time_generation(model: GPT, prompt: t.Tensor, new_tokens: int, use_cache: bool) -> float:
    if prompt.device.type == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    generate(model, prompt, new_tokens, temperature=0.0, use_cache=use_cache)
    if prompt.device.type == "cuda":
        t.cuda.synchronize()
    return time.perf_counter() - start


def hardware_name(device: t.device) -> str:
    if device.type == "cuda":
        return t.cuda.get_device_name(device)
    return f"CPU ({platform.processor() or platform.machine()}, {t.get_num_threads()} threads)"


def plot(results: list[dict], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    surface, text_primary, text_secondary, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    x = [r["new_tokens"] for r in results]
    for use_cache, (label, colour) in SERIES.items():
        key = "seconds_cached" if use_cache else "seconds_uncached"
        y = [r[key] for r in results]
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=5, label=label, zorder=3,
                markeredgecolor=surface, markeredgewidth=1.5)
        ax.annotate(f"{label}\n{y[-1]:.1f} s", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=text_primary)
    ax.set_title(title, loc="left", fontsize=11, color=text_primary, pad=12)
    ax.set_xlabel("Tokens generated", fontsize=9, color=text_secondary)
    ax.set_ylabel("Time (seconds)", fontsize=9, color=text_secondary)
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=9, labelcolor=text_primary, loc="upper left")
    ax.set_xlim(right=x[-1] * 1.25)  # room for the end-of-line labels
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gpt2-small")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(0)
    cfg = PRESETS[args.preset]
    model = GPT(cfg).to(device).eval()
    prompt = t.randint(0, cfg.d_vocab, (args.prompt_len,), device=device)
    time_generation(model, prompt, 8, use_cache=True)  # warm-up
    time_generation(model, prompt, 8, use_cache=False)

    results = []
    for n in args.new_tokens:
        cached = time_generation(model, prompt, n, use_cache=True)
        uncached = time_generation(model, prompt, n, use_cache=False)
        results.append({"new_tokens": n, "seconds_cached": cached, "seconds_uncached": uncached})

    hardware = hardware_name(device)
    print(f"{args.preset}, {args.prompt_len}-token prompt, greedy decoding, {hardware}\n")
    print("| Tokens generated | Without cache | With KV cache | Speedup |")
    print("|---:|---:|---:|---:|")
    for r in results:
        n, c, u = r["new_tokens"], r["seconds_cached"], r["seconds_uncached"]
        print(f"| {n} | {u:.2f} s ({n / u:.1f} tok/s) | {c:.2f} s ({n / c:.1f} tok/s) | {u / c:.1f}x |")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"preset": args.preset, "prompt_len": args.prompt_len, "hardware": hardware, "torch": t.__version__}
    (out_dir / "kv_cache_benchmark.json").write_text(json.dumps({**meta, "results": results}, indent=2))
    title = f"Generation time, {args.preset}, {'GPU' if device.type == 'cuda' else 'CPU'}"
    plot(results, title, out_dir / "kv_cache_benchmark.png")


if __name__ == "__main__":
    main()
