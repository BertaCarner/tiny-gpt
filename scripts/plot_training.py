"""Plot the loss curves of a training run logged to W&B.

    python scripts/plot_training.py berta-dades-personal/tiny-gpt/uaa83yct

Downloads the run's history, and writes it as JSON + a PNG chart to --out-dir,
along with the sample stories the run logged during training.
"""

import argparse
import json
import tempfile
from pathlib import Path

SERIES = {  # (label, colour): categorical slots 1 and 2 of the chart palette, as in benchmarks/kv_cache.py
    "train": ("Train loss", "#2a78d6"),
    "eval": ("Held-out loss", "#eb6834"),
}


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average, so early points average over fewer steps."""
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def plot(history: dict, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    surface, text_primary, text_secondary, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)

    train_steps, train_loss = zip(*history["train"])
    label, colour = SERIES["train"]
    ax.plot(train_steps, train_loss, color=colour, linewidth=0.6, alpha=0.25, zorder=2)
    ax.plot(train_steps, smooth(list(train_loss), 50), color=colour, linewidth=2, label=label, zorder=3)

    eval_steps, eval_loss = [r["step"] for r in history["eval"]], [r["loss"] for r in history["eval"]]
    label, colour = SERIES["eval"]
    ax.plot(eval_steps, eval_loss, color=colour, linewidth=2, marker="o", markersize=5, label=label, zorder=4,
            markeredgecolor=surface, markeredgewidth=1.5)
    ax.annotate(f"{label}\n{eval_loss[-1]:.2f}", (eval_steps[-1], eval_loss[-1]), xytext=(8, 6),
                textcoords="offset points", va="bottom", fontsize=9, color=text_primary)

    ax.set_title(title, loc="left", fontsize=11, color=text_primary, pad=12)
    ax.set_xlabel("Training step", fontsize=9, color=text_secondary)
    ax.set_ylabel("Cross-entropy loss (nats per token)", fontsize=9, color=text_secondary)
    ax.set_ylim(bottom=0, top=8)  # the first steps start near ln(50257) = 10.8; cut them off to show the rest
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=9, labelcolor=text_primary, loc="upper right")
    ax.set_xlim(left=0, right=eval_steps[-1] * 1.18)  # room for the end-of-line label
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="W&B run path: entity/project/run_id")
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()

    import wandb

    run = wandb.Api(timeout=60).run(args.run)
    rows = list(run.scan_history())
    history = {
        "run": args.run,
        "config": dict(run.config),
        "runtime_seconds": run.summary.get("_runtime"),
        "train": [(r["_step"], r["train_loss"]) for r in rows if r.get("train_loss") is not None],
        "eval": [
            {"step": r["_step"], "loss": r["eval_loss"], "accuracy": r["eval_accuracy"]}
            for r in rows if r.get("eval_loss") is not None
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        with run.file(run.summary["samples"]["path"]).download(root=tmp) as samples_file:
            samples = json.load(samples_file)
    history["samples"] = [{"step": step, "text": text} for step, text in samples["data"]]

    out_dir = Path(args.out_dir)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    cfg = history["config"]
    title = f"Training on TinyStories, {cfg['n_params'] / 1e6:.0f}M parameters, {cfg['device']}"
    plot(history, title, out_dir / "training_loss.png")

    print("| Step | Held-out loss | Next-token accuracy |")
    print("|---:|---:|---:|")
    for r in history["eval"]:
        print(f"| {r['step']:,} | {r['loss']:.2f} | {r['accuracy']:.1%} |")


if __name__ == "__main__":
    main()
