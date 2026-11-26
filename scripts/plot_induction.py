"""Charts for the induction-head experiments: when the head forms, which heads form it, and where it is used.

    python scripts/plot_induction.py --checkpoints checkpoints/repeat --out-dir docs

Downloads the induction measurements of each run from W&B (cached in --out-dir/induction_history.json), and
reads the saved snapshots of the runs with repeated sequences for the attention charts and the ablations:
--checkpoints (repeated random sequences), --story-checkpoints (repeated story fragments) and
--dense-checkpoints (the rerun with a snapshot every 100 steps through the jump).
"""

import argparse
import itertools
import json
from pathlib import Path

import torch as t

RUNS = {  # W&B run id: (label, colour). Categorical slots 1-3 of the chart palette, as in plot_training.py.
    "yi5xfn8t": ("Stories only", "#2a78d6"),
    "cbrygscy": ("Stories only, shortformer", "#eb6834"),
    "pzqhjdxm": ("Shortformer + repeated random sequences", "#1baf7a"),
}
REPEAT_RUN = "pzqhjdxm"
STORY_RUN = "iey5kcca"  # repeated story fragments instead of random tokens (--repeat-source story)
DENSE_RUN = "cstyoyq0"  # the repeated-random-sequences run again, 10,000 steps, snapshots every 100 steps
LATER_HEAD = "3.2"  # the induction head in both of these runs
DENSE_JUMP = (8000, 8200, 8400, 8600, 8800)  # the snapshots shown inside the dense run's jump
INDUCTION_HEAD = "2.3"  # found by the induction scores logged during training
# The heads that head 2.3 depends on to find where to look, found by switching heads off (see copy_measurements)
FEEDING_HEADS = {"0.3": "#6250d6", "1.2": "#2a78d6"}
PHASE_CHANGE = (7700, 8300)  # steps where head 2.3's induction score climbs from 0.07 to 0.81
KEPT = (  # the logged measurements the charts use
    "train_loss", "induction/max_induction_score", "induction/random_loss_second",
    f"induction_score/{INDUCTION_HEAD}", *(f"prev_token_score/{head}" for head in FEEDING_HEADS),
    "induction/text_loss_first", "induction/text_loss_second", f"induction_score/{LATER_HEAD}",
    "prev_token_score/2.0",
)

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#f0efec"
HEAT = ["#fcfcfb", "#cfe0f6", "#94bcee", "#4f92e0", "#2a78d6", "#1c58a3", "#0f3a73"]  # one hue, light to dark


def download_histories(entity_project: str) -> dict:
    import wandb

    api = wandb.Api(timeout=60)
    histories = {}
    for run_id in [*RUNS, STORY_RUN, DENSE_RUN]:
        rows = api.run(f"{entity_project}/{run_id}").scan_history()
        histories[run_id] = [
            {k: round(r[k], 4) if k != "_step" else r[k] for k in ("_step", *KEPT) if r.get(k) is not None}
            for r in rows if any(r.get(k) is not None for k in KEPT)
        ]
    return histories


def style(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def title(fig, text: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, text, fontsize=11.5, color=TEXT, va="top", weight="bold")
    fig.text(0.012, 0.915, subtitle, fontsize=9, color=TEXT_2, va="top", linespacing=1.4)


def series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(r["_step"], r[key]) for r in history if r.get(key) is not None]
    return [s for s, _ in points], [v for _, v in points]


def smooth(values: list[float], window: int) -> list[float]:
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def plot_phase_change(histories: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.12)
    title(fig, "An induction head forms only when the training data rewards copying",
          "Induction score of the model's best head: how much it attends to the token after the earlier copy of\n"
          "the current token, in a sequence of random tokens repeated twice. 1 = always, 0 = never.")
    for run_id, (label, colour) in RUNS.items():
        steps, scores = series(histories[run_id], "induction/max_induction_score")
        ax.plot(steps, scores, color=colour, linewidth=2, label=label)
    style(ax, "Training step", "Best induction score")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 40_000)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.annotate("Sudden jump around step 8k\n(the phase change)", (8300, 0.81), xytext=(14_000, 0.62),
                fontsize=9, color=TEXT, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("Both stories-only runs stay near 0, even after 40k steps", (30_000, 0.05), xytext=(18_000, 0.25),
                fontsize=9, color=TEXT, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right", bbox_to_anchor=(1, 1.02))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_circuit(history: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    lo, hi = 6000, 10_000
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), dpi=150, facecolor=SURFACE, sharex=True,
                                      gridspec_kw=dict(height_ratios=[1, 1], hspace=0.18))
    fig.subplots_adjust(left=0.09, right=0.8, top=0.83, bottom=0.09)
    title(fig, "The induction head forms together with the two heads that feed it",
          f"Run with repeated random sequences, zoomed in on steps {lo:,} to {hi:,}. "
          "The shaded band is the phase change.")
    green = RUNS[REPEAT_RUN][1]
    for ax in (top, bottom):
        ax.axvspan(*PHASE_CHANGE, color=BAND, zorder=0)

    def line(ax, steps, values, colour, label, width=2):
        inside = [(s, v) for s, v in zip(steps, values) if lo <= s <= hi]
        ax.plot([s for s, _ in inside], [v for _, v in inside], color=colour, linewidth=width)
        ax.annotate(label, (inside[-1][0], inside[-1][1]), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=TEXT, va="center", annotation_clip=False)

    line(top, *series(history, f"induction_score/{INDUCTION_HEAD}"), green,
         f"Head {INDUCTION_HEAD} (layer 2):\ninduction score")
    for head, colour in FEEDING_HEADS.items():
        line(top, *series(history, f"prev_token_score/{head}"), colour,
             f"Head {head} (layer {head[0]}):\nprevious-token score")
    style(top, ylabel="Attention score")
    top.set_ylim(0, 1)

    steps, loss = series(history, "induction/random_loss_second")
    line(bottom, steps, loss, green, "Loss on the 2nd copy\nof random tokens")
    steps, train_loss = series(history, "train_loss")
    line(bottom, steps, smooth(train_loss, 50), MUTED, "Training loss\n(all data)", width=1.6)
    style(bottom, "Training step", "Loss (nats per token)")
    bottom.set_ylim(0, 8.5)
    bottom.set_xlim(lo, hi)
    bottom.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    bottom.annotate("7.6 = guessing among the\n2,000 possible tokens", (6100, 7.7), xytext=(6100, 5.2), fontsize=8.5,
                    color=TEXT_2, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def load_stories(split: str, n_stories: int, tokenizer, n_ctx: int) -> t.Tensor:
    """The first stories of a split, packed exactly as data.load_tinystories packs them (in batches of 1,000)."""
    import datasets

    from tiny_gpt.data import pack_sequences

    stream = datasets.load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    texts = [r["text"] for r in itertools.islice(stream, n_stories)]
    rows = [
        pack_sequences(tokenizer(texts[i : i + 1000], verbose=False)["input_ids"], tokenizer.eos_token_id, n_ctx)
        for i in range(0, len(texts), 1000)
    ]
    return t.cat([t.from_numpy(r.astype("int64")) for r in rows])


def head_pattern(model, tokens: t.Tensor, head: str) -> t.Tensor:
    from tiny_gpt.induction import run_with_patterns

    layer, index = map(int, head.split("."))
    return run_with_patterns(model, tokens)[1][:, layer, index]


def draw_pattern(ax, pattern: t.Tensor, labels: list[str] | None, seq_len: int, heading: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("heat", HEAT)
    image = ax.imshow(pattern.numpy(), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(heading, fontsize=10, color=TEXT, loc="left", pad=8)
    n = pattern.shape[0]
    for boundary in (0.5, seq_len + 0.5):  # after BOS, and between the two copies
        ax.axhline(boundary, color=AXIS, linewidth=0.8)
        ax.axvline(boundary, color=AXIS, linewidth=0.8)
    if labels:
        ax.set_xticks(range(n), labels, rotation=90, fontsize=6.5, color=TEXT_2)
        ax.set_yticks(range(n), labels, fontsize=6.5, color=TEXT_2)
    else:
        ticks = [0, seq_len // 2 + 1, seq_len + seq_len // 2 + 1]
        ax.set_xticks(ticks, ["BOS", "1st copy", "2nd copy"], fontsize=8, color=TEXT_2)
        ax.set_yticks(ticks, ["BOS", "1st copy", "2nd copy"], fontsize=8, color=TEXT_2, rotation=90, va="center")
    ax.set_xlabel("Looks at this position", fontsize=8.5, color=TEXT_2)
    ax.set_ylabel("Predicting at this position", fontsize=8.5, color=TEXT_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_attention_before_after(checkpoints: Path, tokens: t.Tensor, seq_len: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    from tiny_gpt.checkpoint import load_checkpoint

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.76, bottom=0.1, wspace=0.3)
    title(fig, f"Head {INDUCTION_HEAD} before and after the phase change",
          f"Attention of head {INDUCTION_HEAD} on {seq_len} random tokens repeated twice (darker = more attention).\n"
          "After the jump, every position in the 2nd copy looks at the token right after its earlier copy:\n"
          "the stripe just above the diagonal of the bottom-left block.")
    for ax, (name, step) in zip(axes, [("snapshots/step_007500.pt", "7,500"), ("snapshots/step_010000.pt", "10,000")]):
        model = load_checkpoint(checkpoints / name).eval()
        image = draw_pattern(ax, head_pattern(model, tokens, INDUCTION_HEAD)[0], None, seq_len, f"Step {step}")
    colourbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03)
    colourbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    colourbar.outline.set_visible(False)
    colourbar.set_label("Attention weight", fontsize=8.5, color=TEXT_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_random_vs_story(model, random_tokens: t.Tensor, story_tokens: t.Tensor, tokenizer, seq_len: int,
                         path: Path) -> None:
    import matplotlib.pyplot as plt

    def labels(tokens):
        return ["BOS"] + [tokenizer.decode([i]).replace("\n", "\\n").strip() or "␣" for i in tokens[1:].tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12, wspace=0.28)
    title(fig, f"The trained model uses head {INDUCTION_HEAD} for copying random tokens, but not stories",
          f"Final model, head {INDUCTION_HEAD}, on {seq_len} tokens repeated twice. On random tokens the 2nd copy "
          "looks back at the 1st (the stripe in the\nbottom-left block). On a story the same head only looks at "
          "the last few tokens (along the diagonal), so the repeat is never copied.")
    for ax, tokens, heading in [(axes[0], random_tokens, "Random tokens, repeated"),
                                (axes[1], story_tokens, "A story fragment, repeated")]:
        draw_pattern(ax, head_pattern(model, tokens[None], INDUCTION_HEAD)[0], labels(tokens), seq_len, heading)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_token_sets(results: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.3), dpi=150, facecolor=SURFACE,
                                      gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.66, bottom=0.12)
    title(fig, f"Head {INDUCTION_HEAD} learned where to look for any token, but to copy only the ones it practised",
          "Final model on random tokens repeated twice, for three sets of 2,000 tokens. During training the repeated\n"
          "sequences only ever used the first set. On the 1st copy there is nothing to copy yet, so it is the baseline.")
    rows = list(range(len(results)))[::-1]  # first result at the top
    blue = "#2a78d6"
    left.barh(rows, [r["score"] for r in results], color=blue, height=0.5)
    for y, r in zip(rows, results):
        left.annotate(f"{r['score']:.2f}", (r["score"], y), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=8.5, color=TEXT)
    left.set_xlim(0, 1.1)
    left.set_title("Looks at the right place\n(induction score, higher is better)", fontsize=9.5, color=TEXT,
                   loc="left")

    height = 0.32
    for offset, key, colour, label in [(height / 2 + 0.01, "loss_first", AXIS, "1st copy (no copying possible)"),
                                       (-height / 2 - 0.01, "loss_second", blue, "2nd copy")]:
        values = [r[key] for r in results]
        right.barh([y + offset for y in rows], values, color=colour, height=height, label=label)
        for y, v in zip(rows, values):
            right.annotate(f"{v:.2f}", (v, y + offset), xytext=(4, 0), textcoords="offset points", va="center",
                           fontsize=8, color=TEXT)
    right.set_xlim(0, 13.5)
    right.set_title("Copies the right token\n(loss, lower is better)", fontsize=9.5, color=TEXT, loc="left")
    right.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right", bbox_to_anchor=(1.02, 1.13),
                 ncols=1, handlelength=1, reverse=False)

    for ax in (left, right):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    left.set_yticks(rows, [r["name"] for r in results], fontsize=8.5, color=TEXT)
    right.set_yticks([])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def head_output(attn, normalized_resid_pre: t.Tensor, pattern: t.Tensor, head: int) -> t.Tensor:
    """What one head adds to the residual stream (without the output bias all heads share)."""
    import einops

    v = einops.einsum(normalized_resid_pre, attn.W_V[head], "batch posn d_model, d_model d_head -> batch posn d_head")
    z = einops.einsum(
        v + attn.b_V[head], pattern[:, head], "batch posn_K d_head, batch posn_Q posn_K -> batch posn_Q d_head"
    )
    return einops.einsum(z, attn.W_O[head], "batch posn d_head, d_head d_model -> batch posn d_model")


def mean_head_outputs(model, tokens: t.Tensor) -> dict[str, t.Tensor]:
    """Every head's output at each position, averaged over a batch."""
    captured, hooks = {}, []
    for layer, block in enumerate(model.blocks):
        hooks.append(block.attn.hook_pattern.register_forward_hook(
            lambda module, inputs, pattern, layer=layer: captured.__setitem__(("pattern", layer), pattern)))
        hooks.append(block.attn.register_forward_hook(
            lambda module, inputs, output, layer=layer: captured.__setitem__(("input", layer), inputs[0])))
    try:
        model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return {
        f"{layer}.{head}": head_output(block.attn, captured[("input", layer)], captured[("pattern", layer)], head).mean(0)
        for layer, block in enumerate(model.blocks) for head in range(model.cfg.n_heads)
    }


def copy_measurements(model, tokens: t.Tensor, seq_len: int, ablate: tuple[str, ...] = (),
                      means: dict[str, t.Tensor] | None = None, induction_head: str = INDUCTION_HEAD) -> dict:
    """The induction head's induction score and the loss on each copy of `[BOS] X X`.

    The heads in `ablate` are mean-ablated: their output is replaced by their average output (`means`), which
    removes what they do on this particular input but keeps the constant part that every input gets.
    """
    from tiny_gpt.induction import induction_scores, run_with_patterns

    hooks, patterns = [], {}
    for name in ablate:
        layer, head = map(int, name.split("."))
        attn = model.blocks[layer].attn
        hooks.append(attn.hook_pattern.register_forward_hook(
            lambda module, inputs, pattern, layer=layer: patterns.__setitem__(layer, pattern)))

        def replace(module, inputs, output, layer=layer, head=head, name=name):
            own = head_output(module, inputs[0], patterns[layer], head)
            return output - own + means[name][: output.shape[1]]

        hooks.append(attn.register_forward_hook(replace))
    try:
        logits, all_patterns = run_with_patterns(model, tokens)
    finally:
        for hook in hooks:
            hook.remove()
    log_probs = logits.log_softmax(-1)[:, :-1].gather(-1, tokens[:, 1:, None])[..., 0]
    layer, head = map(int, induction_head.split("."))
    return {
        "score": induction_scores(all_patterns, seq_len)[layer, head].item(),
        "loss_first": -log_probs[:, 1:seq_len].mean().item(),
        "loss_second": -log_probs[:, seq_len + 1 :].mean().item(),
    }


def plot_ablation(rows: list[dict], baseline: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 5.8), dpi=150, facecolor=SURFACE,
                                      gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.74, bottom=0.08)
    title(fig, f"Switching heads off one at a time: head {INDUCTION_HEAD} copies, heads 0.3 and 1.2 tell it where",
          "Final model, random tokens repeated twice. Each head in turn is replaced by its average output, which "
          "removes\nwhat it does on this input. Only the highlighted heads matter; the other 13 change almost nothing.")
    rows = rows[::-1]  # first row at the top
    names = [r["ablated"] for r in rows]
    important = {INDUCTION_HEAD, *FEEDING_HEADS, " + ".join(FEEDING_HEADS)}
    colours = ["#2a78d6" if r["ablated"] in important else AXIS for r in rows]
    left.barh(names, [r["score"] for r in rows], color=colours, height=0.6)
    right.barh(names, [r["loss_second"] for r in rows], color=colours, height=0.6)
    for ax, key in [(left, "score"), (right, "loss_second")]:
        for y, r in enumerate(rows):
            ax.annotate(f"{r[key]:.2f}", (r[key], y), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=TEXT)
        ax.axvline(baseline[key], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    left.tick_params(axis="y", labelsize=8.5, colors=TEXT)
    right.set_yticks([])
    left.set_xlim(0, 1.1)
    right.set_xlim(0, 13)
    left.set_title(f"Does head {INDUCTION_HEAD} still look at the right place?\n(induction score; dashed line: "
                   "nothing switched off)", fontsize=9, color=TEXT, loc="left")
    right.set_title("Does the model still copy?\n(loss on the 2nd copy, lower is better)", fontsize=9, color=TEXT,
                    loc="left")
    left.set_ylabel("Head switched off", fontsize=9, color=TEXT_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ablate_each_head(model, inputs: dict, seq_len: int, induction_head: str,
                     groups: list[tuple[str, ...]] = ()) -> dict:
    """Copying measured with nothing switched off, then with each head (and each group in `groups`) mean-ablated.

    `inputs` maps a kind of input to (tokens, reference): each head's average output comes from the reference
    sequences, which are of the same kind, so the constant part of what it writes is kept.
    """
    heads = [(f"{layer}.{head}",) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    results = {}
    for kind, (tokens, reference) in inputs.items():
        means = mean_head_outputs(model, reference)
        results[kind] = [{"ablated": "none", **copy_measurements(model, tokens, seq_len,
                                                                 induction_head=induction_head)}] + [
            {"ablated": " + ".join(group),
             **copy_measurements(model, tokens, seq_len, group, means, induction_head)}
            for group in heads + list(groups)
        ]
    return results


def plot_jump(checkpoints: Path, history: list[dict], tokens: t.Tensor, seq_len: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    from tiny_gpt.checkpoint import load_checkpoint

    logged = {r["_step"]: r for r in history}
    fig, axes = plt.subplots(1, len(DENSE_JUMP), figsize=(12, 3.9), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.68, bottom=0.08, wspace=0.12)
    title(fig, f"Inside the jump: head {LATER_HEAD} turns into an induction head in about 800 steps",
          f"A second run with repeated random sequences, saved every 100 steps. Attention of head {LATER_HEAD} "
          f"on {seq_len} random tokens repeated twice\n(darker = more attention). The stripe in the bottom-left "
          "block (looking at the token after the earlier copy) fades in, and copying follows.")
    for ax, step in zip(axes, DENSE_JUMP):
        model = load_checkpoint(checkpoints / f"snapshots/step_{step:06d}.pt").eval()
        draw_pattern(ax, head_pattern(model, tokens, LATER_HEAD)[0], None, seq_len, f"Step {step:,}")
        row = logged[step]
        ax.text(0, -0.04, f"induction score {row[f'induction_score/{LATER_HEAD}']:.2f}\n"
                f"loss on 2nd copy {row['induction/random_loss_second']:.2f}", transform=ax.transAxes,
                va="top", fontsize=8, color=TEXT_2)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        if step != DENSE_JUMP[0]:
            ax.set_ylabel("")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_story_copying(histories: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.12)
    title(fig, "Practising on repeated stories, instead of random tokens, makes the model copy stories",
          "Loss on a fragment of a held-out story repeated twice. Solid: the 2nd copy, which can be predicted by "
          "copying.\nDotted: the 1st copy, where there is nothing to copy yet.")
    for run_id, label, colour in [(REPEAT_RUN, "Practised on repeated random tokens", RUNS[REPEAT_RUN][1]),
                                  (STORY_RUN, "Practised on repeated story fragments", "#6250d6")]:
        for key, line_style, width in [("induction/text_loss_first", (0, (1, 2)), 1.4),
                                       ("induction/text_loss_second", "-", 2)]:
            steps, loss = series(histories[run_id], key)
            ax.plot(steps, smooth(loss, 5), color=colour, linewidth=width, linestyle=line_style,
                    label=label if key.endswith("second") else None)
    style(ax, "Training step", "Loss (nats per token)")
    ax.set_xlim(0, 15_000)
    ax.set_ylim(0, 3.5)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", default="checkpoints/repeat", help="Checkpoints of the repeated-sequences run")
    parser.add_argument("--story-checkpoints", default="checkpoints/story",
                        help="Checkpoints of the repeated-story-fragments run")
    parser.add_argument("--dense-checkpoints", default="checkpoints/dense",
                        help="Checkpoints of the rerun with snapshots every 100 steps")
    parser.add_argument("--wandb", default="berta-dades-personal/tiny-gpt")
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()
    out_dir, checkpoints = Path(args.out_dir), Path(args.checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "induction_history.json"
    if not cache.exists():
        cache.write_text(json.dumps(download_histories(args.wandb)), encoding="utf-8")
    histories = json.loads(cache.read_text(encoding="utf-8"))
    plot_phase_change(histories, out_dir / "induction_phase_change.png")
    plot_circuit(histories[REPEAT_RUN], out_dir / "induction_circuit.png")
    plot_story_copying(histories, out_dir / "induction_story_copying.png")

    from transformers import GPT2TokenizerFast

    from tiny_gpt.checkpoint import load_checkpoint
    from tiny_gpt.induction import common_tokens, random_repeated_tokens, repeat_with_bos

    t.set_grad_enabled(False)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    bos = tokenizer.eos_token_id
    model = load_checkpoint(checkpoints / "model.pt").eval()
    n_ctx = model.cfg.n_ctx
    # The 2,000 tokens the repeated training sequences were drawn from: train.py takes them from the first
    # 10,000 training rows.
    train_rows = load_stories("train", 13_000, tokenizer, n_ctx)[:10_000]
    counts = t.bincount(train_rows.flatten(), minlength=model.cfg.d_vocab)
    ranked = counts.argsort(descending=True)
    practised = common_tokens(train_rows, model.cfg.d_vocab)
    validation = load_stories("validation", 400, tokenizer, n_ctx)

    short = 20
    plot_attention_before_after(checkpoints, random_repeated_tokens(practised, short, 1, bos, seed=3), short,
                                out_dir / "induction_attention.png")
    story = validation[3, 1 : short + 1]
    plot_random_vs_story(model, random_repeated_tokens(practised, short, 1, bos, seed=3)[0],
                         repeat_with_bos(story[None], bos)[0], tokenizer, short, out_dir / "induction_random_vs_story.png")

    seq_len = (n_ctx - 1) // 2
    similar = common_tokens(validation, model.cfg.d_vocab)  # the most common tokens of other stories
    overlap = t.isin(similar, practised).float().mean().item()
    token_sets = []
    for name, vocab in [("The 2,000 practised tokens", practised),
                        (f"A similar set ({overlap:.0%} overlap)", similar),
                        ("2,000 never-practised tokens", ranked[2000:4000])]:
        tokens = random_repeated_tokens(vocab, seq_len, 32, bos)
        token_sets.append({"name": name, **copy_measurements(model, tokens, seq_len)})
    plot_token_sets(token_sets, out_dir / "induction_token_sets.png")

    # Switch off each head in turn, on repeated random tokens and on repeated stories. Each head's average output
    # comes from other sequences of the same kind, so the constant part of what it writes is kept.
    inputs = {
        "random": (random_repeated_tokens(practised, seq_len, 32, bos, seed=0),
                   random_repeated_tokens(practised, seq_len, 32, bos, seed=1)),
        "stories": (repeat_with_bos(validation[:32, 1 : seq_len + 1], bos),
                    repeat_with_bos(validation[32:64, 1 : seq_len + 1], bos)),
    }
    ablation = ablate_each_head(model, inputs, seq_len, INDUCTION_HEAD, [tuple(FEEDING_HEADS)])
    plot_ablation(ablation["random"][1:], ablation["random"][0], out_dir / "induction_ablation.png")

    # The two later runs: the same ablations on their final models, whose induction head is 3.2
    later = {}
    for name, run_checkpoints in [("story_run", Path(args.story_checkpoints)),
                                  ("dense_run", Path(args.dense_checkpoints))]:
        later[name] = ablate_each_head(load_checkpoint(run_checkpoints / "model.pt").eval(), inputs, seq_len,
                                       LATER_HEAD)
    plot_jump(Path(args.dense_checkpoints), histories[DENSE_RUN],
              random_repeated_tokens(practised, short, 1, bos, seed=3), short, out_dir / "induction_jump.png")

    results = {"token_sets": token_sets, "ablation": ablation, **{f"ablation_{k}": v for k, v in later.items()}}
    (out_dir / "induction_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
