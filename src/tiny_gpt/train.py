"""Train a GPT from scratch on TinyStories.

    tiny-gpt-train --preset tiny --max-steps 5000 --wandb-project tiny-gpt
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch as t
from jaxtyping import Int
from torch import Tensor
from tqdm.auto import tqdm

from tiny_gpt.checkpoint import save_checkpoint
from tiny_gpt.config import PRESETS, GPTConfig
from tiny_gpt.induction import (
    common_tokens,
    induction_metrics,
    random_repeated_tokens,
    repeat_with_bos,
    repeated_training_rows,
)
from tiny_gpt.model import GPT, next_token_log_probs
from tiny_gpt.sample import generate


@dataclass
class TrainArgs:
    batch_size: int = 32
    max_steps: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 20
    eval_prompt: str = "Once upon a time"
    seed: int = 0
    out_dir: str = "checkpoints"
    # Extra copies kept for studying how the model changes during training (dense early, when induction heads form).
    snapshot_steps: str = "50,100,200,300,400,500,750,1000,1500,2000,3000,5000"
    # Measure induction heads (see induction.py) every this many steps; 0 turns it off. Cheap: two small batches.
    induction_every: int = 50
    induction_seqs: int = 16
    # Fraction of each batch replaced by random token sequences repeated over and over (see
    # induction.repeated_training_rows), which rewards copying from context. 0 trains on stories only.
    repeat_fraction: float = 0.0
    wandb_project: str | None = None
    wandb_name: str | None = None


def learning_rate(step: int, args: TrainArgs) -> float:
    """Linear warmup to `lr`, then cosine decay down to `min_lr` at `max_steps`."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, args: TrainArgs) -> t.optim.AdamW:
    # Weight decay on matrices only; biases and LayerNorm gains are left alone.
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return t.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


def mixed_precision_dtype(device: t.device) -> t.dtype | None:
    """bfloat16 on GPUs that support it natively (Ampere+), float16 on older ones (Kaggle's T4/P100)."""
    if device.type != "cuda":
        return None
    return t.bfloat16 if t.cuda.get_device_capability(device)[0] >= 8 else t.float16


@t.inference_mode()
def evaluate(
    model: GPT, eval_tokens: Int[Tensor, "n_seqs seq_len"], batch_size: int, amp_dtype: t.dtype | None
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_correct, total_predictions = 0.0, 0, 0
    for start in range(0, len(eval_tokens), batch_size):
        tokens = eval_tokens[start : start + batch_size].to(device).long()
        with t.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(tokens)
        log_probs = next_token_log_probs(logits.float(), tokens)
        total_loss -= log_probs.sum().item()
        total_correct += (logits[:, :-1].argmax(-1) == tokens[:, 1:]).sum().item()
        total_predictions += log_probs.numel()
    model.train()
    return {"eval_loss": total_loss / total_predictions, "eval_accuracy": total_correct / total_predictions}


def train(
    cfg: GPTConfig,
    args: TrainArgs,
    train_tokens: Int[Tensor, "n_seqs seq_len"],
    eval_tokens: Int[Tensor, "n_seqs seq_len"],
    tokenizer=None,
) -> GPT:
    """Train on random batches of `train_tokens`; evaluate, sample and checkpoint every `eval_every` steps."""
    t.manual_seed(args.seed)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = GPT(cfg).to(device).train()
    optimizer = make_optimizer(model, args)
    amp_dtype = mixed_precision_dtype(device)
    # float16 can underflow small gradients, so it needs loss scaling; bfloat16 does not.
    scaler = t.amp.GradScaler(device.type, enabled=amp_dtype == t.float16)
    batch_generator = t.Generator().manual_seed(args.seed)
    tokens_per_step = args.batch_size * train_tokens.shape[1]
    snapshot_steps = {int(s) for s in args.snapshot_steps.split(",") if s.strip()}

    bos = tokenizer.eos_token_id if tokenizer is not None else 0
    vocab = common_tokens(train_tokens[:10_000], cfg.d_vocab)
    n_repeat_rows = round(args.repeat_fraction * args.batch_size)
    repeat_generator = t.Generator().manual_seed(args.seed + 1)  # separate, so the story batches stay the same

    if args.induction_every:
        # The same sequences at every measurement, so changes come from the model, not the data.
        seq_len = (cfg.n_ctx - 1) // 2
        induction_random = random_repeated_tokens(vocab, seq_len, args.induction_seqs, bos, args.seed)
        induction_text = repeat_with_bos(eval_tokens[: args.induction_seqs, :seq_len], bos)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        induction_log = Path(args.out_dir) / "induction.jsonl"
        induction_log.write_text("")

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                **asdict(args),
                **asdict(cfg),
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_tokens_available": train_tokens.numel(),
                "repeat_rows_per_batch": n_repeat_rows,
                "device": t.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            },
        )
    samples: list[list] = []

    interval_start, interval_steps = time.perf_counter(), 0
    progress = tqdm(range(args.max_steps), desc="Training")
    for step in progress:
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch_idx = t.randint(0, len(train_tokens), (args.batch_size,), generator=batch_generator)
        tokens = train_tokens[batch_idx].to(device, non_blocking=True).long()
        if n_repeat_rows:
            tokens[:n_repeat_rows] = repeated_training_rows(vocab, n_repeat_rows, cfg.n_ctx, bos, repeat_generator)
        with t.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(tokens)
        loss = -next_token_log_probs(logits.float(), tokens).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        interval_steps += 1

        metrics = {"train_loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item()}
        progress.set_postfix(loss=f"{metrics['train_loss']:.3f}")

        if args.induction_every and (step + 1) % args.induction_every == 0:
            induction = induction_metrics(model, induction_random, induction_text)
            metrics.update(induction)
            with induction_log.open("a") as f:
                f.write(json.dumps({"step": step + 1, "train_loss": metrics["train_loss"], **induction}) + "\n")

        if (step + 1) % args.eval_every == 0 or step + 1 == args.max_steps:
            metrics["tokens_per_sec"] = interval_steps * tokens_per_step / (time.perf_counter() - interval_start)
            metrics.update(evaluate(model, eval_tokens, args.batch_size, amp_dtype))
            line = f"step {step + 1}: eval_loss {metrics['eval_loss']:.3f}, eval_accuracy {metrics['eval_accuracy']:.3f}"
            if "induction/max_induction_score" in metrics:
                line += f", max induction score {metrics['induction/max_induction_score']:.2f}"
            if tokenizer is not None:
                prompt = t.tensor(tokenizer.encode(args.eval_prompt), device=device)
                story = tokenizer.decode(
                    generate(model, prompt, 80, temperature=0.7, top_p=0.95, stop_token=tokenizer.eos_token_id,
                             seed=args.seed)
                )
                model.train()
                samples.append([step + 1, story])
                line += f"\n  {story!r}"
                if run is not None:
                    metrics["samples"] = wandb.Table(columns=["step", "text"], data=samples)
            tqdm.write(line)
            save_checkpoint(model, f"{args.out_dir}/model.pt", step=step + 1, eval_loss=metrics["eval_loss"])
            interval_start, interval_steps = time.perf_counter(), 0

        if step + 1 in snapshot_steps:
            save_checkpoint(model, f"{args.out_dir}/snapshots/step_{step + 1:06d}.pt", step=step + 1)

        if run is not None:
            run.log(metrics, step=step + 1)

    if run is not None:
        run.finish()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT from scratch on TinyStories.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--train-stories", type=int, default=300_000, help="How many stories to tokenize for training")
    parser.add_argument("--eval-stories", type=int, default=5_000, help="Held-out stories from the validation split")
    parser.add_argument("--num-proc", type=int, default=None, help="Processes for tokenizing (default: all CPUs)")
    for f in fields(TrainArgs):
        arg_type = type(f.default) if f.default is not None else str
        parser.add_argument("--" + f.name.replace("_", "-"), type=arg_type, default=f.default)
    ns = parser.parse_args()
    args = TrainArgs(**{f.name: getattr(ns, f.name) for f in fields(TrainArgs)})

    from transformers import GPT2TokenizerFast

    from tiny_gpt.data import load_tinystories

    cfg = PRESETS[ns.preset]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_tokens = load_tinystories("train", tokenizer, cfg.n_ctx, ns.train_stories, ns.num_proc)
    eval_tokens = load_tinystories("validation", tokenizer, cfg.n_ctx, ns.eval_stories, ns.num_proc)
    eval_tokens = eval_tokens[: args.eval_batches * args.batch_size]
    print(f"{train_tokens.numel():,} training tokens, {eval_tokens.numel():,} eval tokens")
    print(f"Training for {args.max_steps:,} steps = {args.max_steps * args.batch_size * cfg.n_ctx:,} tokens")

    train(cfg, args, train_tokens, eval_tokens, tokenizer)


if __name__ == "__main__":
    main()
