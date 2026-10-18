"""TinyStories, tokenized with GPT-2's tokenizer and packed into fixed-length sequences."""

import os

import numpy as np
import torch as t
from jaxtyping import Int
from torch import Tensor


def pack_sequences(token_lists: list[list[int]], separator: int, seq_len: int) -> np.ndarray:
    """Join documents into one stream (each preceded by `separator`) and cut it into rows of `seq_len`.

    The leftover tokens at the end, fewer than `seq_len`, are dropped.
    """
    stream = np.concatenate([np.array([separator, *ids], dtype=np.int32) for ids in token_lists])
    n_rows = len(stream) // seq_len
    return stream[: n_rows * seq_len].reshape(n_rows, seq_len)


def _tokenize_and_pack(batch: dict[str, list], tokenizer, seq_len: int) -> dict[str, np.ndarray]:
    # verbose=False: stories longer than GPT-2's 1024-token context are fine here, since they get packed and cut.
    token_lists = tokenizer(batch["text"], verbose=False)["input_ids"]
    return {"tokens": pack_sequences(token_lists, tokenizer.eos_token_id, seq_len)}


def load_tinystories(
    split: str, tokenizer, seq_len: int, max_stories: int | None = None, num_proc: int | None = None
) -> Int[Tensor, "n_seqs seq_len"]:
    """Tokenized TinyStories as an int32 tensor of shape (number of sequences, seq_len).

    GPT-2's `<|endoftext|>` token separates stories, as in GPT-2's own training data.
    """
    import datasets

    if max_stories is not None:
        split = f"{split}[:{max_stories}]"
    dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)
    packed = dataset.map(
        _tokenize_and_pack,
        batched=True,
        batch_size=1000,
        fn_kwargs={"tokenizer": tokenizer, "seq_len": seq_len},
        remove_columns=dataset.column_names,
        num_proc=num_proc or os.cpu_count(),
        desc=f"Tokenizing {split}",
    )
    return t.from_numpy(np.asarray(packed.with_format("numpy")[:]["tokens"], dtype=np.int32))
