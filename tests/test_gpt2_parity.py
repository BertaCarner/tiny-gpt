"""The from-scratch model, loaded with OpenAI's weights, must reproduce Hugging Face's GPT-2."""

import pytest
import torch as t

transformers = pytest.importorskip("transformers")

from tiny_gpt.pretrained import load_pretrained_gpt2  # noqa: E402
from tiny_gpt.sample import generate  # noqa: E402

TEXT = "The quick brown fox jumps over the lazy dog, and then it keeps running towards the river."


@pytest.fixture(scope="module")
def models():
    hf_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    return load_pretrained_gpt2(), hf_model, tokenizer


def test_logits_match_hugging_face(models):
    ours, hf_model, tokenizer = models
    tokens = t.tensor([tokenizer.encode(TEXT)])
    with t.inference_mode():
        expected = hf_model(tokens).logits
        actual = ours(tokens)
    t.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("use_cache", [True, False])
def test_greedy_generation_matches_hugging_face(models, use_cache):
    ours, hf_model, tokenizer = models
    prompt = t.tensor(tokenizer.encode("Machine learning is"))
    expected = hf_model.generate(prompt.unsqueeze(0), max_new_tokens=15, do_sample=False, pad_token_id=50256)[0]
    actual = generate(ours, prompt, max_new_tokens=15, temperature=0.0, use_cache=use_cache)
    assert tokenizer.decode(actual) == tokenizer.decode(expected)
