"""Load OpenAI's pretrained GPT-2 weights (via Hugging Face) into `GPT`.

Hugging Face stores GPT-2 in its original layout: Q, K and V packed into one
matrix (`c_attn`), and heads flattened into the model dimension. This module
unpacks them into the per-head shapes used by `tiny_gpt.model`.
"""

import einops
import torch as t
from torch import Tensor

from tiny_gpt.config import PRESETS
from tiny_gpt.model import GPT


def convert_hf_gpt2_state_dict(hf: dict[str, Tensor], n_layers: int, n_heads: int) -> dict[str, Tensor]:
    sd = {
        "embed.W_E": hf["transformer.wte.weight"],
        "pos_embed.W_pos": hf["transformer.wpe.weight"],
        "ln_final.w": hf["transformer.ln_f.weight"],
        "ln_final.b": hf["transformer.ln_f.bias"],
        # GPT-2 ties the unembedding to the token embedding.
        "unembed.W_U": hf["transformer.wte.weight"].T,
        "unembed.b_U": t.zeros(hf["transformer.wte.weight"].shape[0]),
    }
    for layer in range(n_layers):
        hf_pre, pre = f"transformer.h.{layer}.", f"blocks.{layer}."
        sd[pre + "ln1.w"] = hf[hf_pre + "ln_1.weight"]
        sd[pre + "ln1.b"] = hf[hf_pre + "ln_1.bias"]
        sd[pre + "ln2.w"] = hf[hf_pre + "ln_2.weight"]
        sd[pre + "ln2.b"] = hf[hf_pre + "ln_2.bias"]

        W_Q, W_K, W_V = hf[hf_pre + "attn.c_attn.weight"].chunk(3, dim=1)
        b_Q, b_K, b_V = hf[hf_pre + "attn.c_attn.bias"].chunk(3, dim=0)
        for name, W, b in (("Q", W_Q, b_Q), ("K", W_K, b_K), ("V", W_V, b_V)):
            sd[pre + f"attn.W_{name}"] = einops.rearrange(
                W, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            sd[pre + f"attn.b_{name}"] = einops.rearrange(
                b, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads
            )
        sd[pre + "attn.W_O"] = einops.rearrange(
            hf[hf_pre + "attn.c_proj.weight"], "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=n_heads
        )
        sd[pre + "attn.b_O"] = hf[hf_pre + "attn.c_proj.bias"]

        sd[pre + "mlp.W_in"] = hf[hf_pre + "mlp.c_fc.weight"]
        sd[pre + "mlp.b_in"] = hf[hf_pre + "mlp.c_fc.bias"]
        sd[pre + "mlp.W_out"] = hf[hf_pre + "mlp.c_proj.weight"]
        sd[pre + "mlp.b_out"] = hf[hf_pre + "mlp.c_proj.bias"]
    return {k: v.contiguous() for k, v in sd.items()}


def load_pretrained_gpt2() -> GPT:
    """GPT-2 small (124M) with OpenAI's weights. Needs the `transformers` package."""
    from transformers import GPT2LMHeadModel

    cfg = PRESETS["gpt2-small"]
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = GPT(cfg)
    model.load_state_dict(convert_hf_gpt2_state_dict(hf_model.state_dict(), cfg.n_layers, cfg.n_heads))
    return model.eval()
