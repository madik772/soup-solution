import torch
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
