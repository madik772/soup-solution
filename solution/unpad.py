"""Padding removal: one Triton row gather that serves every move between padded and packed rows.

`_custom_row_gather_kernel` copies `out[i] = src[map[i]]` row by row, writes zeros where
`map[i] < 0`, can multiply by a scalar read from device memory and converts dtype on the store.
With map = the non-pad positions it unpads, with map = (full position -> packed row or -1) it
pads in a single pass (no zeros + index_copy), with map = token ids over the embedding matrix it
is the embedding lookup, and in the loss it scatters the head's gradient back with the loss
scale applied.
"""

import torch
import triton
import triton.language as tl

_ACTIVE_ATTENTION_MASK = None
_ACTIVE_INDICES = None

GATHER_BLOCK = 512


@triton.jit
def _custom_row_gather_kernel(
    Src, Map, Out, Scale,
    stride_src, stride_out,
    n_cols,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    src_row = tl.load(Map + row).to(tl.int64)
    valid = src_row >= 0
    src_ptr = Src + tl.maximum(src_row, 0) * stride_src
    out_ptr = Out + row * stride_out
    if HAS_SCALE:
        scale = tl.load(Scale).to(tl.float32)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        in_row = cols < n_cols
        vals = tl.load(src_ptr + cols, mask=in_row & valid, other=0.0).to(tl.float32)
        if HAS_SCALE:
            vals = vals * scale
        tl.store(out_ptr + cols, vals.to(Out.dtype.element_ty), mask=in_row)


def row_gather(src, row_map, n_rows=None, out=None, out_dtype=None, scale=None):
    """out[i] = src[row_map[i]] (zero where row_map[i] < 0) for 2D `src` with unit column stride."""
    n_rows = row_map.numel() if n_rows is None else n_rows
    n_cols = src.shape[1]
    if out is None:
        out = torch.empty((n_rows, n_cols), device=src.device, dtype=out_dtype or src.dtype)
    if n_rows == 0:
        return out
    _custom_row_gather_kernel[(n_rows,)](
        src, row_map, out, scale if scale is not None else row_map,
        src.stride(0), out.stride(0),
        n_cols,
        HAS_SCALE=scale is not None,
        BLOCK=GATHER_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out


def inverse_map(indices, total):
    """Full position -> packed row, -1 for the positions not in `indices`."""
    inv = torch.full((total,), -1, device=indices.device, dtype=torch.int32)
    inv[indices] = torch.arange(indices.numel(), device=indices.device, dtype=torch.int32)
    return inv


class _UnpadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, non_pad_indices):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.reshape(-1, hidden_dim)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        ctx.save_for_backward(non_pad_indices)
        ctx.orig_shape = orig_shape
        ctx.total_tokens = flat.shape[0]
        return row_gather(flat, non_pad_indices)

    @staticmethod
    def backward(ctx, grad_unpadded):
        (non_pad_indices,) = ctx.saved_tensors
        grad = grad_unpadded.contiguous()
        inv = inverse_map(non_pad_indices, ctx.total_tokens)
        return row_gather(grad, inv).view(ctx.orig_shape), None


class _PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unpadded_hidden, non_pad_indices, total_tokens, hidden_dim):
        ctx.save_for_backward(non_pad_indices)
        ctx.hidden_dim = hidden_dim
        src = unpadded_hidden.contiguous()
        return row_gather(src, inverse_map(non_pad_indices, total_tokens))

    @staticmethod
    def backward(ctx, grad_flat):
        (non_pad_indices,) = ctx.saved_tensors
        grad_2d = grad_flat.reshape(-1, ctx.hidden_dim)
        if not grad_2d.is_contiguous():
            grad_2d = grad_2d.contiguous()
        return row_gather(grad_2d, non_pad_indices), None, None, None


def _get_or_compute_non_pad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask is _ACTIVE_ATTENTION_MASK and _ACTIVE_INDICES is not None:
        return _ACTIVE_INDICES
    non_pad_indices = getattr(attention_mask, "_cached_non_pad_indices", None)
    if non_pad_indices is None:
        non_pad_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        try:
            attention_mask._cached_non_pad_indices = non_pad_indices
        except Exception:
            pass
    return non_pad_indices


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Compress [batch, seq_len, hidden_dim] to [total_real_tokens, hidden_dim]."""
    non_pad_indices = _get_or_compute_non_pad_indices(attention_mask)
    unpadded_hidden = _UnpadFunction.apply(hidden_states, non_pad_indices)
    return unpadded_hidden, non_pad_indices, hidden_states.shape


def pad_input(
    unpadded_hidden: torch.Tensor,
    non_pad_indices: torch.Tensor,
    original_shape: torch.Size,
):
    """Restore [total_real_tokens, hidden_dim] back to original_shape [batch, seq_len, hidden_dim]."""
    total_tokens = 1
    for d in original_shape[:-1]:
        total_tokens *= int(d)
    hidden_dim = original_shape[-1]
    flat_out = _PadFunction.apply(unpadded_hidden, non_pad_indices, total_tokens, hidden_dim)
    return flat_out.view(original_shape)


def set_active_attention_mask(attention_mask: torch.Tensor | None) -> None:
    global _ACTIVE_ATTENTION_MASK, _ACTIVE_INDICES
    if (
        attention_mask is not None
        and isinstance(attention_mask, torch.Tensor)
        and attention_mask.is_cuda
        and attention_mask.ndim == 2
    ):
        total_tokens = attention_mask.numel()
        non_pad_indices = _get_or_compute_non_pad_indices(attention_mask)
        n_valid = non_pad_indices.numel()
        if 0 < n_valid < total_tokens:
            _ACTIVE_ATTENTION_MASK = attention_mask
            _ACTIVE_INDICES = non_pad_indices
            return
    _ACTIVE_ATTENTION_MASK = None
    _ACTIVE_INDICES = None


def get_active_attention_mask() -> torch.Tensor | None:
    return _ACTIVE_ATTENTION_MASK


def get_active_non_pad_indices() -> torch.Tensor | None:
    return _ACTIVE_INDICES
