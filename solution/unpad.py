import torch

_ACTIVE_ATTENTION_MASK = None
_ACTIVE_INDICES = None


class _UnpadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, indices):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat_hidden = (
            hidden_states if hidden_states.ndim == 2 else hidden_states.reshape(-1, hidden_dim)
        )
        ctx.save_for_backward(indices)
        ctx.orig_shape = orig_shape
        ctx.total_tokens = flat_hidden.shape[0]
        ctx.hidden_dim = hidden_dim

        unpadded_out = torch.empty(
            (indices.numel(), hidden_dim),
            device=flat_hidden.device,
            dtype=flat_hidden.dtype,
        )
        torch.index_select(flat_hidden, 0, indices, out=unpadded_out)
        return unpadded_out

    @staticmethod
    def backward(ctx, grad_unpadded):
        (indices,) = ctx.saved_tensors
        grad_flat = torch.zeros(
            (ctx.total_tokens, ctx.hidden_dim),
            device=grad_unpadded.device,
            dtype=grad_unpadded.dtype,
        )
        grad_flat.index_copy_(0, indices, grad_unpadded)
        return grad_flat.view(ctx.orig_shape), None


class _PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unpadded_hidden, indices, total_tokens, hidden_dim):
        ctx.save_for_backward(indices)
        ctx.hidden_dim = hidden_dim
        flat_out = torch.zeros(
            (total_tokens, hidden_dim),
            device=unpadded_hidden.device,
            dtype=unpadded_hidden.dtype,
        )
        flat_out.index_copy_(0, indices, unpadded_hidden)
        return flat_out

    @staticmethod
    def backward(ctx, grad_flat):
        (indices,) = ctx.saved_tensors
        grad_2d = grad_flat if grad_flat.ndim == 2 else grad_flat.reshape(-1, ctx.hidden_dim)
        grad_unpadded = torch.empty(
            (indices.numel(), ctx.hidden_dim),
            device=grad_2d.device,
            dtype=grad_2d.dtype,
        )
        torch.index_select(grad_2d, 0, indices, out=grad_unpadded)
        return grad_unpadded, None, None, None


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Compress [batch, seq_len, hidden_dim] to [num_valid_tokens, hidden_dim]."""
    if attention_mask is _ACTIVE_ATTENTION_MASK and _ACTIVE_INDICES is not None:
        indices = _ACTIVE_INDICES
    else:
        indices = getattr(attention_mask, "_cached_unpad_indices", None)
        if indices is None:
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            try:
                attention_mask._cached_unpad_indices = indices
            except Exception:
                pass
    unpadded_hidden = _UnpadFunction.apply(hidden_states, indices)
    return unpadded_hidden, indices, hidden_states.shape


def pad_input(unpadded_hidden: torch.Tensor, indices: torch.Tensor, original_shape: torch.Size):
    """Restore [num_valid_tokens, hidden_dim] back to original_shape [batch, seq_len, hidden_dim]."""
    total_tokens = (
        original_shape[0] * original_shape[1]
        if len(original_shape) == 3
        else int(torch.prod(torch.tensor(original_shape[:-1])).item())
    )
    hidden_dim = original_shape[-1]
    flat_out = _PadFunction.apply(unpadded_hidden, indices, total_tokens, hidden_dim)
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
        indices = getattr(attention_mask, "_cached_unpad_indices", None)
        if indices is None:
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            try:
                attention_mask._cached_unpad_indices = indices
            except Exception:
                pass
        n_valid = indices.numel()
        if 0 < n_valid < total_tokens:
            _ACTIVE_ATTENTION_MASK = attention_mask
            _ACTIVE_INDICES = indices
            return
    _ACTIVE_ATTENTION_MASK = None
    _ACTIVE_INDICES = None


def get_active_attention_mask() -> torch.Tensor | None:
    return _ACTIVE_ATTENTION_MASK
