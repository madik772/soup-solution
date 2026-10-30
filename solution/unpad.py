import torch

_ACTIVE_ATTENTION_MASK = None
_ACTIVE_INDICES = None


class _UnpadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, non_pad_indices):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat_hidden = (
            hidden_states if hidden_states.ndim == 2 else hidden_states.reshape(-1, hidden_dim)
        )
        ctx.save_for_backward(non_pad_indices)
        ctx.orig_shape = orig_shape
        ctx.total_tokens = flat_hidden.shape[0]
        ctx.hidden_dim = hidden_dim

        unpadded_out = torch.empty(
            (non_pad_indices.numel(), hidden_dim),
            device=flat_hidden.device,
            dtype=flat_hidden.dtype,
        )
        torch.index_select(flat_hidden, 0, non_pad_indices, out=unpadded_out)
        return unpadded_out

    @staticmethod
    def backward(ctx, grad_unpadded):
        (non_pad_indices,) = ctx.saved_tensors
        grad_flat = torch.zeros(
            (ctx.total_tokens, ctx.hidden_dim),
            device=grad_unpadded.device,
            dtype=grad_unpadded.dtype,
        )
        grad_flat.index_copy_(0, non_pad_indices, grad_unpadded)
        return grad_flat.view(ctx.orig_shape), None


class _PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unpadded_hidden, non_pad_indices, total_tokens, hidden_dim):
        ctx.save_for_backward(non_pad_indices)
        ctx.hidden_dim = hidden_dim
        flat_out = torch.zeros(
            (total_tokens, hidden_dim),
            device=unpadded_hidden.device,
            dtype=unpadded_hidden.dtype,
        )
        flat_out.index_copy_(0, non_pad_indices, unpadded_hidden)
        return flat_out

    @staticmethod
    def backward(ctx, grad_flat):
        (non_pad_indices,) = ctx.saved_tensors
        grad_2d = grad_flat if grad_flat.ndim == 2 else grad_flat.reshape(-1, ctx.hidden_dim)
        grad_unpadded = torch.empty(
            (non_pad_indices.numel(), ctx.hidden_dim),
            device=grad_2d.device,
            dtype=grad_2d.dtype,
        )
        torch.index_select(grad_2d, 0, non_pad_indices, out=grad_unpadded)
        return grad_unpadded, None, None, None


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
    total_tokens = (
        original_shape[0] * original_shape[1]
        if len(original_shape) == 3
        else int(torch.prod(torch.tensor(original_shape[:-1])).item())
    )
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
