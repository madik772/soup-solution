import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def _custom_fused_swiglu_fwd_kernel(
    Gate, Up, Out,
    stride_g_row, stride_g_col,
    stride_u_row, stride_u_col,
    stride_o_row, stride_o_col,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    g_row_ptr = Gate + row_idx * stride_g_row
    u_row_ptr = Up + row_idx * stride_u_row
    o_row_ptr = Out + row_idx * stride_o_row

    for col_off in range(0, n_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        g_ptrs = g_row_ptr + cols * stride_g_col
        u_ptrs = u_row_ptr + cols * stride_u_col
        o_ptrs = o_row_ptr + cols * stride_o_col

        gate = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

        neg_gate = -gate
        denom = 1.0 + tl.exp(neg_gate)
        sig = 1.0 / denom
        silu = gate * sig
        out = silu * up

        tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _custom_fused_swiglu_bwd_kernel(
    dOut, Gate, Up, dGate, dUp,
    stride_do_row, stride_do_col,
    stride_g_row, stride_g_col,
    stride_u_row, stride_u_col,
    stride_dg_row, stride_dg_col,
    stride_du_row, stride_du_col,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    do_row_ptr = dOut + row_idx * stride_do_row
    g_row_ptr = Gate + row_idx * stride_g_row
    u_row_ptr = Up + row_idx * stride_u_row
    dg_row_ptr = dGate + row_idx * stride_dg_row
    du_row_ptr = dUp + row_idx * stride_du_row

    for col_off in range(0, n_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        do_ptrs = do_row_ptr + cols * stride_do_col
        g_ptrs = g_row_ptr + cols * stride_g_col
        u_ptrs = u_row_ptr + cols * stride_u_col
        dg_ptrs = dg_row_ptr + cols * stride_dg_col
        du_ptrs = du_row_ptr + cols * stride_du_col

        d_out = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

        neg_gate = -gate
        denom = 1.0 + tl.exp(neg_gate)
        sig = 1.0 / denom
        silu = gate * sig
        d_up = d_out * silu

        one_minus_sig = 1.0 - sig
        dsilu = sig + gate * sig * one_minus_sig
        d_gate = d_out * up * dsilu

        tl.store(dg_ptrs, d_gate.to(dGate.dtype.element_ty), mask=mask)
        tl.store(du_ptrs, d_up.to(dUp.dtype.element_ty), mask=mask)


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        orig_shape = gate.shape
        N = orig_shape[-1]
        if not gate.is_contiguous():
            gate = gate.contiguous()
        gate_2d = gate if gate.ndim == 2 else gate.view(-1, N)

        up_cast = up if up.dtype == gate.dtype else up.to(gate.dtype)
        if not up_cast.is_contiguous():
            up_cast = up_cast.contiguous()
        up_2d = up_cast if up_cast.ndim == 2 else up_cast.view(-1, N)

        M = gate_2d.shape[0]
        out_2d = torch.empty_like(gate_2d)

        _custom_fused_swiglu_fwd_kernel[(M,)](
            gate_2d, up_2d, out_2d,
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
        base_g = gate._base if gate._base is not None else gate
        base_u = up._base if up._base is not None else up
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.orig_shape = orig_shape
        ctx.inplace_bwd = (not base_g.is_leaf) and (not base_u.is_leaf) and (up.dtype == gate.dtype)
        return out_2d.view(*orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        gate_2d, up_2d = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        M, N = gate_2d.shape

        grad_cast = grad_out if grad_out.dtype == gate_2d.dtype else grad_out.to(gate_2d.dtype)
        if not grad_cast.is_contiguous():
            grad_cast = grad_cast.contiguous()
        grad_2d = grad_cast if grad_cast.ndim == 2 else grad_cast.view(M, N)

        if ctx.inplace_bwd:
            d_gate = gate_2d
            d_up = up_2d
        else:
            d_gate = torch.empty_like(gate_2d)
            d_up = torch.empty_like(up_2d)

        _custom_fused_swiglu_bwd_kernel[(M,)](
            grad_2d, gate_2d, up_2d, d_gate, d_up,
            grad_2d.stride(0), grad_2d.stride(1),
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            d_gate.stride(0), d_gate.stride(1),
            d_up.stride(0), d_up.stride(1),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
        return d_gate.view(*orig_shape), d_up.view(*orig_shape)


def swiglu(gate, up):
    return TritonSwiGLUFunction.apply(gate, up)


def swiglu_packed_fwd(gate_up, n_cols):
    """silu(gate) * up for a fused [M, 2 * n_cols] gate|up projection (one GEMM for both)."""
    M = gate_up.shape[0]
    gate = gate_up[:, :n_cols]
    up = gate_up[:, n_cols:]
    out = torch.empty((M, n_cols), device=gate_up.device, dtype=gate_up.dtype)
    if M == 0:
        return out
    _custom_fused_swiglu_fwd_kernel[(M,)](
        gate, up, out,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        out.stride(0), out.stride(1),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


def swiglu_packed_bwd_(grad_out, gate_up, n_cols):
    """Overwrite the fused gate|up buffer with (d gate | d up): it is then the gradient of the
    fused projection's output, and one GEMM takes it back to the layer input."""
    M = gate_up.shape[0]
    if M == 0:
        return gate_up
    gate = gate_up[:, :n_cols]
    up = gate_up[:, n_cols:]
    _custom_fused_swiglu_bwd_kernel[(M,)](
        grad_out, gate, up, gate, up,
        grad_out.stride(0), grad_out.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return gate_up
