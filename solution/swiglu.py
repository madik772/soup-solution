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
        cols = (col_off + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
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
        cols = (col_off + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
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
        gate_2d = gate.contiguous().reshape(-1, orig_shape[-1])
        up_2d = up.to(gate.dtype).contiguous().reshape(-1, orig_shape[-1])
        M, N = gate_2d.shape
        out_2d = torch.empty_like(gate_2d)

        _custom_fused_swiglu_fwd_kernel[(M,)](
            gate_2d, up_2d, out_2d,
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.orig_shape = orig_shape
        return out_2d.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        gate_2d, up_2d = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        grad_2d = grad_out.to(gate_2d.dtype).contiguous().reshape(-1, orig_shape[-1])
        M, N = gate_2d.shape
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
            num_warps=4,
        )
        return d_gate.reshape(*orig_shape), d_up.reshape(*orig_shape)


def swiglu(gate, up):
    return TritonSwiGLUFunction.apply(gate, up)
