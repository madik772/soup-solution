import torch
import triton
import triton.language as tl

@triton.jit
def _custom_rmsnorm_fwd_kernel(
    X, Y, W, Rsqrt,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row_idx * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x * rsqrt * w).to(tl.float16)

    y_ptrs = Y + row_idx * stride_y_row + cols
    tl.store(y_ptrs, out, mask=mask)

@triton.jit
def _custom_rmsnorm_bwd_kernel(
    dY, X, W, Rsqrt, dX,
    stride_dy_row, stride_x_row, stride_dx_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(dY + row_idx * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(Rsqrt + row_idx)

    dy_w = dy * w
    c1 = tl.sum(dy_w * x, axis=0) / N
    dx = rsqrt * (dy_w - x * (rsqrt * rsqrt * c1))

    tl.store(dX + row_idx * stride_dx_row + cols, dx.to(tl.float16), mask=mask)

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(N)
        _custom_rmsnorm_fwd_kernel[(M,)](
            x_2d, y, weight, rsqrt,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        ctx.save_for_backward(x_2d, weight, rsqrt)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.N = N
        return y.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rsqrt = ctx.saved_tensors
        orig_shape = dy.shape
        dy_2d = dy.reshape(-1, orig_shape[-1])
        M, N = dy_2d.shape
        dx = torch.empty_like(x_2d)

        _custom_rmsnorm_bwd_kernel[(M,)](
            dy_2d, x_2d, weight, rsqrt, dx,
            dy_2d.stride(0), x_2d.stride(0), dx.stride(0),
            ctx.N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=4,
        )
        return dx.reshape(*orig_shape), None, None
