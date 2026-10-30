import torch
import triton
import triton.language as tl

@triton.jit
def _custom_rmsnorm_fwd_kernel(
    X, Y, W,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row_idx * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Вычисление дисперсии в FP32 для численной стабильности
    var = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(var + eps)

    w_ptrs = W + cols
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    out = (x * rsqrt * w).to(tl.float16)
    y_ptrs = Y + row_idx * stride_y_row + cols
    tl.store(y_ptrs, out, mask=mask)

@triton.jit
def _custom_rmsnorm_bwd_kernel(
    dY, X, W, dX,
    stride_dy_row, stride_x_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row_idx * stride_x_row + cols
    dy_ptrs = dY + row_idx * stride_dy_row + cols
    w_ptrs = W + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(var + eps)

    dy_w = dy * w
    c1 = tl.sum(dy_w * x, axis=0) / N
    dx = rsqrt * (dy_w - x * (rsqrt * rsqrt) * c1)

    dx_ptrs = dX + row_idx * stride_dx_row + cols
    tl.store(dx_ptrs, dx.to(tl.float16), mask=mask)

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)

        BLOCK_SIZE = triton.next_power_of_2(N)
        _custom_rmsnorm_fwd_kernel[(M,)](
            x_2d, y, weight,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4 if N <= 2048 else 8,
        )
        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight = ctx.saved_tensors
        orig_shape = dy.shape
        dy_2d = dy.reshape(-1, orig_shape[-1])
        M, N = dy_2d.shape
        dx = torch.empty_like(x_2d)

        _custom_rmsnorm_bwd_kernel[(M,)](
            dy_2d, x_2d, weight, dx,
            dy_2d.stride(0), x_2d.stride(0), dx.stride(0),
            N, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=4 if N <= 2048 else 8,
        )

        # Вычисление градиента по весам при необходимости
        if weight.requires_grad:
            var = torch.mean(x_2d.float() ** 2, dim=-1, keepdim=True)
            rsqrt = torch.rsqrt(var + ctx.eps)
            dw = (dy_2d * (x_2d * rsqrt).to(dy_2d.dtype)).sum(dim=0)
        else:
            dw = None

        return dx.reshape(*orig_shape), dw, None
