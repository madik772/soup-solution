import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def _custom_rmsnorm_fwd_kernel(
    X, Y, W, Rsqrt,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    x_row_ptr = X + row_idx * stride_x_row
    y_row_ptr = Y + row_idx * stride_y_row

    var_sum = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        var_sum += tl.sum(x * x, axis=0)

    var = var_sum / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x * rsqrt * w).to(Y.dtype.element_ty)
        tl.store(y_row_ptr + cols, out, mask=mask)


@triton.jit
def _custom_rmsnorm_bwd_kernel(
    dY, X, W, Rsqrt, dX,
    stride_dy_row, stride_x_row, stride_dx_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    dy_row_ptr = dY + row_idx * stride_dy_row
    x_row_ptr = X + row_idx * stride_x_row
    dx_row_ptr = dX + row_idx * stride_dx_row
    rsqrt = tl.load(Rsqrt + row_idx)

    dot_sum = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dot_sum += tl.sum(dy * w * x, axis=0)

    c1 = dot_sum / N

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dy_w = dy * w
        dx = rsqrt * (dy_w - x * (rsqrt * rsqrt * c1))
        tl.store(dx_row_ptr + cols, dx.to(dX.dtype.element_ty), mask=mask)


@triton.jit
def _custom_fused_add_rmsnorm_fwd_kernel(
    Res, X, W, ResOut, Y, Rsqrt,
    stride_res_row, stride_x_row, stride_res_out_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    res_row_ptr = Res + row_idx * stride_res_row
    x_row_ptr = X + row_idx * stride_x_row
    res_out_row_ptr = ResOut + row_idx * stride_res_out_row
    y_row_ptr = Y + row_idx * stride_y_row

    var_sum = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        res = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_res = res + x
        tl.store(res_out_row_ptr + cols, sum_res.to(ResOut.dtype.element_ty), mask=mask)
        var_sum += tl.sum(sum_res * sum_res, axis=0)

    var = var_sum / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        sum_res = tl.load(res_out_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        out = (sum_res * rsqrt * w).to(Y.dtype.element_ty)
        tl.store(y_row_ptr + cols, out, mask=mask)


@triton.jit
def _custom_fused_add_rmsnorm_bwd_kernel(
    dY, dResOut, ResOut, W, Rsqrt, dSumRes,
    stride_dy_row, stride_dres_out_row, stride_res_out_row, stride_dsum_row,
    N,
    HAS_DRES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    dy_row_ptr = dY + row_idx * stride_dy_row
    res_out_row_ptr = ResOut + row_idx * stride_res_out_row
    dsum_row_ptr = dSumRes + row_idx * stride_dsum_row
    rsqrt = tl.load(Rsqrt + row_idx)

    dot_sum = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_res = tl.load(res_out_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dot_sum += tl.sum(dy * w * sum_res, axis=0)

    c1 = dot_sum / N

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_res = tl.load(res_out_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dy_w = dy * w
        d_sum = rsqrt * (dy_w - sum_res * (rsqrt * rsqrt * c1))
        if HAS_DRES:
            dres = tl.load(
                dResOut + row_idx * stride_dres_out_row + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            d_sum = d_sum + dres
        tl.store(dsum_row_ptr + cols, d_sum.to(dSumRes.dtype.element_ty), mask=mask)


def _as_contiguous_2d(t: torch.Tensor, last_dim: int) -> torch.Tensor:
    if not t.is_contiguous():
        t = t.contiguous()
    return t if t.ndim == 2 else t.view(-1, last_dim)


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = _as_contiguous_2d(x, N)
        M = x_2d.shape[0]
        y = torch.empty_like(x_2d)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

        _custom_rmsnorm_fwd_kernel[(M,)](
            x_2d, y, weight, rsqrt,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(x_2d, weight, rsqrt)
        ctx.N = N
        return y.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rsqrt = ctx.saved_tensors
        orig_shape = dy.shape
        N = ctx.N
        dy_2d = _as_contiguous_2d(dy, N)
        M = dy_2d.shape[0]
        dx = torch.empty_like(x_2d)

        _custom_rmsnorm_bwd_kernel[(M,)](
            dy_2d, x_2d, weight, rsqrt, dx,
            dy_2d.stride(0), x_2d.stride(0), dx.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        if weight is None or not weight.requires_grad:
            return dx.view(*orig_shape), None, None

        dw = (dy_2d.float() * (x_2d.float() * rsqrt.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return dx.view(*orig_shape), dw, None


class TritonFusedAddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, x, weight, eps=1e-6):
        ctx.set_materialize_grads(False)
        orig_shape = x.shape
        N = orig_shape[-1]
        res_2d = _as_contiguous_2d(residual, N)
        x_cast = x if x.dtype == res_2d.dtype else x.to(res_2d.dtype)
        x_2d = _as_contiguous_2d(x_cast, N)
        M = x_2d.shape[0]

        res_out_2d = torch.empty_like(res_2d)
        y_2d = torch.empty_like(x_2d)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

        _custom_fused_add_rmsnorm_fwd_kernel[(M,)](
            res_2d, x_2d, weight, res_out_2d, y_2d, rsqrt,
            res_2d.stride(0), x_2d.stride(0), res_out_2d.stride(0), y_2d.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(res_out_2d, weight, rsqrt)
        ctx.orig_shape = orig_shape
        ctx.N = N
        return y_2d.view(*orig_shape), res_out_2d.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy, d_res_out):
        res_out_2d, weight, rsqrt = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        M, N = res_out_2d.shape

        if dy is None:
            dy_2d = torch.zeros_like(res_out_2d)
        else:
            dy_cast = dy if dy.dtype == res_out_2d.dtype else dy.to(res_out_2d.dtype)
            dy_2d = _as_contiguous_2d(dy_cast, N)

        has_dres = d_res_out is not None
        if has_dres:
            dres_cast = d_res_out if d_res_out.dtype == res_out_2d.dtype else d_res_out.to(res_out_2d.dtype)
            dres_2d = _as_contiguous_2d(dres_cast, N)
            dres_stride = dres_2d.stride(0)
        else:
            dres_2d = res_out_2d
            dres_stride = 0

        d_sum_2d = torch.empty_like(res_out_2d)

        _custom_fused_add_rmsnorm_bwd_kernel[(M,)](
            dy_2d, dres_2d, res_out_2d, weight, rsqrt, d_sum_2d,
            dy_2d.stride(0), dres_stride, res_out_2d.stride(0), d_sum_2d.stride(0),
            N,
            HAS_DRES=has_dres,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        d_sum = d_sum_2d.view(*orig_shape)
        if weight is None or not weight.requires_grad:
            return d_sum, d_sum, None, None

        dw = (dy_2d.float() * (res_out_2d.float() * rsqrt.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return d_sum, d_sum, dw, None


def fused_add_rmsnorm(residual, x, weight, eps=1e-6):
    return TritonFusedAddRMSNormFunction.apply(residual, x, weight, eps)
