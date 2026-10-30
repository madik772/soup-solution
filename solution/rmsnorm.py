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

    cols0 = tl.arange(0, BLOCK_SIZE)
    mask0 = cols0 < N
    x0 = tl.load(x_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)

    cols1 = BLOCK_SIZE + cols0
    mask1 = cols1 < N
    x1 = tl.load(x_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)

    var = (tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w0 = tl.load(W + cols0, mask=mask0, other=0.0).to(tl.float32)
    w1 = tl.load(W + cols1, mask=mask1, other=0.0).to(tl.float32)

    tl.store(y_row_ptr + cols0, (x0 * rsqrt * w0).to(Y.dtype.element_ty), mask=mask0)
    tl.store(y_row_ptr + cols1, (x1 * rsqrt * w1).to(Y.dtype.element_ty), mask=mask1)


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

    cols0 = tl.arange(0, BLOCK_SIZE)
    mask0 = cols0 < N
    dy0 = tl.load(dy_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    x0 = tl.load(x_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    w0 = tl.load(W + cols0, mask=mask0, other=0.0).to(tl.float32)
    dy_w0 = dy0 * w0

    cols1 = BLOCK_SIZE + cols0
    mask1 = cols1 < N
    dy1 = tl.load(dy_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    x1 = tl.load(x_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    w1 = tl.load(W + cols1, mask=mask1, other=0.0).to(tl.float32)
    dy_w1 = dy1 * w1

    c1 = (tl.sum(dy_w0 * x0, axis=0) + tl.sum(dy_w1 * x1, axis=0)) / N
    scale = rsqrt * rsqrt * c1

    dx0 = rsqrt * (dy_w0 - x0 * scale)
    dx1 = rsqrt * (dy_w1 - x1 * scale)

    tl.store(dx_row_ptr + cols0, dx0.to(dX.dtype.element_ty), mask=mask0)
    tl.store(dx_row_ptr + cols1, dx1.to(dX.dtype.element_ty), mask=mask1)


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

    cols0 = tl.arange(0, BLOCK_SIZE)
    mask0 = cols0 < N
    res0 = tl.load(res_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    x0 = tl.load(x_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    sum0 = res0 + x0
    tl.store(res_out_row_ptr + cols0, sum0.to(ResOut.dtype.element_ty), mask=mask0)

    cols1 = BLOCK_SIZE + cols0
    mask1 = cols1 < N
    res1 = tl.load(res_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    x1 = tl.load(x_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    sum1 = res1 + x1
    tl.store(res_out_row_ptr + cols1, sum1.to(ResOut.dtype.element_ty), mask=mask1)

    var = (tl.sum(sum0 * sum0, axis=0) + tl.sum(sum1 * sum1, axis=0)) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w0 = tl.load(W + cols0, mask=mask0, other=0.0).to(tl.float32)
    w1 = tl.load(W + cols1, mask=mask1, other=0.0).to(tl.float32)

    tl.store(y_row_ptr + cols0, (sum0 * rsqrt * w0).to(Y.dtype.element_ty), mask=mask0)
    tl.store(y_row_ptr + cols1, (sum1 * rsqrt * w1).to(Y.dtype.element_ty), mask=mask1)


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

    cols0 = tl.arange(0, BLOCK_SIZE)
    mask0 = cols0 < N
    dy0 = tl.load(dy_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    sum0 = tl.load(res_out_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
    w0 = tl.load(W + cols0, mask=mask0, other=0.0).to(tl.float32)
    dy_w0 = dy0 * w0

    cols1 = BLOCK_SIZE + cols0
    mask1 = cols1 < N
    dy1 = tl.load(dy_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    sum1 = tl.load(res_out_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
    w1 = tl.load(W + cols1, mask=mask1, other=0.0).to(tl.float32)
    dy_w1 = dy1 * w1

    c1 = (tl.sum(dy_w0 * sum0, axis=0) + tl.sum(dy_w1 * sum1, axis=0)) / N
    scale = rsqrt * rsqrt * c1

    d_sum0 = rsqrt * (dy_w0 - sum0 * scale)
    d_sum1 = rsqrt * (dy_w1 - sum1 * scale)

    if HAS_DRES:
        dres_row_ptr = dResOut + row_idx * stride_dres_out_row
        dres0 = tl.load(dres_row_ptr + cols0, mask=mask0, other=0.0).to(tl.float32)
        dres1 = tl.load(dres_row_ptr + cols1, mask=mask1, other=0.0).to(tl.float32)
        d_sum0 = d_sum0 + dres0
        d_sum1 = d_sum1 + dres1

    tl.store(dsum_row_ptr + cols0, d_sum0.to(dSumRes.dtype.element_ty), mask=mask0)
    tl.store(dsum_row_ptr + cols1, d_sum1.to(dSumRes.dtype.element_ty), mask=mask1)


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
