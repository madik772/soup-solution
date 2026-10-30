import torch
import triton
import triton.language as tl

BLOCK_SIZE = 2048


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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_row_ptr + cols, (x * rsqrt * w).to(Y.dtype.element_ty), mask=mask)


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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dy_w = dy * w

    c1 = tl.sum(dy_w * x, axis=0) / N
    scale = rsqrt * rsqrt * c1
    dx = rsqrt * (dy_w - x * scale)

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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    res = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sum_res = res + x
    tl.store(res_out_row_ptr + cols, sum_res.to(ResOut.dtype.element_ty), mask=mask)

    var = tl.sum(sum_res * sum_res, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_row_ptr + cols, (sum_res * rsqrt * w).to(Y.dtype.element_ty), mask=mask)


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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sum_res = tl.load(res_out_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dy_w = dy * w

    c1 = tl.sum(dy_w * sum_res, axis=0) / N
    scale = rsqrt * rsqrt * c1
    d_sum = rsqrt * (dy_w - sum_res * scale)

    if HAS_DRES:
        dres_row_ptr = dResOut + row_idx * stride_dres_out_row
        dres = tl.load(dres_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        d_sum = d_sum + dres

    tl.store(dsum_row_ptr + cols, d_sum.to(dSumRes.dtype.element_ty), mask=mask)


def _as_contiguous_2d(t: torch.Tensor, last_dim: int) -> torch.Tensor:
    if not t.is_contiguous():
        t = t.contiguous()
    return t if t.ndim == 2 else t.view(-1, last_dim)


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, out_dtype=None):
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = _as_contiguous_2d(x, N)
        M = x_2d.shape[0]
        y_dtype = out_dtype if out_dtype is not None else x_2d.dtype
        y = torch.empty((M, N), device=x.device, dtype=y_dtype)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)
        blk = BLOCK_SIZE if N <= BLOCK_SIZE else triton.next_power_of_2(N)

        _custom_rmsnorm_fwd_kernel[(M,)](
            x_2d, y, weight, rsqrt,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=blk,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(x_2d, weight, rsqrt)
        ctx.N = N
        ctx.blk = blk
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
            BLOCK_SIZE=ctx.blk,
            num_warps=4,
            num_stages=2,
        )
        if weight is None or not weight.requires_grad:
            return dx.view(*orig_shape), None, None, None

        dw = (dy_2d.float() * (x_2d.float() * rsqrt.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return dx.view(*orig_shape), dw, None, None


class TritonFusedAddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, x, weight, eps=1e-6, out_dtype=None):
        ctx.set_materialize_grads(False)
        orig_shape = residual.shape
        N = orig_shape[-1]
        res_2d = _as_contiguous_2d(residual, N)
        x_2d = _as_contiguous_2d(x, N)
        M = res_2d.shape[0]

        base_x = x._base if x._base is not None else x
        inplace_fwd = (not base_x.is_leaf) and (x_2d.dtype == res_2d.dtype)
        res_out_2d = x_2d if inplace_fwd else torch.empty_like(res_2d)

        y_dtype = out_dtype if out_dtype is not None else res_2d.dtype
        y_2d = torch.empty((M, N), device=residual.device, dtype=y_dtype)
        rsqrt = torch.empty(M, device=residual.device, dtype=torch.float32)
        blk = BLOCK_SIZE if N <= BLOCK_SIZE else triton.next_power_of_2(N)

        _custom_fused_add_rmsnorm_fwd_kernel[(M,)](
            res_2d, x_2d, weight, res_out_2d, y_2d, rsqrt,
            res_2d.stride(0), x_2d.stride(0), res_out_2d.stride(0), y_2d.stride(0),
            N, eps,
            BLOCK_SIZE=blk,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(res_out_2d, weight, rsqrt)
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.blk = blk
        ctx.inplace_bwd = weight is None or not weight.requires_grad
        ctx.x_dtype = x.dtype
        return y_2d.view(*orig_shape), res_out_2d.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy, d_res_out):
        res_out_2d, weight, rsqrt = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        M, N = res_out_2d.shape

        if dy is None:
            dy_2d = torch.zeros_like(res_out_2d)
        else:
            dy_2d = _as_contiguous_2d(dy, N)

        has_dres = d_res_out is not None
        if has_dres:
            dres_2d = _as_contiguous_2d(d_res_out, N)
            dres_stride = dres_2d.stride(0)
        else:
            dres_2d = res_out_2d
            dres_stride = 0

        d_sum_2d = res_out_2d if ctx.inplace_bwd else torch.empty_like(res_out_2d)

        _custom_fused_add_rmsnorm_bwd_kernel[(M,)](
            dy_2d, dres_2d, res_out_2d, weight, rsqrt, d_sum_2d,
            dy_2d.stride(0), dres_stride, res_out_2d.stride(0), d_sum_2d.stride(0),
            N,
            HAS_DRES=has_dres,
            BLOCK_SIZE=ctx.blk,
            num_warps=4,
            num_stages=2,
        )

        d_sum = d_sum_2d.view(*orig_shape)
        dx = d_sum if d_sum.dtype == ctx.x_dtype else d_sum.to(ctx.x_dtype)
        if weight is None or not weight.requires_grad:
            return d_sum, dx, None, None, None

        dw = (dy_2d.float() * (res_out_2d.float() * rsqrt.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return d_sum, dx, dw, None, None


def fused_add_rmsnorm(residual, x, weight, eps=1e-6, out_dtype=None):
    return TritonFusedAddRMSNormFunction.apply(residual, x, weight, eps, out_dtype)
