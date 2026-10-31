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
        ctx.inplace_bwd = False  # res_out is returned to the caller: never overwrite it
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


# ---------------------------------------------------------------------------------------------
# input_layernorm of the packed decoder layer, with the LoRA dropout of the projections that
# read it (q_proj and v_proj in this task) fused in. peft applies `dropout(x)` to the norm's
# output before lora_A; here the kernel writes x and the dropped copies in one pass from the
# fp32 value, and the masks are never stored: they are a function of (seed, element index), so
# the backward kernel regenerates them while it folds the LoRA branch gradients into the norm's
# input gradient.
# ---------------------------------------------------------------------------------------------


@triton.jit
def _custom_dropout_keep(counter, seed, p_drop):
    """Keep mask, P(keep) = 1 - p_drop, from a counter hash: the murmur3 32-bit finalizer of
    counter + seed * 2^32/phi, top 24 bits as a uniform in [0, 1). Same (seed, counter), same
    mask; far cheaper to compile and run than Philox, and plenty for dropout."""
    x = counter.to(tl.uint32) + seed.to(tl.uint32) * 0x9E3779B9
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return (x >> 8).to(tl.float32) * (1.0 / 16777216.0) >= p_drop


@triton.jit(do_not_specialize=["seed1", "seed2"])
def _custom_rmsnorm_drop_fwd_kernel(
    X, W, Y, Rsqrt, Yd1, Yd2,
    stride_x_row, stride_y_row,
    N, eps, p_drop, drop_scale, seed1, seed2,
    NUM_DROP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt + row_idx, rsqrt)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rsqrt * w
    y_off = row_idx * stride_y_row + cols
    tl.store(Y + y_off, y.to(Y.dtype.element_ty), mask=mask)

    if NUM_DROP >= 1:
        rand_off = row_idx * N + cols
        keep = _custom_dropout_keep(rand_off, seed1, p_drop)
        tl.store(Yd1 + y_off, tl.where(keep, y * drop_scale, 0.0).to(Yd1.dtype.element_ty),
                 mask=mask)
        if NUM_DROP >= 2:
            keep2 = _custom_dropout_keep(rand_off, seed2, p_drop)
            tl.store(Yd2 + y_off, tl.where(keep2, y * drop_scale, 0.0).to(Yd2.dtype.element_ty),
                     mask=mask)


@triton.jit(do_not_specialize=["seed1", "seed2"])
def _custom_rmsnorm_drop_bwd_kernel(
    dY, T1, T2, X, W, Rsqrt, dRes, dX,
    stride_dy_row, stride_t_row, stride_x_row, stride_dres_row, stride_dx_row,
    N, p_drop, drop_scale, seed1, seed2,
    NUM_DROP: tl.constexpr,
    HAS_DRES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dY + row_idx * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_DROP >= 1:
        rand_off = row_idx * N + cols
        t1 = tl.load(T1 + row_idx * stride_t_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy += tl.where(_custom_dropout_keep(rand_off, seed1, p_drop), t1 * drop_scale, 0.0)
        if NUM_DROP >= 2:
            t2 = tl.load(T2 + row_idx * stride_t_row + cols, mask=mask, other=0.0).to(tl.float32)
            dy += tl.where(_custom_dropout_keep(rand_off, seed2, p_drop), t2 * drop_scale, 0.0)

    x = tl.load(X + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(Rsqrt + row_idx)
    dy_w = dy * w
    c1 = tl.sum(dy_w * x, axis=0) / N
    dx = rsqrt * (dy_w - x * (rsqrt * rsqrt * c1))
    if HAS_DRES:
        dx += tl.load(dRes + row_idx * stride_dres_row + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(dX + row_idx * stride_dx_row + cols, dx.to(dX.dtype.element_ty), mask=mask)


def _block_for(n):
    return BLOCK_SIZE if n <= BLOCK_SIZE else triton.next_power_of_2(n)


def rmsnorm_drop_fwd(x, weight, eps, out_dtype, p_drop=0.0, seeds=()):
    """y = RMSNorm(x) in `out_dtype`, plus one dropped copy of y per seed (at most two)."""
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)
    drops = [torch.empty_like(y) for _ in seeds]
    yd1 = drops[0] if len(drops) > 0 else y
    yd2 = drops[1] if len(drops) > 1 else y
    s1 = seeds[0] if len(seeds) > 0 else 0
    s2 = seeds[1] if len(seeds) > 1 else 0
    _custom_rmsnorm_drop_fwd_kernel[(M,)](
        x, weight, y, rsqrt, yd1, yd2,
        x.stride(0), y.stride(0),
        N, eps, p_drop, 1.0 / (1.0 - p_drop), s1, s2,
        NUM_DROP=len(seeds),
        BLOCK_SIZE=_block_for(N),
        num_warps=4,
        num_stages=2,
    )
    return y, rsqrt, drops


def rmsnorm_drop_bwd(dy, x, weight, rsqrt, d_res=None, p_drop=0.0, seeds=(), branch_grads=(),
                     out=None):
    """dx of RMSNorm for dy + sum(dropout_k(branch_grads[k])), plus d_res; fp32 by default."""
    M, N = x.shape
    if out is None:
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    t1 = branch_grads[0] if len(branch_grads) > 0 else dy
    t2 = branch_grads[1] if len(branch_grads) > 1 else dy
    s1 = seeds[0] if len(seeds) > 0 else 0
    s2 = seeds[1] if len(seeds) > 1 else 0
    dres = d_res if d_res is not None else dy
    _custom_rmsnorm_drop_bwd_kernel[(M,)](
        dy, t1, t2, x, weight, rsqrt, dres, out,
        dy.stride(0), t1.stride(0), x.stride(0), dres.stride(0), out.stride(0),
        N, p_drop, 1.0 / (1.0 - p_drop), s1, s2,
        NUM_DROP=len(branch_grads),
        HAS_DRES=d_res is not None,
        BLOCK_SIZE=_block_for(N),
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_add_rmsnorm_fwd(residual, x, weight, eps, out_dtype):
    """(residual + x, RMSNorm(residual + x), rsqrt) with the sum kept in residual's dtype."""
    M, N = residual.shape
    res_out = torch.empty_like(residual)
    y = torch.empty((M, N), device=residual.device, dtype=out_dtype)
    rsqrt = torch.empty(M, device=residual.device, dtype=torch.float32)
    _custom_fused_add_rmsnorm_fwd_kernel[(M,)](
        residual, x, weight, res_out, y, rsqrt,
        residual.stride(0), x.stride(0), res_out.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=_block_for(N),
        num_warps=4,
        num_stages=2,
    )
    return res_out, y, rsqrt


def fused_add_rmsnorm_bwd_(dy, d_res_out, res_out, weight, rsqrt):
    """Gradient w.r.t. the sum (= both addends), written over `res_out` (no longer needed)."""
    M, N = res_out.shape
    _custom_fused_add_rmsnorm_bwd_kernel[(M,)](
        dy, d_res_out, res_out, weight, rsqrt, res_out,
        dy.stride(0), d_res_out.stride(0), res_out.stride(0), res_out.stride(0),
        N,
        HAS_DRES=True,
        BLOCK_SIZE=_block_for(N),
        num_warps=4,
        num_stages=2,
    )
    return res_out
