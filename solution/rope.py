import torch
import triton
import triton.language as tl


@triton.jit
def _custom_fused_rope_fwd_kernel(
    X, Cos, Sin, Out,
    stride_x0, stride_x1, stride_x2, stride_x3,
    stride_c0, stride_c1, stride_c2, stride_c3,
    stride_s0, stride_s1, stride_s2, stride_s3,
    stride_o0, stride_o1, stride_o2, stride_o3,
    D1, D2, half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    i2 = row_idx % D2
    tmp = row_idx // D2
    i1 = tmp % D1
    i0 = tmp // D1

    x_row_ptr = X + i0 * stride_x0 + i1 * stride_x1 + i2 * stride_x2
    c_row_ptr = Cos + i0 * stride_c0 + i1 * stride_c1 + i2 * stride_c2
    s_row_ptr = Sin + i0 * stride_s0 + i1 * stride_s1 + i2 * stride_s2
    o_row_ptr = Out + i0 * stride_o0 + i1 * stride_o1 + i2 * stride_o2

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim

    x1 = tl.load(x_row_ptr + cols * stride_x3, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row_ptr + (cols + half_dim) * stride_x3, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(c_row_ptr + cols * stride_c3, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_row_ptr + cols * stride_s3, mask=mask, other=0.0).to(tl.float32)

    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s

    tl.store(o_row_ptr + cols * stride_o3, out1.to(Out.dtype.element_ty), mask=mask)
    tl.store(o_row_ptr + (cols + half_dim) * stride_o3, out2.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _custom_fused_rope_bwd_kernel(
    dOut, Cos, Sin, dX,
    stride_do0, stride_do1, stride_do2, stride_do3,
    stride_c0, stride_c1, stride_c2, stride_c3,
    stride_s0, stride_s1, stride_s2, stride_s3,
    stride_dx0, stride_dx1, stride_dx2, stride_dx3,
    D1, D2, half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    i2 = row_idx % D2
    tmp = row_idx // D2
    i1 = tmp % D1
    i0 = tmp // D1

    do_row_ptr = dOut + i0 * stride_do0 + i1 * stride_do1 + i2 * stride_do2
    c_row_ptr = Cos + i0 * stride_c0 + i1 * stride_c1 + i2 * stride_c2
    s_row_ptr = Sin + i0 * stride_s0 + i1 * stride_s1 + i2 * stride_s2
    dx_row_ptr = dX + i0 * stride_dx0 + i1 * stride_dx1 + i2 * stride_dx2

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim

    dout1 = tl.load(do_row_ptr + cols * stride_do3, mask=mask, other=0.0).to(tl.float32)
    dout2 = tl.load(do_row_ptr + (cols + half_dim) * stride_do3, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(c_row_ptr + cols * stride_c3, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_row_ptr + cols * stride_s3, mask=mask, other=0.0).to(tl.float32)

    dx1 = dout1 * c + dout2 * s
    dx2 = dout2 * c - dout1 * s

    tl.store(dx_row_ptr + cols * stride_dx3, dx1.to(dX.dtype.element_ty), mask=mask)
    tl.store(dx_row_ptr + (cols + half_dim) * stride_dx3, dx2.to(dX.dtype.element_ty), mask=mask)


def _prepare_cos_sin_4d(x_4d: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    D0, D1, D2, head_dim = x_4d.shape
    half_dim = head_dim // 2

    if cos.shape[-1] == head_dim:
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    if cos.ndim == 2:
        if D1 == cos.shape[0]:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        if D1 == cos.shape[1]:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
    elif cos.ndim == 4:
        if cos.shape[1] == 1 and cos.shape[2] == D1 and D2 != D1:
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)
        elif cos.shape[2] == 1 and cos.shape[1] == D2 and D1 != D2:
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)

    cos_4d = cos.expand(D0, D1, D2, half_dim)
    sin_4d = sin.expand(D0, D1, D2, half_dim)
    return cos_4d, sin_4d


class TritonRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        orig_shape = x.shape
        if x.ndim == 3:
            x_4d = x.unsqueeze(1)
        elif x.ndim == 4:
            x_4d = x
        else:
            x_4d = x.reshape(-1, 1, orig_shape[-2], orig_shape[-1])

        D0, D1, D2, head_dim = x_4d.shape
        half_dim = head_dim // 2
        n_rows = D0 * D1 * D2

        cos_4d, sin_4d = _prepare_cos_sin_4d(x_4d, cos, sin)
        out_4d = torch.empty(x_4d.shape, device=x.device, dtype=x.dtype)

        BLOCK_SIZE = 64 if half_dim == 64 else triton.next_power_of_2(half_dim)

        _custom_fused_rope_fwd_kernel[(n_rows,)](
            x_4d, cos_4d, sin_4d, out_4d,
            x_4d.stride(0), x_4d.stride(1), x_4d.stride(2), x_4d.stride(3),
            cos_4d.stride(0), cos_4d.stride(1), cos_4d.stride(2), cos_4d.stride(3),
            sin_4d.stride(0), sin_4d.stride(1), sin_4d.stride(2), sin_4d.stride(3),
            out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
            D1, D2, half_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(cos_4d, sin_4d)
        ctx.orig_shape = orig_shape
        ctx.x_4d_shape = x_4d.shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return out_4d if x.ndim == 4 else out_4d.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, dout):
        cos_4d, sin_4d = ctx.saved_tensors
        D0, D1, D2, head_dim = ctx.x_4d_shape
        half_dim = head_dim // 2
        n_rows = D0 * D1 * D2

        dout_4d = dout if dout.ndim == 4 else dout.reshape(D0, D1, D2, head_dim)
        base_do = dout._base if dout._base is not None else dout
        inplace_bwd = (not base_do.is_leaf) and dout_4d.stride(-1) == 1
        dx_4d = dout_4d if inplace_bwd else torch.empty(ctx.x_4d_shape, device=dout.device, dtype=dout.dtype)

        _custom_fused_rope_bwd_kernel[(n_rows,)](
            dout_4d, cos_4d, sin_4d, dx_4d,
            dout_4d.stride(0), dout_4d.stride(1), dout_4d.stride(2), dout_4d.stride(3),
            cos_4d.stride(0), cos_4d.stride(1), cos_4d.stride(2), cos_4d.stride(3),
            sin_4d.stride(0), sin_4d.stride(1), sin_4d.stride(2), sin_4d.stride(3),
            dx_4d.stride(0), dx_4d.stride(1), dx_4d.stride(2), dx_4d.stride(3),
            D1, D2, half_dim,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return (dx_4d if len(ctx.orig_shape) == 4 else dx_4d.reshape(*ctx.orig_shape)), None, None


def rope(x, cos, sin):
    return TritonRoPEFunction.apply(x, cos, sin)
