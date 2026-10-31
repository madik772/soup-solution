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


# ---------------------------------------------------------------------------------------------
# Packed (padding-free) attention input: one kernel takes the fused q|k|v projection of the real
# tokens, rotates q and k, repeats the key/value heads for grouped-query attention and writes the
# padded [B, S, H, D] tensors SDPA reads (zeros at padding). The backward kernel undoes all of
# it in one pass per real token: inverse rotation, sum over the repeated heads, back to the
# packed q|k|v layout. Rotation in fp32 as transformers does it (cos/sin are fp32 there).
# ---------------------------------------------------------------------------------------------


@triton.jit(do_not_specialize=["seq_len"])
def _custom_qkv_rope_pad_fwd_kernel(
    QKV, Map, Cos, Sin, Qp, Kp, Vp,
    stride_qkv, stride_cos, seq_len,
    N_HEADS: tl.constexpr,
    N_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(Map + row).to(tl.int64)
    valid = src >= 0
    pos = row % seq_len

    h = tl.arange(0, BLOCK_H)[:, None]
    d = tl.arange(0, HALF)[None, :]
    hmask = h < N_HEADS
    lmask = hmask & valid

    c1 = tl.load(Cos + pos * stride_cos + d).to(tl.float32)
    c2 = tl.load(Cos + pos * stride_cos + HALF + d).to(tl.float32)
    s1 = tl.load(Sin + pos * stride_cos + d).to(tl.float32)
    s2 = tl.load(Sin + pos * stride_cos + HALF + d).to(tl.float32)

    src_ptr = QKV + tl.maximum(src, 0) * stride_qkv
    out_off = row * (N_HEADS * HEAD_DIM) + h * HEAD_DIM + d

    q1 = tl.load(src_ptr + h * HEAD_DIM + d, mask=lmask, other=0.0).to(tl.float32)
    q2 = tl.load(src_ptr + h * HEAD_DIM + HALF + d, mask=lmask, other=0.0).to(tl.float32)
    tl.store(Qp + out_off, (q1 * c1 - q2 * s1).to(Qp.dtype.element_ty), mask=hmask)
    tl.store(Qp + out_off + HALF, (q2 * c2 + q1 * s2).to(Qp.dtype.element_ty), mask=hmask)

    kv = h // GROUP
    k_ptr = src_ptr + N_HEADS * HEAD_DIM + kv * HEAD_DIM
    k1 = tl.load(k_ptr + d, mask=lmask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + HALF + d, mask=lmask, other=0.0).to(tl.float32)
    tl.store(Kp + out_off, (k1 * c1 - k2 * s1).to(Kp.dtype.element_ty), mask=hmask)
    tl.store(Kp + out_off + HALF, (k2 * c2 + k1 * s2).to(Kp.dtype.element_ty), mask=hmask)

    v_ptr = src_ptr + (N_HEADS + N_KV) * HEAD_DIM + kv * HEAD_DIM
    v1 = tl.load(v_ptr + d, mask=lmask, other=0.0)
    v2 = tl.load(v_ptr + HALF + d, mask=lmask, other=0.0)
    tl.store(Vp + out_off, v1.to(Vp.dtype.element_ty), mask=hmask)
    tl.store(Vp + out_off + HALF, v2.to(Vp.dtype.element_ty), mask=hmask)


@triton.jit(do_not_specialize=["seq_len"])
def _custom_qkv_rope_pad_bwd_kernel(
    dQ, dK, dV, Idx, Cos, Sin, dQKV,
    sq_b, sq_h, sq_s,
    sk_b, sk_h, sk_s,
    sv_b, sv_h, sv_s,
    stride_dqkv, stride_cos, seq_len,
    N_HEADS: tl.constexpr,
    N_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    flat = tl.load(Idx + row).to(tl.int64)
    b = flat // seq_len
    pos = flat % seq_len

    d = tl.arange(0, HALF)[None, :]
    c1 = tl.load(Cos + pos * stride_cos + d).to(tl.float32)
    c2 = tl.load(Cos + pos * stride_cos + HALF + d).to(tl.float32)
    s1 = tl.load(Sin + pos * stride_cos + d).to(tl.float32)
    s2 = tl.load(Sin + pos * stride_cos + HALF + d).to(tl.float32)
    out_ptr = dQKV + row * stride_dqkv

    h = tl.arange(0, BLOCK_H)[:, None]
    hmask = h < N_HEADS
    q_ptr = dQ + b * sq_b + pos * sq_s + h * sq_h
    g1 = tl.load(q_ptr + d, mask=hmask, other=0.0).to(tl.float32)
    g2 = tl.load(q_ptr + HALF + d, mask=hmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + h * HEAD_DIM + d, (g1 * c1 + g2 * s2).to(dQKV.dtype.element_ty), mask=hmask)
    tl.store(out_ptr + h * HEAD_DIM + HALF + d, (g2 * c2 - g1 * s1).to(dQKV.dtype.element_ty),
             mask=hmask)

    g = tl.arange(0, BLOCK_G)[:, None]
    gmask = g < GROUP
    d1 = tl.arange(0, HALF)
    for kv in tl.static_range(N_KV):
        heads = kv * GROUP + g
        k_ptr = dK + b * sk_b + pos * sk_s + heads * sk_h
        k1 = tl.load(k_ptr + d, mask=gmask, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + HALF + d, mask=gmask, other=0.0).to(tl.float32)
        dk1 = tl.sum(k1 * c1 + k2 * s2, axis=0)
        dk2 = tl.sum(k2 * c2 - k1 * s1, axis=0)
        k_out = out_ptr + N_HEADS * HEAD_DIM + kv * HEAD_DIM
        tl.store(k_out + d1, dk1.to(dQKV.dtype.element_ty))
        tl.store(k_out + HALF + d1, dk2.to(dQKV.dtype.element_ty))

        v_ptr = dV + b * sv_b + pos * sv_s + heads * sv_h
        v1 = tl.load(v_ptr + d, mask=gmask, other=0.0).to(tl.float32)
        v2 = tl.load(v_ptr + HALF + d, mask=gmask, other=0.0).to(tl.float32)
        v_out = out_ptr + (N_HEADS + N_KV) * HEAD_DIM + kv * HEAD_DIM
        tl.store(v_out + d1, tl.sum(v1, axis=0).to(dQKV.dtype.element_ty))
        tl.store(v_out + HALF + d1, tl.sum(v2, axis=0).to(dQKV.dtype.element_ty))


def qkv_rope_pad(qkv, row_map, cos, sin, batch, seq_len, n_heads, n_kv, head_dim):
    """Packed fused projection [N, (H + 2*KV) * D] -> padded, rotated, head-repeated q, k, v as
    [B, H, S, D] views of [B, S, H, D] buffers, ready for SDPA."""
    n_pos = batch * seq_len
    shape = (n_pos, n_heads * head_dim)
    qp = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    kp = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    vp = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    _custom_qkv_rope_pad_fwd_kernel[(n_pos,)](
        qkv, row_map, cos, sin, qp, kp, vp,
        qkv.stride(0), cos.stride(0), seq_len,
        N_HEADS=n_heads, N_KV=n_kv, HEAD_DIM=head_dim,
        HALF=head_dim // 2, GROUP=n_heads // n_kv,
        BLOCK_H=triton.next_power_of_2(n_heads),
        num_warps=4, num_stages=2,
    )

    def as_bhsd(t):
        return t.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

    return as_bhsd(qp), as_bhsd(kp), as_bhsd(vp)


def qkv_rope_pad_backward(dq, dk, dv, real_idx, cos, sin, seq_len, n_heads, n_kv, head_dim):
    """Gradients of `qkv_rope_pad` w.r.t. its packed input, [N, (H + 2*KV) * D]."""
    n_real = real_idx.numel()
    out = torch.empty((n_real, (n_heads + 2 * n_kv) * head_dim), device=dq.device, dtype=dq.dtype)
    for t in (dq, dk, dv):
        assert t.stride(3) == 1
    _custom_qkv_rope_pad_bwd_kernel[(n_real,)](
        dq, dk, dv, real_idx, cos, sin, out,
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        out.stride(0), cos.stride(0), seq_len,
        N_HEADS=n_heads, N_KV=n_kv, HEAD_DIM=head_dim,
        HALF=head_dim // 2, GROUP=n_heads // n_kv,
        BLOCK_H=triton.next_power_of_2(n_heads),
        BLOCK_G=triton.next_power_of_2(n_heads // n_kv),
        num_warps=4, num_stages=2,
    )
    return out
