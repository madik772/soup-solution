import math
import torch
import triton
import triton.language as tl


@triton.jit
def _custom_causal_attn_fwd_kernel(
    Q, K, V, Out, LSE,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H_q, S, sm_scale,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b_idx = (pid_bh // H_q).to(tl.int64)
    hq_idx = (pid_bh % H_q).to(tl.int64)
    hkv_idx = hq_idx // GROUP_SIZE

    q_base = Q + b_idx * stride_qb + hq_idx * stride_qh
    k_base = K + b_idx * stride_kb + hkv_idx * stride_kh
    v_base = V + b_idx * stride_vb + hkv_idx * stride_vh
    o_base = Out + b_idx * stride_ob + hq_idx * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < S

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float16)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, S) if IS_CAUSAL else S
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float16)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & mask_n[None, :]
        else:
            valid = mask_n[None, :]
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float16)

        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    tl.store(LSE + pid_bh.to(tl.int64) * S + offs_m, lse, mask=mask_m)
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _custom_causal_attn_bwd_kernel(
    Q, K, V, Out, dOut, LSE, Delta,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    H_q, H_kv, S, sm_scale,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MODE: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)

    if MODE == 1:
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b_idx = (pid_bh // H_q).to(tl.int64)
        hq_idx = (pid_bh % H_q).to(tl.int64)
        hkv_idx = hq_idx // GROUP_SIZE
        bh_offset = pid_bh.to(tl.int64) * S

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        q_base = Q + b_idx * stride_qb + hq_idx * stride_qh
        o_base = Out + b_idx * stride_ob + hq_idx * stride_oh
        do_base = dOut + b_idx * stride_dob + hq_idx * stride_doh
        k_base = K + b_idx * stride_kb + hkv_idx * stride_kh
        v_base = V + b_idx * stride_vb + hkv_idx * stride_vh
        dq_base = dQ + b_idx * stride_dqb + hq_idx * stride_dqh

        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float16)
        out_tile = tl.load(
            o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok,
            mask=mask_m[:, None],
            other=0.0,
        )
        do_fp16 = do.to(tl.float16)
        lse = tl.load(LSE + bh_offset + offs_m, mask=mask_m, other=0.0)

        delta = tl.sum(out_tile * do.to(tl.float32), axis=1)
        tl.store(Delta + bh_offset + offs_m, delta, mask=mask_m)

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        hi = tl.minimum((pid_m + 1) * BLOCK_M, S) if IS_CAUSAL else S

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float16)
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float16)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            if IS_CAUSAL:
                valid = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid = mask_m[:, None] & mask_n[None, :]
            p = tl.where(valid, tl.exp(tl.where(valid, qk - lse[:, None], 0.0)), 0.0)

            dp = tl.dot(do_fp16, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq = tl.dot(ds.to(tl.float16), k, dq)

        dq = dq * sm_scale
        tl.store(
            dq_base + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk,
            dq.to(dQ.dtype.element_ty),
            mask=mask_m[:, None],
        )
    else:
        pid_n = tl.program_id(0)
        pid_bkv = tl.program_id(1)
        b_idx = (pid_bkv // H_kv).to(tl.int64)
        hkv_idx = (pid_bkv % H_kv).to(tl.int64)

        start_n = pid_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_base = K + b_idx * stride_kb + hkv_idx * stride_kh
        v_base = V + b_idx * stride_vb + hkv_idx * stride_vh
        dk_base = dK + b_idx * stride_dkb + hkv_idx * stride_dkh
        dv_base = dV + b_idx * stride_dvb + hkv_idx * stride_dvh

        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float16)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float16)

        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        lo = (start_n // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0
        for g in range(0, GROUP_SIZE):
            hq_idx = hkv_idx * GROUP_SIZE + g
            bh_offset = (b_idx * H_q + hq_idx) * S
            q_base = Q + b_idx * stride_qb + hq_idx * stride_qh
            do_base = dOut + b_idx * stride_dob + hq_idx * stride_doh

            for start_m in range(lo, S, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < S

                q = tl.load(
                    q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                    mask=mask_m[:, None],
                    other=0.0,
                ).to(tl.float16)
                do = tl.load(
                    do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok,
                    mask=mask_m[:, None],
                    other=0.0,
                ).to(tl.float16)
                lse = tl.load(LSE + bh_offset + offs_m, mask=mask_m, other=0.0)
                delta = tl.load(Delta + bh_offset + offs_m, mask=mask_m, other=0.0)

                qk = tl.dot(q, tl.trans(k)) * sm_scale
                if IS_CAUSAL:
                    valid = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
                else:
                    valid = mask_m[:, None] & mask_n[None, :]
                p = tl.where(valid, tl.exp(tl.where(valid, qk - lse[:, None], 0.0)), 0.0)

                dv = tl.dot(tl.trans(p.to(tl.float16)), do, dv)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dk = tl.dot(tl.trans(ds.to(tl.float16)), q, dk)

        dk = dk * sm_scale
        tl.store(
            dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk,
            dk.to(dK.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk,
            dv.to(dV.dtype.element_ty),
            mask=mask_n[:, None],
        )


def _prepare_4d(t: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != target_dtype:
        t = t.to(target_dtype)
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


class TritonCausalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True):
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        orig_v_dtype = v.dtype

        out_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else orig_q_dtype
        )

        q_fp16 = _prepare_4d(q, torch.float16)
        k_fp16 = _prepare_4d(k, torch.float16)
        v_fp16 = _prepare_4d(v, torch.float16)

        B, H_q, S, D = q_fp16.shape
        H_kv = k_fp16.shape[1]
        group_size = H_q // H_kv
        sm_scale = 1.0 / math.sqrt(D)
        causal_flag = bool(is_causal)

        out_fp16 = torch.empty((B, S, H_q, D), device=q.device, dtype=torch.float16).transpose(1, 2)
        lse = torch.empty((B, H_q, S), device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 32 if D > 64 else 64
        grid = (triton.cdiv(S, BLOCK_M), B * H_q)

        _custom_causal_attn_fwd_kernel[grid](
            q_fp16, k_fp16, v_fp16, out_fp16, lse,
            q_fp16.stride(0), q_fp16.stride(1), q_fp16.stride(2), q_fp16.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            out_fp16.stride(0), out_fp16.stride(1), out_fp16.stride(2), out_fp16.stride(3),
            H_q, S, sm_scale,
            GROUP_SIZE=group_size,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal_flag,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q_fp16, k_fp16, v_fp16, out_fp16, lse)
        ctx.sm_scale = sm_scale
        ctx.group_size = group_size
        ctx.is_causal = causal_flag
        ctx.orig_q_dtype = orig_q_dtype
        ctx.orig_k_dtype = orig_k_dtype
        ctx.orig_v_dtype = orig_v_dtype

        return out_fp16 if out_dtype == torch.float16 else out_fp16.to(out_dtype)

    @staticmethod
    def backward(ctx, dout):
        q_fp16, k_fp16, v_fp16, out_fp16, lse = ctx.saved_tensors
        dout_fp16 = _prepare_4d(dout, torch.float16)

        B, H_q, S, D = q_fp16.shape
        H_kv = k_fp16.shape[1]
        group_size = ctx.group_size
        sm_scale = ctx.sm_scale
        causal_flag = ctx.is_causal

        dq_fp16 = torch.empty(q_fp16.shape, device=q_fp16.device, dtype=torch.float16)
        dk_fp16 = torch.empty(k_fp16.shape, device=k_fp16.device, dtype=torch.float16)
        dv_fp16 = torch.empty(v_fp16.shape, device=v_fp16.device, dtype=torch.float16)
        delta = torch.empty((B, H_q, S), device=q_fp16.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32

        grid_dq = (triton.cdiv(S, BLOCK_M), B * H_q)
        _custom_causal_attn_bwd_kernel[grid_dq](
            q_fp16, k_fp16, v_fp16, out_fp16, dout_fp16, lse, delta,
            dq_fp16, dk_fp16, dv_fp16,
            q_fp16.stride(0), q_fp16.stride(1), q_fp16.stride(2), q_fp16.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            out_fp16.stride(0), out_fp16.stride(1), out_fp16.stride(2), out_fp16.stride(3),
            dout_fp16.stride(0), dout_fp16.stride(1), dout_fp16.stride(2), dout_fp16.stride(3),
            dq_fp16.stride(0), dq_fp16.stride(1), dq_fp16.stride(2), dq_fp16.stride(3),
            dk_fp16.stride(0), dk_fp16.stride(1), dk_fp16.stride(2), dk_fp16.stride(3),
            dv_fp16.stride(0), dv_fp16.stride(1), dv_fp16.stride(2), dv_fp16.stride(3),
            H_q, H_kv, S, sm_scale,
            GROUP_SIZE=group_size,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal_flag,
            MODE=1,
            num_warps=4,
            num_stages=1,
        )

        grid_dkv = (triton.cdiv(S, BLOCK_N), B * H_kv)
        _custom_causal_attn_bwd_kernel[grid_dkv](
            q_fp16, k_fp16, v_fp16, out_fp16, dout_fp16, lse, delta,
            dq_fp16, dk_fp16, dv_fp16,
            q_fp16.stride(0), q_fp16.stride(1), q_fp16.stride(2), q_fp16.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            out_fp16.stride(0), out_fp16.stride(1), out_fp16.stride(2), out_fp16.stride(3),
            dout_fp16.stride(0), dout_fp16.stride(1), dout_fp16.stride(2), dout_fp16.stride(3),
            dq_fp16.stride(0), dq_fp16.stride(1), dq_fp16.stride(2), dq_fp16.stride(3),
            dk_fp16.stride(0), dk_fp16.stride(1), dk_fp16.stride(2), dk_fp16.stride(3),
            dv_fp16.stride(0), dv_fp16.stride(1), dv_fp16.stride(2), dv_fp16.stride(3),
            H_q, H_kv, S, sm_scale,
            GROUP_SIZE=group_size,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal_flag,
            MODE=0,
            num_warps=4,
            num_stages=1,
        )

        dq = dq_fp16 if ctx.orig_q_dtype == torch.float16 else dq_fp16.to(ctx.orig_q_dtype)
        dk = dk_fp16 if ctx.orig_k_dtype == torch.float16 else dk_fp16.to(ctx.orig_k_dtype)
        dv = dv_fp16 if ctx.orig_v_dtype == torch.float16 else dv_fp16.to(ctx.orig_v_dtype)
        return dq, dk, dv, None


def causal_attention(q, k, v, is_causal=True):
    return TritonCausalAttentionFunction.apply(q, k, v, is_causal)
