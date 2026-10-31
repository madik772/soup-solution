"""Compile the packed step's Triton kernels while Soup loads the data and the model.

Every measured run starts with empty compiler caches, and the first launch of each kernel
variant would compile it on the critical path of training step 1. `start()` (called by
`apply()` when a CUDA context already exists) compiles the variants the Qwen2.5-1.5B step
launches, with the dtypes, constexprs and integer specializations that step uses, in a
background thread; `wait()` joins it before the first packed step. Nothing is launched: this
is `JITFunction.warmup`, i.e. compilation only. Any variant it misses (another model size, a
Triton without `warmup`) is simply compiled on first use, as without this file.
"""

import threading

_THREAD = None


def _jobs(torch):
    from solution import cross_entropy, nf4, rmsnorm, rope, swiglu, unpad

    dev = torch.device("cuda", torch.cuda.current_device())
    f16, f32, i32, u8 = torch.float16, torch.float32, torch.int32, torch.uint8

    def t(dtype):
        return torch.empty(64, device=dev, dtype=dtype)

    hidden, inter, heads, kv, head_dim, vocab = 1536, 8960, 12, 2, 128, 151936
    qkv = (heads + 2 * kv) * head_dim
    seq = 512
    blk = rmsnorm._block_for(hidden)
    return [
        (nf4._custom_nf4_dequant_kernel,
         (t(u8), t(u8), t(f32), t(f32), t(f32), t(f32), t(f16), 4096),
         dict(QBLOCK_HALF=32, QBLOCK2=256, NESTED=True, BLOCK_BYTES=nf4.BLOCK_BYTES,
              BLOCK_OUT=2 * nf4.BLOCK_BYTES,
              num_warps=4, num_stages=2)),
        (rmsnorm._custom_rmsnorm_drop_fwd_kernel,
         (t(f32), t(f32), t(f16), t(f32), t(f16), t(f16), hidden, hidden,
          hidden, 1e-6, 0.05, 1.0 / 0.95, 12345, 67890),
         dict(NUM_DROP=2, BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (rmsnorm._custom_rmsnorm_drop_bwd_kernel,
         (t(f16), t(f16), t(f16), t(f32), t(f32), t(f32), t(f32), t(f32),
          hidden, hidden, hidden, hidden, hidden, hidden, 0.05, 1.0 / 0.95, 12345, 67890),
         dict(NUM_DROP=2, HAS_DRES=True, BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (rmsnorm._custom_fused_add_rmsnorm_fwd_kernel,
         (t(f32), t(f16), t(f32), t(f32), t(f16), t(f32), hidden, hidden, hidden, hidden,
          hidden, 1e-6),
         dict(BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (rmsnorm._custom_fused_add_rmsnorm_bwd_kernel,
         (t(f16), t(f32), t(f32), t(f32), t(f32), t(f32), hidden, hidden, hidden, hidden,
          hidden),
         dict(HAS_DRES=True, BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (rmsnorm._custom_rmsnorm_fwd_kernel,
         (t(f32), t(f16), t(f32), t(f32), hidden, hidden, hidden, 1e-6),
         dict(BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (rmsnorm._custom_rmsnorm_bwd_kernel,
         (t(f16), t(f32), t(f32), t(f32), t(f32), hidden, hidden, hidden, hidden),
         dict(BLOCK_SIZE=blk, num_warps=4, num_stages=2)),
        (swiglu._custom_fused_swiglu_fwd_kernel,
         (t(f16), t(f16), t(f16), 2 * inter, 1, 2 * inter, 1, inter, 1, inter),
         dict(BLOCK_SIZE=swiglu.BLOCK_SIZE, num_warps=8, num_stages=2)),
        (swiglu._custom_fused_swiglu_bwd_kernel,
         (t(f16), t(f16), t(f16), t(f16), t(f16), inter, 1, 2 * inter, 1, 2 * inter, 1,
          2 * inter, 1, 2 * inter, 1, inter),
         dict(BLOCK_SIZE=swiglu.BLOCK_SIZE, num_warps=8, num_stages=2)),
        (rope._custom_qkv_rope_pad_fwd_kernel,
         (t(f16), t(i32), t(f32), t(f32), t(f16), t(f16), t(f16), qkv, head_dim, seq),
         dict(N_HEADS=heads, N_KV=kv, HEAD_DIM=head_dim, HALF=head_dim // 2, GROUP=heads // kv,
              BLOCK_H=16, num_warps=4, num_stages=2)),
        (rope._custom_qkv_rope_pad_bwd_kernel,
         (t(f16), t(f16), t(f16), t(i32), t(f32), t(f32), t(f16),
          seq * heads * head_dim, head_dim, heads * head_dim,
          seq * heads * head_dim, head_dim, heads * head_dim,
          seq * heads * head_dim, head_dim, heads * head_dim,
          qkv, head_dim, seq),
         dict(N_HEADS=heads, N_KV=kv, HEAD_DIM=head_dim, HALF=head_dim // 2, GROUP=heads // kv,
              BLOCK_H=16, BLOCK_G=8,
              num_warps=4, num_stages=2)),
        (unpad._custom_row_gather_kernel,
         (t(f16), t(i32), t(f16), t(i32), hidden, hidden, hidden),
         dict(HAS_SCALE=False, BLOCK=unpad.GATHER_BLOCK, num_warps=4, num_stages=2)),
        (unpad._custom_row_gather_kernel,
         (t(f16), t(i32), t(f32), t(i32), hidden, hidden, hidden),
         dict(HAS_SCALE=False, BLOCK=unpad.GATHER_BLOCK, num_warps=4, num_stages=2)),
        (unpad._custom_row_gather_kernel,
         (t(f16), t(i32), t(f16), t(f32), hidden, hidden, hidden),
         dict(HAS_SCALE=True, BLOCK=unpad.GATHER_BLOCK, num_warps=4, num_stages=2)),
        (cross_entropy._custom_lmhead_ce_kernel,
         (t(f16), t(i32), t(f32), t(f32), t(i32), vocab, 100, vocab, cross_entropy.GRAD_SCALE),
         dict(BLOCK_V=cross_entropy.CE_BLOCK_V, num_warps=8, num_stages=2)),
    ]


def _run():
    try:
        import torch

        for fn, args, kwargs in _jobs(torch):
            try:
                fn.warmup(*args, grid=(1,), **kwargs)
            except Exception:  # noqa: BLE001 - a miss only means compiling on first use
                pass
    except Exception:  # noqa: BLE001
        pass


def start():
    global _THREAD
    try:
        import torch

        if _THREAD is not None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
    except Exception:  # noqa: BLE001
        return
    _THREAD = threading.Thread(target=_run, name="solution-triton-warmup", daemon=True)
    _THREAD.start()


def wait():
    if _THREAD is not None:
        _THREAD.join()
