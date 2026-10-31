"""Compile the packed step's Triton kernels while Soup loads the data and the model.

Every measured run starts with empty compiler caches, and the first launch of each kernel
variant would compile it on the critical path of training step 1. `start()` (called by
`apply()` when a CUDA context already exists) compiles the variants the Qwen2.5-1.5B step
launches, with the dtypes, constexprs and integer specializations that step uses, in
background threads; `wait()` joins them before the first packed step. Nothing is launched:
this is `JITFunction.warmup` plus loading the module and building its launcher, i.e.
compilation only. Any variant it misses (another model size, a Triton without `warmup`) is
simply compiled on first use, as without this file.
"""

import threading

_THREADS = []


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
         dict(QBLOCK=64, QBLOCK_HALF=32, QBLOCK2=256, NESTED=True,
              ROWS=nf4.QBLOCKS_PER_PROGRAM, num_warps=4, num_stages=2)),
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
         (t(f16), t(i32), t(i32), t(f32), t(f32), t(f16), t(f16), t(f16), qkv, head_dim, 1),
         dict(N_HEADS=heads, N_KV=kv, HEAD_DIM=head_dim, HALF=head_dim // 2, GROUP=heads // kv,
              BLOCK_H=16, PACKED=True, num_warps=4, num_stages=2)),
        (rope._custom_qkv_rope_pad_bwd_kernel,
         (t(f16), t(f16), t(f16), t(i32), t(f32), t(f32), t(f16),
          0, head_dim, heads * head_dim,
          0, head_dim, heads * head_dim,
          0, head_dim, heads * head_dim,
          qkv, head_dim, 1),
         dict(N_HEADS=heads, N_KV=kv, HEAD_DIM=head_dim, HALF=head_dim // 2, GROUP=heads // kv,
              BLOCK_H=16, BLOCK_G=8, PACKED=True,
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


N_THREADS = 3
_T0 = None
_TIMES = []


def _run(jobs):
    import time

    for fn, args, kwargs in jobs:
        t = time.perf_counter()
        try:
            kernel = fn.warmup(*args, grid=(1,), **kwargs)
            # also load the module and build the launcher stub (a C compile), which Triton
            # otherwise does lazily at the first real launch
            init = getattr(kernel, "_init_handles", None)
            if callable(init):
                init()
            _TIMES.append((fn.__name__, time.perf_counter() - t, time.perf_counter() - _T0))
        except Exception as exc:  # noqa: BLE001 - a miss only means compiling on first use
            _TIMES.append((f"{fn.__name__} FAILED {type(exc).__name__}: {exc}"[:200], 0.0, 0.0))


def start():
    global _THREADS, _T0
    try:
        import time

        import torch

        if _THREADS or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        _T0 = time.perf_counter()
        jobs = _jobs(torch)
    except Exception:  # noqa: BLE001
        return
    _THREADS = [threading.Thread(target=_run, args=(jobs[i::N_THREADS],),
                                 name=f"solution-triton-warmup-{i}", daemon=True)
                for i in range(N_THREADS)]
    for th in _THREADS:
        th.start()


def wait():
    """Join the compile threads (once), and say in the log how long that took."""
    global _THREADS
    if not _THREADS:
        return
    import sys
    import time

    t = time.perf_counter()
    for th in _THREADS:
        th.join()
    _THREADS = []
    done = max((x[2] for x in _TIMES), default=0.0)
    print(f"[solution] Triton warm-up: {len(_TIMES)} kernels compiled in the background, "
          f"done {done:.1f} s after apply(); the first step waited {time.perf_counter() - t:.2f} s; "
          + ", ".join(f"{n} {c:.1f}s" for n, c, _ in _TIMES), file=sys.stderr, flush=True)
