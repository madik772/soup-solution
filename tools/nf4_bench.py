"""Microbenchmark of NF4 dequant tilings on the GPU (diagnostics; not part of solution/)."""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/kaggle/working/repo")
from solution import nf4  # noqa: E402


@triton.jit
def bench_dequant_flat(Packed, AbsQ, Abs2, Code, Code2, Offset, Out, n_qblocks,
                       QBLOCK: tl.constexpr, QBLOCK_HALF: tl.constexpr, QBLOCK2: tl.constexpr,
                       ROWS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    blk = pid * ROWS + tl.arange(0, ROWS)
    ok = blk < n_qblocks
    aq = tl.load(AbsQ + blk, mask=ok, other=0).to(tl.int32)
    scale = tl.load(Code2 + aq) * tl.load(Abs2 + blk // QBLOCK2, mask=ok, other=0.0) + tl.load(Offset)
    e = tl.arange(0, QBLOCK)[None, :]
    packed = tl.load(Packed + blk[:, None] * QBLOCK_HALF + e // 2, mask=ok[:, None], other=0).to(tl.int32)
    nib = (packed >> ((1 - (e % 2)) * 4)) & 15
    val = tl.load(Code + nib) * scale[:, None]
    tl.store(Out + blk[:, None] * QBLOCK + e, val.to(Out.dtype.element_ty), mask=ok[:, None])


def main():
    dev = "cuda"
    n_out, n_in = 8960, 1536
    n = n_out * n_in
    torch.manual_seed(0)
    w = SimpleW(n, dev)
    out = torch.empty(n, device=dev, dtype=torch.float16)
    ref = torch.empty_like(out)
    nq = n // 64

    def current(rows=nf4.QBLOCKS_PER_PROGRAM, warps=4):
        def run():
            nf4._custom_nf4_dequant_kernel[(triton.cdiv(nq, rows),)](
                w.packed, w.absq, w.abs2, w.code, w.code2, w.offset, out, nq,
                QBLOCK=64, QBLOCK_HALF=32, QBLOCK2=256, NESTED=True, ROWS=rows,
                num_warps=warps, num_stages=2)
        return run

    def flat(rows, warps):
        def run():
            bench_dequant_flat[(triton.cdiv(nq, rows),)](
                w.packed, w.absq, w.abs2, w.code, w.code2, w.offset, out, nq,
                QBLOCK=64, QBLOCK_HALF=32, QBLOCK2=256, ROWS=rows,
                num_warps=warps, num_stages=2)
        return run

    current()()
    ref.copy_(out)
    variants = {"current r32 w4": current(), "current r32 w8": current(32, 8),
                "current r64 w8": current(64, 8), "current r16 w4": current(16, 4),
                "flat r32 w4": flat(32, 4), "flat r16 w4": flat(16, 4),
                "flat r64 w8": flat(64, 8), "flat r32 w8": flat(32, 8)}
    nbytes = n * 2 + n // 2
    for name, fn in variants.items():
        fn()
        torch.cuda.synchronize()
        same = torch.equal(out, ref)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 50
        print(f"BENCH nf4 {name:16s} {ms * 1000:7.1f} us  {nbytes / ms / 1e6:6.1f} GB/s  same={same}")


class SimpleW:
    def __init__(self, n, dev):
        self.packed = torch.randint(0, 256, (n // 2,), device=dev, dtype=torch.uint8)
        self.absq = torch.randint(0, 256, (n // 64,), device=dev, dtype=torch.uint8)
        self.abs2 = torch.rand(n // 64 // 256 + 1, device=dev)
        self.code = torch.linspace(-1, 1, 16, device=dev)
        self.code2 = torch.linspace(-1, 1, 256, device=dev).abs()
        self.offset = torch.tensor([0.01], device=dev)


if __name__ == "__main__":
    main()
