"""Diagnostics: where do the dropout masks of the RMSNorm kernels disagree, if anywhere?
(Not part of solution/; run on a GPU after the measured runs.)"""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/kaggle/working/repo")
from solution.rmsnorm import _custom_dropout_keep, rmsnorm_drop_bwd, rmsnorm_drop_fwd  # noqa: E402


@triton.jit
def probe_keep(Out, N, p_drop, seed, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    keep = _custom_dropout_keep(row * N + cols, seed, p_drop)
    tl.store(Out + row * N + cols, keep.to(tl.int8), mask=mask)


def torch_keep(n, d, seed, p):
    m = 0xFFFFFFFF
    x = (torch.arange(n * d, device="cuda", dtype=torch.int64) + seed * 0x9E3779B9) & m
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & m
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & m
    x ^= x >> 16
    u = (x >> 8).to(torch.float32) * (1.0 / 16777216.0)
    return (u >= torch.tensor(p, dtype=torch.float32)).view(n, d)


def main():
    torch.manual_seed(0)
    n, d, p = 512, 1536, 0.3
    x = torch.randn(n, d, device="cuda").sign() * (0.5 + torch.rand(n, d, device="cuda"))
    w = torch.rand(d, device="cuda") + 0.5
    y, rs, (d1, d2) = rmsnorm_drop_fwd(x, w, 1e-6, torch.float16, p, (123, 456))
    for seed, drop in ((123, d1), (456, d2)):
        probe = torch.empty(n, d, device="cuda", dtype=torch.int8)
        probe_keep[(n,)](probe, d, p, seed, BLOCK=2048)
        fwd = drop != 0
        ref = torch_keep(n, d, seed, p)
        for name, a, b in (("fwd vs probe", fwd, probe.bool()), ("probe vs torch", probe.bool(), ref)):
            bad = (a != b).nonzero().tolist()
            print(f"PROBE seed {seed} {name}: {len(bad)} mismatches {bad[:5]}")
    # backward: route a one-hot through each branch and read which elements came back
    t = torch.ones(n, d, device="cuda", dtype=torch.float16)
    zero = torch.zeros_like(t)
    one_w = torch.ones(d, device="cuda")
    big_rs = torch.full((n,), 1.0, device="cuda")
    for k, (seed, drop) in enumerate(((123, d1), (456, d2))):
        branch = (t, zero) if k == 0 else (zero, t)
        # x = 0 kills the norm's correction term: dx = rsqrt * w * dy exactly
        dx = rmsnorm_drop_bwd(zero, torch.zeros_like(x), one_w, big_rs, p_drop=p,
                              seeds=(123, 456), branch_grads=branch)
        bwd = dx != 0
        bad = (bwd != (drop != 0)).nonzero().tolist()
        print(f"PROBE seed {seed} bwd vs fwd: {len(bad)} mismatches {bad[:5]}")


if __name__ == "__main__":
    main()
