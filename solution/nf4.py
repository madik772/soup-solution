"""Own NF4 dequantization for bitsandbytes 4-bit weights (QLoRA layout, double quant).

bitsandbytes stores a [N, K] weight as (N*K)/2 bytes, two 4-bit codes per byte (element 2i in
the high nibble, 2i+1 in the low one), one absmax per block of 64 elements, and with double
quant the absmax itself as uint8 codes over a 256-entry map with one fp32 scale per 256 blocks
plus a global offset. The value is code[nibble] * absmax, computed in fp32 and rounded once to
the compute dtype, the same arithmetic as bitsandbytes' own dequantize_4bit, so the weights the
GEMMs see are the same.

Every frozen projection of every layer is dequantized here on the fly (forward, and once more in
the backward pass), into one buffer per fused group (q|k|v, gate|up), so one cuBLAS GEMM covers
the whole group.
"""

import torch
import triton
import triton.language as tl

QBLOCKS_PER_PROGRAM = 32


@triton.jit(do_not_specialize=["n_qblocks"])
def _custom_nf4_dequant_kernel(
    Packed, AbsQ, Abs2, Code, Code2, Offset, Out,
    n_qblocks,
    QBLOCK: tl.constexpr,
    QBLOCK_HALF: tl.constexpr,
    QBLOCK2: tl.constexpr,
    NESTED: tl.constexpr,
    ROWS: tl.constexpr,
):
    # One tile = ROWS quantization blocks x QBLOCK elements: the block scale is read once per
    # row of the tile (not per byte) and broadcast over its QBLOCK / 2 packed bytes.
    pid = tl.program_id(0).to(tl.int64)
    blk = pid * ROWS + tl.arange(0, ROWS)
    blk_ok = blk < n_qblocks
    if NESTED:
        aq = tl.load(AbsQ + blk, mask=blk_ok, other=0).to(tl.int32)
        scale = tl.load(Code2 + aq) * tl.load(Abs2 + blk // QBLOCK2, mask=blk_ok, other=0.0)
        scale = scale + tl.load(Offset)
    else:
        scale = tl.load(AbsQ + blk, mask=blk_ok, other=0.0).to(tl.float32)
    scale = scale[:, None]

    byte_idx = blk[:, None] * QBLOCK_HALF + tl.arange(0, QBLOCK_HALF)[None, :]
    packed = tl.load(Packed + byte_idx, mask=blk_ok[:, None], other=0).to(tl.int32)
    first = tl.load(Code + (packed >> 4))
    second = tl.load(Code + (packed & 15))

    vals = tl.interleave(first * scale, second * scale)
    out_idx = blk[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    tl.store(Out + out_idx, vals.to(Out.dtype.element_ty), mask=blk_ok[:, None])


class NF4Weight:
    """What the dequant kernel needs from one bitsandbytes Params4bit, read once."""

    def __init__(self, param):
        qs = param.quant_state
        self.packed = param.data.reshape(-1)
        self.shape = tuple(qs.shape)
        self.n_bytes = (self.shape[0] * self.shape[1]) // 2
        self.blocksize = int(qs.blocksize)
        self.code = qs.code.to(device=param.device, dtype=torch.float32).contiguous()
        self.nested = bool(getattr(qs, "nested", False))
        if self.nested:
            s2 = qs.state2
            self.absq = qs.absmax.contiguous()
            self.abs2 = s2.absmax.to(torch.float32).contiguous()
            self.code2 = s2.code.to(device=param.device, dtype=torch.float32).contiguous()
            self.block2 = int(s2.blocksize)
            off = qs.offset
            if not torch.is_tensor(off):
                off = torch.tensor(float(off), device=param.device)
            self.offset = off.to(device=param.device, dtype=torch.float32).reshape(1)
        else:
            self.absq = qs.absmax.to(torch.float32).contiguous()
            self.abs2 = self.absq
            self.code2 = self.code
            self.block2 = 1
            self.offset = self.code

    @staticmethod
    def supported(param):
        qs = getattr(param, "quant_state", None)
        if qs is None or getattr(qs, "quant_type", None) != "nf4" or param.dtype != torch.uint8:
            return False
        if len(getattr(qs, "shape", ())) != 2 or (qs.shape[0] * qs.shape[1]) % 2:
            return False
        bs = int(qs.blocksize)
        if bs < 2 or bs & (bs - 1) or (qs.shape[0] * qs.shape[1]) % bs or not param.is_cuda:
            return False
        if getattr(qs, "nested", False) and getattr(qs, "state2", None) is None:
            return False
        return True

    def dequant_into(self, out):
        """Write the [N, K] weight into `out` (contiguous, N*K elements)."""
        n_qblocks = (self.shape[0] * self.shape[1]) // self.blocksize
        grid = (triton.cdiv(n_qblocks, QBLOCKS_PER_PROGRAM),)
        _custom_nf4_dequant_kernel[grid](
            self.packed, self.absq, self.abs2, self.code, self.code2, self.offset, out,
            n_qblocks,
            QBLOCK=self.blocksize,
            QBLOCK_HALF=self.blocksize // 2,
            QBLOCK2=self.block2,
            NESTED=self.nested,
            ROWS=QBLOCKS_PER_PROGRAM,
            num_warps=4,
            num_stages=2,
        )
        return out


class NF4Group:
    """Several frozen projections that read the same input, dequantized into one [sum N, K]
    weight so that a single GEMM computes all of them (q|k|v, gate|up)."""

    def __init__(self, params, dtype=torch.float16):
        self.weights = [NF4Weight(p) for p in params]
        k = {w.shape[1] for w in self.weights}
        if len(k) != 1:
            raise ValueError("projections of a group must share the input size")
        self.in_features = k.pop()
        self.out_sizes = [w.shape[0] for w in self.weights]
        self.out_features = sum(self.out_sizes)
        self.dtype = dtype
        self.device = params[0].device

    def dequant(self):
        out = torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        row = 0
        for w, n in zip(self.weights, self.out_sizes):
            w.dequant_into(out[row:row + n])
            row += n
        return out
