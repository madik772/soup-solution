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

BLOCK_BYTES = 1024


@triton.jit(do_not_specialize=["n_bytes"])
def _custom_nf4_dequant_kernel(
    Packed, AbsQ, Abs2, Code, Code2, Offset, Out,
    n_bytes,
    QBLOCK_HALF: tl.constexpr,
    QBLOCK2: tl.constexpr,
    NESTED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    byte_idx = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    in_range = byte_idx < n_bytes

    packed = tl.load(Packed + byte_idx, mask=in_range, other=0).to(tl.int32)
    first = tl.load(Code + (packed >> 4))
    second = tl.load(Code + (packed & 15))

    # Both nibbles of a byte share one quantization block (blocks hold an even count).
    block = byte_idx // QBLOCK_HALF
    if NESTED:
        aq = tl.load(AbsQ + block, mask=in_range, other=0).to(tl.int32)
        scale = tl.load(Code2 + aq) * tl.load(Abs2 + block // QBLOCK2, mask=in_range, other=0.0)
        scale = scale + tl.load(Offset)
    else:
        scale = tl.load(AbsQ + block, mask=in_range, other=0.0).to(tl.float32)

    vals = tl.interleave(first * scale, second * scale)
    out_idx = pid * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
    tl.store(Out + out_idx, vals.to(Out.dtype.element_ty), mask=out_idx < 2 * n_bytes)


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
        if int(qs.blocksize) % 2 or not param.is_cuda:
            return False
        if getattr(qs, "nested", False) and getattr(qs, "state2", None) is None:
            return False
        return True

    def dequant_into(self, out):
        """Write the [N, K] weight into `out` (contiguous, N*K elements)."""
        grid = (triton.cdiv(self.n_bytes, BLOCK_BYTES),)
        _custom_nf4_dequant_kernel[grid](
            self.packed, self.absq, self.abs2, self.code, self.code2, self.offset, out,
            self.n_bytes,
            QBLOCK_HALF=self.blocksize // 2,
            QBLOCK2=self.block2,
            NESTED=self.nested,
            BLOCK_BYTES=BLOCK_BYTES,
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
