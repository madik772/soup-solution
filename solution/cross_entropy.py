import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _custom_ce_fwd_kernel(
    X, Y, Loss, LSE,
    stride_x_row, stride_x_col,
    V, ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(Y + row_idx)

    if target == ignore_index or target < 0 or target >= V:
        tl.store(Loss + row_idx, 0.0)
        tl.store(LSE + row_idx, 0.0)
        return

    # Проход 1: находим максимум строки блоками по 4096
    m = -float('inf')
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(X + row_idx * stride_x_row + cols * stride_x_col, mask=mask, other=-float('inf')).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    # Проход 2: сумма экспонент в FP32
    s = 0.0
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(X + row_idx * stride_x_row + cols * stride_x_col, mask=mask, other=-float('inf')).to(tl.float32)
        s += tl.sum(tl.exp(x - m), axis=0)

    lse = m + tl.log(s)
    tl.store(LSE + row_idx, lse)

    x_target = tl.load(X + row_idx * stride_x_row + target * stride_x_col).to(tl.float32)
    tl.store(Loss + row_idx, lse - x_target)

@triton.jit
def _custom_ce_bwd_kernel(
    dX, X, Y, LSE,
    stride_dx_row, stride_dx_col,
    stride_x_row, stride_x_col,
    V, scale, ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(Y + row_idx)

    if target == ignore_index or target < 0 or target >= V:
        for off in range(0, V, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < V
            tl.store(dX + row_idx * stride_dx_row + cols * stride_dx_col, 0.0, mask=mask)
        return

    lse = tl.load(LSE + row_idx)

    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(X + row_idx * stride_x_row + cols * stride_x_col, mask=mask, other=-float('inf')).to(tl.float32)
        p = tl.exp(x - lse)
        is_target = (cols == target).to(tl.float32)
        grad = (p - is_target) * scale
        tl.store(dX + row_idx * stride_dx_row + cols * stride_dx_col, grad, mask=mask)

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index=-100):
        N, V = logits.shape
        Loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        LSE = torch.empty(N, device=logits.device, dtype=torch.float32)

        BLOCK_SIZE = 4096
        _custom_ce_fwd_kernel[(N,)](
            logits, targets, Loss, LSE,
            logits.stride(0), logits.stride(1),
            V, ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        valid_mask = (targets != ignore_index) & (targets >= 0) & (targets < V)
        n_valid = max(int(valid_mask.sum().item()), 1)

        ctx.save_for_backward(logits, targets, LSE)
        ctx.n_valid = n_valid
        ctx.ignore_index = ignore_index
        ctx.V = V
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return Loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, LSE = ctx.saved_tensors
        N, V = logits.shape
        dX = torch.empty_like(logits)

        scale = (grad_output.item() if grad_output.numel() == 1 else 1.0) / ctx.n_valid

        _custom_ce_bwd_kernel[(N,)](
            dX, logits, targets, LSE,
            dX.stride(0), dX.stride(1),
            logits.stride(0), logits.stride(1),
            V, scale, ctx.ignore_index,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=8,
        )
        return dX, None, None
