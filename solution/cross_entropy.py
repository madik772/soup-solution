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
    row_idx = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row_idx).to(tl.int64)

    if target == ignore_index or target < 0 or target >= V:
        tl.store(Loss + row_idx, 0.0)
        tl.store(LSE + row_idx, 0.0)
        return

    # Проход 1: находим максимум строки блоками по 4096 в FP32
    m = -float('inf')
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float('inf'),
        ).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    # Проход 2: сумма экспонент в FP32 без переполнения
    s = 0.0
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float('inf'),
        ).to(tl.float32)
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
    row_idx = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row_idx).to(tl.int64)

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
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float('inf'),
        ).to(tl.float32)
        p = tl.exp(x - lse)
        is_target = (cols == target).to(tl.float32)
        grad = (p - is_target) * scale
        tl.store(
            dX + row_idx * stride_dx_row + cols * stride_dx_col,
            grad.to(dX.dtype.element_ty),
            mask=mask,
        )


class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index=-100):
        orig_shape = logits.shape
        logits_2d = logits.reshape(-1, orig_shape[-1])
        targets_1d = targets.reshape(-1).contiguous()

        N, V = logits_2d.shape
        Loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        LSE = torch.empty(N, device=logits.device, dtype=torch.float32)

        BLOCK_SIZE = 4096
        _custom_ce_fwd_kernel[(N,)](
            logits_2d, targets_1d, Loss, LSE,
            logits_2d.stride(0), logits_2d.stride(1),
            V, ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
        n_valid = max(int(valid_mask.sum().item()), 1)

        ctx.save_for_backward(logits_2d, targets_1d, LSE)
        ctx.orig_shape = orig_shape
        ctx.n_valid = n_valid
        ctx.ignore_index = ignore_index
        ctx.V = V
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return Loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, targets_1d, LSE = ctx.saved_tensors
        N, V = logits_2d.shape
        dX = torch.empty_like(logits_2d)

        scale = (grad_output.float().item() if grad_output.numel() == 1 else 1.0) / ctx.n_valid

        _custom_ce_bwd_kernel[(N,)](
            dX, logits_2d, targets_1d, LSE,
            dX.stride(0), dX.stride(1),
            logits_2d.stride(0), logits_2d.stride(1),
            V, scale, ctx.ignore_index,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=8,
        )
        return dX.reshape(*ctx.orig_shape), None, None


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    """Chunked Cross-Entropy over hidden_states and lm_head.weight without saving full logits."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        targets,
        ignore_index=-100,
        chunk_size=1024,
        reduction="mean",
        precomputed_logits=None,
    ):
        orig_hidden_shape = hidden_states.shape
        compute_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else hidden_states.dtype
        )
        hidden_2d = hidden_states.to(compute_dtype).contiguous().reshape(-1, orig_hidden_shape[-1])
        targets_1d = targets.reshape(-1).contiguous()

        N, D = hidden_2d.shape
        V = weight.shape[0]
        Loss = torch.empty(N, device=hidden_2d.device, dtype=torch.float32)
        LSE = torch.empty(N, device=hidden_2d.device, dtype=torch.float32)

        BLOCK_SIZE = 4096
        w_cast = None
        logits_2d = (
            precomputed_logits.reshape(N, V)
            if (precomputed_logits is not None and precomputed_logits.numel() == N * V)
            else None
        )

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_n = end - start
            t_chunk = targets_1d[start:end]
            if logits_2d is not None:
                logits_chunk = logits_2d[start:end]
            else:
                if w_cast is None:
                    w_cast = weight.to(compute_dtype) if weight.dtype != compute_dtype else weight
                h_chunk = hidden_2d[start:end]
                logits_chunk = torch.matmul(h_chunk, w_cast.t())

            _custom_ce_fwd_kernel[(chunk_n,)](
                logits_chunk,
                t_chunk,
                Loss[start:end],
                LSE[start:end],
                logits_chunk.stride(0),
                logits_chunk.stride(1),
                V,
                ignore_index,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
            )

        valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
        n_valid = max(int(valid_mask.sum().item()), 1)

        ctx.save_for_backward(hidden_2d, weight, targets_1d, LSE)
        ctx.orig_hidden_shape = orig_hidden_shape
        ctx.orig_hidden_dtype = hidden_states.dtype
        ctx.compute_dtype = compute_dtype
        ctx.n_valid = n_valid if reduction == "mean" else 1
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.BLOCK_SIZE = BLOCK_SIZE

        if reduction == "sum":
            return Loss.sum()
        return Loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        hidden_2d, weight, targets_1d, LSE = ctx.saved_tensors
        N, D = hidden_2d.shape
        V = weight.shape[0]
        chunk_size = ctx.chunk_size
        compute_dtype = ctx.compute_dtype

        w_cast = weight.to(compute_dtype) if weight.dtype != compute_dtype else weight
        d_hidden = torch.empty_like(hidden_2d)

        need_dw = weight is not None and weight.requires_grad
        d_weight = (
            torch.zeros((V, D), device=weight.device, dtype=compute_dtype) if need_dw else None
        )

        scale = (grad_output.float().item() if grad_output.numel() == 1 else 1.0) / ctx.n_valid

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_n = end - start
            h_chunk = hidden_2d[start:end]
            t_chunk = targets_1d[start:end]
            lse_chunk = LSE[start:end]

            logits_chunk = torch.matmul(h_chunk, w_cast.t())

            # Write d_logits_chunk in-place into logits_chunk
            _custom_ce_bwd_kernel[(chunk_n,)](
                logits_chunk,
                logits_chunk,
                t_chunk,
                lse_chunk,
                logits_chunk.stride(0),
                logits_chunk.stride(1),
                logits_chunk.stride(0),
                logits_chunk.stride(1),
                V,
                scale,
                ctx.ignore_index,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=8,
            )

            torch.matmul(logits_chunk, w_cast, out=d_hidden[start:end])

            if need_dw:
                d_weight.addmm_(logits_chunk.t(), h_chunk)

        if d_hidden.dtype != ctx.orig_hidden_dtype:
            d_hidden = d_hidden.to(ctx.orig_hidden_dtype)
        if need_dw and d_weight.dtype != weight.dtype:
            d_weight = d_weight.to(weight.dtype)

        return (
            d_hidden.reshape(*ctx.orig_hidden_shape),
            d_weight,
            None,
            None,
            None,
            None,
            None,
        )
