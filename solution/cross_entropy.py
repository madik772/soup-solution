import torch
import triton
import triton.language as tl

CHUNK_SIZE = 4096
DEFAULT_CHUNK_SIZE = 4096
BLOCK_SIZE = 4096


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

    m = -float("inf")
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    s = 0.0
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        s += tl.sum(tl.exp(x - m), axis=0)

    lse = m + tl.log(s)
    tl.store(LSE + row_idx, lse)

    x_target = tl.load(X + row_idx * stride_x_row + target * stride_x_col).to(tl.float32)
    tl.store(Loss + row_idx, lse - x_target)


@triton.jit
def _custom_ce_bwd_kernel(
    dX, X, Y, LSE, GradOut, NValid,
    stride_dx_row, stride_dx_col,
    stride_x_row, stride_x_col,
    V, ignore_index,
    IS_MEAN: tl.constexpr,
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
    scale = tl.load(GradOut).to(tl.float32)
    if IS_MEAN:
        scale = scale / tl.load(NValid).to(tl.float32)

    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        x = tl.load(
            X + row_idx * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        p = tl.exp(x - lse)
        is_target = (cols == target).to(tl.float32)
        grad = (p - is_target) * scale
        tl.store(
            dX + row_idx * stride_dx_row + cols * stride_dx_col,
            grad.to(dX.dtype.element_ty),
            mask=mask,
        )


def _ce_forward_nograd(logits_2d, targets_1d, ignore_index=-100, reduction="mean"):
    N, V = logits_2d.shape
    if not targets_1d.is_contiguous():
        targets_1d = targets_1d.contiguous()

    Loss = torch.empty(N, device=logits_2d.device, dtype=torch.float32)
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_n = end - start
        _custom_ce_fwd_kernel[(chunk_n,)](
            logits_2d[start:end],
            targets_1d[start:end],
            Loss[start:end],
            Loss[start:end],
            logits_2d.stride(0),
            logits_2d.stride(1),
            V,
            ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    if reduction == "none":
        return Loss
    if reduction == "sum":
        return Loss.sum()
    valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
    n_valid = valid_mask.sum().clamp(min=1)
    return Loss.sum() / n_valid


class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index=-100, reduction="mean"):
        orig_shape = logits.shape
        logits_2d = logits if logits.ndim == 2 else logits.reshape(-1, orig_shape[-1])
        targets_1d = targets if targets.ndim == 1 else targets.reshape(-1)
        if not targets_1d.is_contiguous():
            targets_1d = targets_1d.contiguous()

        N, V = logits_2d.shape
        Loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        LSE = torch.empty(N, device=logits.device, dtype=torch.float32)

        for start in range(0, N, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N)
            chunk_n = end - start
            _custom_ce_fwd_kernel[(chunk_n,)](
                logits_2d[start:end],
                targets_1d[start:end],
                Loss[start:end],
                LSE[start:end],
                logits_2d.stride(0),
                logits_2d.stride(1),
                V,
                ignore_index,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )

        is_mean = reduction == "mean"
        if is_mean:
            valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
            n_valid = valid_mask.sum().clamp(min=1)
            out = Loss.sum() / n_valid
        else:
            n_valid = Loss
            out = Loss.sum()

        base = logits._base if logits._base is not None else logits
        ctx.save_for_backward(logits_2d, targets_1d, LSE, n_valid)
        ctx.orig_shape = orig_shape
        ctx.ignore_index = ignore_index
        ctx.is_mean = is_mean
        ctx.inplace_bwd = (not base.is_leaf) and logits_2d.is_contiguous()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, targets_1d, LSE, n_valid = ctx.saved_tensors
        N, V = logits_2d.shape

        dX = logits_2d if ctx.inplace_bwd else torch.empty_like(logits_2d)

        for start in range(0, N, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N)
            chunk_n = end - start
            _custom_ce_bwd_kernel[(chunk_n,)](
                dX[start:end],
                logits_2d[start:end],
                targets_1d[start:end],
                LSE[start:end],
                grad_output,
                n_valid,
                dX.stride(0),
                dX.stride(1),
                logits_2d.stride(0),
                logits_2d.stride(1),
                V,
                ctx.ignore_index,
                IS_MEAN=ctx.is_mean,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )
        return dX.view(*ctx.orig_shape), None, None, None


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    """Streaming Chunked Cross-Entropy over hidden_states and lm_head.weight with CHUNK_SIZE=4096."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        targets,
        ignore_index=-100,
        chunk_size=CHUNK_SIZE,
        reduction="mean",
        precomputed_logits=None,
    ):
        orig_hidden_shape = hidden_states.shape
        compute_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else hidden_states.dtype
        )
        h_cast = (
            hidden_states
            if hidden_states.dtype == compute_dtype
            else hidden_states.to(compute_dtype)
        )
        hidden_2d = h_cast if h_cast.ndim == 2 else h_cast.reshape(-1, orig_hidden_shape[-1])
        if not hidden_2d.is_contiguous():
            hidden_2d = hidden_2d.contiguous()

        targets_1d = targets if targets.ndim == 1 else targets.reshape(-1)
        if not targets_1d.is_contiguous():
            targets_1d = targets_1d.contiguous()

        N, D = hidden_2d.shape
        V = weight.shape[0]
        Loss = torch.empty(N, device=hidden_2d.device, dtype=torch.float32)
        LSE = torch.empty(N, device=hidden_2d.device, dtype=torch.float32)

        w_cast = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)
        w_t = w_cast.t()

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
                logits_chunk = torch.mm(hidden_2d[start:end], w_t)

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
                num_stages=2,
            )

        is_mean = reduction == "mean"
        if is_mean:
            valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
            n_valid = valid_mask.sum().clamp(min=1)
            out = Loss.sum() / n_valid
        else:
            n_valid = Loss
            out = Loss.sum()

        ctx.save_for_backward(hidden_2d, weight, targets_1d, LSE, n_valid)
        ctx.orig_hidden_shape = orig_hidden_shape
        ctx.orig_hidden_dtype = hidden_states.dtype
        ctx.compute_dtype = compute_dtype
        ctx.is_mean = is_mean
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden_2d, weight, targets_1d, LSE, n_valid = ctx.saved_tensors
        N, D = hidden_2d.shape
        V = weight.shape[0]
        chunk_size = ctx.chunk_size
        compute_dtype = ctx.compute_dtype

        w_cast = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)
        w_t = w_cast.t()
        d_hidden = torch.empty_like(hidden_2d)

        need_dw = weight is not None and weight.requires_grad
        d_weight = (
            torch.zeros((V, D), device=weight.device, dtype=compute_dtype) if need_dw else None
        )

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_n = end - start
            h_chunk = hidden_2d[start:end]
            t_chunk = targets_1d[start:end]
            lse_chunk = LSE[start:end]

            logits_chunk = torch.mm(h_chunk, w_t)

            _custom_ce_bwd_kernel[(chunk_n,)](
                logits_chunk,
                logits_chunk,
                t_chunk,
                lse_chunk,
                grad_output,
                n_valid,
                logits_chunk.stride(0),
                logits_chunk.stride(1),
                logits_chunk.stride(0),
                logits_chunk.stride(1),
                V,
                ctx.ignore_index,
                IS_MEAN=ctx.is_mean,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )

            torch.mm(logits_chunk, w_cast, out=d_hidden[start:end])

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


# ---------------------------------------------------------------------------------------------
# Output head + loss + TRL's metrics without the [B, S, vocab] logits.
#
# The packed real tokens are ordered labelled-first. Chunk by chunk, one GEMM gives the fp16
# logits (as transformers' fp16 head does), and one kernel pass per row reads them once with an
# online max / sum-exp / sum(exp * x) to get log-sum-exp, the entropy TRL logs (every real
# token) and the argmax for its token accuracy; for labelled rows it then takes the target
# logit, writes the token loss, and overwrites the logits with (softmax - onehot) * GRAD_SCALE,
# the loss gradient, which a second GEMM takes back to the hidden states. Prompt rows skip the
# gradient, so that GEMM only covers the answer tokens, and nothing is recomputed in backward:
# the gradient is scaled by the incoming loss scale / denominator there.
#
# GRAD_SCALE keeps small probabilities representable in fp16, the role fp16 loss scaling plays
# in the unfused step (where the scaled fp32 gradient is rounded to fp16 for the head GEMM).
# ---------------------------------------------------------------------------------------------

CE_CHUNK = 512
CE_BLOCK_V = 2048
GRAD_SCALE = 1024.0


@triton.jit(do_not_specialize=["n_lab"])
def _custom_lmhead_ce_kernel(
    Logits, Targets, Loss, Ent, Correct,
    stride_row, n_lab, V, grad_scale,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ptr = Logits + row * stride_row

    m = -float("inf")
    s = 0.0
    t = 0.0
    best = -float("inf")
    best_idx = 0
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        blk_arg = tl.argmax(x, axis=0)
        best_idx = tl.where(blk_max > best, off + blk_arg, best_idx)
        best = tl.maximum(best, blk_max)
        m_new = tl.maximum(m, blk_max)
        alpha = tl.exp(m - m_new)
        e = tl.exp(x - m_new)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(mask, e * x, 0.0), axis=0)
        m = m_new

    lse = m + tl.log(s)
    tl.store(Ent + row, lse - t / s)

    if row < n_lab:
        tgt = tl.load(Targets + row).to(tl.int64)
        x_t = tl.load(ptr + tgt).to(tl.float32)
        tl.store(Loss + row, lse - x_t)
        tl.store(Correct + row, (best_idx == tgt).to(tl.int32))
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            mask = cols < V
            x = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(x - lse)
            g = tl.where(cols == tgt, p - 1.0, p) * grad_scale
            tl.store(ptr + cols, g.to(Logits.dtype.element_ty), mask=mask)


class LMHeadLossFunction(torch.autograd.Function):
    """loss = sum of the labelled tokens' cross-entropy / denom, from packed hidden states.

    `stats` (a plain object) receives the per-token entropy of every real token, the labelled
    tokens' losses and argmax hits in labelled-first order; `plan` carries that order.
    """

    @staticmethod
    def forward(ctx, hidden, weight, plan, stats):
        from solution.unpad import row_gather

        n_real, d = hidden.shape
        n_lab = plan.n_lab
        vocab = weight.shape[0]
        dev = hidden.device
        hidden_sorted = row_gather(hidden, plan.order)
        loss_tok = torch.empty(n_lab, device=dev, dtype=torch.float32)
        correct = torch.empty(n_lab, device=dev, dtype=torch.int32)
        ent = torch.empty(n_real, device=dev, dtype=torch.float32)
        d_sorted = torch.empty((n_lab, d), device=dev, dtype=hidden.dtype)

        chunk = min(CE_CHUNK, n_real)
        buf = torch.empty((chunk, vocab), device=dev, dtype=hidden.dtype)
        w_t = weight.t()
        for start in range(0, n_real, chunk):
            end = min(start + chunk, n_real)
            rows = end - start
            logits = buf[:rows]
            torch.mm(hidden_sorted[start:end], w_t, out=logits)
            lab_rows = max(0, min(end, n_lab) - start)
            _custom_lmhead_ce_kernel[(rows,)](
                logits, plan.tgt_sorted[start:],
                loss_tok[start:] if lab_rows else loss_tok,
                ent[start:],
                correct[start:] if lab_rows else correct,
                logits.stride(0), lab_rows, vocab, GRAD_SCALE,
                BLOCK_V=CE_BLOCK_V,
                num_warps=8,
                num_stages=2,
            )
            if lab_rows:
                torch.mm(logits[:lab_rows], weight, out=d_sorted[start:start + lab_rows])
        del buf, hidden_sorted

        stats.ent, stats.loss_tok, stats.correct = ent, loss_tok, correct
        denom = plan.denom
        loss = loss_tok.sum() / denom
        ctx.save_for_backward(d_sorted)
        ctx.plan = plan
        ctx.denom = denom
        ctx.n_real = n_real
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        from solution.unpad import row_gather

        (d_sorted,) = ctx.saved_tensors
        scale = (grad_loss.float() / (ctx.denom * GRAD_SCALE)).reshape(1)
        d_hidden = row_gather(d_sorted, ctx.plan.lab_map, n_rows=ctx.n_real, scale=scale)
        return d_hidden, None, None, None
