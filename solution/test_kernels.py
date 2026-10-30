"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import TritonSwiGLUFunction

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = TritonSwiGLUFunction.apply(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_rmsnorm_matches_pytorch(dtype, weight_grad):
    from solution.rmsnorm import TritonRMSNormFunction

    torch.manual_seed(0)
    shape = (4, 128, 1536)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=weight_grad)
    dy = torch.randn(shape, device="cuda", dtype=dtype)

    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (x_f * torch.rsqrt(var + eps) * w.float()).to(dtype)
    ref.backward(dy)
    ref_dx = x.grad.float().clone()
    ref_dw = w.grad.float().clone() if weight_grad else None
    x.grad = None
    if weight_grad:
        w.grad = None

    out = TritonRMSNormFunction.apply(x, w, eps)
    out.backward(dy)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)
    if weight_grad:
        torch.testing.assert_close(w.grad.float(), ref_dw, rtol=tol, atol=tol)
    else:
        assert w.grad is None


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_rope_matches_pytorch(dtype, layout):
    from solution.rope import TritonRoPEFunction

    torch.manual_seed(0)
    B, S, H, D = 2, 64, 12, 128
    half = D // 2
    if layout == "bshd":
        x_shape = (B, S, H, D)
        cos_u_dim = 2
    else:
        x_shape = (B, H, S, D)
        cos_u_dim = 1

    x = torch.randn(x_shape, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn(x_shape, device="cuda", dtype=dtype)
    freqs = torch.randn((B, S, half), device="cuda", dtype=dtype)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    x_f = x.float()
    c_f = cos.float().unsqueeze(cos_u_dim)[..., :half]
    s_f = sin.float().unsqueeze(cos_u_dim)[..., :half]
    x1, x2 = x_f[..., :half], x_f[..., half:]
    ref = torch.cat([x1 * c_f - x2 * s_f, x2 * c_f + x1 * s_f], dim=-1)
    ref.backward(dout.float())
    ref_dx = x.grad.float().clone()
    x.grad = None

    out = TritonRoPEFunction.apply(x, cos, sin)
    out.backward(dout)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("ndim", [2, 3])
def test_cross_entropy_matches_pytorch(dtype, ndim):
    import solution

    solution.apply()
    torch.manual_seed(0)
    V = 16384
    if ndim == 2:
        logits_shape = (32, V)
        targets_shape = (32,)
    else:
        logits_shape = (2, 16, V)
        targets_shape = (2, 16)

    logits = torch.randn(logits_shape, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.randint(0, V, targets_shape, device="cuda", dtype=torch.long)
    targets.view(-1)[::5] = -100

    ref_logits = logits.detach().float().requires_grad_(True)
    ref_loss = solution._orig_F_cross_entropy(
        ref_logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    ref_loss.backward()
    ref_grad = ref_logits.grad.clone()

    loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
    loss.backward()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(logits.grad.float(), ref_grad, rtol=tol, atol=tol)


def test_apply_patches_cross_entropy_and_model():
    import solution

    dummy_model = SimpleNamespace(loss_function=None, loss_fct=None)
    solution.apply(model=dummy_model)

    assert F.cross_entropy is solution.custom_cross_entropy
    assert nn.CrossEntropyLoss.forward is solution.custom_ce_forward
    assert dummy_model.loss_function is solution.custom_model_loss_function
    assert dummy_model.loss_fct is solution.custom_cross_entropy
