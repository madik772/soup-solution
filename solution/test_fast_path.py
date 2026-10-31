"""The packed training step against transformers + peft + bitsandbytes, on a small random Qwen2.

Run on a GPU: `python -m pytest solution -q`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


@needs_cuda
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_dequant_matches_bitsandbytes(nested):
    bnb_f = pytest.importorskip("bitsandbytes.functional")
    from solution.nf4 import NF4Weight

    torch.manual_seed(0)
    w = torch.randn(384, 256, device="cuda", dtype=torch.float16) * 0.05
    packed, state = bnb_f.quantize_4bit(w, blocksize=64, compress_statistics=nested, quant_type="nf4")
    ref = bnb_f.dequantize_4bit(packed, state)
    param = SimpleNamespace(quant_state=state, data=packed, device=packed.device, dtype=packed.dtype,
                            is_cuda=True)
    out = torch.empty_like(ref)
    NF4Weight(param).dequant_into(out)
    torch.testing.assert_close(out, ref)


@needs_cuda
def test_lmhead_loss_matches_pytorch():
    from solution.cross_entropy import LMHeadLossFunction
    from solution.unpad import inverse_map

    torch.manual_seed(0)
    n, d, vocab = 700, 64, 5003
    hidden = (torch.randn(n, d, device="cuda") * 0.5).half().requires_grad_(True)
    weight = (torch.randn(vocab, d, device="cuda") * 0.3).half()
    tgt = torch.randint(0, vocab, (n,), device="cuda")
    tgt[torch.rand(n, device="cuda") < 0.4] = -100
    is_lab = tgt != -100
    lab_pos, unl_pos = is_lab.nonzero().squeeze(1), (~is_lab).nonzero().squeeze(1)
    order = torch.cat([lab_pos, unl_pos])
    n_lab = lab_pos.numel()
    plan = SimpleNamespace(n_lab=n_lab, order=order.int(), tgt_sorted=tgt[order].int(),
                           lab_map=inverse_map(lab_pos, n), denom=n_lab)
    stats = SimpleNamespace()
    loss = LMHeadLossFunction.apply(hidden, weight, plan, stats)
    (loss * 8.0).backward()

    h_ref = hidden.detach().clone().requires_grad_(True)
    logits = h_ref @ weight.t()
    ref = F.cross_entropy(logits.float(), tgt, ignore_index=-100)
    (ref * 8.0).backward()
    logp = F.log_softmax(logits.float(), dim=-1)
    ent_ref = -(logp.exp() * logp).sum(-1)
    acc_ref = ((logits.argmax(-1) == tgt) & is_lab).sum()

    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)
    assert _rel(hidden.grad, h_ref.grad) < 1e-2
    torch.testing.assert_close(stats.ent, ent_ref[order], rtol=1e-3, atol=1e-3)
    assert stats.correct.sum().item() == acc_ref.item()
    torch.testing.assert_close(stats.loss_tok, -logp[lab_pos, tgt[lab_pos]], rtol=1e-4, atol=1e-4)


def _tiny_qlora_qwen2(dropout):
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=2048, hidden_size=256, intermediate_size=640, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).half()
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                lin = getattr(parent, name)
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                      compute_dtype=torch.float16, compress_statistics=True,
                                      quant_type="nf4")
                q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                             compress_statistics=True, quant_type="nf4")
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                setattr(parent, name, q)
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad_(False)
        if p.dtype == torch.float16:  # as peft.prepare_model_for_kbit_training does
            p.data = p.data.float()
    try:
        model.is_loaded_in_4bit = True  # what from_pretrained(quantization_config=...) sets
    except Exception:  # noqa: BLE001
        pass
    lcfg = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=dropout, target_modules=["q_proj", "v_proj"],
                           task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lcfg)
    torch.manual_seed(1)
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            if "lora_B" in n:
                p.data.normal_(0, 0.05)
    pm.train()
    return pm


def _batch(vocab):
    torch.manual_seed(2)
    B, S = 3, 40
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    lengths = torch.tensor([40, 31, 17], device="cuda")
    am = (torch.arange(S, device="cuda")[None] < lengths[:, None]).long()
    labels = ids.clone()
    labels[:, :6] = -100
    labels[am == 0] = -100
    return {"input_ids": ids, "attention_mask": am, "labels": labels}


@needs_cuda
def test_packed_step_matches_transformers():
    from solution import fast_qwen2

    pm = _tiny_qlora_qwen2(dropout=0.0)
    inputs = _batch(pm.config.vocab_size)
    lora = [p for n, p in pm.named_parameters() if p.requires_grad]

    with torch.autocast("cuda", dtype=torch.float16):
        out = pm(**inputs)  # no plan is active: the standard transformers + peft path
    ref_loss = out.loss
    ref_loss.backward()
    ref_grads = [p.grad.clone() for p in lora]
    logits = out.logits.float()
    logp = F.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1)
    am = inputs["attention_mask"]
    ref_entropy = (ent * am).sum() / am.sum()
    for p in lora:
        p.grad = None

    trainer = SimpleNamespace(args=SimpleNamespace(use_liger_kernel=False), compute_loss_func=None,
                              label_smoother=None)
    plan = fast_qwen2.plan_batch(trainer, pm, dict(inputs), None)
    assert plan is not None
    with torch.autocast("cuda", dtype=torch.float16):
        loss = fast_qwen2.causal_lm_forward(fast_qwen2._causal_lm(pm), plan)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss.float(), rtol=3e-3, atol=3e-3)
    for g, r in zip((p.grad for p in lora), ref_grads):
        assert _rel(g, r) < 3e-2
    entropy = plan.stats.ent.sum() / am.sum()
    torch.testing.assert_close(entropy, ref_entropy, rtol=3e-3, atol=3e-3)
    rw = fast_qwen2.rewind_step(plan).token_loss
    assert rw.shape == inputs["labels"].shape
    torch.testing.assert_close(rw.sum() / plan.n_lab, loss.detach(), rtol=1e-4, atol=1e-4)


@needs_cuda
def test_packed_step_dropout_masks_are_consistent():
    """With LoRA dropout, forward and the recomputing backward must use the same masks: the
    gradient must match a finite difference of the loss along the gradient direction."""
    from solution import fast_qwen2

    pm = _tiny_qlora_qwen2(dropout=0.3)
    inputs = _batch(pm.config.vocab_size)
    lora = [p for n, p in pm.named_parameters() if p.requires_grad]
    trainer = SimpleNamespace(args=SimpleNamespace(use_liger_kernel=False), compute_loss_func=None,
                              label_smoother=None)
    plan = fast_qwen2.plan_batch(trainer, pm, dict(inputs), None)
    lm = fast_qwen2._causal_lm(pm)
    loss = fast_qwen2.causal_lm_forward(lm, plan)
    loss.backward()
    grads = [p.grad.clone() for p in lora]
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    eps = 2e-2 / gnorm
    with torch.no_grad():
        for p, g in zip(lora, grads):
            p.add_(g, alpha=eps.item())
        plan.consumed = False
        loss2 = fast_qwen2.causal_lm_forward(lm, plan)
        for p, g in zip(lora, grads):
            p.sub_(g, alpha=eps.item())
    predicted = (eps * gnorm ** 2).item()
    actual = (loss2 - loss).item()
    assert abs(actual - predicted) < 0.25 * abs(predicted) + 1e-4
