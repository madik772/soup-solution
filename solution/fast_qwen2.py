"""Packed (padding-free) Qwen2 + 4-bit LoRA training step, with layer recomputation.

What one labelled training forward does here instead of transformers' path:

* the real tokens of the batch (attention_mask == 1; TRL pads on the right) are packed into one
  [N, hidden] matrix right after the embedding lookup and stay packed to the loss: every
  projection, norm and activation runs on N rows, not B * S;
* each decoder layer is one autograd Function. Its forward keeps only its fp32 input (the
  residual stream); the backward recomputes what it needs, skipping down_proj, whose input
  gradient needs only the weight, and reuses the recomputed dequantized weights for the
  backward GEMMs, so every frozen weight is dequantized twice per step (forward, backward);
* q|k|v and gate|up are one GEMM each over a jointly dequantized weight; LoRA (q_proj, v_proj)
  is added in the GEMM epilogue (addmm with alpha=scaling); its dropout masks are made inside
  the input RMSNorm kernel and regenerated, never stored;
* attention: SDPA (memory-efficient kernel, causal) on padded q/k/v written by one Triton kernel
  that also applies RoPE and repeats the KV heads;
* the head and the loss run chunked over the packed tokens (cross_entropy.LMHeadLossFunction),
  which also yields TRL's entropy / token accuracy and Soup's per-row rewind losses, so the
  [B, S, 151936] logits are never built.

Values follow the transformers + peft + bitsandbytes step: same dequantized weights, the
residual stream in fp32, RMSNorm / RoPE / softmax statistics in fp32, matmuls in fp16.
Anything this path does not cover (other architectures, LoRA on other projections, DoRA, left
padding, packed batches, ...) is detected before the step and goes through the original code.
"""

import random
import sys
import time
import warnings
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from solution.cross_entropy import LMHeadLossFunction
from solution.nf4 import NF4Group, NF4Weight
from solution.rmsnorm import (
    TritonRMSNormFunction,
    fused_add_rmsnorm_bwd_,
    fused_add_rmsnorm_fwd,
    rmsnorm_drop_bwd,
    rmsnorm_drop_fwd,
)
from solution.rope import (
    qkv_rope_pack,
    qkv_rope_pack_backward,
    qkv_rope_pad,
    qkv_rope_pad_backward,
)
from solution.swiglu import swiglu_packed_bwd_, swiglu_packed_fwd
from solution.unpad import inverse_map, row_gather
from solution import timeline, warmup

F16 = torch.float16
STEP_ATTR = "_soup_chunked_step"  # where Soup's rewind recorder looks for per-token losses

_SDPA = getattr(F, "_orig_scaled_dot_product_attention", F.scaled_dot_product_attention)
_STATE = SimpleNamespace(plan=None, disabled=False, rng=None, varlen=None)

# Bytes per step for what the forward leaves for the backward: every layer's fp32 input (6 KB
# per token and layer) plus, for the last layers of the step, their gate|up output and residual
# sum (~41 KB per token and layer), which those layers then do not recompute. A fixed budget
# keeps the peak flat whatever the batch length: short batches keep more layers, long ones
# fewer (none once the inputs alone fill it).
ACTIVATION_BUDGET_BYTES = 512 * 2**20


class Unsupported(Exception):
    pass


def _warn_once(reason):
    if not _STATE.disabled:
        _STATE.disabled = True
        warnings.warn(f"solution: packed fast path off, using the standard path ({reason})")


# --------------------------------------------------------------------------- model inspection


def _projection(module):
    """(bitsandbytes Linear4bit, LoRA spec or None) of one projection, or Unsupported."""
    base = getattr(module, "base_layer", None)
    lora = None
    if base is None:
        base = module
    else:
        if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
            raise Unsupported("merged or disabled adapter")
        names = [a for a in module.active_adapters if a in module.lora_A]
        if len(names) > 1:
            raise Unsupported("several active adapters")
        if names:
            name = names[0]
            if name in getattr(module, "lora_variant", {}) or getattr(module, "use_dora", {}).get(name):
                raise Unsupported("LoRA variant")
            a, b = module.lora_A[name], module.lora_B[name]
            if getattr(a, "bias", None) is not None or getattr(b, "bias", None) is not None:
                raise Unsupported("LoRA bias")
            drop = module.lora_dropout[name]
            if isinstance(drop, torch.nn.Dropout):
                p = float(drop.p)
            elif isinstance(drop, torch.nn.Identity):
                p = 0.0
            else:
                raise Unsupported("LoRA dropout module")
            lora = SimpleNamespace(A=a.weight, B=b.weight, scaling=float(module.scaling[name]), p=p)
    if type(base).__name__ != "Linear4bit" or not NF4Weight.supported(base.weight):
        raise Unsupported("projection is not an NF4 bitsandbytes Linear4bit")
    if getattr(base, "compute_dtype", F16) not in (None, F16):
        raise Unsupported("compute dtype is not fp16")
    return base, lora


class _LayerInfo:
    def __init__(self, layer, config):
        attn, mlp = layer.self_attn, layer.mlp
        q, qa = _projection(attn.q_proj)
        k, ka = _projection(attn.k_proj)
        v, va = _projection(attn.v_proj)
        o, oa = _projection(attn.o_proj)
        g, ga = _projection(mlp.gate_proj)
        u, ua = _projection(mlp.up_proj)
        d, da = _projection(mlp.down_proj)
        if any(x is not None for x in (oa, ga, ua, da)):
            raise Unsupported("LoRA outside q/k/v")
        if any(m.bias is not None for m in (o, g, u, d)):
            raise Unsupported("bias on o/gate/up/down")

        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = getattr(attn, "head_dim", config.hidden_size // self.n_heads)
        self.scale = float(attn.scaling)
        self.hidden = config.hidden_size
        self.inter = g.weight.quant_state.shape[0]
        if self.n_heads % self.n_kv or self.head_dim % 2:
            raise Unsupported("head layout")

        self.qkv = NF4Group([q.weight, k.weight, v.weight])
        self.o = NF4Group([o.weight])
        self.gu = NF4Group([g.weight, u.weight])
        self.down = NF4Group([d.weight])
        sizes = self.qkv.out_sizes
        if sizes != [self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim]:
            raise Unsupported("q/k/v sizes")
        biases = []
        for base, n in zip((q, k, v), sizes):
            b = base.bias
            biases.append(b.detach().to(F16) if b is not None else
                          torch.zeros(n, device=base.weight.device, dtype=F16))
        self.qkv_bias = torch.cat(biases)

        # (first column, last column, spec) of every LoRA inside the fused q|k|v output
        self.loras = []
        col = 0
        for spec, n in zip((qa, ka, va), sizes):
            if spec is not None:
                self.loras.append((col, col + n, spec))
            col += n
        ps = {spec.p for _, _, spec in self.loras}
        if len(ps) > 1 or len(self.loras) > 2:
            raise Unsupported("LoRA dropout layout")
        self.p_drop = ps.pop() if ps else 0.0
        self.params = [t for _, _, spec in self.loras for t in (spec.A, spec.B)]

        n1, n2 = layer.input_layernorm, layer.post_attention_layernorm
        self.norm1_w, self.eps1 = n1.weight, float(n1.variance_epsilon)
        self.norm2_w, self.eps2 = n2.weight, float(n2.variance_epsilon)


def _causal_lm(model):
    get_base = getattr(model, "get_base_model", None)
    lm = get_base() if callable(get_base) else model
    if type(lm).__name__ != "Qwen2ForCausalLM":
        return None
    return lm


def _model_infos(lm):
    infos = getattr(lm, "_soup_fast_infos", None)
    if infos is not None:
        return infos
    cfg = lm.config
    if getattr(cfg, "hidden_act", "silu") != "silu":
        raise Unsupported("activation")
    if getattr(cfg, "final_logit_softcapping", None) is not None:
        raise Unsupported("logit softcapping")
    if float(getattr(cfg, "attention_dropout", 0.0) or 0.0) != 0.0:
        raise Unsupported("attention dropout")
    layer_types = getattr(cfg, "layer_types", None) or []
    if any(t != "full_attention" for t in layer_types):
        raise Unsupported("sliding-window layers")
    if len({p.device for p in lm.parameters()}) != 1:
        raise Unsupported("model spread over several devices")
    infos = [_LayerInfo(layer, cfg) for layer in lm.model.layers[: cfg.num_hidden_layers]]
    if len(infos) != len(lm.model.layers):
        raise Unsupported("layer count")
    lm._soup_fast_infos = infos
    return infos


# --------------------------------------------------------------------------- batch plan


def plan_batch(trainer, model, inputs, num_items_in_batch):
    """Everything the packed step needs about this batch, or None to take the standard path."""
    if _STATE.disabled or not getattr(model, "training", False):
        return None
    args = getattr(trainer, "args", None)
    if getattr(args, "use_liger_kernel", False) or getattr(trainer, "num_virtual_tokens", 0):
        return None
    if getattr(trainer, "aux_loss_enabled", False) or getattr(trainer, "compute_loss_func", None):
        return None
    if getattr(trainer, "label_smoother", None) is not None:
        return None
    if set(inputs) - {"input_ids", "attention_mask", "labels", "use_cache"}:
        return None
    ids, am, labels = inputs.get("input_ids"), inputs.get("attention_mask"), inputs.get("labels")
    if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.ndim == 2 for t in (ids, am, labels)):
        return None
    if not (ids.shape == am.shape == labels.shape):
        return None
    lm = _causal_lm(model)
    if lm is None or not lm.training:
        return None
    try:
        infos = _model_infos(lm)
    except Unsupported as exc:
        _warn_once(str(exc))
        return None
    warmup.wait()
    timeline.stop_sampler()

    B, S = ids.shape
    dev = ids.device
    am_b = am.bool()
    lengths = am_b.sum(1)
    right_padded = (am_b == (torch.arange(S, device=dev)[None, :] < lengths[:, None])).all()
    shift = torch.full_like(labels, -100)
    shift[:, :-1] = labels[:, 1:]
    lab_all = shift != -100
    stats = torch.cat([torch.stack([right_padded.long(), lengths.sum(), lab_all.sum(),
                                    (lab_all & am_b).sum()]), lengths.long()]).tolist()
    ok, n_real, n_lab_all, n_lab = stats[:4]
    row_lengths = [n for n in stats[4:] if n > 0]
    if not ok or n_lab_all != n_lab or n_real == 0 or n_lab == 0:
        return None

    real_idx = am_b.reshape(-1).nonzero().squeeze(1)
    tgt_real = shift.reshape(-1)[real_idx]
    is_lab = tgt_real != -100
    lab_pos = is_lab.nonzero().squeeze(1)
    unl_pos = (~is_lab).nonzero().squeeze(1)
    order = torch.cat([lab_pos, unl_pos])

    cu = torch.tensor([0] + row_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    if _STATE.varlen is None:
        _STATE.varlen = _varlen_attention_works(dev, infos[0])

    inputs_bytes = len(infos) * n_real * infos[0].hidden * 4
    per_layer = n_real * (infos[0].inter * 2 * 2 + infos[0].hidden * 4 + 4)
    n_keep = min(len(infos), max(0, ACTIVATION_BUDGET_BYTES - inputs_bytes) // per_layer)
    keep = [i >= len(infos) - n_keep for i in range(len(infos))]

    rotary = lm.model.rotary_emb
    cos, sin = rotary(torch.empty(0, device=dev, dtype=torch.float32),
                      torch.arange(S, device=dev)[None, :])

    if _STATE.rng is None:
        _STATE.rng = random.Random(torch.initial_seed())
    rng = _STATE.rng
    seeds = [tuple(rng.randrange(1, 2**31 - 1) for _ in info.loras) for info in infos]

    return SimpleNamespace(
        input_ids=ids, infos=infos, lm=lm, consumed=False,
        B=B, S=S, n_real=n_real, n_lab=n_lab,
        real_idx=real_idx.to(torch.int32),
        pad_map=inverse_map(real_idx, B * S),
        token_ids=ids.reshape(-1)[real_idx].to(torch.int32),
        order=order.to(torch.int32),
        tgt_sorted=tgt_real[order].to(torch.int32),
        lab_map=inverse_map(lab_pos, n_real),
        lab_flat_idx=real_idx[lab_pos],
        cos=cos[0].contiguous(), sin=sin[0].contiguous(),
        seeds=seeds, denom=n_lab, lora16={}, keep=keep,
        pos=(real_idx % S).to(torch.int32),
        cu=cu.to(dev, non_blocking=True), max_len=max(row_lengths),
    )


# --------------------------------------------------------------------------- decoder layer


def _qkv_forward(info, plan, h, seeds, params):
    """input_layernorm (+ LoRA dropout copies) and the fused q|k|v projection with LoRA."""
    drop_seeds = seeds if info.p_drop > 0 else ()
    x, rs1, drops = rmsnorm_drop_fwd(h, info.norm1_w, info.eps1, F16, info.p_drop, drop_seeds)
    if not drop_seeds:
        drops = [x] * len(info.loras)
    w_qkv = info.qkv.dequant()
    qkv = torch.addmm(info.qkv_bias, x, w_qkv.t())
    w16 = plan.lora16.get(id(info))
    if w16 is None:  # fp16 copies of this step's LoRA weights, shared by forward and recompute
        w16 = plan.lora16[id(info)] = [p.detach().to(F16) for p in params]
    loras = []
    for i, ((c0, c1, spec), xd) in enumerate(zip(info.loras, drops)):
        a16, b16 = w16[2 * i], w16[2 * i + 1]
        xa = torch.mm(xd, a16.t())
        qkv[:, c0:c1].addmm_(xa, b16.t(), alpha=spec.scaling)
        loras.append((c0, c1, spec.scaling, a16, b16, xd, xa))
    return x, rs1, w_qkv, qkv, loras


def _varlen_attention(q, k, v, plan, scale, need_lse):
    """Causal attention of every packed sequence at once: PyTorch's memory-efficient kernel
    with cu_seqlens (no padding, one launch). q, k, v: [1, N, H, D]."""
    out = torch.ops.aten._efficient_attention_forward(
        q, k, v, None, plan.cu, plan.cu, plan.max_len, plan.max_len, 0.0, 1, need_lse,
        scale=scale)
    return out[0]


def _varlen_attention_works(dev, info):
    """One small forward + backward against padded SDPA; False on any error or mismatch."""
    try:
        g = torch.Generator(device=dev).manual_seed(0)
        lens = [5, 9, 3]
        n, h, d = sum(lens), info.n_heads, info.head_dim
        q, k, v = (torch.randn((1, n, h, d), device=dev, dtype=F16, generator=g).requires_grad_(True)
                   for _ in range(3))
        plan = SimpleNamespace(cu=torch.tensor([0, 5, 14, 17], device=dev, dtype=torch.int32),
                               max_len=max(lens))
        with torch.enable_grad():
            out = _varlen_attention(q, k, v, plan, info.scale, True)
            grads = torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
        start = 0
        for n_i in lens:
            sl = slice(start, start + n_i)
            qs, ks, vs = (t.detach()[:, sl].transpose(1, 2).requires_grad_(True) for t in (q, k, v))
            with torch.enable_grad():
                ref = _SDPA(qs, ks, vs, is_causal=True, scale=info.scale)
                ref_g = torch.autograd.grad(ref, (qs, ks, vs), torch.ones_like(ref))
            same = torch.allclose(out[:, sl].transpose(1, 2).float(), ref.float(),
                                  atol=2e-2, rtol=2e-2)
            same = same and all(
                torch.allclose(a[:, sl].transpose(1, 2).float(), r.float(), atol=5e-2, rtol=5e-2)
                for a, r in zip(grads, ref_g))
            if not same:
                warnings.warn("solution: varlen attention disagrees with SDPA; using padded SDPA")
                return False
            start += n_i
        return True
    except Exception as exc:  # noqa: BLE001 - any failure: padded SDPA instead
        warnings.warn(f"solution: varlen attention unavailable ({type(exc).__name__}: {exc}); "
                      "using padded SDPA")
        return False


def _attention(info, plan, qkv, need_grad):
    if _STATE.varlen:
        q, k, v = qkv_rope_pack(qkv, plan.pos, plan.cos, plan.sin, info.n_heads, info.n_kv,
                                info.head_dim)
        if need_grad:
            q = q.requires_grad_(True)
            k = k.requires_grad_(True)
            v = v.requires_grad_(True)
            with torch.enable_grad():
                o = _varlen_attention(q, k, v, plan, info.scale, True)
        else:
            o = _varlen_attention(q, k, v, plan, info.scale, False)
        return o.view(plan.n_real, -1), (q, k, v, o)

    q, k, v = qkv_rope_pad(qkv, plan.pad_map, plan.cos, plan.sin, plan.B, plan.S,
                           info.n_heads, info.n_kv, info.head_dim)
    if need_grad:
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        with torch.enable_grad():
            o = _SDPA(q, k, v, is_causal=True, scale=info.scale)
    else:
        o = _SDPA(q, k, v, is_causal=True, scale=info.scale)
    o_rows = o.transpose(1, 2)
    if not o_rows.is_contiguous():
        o_rows = o_rows.contiguous()
    attn = row_gather(o_rows.view(plan.B * plan.S, -1), plan.real_idx)
    return attn, (q, k, v, o)


class _DecoderLayerFunction(torch.autograd.Function):
    """One decoder layer on packed tokens. Saves its fp32 input; with `keep` also the residual
    sum h2 and the gate|up output, so the backward does not recompute o_proj / gate|up."""

    @staticmethod
    def forward(ctx, h, info, plan, seeds, keep, *params):
        x, rs1, w_qkv, qkv, _ = _qkv_forward(info, plan, h, seeds, params)
        del x, rs1, w_qkv
        attn, _ = _attention(info, plan, qkv, need_grad=False)
        del qkv
        a = torch.mm(attn, info.o.dequant().t())
        del attn
        h2, x2, rs2 = fused_add_rmsnorm_fwd(h, a, info.norm2_w, info.eps2, F16)
        del a
        gu = torch.mm(x2, info.gu.dequant().t())
        del x2
        s = swiglu_packed_fwd(gu, info.inter)
        m = torch.mm(s, info.down.dequant().t())
        del s
        if keep:
            out = torch.add(h2, m)
            ctx.save_for_backward(h, h2, gu, rs2, *params)
        else:
            del gu
            out = h2.add_(m)
            ctx.save_for_backward(h, *params)
        ctx.info, ctx.plan, ctx.seeds, ctx.keep = info, plan, seeds, keep
        return out

    @staticmethod
    def backward(ctx, g):
        info, plan, seeds = ctx.info, ctx.plan, ctx.seeds
        if ctx.keep:
            h, h2, gu, rs2, *params = ctx.saved_tensors
        else:
            h, *params = ctx.saved_tensors
        g = g.contiguous()

        # recompute what the backward reads (down_proj's output is never one of them)
        x, rs1, w_qkv, qkv, loras = _qkv_forward(info, plan, h, seeds, params)
        attn, (q, k, v, o) = _attention(info, plan, qkv, need_grad=True)
        del qkv
        w_o = info.o.dequant()
        w_gu = info.gu.dequant()
        if not ctx.keep:
            a = torch.mm(attn, w_o.t())
            h2, x2, rs2 = fused_add_rmsnorm_fwd(h, a, info.norm2_w, info.eps2, F16)
            del a
            gu = torch.mm(x2, w_gu.t())
            del x2
        del attn

        # MLP
        ds = torch.mm(g.to(F16), info.down.dequant())
        swiglu_packed_bwd_(ds, gu, info.inter)
        del ds
        dx2 = torch.mm(gu, w_gu)
        del gu, w_gu
        d_sum = fused_add_rmsnorm_bwd_(dx2, g, h2, info.norm2_w, rs2)
        del dx2, h2

        # attention
        d_attn = torch.mm(d_sum.to(F16), w_o)
        del w_o
        if _STATE.varlen:
            dq, dk, dv = torch.autograd.grad(o, (q, k, v), d_attn.view(o.shape))
            del o, q, k, v, d_attn
            d_qkv = qkv_rope_pack_backward(dq, dk, dv, plan.pos, plan.cos, plan.sin,
                                           info.n_heads, info.n_kv, info.head_dim)
        else:
            d_o = row_gather(d_attn, plan.pad_map).view(plan.B, plan.S, info.n_heads,
                                                        info.head_dim)
            del d_attn
            dq, dk, dv = torch.autograd.grad(o, (q, k, v), d_o.transpose(1, 2))
            del o, q, k, v, d_o
            d_qkv = qkv_rope_pad_backward(dq, dk, dv, plan.real_idx, plan.cos, plan.sin, plan.S,
                                          info.n_heads, info.n_kv, info.head_dim)
        del dq, dk, dv

        # LoRA weight gradients, and the LoRA branches' gradient into the dropped inputs
        need_dh = ctx.needs_input_grad[0]
        grads, branch = [], []
        for c0, c1, scaling, a16, b16, xd, xa in loras:
            gm = d_qkv[:, c0:c1]
            d_b = torch.mm(gm.t(), xa).mul_(scaling)
            d_xa = torch.mm(gm, b16).mul_(scaling)
            d_a = torch.mm(d_xa.t(), xd)
            grads += [d_a.float(), d_b.float()]
            if need_dh:
                branch.append(torch.mm(d_xa, a16))
        del loras

        dh = None
        if need_dh:
            dx = torch.mm(d_qkv, w_qkv)
            del d_qkv, w_qkv
            if info.p_drop > 0:
                dh = rmsnorm_drop_bwd(dx, h, info.norm1_w, rs1, d_res=d_sum, p_drop=info.p_drop,
                                      seeds=seeds, branch_grads=branch)
            else:
                for t in branch:
                    dx += t
                dh = rmsnorm_drop_bwd(dx, h, info.norm1_w, rs1, d_res=d_sum)
        return (dh, None, None, None, None, *grads)


# --------------------------------------------------------------------------- model / trainer


def _head_weight(lm):
    w = lm.lm_head.weight
    return w.detach() if w.dtype == F16 else w.detach().to(F16)


def causal_lm_forward(lm, plan, num_items_in_batch=None):
    """Loss of the packed step; fills plan.stats for the trainer's metrics."""
    plan.consumed = True
    if num_items_in_batch is not None:
        plan.denom = num_items_in_batch
    model = lm.model
    with torch.autocast("cuda", enabled=False):
        h = row_gather(model.embed_tokens.weight.detach(), plan.token_ids, out_dtype=torch.float32)
        for info, seeds, keep in zip(plan.infos, plan.seeds, plan.keep):
            h = _DecoderLayerFunction.apply(h, info, plan, seeds, keep, *info.params)
        hn = TritonRMSNormFunction.apply(h, model.norm.weight, model.norm.variance_epsilon, F16)
        del h
        plan.stats = SimpleNamespace()
        loss = LMHeadLossFunction.apply(hn, _head_weight(lm), plan, plan.stats)
    return loss


def log_trl_metrics(trainer, mode, inputs, plan):
    """entropy, num_tokens and mean_token_accuracy, by TRL's formulas, from the fused pass."""
    gather = trainer.accelerator.gather_for_metrics
    metrics = trainer._metrics[mode]
    st = plan.stats
    entropy = st.ent.sum() / inputs["attention_mask"].sum()
    metrics["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += gather(inputs["attention_mask"].sum()).sum().item()
    metrics["num_tokens"] = [trainer._total_train_tokens]
    correct = gather(st.correct.sum())
    total = gather(torch.tensor(plan.n_lab, device=st.correct.device))
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    metrics["mean_token_accuracy"].append(accuracy)


def rewind_step(plan):
    """Per-token losses in the layout Soup's rewind recorder reads for a step without logits:
    [B, S] float32, position t holding the loss of label t + 1."""
    token_loss = torch.zeros(plan.B * plan.S, device=plan.stats.loss_tok.device, dtype=torch.float32)
    token_loss[plan.lab_flat_idx] = plan.stats.loss_tok
    return SimpleNamespace(token_loss=token_loss.view(plan.B, plan.S))


# Diagnostics for a few steps: where a step's wall time goes (CPU issue vs GPU wait).
DIAG_STEPS = range(3, 9)
_DIAG = SimpleNamespace(step=0, prev=None)


def _diag_report(now):
    d = _DIAG.prev
    if d is None:
        return
    _DIAG.prev = None
    period = (now - d["t0"]) * 1000
    parts = {k: v * 1000 for k, v in d.items() if k not in ("t0", "_step")}
    rest = period - sum(parts.values())
    print(f"[solution-step] step {d_step(d)}: {period:6.1f} ms = "
          + " | ".join(f"{k} {v:5.1f}" for k, v in parts.items()) + f" | rest {rest:5.1f}",
          file=sys.stderr, flush=True)


def d_step(d):
    return d.get("_step", "?")


def make_compute_loss(original):
    """TRL SFTTrainer.compute_loss with the packed step when the batch and model allow it."""
    from transformers import Trainer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        t0 = time.perf_counter()
        _diag_report(t0)
        step = _DIAG.step
        _DIAG.step += 1
        diag = step in DIAG_STEPS and torch.cuda.is_available()
        try:
            plan = plan_batch(self, model, inputs, num_items_in_batch)
        except Exception as exc:  # noqa: BLE001 - a planning failure must not stop training
            _warn_once(f"{type(exc).__name__}: {exc}")
            plan = None
        if plan is None:
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        t1 = time.perf_counter()
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _STATE.plan = plan
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            _STATE.plan = None
        if not plan.consumed:
            raise RuntimeError("solution: the packed step was planned but the model did not run it")
        t2 = time.perf_counter()
        if diag:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        log_trl_metrics(self, mode, inputs, plan)
        setattr(self, STEP_ATTR, rewind_step(plan))
        t4 = time.perf_counter()
        if diag:
            times = {"_step": step, "t0": t0, "plan": t1 - t0, "fwd issue": t2 - t1,
                     "fwd gpu": t3 - t2, "metrics": t4 - t3}
            _DIAG.prev = times
            acc = self.accelerator
            plain_backward = acc.backward

            def timed_backward(*args, **kwargs):
                b0 = time.perf_counter()
                try:
                    return plain_backward(*args, **kwargs)
                finally:
                    b1 = time.perf_counter()
                    torch.cuda.synchronize()
                    times["bwd issue"] = b1 - b0
                    times["bwd gpu"] = time.perf_counter() - b1
                    acc.backward = plain_backward

            acc.backward = timed_backward
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_fast = True
    return compute_loss


def make_lm_forward(original):
    """Qwen2ForCausalLM.forward that runs the planned packed step for the trainer's batch."""

    def forward(self, *args, **kwargs):
        plan = _STATE.plan
        if (
            plan is not None
            and not plan.consumed
            and plan.lm is self
            and not args
            and kwargs.get("input_ids") is plan.input_ids
            and kwargs.get("labels") is not None
            and kwargs.get("inputs_embeds") is None
            and kwargs.get("past_key_values") is None
            and not kwargs.get("output_hidden_states")
            and not kwargs.get("output_attentions")
        ):
            from transformers.modeling_outputs import CausalLMOutputWithPast

            loss = causal_lm_forward(self, plan, kwargs.get("num_items_in_batch"))
            return CausalLMOutputWithPast(loss=loss, logits=None)
        return original(self, *args, **kwargs)

    forward._soup_fast = True
    return forward


def make_prepare_kbit(original):
    """peft.prepare_model_for_kbit_training that leaves the frozen tied embedding / lm_head in
    the fp16 it was loaded in (the head runs in fp16 anyway, and fp32 -> fp16 of these values
    is exact), instead of a 2x larger fp32 copy. Everything else is the original."""

    def prepare(model, *args, **kwargs):
        held = None
        try:
            emb = model.get_input_embeddings().weight
            head = model.get_output_embeddings()
            if (
                type(model).__name__ == "Qwen2ForCausalLM"
                and emb.is_cuda
                and emb.dtype == F16
                and head is not None
                and head.weight is emb
            ):
                held = (emb, emb.data)
                emb.data = torch.empty(0, device=emb.device, dtype=F16)
        except Exception:  # noqa: BLE001 - anything unexpected: keep the original behaviour
            held = None
        try:
            return original(model, *args, **kwargs)
        finally:
            if held is not None:
                held[0].data = held[1]

    prepare._soup_fast = True
    return prepare
