import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from solution.rmsnorm import TritonRMSNormFunction, TritonFusedAddRMSNormFunction
from solution.cross_entropy import (
    TritonCrossEntropyFunction,
    ChunkedCrossEntropyFunction,
    _ce_forward_nograd,
)
from solution.swiglu import TritonSwiGLUFunction
from solution.rope import TritonRoPEFunction

_orig_F_cross_entropy = getattr(F, "_orig_cross_entropy", F.cross_entropy)
F._orig_cross_entropy = _orig_F_cross_entropy

_orig_ce_forward = getattr(nn.CrossEntropyLoss, "_orig_forward", nn.CrossEntropyLoss.forward)
nn.CrossEntropyLoss._orig_forward = _orig_ce_forward


def custom_cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    if (
        isinstance(input, torch.Tensor)
        and input.is_cuda
        and input.ndim in (2, 3)
        and input.shape[-1] > 10000
        and weight is None
        and size_average is None
        and reduce is None
        and label_smoothing == 0.0
        and reduction in ("mean", "sum")
    ):
        n_rows = input.shape[0] if input.ndim == 2 else input.shape[0] * input.shape[1]
        if n_rows == target.numel():
            if not torch.is_grad_enabled() or not input.requires_grad:
                input_2d = input if input.ndim == 2 else input.reshape(-1, input.shape[-1])
                target_1d = target if target.ndim == 1 else target.reshape(-1)
                return _ce_forward_nograd(input_2d, target_1d, ignore_index, reduction)
            return TritonCrossEntropyFunction.apply(
                input, target, ignore_index, reduction
            )
    return _orig_F_cross_entropy(
        input,
        target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def custom_ce_forward(self, input, target):
    reduction = getattr(self, "reduction", "mean")
    if (
        isinstance(input, torch.Tensor)
        and input.is_cuda
        and input.ndim in (2, 3)
        and input.shape[-1] > 10000
        and getattr(self, "weight", None) is None
        and getattr(self, "label_smoothing", 0.0) == 0.0
        and reduction in ("mean", "sum")
    ):
        n_rows = input.shape[0] if input.ndim == 2 else input.shape[0] * input.shape[1]
        if n_rows == target.numel():
            ignore_index = getattr(self, "ignore_index", -100)
            if not torch.is_grad_enabled() or not input.requires_grad:
                input_2d = input if input.ndim == 2 else input.reshape(-1, input.shape[-1])
                target_1d = target if target.ndim == 1 else target.reshape(-1)
                return _ce_forward_nograd(input_2d, target_1d, ignore_index, reduction)
            return TritonCrossEntropyFunction.apply(
                input, target, ignore_index, reduction
            )
    return _orig_ce_forward(self, input, target)


def custom_model_loss_function(
    logits=None,
    labels=None,
    vocab_size=None,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    **kwargs,
):
    if vocab_size is None and shift_labels is None and num_items_in_batch is None:
        return custom_cross_entropy(logits, labels, ignore_index=ignore_index)

    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Keep logits in native dtype (fp16/bf16); Triton kernel computes reduction in fp32
    logits_2d = (
        logits.view(-1, vocab_size)
        if vocab_size is not None
        else logits.reshape(-1, logits.shape[-1])
    )
    shift_labels_1d = shift_labels.view(-1)
    if shift_labels_1d.device != logits_2d.device:
        shift_labels_1d = shift_labels_1d.to(logits_2d.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = custom_cross_entropy(
        logits_2d,
        shift_labels_1d,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch) and num_items_in_batch.device != loss.device:
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def _patch_model_loss_attributes(model):
    if model is None:
        return
    visited = set()
    queue = [model]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in visited:
            continue
        visited.add(id(obj))
        if hasattr(obj, "loss_function"):
            try:
                obj.loss_function = custom_model_loss_function
            except Exception:
                pass
        if hasattr(obj, "loss_fct"):
            try:
                obj.loss_fct = custom_cross_entropy
            except Exception:
                pass
        for attr in ("model", "base_model"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in visited:
                queue.append(child)


def apply(model=None):
    try:
        from transformers.models.qwen2 import modeling_qwen2

        # 1. RMSNorm (runs on input_layernorm and model.norm)
        _orig_rmsnorm_forward = getattr(
            modeling_qwen2.Qwen2RMSNorm,
            "_orig_forward",
            modeling_qwen2.Qwen2RMSNorm.forward,
        )
        modeling_qwen2.Qwen2RMSNorm._orig_forward = _orig_rmsnorm_forward

        def custom_rmsnorm_forward(self, hidden_states):
            if not hidden_states.is_cuda:
                return _orig_rmsnorm_forward(self, hidden_states)
            return TritonRMSNormFunction.apply(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )

        modeling_qwen2.Qwen2RMSNorm.forward = custom_rmsnorm_forward

        # 2. SwiGLU MLP
        _orig_mlp_forward = getattr(
            modeling_qwen2.Qwen2MLP,
            "_orig_forward",
            modeling_qwen2.Qwen2MLP.forward,
        )
        modeling_qwen2.Qwen2MLP._orig_forward = _orig_mlp_forward

        def custom_mlp_forward(self, x):
            act = getattr(getattr(self, "config", None), "hidden_act", "silu")
            if act != "silu" or not x.is_cuda:
                return _orig_mlp_forward(self, x)
            return self.down_proj(TritonSwiGLUFunction.apply(self.gate_proj(x), self.up_proj(x)))

        modeling_qwen2.Qwen2MLP.forward = custom_mlp_forward

        # 3. Fused RoPE
        _orig_apply_rotary_pos_emb = getattr(
            modeling_qwen2,
            "_orig_apply_rotary_pos_emb",
            modeling_qwen2.apply_rotary_pos_emb,
        )
        modeling_qwen2._orig_apply_rotary_pos_emb = _orig_apply_rotary_pos_emb

        def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            if not q.is_cuda:
                return _orig_apply_rotary_pos_emb(
                    q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim
                )
            cos_u = cos.unsqueeze(unsqueeze_dim)
            sin_u = sin.unsqueeze(unsqueeze_dim)
            q_embed = TritonRoPEFunction.apply(q, cos_u, sin_u)
            k_embed = TritonRoPEFunction.apply(k, cos_u, sin_u)
            return q_embed, k_embed

        modeling_qwen2.apply_rotary_pos_emb = custom_apply_rotary_pos_emb
        if hasattr(modeling_qwen2, "Qwen2Attention") and hasattr(
            modeling_qwen2.Qwen2Attention, "forward"
        ):
            fwd_globals = getattr(modeling_qwen2.Qwen2Attention.forward, "__globals__", None)
            if isinstance(fwd_globals, dict) and "apply_rotary_pos_emb" in fwd_globals:
                fwd_globals["apply_rotary_pos_emb"] = custom_apply_rotary_pos_emb

        # 4. Fused Add + RMSNorm in Qwen2DecoderLayer
        if hasattr(modeling_qwen2, "Qwen2DecoderLayer"):
            _orig_decoder_layer_forward = getattr(
                modeling_qwen2.Qwen2DecoderLayer,
                "_orig_forward",
                modeling_qwen2.Qwen2DecoderLayer.forward,
            )
            modeling_qwen2.Qwen2DecoderLayer._orig_forward = _orig_decoder_layer_forward

            _dec_src = ""
            try:
                _dec_src = inspect.getsource(_orig_decoder_layer_forward)
            except Exception:
                pass
            _dec_returns_tuple = (
                "return outputs" in _dec_src or "outputs = (hidden_states" in _dec_src
            )

            def custom_decoder_layer_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                if not hidden_states.is_cuda or kwargs.get("output_attentions", False):
                    return _orig_decoder_layer_forward(self, hidden_states, *args, **kwargs)

                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

                attn_outputs = self.self_attn(hidden_states, *args, **kwargs)
                if isinstance(attn_outputs, tuple):
                    hidden_states = attn_outputs[0]
                    rest = attn_outputs[1:]
                else:
                    hidden_states = attn_outputs
                    rest = ()

                hidden_states, residual = TritonFusedAddRMSNormFunction.apply(
                    residual,
                    hidden_states,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.variance_epsilon,
                )
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states

                if _dec_returns_tuple:
                    outputs = (hidden_states,)
                    if kwargs.get("use_cache", False) and rest:
                        outputs += rest
                    return outputs
                return hidden_states

            modeling_qwen2.Qwen2DecoderLayer.forward = custom_decoder_layer_forward

        # 5. Patch Qwen2ForCausalLM.loss_function and transformers.loss.loss_utils to avoid fp32 logits upcast
        if hasattr(modeling_qwen2, "Qwen2ForCausalLM"):
            try:
                modeling_qwen2.Qwen2ForCausalLM.loss_function = staticmethod(
                    custom_model_loss_function
                )
            except Exception:
                pass
            if hasattr(modeling_qwen2.Qwen2ForCausalLM, "forward"):
                clm_globals = getattr(
                    modeling_qwen2.Qwen2ForCausalLM.forward, "__globals__", None
                )
                if isinstance(clm_globals, dict) and "CrossEntropyLoss" in clm_globals:
                    clm_globals["CrossEntropyLoss"] = nn.CrossEntropyLoss

        try:
            from transformers.loss import loss_utils

            loss_utils.ForCausalLMLoss = custom_model_loss_function
            if hasattr(loss_utils, "LOSS_MAPPING") and isinstance(
                loss_utils.LOSS_MAPPING, dict
            ):
                loss_utils.LOSS_MAPPING["ForCausalLM"] = custom_model_loss_function
                loss_utils.LOSS_MAPPING["ForCausalLMLoss"] = custom_model_loss_function
        except Exception:
            pass
    except Exception:
        pass

    # 6. Patch functional.cross_entropy and nn.CrossEntropyLoss.forward
    torch.nn.functional.cross_entropy = custom_cross_entropy
    F.cross_entropy = custom_cross_entropy
    nn.CrossEntropyLoss.forward = custom_ce_forward

    # 7. Patch model instance loss_function / loss_fct if provided
    if model is not None:
        _patch_model_loss_attributes(model)
