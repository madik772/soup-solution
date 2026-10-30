import torch
import torch.nn as nn
import torch.nn.functional as F
from solution.rmsnorm import TritonRMSNormFunction
from solution.cross_entropy import TritonCrossEntropyFunction
from solution.swiglu import TritonSwiGLUFunction

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
        input_2d = input.reshape(-1, input.shape[-1])
        target_1d = target.reshape(-1)
        if input_2d.shape[0] == target_1d.shape[0]:
            loss = TritonCrossEntropyFunction.apply(input_2d, target_1d, ignore_index)
            if reduction == "sum":
                V = input_2d.shape[-1]
                valid_mask = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
                loss = loss * int(valid_mask.sum().item())
            return loss
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
    if (
        isinstance(input, torch.Tensor)
        and input.is_cuda
        and input.ndim in (2, 3)
        and input.shape[-1] > 10000
        and getattr(self, "weight", None) is None
        and getattr(self, "label_smoothing", 0.0) == 0.0
        and getattr(self, "reduction", "mean") in ("mean", "sum")
    ):
        input_2d = input.reshape(-1, input.shape[-1])
        target_1d = target.reshape(-1)
        if input_2d.shape[0] == target_1d.shape[0]:
            ignore_index = getattr(self, "ignore_index", -100)
            loss = TritonCrossEntropyFunction.apply(input_2d, target_1d, ignore_index)
            if getattr(self, "reduction", "mean") == "sum":
                V = input_2d.shape[-1]
                valid_mask = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
                loss = loss * int(valid_mask.sum().item())
            return loss
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

    logits = logits.float()
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_2d = (
        logits.view(-1, vocab_size)
        if vocab_size is not None
        else logits.reshape(-1, logits.shape[-1])
    )
    shift_labels_1d = shift_labels.view(-1).to(logits_2d.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = custom_cross_entropy(
        logits_2d,
        shift_labels_1d,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
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
    # 1. Подменяем Qwen2RMSNorm.forward и Qwen2MLP.forward на кастомные Triton-функции
    try:
        from transformers.models.qwen2 import modeling_qwen2

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
    except Exception:
        pass

    # 2. Активируем подмену functional.cross_entropy и nn.CrossEntropyLoss.forward
    torch.nn.functional.cross_entropy = custom_cross_entropy
    F.cross_entropy = custom_cross_entropy
    nn.CrossEntropyLoss.forward = custom_ce_forward

    # 3. Если передан model, обновляем атрибуты loss_function / loss_fct
    if model is not None:
        _patch_model_loss_attributes(model)
