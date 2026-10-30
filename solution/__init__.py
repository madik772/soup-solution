import torch.nn as nn
from solution.rmsnorm import TritonRMSNormFunction
from solution.cross_entropy import TritonCrossEntropyFunction

_orig_ce_forward = nn.CrossEntropyLoss.forward

def custom_ce_forward(self, input, target):
    # Применяем кастомный Triton-кернел для большого словаря модели
    if (
        input.is_cuda 
        and input.ndim == 2 
        and input.shape[-1] > 10000 
        and self.weight is None 
        and getattr(self, "label_smoothing", 0.0) == 0.0
        and self.reduction == "mean"
    ):
        return TritonCrossEntropyFunction.apply(input, target, self.ignore_index)
    return _orig_ce_forward(self, input, target)

def apply(model=None):
    from transformers.models.qwen2 import modeling_qwen2

    # 1. Подменяем нормализацию
    def custom_rmsnorm_forward(self, hidden_states):
        return TritonRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )
    modeling_qwen2.Qwen2RMSNorm.forward = custom_rmsnorm_forward

    # 2. Подменяем расчет CrossEntropyLoss
    nn.CrossEntropyLoss.forward = custom_ce_forward
