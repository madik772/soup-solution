from solution.rmsnorm import TritonRMSNormFunction

def apply(model=None):
    from transformers.models.qwen2 import modeling_qwen2

    def custom_rmsnorm_forward(self, hidden_states):
        return TritonRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )

    # Подменяем forward во всех слоях нормализации Qwen2
    modeling_qwen2.Qwen2RMSNorm.forward = custom_rmsnorm_forward
