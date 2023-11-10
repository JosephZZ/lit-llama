import torch
import torch.nn as nn
import torch.nn.functional as F

#combines both lora method and adapter V2 method that adds bias and scale
def enable_lora(module, config):
    for layer in module.modules():
        if isinstance(layer, nn.Linear) and (layer.weight.shape[0] != config.lora_r or layer.weight.shape[1] != config.lora_r):
            if hasattr(layer, 'is_lora') and layer.is_lora:
                continue
            try:
                setattr(layer, 'forward', layer.custom_forward)
            except:
                init_lora_for_all(layer, config)
            layer.forward_method_type = "custom"

def init_lora_for_all(layer, config):
    layer.lora_linear1 = nn.Linear(layer.weight.shape[1], config.lora_r, bias=False)
    layer.lora_linear1.is_lora = True
    layer.lora_linear2 = nn.Linear(config.lora_r, layer.weight.shape[0], bias=False)
    layer.lora_linear2.is_lora = True
    
    layer.adapter_bias = torch.nn.Parameter(torch.zeros(layer.weight.shape[0]), requires_grad=True)
    layer.adapter_scale = torch.nn.Parameter(torch.ones(layer.weight.shape[0]), requires_grad=True)

    layer.original_forward_method = layer.forward

    layer.lora_alpha = config.lora_alpha
    layer.lora_r = config.lora_r
    layer.lora_dropout = nn.Dropout(config.lora_dropout)

    bound_method = lora_new_forward.__get__(layer, layer.__class__)
    layer.custom_forward = bound_method
    setattr(layer, 'forward', bound_method)

def lora_new_forward(self, input: torch.Tensor) -> torch.Tensor:
    x = F.linear(input, self.weight, self.bias)  \
            + (self.lora_alpha/self.lora_r)* self.lora_dropout(self.lora_linear2(self.lora_linear1(input)))
    return self.adapter_scale * (
       x + self.adapter_bias
    )

def mark_only_lora_as_trainable(model) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name or "adapter" in name or "gating_factor" in name or "aligner" in name or "generator" in name
        if (param.requires_grad):
                print(name, param.size())

def mark_lm_head_as_trainable(model) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "lm_head" in name
        if (param.requires_grad):
                print(name, param.size())

def lora_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if  "lora" in name or "adapter" in name or "gating_factor" in name or "aligner" in name or "generator" in name}

def lora_and_lm_head_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if  "lora" in name or "adapter" in name or "gating_factor" in name or "aligner" in name or "generator" in name or "lm_head" in name}