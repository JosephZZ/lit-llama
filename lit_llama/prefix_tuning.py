"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

                                                                             |              Prefix cross-attention
                                                                             |
  ┌─────────────────┐                                                        |               ┌──────────────────┐
  ┆        x        ┆                                                        |               ┆      prefix      ┆
  └─────────────────┘                                                        |               └──────────────────┘
           |                                                                 |                        |
           ▼                                                                 |                        ▼
  ┌──────────────────┐                                                       |              ┌─────────────────────┐
  ┆  self-attention  ┆ --------------------------------------------------------------┐      ┆  linear projection  ┆
  └──────────────────┘                                                       |       ┆      └─────────────────────┘
           |                                                                 |       ┆                |         \
           ▼                                                                 |       ▼                ▼          ▼
         ╭───╮     ┌────────────────┐ ╭───╮ ┌──────────────────────────┐     |  ┌─────────┐    ┌──────────────┐  ┌────────────────┐
         ┆ + ┆ ◀── ┆  gating factor ┆-┆ x ┆-┆  prefix cross-attention  ┆     |  ┆  query  ┆    ┆  prefix key  ┆  ┆  prefix value  ┆
         ╰───╯     └────────────────┘ ╰───╯ └──────────────────────────┘     |  └─────────┘    └──────────────┘  └────────────────┘
           |                                                                 |          \             |           /
           ▼                                                                 |           ▼            ▼          ▼
                                                                             |         ┌────────────────────────────────┐
                                                                             |         ┆  scaled dot-product attention  ┆
                                                                             |         └────────────────────────────────┘


In order to inject learnable information from the prefix to pretrained weights we need to sum outputs from
self-attention and prefix cross-attention (times gating factor). For prefix cross-attention we need `query` (from
self-attention as a result of linear projection), `prefix key` and `prefix value` (from cross-attention as a result of
linear projection).
The output of prefix cross-attention is multiplied by gating factor, which is a learnable parameter that is needed to
avoid potential disruption of pretrained weights caused by incorporating randomly initialized tensors. This factor is
initialized with zeros to avoid noise from the adaption prompts at the early training stage.
More about it: 

Notes about implementation: as per paper adapter's prefix is concatenated with the input, while here outputs of
self-attention and prefix cross-attention are summed. Both variants are mathematically equivalent:
https://github.com/ZrrSkywalker/LLaMA-Adapter/issues/47
"""
# mypy: ignore-errors
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import lit_llama.model as llama
from lit_llama.model import build_rope_cache, apply_rope, Block, RMSNorm, MLP, KVCache, RoPECache, MaskCache


@dataclass
class LLaMAConfig(llama.LLaMAConfig):
    prefix_length: int = 1


class LLaMA(llama.LLaMA):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__(config)
        # assert config.padded_vocab_size is not None
        # self.config = config

        # self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        # self.transformer = nn.ModuleDict(
        #     dict(
        #         wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
        #         h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
        #         ln_f=RMSNorm(config.n_embd),
        #     )
        # )

        # self.rope_cache: Optional[RoPECache] = None
        # self.mask_cache: Optional[MaskCache] = None
        # self.kv_caches: List[KVCache] = []

        #initialize prefix embedding parameters of size (prefix_length, n_embd)
        self.prefix_embd = nn.Parameter(torch.randn(config.prefix_length, config.n_embd) / config.n_embd ** 0.5)
    
    def init_prefix_embd_with_token(self):
        # init as token correspond to "summary" for config.prefix_length times to be shape [prefix_length, n_embd] using repeat_interleave
        init_token = self.transformer.wte(torch.tensor([15387],device=self.prefix_embd.device)).data.detach().clone()
        init_token = torch.repeat_interleave(init_token, self.config.prefix_length, dim=0)
        self.prefix_embd.data = init_token
    
    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()
        T = T + self.config.prefix_length

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            input_pos = torch.arange(0, T, device=input_pos.device)
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # prepend the prefix embeddings to the input
        prefix_embd = self.prefix_embd.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([prefix_embd, x], dim=1)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        # remove the prefix embeddings from the output
        x = x[:, self.config.prefix_length:]
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
    
    @classmethod
    def from_name(cls, name: str, prefix_length: int) -> "LLaMA":
        config = LLaMAConfig.from_name(name)
        config.prefix_length = prefix_length
        return cls(config)

def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "prefix" in name
        if param.requires_grad:
            print(f"Training {name}")

def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if "prefix" in name }

