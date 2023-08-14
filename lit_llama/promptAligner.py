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

import torch
import torch.nn as nn
from torch.nn import functional as F

import lit_llama.model as llama
from lit_llama.model import build_rope_cache, apply_rope, RMSNorm, MLP, KVCache, RoPECache


@dataclass
class LLaMAConfig(llama.LLaMAConfig):
    #used for aligner only model, compatible with adapter, not used for aligner generator model
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2

    #used for aligner generator model
    aligner_length: int = 1
    aligner_start_layer: int = 2

    aligner_generator_length: int = 10
    aligner_generator_start_layer: int = 2

class CausalSelfAttention(nn.Module):
    """A modification of `lit_llama.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)


        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.block_idx = block_idx
        self.aligner_generator_length = config.aligner_generator_length
        self.aligner_generator_start_layer = config.aligner_generator_start_layer
        self.aligner_length = config.aligner_length
        self.aligner_start_layer = config.aligner_start_layer

        self.current_model_mode = None # "aligner" or "aligner_generator"

        #initialize aligner generator embedding
        if block_idx >= config.aligner_generator_start_layer:
            self.aligner_generator_embedding = nn.Embedding(self.aligner_generator_length, self.n_embd)
            self.gating_factor_aligner_generator = torch.nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        #initialize aligner embedding place holder (for ease of fabric handling dtype to be bfloat16 or something)
        #not gonna use it during training since all aligner embedding should be from aligner generator
        if block_idx >= config.aligner_start_layer:
            self.gating_factor_aligner = torch.nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
            self.aligner_embedding = None #to be fed


    def _set_model_mode(self, is_aligner : bool, aligner_embedding:torch.tensor=None):
        if is_aligner:
            self.current_model_mode = "aligner"
            self.aligner_embedding = aligner_embedding
            self.adapter_start_layer = self.aligner_start_layer
        else:
            self.current_model_mode = "aligner_generator"
            self.adapter_start_layer = self.aligner_generator_start_layer

    def _get_current_adapter_params(self):
        if self.current_model_mode == "aligner":
            return self.aligner_embedding, self.gating_factor_aligner, self.aligner_length
        elif self.current_model_mode == "aligner_generator":
            return self.aligner_generator_embedding, self.gating_factor_aligner_generator, self.aligner_generator_length

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        # notation:
        # - B  | batch
        # - T  | time-step (sequence length)
        # - C  | embeddings size (n_embd) = head size * num heads
        # - hs | head size
        # - nh | number of heads

        B, T, C = x.size()

        # instead of calculating `query`, `key` and `value` by separately multiplying input `x` with corresponding
        # weight matrices do it (for all heads) in a single multiplication with a matrix of 3x size (concatenated
        # weights for q, k, v) and then split the result along `embedding size` dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, 3 * C) --> 3 * (B, T, C)

        # in order to move head_size (hs) dimension right after batch (B) dimension, we need to first split
        # embedding size (C) dimension into num_heads (nh) and head_size (hs)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        # "Unlike standard positional embeddings rotary embeddings must be applied at every layer"
        q = apply_rope(q, rope) # (B, T, nh, hs)
        k = apply_rope(k, rope) # (B, T, nh, hs)

        # now `key`, 'query` and `value` tensors are correctly represented: for each element in a batch (B)
        # there is a number of heads (nh) and for each head there is a sequence of elements (T), each of them is
        # represented by a vector of size `hs`
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache # 2 * (B, nh, max_seq_length, hs)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                # if we reached token limit and thus there is no space to put newly calculated `key` and `value`
                # right next to cached ones, we need to rotate cache tensor along `max_seq_length` dimension by one
                # element to the left: this will free up space for new `key` and `value`
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k) # (B, nh, max_seq_length, hs)
            v = cache_v.index_copy(2, input_pos, v) # (B, nh, max_seq_length, hs)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) # (B, nh, T, hs)

        # "Adapters are applied to the topmost layers to better tune the language
        # representations with higher-level semantics".
        if self.block_idx >= self.adapter_start_layer:
            adapter_embedding, gating_factor, adapter_length = self._get_current_adapter_params()

            if adapter_kv_cache is not None:
                ak, av = adapter_kv_cache # 2 * (B, nh, aT, hs)
            else:
                if isinstance(adapter_embedding, nn.Embedding):
                    prefix = adapter_embedding.weight.reshape(1, adapter_length, self.n_embd)
                elif isinstance(adapter_embedding, torch.Tensor):
                    prefix = adapter_embedding.reshape(1, adapter_length, self.n_embd)
                else:
                    raise ValueError(f"Unsupported type of adapter_wte: {type(adapter_embedding)}")

                aT = prefix.size(1)
                _, ak, av = self.c_attn(prefix).split(self.n_embd, dim=2) # (1, aT, 3 * C) --> 3 * (1, aT, C)
                ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) # (B, nh, aT, hs)
                av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) # (B, nh, aT, hs)
                adapter_kv_cache = (ak, av)

            # Apply cross-attention with `query`, `adapter_key`, `adapter_value` and sum the output with the output
            # obtained from self-attention step. This is mathematically equivalent to concatenation of prefix and input as per paper.
            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device) # (T, aT)
            # ↓ (B, nh, T, hs) @ (B, nh, aT, hs).mT --> (B, nh, T, aT) @ (B, nh, aT, hs) --> (B, nh, T, hs)
            # TODO: add a network that adapts the k and v of value_embedding based on the input q
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False) # (B, nh, T, hs)
            y = y + gating_factor * ay

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y) # (B, T, C)

        return y, kv_cache, adapter_kv_cache

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """For backward compatibility with old checkpoints that have a single gating value for all heads."""
        name = prefix + "gating_factor"
        if name in state_dict:
            tensor = state_dict[name]
            # in case we are loading with `utils.lazy_load()`
            tensor = tensor._load_tensor() if hasattr(tensor, "_load_tensor") else tensor

            if len(tensor.shape) < 4:
                # For old checkpoints with unified gating value
                state_dict[name] = tensor.reshape(1, 1, 1, 1).repeat(1, self.n_head, 1, 1)
            else:
                state_dict[name] = tensor

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: LLaMAConfig, block_idx: int, aligner_embedding=None) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        h, new_kv_cache, new_adapter_kv_cache = self.attn(
            self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache, adapter_kv_cache
        )
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache, new_adapter_kv_cache


class LLaMA(llama.LLaMA):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig, aligner_embedding=None) -> None:
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # if aligner_embedding is None:
        #     aligner_embedding = nn.Embedding(config.vocab_size, config.n_embd) #will never use this initialized one, but put it here to let fabric be able to set the dtype to be bfloat16 or something

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i, aligner_embedding) for i in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )# leave the aligner_embedding to be None
        
        self.rope_cache_aligner: Optional[RoPECache] = None
        self.mask_cache_aligner: Optional[torch.Tensor] = None
        self.kv_caches_aligner: List[KVCache] = []
        self.adapter_kv_caches_aligner: List[KVCache] = []

        self.rope_cache_aligner_generator: Optional[RoPECache] = None
        self.mask_cache_aligner_generator: Optional[torch.Tensor] = None
        self.kv_caches_aligner_generator: List[KVCache] = []
        self.adapter_kv_caches_aligner_generator: List[KVCache] = []

        self.current_model_mode = None # "aligner" or "aligner_generator"
        self.aligner_embedding = aligner_embedding
        self.aligner_generator_head_MLP = MLP(config)
        self.rms_generator = RMSNorm(config.n_embd)

    def set_model_mode(self, is_aligner:bool, aligner_embedding:torch.Tensor=None):
        if is_aligner:
            self.aligner_embedding = aligner_embedding
            for block in self.transformer.h:
                block.attn._set_model_mode(is_aligner, aligner_embedding)
            self.current_model_mode = "aligner"
        else:
            for block in self.transformer.h:
                block.attn._set_model_mode(is_aligner)
            self.current_model_mode = "aligner_generator"


    def get_current_caches(self):
        if self.current_model_mode == "aligner":
            return self.rope_cache_aligner, self.mask_cache_aligner, self.kv_caches_aligner, self.adapter_kv_caches_aligner
        else:
            return self.rope_cache_aligner_generator, self.mask_cache_aligner_generator, self.kv_caches_aligner_generator, self.adapter_kv_caches_aligner_generator


    @classmethod
    def from_name(cls, name: str, aligner_length: int = 1, aligner_start_layer: int = 2, \
                  aligner_generator_length: int=10, aligner_generator_start_layer: int = 2) -> "LLaMA":
        config = LLaMAConfig.from_name(name)
        config.aligner_length = aligner_length
        config.aligner_start_layer = aligner_start_layer
        config.aligner_generator_length = aligner_generator_length
        config.aligner_generator_start_layer = aligner_generator_start_layer
        return cls(config)

    def reset_cache(self) -> None:
        super().reset_cache()
        self.adapter_kv_caches_aligner.clear()
        self.adapter_kv_caches_aligner_generator.clear()

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None \
        
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"


        rope_cache, mask_cache, kv_caches, adapter_kv_caches = self.get_current_caches()

        if rope_cache is None:
            rope_cache = self.build_rope_cache(idx) # (block_size, head_size / 2, 2)
        if mask_cache is None:
            mask_cache = self.build_mask_cache(idx) # (1, 1, block_size, block_size)

        if input_pos is not None:
            rope = rope_cache.index_select(0, input_pos)
            mask = mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = rope_cache[:T]
            mask = mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, *_ = block(x, rope, mask, max_seq_length)
        else:
            if not kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            if not adapter_kv_caches:
                adapter_kv_caches = [None for _ in range(self.config.n_layer)]
            for i, block in enumerate(self.transformer.h):
                x, kv_caches[i], adapter_kv_caches[i] = block(
                    x, rope, mask, max_seq_length, input_pos, kv_caches[i], adapter_kv_caches[i]
                )


        results = None
        if self.current_model_mode == "aligner":
            #if aligner already given (inference mode), we use it to generate normal text output
            x = self.transformer.ln_f(x) # (B, T, n_embd) # RMS normalization
            logits = self.lm_head(x)  # (B, T, vocab_size)
            results = logits
        else:
            #if aligner not given (at generator mode), we use the last layer output as aligner embedding
            generated_align_embedding = x[:,-1,:] #take the last output to be our aligner embedding, assuming batch size 1
            generated_align_embedding = generated_align_embedding + self.aligner_generator_head_MLP(self.rms_generator(generated_align_embedding))#further process the aligner embedding with MLP
            results = generated_align_embedding

        return results


def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad =  "gating_factor" in name or "aligner" in name or "generator" in name
        if (param.requires_grad):
            try:
                print(name, param.size())
            except:
                print(name)

def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if"gating_factor" in name or "aligner" in name or "generator" in name}
