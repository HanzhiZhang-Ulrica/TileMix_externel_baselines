# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 ByteDance and/or its affiliates.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Optional, Tuple

import torch
import triton
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, logger

from flex_prefill.modules.llama.apply_rope import triton_apply_rotary_pos_emb
from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention


_APPLY_ROTARY_PARAMS = tuple(inspect.signature(apply_rotary_pos_emb).parameters)


def _apply_llama_rotary(query_states, key_states, cos, sin, position_ids=None):
    use_triton_rope = (
        triton.__version__ == "3.0.0"
        and query_states.dtype == torch.bfloat16
        and key_states.dtype == torch.bfloat16
    )
    if use_triton_rope:
        return triton_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if "position_ids" in _APPLY_ROTARY_PARAMS:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _rotary_uses_seq_len(rotary_emb) -> bool:
    try:
        params = tuple(inspect.signature(rotary_emb.forward).parameters)
    except (TypeError, ValueError):
        return False
    return "seq_len" in params


def _build_position_embeddings(
    self,
    hidden_states: torch.Tensor,
    value_states: torch.Tensor,
    position_ids: torch.LongTensor,
    kv_seq_len: int,
):
    if _rotary_uses_seq_len(self.rotary_emb):
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        return self.rotary_emb(value_states, seq_len=rotary_seq_len)
    try:
        return self.rotary_emb(hidden_states, position_ids)
    except (TypeError, RuntimeError, ValueError):
        return self.rotary_emb(value_states, position_ids)


def llama_flex_prefill_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.45
    past_key_values: Optional[Cache] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    cache = past_key_values
    if cache is None:
        cache = past_key_value
    if cache is None:
        cache = kwargs.get("past_key_values")
    if cache is None:
        cache = kwargs.get("past_key_value")

    if isinstance(cache, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    del attention_mask, output_attentions, use_cache

    if position_embeddings is not None and not (
        isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
    ):
        position_embeddings = kwargs.get("position_embeddings")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if cache is not None and self.layer_idx is not None:
        if hasattr(cache, "get_usable_length"):
            kv_seq_len += cache.get_usable_length(kv_seq_len, self.layer_idx)
        elif hasattr(cache, "get_seq_length"):
            try:
                kv_seq_len += cache.get_seq_length(self.layer_idx)
            except TypeError:
                kv_seq_len += cache.get_seq_length()

    if position_embeddings is None:
        position_ids = position_ids if position_ids is not None else kwargs.get("position_ids")
        if position_ids is None:
            if cache_position is None:
                cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)

        if position_ids.device != hidden_states.device:
            position_ids = position_ids.to(hidden_states.device)
        if cache_position is not None and cache_position.device != hidden_states.device:
            cache_position = cache_position.to(hidden_states.device)

        position_embeddings = _build_position_embeddings(
            self, hidden_states, value_states, position_ids, kv_seq_len
        )
    else:
        if position_ids is None:
            position_ids = kwargs.get("position_ids")
        if position_ids is not None and position_ids.device != hidden_states.device:
            position_ids = position_ids.to(hidden_states.device)
        if cache_position is not None and cache_position.device != hidden_states.device:
            cache_position = cache_position.to(hidden_states.device)

    cos, sin = position_embeddings
    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
    if position_ids is not None and position_ids.device != query_states.device:
        position_ids = position_ids.to(query_states.device)
    if cache_position is not None and cache_position.device != query_states.device:
        cache_position = cache_position.to(query_states.device)

    query_states, key_states = _apply_llama_rotary(
        query_states, key_states, cos, sin, position_ids
    )

    if cache is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        key_states, value_states = cache.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = flex_prefill_attention(
        query_states,
        key_states,
        value_states,
        gamma=self.config.flex_prefill_gamma,
        tau=self.config.flex_prefill_tau,
        block_size=self.config.block_size,
        min_budget=getattr(self.config, "flex_prefill_min_budget", None),
        max_budget=getattr(self.config, "flex_prefill_max_budget", None),
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, cache
