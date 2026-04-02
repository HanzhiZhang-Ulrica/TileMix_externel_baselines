# Copyright 2024 ByteDance and/or its affiliates.
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
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention


_APPLY_ROTARY_PARAMS = tuple(inspect.signature(apply_rotary_pos_emb).parameters)


def _apply_qwen2_rotary(query_states, key_states, cos, sin, position_ids=None):
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
        return self.rotary_emb(value_states, seq_len=rotary_seq_len), True
    return self.rotary_emb(hidden_states, position_ids), False


def qwen2_flex_prefill_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    del attention_mask, output_attentions, use_cache

    cache = past_key_values
    if cache is None:
        cache = past_key_value
    if cache is None:
        cache = kwargs.get("past_key_values")
    if cache is None:
        cache = kwargs.get("past_key_value")

    if position_embeddings is not None and not (
        isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
    ):
        position_embeddings = kwargs.get("position_embeddings")

    legacy_api = position_embeddings is None
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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

        position_embeddings, legacy_api = _build_position_embeddings(
            self, hidden_states, value_states, position_ids, kv_seq_len
        )

    cos, sin = position_embeddings
    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

    query_states, key_states = _apply_qwen2_rotary(
        query_states, key_states, cos, sin, position_ids
    )

    if cache is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        key_states, value_states = cache.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
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

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    if legacy_api:
        return attn_output, None, cache
    return attn_output, None
