###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import math
import torch
from torch import nn
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
from tpp_pytorch_extension._C import _fused_llm_infer as fused_llm_cpp
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import transformers

from .llm_common import (
    BlockedLinear,
    BlockedLayerNorm,
    FixLinear,
    ShardLinear,
    get_rank,
    get_size,
    set_pg,
    _reorder_cache,
    block,
    compare,
    global_layer_dtype,
    get_layer_past_and_offset,
)

class LlamaDecoderLayer(BlockedModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # print("Attention mask shape: ", attention_mask.shape)

        if not hasattr(self, "cpp_block"):
            raise
        orig_hidden_states = hidden_states
        S = hidden_states.size(-2)
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.features_block_size])
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        discrete_kv = getattr(self, "discrete_kv", True)
        past_key_value, offset = get_layer_past_and_offset(past_key_value, discrete_kv)
        
        add_tensor_or_empty(attention_mask)
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(offset, offset+seq_len).repeat(hidden_states.shape[0], 1)

        add_tensor_or_empty(position_ids)
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        
        past_key_value = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in past_key_value
        ]

        outputs = self.cpp_block.forward(inputs, past_key_value, use_cache)
        hs = outputs[0]
        # hs = BlockedTensor(hs, self.blocked_input_signature, orig_hidden_states.dtype)

        present_key_value = tuple(outputs[1:])
        outputs = (hs,)
        if output_attentions:
            print("This feature is not available yet")
            # outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        """

        return outputs

def FixLlamaDecoderLayer(self, bk=None, bc=None, layer_dtype=global_layer_dtype):
    if not isinstance(self, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
        return
    self.__class__ = LlamaDecoderLayer
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        ShardLinear(self.self_attn.q_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.k_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.v_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.o_proj, 1, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.mlp.gate_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.mlp.up_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.mlp.down_proj, 1, rank, wsize, self.self_attn.head_dim)
        self.model_parallel = True
    else:
        self.model_parallel = False
    for m in self.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta: continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(torch.empty_like(m._parameters[name], device='cpu'), **kwargs)

        if isinstance(m, torch.nn.Linear):
            FixLinear(m, bk, bc, layer_dtype)
    block(self)

    if not hasattr(self, "cpp_block"):
        params = [self.input_layernorm.weight]
        params += [
            self.self_attn.q_proj.weight,
            self.self_attn.k_proj.weight,
            self.self_attn.v_proj.weight,
            self.self_attn.o_proj.weight,
        ]
        params += [self.post_attention_layernorm.weight]
        params += [self.mlp.gate_proj.weight,
                   self.mlp.up_proj.weight,
                   self.mlp.down_proj.weight,
                  ]       # since bias is False
        
        if hasattr(self.self_attn, "rotary_emb"):
            embed_positions = torch.cat(self.self_attn.rotary_emb(self.self_attn.q_proj.weight, self.self_attn.max_position_embeddings), dim=0).view([2, self.self_attn.max_position_embeddings, -1]).to(torch.float)
        else:
            raise NotImplementedError("Requires self.self_attn.rotary_emb")
        params += [embed_positions]

        # self.cpp_block = fused_llama_cpp.LlamaDecoderLayer(
        self.cpp_block = torch.classes.tpp_llm.LlamaDecoderLayer(
            params,
            self.input_layernorm.variance_epsilon,
            self.self_attn.head_dim,
            self.self_attn.max_position_embeddings,
            self.self_attn.head_dim,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")
        
def OptimizeModelForLlama(model, dtype, device='cpu'):
    set_pg()

    for m in model.modules():
        if isinstance(m, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
            FixLlamaDecoderLayer(m, 64, 64, dtype)
        elif isinstance(m, torch.nn.Linear):
            FixLinear(m, 100, 64, dtype, parallel_dim=0)
            block(m)
    for m in model.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta: continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(torch.empty_like(m._parameters[name], device=device), **kwargs)

def UpdateLLamaModel(model):
    for m in model.modules():
        if isinstance(m, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
            m.cpp_block.update_sparse_copy()

transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache = staticmethod(_reorder_cache)

