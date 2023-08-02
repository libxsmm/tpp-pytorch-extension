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
from typing import Optional, Tuple, Union
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
from transformers.modeling_outputs import CausalLMOutputWithPast

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


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


class GPTJBlock(BlockedModule):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        #print("HS:", hidden_states.shape, hidden_states.device, hidden_states.dtype)
        # print("layer_past:", layer_past[0].shape if layer_past is not None else layer_past)
        # print("attention_mask:", attention_mask.shape if attention_mask is not None else attention_mask)
        # print("position_ids:", position_ids.shape if position_ids is not None else position_ids)
        if not hasattr(self, "cpp_block"):
            raise
        orig_hidden_states = hidden_states
        S = hidden_states.size(-2)
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.features_block_size])
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)
        dummy_tensor_int = torch.Tensor().to(torch.long)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        add_tensor_or_empty(attention_mask)
        discrete_kv = getattr(self, "discrete_kv", True)
        layer_past, offset = get_layer_past_and_offset(layer_past, discrete_kv)
        # print("position_ids:", position_ids)
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(offset, offset+seq_len).repeat(hidden_states.shape[0], 1)
        add_tensor_or_empty(position_ids)
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        #print("AM: ", inputs[-2].shape, inputs[-2].dtype)

        # print("PHS:", hidden_states.shape)
        layer_past = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in layer_past
        ]

        #print(f"attention_mask: {attention_mask.shape} {attention_mask}")
        outputs = self.cpp_block.forward(inputs, layer_past, use_cache)
        hs = outputs[0]
        # print("hs: ", hs.sum().item(), hs.shape)
        # print("HS: ", hs[:2,:2,:2])
        present = tuple(outputs[1:])
        #print("K: ", k.shape)
        #print("V: ", v.shape)

        hs = BlockedTensor(hs, self.blocked_input_signature, orig_hidden_states.dtype)
        # k = BlockedTensor(k, self.blocked_input_signature).unblocked_tensor()
        # v = BlockedTensor(v, self.blocked_input_signature).unblocked_tensor()

        if use_cache:
            outputs = (hs, present)
        else:
            outputs = (hs,)

        return outputs  # hidden_states, present, (attentions)

def FixGPTJBlock(self, bk=None, bc=None, layer_dtype=global_layer_dtype):
    if not isinstance(self, transformers.models.gptj.modeling_gptj.GPTJBlock):
        return
    self.__class__ = GPTJBlock
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        ShardLinear(self.attn.q_proj, 0, rank, wsize, self.attn.head_dim)
        ShardLinear(self.attn.k_proj, 0, rank, wsize, self.attn.head_dim)
        ShardLinear(self.attn.v_proj, 0, rank, wsize, self.attn.head_dim)
        ShardLinear(self.attn.out_proj, 1, rank, wsize, self.attn.head_dim)
        ShardLinear(self.mlp.fc_in, 0, rank, wsize, self.attn.head_dim)
        ShardLinear(self.mlp.fc_out, 1, rank, wsize, self.attn.head_dim)
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
        params = [self.ln_1.weight, self.ln_1.bias]
        params += [
            self.attn.q_proj.weight,
            self.attn.k_proj.weight,
            self.attn.v_proj.weight,
            self.attn.out_proj.weight,
        ]
        params += [self.mlp.fc_in.weight, self.mlp.fc_in.bias]
        params += [self.mlp.fc_out.weight, self.mlp.fc_out.bias]
        if hasattr(self.attn, "embed_positions"):
            embed_positions = self.attn.embed_positions
        else:
            max_positions = self.attn.bias.size(-1)
            pos_embd_dim = self.attn.rotary_dim or self.attn.embed_dim
            embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        params += [embed_positions]
        self.cpp_block = torch.classes.tpp_llm.GPTJBlock(
            params,
            self.ln_1.eps,
            self.attn.head_dim,
            self.attn.bias.size(-1),
            self.attn.rotary_dim,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")

def OptimizeModelForGPTJ(model, dtype, device='cpu'):
    set_pg()

    for m in model.modules():
        if isinstance(m, transformers.models.gptj.modeling_gptj.GPTJBlock):
            FixGPTJBlock(m, 16, 64, dtype)
        elif isinstance(m, torch.nn.Linear):
            FixLinear(m, 100, 64, dtype, parallel_dim=0)
            block(m)
    for m in model.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta: continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(torch.empty_like(m._parameters[name], device=device), **kwargs)


transformers.models.gptj.modeling_gptj.GPTJForCausalLM._reorder_cache = staticmethod(_reorder_cache)

GPTJForCausalLM_forward = transformers.models.gptj.modeling_gptj.GPTJForCausalLM.forward

def GPTJForCausalLM_forward_patched(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return GPTJForCausalLM_forward(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
    )

transformers.models.gptj.modeling_gptj.GPTJForCausalLM.forward = GPTJForCausalLM_forward_patched
