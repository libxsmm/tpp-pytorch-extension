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
from typing import Optional, List, Tuple, Union
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


class OPTDecoderLayer(BlockedModule):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ]:
        # print("HS:", hidden_states.shape, hidden_states.device, hidden_states.dtype)
        # print("layer_past:", layer_past[0].shape if layer_past is not None else layer_past)
        # print("attention_mask:", attention_mask.shape if attention_mask is not None else attention_mask)
        # print("position_ids:", position_ids.shape if position_ids is not None else position_ids)
        if not hasattr(self, "cpp_block"):
            raise
        orig_hidden_states = hidden_states
        S = hidden_states.size(-2)
        hidden_states = self.get_blocked_tensor(
            hidden_states,
            self.blocked_input_signature,
            [None, None, self.features_block_size],
        )
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        discrete_kv = getattr(self, "discrete_kv", False)
        past_key_value, offset = get_layer_past_and_offset(past_key_value, discrete_kv)
        add_tensor_or_empty(attention_mask)

        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        # print("AM: ", inputs[-2].shape, inputs[-2].dtype)

        # print("PHS:", hidden_states.shape)
        past_key_value = [
            i.to(self.layer_dtype) if i.is_floating_point() else i
            for i in past_key_value
        ]

        outputs = self.cpp_block.forward(inputs, past_key_value, use_cache)
        hs = outputs[0]
        present = tuple(outputs[1:])

        hs = BlockedTensor(hs, self.blocked_input_signature, orig_hidden_states.dtype)
        # k = BlockedTensor(k, self.blocked_input_signature).unblocked_tensor()
        # v = BlockedTensor(v, self.blocked_input_signature).unblocked_tensor()

        if use_cache:
            outputs = (hs, present)
        else:
            outputs = (hs,)

        return outputs  # hidden_states, present, (attentions)


def FixOPTDecoderLayer(
    self,
    bk=None,
    bc=None,
    layer_dtype=global_layer_dtype,
    weight_dtype=global_layer_dtype,
):
    if not isinstance(self, transformers.models.opt.modeling_opt.OPTDecoderLayer):
        return
    self.__class__ = OPTDecoderLayer
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        ShardLinear(self.self_attn.q_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.k_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.v_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.out_proj, 1, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.fc1, 0, rank, wsize, 64)
        ShardLinear(self.fc2, 1, rank, wsize, 64)
        self.model_parallel = True
    else:
        self.model_parallel = False
    for m in self.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta:
                continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(
                torch.empty_like(m._parameters[name], device="cpu"), **kwargs
            )

        if isinstance(m, torch.nn.Linear):
            FixLinear(m, bk, bc, layer_dtype, weight_dtype=weight_dtype)
    block(self)
    if not hasattr(self, "cpp_block"):
        params = [
            self.self_attn_layer_norm.weight,
            self.self_attn_layer_norm.bias,
            self.final_layer_norm.weight,
            self.final_layer_norm.bias,
        ]
        params += [
            self.self_attn.q_proj.weight,
            self.self_attn.q_proj.bias,
            self.self_attn.k_proj.weight,
            self.self_attn.k_proj.bias,
            self.self_attn.v_proj.weight,
            self.self_attn.v_proj.bias,
            self.self_attn.out_proj.weight,
            self.self_attn.out_proj.bias,
        ]
        params += [self.fc1.weight, self.fc1.bias]
        params += [self.fc2.weight, self.fc2.bias]

        self.cpp_block = torch.classes.tpp_llm.OPTDecoderLayer(
            params,
            self.self_attn_layer_norm.eps,
            self.final_layer_norm.eps,
            self.self_attn.head_dim,
            self.do_layer_norm_before,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")


def OptimizeModelForOPT(model, dtype, device="cpu", weight_dtype=None):
    set_pg()

    if weight_dtype is None:
        weight_dtype = dtype
    for m in model.modules():
        if isinstance(m, transformers.models.opt.modeling_opt.OPTDecoderLayer):
            FixOPTDecoderLayer(m, 16, 64, dtype, weight_dtype=weight_dtype)
        elif isinstance(m, torch.nn.Linear):
            FixLinear(m, 16, 64, dtype, parallel_dim=None)
            block(m)
    for m in model.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta:
                continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(
                torch.empty_like(m._parameters[name], device=device), **kwargs
            )


transformers.models.opt.modeling_opt.OPTForCausalLM._reorder_cache = staticmethod(
    _reorder_cache
)

OPTForCausalLM_forward = transformers.models.opt.modeling_opt.OPTForCausalLM.forward


def OPTForCausalLM_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return OPTForCausalLM_forward(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


transformers.models.opt.modeling_opt.OPTForCausalLM.forward = (
    OPTForCausalLM_forward_patched
)
