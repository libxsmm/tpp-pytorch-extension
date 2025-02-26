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
from torch.nn import CrossEntropyLoss
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
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache

from .llm_common import (
    BlockedLinear,
    BlockedLayerNorm,
    FixLinear,
    ShardLinear,
    get_rank,
    get_size,
    set_pg,
    block,
    compare,
    global_layer_dtype,
    TppCache,
)


class OPTDecoderLayer(BlockedModule):
    # def __init__(self, config):
    #     super().__init__()
    #     self.embed_dim = config.hidden_size
    #     self.self_attn = OPTAttention(
    #         embed_dim=self.embed_dim,
    #         num_heads=config.num_attention_heads,
    #         dropout=config.attention_dropout,
    #         is_decoder=True,
    #         bias=config.enable_bias,
    #     )
    #     self.do_layer_norm_before = config.do_layer_norm_before
    #     self.dropout = config.dropout
    #     self.activation_fn = ACT2FN[config.activation_function]

    #     self.self_attn_layer_norm = nn.LayerNorm(
    #         self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
    #     )
    #     self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
    #     self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
    #     self.final_layer_norm = nn.LayerNorm(
    #         self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
    #     )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ]:
        # print("HS:", hidden_states.shape, hidden_states.device, hidden_states.dtype)
        # print("layer_past:", layer_past[0].shape if layer_past is not None else layer_past)
        # print("attention_mask:", attention_mask.shape if attention_mask is not None else attention_mask)
        # print("position_ids:", position_ids.shape if position_ids is not None else position_ids)
        if not hasattr(self, "cpp_block"):
            raise ValueError("cpp_block is not initialized")
        orig_hidden_states = hidden_states
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        add_tensor_or_empty(attention_mask)

        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]

        outputs = self.cpp_block.forward(inputs, past_key_value.get_cpp(), use_cache)
        hs = outputs[0]

        if use_cache:
            outputs += (past_key_value,)

        return outputs


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
            self.self_attn.layer_idx,
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
    layers = model.model.decoder.layers
    for i, l in enumerate(layers):
        if not hasattr(l.self_attn, "layer_idx"):
            l.self_attn.layer_idx = i
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
    model._supports_cache_class = True


def OPTDecoder_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    position_ids: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = False
    output_hidden_states = False
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and not isinstance(past_key_values, TppCache):
        if past_key_values is None:
            past_key_values = TppCache()
        else:
            if (
                isinstance(past_key_values, Cache)
                and past_key_values.get_seq_length() == 0
            ):
                past_key_values = TppCache()
            else:
                raise ValueError("past_key_values must be of TppCache type")

    # create causal mask
    causal_mask = past_key_values.align_and_invert_mask(attention_mask, inputs_embeds)

    past_key_values_length = past_key_values.get_seq_length()

    # embed positions

    if position_ids is None:
        position_ids = torch.cumsum(attention_mask, dim=1)
        position_ids = (position_ids * attention_mask - 1).long()
        # cut positions if `past_key_values_length` is > 0
        position_ids = position_ids[:, past_key_values_length:]

    pos_embeds = self.embed_positions(
        attention_mask, past_key_values_length, position_ids=position_ids
    )

    if self.project_in is not None:
        inputs_embeds = self.project_in(inputs_embeds)

    hidden_states = inputs_embeds + pos_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            # position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[1]

    if self.final_layer_norm is not None:
        hidden_states = self.final_layer_norm(hidden_states)

    if self.project_out is not None:
        hidden_states = self.project_out(hidden_states)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


transformers.models.opt.modeling_opt.OPTDecoder.forward = OPTDecoder_forward


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
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = False
    output_hidden_states = False
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # only_last_logit = (
    #     self.config.only_last_logit
    #     if hasattr(self.config, "only_last_logit")
    #     else False
    # )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


transformers.models.opt.modeling_opt.OPTForCausalLM.forward = (
    OPTForCausalLM_forward_patched
)
