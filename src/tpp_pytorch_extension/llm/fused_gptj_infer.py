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
from typing import Optional, Tuple, Union
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union
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


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


class GPTJBlock(BlockedModule):
    # def __init__(self, config):
    #     super().__init__()
    #     inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    #     self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    #     self.attn = GPTJAttention(config)
    #     self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        if not hasattr(self, "cpp_block"):
            raise ValueError("cpp_block is not initialized")
        orig_hidden_states = hidden_states
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        add_tensor_or_empty(attention_mask)
        # print("position_ids:", position_ids)
        if position_ids is None:
            raise ValueError("position_ids is required")

        inputs.append(position_ids)
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]

        outputs = self.cpp_block.forward(inputs, layer_past.get_cpp(), use_cache)
        hs = outputs[0]

        outputs = (hs,)

        if use_cache:
            outputs += (layer_past,)

        return outputs


def FixGPTJBlock(
    self,
    bk=None,
    bc=None,
    layer_dtype=global_layer_dtype,
    weight_dtype=global_layer_dtype,
):
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
        ShardLinear(self.mlp.fc_in, 0, rank, wsize, 64)
        ShardLinear(self.mlp.fc_out, 1, rank, wsize, 64)
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
            raise ValueError("embed_positions not created")
            max_positions = self.attn.bias.size(-1)
            pos_embd_dim = self.attn.rotary_dim or self.attn.embed_dim
            embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        params += [embed_positions]
        self.cpp_block = torch.classes.tpp_llm.GPTJBlock(
            params,
            self.attn.layer_idx,
            self.ln_1.eps,
            self.attn.head_dim,
            self.attn.embed_positions.size(0),
            self.attn.rotary_dim,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")


def OptimizeModelForGPTJ(model, dtype, device="cpu", weight_dtype=None):
    set_pg()

    if weight_dtype is None:
        weight_dtype = dtype
    for m in model.modules():
        if isinstance(m, transformers.models.gptj.modeling_gptj.GPTJBlock):
            FixGPTJBlock(m, 16, 64, dtype, weight_dtype=weight_dtype)
        elif isinstance(m, torch.nn.Linear):
            if m.weight.shape[0] % 100 == 0 and m.weight.shape[1] % 64 == 0:
                FixLinear(m, 100, 64, dtype, parallel_dim=0, block_size=100)
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


def GPTJModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = False
    output_hidden_states = False
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if use_cache and not isinstance(past_key_values, TppCache):
        if past_key_values is None:
            past_key_values = TppCache()
        else:
            if past_key_values.get_seq_length() == 0:
                past_key_values = TppCache()
            else:
                raise ValueError("past_key_values must be of TppCache type")

    seq_length = inputs_embeds.shape[1]
    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_length,
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # create causal mask
    causal_mask = past_key_values.align_and_invert_mask(attention_mask, inputs_embeds)
    # causal_mask = self._update_causal_mask(
    #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    # )

    # embed positions
    hidden_states = inputs_embeds

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, seq_length)
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    output_shape = (-1, seq_length, hidden_states.size(-1))

    next_decoder_cache = None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        outputs = block(
            hidden_states=hidden_states,
            layer_past=past_key_values,
            attention_mask=causal_mask,
            position_ids=position_ids,
            # head_mask=head_mask[i],
            use_cache=use_cache,
            # output_attentions=output_attentions,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            next_decoder_cache = outputs[1]

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
            ]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


transformers.models.gptj.modeling_gptj.GPTJModel.forward = GPTJModel_forward

GPTJForCausalLM_forward = transformers.models.gptj.modeling_gptj.GPTJForCausalLM.forward


def GPTJForCausalLM_forward_patched(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: int = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
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

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        cache_position=cache_position,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    # make sure sampling in fp16 works correctly and
    # compute loss in fp32 to match with mesh-tf version
    # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179

    # We only need logits for last token doing text generation
    # if only_last_logit == True and labels is None:
    #    hidden_states = hidden_states[:, -1:, :]

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    lm_logits = self.lm_head(hidden_states[:, slice_indices, :]).to(torch.float32)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = loss.to(hidden_states.dtype)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


transformers.models.gptj.modeling_gptj.GPTJForCausalLM.forward = (
    GPTJForCausalLM_forward_patched
)
