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

# from torch.nn import CrossEntropyLoss
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
from transformers.processing_utils import Unpack

from .llm_common import (
    FixLinear,
    ShardLinear,
    get_rank,
    get_size,
    set_pg,
    block,
    global_layer_dtype,
    TppCache,
)


class LlamaDecoderLayer(BlockedModule):
    # def __init__(self, config: LlamaConfig, layer_idx: int):
    #     super().__init__()
    #     self.hidden_size = config.hidden_size

    #     self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

    #     self.mlp = LlamaMLP(config)
    #     self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    #     self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # print("Attention mask shape: ", attention_mask.shape)

        if not hasattr(self, "cpp_block"):
            raise ValueError("cpp_block is not initialized")
        orig_hidden_states = hidden_states
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        add_tensor_or_empty(attention_mask)
        if position_ids is None:
            raise ValueError("position_ids is required")

        inputs.append(position_ids)
        # breakpoint()
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]

        outputs = self.cpp_block.forward(inputs, past_key_value.get_cpp(), use_cache)
        hs = outputs[0]

        outputs = (hs,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


def FixLlamaDecoderLayer(
    self,
    bk=None,
    bc=None,
    layer_dtype=global_layer_dtype,
    weight_dtype=global_layer_dtype,
    rotary_emb=None,
):
    if not isinstance(self, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
        return
    self.__class__ = LlamaDecoderLayer
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        block_size = self.self_attn.head_dim
        if hasattr(self.self_attn, "num_key_value_groups"):
            block_size = block_size * self.self_attn.num_key_value_groups
        ShardLinear(self.self_attn.q_proj, 0, rank, wsize, block_size)
        ShardLinear(self.self_attn.k_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.v_proj, 0, rank, wsize, self.self_attn.head_dim)
        ShardLinear(self.self_attn.o_proj, 1, rank, wsize, block_size)
        ShardLinear(self.mlp.gate_proj, 0, rank, wsize, 64)
        ShardLinear(self.mlp.up_proj, 0, rank, wsize, 64)
        ShardLinear(self.mlp.down_proj, 1, rank, wsize, 64)
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
        params = [self.input_layernorm.weight]
        params += [
            self.self_attn.q_proj.weight,
            self.self_attn.k_proj.weight,
            self.self_attn.v_proj.weight,
            self.self_attn.o_proj.weight,
        ]
        params += [self.post_attention_layernorm.weight]
        params += [
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
            self.mlp.down_proj.weight,
        ]  # since bias is False

        if rotary_emb is None and hasattr(self.self_attn, "rotary_emb"):
            rotary_emb = self.self_attn.rotary_emb

        if rotary_emb is not None:
            # max_position_embeddings = self.self_attn.max_position_embeddings
            max_position_embeddings = rotary_emb.config.max_position_embeddings
            position_ids = torch.arange(max_position_embeddings).unsqueeze(0)
            embed_positions = (
                torch.cat(
                    rotary_emb(
                        self.self_attn.q_proj.weight,
                        position_ids,
                    ),
                    dim=0,
                )
                .view([2, max_position_embeddings, -1])
                .to(torch.float)
            )
        else:
            raise NotImplementedError("Requires self.self_attn.rotary_emb")
        params += [embed_positions]

        # self.cpp_block = fused_llama_cpp.LlamaDecoderLayer(
        self.cpp_block = torch.classes.tpp_llm.LlamaDecoderLayer(
            params,
            self.self_attn.layer_idx,
            self.input_layernorm.variance_epsilon,
            self.self_attn.head_dim,
            max_position_embeddings,
            self.self_attn.head_dim,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")


def OptimizeModelForLlama(model, dtype, device="cpu", weight_dtype=None):
    set_pg()

    model.config._attn_implementation = "tpp"
    if hasattr(model.config, "attention_bias"):
        assert (
            model.config.attention_bias == False
        ), "attention_bias is not supported for Llama yet!"
    if weight_dtype is None:
        weight_dtype = dtype
    rotary_emb = None
    for m in model.modules():
        if isinstance(m, transformers.models.llama.modeling_llama.LlamaPreTrainedModel):
            if hasattr(m, "rotary_emb"):
                rotary_emb = m.rotary_emb
                break
    for m in model.modules():
        if isinstance(m, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
            FixLlamaDecoderLayer(
                m, 16, 64, dtype, weight_dtype=weight_dtype, rotary_emb=rotary_emb
            )
        elif isinstance(m, torch.nn.Linear):
            if m.weight.shape[0] % 100 == 0 and m.weight.shape[1] % 64 == 0:
                FixLinear(m, 100, 64, dtype, parallel_dim=1, block_size=64)
                block(m)
            elif m.weight.shape[0] % 64 == 0 and m.weight.shape[1] % 64 == 0:
                FixLinear(m, 64, 64, dtype, parallel_dim=1, block_size=64)
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


def LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,  #: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = False
    output_hidden_states = False
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and not isinstance(past_key_values, TppCache):
        if past_key_values is None:
            past_key_values = TppCache()
        else:
            if past_key_values.get_seq_length() == 0:
                past_key_values = TppCache()
            else:
                raise ValueError("past_key_values must be of TppCache type")

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
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

    # create position embeddings to be shared across the decoder layers
    # position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # position_embeddings Not used
    position_embeddings = None

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            # position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[1]

    hidden_states = self.norm(hidden_states)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward

LlamaForCausalLM_forward = (
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward
)


def LlamaForCausalLM_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,  #: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = False
    output_hidden_states = False

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
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


transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
    LlamaForCausalLM_forward_patched
)
