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
from transformers.utils import is_torch_fx_proxy
import numpy as np

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32
unpad = True
print_cou = 0


def compare(ref, opt, name=""):
    ref = ref.detach()
    opt = opt.detach()
    allclose = ref.allclose(opt, atol=1e-6, rtol=1e-6)
    # print(f"{name}: ref: {ref.abs().mean():14g} allclose: {allclose}  shape: {ref.shape}")
    if not allclose:
        print(f"ref = {ref.view([-1])[:8]}, xsmm = {opt.view([-1])[:8]}")
        avg = ref.abs().mean()
        adiff = (ref - opt).abs()
        rdiff = adiff / avg
        err = 1e-6
        for ind, rd in np.ndenumerate(rdiff):
            if rd > err:
                print(
                    f"{ind}: ref: {ref[ind].item():.7g} opt: {opt[ind].item():.7g} diff: {adiff[ind].item():.7g}  rdiff: {rd:.7g}"
                )
                err = rd


class BlockedLinear(BlockedModule, torch.nn.Linear):
    def maybe_block_params(self):
        self.weight.block()
        if self.bias is not None:
            self.bias.block()

    def parallelize(self, dim, rank, size):
        if size <= 1: return
        ShardLinear(self, dim, rank, size)
        self.model_parallel = True
        self.parallel_dim = dim
        self.parallel_rank = rank
        self.parallel_size = size

    def forward(self, input):
        #if self.model_parallel == True and self.parallel_dim == 1:
        #    input = input.chunk(self.parallel_size, dim=-1)[self.parallel_rank].contiguous()
        #self.maybe_block_params()
        bias = (
            self.bias if self.bias is not None else torch.Tensor().to(self.weight.dtype)
        )
        # print("BIas:", bias.shape, bias.dtype)
        input = input.to(self.weight.dtype)
        parallel_dim = self.parallel_dim if self.model_parallel == True else -1
        ret = torch.ops.tpp_llm.fc_plain(input, self.weight, bias, parallel_dim)
        # if self.model_parallel == True:
        #     with torch.inference_mode(False):
        #         if self.parallel_dim == 0:
        #             agret = [t.view_as(ret) for t in ret.new_empty([self.parallel_size]+list(ret.shape)).chunk(self.parallel_size)]
        #             torch.distributed.all_gather(agret, ret)
        #             ret = torch.cat(agret, dim = -1)
        #         else:
        #             torch.distributed.all_reduce(ret)
        return ret


class BlockedLayerNorm(BlockedModule, torch.nn.LayerNorm):
    def maybe_block_params(self):
        if self.elementwise_affine:
            self.weight.block()
            if self.bias is not None:
                self.bias.block()


def FixLinear(self, bk=None, bc=None, layer_dtype=global_layer_dtype, parallel_dim=None):
    if not isinstance(self, torch.nn.Linear):
        return
    if isinstance(self, BlockedLinear): return
    self.__class__ = BlockedLinear
    self.model_parallel = False
    if parallel_dim is not None:
        self.parallelize(parallel_dim, get_rank(), get_size())
    self.weight = BlockedParameter(self.weight.data)
    self.weight.set_blocking_param(
        (
            [bk, bc],
            [0, 2, 3, 1],
        )
    )
    layer_use_low_prec = layer_dtype != torch.float32
    if layer_use_low_prec == True and USE_LOW_PREC_PARAMS:
        low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
        self.weight.set_blocking_param(
            (
                [
                    bk,
                    [
                        bc // low_prec_vnni_blocking,
                        low_prec_vnni_blocking,
                    ],
                ],
                [0, 2, 3, 1, 4],
                layer_dtype,
            )
        )

    if self.bias is not None:
        self.bias = BlockedParameter(self.bias.data)
        self.bias.set_blocking_param((None, None, layer_dtype))
    


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

        if layer_past is not None:
            #print("KP: ", layer_past[0].shape)
            #print("VP: ", layer_past[1].shape)
            add_tensor_or_empty(layer_past[0])
            add_tensor_or_empty(layer_past[1])
            if len(layer_past) > 2:
                add_tensor_or_empty(layer_past[2].to(torch.long))
            else:
                inputs.append(dummy_tensor_int)
        else:
            inputs += [dummy_tensor, dummy_tensor, dummy_tensor_int]
        add_tensor_or_empty(attention_mask)
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            offset = 0
            if layer_past is not None:
                offset = layer_past[0].shape[1]
            position_ids = torch.arange(offset, offset+seq_len).repeat(hidden_states.shape[0], 1)

        add_tensor_or_empty(position_ids)
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        #print("AM: ", inputs[-2].shape, inputs[-2].dtype)

        # print("PHS:", hidden_states.shape)
        hs, k, v = self.cpp_block.forward(inputs, use_cache)
        #print("K: ", k.shape)
        #print("V: ", v.shape)

        hs = BlockedTensor(hs, self.blocked_input_signature, orig_hidden_states.dtype)
        # k = BlockedTensor(k, self.blocked_input_signature).unblocked_tensor()
        # v = BlockedTensor(v, self.blocked_input_signature).unblocked_tensor()

        if use_cache:
            outputs = (hs, (k, v))
        else:
            outputs = (hs,)

        return outputs  # hidden_states, present, (attentions)

def ShardLinear(m, dim, rank, size):
    # dim = 0 - shard output features
    # dim = 1 - shard input features
    m.weight.data = torch.chunk(m.weight.data, size, dim)[rank].contiguous()
    if m.weight.is_meta:
        m.weight = torch.nn.Parameter(torch.empty_like(m.weight.data, device='cpu'))
    if m.bias is not None:
        if dim == 0:
            m.bias.data = torch.chunk(m.bias.data, size, dim)[rank].contiguous()
        else:
            m.bias.data = m.bias.data / size
        if m.bias.is_meta:
            m.bias = torch.nn.Parameter(torch.empty_like(m.bias.data, device='cpu'))

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def get_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        size = torch.distributed.get_world_size()
    else:
        size = 1
    return size

def all_reduce(t):
    with torch.autograd.profiler.record_function("allreduce"):
        torch.distributed.all_reduce(t)

def set_pg():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fused_llm_cpp.set_pg(torch.distributed.distributed_c10d._get_default_group())

def FixGPTJBlock(self, bk=None, bc=None, layer_dtype=global_layer_dtype):
    if not isinstance(self, transformers.models.gptj.modeling_gptj.GPTJBlock):
        return
    self.__class__ = GPTJBlock
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        ShardLinear(self.attn.q_proj, 0, rank, wsize)
        ShardLinear(self.attn.k_proj, 0, rank, wsize)
        ShardLinear(self.attn.v_proj, 0, rank, wsize)
        ShardLinear(self.attn.out_proj, 1, rank, wsize)
        ShardLinear(self.mlp.fc_in, 0, rank, wsize)
        ShardLinear(self.mlp.fc_out, 1, rank, wsize)
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
            self.attn.num_attention_heads // wsize,
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


def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
    This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
    [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
    beam_idx at every generation step.
    """
    return tuple(layer_past + (beam_idx,) for layer_past in past)

    # ret = fused_llm_cpp.reorder_cache(past, beam_idx)
    # return tuple(
    #     tuple(p for p in layer_past) for layer_past in ret
    # )

    # return tuple(
    #     tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
    #     for layer_past in past
    # )

transformers.models.gptj.modeling_gptj.GPTJForCausalLM._reorder_cache = staticmethod(_reorder_cache)

# bm_default_blocking_factors = BlockedModule.default_blocking_factors
# @staticmethod
# def custom_blocking_factors(S):
#     print(f"S = {S}")
#     if S % 32 == 0: return [S//32, 32]
#     return bm_default_blocking_factors
# BlockedModule.default_blocking_factors = custom_blocking_factors

try:
    import transformers

    transformers_orig_is_tensor = transformers.file_utils.is_tensor

    def is_tensor(x):
        """Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`."""
        if transformers_orig_is_tensor(x):
            return True
        if isinstance(x, BlockedTensor):
            return True
        return False

    transformers.file_utils.is_tensor = is_tensor
except:
    pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
