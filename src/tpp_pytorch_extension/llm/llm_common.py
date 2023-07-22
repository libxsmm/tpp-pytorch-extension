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
import numpy as np
import os

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32

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
        split_sizes = self.split_sizes if hasattr(self, "split_sizes") else []
        ret = torch.ops.tpp_llm.fc_plain(input, self.weight, bias, parallel_dim, split_sizes)
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
    
def ShardLinear(m, dim, rank, size, block_size=1):
    # dim = 0 - shard output features
    # dim = 1 - shard input features
    dim_size = m.weight.shape[dim]
    assert(dim_size % block_size == 0, f"dim_size ({dim_size}) is not multiple of block_size ({block_size})")
    num_blocks = dim_size // block_size
    split_size = ((num_blocks + size - 1) // size) * block_size
    m.split_sizes = [split_size] * size
    m.split_sizes[-1] -= (split_size * size - dim_size)
    # print(m.split_sizes)
    assert(sum(m.split_sizes) == dim_size, "Sum of split sizes doesn't match dim size")
    # m.weight.data = torch.chunk(m.weight.data, size, dim)[rank].contiguous()
    m.weight.data = torch.split(m.weight.data, split_size, dim)[rank].contiguous()
    if m.weight.is_meta:
        m.weight = torch.nn.Parameter(torch.empty_like(m.weight.data, device='cpu'))
    if m.bias is not None:
        if dim == 0:
            # m.bias.data = torch.chunk(m.bias.data, size, dim)[rank].contiguous()
            m.bias.data = torch.split(m.bias.data, split_size, dim)[rank].contiguous()
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

def get_layer_past_and_offset(layer_past: Optional[Tuple[torch.Tensor]], discrete_kv: bool):
    if layer_past is None:
        return ([], 0)
    n = len(layer_past)
    if n < 4: # cache with beam_idx
        if discrete_kv == True:
            B1, N, S, H = layer_past[0].shape
            if n == 3:
                B2 = layer_past[2].shape[0]
            else:
                B2 = B1
            inc_size = int(os.environ.get("KV_CACHE_INC_SIZE", "64"))
            capacity = S + inc_size
            new_key = layer_past[0].new_zeros([B2, N, capacity, H])
            new_value = layer_past[1].new_zeros([B2, N, capacity, H])
            if B1 == B2:
                new_key[:,:,:S,:].copy_(layer_past[0])
                new_value[:,:,:S,:].copy_(layer_past[1])
                new_beam_idx = torch.arange(B2).unsqueeze(0).expand([capacity, B2]).contiguous()
                if n == 3:
                    new_beam_idx[S-1] = layer_past[2]
            else:
                assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
                assert n == 3, f"Must use lazy kv cache reorder but n = {n}"
                num_beams = B2 // B1
                new_key[::num_beams,:,:S,:] = layer_past[0]
                new_value[::num_beams,:,:S,:] = layer_past[1]
                new_beam_idx = layer_past[2].new_zeros([capacity, B2]).contiguous()
                # beam_idx was adjusted in reorder_cache by num_beams, so fit it back
                new_beam_idx[:S] = layer_past[2] * num_beams
            offset = torch.tensor(S)
            layer_past = (new_key, new_value, new_beam_idx, offset,)
            return (layer_past, offset)
        else:
            return (layer_past, layer_past[0].shape[2])

    else:
        B1, N, S, H = layer_past[0].shape
        B2 = layer_past[2].shape[0]
        #print(f"pkv{n} B1: {B1}  B2: {B2} t_offset: {layer_past[3]}")
        #print(f"pkv{n} : layer_past[3]{layer_past[3]}")
        return (layer_past, layer_past[3])


def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
    This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
    [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
    beam_idx at every generation step.
    """
    # print(f"_reorder_cache: len(pkv) = {len(past[0])}, {past[0][0].shape}  beam_idx = {beam_idx}")
    if len(past[0]) == 4: #discrete kv_cache
        B1 = past[0][0].shape[0]
        B2 = beam_idx.shape[0]
        # print(f"_reorder_cache: B1: {past[0][0].shape}, beam_idx: {beam_idx}")
        # print(f"B1 = {B1}, B2 = {B2}")
        xyz = True
        if B1 != B2:
            assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
            num_beams = B2 // B1
            new_past = []
            print(f"past[2]: {past[0][2]}")
            print(f"beam_idx: {beam_idx}")
            for layer_past in past:
                layer_past_0 = layer_past[0].repeat_interleave(num_beams, dim=0).contiguous()
                layer_past_1 = layer_past[1].repeat_interleave(num_beams, dim=0).contiguous()
                layer_past_2 = layer_past[2].repeat_interleave(num_beams, dim=1).mul(num_beams).contiguous()
                if xyz == True:
                    print(f"layer_past[2]: {layer_past[2]}")
                    print(f"layer_past_2: {layer_past_2}")
                    xyz = False
                layer_past_2[layer_past[3]-1]=beam_idx
                new_past.append((layer_past_0, layer_past_1, layer_past_2, layer_past[3],))
            print(f"new_past[0][2]: {new_past[0][2]}")

            return tuple(new_past)
        else:
            for layer_past in past:
                layer_past[2][layer_past[3]-1]=beam_idx
            return past
    else:
        B1 = past[0][0].shape[0]
        B2 = beam_idx.shape[0]
        if B1 != B2:
            assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
            num_beams = B2 // B1
            beam_idx = beam_idx // num_beams
            # print(f"B1 = {B1}, B2 = {B2}, beam_idx: {beam_idx}")
        return tuple(tuple(layer_past) + (beam_idx,) for layer_past in past)

    # ret = fused_llm_cpp.reorder_cache(past, beam_idx)
    # return tuple(
    #     tuple(p for p in layer_past) for layer_past in ret
    # )

    # return tuple(
    #     tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
    #     for layer_past in past
    # )

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
