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
