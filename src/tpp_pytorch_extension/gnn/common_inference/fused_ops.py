###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
from dgl.utils import expand_as_pair
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
from tpp_pytorch_extension._C import _fused_ops_inf as fused_ops_cpp


USE_LOW_PREC_PARAMS = True
global_layer_dtype = torch.float32


def FixLinear(
    self,
    bk=None,
    bc=None,
    layer_dtype=global_layer_dtype,
    block_size=1,
    weight_dtype=None,
):
    if weight_dtype is None:
        weight_dtype = layer_dtype
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
                weight_dtype,
            )
        )

    if self.bias is not None:
        self.bias = BlockedParameter(self.bias.data)
        self.bias.set_blocking_param((None, None, layer_dtype))


##################LeakyReLU#################
class LeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, input):
        (out, mask) = fused_ops_cpp.leakyrelu(alpha, input)

        return out

class LeakyReLU(nn.Module):
    __constants__ = ["inplace"]

    def __init__(self, alpha, inplace: bool = False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace = False  # inplace

    def forward(self, input):
        input = input.contiguous()
        output = LeakyReLUFn.apply(self.alpha, input)
        return output


####################ReLU########################
class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        (out, mask) = fused_ops_cpp.relu(inp)

        return out

class ReLU(nn.Module):
    __constants__ = ["inplace"]

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = False  # inplace

    def forward(self, inp):
        output = ReLUFn.apply(inp)
        return output


#######################FusedBiasReLU################

class FusedBiasReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, training):
        inputs = [inp, bias]
        (out, rmask) = fused_ops_cpp.bias_relu(inputs)

        return out

class FusedBiasReLU(nn.Module):
    def __init__(self):
        super(FusedBiasReLU, self).__init__()

    def forward(self, inp, bias):
        output = FusedBiasReLUFn.apply(inp, bias, self.training)
        return output

#######################FusedBiasLeakyReLU################
class FusedBiasLeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, alpha, training):
        inputs = [inp, bias]
        (out, rmask) = fused_ops_cpp.bias_lrelu(inputs, alpha)

        return out

class FusedBiasLeakyReLU(nn.Module):
    __constants__ = ["alpha"]

    def __init__(self, alpha: float = 0.01):
        super(FusedBiasLeakyReLU, self).__init__()
        self.alpha = alpha
        self.align = 64

    def extra_repr(self) -> str:
        return "alpha={}".format(self.alpha)

    def forward(self, inp, bias):
        output = FusedBiasLeakyReLUFn.apply(
            inp, bias, self.alpha, self.training
        )
        return output

#######################AddBias################
class AddBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias):
        inputs = [inp, bias]
        out = fused_ops_cpp.add_bias(inputs)

        return out

class AddBias(nn.Module):
    def __init__(self):
        super(AddBias, self).__init__()

    def forward(self, inp, bias):
        output = AddBiasFn.apply(inp, bias)
        return output
