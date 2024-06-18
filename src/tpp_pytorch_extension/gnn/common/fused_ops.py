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
from tpp_pytorch_extension._C import _fused_ops as fused_ops_cpp


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
        (out, mask) = fused_ops_cpp.leakyrelu_fwd(alpha, input)
        ctx.save_for_backward(input, mask)
        ctx.alpha = alpha

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_ops_cpp.leakyrelu_bwd(ctx.alpha, inputs)

        return (None, grad_inp)


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
        (out, mask) = fused_ops_cpp.relu_fwd(inp)
        ctx.save_for_backward(mask)

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_ops_cpp.relu_bwd(inputs)

        return grad_inp


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
        (out, rmask) = fused_ops_cpp.bias_relu_fwd(inputs)
        if training:
            ctx.save_for_backward(rmask)
            ctx.dprm = 0
            if bias.dtype == torch.bfloat16:
                ctx.dprm = 1

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_ops_cpp.bias_relu_bwd(inputs, ctx.dprm)

        return (grad_inp, grad_bias, None)


class FusedBiasReLU(nn.Module):
    def __init__(self):
        super(FusedBiasReLU, self).__init__()

    def forward(self, inp, bias):
        output = FusedBiasReLUFn.apply(inp, bias, self.training)
        return output


#######################FusedBiasReLUDropout################
class FusedBiasReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, p, training):
        inputs = [inp, bias]
        if training and p > 0:
            (out, rmask, dpmask) = fused_ops_cpp.bias_relu_drop_fwd(inputs, p, 1)
            ctx.save_for_backward(rmask, dpmask)
        else:
            (out, rmask) = fused_ops_cpp.bias_relu_drop_fwd(inputs, p, 0)
            ctx.save_for_backward(rmask)
        ctx.p = p
        ctx.dprm = 0
        if bias.dtype == torch.bfloat16:
            ctx.dprm = 1

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_ops_cpp.bias_relu_drop_bwd(inputs, ctx.p, ctx.dprm)

        return (grad_inp, grad_bias, None, None)


class FusedBiasReLUDrop(nn.Module):
    __constants__ = ["p", "inplace"]

    def __init__(self, p, inplace: bool = False):
        super(FusedBiasReLUDrop, self).__init__()
        self.inplace = False  # inplace
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)

    def forward(self, inp, bias):
        output = FusedBiasReLUDropFn.apply(inp, bias, self.p, self.training)
        return output


#######################FusedBiasLeakyReLUDropout################
class FusedBiasLeakyReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, alpha, p, training, align):
        inputs = [inp, bias]
        N = inp.size(0)
        align = align if (N > align or N == 0) else N
        if training and p > 0:
            (out, rmask, dpmask) = fused_ops_cpp.bias_lrelu_drop_fwd(
                align, inputs, alpha, p, 1
            )
            ctx.save_for_backward(inp, rmask, dpmask)
        else:
            (out, rmask) = fused_ops_cpp.bias_lrelu_drop_fwd(align, inputs, alpha, p, 0)
            if training:
                ctx.save_for_backward(inp, rmask)
        ctx.dprm = 0
        if bias.dtype == torch.bfloat16:
            ctx.dprm = 1
        ctx.alpha = alpha
        ctx.p = p
        ctx.align = align

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_ops_cpp.bias_lrelu_drop_bwd(
            ctx.align, inputs, ctx.alpha, ctx.p, ctx.dprm
        )

        return (grad_inp, grad_bias, None, None, None, None)


class FusedBiasLeakyReLUDrop(nn.Module):
    __constants__ = ["alpha", "p", "inplace"]

    def __init__(self, alpha: float = 0.01, p: float = 0.2, inplace: bool = False):
        super(FusedBiasLeakyReLUDrop, self).__init__()
        self.inplace = False  # inplace
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.alpha = alpha
        self.p = p
        self.align = 64

    def extra_repr(self) -> str:
        return "alpha={}, p={}, inplace={}".format(self.alpha, self.p, self.inplace)

    def forward(self, inp, bias):
        output = FusedBiasLeakyReLUDropFn.apply(
            inp, bias, self.alpha, self.p, self.training, self.align
        )
        return output


#######################FusedBiasLeakyReLU################
class FusedBiasLeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, alpha, training):
        inputs = [inp, bias]
        (out, rmask) = fused_ops_cpp.bias_lrelu_fwd(inputs, alpha)
        if training:
            ctx.save_for_backward(inp, rmask)
        ctx.alpha = alpha
        ctx.dprm = 0
        if bias.dtype == torch.bfloat16:
            ctx.dprm = 1

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_ops_cpp.bias_lrelu_bwd(inputs, ctx.alpha, ctx.dprm)

        return (grad_inp, grad_bias, None, None)


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
            inp, bias, self.alpha, self.training, self.align
        )
        return output


#######################FusedReLUDropout################
class FusedReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, inp, training):
        (out, rmask, dpmask) = fused_ops_cpp.relu_drop_fwd(p, inp, 1 if training else 0)
        if training:
            ctx.save_for_backward(rmask, dpmask)
            ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_ops_cpp.relu_drop_bwd(ctx.p, inputs)
        return (None, grad_inp, None)


class FusedReLUDrop(nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(FusedReLUDrop, self).__init__()
        self.inplace = False  # inplace
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)

    def forward(self, input):
        output = FusedReLUDropFn.apply(self.p, input, self.training)
        return output


#######################FusedLeakyReLUDropout################
class FusedLeakyReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, p, inp, training):
        (out, rmask, dpmask) = fused_ops_cpp.leaky_relu_drop_fwd(
            alpha, p, inp, 1 if training else 0
        )
        if training:
            ctx.save_for_backward(inp, rmask, dpmask)
            ctx.alpha = alpha
            ctx.p = p

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_ops_cpp.leaky_relu_drop_bwd(ctx.alpha, ctx.p, inputs)
        return (None, None, grad_inp, None)


class FusedLeakyReLUDrop(nn.Module):
    __constants__ = ["alpha", "p", "inplace"]
    p: float
    inplace: bool

    def __init__(
        self, alpha: float = 0.01, p: float = 0.5, inplace: bool = False
    ) -> None:
        super(FusedLeakyReLUDrop, self).__init__()
        self.inplace = False  # inplace
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.alpha = alpha
        self.p = p

    def extra_repr(self) -> str:
        return "alpha={}, p={}, inplace={}".format(self.alpha, self.p, self.inplace)

    def forward(self, input):
        output = FusedLeakyReLUDropFn.apply(self.alpha, self.p, input, self.training)
        return output


#######################AddBias################
class AddBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias):
        inputs = [inp, bias]
        out = fused_ops_cpp.add_bias_fwd(inputs)
        ctx.dprm = 0
        if bias.dtype == torch.bfloat16:
            ctx.dprm = 1

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        grad_bias = fused_ops_cpp.add_bias_bwd(inputs, ctx.dprm)

        return (grad_outs[0], grad_bias)


class AddBias(nn.Module):
    def __init__(self):
        super(AddBias, self).__init__()

    def forward(self, inp, bias):
        output = AddBiasFn.apply(inp, bias)
        return output
