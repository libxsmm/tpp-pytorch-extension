###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################

import math
from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer
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
from tpp_pytorch_extension.utils import blocked_layout, xsmm
from tpp_pytorch_extension._C import _fused_gat as fused_gat_cpp
from ..common.gnn_utils import glorot_initializer

import time
from contextlib import contextmanager

import numpy as np

torch.autograd.set_detect_anomaly(False)

USE_BF16_PARAMS = True
MATCH_PYG_GATCONV = True


class DummyLinear(BlockedModule):
    def __init__(self, in_features, out_features, bias=True):
        super(DummyLinear, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        raise NotImplemented
        return input


class LinearOut(BlockedModule):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearOut, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.bk = out_features
        self.bc = 64
        self.weight.set_blocking_param(
            (
                [self.bk, self.bc],
                [0, 2, 3, 1],
            )
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def maybe_block_params(self):
        self.weight.block()

    def forward(self, input):
        inputs = [input, self.weight, self.bias]
        N = input.size(0)
        align = 64 if (N > 64 or N == 0) else N
        out = GATMLPFunction.apply(align, True, *inputs)

        return out


class LinearOutBF16(LinearOut):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearOutBF16, self).__init__(in_features, out_features)
        self.weight.set_blocking_param(
            (
                [self.bk, [self.bc // 2, 2]],
                [0, 2, 3, 1, 4],
                torch.bfloat16,
            )
        )
        self.bias.data = self.bias.to(torch.bfloat16)

    def forward(self, input):
        inputs = [input, self.weight, self.bias]
        N = input.size(0)
        align = 64 if (N > 64 or N == 0) else N
        out = GATMLPFunction.apply(align, True, *inputs)

        return out


class GATMLPAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, fuse_bias, *inputs):

        if fuse_bias:
            (mlp_inp, wt, attn, bias) = inputs
        else:
            (mlp_inp, wt, attn) = inputs
        (mlp_out, attn_out) = fused_gat_cpp.fused_gat_mlp_attn_fwd(
            align, 1 if fuse_bias else 0, inputs
        )

        ctx.save_for_backward(mlp_out, attn, mlp_inp, wt)  # attn_input = mlp_out
        ctx.align = align
        ctx.fuse_bias = fuse_bias

        return (attn_out, mlp_out)  # mlp_out = attn_inp

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        if ctx.fuse_bias:
            (
                grad_inp,
                grad_wt,
                grad_attn,
                grad_bias,
            ) = fused_gat_cpp.fused_gat_mlp_attn_bwd(ctx.align, 1, inputs)
        else:
            grad_inp, grad_wt, grad_attn = fused_gat_cpp.fused_gat_mlp_attn_bwd(
                ctx.align, 0, inputs
            )
        if ctx.fuse_bias:
            return (None, None, grad_inp, grad_wt, grad_attn, grad_bias)
        else:
            return (None, None, grad_inp, grad_wt, grad_attn)


class GATMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, fuse_bias, *inputs):

        if fuse_bias:
            (inp, wt, bias) = inputs
        else:
            (inp, wt) = inputs
        out = fused_gat_cpp.fused_mlp_fwd(align, 1 if fuse_bias else 0, inputs)

        ctx.save_for_backward(inp, wt)
        ctx.align = align
        ctx.fuse_bias = fuse_bias

        return out

    @staticmethod
    def backward(ctx, *grad_out):
        inputs = list(grad_out)
        inputs += ctx.saved_tensors
        if ctx.fuse_bias:
            grad_inp, grad_wt, grad_bias = fused_gat_cpp.fused_mlp_bwd(
                ctx.align, 1, inputs
            )
            return (None, None, grad_inp, grad_wt, grad_bias)
        else:
            grad_inp, grad_wt = fused_gat_cpp.fused_mlp_bwd(ctx.align, 0, inputs)
            return (None, None, grad_inp, grad_wt)


class GATAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, *inputs):

        inp, attn = inputs
        out = fused_gat_cpp.attn_fwd(align, inputs)

        ctx.save_for_backward(inp, attn)  # attn_input = mlp_out
        ctx.align = align

        return out

    @staticmethod
    def backward(ctx, *grad_out):
        inputs = list(grad_out)
        inputs += ctx.saved_tensors
        grad_inp, grad_attn = fused_gat_cpp.attn_bwd(ctx.align, inputs)
        return (None, grad_inp, grad_attn)


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, inp, training):
        outputs = fused_gat_cpp.gat_dropout_fwd(p, inp, training)
        (out, dp_mask) = outputs
        ctx.save_for_backward(dp_mask)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        # breakpoint()
        grad_inp = fused_gat_cpp.gat_dropout_bwd(ctx.p, inputs)
        return (None, grad_inp, None)


class Dropout_(nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout_, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class Dropout(Dropout_):
    r"""Train"""

    def forward(self, input):
        input = input.contiguous()
        output = DropoutFunction.apply(self.p, input, self.training)

        return output


class LeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, input):
        (out, mask) = fused_gat_cpp.leakyrelu_fwd(alpha, input)
        ctx.save_for_backward(input, mask)
        ctx.alpha = alpha

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_gat_cpp.leakyrelu_bwd(ctx.alpha, inputs)

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


class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        (out, mask) = fused_gat_cpp.relu_fwd(inp)
        ctx.save_for_backward(mask)

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_gat_cpp.relu_bwd(inputs)

        return grad_inp


class ReLU(nn.Module):
    __constants__ = ["inplace"]

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = False  # inplace

    def forward(self, inp):
        output = ReLUFn.apply(inp)
        return output


class FusedBiasReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, p, training):
        inputs = [inp, bias]
        if training and p > 0:
            (out, rmask, dpmask) = fused_gat_cpp.bias_relu_drop_fwd(inputs, p, 1)
            ctx.save_for_backward(rmask, dpmask)
        else:
            (out, rmask) = fused_gat_cpp.bias_relu_drop_fwd(inputs, p, 0)
            ctx.save_for_backward(rmask)
        ctx.p = p

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_gat_cpp.bias_relu_drop_bwd(inputs, ctx.p)

        return (grad_inp, grad_bias, None, None)


class FusedBiasReLUDrop(nn.Module):
    __constants__ = ["p", "inplace"]

    def __init__(self, inplace: bool = False):
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


class FusedBiasLeakyReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, alpha, p, training):
        inputs = [inp, bias]
        if training and p > 0:
            (out, rmask, dpmask) = fused_gat_cpp.bias_lrelu_drop_fwd(
                inputs, alpha, p, 1
            )
            ctx.save_for_backward(inp, rmask, dpmask)
        else:
            (out, rmask) = fused_gat_cpp.bias_lrelu_drop_fwd(inputs, alpha, p, 0)
            ctx.save_for_backward(inp, rmask)
        ctx.alpha = alpha
        ctx.p = p

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_bias = fused_gat_cpp.bias_lrelu_drop_bwd(
            inputs, ctx.alpha, ctx.p
        )

        return (grad_inp, grad_bias, None, None, None)


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

    def extra_repr(self) -> str:
        return "alpha={}, p={}, inplace={}".format(self.alpha, self.p, self.inplace)

    def forward(self, inp, bias):
        output = FusedBiasLeakyReLUDropFn.apply(
            inp, bias, self.alpha, self.p, self.training
        )
        return output


class FusedReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, inp, training):
        (out, rmask, dpmask) = fused_gat_cpp.relu_drop_fwd(p, inp, 1 if training else 0)
        if training:
            ctx.save_for_backward(rmask, dpmask)
            ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_gat_cpp.relu_drop_bwd(ctx.p, inputs)
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


class FusedLeakyReLUDropFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, p, inp, training):
        (out, rmask, dpmask) = fused_gat_cpp.leaky_relu_drop_fwd(
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
        grad_inp = fused_gat_cpp.leaky_relu_drop_bwd(ctx.alpha, ctx.p, inputs)
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


class AddBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias):
        inputs = [inp, bias]
        out = fused_gat_cpp.add_bias_fwd(inputs)

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        grad_bias = fused_gat_cpp.add_bias_bwd(inputs)

        return (grad_outs[0], grad_bias)


class AddBias(nn.Module):
    def __init__(self):
        super(AddBias, self).__init__()

    def forward(self, inp, bias):
        output = AddBiasFn.apply(inp, bias)
        return output


class GATConvOpt(BlockedModule):
    r"""

    Description
    -----------
    Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        alpha=0.01,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConvOpt, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._num_heads = num_heads
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.bc = self._in_dst_feats
        self.bk = num_heads * self._out_feats
        self.res = False
        self.align = 64
        self.fdp = feat_drop
        self.adp = attn_drop
        for cbf in [50, 32, 16]:
            if self._in_dst_feats % cbf == 0:
                self.bc = cbf
                break

        for kbf in [50, 32, 16]:
            if self._out_feats % kbf == 0:
                self.bk = kbf
                break
        self.fc_src = DummyLinear(
            self._in_src_feats,
            out_feats * num_heads,
            bias=False,
        )
        self.fc_src.weight.set_blocking_param(
            (
                [self.bk, self.bc],
                [0, 2, 3, 1],
            )
        )
        self.fc_dst = DummyLinear(self._in_dst_feats, out_feats * num_heads, bias=False)
        self.fc_dst.weight.set_blocking_param(
            (
                [self.bk, self.bc],
                [0, 2, 3, 1],
            )
        )

        self.attn_l = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )

        self.feat_drop = Dropout(feat_drop)
        self.attn_drop = Dropout(attn_drop)

        self.leaky_relu = LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = DummyLinear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
                self.res_fc.weight.set_blocking_param(
                    (
                        [self.bk, self.bc],
                        [0, 2, 3, 1],
                    )
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        self.fuse_src_bias = True if self.fc_src.bias is not None else False
        self.fuse_dst_bias = True if self.fc_dst.bias is not None else False
        self.bias = None

        if bias and (not self.fuse_src_bias or not self.fuse_dst_bias):
            self.bias = BlockedParameter(
                torch.FloatTensor(size=(num_heads * out_feats,))
            )
            self.set_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.act_drop = None
        self.activation = None
        self.bias_act_drop = None
        self.add_bias = None

        if self.set_explicit_bias:
            if activation is not None and feat_drop > 0.0:
                if activation == F.relu:
                    self.bias_act_drop = FusedBiasReLUDrop(feat_drop)
                elif activation == F.leaky_relu:
                    self.bias_act_drop = FusedBiasLeakyReLUDrop(alpha, feat_drop)
            else:
                self.add_bias = AddBias()

        self.use_bf16 = False
        self.reset_parameters()

    def maybe_block_params(self):

        if not hasattr(self, "fc_src"):
            self.fc.weight.block()
        else:
            self.fc_src.weight.block()
            self.fc_dst.weight.block()

        if self.res_fc is not None:
            self.res_fc.weight.block()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        if not MATCH_PYG_GATCONV:
            gain = nn.init.calculate_gain("relu")
        else:
            gain = 1.0

        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            if not MATCH_PYG_GATCONV:
                nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
                nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            else:
                glorot_initializer(self.fc_src.weight)
                glorot_initializer(self.fc_dst.weight)

            if not MATCH_PYG_GATCONV:
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
            else:
                glorot_initializer(self.attn_l)
                glorot_initializer(self.attn_r)

        if self.set_explicit_bias:
            nn.init.constant_(self.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, distgnn=False, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]

                h_src = feat[0]
                h_dst = feat[1]

                inputs_src = [h_src, self.fc_src.weight, self.attn_l]
                if self.fuse_src_bias:
                    inputs_src.append(self.fc_src.bias)

                N = h_src.size(0)
                align = self.align if (N > self.align or N == 0) else N

                el, feat_src_ = GATMLPAttentionFunction.apply(
                    align,
                    self.fuse_src_bias,
                    *inputs_src,
                )
                feat_src = feat_src_.view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )

                N = h_dst.size(0)
                inputs_dst = [
                    h_dst,
                    self.fc_dst.weight,
                ]
                if self.fuse_dst_bias:
                    inputs_dst.append(self.fc_dst.bias)

                align = self.align if (N > self.align or N == 0) else N

                feat_dst_ = GATMLPFunction.apply(
                    align,
                    self.fuse_dst_bias,
                    *inputs_dst,
                )  #
                feat_dst = feat_dst_.view(
                    *dst_prefix_shape, self._num_heads, self._out_feats
                )

            else:

                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = feat
                inputs_src = [h_src, self.fc_src.weight, self.attn_l]
                if self.fuse_src_bias:
                    inputs_src.append(self.fc_src.bias)

                N = h_src.size(0)
                align = self.align if (N > self.align or N == 0) else N

                el, feat_src_ = GATMLPAttentionFunction.apply(
                    align,
                    self.fuse_src_bias,
                    *inputs_src,
                )
                feat_src = feat_src_.view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )

                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    inputs_dst = [
                        h_dst,
                        self.fc_dst.weight,
                    ]
                    if self.fuse_dst_bias:
                        inputs_dst.append(self.fc_dst.bias)

                    N = h_dst.size(0)
                    align = self.align if (N > self.align or N == 0) else N
                    feat_dst_ = GATMLPFunction.apply(
                        align,
                        self.fuse_dst_bias,
                        *inputs_dst,
                    )  #
                    feat_dst = feat_dst_.view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )

                    # if graph.is_block:
                    #    feat_dst = self.feat_src[: graph.number_of_dst_nodes()]
                    #    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    #    dst_prefix_shape = (
                    #        graph.number_of_dst_nodes(),
                    #    ) + dst_prefix_shape[1:]

                else:
                    feat_dst = feat_src

            inputs_dst = [feat_dst, self.attn_r]
            N = feat_dst.size(0)
            align = self.align if (N > self.align or N == 0) else N
            er = GATAttentionFunction.apply(align, *inputs_dst).view(
                -1, self._num_heads, 1
            )

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e.float()))

            if self.use_bf16:
                graph.edata["a"] = graph.edata["a"].to(torch.bfloat16)

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            rst = rst.view(*dst_prefix_shape, self._num_heads * self._out_feats)
            if self.bias_act_drop is not None:
                rst = self.bias_act_drop(rst, self.bias)
            elif self.add_bias is not None:
                rst = self.add_bias(rst, self.bias)
            rst = rst.view(*dst_prefix_shape, self._num_heads, self._out_feats)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATConvOptBF16(GATConvOpt):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        alpha=0.01,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):

        super(GATConvOptBF16, self).__init__(
            in_feats,
            out_feats,
            num_heads,
            feat_drop,
            attn_drop,
            negative_slope,
            alpha,
            residual,
            activation,
            allow_zero_in_degree,
            bias,
        )
        if USE_BF16_PARAMS:
            self.fc_src.weight.set_blocking_param(
                (
                    [self.bk, [self.bc // 2, 2]],
                    [0, 2, 3, 1, 4],
                    torch.bfloat16,
                )
            )
            self.fc_dst.weight.set_blocking_param(
                (
                    [self.bk, [self.bc // 2, 2]],
                    [0, 2, 3, 1, 4],
                    torch.bfloat16,
                )
            )
            self.attn_l.data = self.attn_l.to(torch.bfloat16)
            self.attn_r.data = self.attn_r.to(torch.bfloat16)
            if self.set_explicit_bias:
                self.bias.data = self.bias.to(torch.bfloat16)

        self.use_bf16 = True
