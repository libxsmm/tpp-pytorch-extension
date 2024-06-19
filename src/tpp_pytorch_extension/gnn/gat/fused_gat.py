###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)               #
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
from ..common.fused_ops import *

import time
from contextlib import contextmanager

import numpy as np

torch.autograd.set_detect_anomaly(False)

global_layer_dtype = torch.float32


class MLPAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, fuse_bias, inp_needs_grad, res_spmm, *inputs):

        if fuse_bias:
            (mlp_inp, wt, attn, bias) = inputs
            (mlp_out, attn_out) = fused_gat_cpp.mlp_attn_fwd(align, 1, inputs)
        else:
            (mlp_inp, wt, attn) = inputs
            (mlp_out, attn_out) = fused_gat_cpp.mlp_attn_fwd(align, 0, inputs)

        ctx.save_for_backward(mlp_out, attn, mlp_inp, wt)  # attn_input = mlp_out
        ctx.align = align
        ctx.fuse_bias = fuse_bias
        ctx.inp_needs_grad = inp_needs_grad
        ctx.res_spmm = res_spmm

        return (attn_out, mlp_out)  # mlp_out = attn_inp

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        if ctx.fuse_bias:
            (grad_inp, grad_wt, grad_attn, grad_bias,) = fused_gat_cpp.mlp_attn_bwd(
                ctx.align,
                1 if ctx.inp_needs_grad else 0,
                1,
                1 if ctx.res_spmm else 0,
                inputs,
            )
        else:
            grad_inp, grad_wt, grad_attn = fused_gat_cpp.mlp_attn_bwd(
                ctx.align,
                1 if ctx.inp_needs_grad else 0,
                0,
                1 if ctx.res_spmm else 0,
                inputs,
            )
        if ctx.fuse_bias:
            return (None, None, None, None, grad_inp, grad_wt, grad_attn, grad_bias)
        else:
            return (None, None, None, None, grad_inp, grad_wt, grad_attn)


class MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, fuse_bias, inp_needs_grad, *inputs):

        if fuse_bias:
            (inp, wt, bias) = inputs
            out = fused_gat_cpp.mlp_fwd(align, 1, inputs)
        else:
            (inp, wt) = inputs
            out = fused_gat_cpp.mlp_fwd(align, 0, inputs)

        ctx.save_for_backward(inp, wt)
        ctx.align = align
        ctx.fuse_bias = fuse_bias
        ctx.inp_needs_grad = inp_needs_grad

        return out

    @staticmethod
    def backward(ctx, *grad_out):
        inputs = list(grad_out)
        inputs += ctx.saved_tensors
        if ctx.fuse_bias:
            grad_inp, grad_wt, grad_bias = fused_gat_cpp.mlp_bwd(
                ctx.align, 1 if ctx.inp_needs_grad else 0, 1, inputs
            )
            return (None, None, None, grad_inp, grad_wt, grad_bias)
        else:
            grad_inp, grad_wt = fused_gat_cpp.mlp_bwd(
                ctx.align, 1 if ctx.inp_needs_grad else 0, 0, inputs
            )
            return (None, None, None, grad_inp, grad_wt)


class LinearOut(BlockedModule):
    def __init__(self, in_features, out_features, layer_dtype, bias=True):
        super(LinearOut, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.bk = out_features
        self.bc = 32
        FixLinear(self, self.bk, self.bc, layer_dtype)
        if self.bias is not None:
            self.bias.set_blocking_param((None, None, layer_dtype))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def maybe_block_params(self):
        self.weight.block()
        self.bias.block()

    def forward(self, input):
        inputs = [input, self.weight, self.bias]
        N = input.size(0)
        align = 64 if (N > 64 or N == 0) else N
        out = MLPFunction.apply(align, 1, 1, *inputs)

        return out


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
        input_needs_grad=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        match_pyg_gatconv=True,
        layer_dtype=global_layer_dtype,
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
        self.inp_needs_grad = input_needs_grad

        for cbf in [64, 32]:
            if self.bc % cbf == 0:
                self.bc = cbf
                break

        for kbf in [64, 32]:
            if self.bk % kbf == 0:
                self.bk = kbf
                break

        if not match_pyg_gatconv:
            self.fc_src = torch.nn.Linear(
                self._in_src_feats,
                out_feats * num_heads,
                bias=False,
            )
            FixLinear(self.fc_src, self.bk, self.bc, layer_dtype)

            self.fc_dst = torch.nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
            FixLinear(self.fc_dst, self.bk, self.bc, layer_dtype)
        else:
            self.fc = torch.nn.Linear(
                self._in_src_feats,
                out_feats * num_heads,
                bias=False,
            )
            FixLinear(self.fc, self.bk, self.bc, layer_dtype)

        self.attn_l = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_l.set_blocking_param((None, None, layer_dtype))
        self.attn_r = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r.set_blocking_param((None, None, layer_dtype))

        self.attn_drop = torch.nn.Dropout(attn_drop)

        self.leaky_relu = LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = torch.nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
                FixLinear(self.res_fc, self.bk, self.bc, layer_dtype)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        self.set_explicit_bias = False
        if not match_pyg_gatconv:
            self.fuse_src_bias = self.fc_src.bias is not None
            self.fuse_dst_bias = self.fc_dst.bias is not None
        else:
            self.fuse_src_bias = False
            self.fuse_dst_bias = False

        if bias and not (self.fuse_src_bias and self.fc_dst_bias):
            self.set_explicit_bias = True
            self.bias = BlockedParameter(
                torch.FloatTensor(size=(num_heads * out_feats,))
            )
            self.bias.set_blocking_param((None, None, layer_dtype))
        else:
            self.register_buffer("bias", None)

        self.act_drop = None
        self.activation = None
        self.bias_act_drop = None
        self.add_bias = None
        self.bias_act = None

        if self.set_explicit_bias:
            if activation is not None and feat_drop > 0.0:
                if activation == F.relu:
                    self.bias_act_drop = FusedBiasReLUDrop(feat_drop)
                elif activation == F.leaky_relu:
                    self.bias_act_drop = FusedBiasLeakyReLUDrop(alpha, feat_drop)
            elif activation is not None and feat_drop == 0.0:
                if activation == F.relu:
                    self.bias_act = FusedBiasReLU()
                elif activation == F.leaky_relu:
                    self.bias_act = FusedBiasLeakyReLU(alpha)
            else:
                self.add_bias = AddBias()
        else:
            if activation is not None and feat_drop > 0.0:
                if activation == F.relu:
                    self.act_drop = FusedReLUDrop(feat_drop)
                elif activation == F.leaky_relu:
                    self.act_drop = FusedLeakyReLUDrop(feat_drop)
            elif activation is not None and feat_drop == 0.0:
                if activation == F.relu:
                    self.activation = ReLU()
                elif activation == F.leaky_relu:
                    self.activation = LeakyReLU(alpha)

        self.reset_parameters()

    def maybe_block_params(self):

        if hasattr(self, "fc"):
            self.fc.weight.block()
            if self.fc.bias is not None:
                self.fc.bias.block()
        else:
            self.fc_src.weight.block()
            if self.fc_src.bias is not None:
                self.fc_src.bias.block()
            self.fc_dst.weight.block()
            if self.fc_dst.bias is not None:
                self.fc_dst.bias.block()

        self.attn_l.block()
        self.attn_r.block()

        if self.res_fc is not None:
            self.res_fc.weight.block()

        if self.bias is not None:
            self.bias.block()

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
        if hasattr(self, "fc"):
            glorot_initializer(self.fc.weight)
        else:
            glorot_initializer(self.fc_src.weight)
            glorot_initializer(self.fc_dst.weight)

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

                if not hasattr(self, "fc"):
                    assert hasattr(self, "fc_src") and hasattr(self, "fc_dst")
                    inputs_src = [h_src, self.fc_src.weight, self.attn_l]

                    if self.fuse_src_bias:
                        inputs_src.append(self.fc_src.bias)

                    N = h_src.size(0)
                    align = self.align if (N > self.align or N == 0) else N

                    el, feat_src_ = MLPAttentionFunction.apply(
                        align,
                        self.fuse_src_bias,
                        self.inp_needs_grad,
                        True,
                        *inputs_src,
                    )

                    feat_src = feat_src_.view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )

                    N = h_dst.size(0)
                    inputs_dst = [h_dst, self.fc_dst.weight, self.attn_r]
                    if self.fuse_dst_bias:
                        inputs_dst.append(self.fc_dst.bias)

                    align = self.align if (N > self.align or N == 0) else N

                    er, feat_dst_ = MLPAttentionFunction.apply(
                        align,
                        self.fuse_dst_bias,
                        self.inp_needs_grad,
                        False,
                        *inputs_dst,
                    )
                    feat_dst = feat_dst_.view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    inputs_src = [h_src, self.fc.weight, self.attn_l]

                    N = h_src.size(0)
                    align = self.align if (N > self.align or N == 0) else N

                    el, feat_src_ = MLPAttentionFunction.apply(
                        align,
                        False,
                        self.inp_needs_grad,
                        True,
                        *inputs_src,
                    )

                    feat_src = feat_src_.view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )

                    N = h_dst.size(0)

                    align = self.align if (N > self.align or N == 0) else N

                    inputs_dst = [h_dst, self.fc.weight, self.attn_r]
                    er, feat_dst_ = MLPAttentionFunction.apply(
                        align,
                        False,
                        self.inp_needs_grad,
                        False,
                        *inputs_dst,
                    )
                    feat_dst = feat_dst_.view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                assert hasattr(self, "fc")
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = feat

                inputs_src = [h_src, self.fc.weight, self.attn_l]

                N = h_src.size(0)
                align = self.align if (N > self.align or N == 0) else N

                el, feat_src_ = MLPAttentionFunction.apply(
                    align,
                    False,
                    self.inp_needs_grad,
                    *inputs_src,
                )
                feat_src = feat_src_.view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )

                if graph.is_block:
                    feat_dst = self.feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            etype = e.dtype
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e.float()))

            graph.edata["a"] = graph.edata["a"].to(etype)

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            rst = rst.view(*dst_prefix_shape, self._num_heads * self._out_feats)

            if self.bias_act_drop is not None:
                rst = self.bias_act_drop(rst, self.bias)
            elif self.bias_act is not None:
                rst = self.bias_act(rst, self.bias)
            elif self.add_bias is not None:
                rst = self.add_bias(rst, self.bias)
            elif self.act_drop is not None:
                rst = self.act_drop(rst)
            elif self.activation is not None:
                rst = self.activation(rst)
            elif self.bias_act is not None:
                rst = self.bias_act(rst, self.bias)

            rst = rst.view(*dst_prefix_shape, self._num_heads, self._out_feats)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
