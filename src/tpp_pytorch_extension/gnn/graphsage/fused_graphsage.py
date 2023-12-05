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
from torch.optim import Optimizer
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
)
from tpp_pytorch_extension.utils import blocked_layout, xsmm
from tpp_pytorch_extension._C import _fused_gsage as fused_gsage_cpp
from tpp_pytorch_extension._C import _xsmm as xsmm_cpp
import time
from contextlib import contextmanager
from dgl.nn.pytorch.conv.sageconv import *

torch.autograd.set_detect_anomaly(False)

USE_BF16_PARAMS = True


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


class SAGEMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, apply_bias, p, act, res, training, *inputs):
        if res:
            (inp, inp_res, wt, res_wt, bias) = inputs
            (out, act_mask, dp_mask,) = fused_gsage_cpp.fused_gsage_mlp_fwd(
                align, apply_bias, p, act, res, training, inputs
            )
        else:
            (inp, wt, bias) = inputs
            out, act_mask, dp_mask = fused_gsage_cpp.fused_gsage_mlp_fwd(
                align, apply_bias, p, act, res, training, inputs
            )

        if training:
            if act == "None":
                act_mask = torch.tensor([], dtype=torch.short)
            if p == 0.0:
                dp_mask = torch.tensor([], dtype=torch.short)

            if res:
                ctx.save_for_backward(inp, inp_res, wt, res_wt, act_mask, dp_mask)
            else:
                ctx.save_for_backward(inp, wt, act_mask, dp_mask)

            ctx.act = act
            ctx.res = res
            ctx.p = p
            ctx.align = align
            ctx.apply_bias = apply_bias

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors

        if ctx.res:
            (
                grad_inp,
                grad_res_inp,
                grad_wt,
                grad_res_wt,
                grad_bias,
            ) = fused_gsage_cpp.fused_gsage_mlp_bwd(
                ctx.align, ctx.apply_bias, ctx.p, ctx.act, ctx.res, inputs
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                grad_inp,
                grad_res_inp,
                grad_wt,
                grad_res_wt,
                grad_bias,
            )
        else:
            grad_inp, grad_wt, grad_bias = fused_gsage_cpp.fused_gsage_mlp_bwd(
                ctx.align, ctx.apply_bias, ctx.p, ctx.act, ctx.res, inputs
            )
            return (None, None, None, None, None, None, grad_inp, grad_wt, grad_bias)


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, training, inp):
        outputs = fused_gsage_cpp.dropout_fwd(p, inp, training)
        (out, dp_mask) = outputs
        ctx.save_for_backward(dp_mask)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_gsage_cpp.dropout_bwd(ctx.p, inputs)
        return (None, None, None, grad_inp)


class SAGEConvOpt(BlockedModule):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGEConvOpt, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = feat_drop
        self.activation = "relu" if activation == F.relu else "None"
        self.bc = self._in_dst_feats
        self.bk = self._out_feats
        self.res = False
        self.apply_bias = False
        self.fc_self = None
        self.align = 32

        for cbf in [50, 32, 16]:
            if self._in_dst_feats % cbf == 0:
                self.bc = cbf
                break

        for kbf in [50, 32, 16]:
            if self._out_feats % kbf == 0:
                self.bk = kbf
                break

        if aggregator_type.startswith("pool"):
            self.fc_pool = DummyLinear(self._in_src_feats, self._in_src_feats)
            bc = self._in_src_feats
            for cb in [50, 32, 16]:
                if self._in_src_feats % cb == 0:
                    bc = cb
                    break
            bk = bc

            self.fc_pool.weight.set_blocking_param(
                (
                    [bk, bc],
                    [0, 2, 3, 1],
                )
            )
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )
        if aggregator_type != "gcn" and ("concat" not in aggregator_type):
            self.res = True
            self.fc_self = DummyLinear(self._in_dst_feats, out_feats, bias=bias)
            self.fc_self.weight.set_blocking_param(
                (
                    [self.bk, self.bc],
                    [0, 2, 3, 1],
                )
            )

        if "concat" not in aggregator_type:
            self.fc_neigh = DummyLinear(self._in_dst_feats, out_feats, bias=False)
        else:
            self.fc_neigh = DummyLinear(
                self._in_dst_feats + self._in_dst_feats, out_feats, bias=False
            )

        self.fc_neigh.weight.set_blocking_param(
            (
                [self.bk, self.bc],
                [0, 2, 3, 1],
            )
        )

        if bias:
            self.apply_bias = True

            if self.fc_self is not None:
                if self.fc_self.bias is None:
                    self.bias = BlockedParameter(torch.zeros(out_feats))
            else:
                self.bias = BlockedParameter(torch.zeros(out_feats))
        else:
            self.register_buffer("bias", None)

        self.use_bf16 = False

        self.reset_parameters()

    def maybe_block_params(self):
        self.fc_neigh.weight.block()
        if self._aggre_type != "gcn" and ("concat" not in self._aggre_type):
            self.fc_self.weight.block()
        if self._aggre_type.startswith("pool"):
            self.fc_pool.weight.block()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type.startswith("pool"):
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn" and ("concat" not in self._aggre_type):
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            feat_src = None
            feat_dst = None

            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat

                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = feat_dst
                else:
                    graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
            elif self._aggre_type.startswith("pool"):
                inputs = [feat_src, self.fc_pool.weight]
                if self.use_bf16:
                    inputs = [
                        i.to(torch.bfloat16) if i.is_floating_point() else i
                        for i in inputs
                    ]
                inputs.append(self.fc_pool.bias)
                graph.srcdata["h"] = SAGEMLPFunction.apply(
                    self.align, 0.0, self.activation, False, self.training, *inputs
                )

                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "none":
                h_neigh = feat_src
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(self._aggre_type)
                )

            self.maybe_block_params()

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                inputs = [h_neigh, self.fc_neigh.weight, self.bias]
                N = h_neigh.size(0)
                align = self.align if (N > self.align or N == 0) else N

                rst = SAGEMLPFunction.apply(
                    align,
                    self.apply_bias,
                    self.feat_drop,
                    self.activation,
                    self.res,
                    self.training,
                    *inputs
                )
            elif "concat" in self._aggre_type:
                h = torch.cat((h_self, h_neigh), 1)
                N = h.size(0)
                align = self.align if (N > self.align or N == 0) else N
                inputs = [h, self.fc_neigh.weight, self.bias]
                rst = SAGEMLPFunction.apply(
                    align,
                    self.apply_bias,
                    self.self.feat_drop,
                    self.activation,
                    self.res,
                    self.training,
                    *inputs
                )

            else:
                inputs = [
                    h_self,
                    h_neigh,
                    self.fc_self.weight,
                    self.fc_neigh.weight,
                    self.fc_self.bias,
                ]
                N = h_self.size(0)
                align = self.align if (N > self.align or N == 0) else N
                rst = SAGEMLPFunction.apply(
                    align,
                    self.apply_bias,
                    self.feat_drop,
                    self.activation,
                    self.res,
                    self.training,
                    *inputs
                )
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class SAGEConvOptBF16(SAGEConvOpt):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGEConvOptBF16, self).__init__(
            in_feats, out_feats, aggregator_type, feat_drop, bias, norm, activation
        )
        if USE_BF16_PARAMS:
            self.fc_neigh.weight.set_blocking_param(
                (
                    [self.bk, [self.bc // 2, 2]],
                    [0, 2, 3, 1, 4],
                    torch.bfloat16,
                )
            )
            if aggregator_type != "gcn":
                self.fc_self.weight.set_blocking_param(
                    (
                        [self.bk, [self.bc // 2, 2]],
                        [0, 2, 3, 1, 4],
                        torch.bfloat16,
                    )
                )

        self.use_bf16 = True


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
