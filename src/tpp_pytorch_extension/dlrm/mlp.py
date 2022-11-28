#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.autograd import Function
#from torchrec.modules.activation import SwishLayerNorm
#from torchrec.modules.utils import extract_module_or_tensor_callable
from torch.autograd import Function

from tpp_pytorch_extension._C import _mlp_cpp as mlp_cpp

class MLPFunction(Function):
    @staticmethod
    def forward(ctx, bias, nLayers, *inputs):
        bias = True
        output = mlp_cpp.forward(bias, nLayers, inputs)
        ctx.nLayers = nLayers
        ctx.bwd_list = output
        #ctx.save_for_backward(output)
        lastlayer_act = output[-1]
        return lastlayer_act

    @staticmethod
    def backward(ctx, grad_out):
        saved_bwd = ctx.bwd_list #ctx.saved_tensors
        nLayers = ctx.nLayers
        output = mlp_cpp.backward(nLayers, saved_bwd, grad_out)
        return None, None, *output

class MLP(torch.nn.Module):
    """
    Applies a stack of Perceptron modules sequentially (i.e. Multi-Layer Perceptron).

    Args:
        in_size (int): `in_size` of the input.
        layer_sizes (List[int]): `out_size` of each Perceptron module.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (str, Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation of
            each Perceptron module.
            If `activation` is a `str`, we currently only support the follow strings, as
            "relu", "sigmoid", and "swish_layernorm".
            If `activation` is a `Callable[[], torch.nn.Module]`, `activation()` will be
            called once per Perceptron module to generate the activation module for that
            Perceptron module, and the parameters won't be shared between those activation
            modules.
            One use case is when all the activation modules share the same constructor
            arguments, but don't share the actual module parameters.
            Default: torch.relu.
    device (Optional[torch.device]): default compute device.

    Example::

        batch_size = 3
        in_size = 40
        input = torch.randn(batch_size, in_size)

        layer_sizes = [16, 8, 4]
        mlp_module = MLP(in_size, layer_sizes, bias=True)
        output = mlp_module(input)
        assert list(output.shape) == [batch_size, layer_sizes[-1]]
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: Union[
            str,
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        torch.manual_seed(42)

        self.layer_sizes = layer_sizes

        if activation == "relu":
            activation = torch.relu
        elif activation == "sigmoid":
            activation = torch.sigmoid

        self._linear = [0 for i in range(len(layer_sizes))]

        for i in range(len(layer_sizes)):
            self._linear[i]: nn.Linear = nn.Linear(
                layer_sizes[i-1] if i>0 else in_size, layer_sizes[i], bias=bias, device=device
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O) where O is `out_size` of the last Perceptron module.
        """
        bias = True
        if self._linear[0].bias is None:
            bias = False
        inputs = [input]

        for i in range(len(self.layer_sizes)):
            inputs += [self._linear[i].weight]        
            if bias:
                inputs += [self._linear[i].bias]

        output = MLPFunction.apply(bias, len(self.layer_sizes), *inputs)
        return output
