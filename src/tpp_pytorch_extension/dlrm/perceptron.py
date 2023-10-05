import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Union, Callable, Optional

from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension import get_vnni_blocking

from tpp_pytorch_extension._C import _perceptron_cpp as perceptron_cpp

##class PerceptronFunction(Function):
##    @staticmethod
##    def forward(ctx, input, weight, bias):
##        output = perceptron_cpp.forward(input, weight, bias)
##        ctx.save_for_backward(input, weight, bias, output)
##        return output
##
##    @staticmethod
##    def backward(ctx, grad_output):
##        (input, weight, bias, act) = ctx.saved_tensors
##        (grad_input, grad_weight, grad_bias) = perceptron_cpp.backward(grad_output, input, weight, act)
##        return grad_input, grad_weight, grad_bias


class Perceptron(BlockedModule, torch.nn.Linear):  # nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        bias: bool = True,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.sigmoid,
        device: Optional[torch.device] = None,
    ) -> None:
        torch.nn.Linear.__init__(self, in_size, out_size, bias)
        self.blocking_enabled = False
        self.layer_dtype = torch.float32
        self.blocked_input_signature = None
        self.out_block_size = out_size
        self.in_block_size = in_size
        self.act_sigmoid = True if activation == torch.sigmoid else False
        self._activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation
        if bias:
            self.bias = BlockedParameter(self.bias.data)
        self.weight = BlockedParameter(self.weight.data)

    def set_blocking(self, in_block_size, out_block_size, layer_dtype=torch.float32):
        self.layer_dtype = layer_dtype
        use_low_prec = layer_dtype != torch.float32
        low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
        self.blocked_input_signature = get_blocking_signature("NC", "NCNC")
        _, out_block_size = BlockedModule.default_blocking_factors(
            self.weight.shape[0], out_block_size
        )  # TODO make it self.out_block_size
        _, in_block_size = BlockedModule.default_blocking_factors(
            self.weight.shape[1], in_block_size, low_prec_vnni_blocking
        )
        self.in_block_size_input = in_block_size
        if use_low_prec and (in_block_size % low_prec_vnni_blocking == 0):
            self.weight.set_blocking_param(
                (
                    [
                        out_block_size,
                        [
                            in_block_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.bias.set_blocking_param((None, None, layer_dtype))
        else:
            self.weight.set_blocking_param(
                (
                    [out_block_size, in_block_size],
                    [0, 2, 3, 1],
                    layer_dtype,
                )
            )
        self.blocking_enabled = True

    def maybe_block_params(self):
        self.weight.block()
        self.bias.block()

    def forward(self, input):
        if self.blocking_enabled:
            self.maybe_block_params()
            orig_input_dtype = input.dtype

            input = self.get_blocked_tensor(
                input,
                self.blocked_input_signature,
                [None, self.in_block_size_input],  # self.in_block_size]
            )

            if self.bias is None:
                self.bias = torch.Tensor().cvt_to(self.layer_dtype)

            inputs = [
                input,
                self.weight,
                self.bias if self.bias is not None else torch.Tensor(),
            ]
            inputs = [
                i.cvt_to(self.layer_dtype) if i.is_floating_point() else i
                for i in inputs
            ]

            output = perceptron_cpp.perceptron_global(self.act_sigmoid, *inputs)
            ##            output = PerceptronFunction.apply(*inputs)
            output = BlockedTensor(
                output, self.blocked_input_signature, orig_input_dtype
            )
        else:
            output = torch.sigmoid(torch.nn.Linear.forward(self, input))

        return output
