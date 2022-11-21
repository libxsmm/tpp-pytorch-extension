import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Union, Callable, Optional

from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
    get_vnni_blocking,
)

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

class Perceptron(BlockedModule, nn.Module):
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
        super(Perceptron, self).__init__()
        self.blocking_enabled = False
        self.layer_dtype = torch.float32
        self.blocked_input_signature = None
        self.out_block_size = out_size
        self.in_block_size = in_size
        self._linear: nn.Linear = nn.Linear(
            self.in_block_size, self.out_block_size, bias=bias, device=device
        )
        self._activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation
        if bias:
            self._linear.bias = BlockedParameter(self._linear.bias.data)
        self._linear.weight = BlockedParameter(self._linear.weight.data)

    def set_blocking(self, in_block_size, out_block_size, layer_dtype=torch.float32):
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size
        self.layer_dtype = layer_dtype
        use_low_prec = layer_dtype != torch.float32
        self.blocked_input_signature = get_blocking_signature("NC", "NCNC")
        if use_low_prec:
            low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
            self._linear.weight.set_blocking_param(
                (
                    [
                        self.out_block_size,
                        [
                            self.in_block_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self._linear.bias.set_blocking_param((None, None, layer_dtype))
        else:
            self._linear.weight.set_blocking_param(
                (
                    [self.out_block_size, self.in_block_size],
                    [0, 2, 3, 1],
                )
            )
        self.blocking_enabled = True

    def maybe_block_params(self):
        self._linear.weight.block()
        self._linear.bias.block()

    def forward(self, input):
        if self.blocking_enabled:
            self.maybe_block_params()
            orig_input_dtype = input.dtype
            
            input = self.get_blocked_tensor(
                input, self.blocked_input_signature, [None, self.in_block_size]
            )
            if self._linear.bias is None:
                self._linear.bias = torch.Tensor()

            inputs = [
                input,
                self._linear.weight,
                self._linear.bias if self._linear.bias is not None else torch.Tensor(),
            ]
##            inputs = [
##                i.cvt_to(self.layer_dtype) if i.is_floating_point() else i
##                for i in inputs
##            ]
            output = perceptron_cpp.perceptron_global(input, self._linear.weight, self._linear.bias) 

##            output = PerceptronFunction.apply(*inputs)
            output = BlockedTensor(output, self.blocked_input_signature, orig_input_dtype)
        else:
            output = torch.sigmoid(torch.nn.Linear.forward(self, input))

        return output
