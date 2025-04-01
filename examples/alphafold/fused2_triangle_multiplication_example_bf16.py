###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################

import time
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

# import intel_extension_for_pytorch as ipex

import tpp_pytorch_extension
from tpp_pytorch_extension.alphafold.Alpha_FusedTriangleMultiplication import (
    FusedTriangleMultiplicationOpti,
)

torch.set_default_tensor_type(torch.FloatTensor)


class FusedTriangleMultiplication(nn.Module):
    #   def __init__(self,config, global_config, act_dim):
    def __init__(self, equation, num_intermediate_channel, act_dim):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [N_res, N_res, c_z]
        mask: Pair mask, shape [N_res, N_res].
        is_training: Whether the module is in training mode.

        Returns:
        Outputs, same shape/type as act.
        """
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        # self.c_equation = self.config['equation']
        self.c_equation = equation
        self.num_intermediate_channel = num_intermediate_channel
        self.left_norm_input = nn.LayerNorm(act_dim)
        self.projection = nn.Linear(act_dim, 2 * num_intermediate_channel)
        self.gate = nn.Linear(num_intermediate_channel, 2 * num_intermediate_channel)
        self.center_norm = nn.LayerNorm(num_intermediate_channel)
        self.output_projection = nn.Linear(num_intermediate_channel, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask):
        mask = mask[..., None]
        left_act = self.left_norm_input(act)
        proj_act = mask * self.projection(left_act)
        proj_act *= torch.sigmoid(self.gate(left_act))
        left_proj_act = proj_act[:, :, : self.num_intermediate_channel]
        right_proj_act = proj_act[:, :, self.num_intermediate_channel :]
        act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
        act = self.center_norm(act)
        act = self.output_projection(act)
        act *= torch.sigmoid(self.gating_linear(left_act))
        return act


set = 1
length = 764
if set == 1:
    B, S, HS = length, length, 64
    equation = "ikc,jkc->ijc"
    num_intermediate_channel = 64

if set == 2:
    B, S, HS = length, length, 128
    equation = "ikc,jkc->ijc"
    num_intermediate_channel = 128

if set == 3:
    B, S, HS = length, length, 64
    equation = "kjc,kic->ijc"
    num_intermediate_channel = 64

if set == 4:
    B, S, HS = length, length, 128
    equation = "kjc,kic->ijc"
    num_intermediate_channel = 128


class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.fused_triangle_multiplication = FusedTriangleMultiplication(
            equation=equation,
            num_intermediate_channel=num_intermediate_channel,
            act_dim=HS,
        )  # Attention layer

    def forward(self, act, mask):
        x = self.fused_triangle_multiplication(act, mask)
        return x


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()
        self.fused_triangle_multiplication = FusedTriangleMultiplicationOpti(
            equation=equation,
            num_intermediate_channel=num_intermediate_channel,
            act_dim=HS,
        )  # Attention layer

    def forward(self, act, mask):
        x = self.fused_triangle_multiplication(act, mask)
        return x


net1 = Net1().to(torch.bfloat16)
net2 = Net2().to(torch.bfloat16)

torch.manual_seed(11)  # Set random seed for reproducibility

act = 0.1 * torch.randn(B, S, HS, requires_grad=False).to(torch.bfloat16)
mask = (torch.rand(B, S, requires_grad=False) > 0.5).to(torch.bfloat16)

left_norm_input_weight = 0.1 * torch.randn(HS).to(torch.bfloat16)
left_norm_input_bias = 0.1 * torch.randn(HS).to(torch.bfloat16)
projection_weight = 0.1 * torch.randn(2 * num_intermediate_channel, HS).to(torch.bfloat16)
projection_bias = 0.1 * torch.randn(2 * num_intermediate_channel).to(torch.bfloat16)
gate_weight = 0.1 * torch.randn(2 * num_intermediate_channel, num_intermediate_channel).to(torch.bfloat16)
gate_bias = 0.1 * torch.randn(2 * num_intermediate_channel).to(torch.bfloat16)

output_projection_weight = 0.1 * torch.randn(HS, num_intermediate_channel).to(torch.bfloat16)
output_projection_bias = 0.1 * torch.randn(HS).to(torch.bfloat16)
center_norm_weight = 0.1 * torch.randn(num_intermediate_channel).to(torch.bfloat16)
center_norm_bias = 0.1 * torch.randn(num_intermediate_channel).to(torch.bfloat16)
gating_linear_weight = 0.1 * torch.randn(HS, HS).to(torch.bfloat16)
gating_linear_bias = 0.1 * torch.randn(HS).to(torch.bfloat16)

net1.fused_triangle_multiplication.left_norm_input.weight = torch.nn.Parameter(
    left_norm_input_weight
)
net1.fused_triangle_multiplication.left_norm_input.bias = torch.nn.Parameter(
    left_norm_input_bias
)
net1.fused_triangle_multiplication.projection.weight = torch.nn.Parameter(
    projection_weight
)
net1.fused_triangle_multiplication.projection.bias = torch.nn.Parameter(projection_bias)
net1.fused_triangle_multiplication.gate.weight = torch.nn.Parameter(gate_weight)
net1.fused_triangle_multiplication.gate.bias = torch.nn.Parameter(gate_bias)
net1.fused_triangle_multiplication.center_norm.weight = torch.nn.Parameter(
    center_norm_weight
)
net1.fused_triangle_multiplication.center_norm.bias = torch.nn.Parameter(
    center_norm_bias
)
net1.fused_triangle_multiplication.output_projection.weight = torch.nn.Parameter(
    output_projection_weight
)
net1.fused_triangle_multiplication.output_projection.bias = torch.nn.Parameter(
    output_projection_bias
)
net1.fused_triangle_multiplication.gating_linear.weight = torch.nn.Parameter(
    gating_linear_weight
)
net1.fused_triangle_multiplication.gating_linear.bias = torch.nn.Parameter(
    gating_linear_bias
)

net2.fused_triangle_multiplication.left_norm_input.weight = torch.nn.Parameter(
    left_norm_input_weight
)
net2.fused_triangle_multiplication.left_norm_input.bias = torch.nn.Parameter(
    left_norm_input_bias
)
net2.fused_triangle_multiplication.projection.weight = torch.nn.Parameter(
    projection_weight
)
net2.fused_triangle_multiplication.projection.bias = torch.nn.Parameter(projection_bias)
net2.fused_triangle_multiplication.gate.weight = torch.nn.Parameter(gate_weight)
net2.fused_triangle_multiplication.gate.bias = torch.nn.Parameter(gate_bias)
net2.fused_triangle_multiplication.center_norm.weight = torch.nn.Parameter(
    center_norm_weight
)
net2.fused_triangle_multiplication.center_norm.bias = torch.nn.Parameter(
    center_norm_bias
)
net2.fused_triangle_multiplication.output_projection.weight = torch.nn.Parameter(
    output_projection_weight
)
net2.fused_triangle_multiplication.output_projection.bias = torch.nn.Parameter(
    output_projection_bias
)
net2.fused_triangle_multiplication.gating_linear.weight = torch.nn.Parameter(
    gating_linear_weight
)
net2.fused_triangle_multiplication.gating_linear.bias = torch.nn.Parameter(
    gating_linear_bias
)

Y1 = net1(act, mask)
Y2 = net2(act, mask)

r = Y1.max().to(torch.float32) - Y1.min().to(torch.float32)
# print((torch.abs(Y1 - Y2) / r > 0.1)[:, :, :].sum())
print(
    "    Foward pass check: ",
    ((torch.abs(Y1.to(torch.float32) - Y2.to(torch.float32)) / r < 0.01).sum() == B * S * HS).item(),
)
# print("diff: ", r)
print(
    " Number of errors: ",
    B * S * HS - (torch.abs(Y1.to(torch.float32) - Y2.to(torch.float32)) / r < 0.01).sum(),
)


forward1 = 0  # variables to store time values
forward2 = 0

N = 10  # Number of iterations

# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/original_triangle_multiplication'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True
#     ) as prof:
for _ in range(N):  # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1(act, mask)
    forward1 += time.time() - start
    # prof.step()

tpp_pytorch_extension.reset_debug_timers()
# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/long_triangle_multiplication'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True
#     ) as prof:
for _ in range(N):  # Optimized PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y2 = net2(act, mask)
    forward2 += time.time() - start
    # prof.step()
tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
