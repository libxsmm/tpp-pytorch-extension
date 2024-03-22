###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                    #
###############################################################################


import math
import torch
from torch import nn
from torch.autograd import Function

from tpp_pytorch_extension._C import (
    _alpha_attention as Alpha_FusedTriangleMultiplication_cpp,
)


class FusedTriangleMultiplicationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        mask,
        c_equation,
        left_norm_input_weight,
        left_norm_input_bias,
        projection_weight,
        projection_bias,
        gate_weight,
        gate_bias,
        center_norm_weight,
        center_norm_bias,
        output_projection_weight,
        output_projection_bias,
        gating_linear_weight,
        gating_linear_bias,
    ):
        equation_flag = int(0)
        if c_equation == "ikc,jkc->ijc":  # "Outgoing" edges equation
            equation_flag = 0
        else:  # "Incoming" edges equation
            equation_flag = 1
        act = Alpha_FusedTriangleMultiplication_cpp.fusedtrianglemulti_forward(
            act.contiguous(),
            mask.contiguous(),
            equation_flag,
            left_norm_input_weight,
            left_norm_input_bias,
            projection_weight,
            projection_bias,
            gate_weight,
            gate_bias,
            center_norm_weight,
            center_norm_bias,
            output_projection_weight,
            output_projection_bias,
            gating_linear_weight,
            gating_linear_bias,
        )
        return act


def FusedTriangleMultiplicationOpti_forward(self, act, mask):
    mask = mask[..., None]
    if (
        act.dtype == torch.bfloat16
        or mask.dtype == torch.bfloat16
        or self.left_norm_input.weight.dtype == torch.bfloat16
        or self.left_norm_input.bias.dtype == torch.bfloat16
        or self.projection.weight.dtype == torch.bfloat16
        or self.projection.bias.dtype == torch.bfloat16
        or self.gate.weight.dtype == torch.bfloat16
        or self.gate.bias.dtype == torch.bfloat16
        or self.center_norm.weight.dtype == torch.bfloat16
        or self.center_norm.bias.dtype == torch.bfloat16
        or self.output_projection.weight.dtype == torch.bfloat16
        or self.output_projection.bias.dtype == torch.bfloat16
        or self.gating_linear.weight.dtype == torch.bfloat16
        or self.gating_linear.bias.dtype == torch.bfloat16
    ):
        act = FusedTriangleMultiplicationFunction.apply(
            act.to(torch.bfloat16),
            mask.to(torch.float32),
            self.c_equation,
            self.left_norm_input.weight.to(torch.bfloat16),
            self.left_norm_input.bias.to(torch.bfloat16),
            self.projection.weight.to(torch.bfloat16),
            self.projection.bias.to(torch.float32),
            self.gate.weight.to(torch.bfloat16),
            self.gate.bias.to(torch.float32),
            self.center_norm.weight.to(torch.bfloat16),
            self.center_norm.bias.to(torch.bfloat16),
            self.output_projection.weight.to(torch.bfloat16),
            self.output_projection.bias.to(torch.float32),
            self.gating_linear.weight.to(torch.bfloat16),
            self.gating_linear.bias.to(torch.float32),
        )
    else:
        act = FusedTriangleMultiplicationFunction.apply(
            act,
            mask,
            self.c_equation,
            self.left_norm_input.weight,
            self.left_norm_input.bias,
            self.projection.weight,
            self.projection.bias,
            self.gate.weight,
            self.gate.bias,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection.weight,
            self.output_projection.bias,
            self.gating_linear.weight,
            self.gating_linear.bias,
        )
    return act


class FusedTriangleMultiplicationOpti(nn.Module):

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
        # self.num_intermediate_channel = num_intermediate_channel
        self.left_norm_input = nn.LayerNorm(act_dim)
        self.projection = nn.Linear(act_dim, 2 * num_intermediate_channel)
        self.gate = nn.Linear(num_intermediate_channel, 2 * num_intermediate_channel)
        self.center_norm = nn.LayerNorm(num_intermediate_channel)
        self.output_projection = nn.Linear(num_intermediate_channel, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask):
        mask = mask[..., None]
        # left_act = self.left_norm_input(act)
        # proj_act = mask * self.projection(left_act)
        # proj_act *= torch.sigmoid(self.gate(left_act))
        # left_proj_act = proj_act[:, :, :self.num_intermediate_channel]
        # right_proj_act = proj_act[:, :, self.num_intermediate_channel:]
        # act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
        # act = self.center_norm(act)
        # act = self.output_projection(act)
        # act *= torch.sigmoid(self.gating_linear(left_act))
        if (
            act.dtype == torch.bfloat16
            or mask.dtype == torch.bfloat16
            or self.left_norm_input.weight.dtype == torch.bfloat16
            or self.left_norm_input.bias.dtype == torch.bfloat16
            or self.projection.weight.dtype == torch.bfloat16
            or self.projection.bias.dtype == torch.bfloat16
            or self.gate.weight.dtype == torch.bfloat16
            or self.gate.bias.dtype == torch.bfloat16
            or self.center_norm.weight.dtype == torch.bfloat16
            or self.center_norm.bias.dtype == torch.bfloat16
            or self.output_projection.weight.dtype == torch.bfloat16
            or self.output_projection.bias.dtype == torch.bfloat16
            or self.gating_linear.weight.dtype == torch.bfloat16
            or self.gating_linear.bias.dtype == torch.bfloat16
        ):
            act = FusedTriangleMultiplicationFunction.apply(
                act.to(torch.bfloat16),
                mask.to(torch.float32),
                self.c_equation,
                self.left_norm_input.weight.to(torch.bfloat16),
                self.left_norm_input.bias.to(torch.bfloat16),
                self.projection.weight.to(torch.bfloat16),
                self.projection.bias.to(torch.float32),
                self.gate.weight.to(torch.bfloat16),
                self.gate.bias.to(torch.float32),
                self.center_norm.weight.to(torch.bfloat16),
                self.center_norm.bias.to(torch.bfloat16),
                self.output_projection.weight.to(torch.bfloat16),
                self.output_projection.bias.to(torch.float32),
                self.gating_linear.weight.to(torch.bfloat16),
                self.gating_linear.bias.to(torch.float32),
            )
        else:
            act = FusedTriangleMultiplicationFunction.apply(
                act,
                mask,
                self.c_equation,
                self.left_norm_input.weight,
                self.left_norm_input.bias,
                self.projection.weight,
                self.projection.bias,
                self.gate.weight,
                self.gate.bias,
                self.center_norm.weight,
                self.center_norm.bias,
                self.output_projection.weight,
                self.output_projection.bias,
                self.gating_linear.weight,
                self.gating_linear.bias,
            )
        return act
