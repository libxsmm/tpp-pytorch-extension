###############################################################################
# Copyright (c) 2024 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import torch
from tpp_pytorch_extension._C import _qtype as qtype
from tpp_pytorch_extension._C._qtype import (
    q_per_block_scales,
    q_per_block_block_size,
    q_per_block_axis,
    remap_and_quantize_mxfp4,
    remap_and_quantize_qint8,
    remap_and_quantize_qint4,
    quantize_mxfp4,
    quantize_int8sym,
    quantize_int4sym,
    create_qtensor_int8sym,
)


def q_per_block_qval(qtensor: torch.Tensor):
    qval = torch.tensor(qtensor.untyped_storage(), dtype=torch.int8).view_as(qtensor)
    return qval


def get_qval(qtensor: torch.Tensor):
    return q_per_block_qval(qtensor)


def get_scales(qtensor: torch.Tensor):
    return q_per_block_scales(qtensor)


def get_axis(qtensor: torch.Tensor):
    return q_per_block_axis(qtensor)


def get_block_size(qtensor: torch.Tensor):
    return q_per_block_block_size(qtensor)
