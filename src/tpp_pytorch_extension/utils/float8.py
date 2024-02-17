###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import torch

pytorch_supports_float8 = hasattr(torch, "float8_e5m2") and hasattr(
    torch, "float8_e4m3fn"
)

if pytorch_supports_float8:
    torch.bfloat8 = torch.float8_e5m2
    torch.hfloat8 = torch.float8_e4m3fn
else:
    torch.bfloat8 = None
    torch.hfloat8 = None


def cvt_to(self, dtype):
    # print(f"CVT_TO called: self.dtype: {self.dtype}, dtype: {dtype}")
    if self.dtype == dtype:
        return self
    return self.to(dtype)


torch.Tensor.cvt_to = cvt_to
