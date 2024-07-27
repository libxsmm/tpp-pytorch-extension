/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"
#include "fused_gemm.h"
#include "qtypes.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(o_mlp_attn, "o_mlp_attn");
REGISTER_SCOPE(o_mlp, "o_mlp");
REGISTER_SCOPE(o_attn, "o_attn");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################

std::vector<at::Tensor> mlp_attn(
    long align,
    int add_bias,
    int use_qint8_gemm,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);

  if(!use_qint8_gemm) {
    auto dact = -1;
    if(inputs[0].dtype() == at::kChar) dact=0;
    else if(inputs[0].dtype() == at::kFloat8_e4m3fn) dact=1;
    else if(inputs[0].dtype() == at::kFloat8_e5m2) dact=2;
    else if(inputs[0].dtype() == at::kBFloat16) dact=3;

    assert(inputs[2].dtype() == at::kBFloat16);
    assert(inputs[3].dtype() == at::kBFloat16);

    if(dact==0) {
      typedef int8_t Tact;
#include "mlp_attn_scf.h"
    }
    else if(dact==1) {
      typedef hfloat8 Tact;
#include "mlp_attn.h"
    }
    else if(dact==2) {
      typedef bfloat8 Tact;
#include "mlp_attn.h"
    }
    else if(dact==3) {
      typedef bfloat16 Tact;
#include "mlp_attn.h"
    }
    else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
    }
  }
  else {
#include "mlp_attn_qint8.h"
  }
}

at::Tensor mlp(long align, int add_bias, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);

  auto dact=-1;
  if(inputs[0].dtype() == at::kFloat) dact=0;
  else if(inputs[0].dtype() == at::kBFloat16) dact=1;

  auto dwt = -1;
  if (inputs[1].dtype() == at::kFloat)
    dwt = 0;
  else if (inputs[1].dtype() == at::kBFloat16)
    dwt = 1;

  if (dact == 0) {
    typedef float Tact;
    if (dwt == 0) {
      typedef float Tprm;
#include "mlp.h"
    } else if (dwt == 1) {
      typedef bfloat16 Tprm;
#include "mlp.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  }
  else if (dact == 1) {
    typedef bfloat16 Tact;
    if (dwt == 1) {
      typedef bfloat16 Tprm;
#include "mlp.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  }
  else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

REGISTER_SUBMODULE(_fused_gat_inf, m) {
  m.def("mlp_attn", &mlp_attn, "TPP fused MLP-Attention");
  m.def("mlp", &mlp, "TPP fused MLP");
}
