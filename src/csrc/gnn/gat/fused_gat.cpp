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

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(gao_gemm, "gao_gemm");
REGISTER_SCOPE(gadi_gemm, "gadi_gemm");
REGISTER_SCOPE(gadw_gemm, "gadw_gemm");
REGISTER_SCOPE(gmo_gemm, "gmo_gemm");
REGISTER_SCOPE(gmdi_gemm, "gmdi_gemm");
REGISTER_SCOPE(gmdw_gemm, "gmdw_gemm");
REGISTER_SCOPE(gobias, "gobias");
REGISTER_SCOPE(gm_dbias, "gm_dbias");
REGISTER_SCOPE(go_attn, "go_attn");
REGISTER_SCOPE(gdo_attn, "gdo_attn");
REGISTER_SCOPE(go_mlp_attn, "go_mlp_attn");
REGISTER_SCOPE(gdo_mlp_attn, "gdo_mlp_attn");
REGISTER_SCOPE(go_mlp, "go_mlp");
REGISTER_SCOPE(gdo_mlp, "gdo_mlp");
REGISTER_SCOPE(ga_dattn, "ga_dattn");
REGISTER_SCOPE(ga_fused_dattn, "ga_fused_dattn");
REGISTER_SCOPE(ga_fused_dbias, "ga_fused_dbias");
REGISTER_SCOPE(ga_fused_din, "ga_fused_din");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################

std::vector<at::Tensor> mlp_attn_fwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "mlp_attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "mlp_attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "mlp_attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "mlp_attn_flat_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> mlp_attn_bwd(
    long align,
    int inp_needs_grad,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "mlp_attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "mlp_attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "mlp_attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "mlp_attn_flat_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

at::Tensor mlp_fwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "mlp_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "mlp_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "mlp_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "mlp_flat_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> mlp_bwd(
    long align,
    int inp_needs_grad,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "mlp_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "mlp_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "mlp_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "mlp_flat_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

at::Tensor attn_fwd(
    long align,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "attn_flat_fwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "attn_flat_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> attn_bwd(
    long align,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e5m2) {
    typedef bfloat8 T;
#include "attn_flat_bwd.h"
  } else if(inputs[0].dtype() == at::kFloat8_e4m3fn) {
    typedef hfloat8 T;
#include "attn_flat_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}


REGISTER_SUBMODULE(_fused_gat, m) {
  m.def(
      "mlp_attn_fwd",
      &mlp_attn_fwd,
      "Tpp fused MLP-Attention forward");
  m.def(
      "mlp_attn_bwd",
      &mlp_attn_bwd,
      "Tpp fused MLP-Attention backward");
  m.def("mlp_fwd", &mlp_fwd, "Tpp fused MLP forward");
  m.def("mlp_bwd", &mlp_bwd, "Tpp GAT fused MLP backward");
  m.def("attn_fwd", &attn_fwd, "Tpp Attention forward");
  m.def("attn_bwd", &attn_bwd, "Tpp Attention backward");
}
