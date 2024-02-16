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
REGISTER_SCOPE(gobias, "gobias");
REGISTER_SCOPE(gadbias, "gadbias");
REGISTER_SCOPE(gao_dropout, "gao_dropout");
REGISTER_SCOPE(gado_dropout, "gado_dropout");
REGISTER_SCOPE(go_lrelu, "go_lrelu");
REGISTER_SCOPE(gdo_lrelu, "gdo_lrelu");
REGISTER_SCOPE(go_relu, "go_relu");
REGISTER_SCOPE(gdo_relu, "gdo_relu");
REGISTER_SCOPE(go_bias_relu, "go_bias_relu");
REGISTER_SCOPE(gdo_bias_relu, "gdo_bias_relu");
REGISTER_SCOPE(go_add_bias, "go_add_bias");
REGISTER_SCOPE(gdo_add_bias, "gdo_add_bias");
REGISTER_SCOPE(go_attn, "go_attn");
REGISTER_SCOPE(gdo_attn, "gdo_attn");
REGISTER_SCOPE(go_mlp_attn, "go_mlp_attn");
REGISTER_SCOPE(gdo_mlp_attn, "gdo_mlp_attn");
REGISTER_SCOPE(go_mlp, "go_mlp");
REGISTER_SCOPE(gdo_mlp, "gdo_mlp");
REGISTER_SCOPE(ga_dattn, "ga_dattn");
REGISTER_SCOPE(ga_dbias, "ga_dbias");
REGISTER_SCOPE(ga_din, "ga_din");
REGISTER_SCOPE(ga_fused_dattn, "ga_fused_dattn");
REGISTER_SCOPE(ga_fused_dbias, "ga_fused_dbias");
REGISTER_SCOPE(ga_fused_din, "ga_fused_din");
REGISTER_SCOPE(gao_gemm_attn, "gao_gemm_attn");
REGISTER_SCOPE(gadi_gemm_attn, "gadi_gemm_attn");
REGISTER_SCOPE(gadw_gemm_attn, "gadw_gemm_attn");
REGISTER_SCOPE(gao_relu_drop, "gao_relu_drop");
REGISTER_SCOPE(gado_relu_drop, "gado_relu_drop");
REGISTER_SCOPE(gao_lrelu_drop, "gao_lrelu_drop");
REGISTER_SCOPE(gado_lrelu_drop, "gado_lrelu_drop");
REGISTER_SCOPE(go_bias_relu_drop, "gao_bias_relu_drop");
REGISTER_SCOPE(gdo_bias_relu_drop, "gado_bias_relu_drop");
REGISTER_SCOPE(go_bias_lrelu_drop, "gao_bias_lrelu_drop");
REGISTER_SCOPE(gdo_bias_lrelu_drop, "gado_bias_lrelu_drop");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################

std::vector<at::Tensor> fused_gat_mlp_attn_fwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gat_mlp_attn_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gat_mlp_attn_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_gat_mlp_attn_bwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gat_mlp_attn_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gat_mlp_attn_flat_bwd.h"
  }
}

at::Tensor fused_mlp_fwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_mlp_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_mlp_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_mlp_bwd(
    long align,
    int add_bias,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_mlp_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_mlp_flat_bwd.h"
  }
}

at::Tensor attn_fwd(long align, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "attn_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "attn_flat_fwd.h"
  }
}

std::vector<at::Tensor> attn_bwd(long align, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "attn_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "attn_flat_bwd.h"
  }
}

at::Tensor gemm_attn_fwd(long align, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "gemm_attn_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "gemm_attn_flat_fwd.h"
  }
}

std::vector<at::Tensor> gemm_attn_bwd(
    long align,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "gemm_attn_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "gemm_attn_flat_bwd.h"
  }
}

// ######################################## Dropout
// ################################################

std::vector<at::Tensor> gat_dropout_fwd(
    float p,
    at::Tensor inp,
    bool training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "dropout_fwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_fwd.h"
  }
}

at::Tensor gat_dropout_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_bwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_bwd.h"
  }
}

// ######################################## Leaky ReLU
// ################################################

std::vector<at::Tensor> leakyrelu_fwd(float alpha, at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    // std::cout << " In Cpp "<< inp << std::endl;
    typedef float T;
#include "leakyrelu_fwd.h"
  } else {
    typedef bfloat16 T;
#include "leakyrelu_fwd.h"
  }
}

at::Tensor leakyrelu_bwd(float alpha, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "leakyrelu_bwd.h"
  } else {
    typedef bfloat16 T;
#include "leakyrelu_bwd.h"
  }
}

// ######################################## ReLU
// ################################################

std::vector<at::Tensor> relu_fwd(at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "relu_fwd.h"
  } else {
    typedef bfloat16 T;
#include "relu_fwd.h"
  }
}

at::Tensor relu_bwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "relu_bwd.h"
  } else {
    typedef bfloat16 T;
#include "relu_bwd.h"
  }
}

// ######################################## Fused Bias Add with ReLU
// ################################################

std::vector<at::Tensor> bias_relu_fwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_relu_fwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_relu_fwd.h"
  }
}

std::vector<at::Tensor> bias_relu_bwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_relu_bwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_relu_bwd.h"
  }
}

// ######################################## Fused Dropout with ReLU
// ################################################
std::vector<at::Tensor> relu_drop_fwd(float p, at::Tensor inp, int training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "relu_drop_fwd.h"
  } else {
    typedef bfloat16 T;
#include "relu_drop_fwd.h"
  }
}

at::Tensor relu_drop_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "relu_drop_bwd.h"
  } else {
    typedef bfloat16 T;
#include "relu_drop_bwd.h"
  }
}

// ######################################## Fuse Bias, ReLU, Dropout
// ################################################
std::vector<at::Tensor> bias_relu_drop_fwd(
    std::vector<at::Tensor> inputs,
    float p,
    int training) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_relu_drop_fwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_relu_drop_fwd.h"
  }
}

std::vector<at::Tensor> bias_relu_drop_bwd(
    std::vector<at::Tensor> inputs,
    float p) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_relu_drop_bwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_relu_drop_bwd.h"
  }
}

// ######################################## Fused Dropout with LeakyReLU
// ################################################
std::vector<at::Tensor> leaky_relu_drop_fwd(
    float alpha,
    float p,
    at::Tensor inp,
    int training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "leaky_relu_drop_fwd.h"
  } else {
    typedef bfloat16 T;
#include "leaky_relu_drop_fwd.h"
  }
}

at::Tensor leaky_relu_drop_bwd(
    float alpha,
    float p,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "leaky_relu_drop_bwd.h"
  } else {
    typedef bfloat16 T;
#include "leaky_relu_drop_bwd.h"
  }
}

// ######################################## Fuse Bias, LeakyReLU, Dropout
// ################################################
std::vector<at::Tensor> bias_lrelu_drop_fwd(
    std::vector<at::Tensor> inputs,
    float alpha,
    float p,
    int training) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_lrelu_drop_fwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_lrelu_drop_fwd.h"
  }
}

std::vector<at::Tensor> bias_lrelu_drop_bwd(
    std::vector<at::Tensor> inputs,
    float alpha,
    float p) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "bias_lrelu_drop_bwd.h"
  } else {
    typedef bfloat16 T;
#include "bias_lrelu_drop_bwd.h"
  }
}

// ######################################## Bias Add
// ################################################

at::Tensor add_bias_fwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "add_bias_fwd.h"
  } else {
    typedef bfloat16 T;
#include "add_bias_fwd.h"
  }
}

at::Tensor add_bias_bwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "add_bias_bwd.h"
  } else {
    typedef bfloat16 T;
#include "add_bias_bwd.h"
  }
}

REGISTER_SUBMODULE(_fused_gat, m) {
  m.def(
      "fused_gat_mlp_attn_fwd",
      &fused_gat_mlp_attn_fwd,
      "Tpp fused MLP-Attention forward");
  m.def(
      "fused_gat_mlp_attn_bwd",
      &fused_gat_mlp_attn_bwd,
      "Tpp fused MLP-Attention backward");
  m.def("fused_mlp_fwd", &fused_mlp_fwd, "Tpp fused MLP forward");
  m.def("fused_mlp_bwd", &fused_mlp_bwd, "Tpp GAT fused MLP backward");
  m.def("attn_fwd", &attn_fwd, "Tpp Attention forward");
  m.def("attn_bwd", &attn_bwd, "Tpp Attention backward");
  m.def("gemm_attn_fwd", &gemm_attn_fwd, "Tpp GEMM Attention forward");
  m.def("gemm_attn_bwd", &gemm_attn_bwd, "Tpp GEMM Attention backward");
  m.def("gat_dropout_fwd", &gat_dropout_fwd, "Tpp Optimized Dropout FWD");
  m.def("gat_dropout_bwd", &gat_dropout_bwd, "Tpp Optimized Dropout BWD");
  m.def("leakyrelu_fwd", &leakyrelu_fwd, "Tpp Optimized Leaky Relu FWD");
  m.def("leakyrelu_bwd", &leakyrelu_bwd, "Tpp Optimized Leaky Relu BWD");
  m.def("relu_fwd", &relu_fwd, "Tpp Optimized Relu FWD");
  m.def("relu_bwd", &relu_bwd, "Tpp Optimized Relu BWD");
  m.def("bias_relu_fwd", &bias_relu_fwd, "Tpp Optimized Bias+Relu FWD");
  m.def("bias_relu_bwd", &bias_relu_bwd, "Tpp Optimized Relu+GradBias BWD");
  m.def("add_bias_fwd", &add_bias_fwd, "Tpp Optimized Bias Add FWD");
  m.def("add_bias_bwd", &add_bias_bwd, "Tpp Optimized GradBias BWD");
  m.def("relu_drop_fwd", &relu_drop_fwd, "Fused ReLU + Dropout FWD");
  m.def("relu_drop_bwd", &relu_drop_bwd, "Fused Relu + Dropout BWD");
  m.def(
      "leaky_relu_drop_fwd",
      &leaky_relu_drop_fwd,
      "Fused Leaky ReLU + Dropout FWD");
  m.def(
      "leaky_relu_drop_bwd",
      &leaky_relu_drop_bwd,
      "Fused Leaky Relu + Dropout BWD");
  m.def("bias_relu_drop_fwd", &bias_relu_drop_fwd, "Fuse Bias,ReLU,Dropout");
  m.def("bias_relu_drop_bwd", &bias_relu_drop_bwd, "Fuse Bias,ReLU,Dropout");
  m.def(
      "bias_lrelu_drop_fwd",
      &bias_lrelu_drop_fwd,
      "Fuse Bias,LeakyReLU,Dropout");
  m.def(
      "bias_lrelu_drop_bwd",
      &bias_lrelu_drop_bwd,
      "Fuse Bias,LeakyReLU,Dropout");
}
