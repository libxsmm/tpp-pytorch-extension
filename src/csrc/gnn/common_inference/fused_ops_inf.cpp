/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Sasikanth Avancha (Intel Corp.)
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

REGISTER_SCOPE(o_lrelu, "o_lrelu");
REGISTER_SCOPE(o_relu, "o_relu");
REGISTER_SCOPE(o_add_bias, "o_add_bias");
REGISTER_SCOPE(o_bias_relu, "o_bias_relu");
REGISTER_SCOPE(o_bias_lrelu, "o_bias_lrelu");

// ######################################## Leaky ReLU
// ################################################

std::vector<at::Tensor> leakyrelu(float alpha, at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "leakyrelu.h"
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "leakyrelu.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## ReLU
// ################################################

std::vector<at::Tensor> relu(at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "relu.h"
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "relu.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## Fused Bias Add with ReLU
// ################################################

std::vector<at::Tensor> bias_relu(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_relu.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

// ######################################## Fused Bias Add with LeakyReLU
// ################################################

std::vector<at::Tensor> bias_lrelu(
    std::vector<at::Tensor> inputs,
    float alpha) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_lrelu.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## Bias Add
// ################################################

at::Tensor add_bias(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "add_bias.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "add_bias.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "add_bias.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

REGISTER_SUBMODULE(_fused_ops_inf, m) {
  m.def("leakyrelu", &leakyrelu, "Tpp Optimized Leaky Relu FWD");
  m.def("relu", &relu, "Tpp Optimized Relu FWD");
  m.def("bias_relu", &bias_relu, "Tpp Optimized Bias+Relu FWD");
  m.def("bias_lrelu", &bias_lrelu, "Tpp Optimized Bias+LeakyRelu FWD");
  m.def("add_bias", &add_bias, "Tpp Optimized Bias Add FWD");
}
