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

REGISTER_SCOPE(go_lrelu, "go_lrelu");
REGISTER_SCOPE(gdo_lrelu, "gdo_lrelu");
REGISTER_SCOPE(go_relu, "go_relu");
REGISTER_SCOPE(gdo_relu, "gdo_relu");
REGISTER_SCOPE(go_add_bias, "go_add_bias");
REGISTER_SCOPE(gdo_add_bias, "gdo_add_bias");
REGISTER_SCOPE(go_relu_drop, "go_relu_drop");
REGISTER_SCOPE(gdo_relu_drop, "gdo_relu_drop");
REGISTER_SCOPE(go_lrelu_drop, "go_lrelu_drop");
REGISTER_SCOPE(gdo_lrelu_drop, "gdo_lrelu_drop");
REGISTER_SCOPE(go_bias_relu_drop, "go_bias_relu_drop");
REGISTER_SCOPE(gdo_bias_relu_drop, "gdo_bias_relu_drop");
REGISTER_SCOPE(go_bias_lrelu_drop, "go_bias_lrelu_drop");
REGISTER_SCOPE(gdo_bias_lrelu_drop, "gdo_bias_lrelu_drop");
REGISTER_SCOPE(go_bias_relu, "go_bias_relu");
REGISTER_SCOPE(gdo_bias_relu, "gdo_bias_relu");
REGISTER_SCOPE(go_bias_lrelu, "go_bias_lrelu");
REGISTER_SCOPE(gdo_bias_lrelu, "gdo_bias_lrelu");

// ######################################## Leaky ReLU
// ################################################

std::vector<at::Tensor> leakyrelu_fwd(float alpha, at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "leakyrelu_fwd.h"
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "leakyrelu_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

at::Tensor leakyrelu_bwd(float alpha, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "leakyrelu_bwd.h"
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "leakyrelu_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## ReLU
// ################################################

std::vector<at::Tensor> relu_fwd(at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "relu_fwd.h"
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "relu_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

at::Tensor relu_bwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "relu_bwd.h"
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "relu_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## Fused Bias Add with ReLU
// ################################################

std::vector<at::Tensor> bias_relu_fwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_relu_fwd.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu_fwd.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu_fwd.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> bias_relu_bwd(
    std::vector<at::Tensor> inputs,
    int dparam) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (dparam == 0) {
      typedef float Tprm;
#include "bias_relu_bwd.h"
    } else if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_relu_bwd.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_relu_bwd.h"
    } else {
      TPP_ASSERT(0, "%s:%d Unsupported type for bias\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported typei activations\n", __FILE__, __LINE__);
  }
}

// ######################################## Fused Dropout with ReLU
// ################################################
std::vector<at::Tensor> relu_drop_fwd(float p, at::Tensor inp, int training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "relu_drop_fwd.h"
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "relu_drop_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

at::Tensor relu_drop_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "relu_drop_bwd.h"
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "relu_drop_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
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
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_relu_drop_fwd.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu_drop_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_relu_drop_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> bias_relu_drop_bwd(
    std::vector<at::Tensor> inputs,
    float p,
    int dparam) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (dparam == 0) {
      typedef float Tprm;
#include "bias_relu_drop_bwd.h"
    } else if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_relu_drop_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_relu_drop_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

// ######################################## Fused Bias Add with LeakyReLU
// ################################################

std::vector<at::Tensor> bias_lrelu_fwd(
    int align,
    std::vector<at::Tensor> inputs,
    float alpha) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_lrelu_fwd.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> bias_lrelu_bwd(
    std::vector<at::Tensor> inputs,
    float alpha,
    int dparam) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (dparam == 0) {
      typedef float Tprm;
#include "bias_lrelu_bwd.h"
    } else if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
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
  } else if (inp.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "leaky_relu_drop_fwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
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
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "leaky_relu_drop_bwd.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

// ######################################## Fuse Bias, LeakyReLU, Dropout
// ################################################
std::vector<at::Tensor> bias_lrelu_drop_fwd(
    int align,
    std::vector<at::Tensor> inputs,
    float alpha,
    float p,
    int training) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "bias_lrelu_drop_fwd.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_drop_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_drop_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> bias_lrelu_drop_bwd(
    int align,
    std::vector<at::Tensor> inputs,
    float alpha,
    float p,
    int dparam) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (dparam == 0) {
      typedef float Tprm;
#include "bias_lrelu_drop_bwd.h"
    } else if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_drop_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "bias_lrelu_drop_bwd.h"
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

at::Tensor add_bias_fwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (inputs[1].dtype() == at::kFloat) {
      typedef float Tprm;
#include "add_bias_fwd.h"
    } else if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "add_bias_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (inputs[1].dtype() == at::kBFloat16) {
      typedef bfloat16 Tprm;
#include "add_bias_fwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

at::Tensor add_bias_bwd(std::vector<at::Tensor> inputs, int dparam) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float Tact;
    if (dparam == 0) {
      typedef float Tprm;
#include "add_bias_bwd.h"
    } else if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "add_bias_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 Tact;
    if (dparam == 1) {
      typedef bfloat16 Tprm;
#include "add_bias_bwd.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
  }
}

REGISTER_SUBMODULE(_fused_ops, m) {
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
  m.def("bias_lrelu_fwd", &bias_lrelu_fwd, "Fuse Bias,LeakyReLU, No-dropout");
  m.def(
      "bias_lrelu_drop_fwd",
      &bias_lrelu_drop_fwd,
      "Fuse Bias,LeakyReLU,Dropout");
  m.def(
      "bias_lrelu_drop_bwd",
      &bias_lrelu_drop_bwd,
      "Fuse Bias,LeakyReLU,Dropout");
}
