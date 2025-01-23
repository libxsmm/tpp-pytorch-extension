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
REGISTER_SCOPE(mlp_attn, "mlp_attn");
REGISTER_SCOPE(mlp, "mlp");
REGISTER_SCOPE(attn, "attn");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################



static std::vector<at::Tensor> gather_mlp_attn(
    std::vector<at::Tensor> inp) {
  GlobalPass _gp(FWD);

  at::Tensor t_in = inp[0];
  at::Tensor t_idx = inp[1];
  at::Tensor t_wt = inp[2];
  at::Tensor t_bias = inp[3];
  at::Tensor t_attn_3d = inp[4];

  auto wt_sizes = t_wt.sizes();
  auto K = wt_sizes[0] * wt_sizes[3];
  auto BS = t_idx.size(0);
  auto out_dt = t_in.is_quantized() ? at::kBFloat16 : t_in.scalar_type();
  auto t_out = t_in.new_empty({BS, K}, out_dt);

  auto attn_sizes = t_attn_3d.sizes(); 
  auto H = attn_sizes[1]; 
  auto F = attn_sizes[2];
  auto t_out_attn = t_out.new_empty({BS, H});

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    typedef float T;
    if(t_idx.dtype() == at::kInt) {
      typedef int Tidx;
#include "gather_mlp_attn.h"
    }
    else if(t_idx.dtype() == at::kLong) {
      typedef long Tidx;
#include "gather_mlp_attn.h"
    }
  } else if (dt == at::kBFloat16) {
    typedef bfloat16 T;
    if(t_idx.dtype() == at::kInt) {
      typedef int Tidx;
#include "gather_mlp_attn.h"
    }
    else if(t_idx.dtype() == at::kLong) {
      typedef long Tidx;
#include "gather_mlp_attn.h"
    }
  } else if (dt == at::kBFloat8) {
    typedef bfloat8 T;
    if(t_idx.dtype() == at::kInt) {
      typedef int Tidx;
#include "gather_mlp_attn.h"
    }
    else if(t_idx.dtype() == at::kLong) {
      typedef long Tidx;
#include "gather_mlp_attn.h"
    }
  } else if (dt == at::kQInt8) {
    typedef int8_t T;
    typedef bfloat16 Tout;
    if(t_idx.dtype() == at::kInt) {
      typedef int Tidx;
#include "gather_mlp_attn_quant.h"
    }
    else if(t_idx.dtype() == at::kLong) {
      typedef long Tidx;
#include "gather_mlp_attn_quant.h"
    }
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return {t_out, t_out_attn.view({BS, H, 1})};
}

std::vector<at::Tensor> mlp_attn(
    std::vector<at::Tensor> inp) {
  GlobalPass _gp(FWD);

  at::Tensor t_in = inp[0];
  auto dt = t_in.dtype();

  if(dt==at::kFloat) {
    typedef float Tact;
#include "mlp_attn.h"
  } else if(dt == at::kBFloat16 || dt == at::kQInt8) {
    typedef bfloat16 Tact;
#include "mlp_attn.h"
  } else if(dt == at::kFloat8_e4m3fn) {
    typedef hfloat8 Tact;
#include "mlp_attn.h"
  } else if(dt == at::kFloat8_e5m2) {
    typedef bfloat8 Tact;
#include "mlp_attn.h"
  } else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

at::Tensor mlp(std::vector<at::Tensor> inp) {
  GlobalPass _gp(FWD);

  at::Tensor t_in = inp[0];
  at::Tensor t_wt = inp[1];
  at::Tensor t_bias = inp[2];
  auto dt = t_wt.dtype();

  if(dt == at::kFloat) {
    at::Tensor t_out = fc_plain<float>(t_in, t_wt, t_bias);
    return t_out;
  }
  else if(dt == at::kBFloat16) {
    at::Tensor t_out = fc_plain<bfloat16>(t_in, t_wt, t_bias);
    return t_out;
  }
  else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
  }
}

REGISTER_SUBMODULE(_fused_gat_inf, m) {
  m.def("gather_mlp_attn", &gather_mlp_attn, "TPP fused Gather with MLP-Attention");
  m.def("mlp_attn", &mlp_attn, "TPP fused MLP-Attention");
  m.def("mlp", &mlp, "TPP fused MLP");
}
