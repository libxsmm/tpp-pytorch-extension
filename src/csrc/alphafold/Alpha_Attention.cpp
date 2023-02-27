/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <iostream>
#include <tuple>

#include <ATen/record_function.h>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

#define QKV_BLOCKSIZE 32
#define A_BLOCKSIZE 32
#define C_BLOCKSIZE 32

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_a_gemm, "alpha_a_gemm");
REGISTER_SCOPE(alpha_c_gemm, "alpha_c_gemm");

at::Tensor fused_gating_attention_fwd(
    at::Tensor& q_data,
    at::Tensor& m_data,
    at::Tensor& bias,
    at::Tensor& nonbatched_bias,
    at::Tensor& query_w,
    at::Tensor& key_w,
    at::Tensor& value_w,
    at::Tensor& gating_w,
    at::Tensor& gating_b,
    at::Tensor& output_w,
    at::Tensor& output_b,
    int key_dim,
    int value_dim) {
  GlobalPass _gp(FWD);
  if (q_data.dtype() == at::kFloat) {
    typedef float T;
#include "fused_gating_attention_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_gating_attention_fwd_tmpl_bf16.h"
  }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
REGISTER_SUBMODULE(_alpha_attention, m) {
  m.def("forward", &fused_gating_attention_fwd, "Gating attention forward");
}
