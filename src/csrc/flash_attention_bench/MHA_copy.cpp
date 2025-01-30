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
// #include <torch/extension.h>
#include <cmath>
#include <iostream>
#include <tuple>

// #include <ATen/record_function.h>
#include "ext_tpp.h"
// #include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

#define QKV_BLOCKSIZE 64
#define A_BLOCKSIZE 64
#define Ak_BLOCKSIZE 512
#define C_BLOCKSIZE 64

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_ac_gemm, "alpha_ac_gemm");
REGISTER_SCOPE(alpha_o_gemm, "alpha_o_gemm");

REGISTER_SCOPE(proj_gemm, "proj_gemm");
REGISTER_SCOPE(out_gemm, "out_gemm");
REGISTER_SCOPE(gate_gemm, "gate_gemm");
REGISTER_SCOPE(eq_bmm, "eq_gemm");
REGISTER_SCOPE(layer_norm_input, "layer_norm_input");
REGISTER_SCOPE(left_norm_input, "left_norm_input");

float* fused_gating_attention_fwd_fp32(
    float* q_data,
    float* m_data,
    float* bias,
    float* nonbatched_bias,
    float* query_w,
    float* key_w,
    float* value_w,
    float* gating_w,
    float* gating_b,
    float* output_w,
    float* output_b,
    float* output_a
    int64_t B_t,
    int64_t Sp_t,
    int64_t N_t,
    int64_t H_t,
    int64_t HS_t) {
  GlobalPass _gp(FWD);
    typedef float T;
#include "fused_gating_attention_fwd_tmpl.h"
}

// at::Tensor fused_gating_attention_fwd(
//     at::Tensor& q_data,
//     at::Tensor& m_data,
//     at::Tensor& bias,
//     at::Tensor& nonbatched_bias,
//     at::Tensor& query_w,
//     at::Tensor& key_w,
//     at::Tensor& value_w,
//     at::Tensor& gating_w,
//     at::Tensor& gating_b,
//     at::Tensor& output_w,
//     at::Tensor& output_b,
//     int key_dim,
//     int value_dim) {
//   GlobalPass _gp(FWD);
//   if (q_data.dtype() == at::kFloat) {
//     typedef float T;
// #include "fused_gating_attention_fwd_tmpl.h"
//   } else {
//     typedef bfloat16 T;
// #include "fused_gating_attention_fwd_tmpl_bf16.h"
//   }
// }

// // PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// REGISTER_SUBMODULE(_alpha_attention, m) {
//   m.def("forward", &fused_gating_attention_fwd, "Gating attention forward");
// }


// Main function that takes Batch size, Sequence length, Number of heads, Head size as input

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias>" << std::endl;
    return 1;
  }

  int batch_size = std::stoi(argv[1]);
  int seq_len = std::stoi(argv[2]);
  int num_heads = std::stoi(argv[3]);
  int head_size = std::stoi(argv[4]);
  int embedding_dim = num_heads * head_size;
  bool bias = std::stoi(argv[5]);
  bool nb_bias = std::stoi(argv[6]);

  // Allocate memory for input tensors
  float* q_data_a = new float[batch_size][seq_len][embedding_dim];
  float* m_data_a = new float[batch_size][seq_len][embedding_dim];
  float* bias_a = new float[batch_size][seq_len];
  float* nonbatched_bias_a = new float[1][num_heads][seq_len][seq_len];
  float* query_w_a = new float[embedding_dim][num_heads][head_size];
  float* key_w_a = new float[embedding_dim][num_heads][head_size];
  float* value_w_a = new float[embedding_dim][num_heads][head_size];
  float* gating_w_a = new float[embedding_dim][num_heads][head_size];
  float* gating_b_a = new float[num_heads][head_size];
  float* output_w_a = new float[num_heads][head_size][embedding_dim];
  float* output_b_a = new float[embedding_dim][1];

  float* output_a = new float[batch_size][seq_len][embedding_dim];

  // Initialize input tensors with random values
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < embedding_dim; ++k) {
        q_data_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
        m_data_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      bias_a[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  for (int i = 0; i < num_heads; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < seq_len; ++k) {
        nonbatched_bias_a[0][i][j][k] = static_cast<float>(rand()) / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < embedding_dim; ++i) {
    for (int j = 0; j < num_heads; ++j) {
      for (int k = 0; k < head_size; ++k) {
        query_w_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
        key_w_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
        value_w_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
        gating_w_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < num_heads; ++i) {
    for (int j = 0; j < head_size; ++j) {
      gating_b_a[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  for (int i = 0; i < num_heads; ++i) {
    for (int j = 0; j < head_size; ++j) {
      for (int k = 0; k < embedding_dim; ++k) {
        output_w_a[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
      }
    }
  }
  for (int i = 0; i < embedding_dim; ++i) {
    output_b_a[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Run the forward function
  auto output = fused_gating_attention_fwd_fp32(
      q_data_a, m_data_a, bias_a, nonbatched_bias_a, 
      query_w_a, key_w_a, value_w_a, gating_w_a, gating_b_a, 
      output_w_a, output_b_a, output_a, 
      batch_size, seq_len, num_heads, head_size, embedding_dim);

  delete[] q_data_a;
  delete[] m_data_a;
  delete[] bias_a;
  delete[] nonbatched_bias_a;
  delete[] query_w_a;
  delete[] key_w_a;
  delete[] value_w_a;
  delete[] gating_w_a;
  delete[] gating_b_a;
  delete[] output_w_a;
  delete[] output_b_a;
  delete[] output_a;

  return 0;
}



