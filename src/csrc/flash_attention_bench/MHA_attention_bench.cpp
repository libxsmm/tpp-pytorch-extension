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
// #include "tensor_helper.h"

#define QKV_BLOCKSIZE 64
#define A_BLOCKSIZE 64
#define Ak_BLOCKSIZE 512
#define C_BLOCKSIZE 64

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_ac_gemm, "alpha_ac_gemm");
REGISTER_SCOPE(alpha_o_gemm, "alpha_o_gemm");

template<typename T> 
void fused_gating_attention_fwd_fp32(
    T* q_data,
    T* m_data,
    T* bias,
    T* nonbatched_bias,
    T* query_w,
    T* key_w,
    T* value_w,
    T* gating_w,
    T* gating_b,
    T* output_w,
    T* output_b,
    T* output,
    int64_t B_t,
    int64_t S_t,
    int64_t N_t,
    int64_t H_t,
    int64_t HS_t) {
  GlobalPass _gp(FWD);
  //   typedef float T;
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
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag>" << std::endl;
    return 1;
  }

  int batch_size = std::stoi(argv[1]);
  int seq_len = std::stoi(argv[2]);
  int num_heads = std::stoi(argv[3]);
  int head_size = std::stoi(argv[4]);
  int embedding_dim = num_heads * head_size;
  bool bias_flag = std::stoi(argv[5]);
  bool nb_bias_flag = std::stoi(argv[6]);

  typedef float T;

  // Allocate memory for input tensors
  float* q_data = new float[batch_size * seq_len * embedding_dim];
  // auto q_data_a = GetVLAPtr<float>(q_data, {seq_len, embedding_dim});

  float* m_data = new float[batch_size * seq_len * embedding_dim];
  // auto m_data_a = GetVLAPtr<float>(m_data, {seq_len, embedding_dim});

  float* bias = new float[batch_size * seq_len];
  // auto bias_a = GetVLAPtr<float>(bias, {seq_len});

  float* nonbatched_bias = new float[1 * num_heads * seq_len * seq_len];
  // auto nonbatched_bias_a = GetVLAPtr<float>(nonbatched_bias, {num_heads, seq_len, seq_len});

  float* query_w = new float[embedding_dim * num_heads * head_size];
  // auto query_w_a = GetVLAPtr<float>(query_w, {num_heads, head_size});

  float* key_w = new float[embedding_dim * num_heads * head_size];
  // auto key_w_a = GetVLAPtr<float>(key_w, {num_heads, head_size});

  float* value_w = new float[embedding_dim * num_heads * head_size];
  // auto value_w_a = GetVLAPtr<float>(value_w, {num_heads, head_size});

  float* gating_w = new float[embedding_dim * num_heads * head_size];
  // auto gating_w_a = GetVLAPtr<float>(gating_w, {num_heads, head_size});

  float* gating_b = new float[num_heads * head_size];
  // auto gating_b_a = GetVLAPtr<float>(gating_b, {head_size});

  float* output_w = new float[num_heads * head_size * embedding_dim];
  // auto output_w_a = GetVLAPtr<float>(output_w, {head_size, embedding_dim});

  float* output_b = new float[embedding_dim];
  // auto output_b_a = GetVLAPtr<float>(output_b, {1});

  float* output = new float[batch_size * seq_len * embedding_dim];
  // auto output_a = GetVLAPtr<float>(output, {seq_len, embedding_dim});

  // Initialize input tensors with random values
  for (int i = 0; i < batch_size*seq_len*embedding_dim; ++i) {
    q_data[i] = static_cast<float>(rand()) / RAND_MAX;
    m_data[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < batch_size*seq_len; ++i) {
    bias[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < num_heads * seq_len * seq_len; ++i) {
    nonbatched_bias[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < embedding_dim * num_heads * head_size; ++i) {
        query_w[i] = static_cast<float>(rand()) / RAND_MAX;
        key_w[i] = static_cast<float>(rand()) / RAND_MAX;
        value_w[i] = static_cast<float>(rand()) / RAND_MAX;
        gating_w[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < num_heads * head_size; ++i) {
    gating_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < num_heads * head_size * embedding_dim; ++i) {
    output_w[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < embedding_dim; ++i) {
    output_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  auto t0 = getTime();
  // Run the forward function
  fused_gating_attention_fwd_fp32<float>(
      q_data, m_data, bias, nonbatched_bias, 
      query_w, key_w, value_w, gating_w, gating_b, 
      output_w, output_b, output, 
      batch_size, seq_len, num_heads, head_size, embedding_dim);

  auto t1 = getTime();
  printf("Time taken: %f ms\n", (t1 - t0) * 1e3);

  delete[] q_data;
  delete[] m_data;
  delete[] bias;
  delete[] nonbatched_bias;
  delete[] query_w;
  delete[] key_w;
  delete[] value_w;
  delete[] gating_w;
  delete[] gating_b;
  delete[] output_w;
  delete[] output_b;
  delete[] output;

  return 0;
}



