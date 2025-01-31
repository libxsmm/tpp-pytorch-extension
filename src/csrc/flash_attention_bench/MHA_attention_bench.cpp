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

// REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
// REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
// REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

// REGISTER_SCOPE(alpha_ac_gemm, "alpha_ac_gemm");
// REGISTER_SCOPE(alpha_o_gemm, "alpha_o_gemm");
 
// void fused_gating_attention_fwd_bf16(
//     bfloat16* q_data,
//     bfloat16* m_data,
//     float* bias,
//     float* nonbatched_bias,
//     bfloat16* query_w,
//     bfloat16* key_w,
//     bfloat16* value_w,
//     bfloat16* gating_w,
//     float* gating_b,
//     bfloat16* output_w,
//     float* output_b,
//     bfloat16* output,
//     int64_t B_t,
//     int64_t S_t,
//     int64_t N_t,
//     int64_t H_t,
//     int64_t HS_t,
//     bool flag) {
  // GlobalPass _gp(FWD);
  
//   typedef bfloat16 T;
//   #include "fused_gating_attention_fwd_tmpl_bf16.h"
// }

void fused_gating_attention_fwd_fp32(
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
    float* output,
    int64_t B_t,
    int64_t S_t,
    int64_t N_t,
    int64_t H_t,
    int64_t HS_t,
    bool flag) {
  // GlobalPass _gp(FWD);
  
  typedef float T;
  #include "fused_gating_attention_fwd_tmpl.h"
}

template<typename T>
std::tuple<T*, T*, float*, float*, T*, T*, T*, T*, float*, T*, float*, T*> allocate_and_initialize(int batch_size, int seq_len, int num_heads, int head_size, int embedding_dim) {

  // Allocate memory for input tensors
  T* q_data = new T[batch_size * seq_len * embedding_dim];
  T* m_data = new T[batch_size * seq_len * embedding_dim];
  float* bias = new float[batch_size * seq_len];
  float* nonbatched_bias = new float[1 * num_heads * seq_len * seq_len];
  T* query_w = new T[embedding_dim * num_heads * head_size];
  T* key_w = new T[embedding_dim * num_heads * head_size];
  T* value_w = new T[embedding_dim * num_heads * head_size];
  T* gating_w = new T[embedding_dim * num_heads * head_size];
  float* gating_b = new float[num_heads * head_size];
  T* output_w = new T[num_heads * head_size * embedding_dim];
  float* output_b = new float[embedding_dim];
  T* output = new T[batch_size * seq_len * embedding_dim];

  // Initialize input tensors with random values
  for (int i = 0; i < batch_size*seq_len*embedding_dim; ++i) {
    q_data[i] = static_cast<T>(rand()) / RAND_MAX;
    m_data[i] = static_cast<T>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < batch_size*seq_len; ++i) {
    bias[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < num_heads * seq_len * seq_len; ++i) {
    nonbatched_bias[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < embedding_dim * num_heads * head_size; ++i) {
        query_w[i] = static_cast<T>(rand()) / RAND_MAX;
        key_w[i] = static_cast<T>(rand()) / RAND_MAX;
        value_w[i] = static_cast<T>(rand()) / RAND_MAX;
        gating_w[i] = static_cast<T>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < num_heads * head_size; ++i) {
    gating_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < num_heads * head_size * embedding_dim; ++i) {
    output_w[i] = static_cast<T>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < embedding_dim; ++i) {
    output_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  return std::make_tuple(q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output);
}


// Main function that takes Batch size, Sequence length, Number of heads, Head size as input

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <BF16>" << std::endl;
    return 1;
  }

  int batch_size = std::stoi(argv[1]);
  int seq_len = std::stoi(argv[2]);
  int num_heads = std::stoi(argv[3]);
  int head_size = std::stoi(argv[4]);
  int embedding_dim = num_heads * head_size;
  bool bias_flag = std::stoi(argv[5]);
  bool nb_bias_flag = std::stoi(argv[6]);
  bool bf16_flag = std::stoi(argv[7]);
  if (bf16_flag) {
    printf("Running with BF16\n");
    typedef bfloat16 T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim);
  
    auto t0 = getTime();
    // Run the forward function
    // fused_gating_attention_fwd_bf16(
    //     q_data, m_data, bias, nonbatched_bias, 
    //     query_w, key_w, value_w, gating_w, gating_b, 
    //     output_w, output_b, output, 
    //     batch_size, seq_len, num_heads, head_size, embedding_dim, nb_bias_flag);

    auto t1 = getTime();
    printf("Time taken: %f ms\n", (t1 - t0) * 1e3);
    delete[] q_data; delete[] m_data; delete[] bias; delete[] nonbatched_bias; delete[] query_w; delete[] key_w; delete[] value_w; delete[] gating_w; delete[] gating_b; delete[] output_w; delete[] output_b; delete[] output;
  } else {
    printf("Running with FP32\n");
    typedef float T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim);
  
    auto t0 = getTime();
    // Run the forward function
    fused_gating_attention_fwd_fp32(
        q_data, m_data, bias, nonbatched_bias, 
        query_w, key_w, value_w, gating_w, gating_b, 
        output_w, output_b, output, 
        batch_size, seq_len, num_heads, head_size, embedding_dim, nb_bias_flag);

    auto t1 = getTime();
    printf("Time taken: %f ms\n", (t1 - t0) * 1e3);
    delete[] q_data; delete[] m_data; delete[] bias; delete[] nonbatched_bias; delete[] query_w; delete[] key_w; delete[] value_w; delete[] gating_w; delete[] gating_b; delete[] output_w; delete[] output_b; delete[] output;
  }

  return 0;
}



