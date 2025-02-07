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
// #include "shm_coll.h"

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

class Barrier {

  public:
    static constexpr int size = 2; 

  volatile int bar1;
  volatile int bar2;
  int count[size];
  int pad[12];
  Barrier() {
    bar1 = 0;
    bar2 = 0;
    for (int i = 0; i < size; i++) {
      count[i] = 0;
    }
  }
  void wait(int tid) {
      if (size == 1) {
        return;
      }
      if (count[tid] % 2) {
        __sync_fetch_and_add(&bar1, 1);
        while ((bar1 % size) != 0)
          ;
      } else {
        __sync_fetch_and_add(&bar2, 1);
        while ((bar2 % size) != 0)
          ;
      }
      count[tid]++;
    }
};

void fused_gating_attention_fwd_bf16(
    bfloat16* q_data,
    bfloat16* m_data,
    float* bias,
    float* nonbatched_bias,
    bfloat16* query_w,
    bfloat16* key_w,
    bfloat16* value_w,
    bfloat16* gating_w,
    float* gating_b,
    bfloat16* output_w,
    float* output_b,
    bfloat16* output,
    int64_t B_t,
    int64_t S_t,
    int64_t N_t,
    int64_t H_t,
    int64_t HS_t,
    bool bias_flag,
    bool flag) {
  GlobalPass _gp(FWD);
  
  typedef bfloat16 T;
  #include "fused_gating_attention_fwd_tmpl_bf16.h"
}

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
    bool bias_flag,
    bool flag) {
  // GlobalPass _gp(FWD);
  
  typedef float T;
  #include "fused_gating_attention_fwd_tmpl.h"
}

template<typename T>
std::tuple<T**, T**, float**, float**, T**, T**, T**, T**, float**, T**, float**, T**> allocate_and_initialize(int batch_size, int seq_len, int num_heads, int head_size, int embedding_dim, int num_layer) {

  T** q_data = new T*[num_layer];
  T** m_data =  new T*[num_layer];
  float** bias = new float*[num_layer];
  float** nonbatched_bias = new float*[num_layer];
  T** query_w = new T*[num_layer];
  T** key_w = new T*[num_layer];
  T** value_w = new T*[num_layer];
  T** gating_w = new T*[num_layer];
  float** gating_b = new float*[num_layer];
  T** output_w = new T*[num_layer];
  float** output_b = new float*[num_layer];
  T** output = new T*[num_layer];


  // // Allocate memory for input tensors
  // T* q_data = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
  // T* m_data = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
  // float* bias = new (std::align_val_t(64)) float[batch_size * seq_len];
  // float* nonbatched_bias = new (std::align_val_t(64)) float[1 * num_heads * seq_len * seq_len];
  // T* query_w = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
  // T* key_w = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
  // T* value_w = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
  // T* gating_w = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
  // float* gating_b = new (std::align_val_t(64)) float[num_heads * head_size];
  // T* output_w = new (std::align_val_t(64)) T[num_heads * head_size * embedding_dim];
  // float* output_b = new (std::align_val_t(64)) float[embedding_dim];
  // T* output = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];

  // Initialize input tensors with random values
  for (int l = 0; l < num_layer; l++) {
    q_data[l] = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
    m_data[l] = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
    bias[l] = new (std::align_val_t(64)) float[batch_size * seq_len];
    nonbatched_bias[l] = new (std::align_val_t(64)) float[num_heads * seq_len * seq_len];
    query_w[l] = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
    key_w[l] = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
    value_w[l] = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
    gating_w[l] = new (std::align_val_t(64)) T[embedding_dim * num_heads * head_size];
    gating_b[l] = new (std::align_val_t(64)) float[num_heads * head_size];
    output_w[l] = new (std::align_val_t(64)) T[num_heads * head_size * embedding_dim];
    output_b[l] = new (std::align_val_t(64)) float[embedding_dim];
    output[l] = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];

    for (int i = 0; i < batch_size*seq_len*embedding_dim; ++i) {
      q_data[l][i] = static_cast<T>(rand()) / RAND_MAX;
      m_data[l][i] = static_cast<T>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < batch_size*seq_len; ++i) {
      bias[l][i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < num_heads * seq_len * seq_len; ++i) {
      nonbatched_bias[l][i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < embedding_dim * num_heads * head_size; ++i) {
          query_w[l][i] = static_cast<T>(rand()) / RAND_MAX;
          key_w[l][i] = static_cast<T>(rand()) / RAND_MAX;
          value_w[l][i] = static_cast<T>(rand()) / RAND_MAX;
          gating_w[l][i] = static_cast<T>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < num_heads * head_size; ++i) {
      gating_b[l][i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < num_heads * head_size * embedding_dim; ++i) {
      output_w[l][i] = static_cast<T>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < embedding_dim; ++i) {
      output_b[l][i] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  return std::make_tuple(q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output);
}


// Main function that takes Batch size, Sequence length, Number of heads, Head size as input

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <BF16> <num_layer> <num_iter>" << std::endl;
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
  int num_layer = std::stoi(argv[8]);
  int num_iter = std::stoi(argv[9]);
  if (bf16_flag) {
    printf("Running with BF16\n");
    typedef bfloat16 T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim, num_layer);
  
    for (int l = 0; l < num_layer; l++) {
      fused_gating_attention_fwd_bf16(
            q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nb_bias_flag);
    }
    auto t0 = getTime();
    for (int i = 0; i < num_iter; i++) {
      // Run the forward function
      for (int l = 0; l < num_layer; l++) {
        fused_gating_attention_fwd_bf16(
              q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nb_bias_flag);
      }
    }

    auto t1 = getTime();
    printf("Time taken: %f ms\n", ((t1 - t0)/num_iter) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
    // delete[] q_data; delete[] m_data; delete[] bias; delete[] nonbatched_bias; delete[] query_w; delete[] key_w; delete[] value_w; delete[] gating_w; delete[] gating_b; delete[] output_w; delete[] output_b; delete[] output;
  } else {
    printf("Running with FP32\n");
    typedef float T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim, num_layer);
  
    for (int l = 0; l < num_layer; l++) {
      fused_gating_attention_fwd_fp32(
            q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nb_bias_flag);
    }
    auto t0 = getTime();
    for (int i = 0; i < num_iter; i++) {
      for (int l = 0; l < num_layer; l++) {
        fused_gating_attention_fwd_fp32(
              q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nb_bias_flag);
      }
    }

    auto t1 = getTime();
    printf("Time taken: %f ms\n", ((t1 - t0)/num_iter) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
    // delete[] q_data; delete[] m_data; delete[] bias; delete[] nonbatched_bias; delete[] query_w; delete[] key_w; delete[] value_w; delete[] gating_w; delete[] gating_b; delete[] output_w; delete[] output_b; delete[] output;
  }

  return 0;
}



