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
#include <cmath>
#include <iostream>
#include <tuple>
#include <cpuid.h>
#include <chrono>

#include "ext_tpp.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;

#define QKV_BLOCKSIZE 64
#define A_BLOCKSIZE 64
#define Ak_BLOCKSIZE 512
#define C_BLOCKSIZE 64

// REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
// REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
// REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");
// REGISTER_SCOPE(alpha_ac_gemm, "alpha_ac_gemm");
// REGISTER_SCOPE(alpha_o_gemm, "alpha_o_gemm");

unsigned long long rdtsc_ordered() {
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx); // Using specific function for cpuid

  return __rdtsc();
  // unsigned int lo, hi;
  // __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
  // return ((unsigned long long)hi << 32) | lo;
}

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::plus<T>());
    return result;
}

std::vector<long int> fused_gating_attention_fwd_bf16(
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
    bool nbbias_flag,
    bool gate_flag) {
  GlobalPass _gp(FWD);
  
  typedef bfloat16 T;
  #include "fused_gating_attention_fwd_tmpl_bf16.h"

  return times;
}

std::vector<long int> fused_gating_attention_fwd_fp32(
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
    bool nbbias_flag,
    bool gate_flag) {
  // GlobalPass _gp(FWD);
  
  typedef float T;
  #include "fused_gating_attention_fwd_tmpl.h"
  return times;
}

template<typename T>
std::tuple<T**, T**, float**, float**, T**, T**, T**, T**, float**, T**, float**, T**> allocate_and_initialize(long batch_size, long seq_len, long num_heads, long head_size, long embedding_dim, long num_layer) {

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

    // check if any allocation failed
    if (!q_data[l] || !m_data[l] || !bias[l] || !nonbatched_bias[l] || !query_w[l] || !key_w[l] || !value_w[l] || !gating_w[l] || !gating_b[l] || !output_w[l] || !output_b[l] || !output[l]) {
      std::cerr << "Memory allocation failed for layer " << l << std::endl;
      exit(1);
    }

    for (int i = 0; i < batch_size*seq_len*embedding_dim; ++i) {
      q_data[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
      m_data[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
    }
    for (int i = 0; i < batch_size*seq_len; ++i) {
      bias[l][i] = static_cast<float>(rand() % 10)*0.1; // / RAND_MAX;
    }
    for (int i = 0; i < num_heads * seq_len * seq_len; ++i) {
      nonbatched_bias[l][i] = static_cast<float>(rand() % 10)*0.1; /// RAND_MAX;
    }
    for (int i = 0; i < embedding_dim * num_heads * head_size; ++i) {
          query_w[l][i] = static_cast<T>(rand() % 10)*0.1; // / RAND_MAX;
          key_w[l][i] = static_cast<T>(rand() % 10)*0.1; // / RAND_MAX;
          value_w[l][i] = static_cast<T>(rand() % 10)*0.1; // / RAND_MAX;
          gating_w[l][i] = static_cast<T>(rand() % 10)*0.1; // / RAND_MAX;
    }
    for (int i = 0; i < num_heads * head_size; ++i) {
      gating_b[l][i] = static_cast<float>(rand() % 10)*0.1; // / RAND_MAX;
    }

    for (int i = 0; i < num_heads * head_size * embedding_dim; ++i) {
      output_w[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
    }
    for (int i = 0; i < embedding_dim; ++i) {
      output_b[l][i] = static_cast<float>(rand() % 10)*0.1; // / RAND_MAX;
    }
  }

  return std::make_tuple(q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output);
}


// Main function that takes Batch size, Sequence length, Number of heads, Head size as input

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <gate_flag> <BF16> <num_layer> <num_iter>" << std::endl;
    return 1;
  }

  long batch_size = std::stoi(argv[1]);
  long seq_len = std::stoi(argv[2]);
  long num_heads = std::stoi(argv[3]);
  long head_size = std::stoi(argv[4]);
  long embedding_dim = num_heads * head_size;
  bool bias_flag = std::stoi(argv[5]);
  bool nbbias_flag = std::stoi(argv[6]);
  bool gate_flag = std::stoi(argv[7]);
  bool bf16_flag = std::stoi(argv[8]);
  long num_layer = std::stoi(argv[9]);
  long num_iter = std::stoi(argv[10]);
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
        batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
    }
    auto t0 = getTime();
    auto times = std::vector<long int>(5, 0);
    for (int i = 0; i < num_iter; i++) {
      // Run the forward function
      for (int l = 0; l < num_layer; l++) {
        auto layer_time = fused_gating_attention_fwd_bf16(
              q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
        times = times + layer_time;
      }
    }
    auto t1 = getTime();
    // print the times vector elements with names
    printf("Time taken for q_gemm: %f ms\n", times[0] / ((double)1e3 * num_iter * num_layer));
    printf("Time taken for k_gemm: %f ms\n", times[1] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for v_gemm: %f ms\n", times[2] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for ac_gemm (SDPA): %f ms\n", times[3] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for o_gemm: %f ms\n", times[4] / ((double)1e3 *num_iter * num_layer));

    printf("Total Time taken for one layer: %f ms\n", ((t1 - t0)/(num_iter * num_layer)) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
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
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
    }
    auto t0 = getTime();
    auto times = std::vector<long int>(5, 0);
    for (int i = 0; i < num_iter; i++) {
      for (int l = 0; l < num_layer; l++) {
        auto layer_time = fused_gating_attention_fwd_fp32(
              q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
        times = times + layer_time;
      }
    }

    auto t1 = getTime();
    // print the times vector elements with names
    printf("Time taken for q_gemm: %f ms\n", times[0] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for k_gemm: %f ms\n", times[1] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for v_gemm: %f ms\n", times[2] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for ac_gemm (SDPA): %f ms\n", times[3] / ((double)1e3 *num_iter * num_layer));
    printf("Time taken for o_gemm: %f ms\n", times[4] / ((double)1e3 *num_iter * num_layer));

    printf("Total Time taken for one layer: %f ms\n", ((t1 - t0)/(num_iter * num_layer)) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
  }

  return 0;
}



