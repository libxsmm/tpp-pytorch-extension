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

#ifndef __riscv
#include <cpuid.h>
#endif

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
#ifdef __riscv
  return 0;
#else
  unsigned int eax, ebx, ecx, edx;
  //__cpuid(1, eax, ebx, ecx, edx); // Using specific function for cpuid

  return __rdtsc();
  unsigned int lo, hi;
  __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
  return ((unsigned long long)hi << 32) | lo;
#endif
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
    bool gate_flag,
    bool b_vnni) {
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

template<typename T>
void flops_and_bandwidth(
  const std::vector<long int>& times,
  long batch_size,
  long seq_len,
  long num_heads,
  long head_size,
  long embedding_dim,
  bool gate_flag,
  int num_iter,
  int num_layer,
  bool bias_flag,
  bool nbbias_flag
  ) {
    auto times_ms = std::vector<double>(5, 0);
    for(int i = 0; i < 5; i++) {
      times_ms[i] = (double)times[i] / ((double)1e3 * num_iter * num_layer);
    }
    auto q_flops = ((double)2.0 * batch_size * seq_len * num_heads * head_size * embedding_dim) / (times_ms[0] * (double)1e9);
    auto k_flops = ((double)2.0 * batch_size * seq_len * num_heads * head_size * embedding_dim) / (times_ms[1] * (double)1e9);
    auto v_flops = ((double)2.0 * batch_size * seq_len * num_heads * head_size * embedding_dim) / (times_ms[2] * (double)1e9);
    auto ac_flops = ((double)4.0 * batch_size * num_heads * head_size * seq_len * seq_len) / (times_ms[3] * (double)1e9);
    auto o_flops = ((double)2.0 * batch_size * seq_len * num_heads * head_size * embedding_dim);
    if (gate_flag)
      o_flops += ((double)2.0 * batch_size * seq_len * num_heads * head_size * embedding_dim);
    o_flops = o_flops / (times_ms[4] * (double)1e9);
    
    auto q_gb = ((((double)2.0 * batch_size * seq_len * num_heads * head_size) + ((double)num_heads * head_size * embedding_dim)) * sizeof(T)) / (times_ms[0] * (double)1e6);
    auto k_gb = ((((double)2.0 *batch_size * seq_len * num_heads * head_size) + ((double)num_heads * head_size * embedding_dim)) * sizeof(T)) / (times_ms[1] * (double)1e6);
    auto v_gb = ((((double)2.0 *batch_size * seq_len * num_heads * head_size) + ((double)num_heads * head_size * embedding_dim)) * sizeof(T)) / (times_ms[2] * (double)1e6);

    auto ac_gb = ((((double)4.0 *batch_size * seq_len * num_heads * head_size)) * sizeof(T));
    if (bias_flag)
      ac_gb += ((((double)batch_size * seq_len)) * sizeof(float));
    if (nbbias_flag)
      ac_gb += ((((double)num_heads * seq_len * seq_len)) * sizeof(float));
    ac_gb = ac_gb / (times_ms[3] * (double)1e6);

    auto o_gb = ((((double)2.0 *batch_size * seq_len * num_heads * head_size) + ((double)num_heads * head_size * embedding_dim)) * sizeof(T));
    if (gate_flag)
      o_gb += ((((double)batch_size * seq_len * num_heads * head_size) + ((double)num_heads * head_size * embedding_dim)) * sizeof(T));
    o_gb = o_gb / (times_ms[4] * (double)1e6);
    // print the times_ms vector elements with names 
    printf("Time taken for q_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", times_ms[0], q_flops, q_gb);
    printf("Time taken for k_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", times_ms[1], k_flops, k_gb);
    printf("Time taken for v_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", times_ms[2], v_flops, v_gb);
    printf("Time   ac_gemm (SDPA): %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", times_ms[3], ac_flops, ac_gb);
    printf("Time taken for o_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", times_ms[4], o_flops, o_gb);
  }

// Main function that takes Batch size, Sequence length, Number of heads, Head size as input

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <gate_flag> <BF16> <b_vnni> <num_layer> <num_iter> <self_attention_flag>" << std::endl;
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
  bool b_vnni = std::stoi(argv[9]);
  long num_layer = std::stoi(argv[10]);
  long num_iter = std::stoi(argv[11]);
  bool self_attention_flag = std::stoi(argv[12]);
  if (bf16_flag) {
    printf("Running with BF16\n");
    typedef bfloat16 T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim, num_layer);
    
    for (int l = 0; l < num_layer; l++) {
      if (self_attention_flag){
        if (l==0) {
          fused_gating_attention_fwd_bf16(
            q_data[l], q_data[l], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
        } else {
          fused_gating_attention_fwd_bf16(
            output[l-1], output[l-1], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
        }
      } else {
        fused_gating_attention_fwd_bf16(
          q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
          query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
          output_w[l], output_b[l], output[l], 
          batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
      }
    }
    auto t0 = getTime();
    auto times = std::vector<long int>(5, 0);
    for (int i = 0; i < num_iter; i++) {
      // Run the forward function
      for (int l = 0; l < num_layer; l++) {
        if (self_attention_flag){
          if (l==0) {
            auto layer_time = fused_gating_attention_fwd_bf16(
              q_data[l], q_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
            times = times + layer_time;
          } else {
            auto layer_time = fused_gating_attention_fwd_bf16(
              output[l-1], output[l-1], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
            times = times + layer_time;
          }
        } else {
          auto layer_time = fused_gating_attention_fwd_bf16(
                q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
                query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
                output_w[l], output_b[l], output[l], 
                batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag, b_vnni);
          times = times + layer_time;
        }
      }
    }
    auto t1 = getTime();
    flops_and_bandwidth<T>(times, batch_size, seq_len, num_heads, head_size, embedding_dim, gate_flag, num_iter, num_layer, bias_flag, nbbias_flag);

    printf("Total Time taken for one layer: %0.2f ms\n", ((t1 - t0)/(num_iter * num_layer)) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
  } else {
    printf("Running with FP32\n");
    typedef float T;
    // Allocate and initialize input tensors
    auto [q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, gating_w, gating_b, output_w, output_b, output] = allocate_and_initialize<T>(batch_size, seq_len, num_heads, head_size, embedding_dim, num_layer);
  
    for (int l = 0; l < num_layer; l++) {
      if (self_attention_flag){
        if (l==0) {
          fused_gating_attention_fwd_fp32(
            q_data[l], q_data[l], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
        } else {
          fused_gating_attention_fwd_fp32(
            output[l-1], output[l-1], bias[l], nonbatched_bias[l], 
            query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
            output_w[l], output_b[l], output[l], 
            batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
        }
      } else {
        fused_gating_attention_fwd_fp32(
              q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
      }
    }
    auto t0 = getTime();
    auto times = std::vector<long int>(5, 0);
    for (int i = 0; i < num_iter; i++) {
      for (int l = 0; l < num_layer; l++) {
        if (self_attention_flag){
          if (l==0) {
            auto layer_time = fused_gating_attention_fwd_fp32(
              q_data[l], q_data[l], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
            times = times + layer_time;
          } else {
            auto layer_time = fused_gating_attention_fwd_fp32(
              output[l-1], output[l-1], bias[l], nonbatched_bias[l], 
              query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
              output_w[l], output_b[l], output[l], 
              batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
            times = times + layer_time;
          }
        } else {
          auto layer_time = fused_gating_attention_fwd_fp32(
                q_data[l], m_data[l], bias[l], nonbatched_bias[l], 
                query_w[l], key_w[l], value_w[l], gating_w[l], gating_b[l], 
                output_w[l], output_b[l], output[l], 
                batch_size, seq_len, num_heads, head_size, embedding_dim, bias_flag, nbbias_flag, gate_flag);
          times = times + layer_time;
        }
      }
    }

    auto t1 = getTime();
    flops_and_bandwidth<T>(times, batch_size, seq_len, num_heads, head_size, embedding_dim, gate_flag, num_iter, num_layer, bias_flag, nbbias_flag);

    printf("Total Time taken for one layer: %0.2f ms\n", ((t1 - t0)/(num_iter * num_layer)) * 1e3);
    for(int l = 0; l < num_layer; l++) {
      delete[] q_data[l]; delete[] m_data[l]; delete[] bias[l]; delete[] nonbatched_bias[l]; delete[] query_w[l]; delete[] key_w[l]; delete[] value_w[l]; delete[] gating_w[l]; delete[] gating_b[l]; delete[] output_w[l]; delete[] output_b[l]; delete[] output[l];
    }
  }

  return 0;
}



