#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <list>
#include <tuple>
#include <chrono>
#include <memory>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>

// #include <ATen/record_function.h>
// #include <torch/csrc/autograd/VariableTypeUtils.h>
// #include <torch/extension.h>

#include "ext_tpp.h"
#include "timing.h"
#include "xsmm_functors.h"
// #include "fused_gemm.h"
// #include "qtypes.h"

using namespace tpp;

#ifdef USE_FLOAT
#define MLP_BLOCKSIZE 32L
#else
#define MLP_BLOCKSIZE 64L
#endif

#ifdef USE_FLOAT
  typedef float T;
#else
  typedef bfloat16 T;
#endif

// #define USE_TBB

#ifdef USE_TBB
  #include <tbb/parallel_for.h>
  #include <tbb/blocked_range.h>
  #include <tbb/global_control.h>
  #include <tbb/concurrent_vector.h>
#endif

// #define SCOPE_TPP
#ifndef SCOPE_TPP
  template<typename T>
  struct FFNTPPs {
    BrgemmTPP<T, float> i_gemm_tpp;
    SiLUFwdTPP<float> silu_tpp;
    Mul2TPP<float, T, T> mul_tpp;
    Mul2TPP<float, T, float> gateup_mul_tpp;
    BrgemmTPP<T, float> o_gemm_tpp;
    ScaleTPP<float, float> scale_tpp;
    ConvertTPP<float, T> downconvert_inter_tpp;
    ConvertTPP<float, T> downconvert_embed_tpp;
  };

#else
  template<typename T>
  struct FFNTPPs {
    SCOPEIT_DECL(BrgemmTPP<T, float>) i_gemm_tpp;
    SCOPEIT_DECL(SiLUFwdTPP<float>) silu_tpp;
    SCOPEIT_DECL(Mul2TPP<float, T, T>) mul_tpp;
    SCOPEIT_DECL(Mul2TPP<float, T, float>) gateup_mul_tpp;
    SCOPEIT_DECL(BrgemmTPP<T, float>) o_gemm_tpp;
    SCOPEIT_DECL(ScaleTPP<float, float>) scale_tpp;
    SCOPEIT_DECL(ConvertTPP<float, T>) downconvert_inter_tpp;
    SCOPEIT_DECL(ConvertTPP<float, T>) downconvert_embed_tpp;
  };
#endif

template<typename T>
void activation_allocate_and_initialize(std::vector<std::unique_ptr<T[]>>& t_Out,
                               std::vector<std::unique_ptr<T[]>>& t_In,
  long token_len, long embedding_dim, long intermediate_dim, long num_layer) {

  auto downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(embedding_dim)), EW_ZERO);

  for (int l = 0; l < num_layer; l++) {
    t_In[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);   // allocate an dynamic array at 64 bit boundary and make unique pointer 
    t_Out[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);

#ifdef USE_TBB
    tbb::parallel_for(0L, token_len, [&](long i) {
#else
    #pragma omp parallel for
    for (int i = 0; i < token_len; ++i) {
#endif
      std::vector<float> t_In_tmp(embedding_dim), t_Out_tmp(embedding_dim);
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 0.9f);

      std::generate(t_In_tmp.begin(), t_In_tmp.end(), 
            [&]() { return dist(gen); });
      std::generate(t_Out_tmp.begin(), t_Out_tmp.end(), 
            [&]() { return dist(gen); });

      downconvert_embed_tpp(t_In_tmp.data(), &t_In[l][i*embedding_dim]);
      downconvert_embed_tpp(t_Out_tmp.data(), &t_Out[l][i*embedding_dim]);

#ifndef USE_TBB
    }
#else
    });
#endif
  }
}


template<typename T>
void moe_split_layer_input(std::unique_ptr<T[]>& t_In,
                           std::vector<std::unique_ptr<T[]>>& t_split_In,
                           std::vector<std::unique_ptr<T[]>>& t_split_Out,
                           std::vector<long>& expert_token_counts,
  std::vector<std::vector<long>>& token_to_expert,
  long token_len, long embedding_dim, long num_expert, long num_expert_per_token) {

  for (long t = 0; t < token_len; t++){
    // randomly select num_experts_per_token from num_experts for each token
    while (token_to_expert[t].size() < num_expert_per_token) {
      long rand_expert = rand() % num_expert;
      if (std::find(token_to_expert[t].begin(), token_to_expert[t].end(), rand_expert) == token_to_expert[t].end()) {
        token_to_expert[t].push_back(rand_expert);
        expert_token_counts[rand_expert]++;
      }
    }
  }
  for (long e = 0; e < num_expert; e++){
    // allocate input buffer for each expert
    t_split_In[e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[expert_token_counts[e] * embedding_dim]);
    t_split_Out[e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[expert_token_counts[e] * embedding_dim]);
    long expert_token_index = 0;
    for (long t = 0; t < token_len; t++){
      if (std::find(token_to_expert[t].begin(), token_to_expert[t].end(), e) != token_to_expert[t].end()) {
        // copy input token to expert input buffer
        std::memcpy(&t_split_In[e][expert_token_index * embedding_dim],
                    &t_In[t * embedding_dim],
                    embedding_dim * sizeof(T));
        expert_token_index++;
      }
    }
  }
}

template<typename T>
void moe_combine_layer_output(std::unique_ptr<T[]>& t_Out,
                           std::vector<std::unique_ptr<T[]>>& t_split_Out,
  std::vector<std::vector<long>>& token_to_expert,
  long token_len, long embedding_dim, long num_expert) {

  auto add_embed_tpp = AddTPP<T, T>(embedding_dim);
  std::vector<long> expert_offsets(num_expert, 0);
  for (long t = 0; t < token_len; t++){
    for (auto& expert : token_to_expert[t]) {
      // accumulate outputs from different experts
      add_embed_tpp(&t_Out[t * embedding_dim],
                    &t_split_Out[expert][expert_offsets[expert]],
                    &t_Out[t * embedding_dim]);
      // move expert output pointer
      expert_offsets[expert] += embedding_dim;
    }
  }
}


template<typename T>
void flat_weight_allocate_and_initialize(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
  long embedding_dim, long intermediate_dim, long num_layer, long num_expert) {

  auto downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(embedding_dim)), EW_ZERO);
  auto downconvert_inter_tpp = SCOPEIT((ConvertTPP<float, T>(intermediate_dim)), EW_ZERO);

  for (int l = 0; l < num_layer; l++) {
    for (int e=0; e < num_expert; e++){
      t_Wg[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      t_Wu[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      t_Wd[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

#ifdef USE_TBB
      tbb::parallel_for(0L, intermediate_dim, [&](long i) {
#else
      #pragma omp parallel for
      for(int i=0; i < intermediate_dim; i++){
#endif
        std::vector<float> t_Wd_tmp(embedding_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 0.9f);

        std::generate(t_Wd_tmp.begin(), t_Wd_tmp.end(), 
              [&]() { return dist(gen); });

        downconvert_embed_tpp(t_Wd_tmp.data(), &t_Wd[l*num_expert + e][i*embedding_dim]);
#ifndef USE_TBB
      }
#else
      });
#endif

#ifdef USE_TBB
      tbb::parallel_for(0L, embedding_dim, [&](long i) {
#else
      #pragma omp parallel for
      for (int i = 0; i < embedding_dim; ++i) {
#endif
        std::vector<float> t_Wg_tmp(intermediate_dim), t_Wu_tmp(intermediate_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 0.9f);

        std::generate(t_Wg_tmp.begin(), t_Wg_tmp.end(), 
              [&]() { return dist(gen); });
        std::generate(t_Wu_tmp.begin(), t_Wu_tmp.end(), 
              [&]() { return dist(gen); });

        downconvert_inter_tpp(t_Wg_tmp.data(), &t_Wg[l*num_expert + e][i*intermediate_dim]);
        downconvert_inter_tpp(t_Wu_tmp.data(), &t_Wu[l*num_expert + e][i*intermediate_dim]);
#ifndef USE_TBB
      }
#else
      });
#endif
    }
  }
}

template<typename T>
void flat_to_blocked_weights(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
  long embedding_dim, long intermediate_dim, long num_layer, long num_expert) {

  for(int l = 0; l < num_layer; l++) {
    for(int e=0; e < num_expert; e++) {
      auto t_Wg_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wu_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wd_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

      for(int n1 = 0; n1 < embedding_dim/MLP_BLOCKSIZE; n1++) {
        for(int h1 = 0; h1 < MLP_BLOCKSIZE; h1++) {
          for(int n2 = 0; n2 < intermediate_dim/MLP_BLOCKSIZE; n2++) {
            for(int h2 = 0; h2 < MLP_BLOCKSIZE; h2++) {
              t_Wg_tmp[n2 * (embedding_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]     //[n2, n1, h1, h2]
                = t_Wg[l*num_expert + e][n1 * (MLP_BLOCKSIZE * intermediate_dim) + h1 * (intermediate_dim) + n2 * (MLP_BLOCKSIZE) + h2];             //[n1, h1, n2, h2]

              t_Wu_tmp[n2 * (embedding_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]
                = t_Wu[l*num_expert + e][n1 * (MLP_BLOCKSIZE * intermediate_dim) + h1 * (intermediate_dim) + n2 * (MLP_BLOCKSIZE) + h2];
            }
          }
        }
      }

      for(int n1 = 0; n1 < intermediate_dim/MLP_BLOCKSIZE; n1++) {
        for(int h1 = 0; h1 < MLP_BLOCKSIZE; h1++) {
          for(int n2 = 0; n2 < embedding_dim/MLP_BLOCKSIZE; n2++) {
            for(int h2 = 0; h2 < MLP_BLOCKSIZE; h2++) {
              t_Wd_tmp[n2 * (intermediate_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]
                = t_Wd[l*num_expert + e][n1 * MLP_BLOCKSIZE * embedding_dim + h1 * (embedding_dim) + n2 * (MLP_BLOCKSIZE) + h2];
            }
          }
        }
      }

      t_Wg[l*num_expert + e].swap(t_Wg_tmp);
      t_Wu[l*num_expert + e].swap(t_Wu_tmp);
      t_Wd[l*num_expert + e].swap(t_Wd_tmp);
    }
  }
}

template<typename T>
FFNTPPs<T> create_flat_ffn_tpps(long M_dim, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni){
  FFNTPPs<T> tpps;
#ifndef SCOPE_TPP
  tpps.i_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));

  tpps.silu_tpp = SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE);
  tpps.downconvert_inter_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.mul_tpp = Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim);
  tpps.o_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));

  tpps.scale_tpp = ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE);
  tpps.downconvert_embed_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);

#else
  tpps.i_gemm_tpp = SCOPEITGEMM((BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value))));
  
  tpps.silu_tpp = SCOPEIT((SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE)), ACT);
  tpps.downconvert_inter_tpp = SCOPEIT((ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim)), EW_COPY);
  tpps.mul_tpp = SCOPEIT((Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim)), EW_MUL);
  tpps.o_gemm_tpp = SCOPEITGEMM((BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value))));

  tpps.scale_tpp = SCOPEIT((ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE)), EW_SCL);
  tpps.downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim)), EW_COPY);

#endif
  return tpps;

}


template<typename T>
FFNTPPs<T> create_blocked_ffn_tpps(long M_dim, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni){
  FFNTPPs<T> tpps;

#ifndef SCOPE_TPP
  tpps.i_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, embedding_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  
  tpps.silu_tpp = SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE);
  tpps.downconvert_inter_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.mul_tpp = Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim);
  tpps.gateup_mul_tpp = Mul2TPP<float, T, float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.o_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, intermediate_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));

  tpps.scale_tpp = ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE);
  tpps.downconvert_embed_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);

#else
  tpps.i_gemm_tpp = SCOPEITGEMM((BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, embedding_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value))));
  
  tpps.silu_tpp = SCOPEIT((SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE)), ACT);
  tpps.downconvert_inter_tpp = SCOPEIT((ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim)), EW_COPY);
  tpps.mul_tpp = SCOPEIT((Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim)), EW_MUL);
  tpps.gateup_mul_tpp = SCOPEIT((Mul2TPP<float, T, float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim)), EW_MUL);
  tpps.o_gemm_tpp = SCOPEITGEMM((BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, intermediate_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value))));
  
  tpps.scale_tpp = SCOPEIT((ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE)), EW_SCL);
  tpps.downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim)), EW_COPY);
#endif
  return tpps;
}


template<typename T> 
std::vector<FFNTPPs<T>> create_ffn_tpp_set(bool blocked, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni){

  std::vector<FFNTPPs<T>> ffn_tpp_set(MLP_BLOCKSIZE);

  for (long M=0; M < MLP_BLOCKSIZE; M++){
    if(M==0){            // zero remainder case
      if (blocked)
        ffn_tpp_set[M] = create_blocked_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      else
        ffn_tpp_set[M] = create_flat_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    }
    else{
      if (blocked)
        ffn_tpp_set[M] = create_blocked_ffn_tpps<T>(M, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      else
        ffn_tpp_set[M] = create_flat_ffn_tpps<T>(M, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    }
  }

  return ffn_tpp_set;
}

template<typename T>
void flat_weight_vnni_transform(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
                            long embedding_dim, long intermediate_dim, long num_layer, long num_expert) {
#ifndef SCOPE_TPP
  auto i_vnni_tpp = XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        intermediate_dim,
        intermediate_dim,
        XformTPP::XFORM_N2V_TPP);

  auto o_vnni_tpp = XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        embedding_dim,
        embedding_dim,
        XformTPP::XFORM_N2V_TPP);
#else
  auto i_vnni_tpp = SCOPEIT((XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        intermediate_dim,
        intermediate_dim,
        XformTPP::XFORM_N2V_TPP)), XPOSE);
  
  auto o_vnni_tpp = SCOPEIT((XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        embedding_dim,
        embedding_dim,
        XformTPP::XFORM_N2V_TPP)), XPOSE);
#endif

  long embedding_blocks = embedding_dim/MLP_BLOCKSIZE ;
  long intermediate_blocks = intermediate_dim/MLP_BLOCKSIZE ;

  for (int l = 0; l < num_layer; l++) {
    for(int e = 0; e < num_expert; e++){
    
      auto t_Wg_a = GetVLAPtr<T>(t_Wg[l*num_expert + e].get(), {intermediate_dim});
      auto t_Wu_a = GetVLAPtr<T>(t_Wu[l*num_expert + e].get(), {intermediate_dim});
      auto t_Wd_a = GetVLAPtr<T>(t_Wd[l*num_expert + e].get(), {embedding_dim});

      auto t_Wg_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wu_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wd_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

      auto t_Wg_vnni_a = GetVLAPtr<T>(t_Wg_vnni.get(), {intermediate_dim});
      auto t_Wu_vnni_a = GetVLAPtr<T>(t_Wu_vnni.get(), {intermediate_dim});
      auto t_Wd_vnni_a = GetVLAPtr<T>(t_Wd_vnni.get(), {embedding_dim});

  #ifdef USE_TBB
      tbb::parallel_for(0L, embedding_blocks*intermediate_blocks, [&i_vnni_tpp, &t_Wg_a, &t_Wg_vnni_a, embedding_blocks, intermediate_blocks]
      (long idx) {
        long i = (idx / intermediate_blocks);
        long j = (idx % intermediate_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < embedding_blocks; i++){
        for(int j = 0; j < intermediate_blocks; j++){
  #endif
          i_vnni_tpp(&t_Wg_a[i*MLP_BLOCKSIZE][j*MLP_BLOCKSIZE], &t_Wg_vnni_a[i*MLP_BLOCKSIZE][2*j*MLP_BLOCKSIZE]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif

  #ifdef USE_TBB
      tbb::parallel_for(0L, embedding_blocks*intermediate_blocks, [&i_vnni_tpp, &t_Wu_a, &t_Wu_vnni_a, embedding_blocks, intermediate_blocks]
      (long idx) {
        long i = (idx / intermediate_blocks);
        long j = (idx % intermediate_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < embedding_blocks; i++){
        for(int j = 0; j < intermediate_blocks; j++){
  #endif
          i_vnni_tpp(&t_Wu_a[i*MLP_BLOCKSIZE][j*MLP_BLOCKSIZE], &t_Wu_vnni_a[i*MLP_BLOCKSIZE][2*j*MLP_BLOCKSIZE]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif

  #ifdef USE_TBB
      tbb::parallel_for(0L, intermediate_blocks*embedding_blocks, [&o_vnni_tpp, &t_Wd_a, &t_Wd_vnni_a, embedding_blocks, intermediate_blocks]
      (long idx) {
        long i = (idx / embedding_blocks);
        long j = (idx % embedding_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < intermediate_blocks; i++){
        for(int j = 0; j < embedding_blocks; j++){
  #endif
          o_vnni_tpp(&t_Wd_a[i*MLP_BLOCKSIZE][j*MLP_BLOCKSIZE], &t_Wd_vnni_a[i*MLP_BLOCKSIZE][2*j*MLP_BLOCKSIZE]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif
      // copy t_Wg_vnni contents t_Wg[l]
      t_Wg[l*num_expert + e].swap(t_Wg_vnni);
      t_Wu[l*num_expert + e].swap(t_Wu_vnni);
      t_Wd[l*num_expert + e].swap(t_Wd_vnni);
    }
  }
}


template<typename T>
void blocked_weight_vnni_transform(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
                            long embedding_dim, long intermediate_dim, long num_layer, long num_expert) {

#ifndef SCOPE_TPP
  auto i_vnni_tpp = XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        XformTPP::XFORM_N2V_TPP);

  auto o_vnni_tpp = XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        XformTPP::XFORM_N2V_TPP);
#else
  auto i_vnni_tpp = SCOPEIT((XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        XformTPP::XFORM_N2V_TPP)), XPOSE);
  
  auto o_vnni_tpp = SCOPEIT((XformExtTPP<T>(
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        MLP_BLOCKSIZE,
        XformTPP::XFORM_N2V_TPP)), XPOSE);
#endif

  long embedding_blocks = embedding_dim/MLP_BLOCKSIZE ;
  long intermediate_blocks = intermediate_dim/MLP_BLOCKSIZE ;

  for (int l = 0; l < num_layer; l++) {
    for(int e = 0; e < num_expert; e++){
    
      auto t_Wg_a = GetVLAPtr<T>(t_Wg[l*num_expert + e].get(), {embedding_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
      auto t_Wu_a = GetVLAPtr<T>(t_Wu[l*num_expert + e].get(), {embedding_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
      auto t_Wd_a = GetVLAPtr<T>(t_Wd[l*num_expert + e].get(), {intermediate_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

      auto t_Wg_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wu_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wd_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

      auto t_Wg_vnni_a = GetVLAPtr<T>(t_Wg_vnni.get(), {embedding_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
      auto t_Wu_vnni_a = GetVLAPtr<T>(t_Wu_vnni.get(), {embedding_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
      auto t_Wd_vnni_a = GetVLAPtr<T>(t_Wd_vnni.get(), {intermediate_blocks, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

  #ifdef USE_TBB
      tbb::parallel_for(0L, intermediate_blocks*embedding_blocks, [&i_vnni_tpp, &t_Wg_a, &t_Wg_vnni_a, embedding_blocks, intermediate_blocks](long idx) {
        long i = (idx / embedding_blocks);
        long j = (idx % embedding_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < intermediate_blocks; i++){
        for(int j = 0; j < embedding_blocks; j++){
  #endif
          i_vnni_tpp(&t_Wg_a[i][j][0][0], &t_Wg_vnni_a[i][j][0][0]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif

  #ifdef USE_TBB
      tbb::parallel_for(0L, intermediate_blocks*embedding_blocks, [&i_vnni_tpp, &t_Wu_a, &t_Wu_vnni_a, embedding_blocks, intermediate_blocks](long idx) {
        long i = (idx / embedding_blocks);
        long j = (idx % embedding_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < intermediate_blocks; i++){
        for(int j = 0; j < embedding_blocks; j++){
  #endif
          i_vnni_tpp(&t_Wu_a[i][j][0][0], &t_Wu_vnni_a[i][j][0][0]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif

  #ifdef USE_TBB
      tbb::parallel_for(0L, embedding_blocks*intermediate_blocks, [&o_vnni_tpp, &t_Wd_a, &t_Wd_vnni_a, embedding_blocks, intermediate_blocks](long idx) {
        long i = (idx / intermediate_blocks);
        long j = (idx % intermediate_blocks);
  #else
      #pragma omp parallel for collapse(2)
      for(int i = 0; i < embedding_blocks; i++){
        for(int j = 0; j < intermediate_blocks; j++){
  #endif
          o_vnni_tpp(&t_Wd_a[i][j][0][0], &t_Wd_vnni_a[i][j][0][0]);
  #ifndef USE_TBB
        }
      }
  #else
      });
  #endif

      // copy t_Wg_vnni contents t_Wg[l]
      t_Wg[l*num_expert + e].swap(t_Wg_vnni);
      t_Wu[l*num_expert + e].swap(t_Wu_vnni);
      t_Wd[l*num_expert + e].swap(t_Wd_vnni);
    }
  }
}


template<typename T>
void ffn_compute_flat(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1) {


  // auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  auto t_In_a = GetVLAPtr<T>(t_In.get(), {embedding_dim});
  auto t_Out_a = GetVLAPtr<T>(t_Out.get(), {embedding_dim});
  auto t_Wg_a = GetVLAPtr<T>(t_Wg.get(), {intermediate_dim});
  auto t_Wu_a = GetVLAPtr<T>(t_Wu.get(), {intermediate_dim});
  auto t_Wd_a = GetVLAPtr<T>(t_Wd.get(), {embedding_dim});

  long token_len_q = (token_len / MLP_BLOCKSIZE)*MLP_BLOCKSIZE;
  long token_len_re = (token_len % MLP_BLOCKSIZE);

  std::unique_ptr<T[]> t_inter (new (std::align_val_t(64)) T[token_len * intermediate_dim]);
  auto t_inter_a = GetVLAPtr<T>(t_inter.get(), {intermediate_dim});

  long embedding_blocks = embedding_dim/MLP_BLOCKSIZE ;
  long intermediate_blocks = intermediate_dim/MLP_BLOCKSIZE ;

  {
    // Gate computation
    {

#ifdef USE_TBB
      tbb::parallel_for(0L, (token_len/MLP_BLOCKSIZE)*intermediate_blocks, [&tpps_main, &t_In_a, &t_Wg_a, &t_inter_a, token_len_q, embedding_blocks, intermediate_blocks, b_vnni]
      (long idx) {

        long i = (idx / intermediate_blocks);
        long k = (idx % intermediate_blocks);
#else 
      // #pragma omp parallel for collapse(2)
      // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
      //   for (long k = 0; k < intermediate_blocks; k++) {
      #pragma omp parallel for
      for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*intermediate_blocks; idx++) {
        long i = (idx / intermediate_blocks);
        long k = (idx % intermediate_blocks);
#endif
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], embedding_blocks);
          else
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[0][k*MLP_BLOCKSIZE], &tmp[0], embedding_blocks);
          tpps_main.silu_tpp(&tmp[0], &tmp[0]);
          tpps_main.downconvert_inter_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        // }
      }
#else
      });
#endif
      if(token_len_re != 0){   //edge case or decode case
#ifdef USE_TBB
        tbb::parallel_for(0L, intermediate_blocks, [&tpps_edge, &t_In_a, &t_Wg_a, &t_inter_a, token_len_q, token_len_re, embedding_blocks, intermediate_blocks, b_vnni]
        (long k) {
#else
        #pragma omp parallel for
        for (long k = 0L; k < intermediate_blocks; k++) {
#endif
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], embedding_blocks);
          else
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[0][k*MLP_BLOCKSIZE], &tmp[0], embedding_blocks);
          tpps_edge.silu_tpp(&tmp[0], &tmp[0]);
          tpps_edge.downconvert_inter_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        }
#else
        });
#endif
      }
    }
  }


  if(gate_flag){
    // Up computation
    {

#ifdef USE_TBB
      tbb::parallel_for(0L, (token_len/MLP_BLOCKSIZE)*intermediate_blocks, [&tpps_main, &t_In_a, &t_Wu_a, &t_inter_a, token_len_q, embedding_blocks, intermediate_blocks, b_vnni]
      (long idx) {

        long i = (idx / intermediate_blocks);
        long k = (idx % intermediate_blocks);
#else      
      // #pragma omp parallel for collapse(2)
      // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
      //   for (long k = 0; k < intermediate_blocks; k++) {
      #pragma omp parallel for
      for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*intermediate_blocks; idx++) {
        long i = (idx / intermediate_blocks);
        long k = (idx % intermediate_blocks);
#endif
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], embedding_blocks);
          else
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[0][k*MLP_BLOCKSIZE], &tmp[0], embedding_blocks);
          tpps_main.mul_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        // }
      }
#else
      });
#endif
      if(token_len_re != 0){       //edge case or decode case

#ifdef USE_TBB
        tbb::parallel_for(0L, intermediate_blocks, [&tpps_edge, &t_In_a, &t_Wu_a, &t_inter_a, token_len_q, token_len_re, embedding_blocks, intermediate_blocks, b_vnni]
        (long k) {
#else
        #pragma omp parallel for
        for (long k = 0L; k < intermediate_blocks; k++) {
#endif
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], embedding_blocks);
          else
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[0][k*MLP_BLOCKSIZE], &tmp[0], embedding_blocks);
          tpps_edge.mul_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        }
#else
        });
#endif
      }
    }
  }

  {
    // Down computation
    {

#ifdef USE_TBB
      tbb::parallel_for(0L, (token_len/MLP_BLOCKSIZE)*embedding_blocks, [&tpps_main, &t_Out_a, &t_Wd_a, &t_inter_a, token_len_q, embedding_blocks, intermediate_blocks, b_vnni, scale]
      (long idx) {

        long i = (idx / embedding_blocks);
        long k = (idx % embedding_blocks);
#else 
      // #pragma omp parallel for collapse(2)
      // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
      //   for (long k = 0; k < embedding_blocks; k++) {
      #pragma omp parallel for
      for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*embedding_blocks; idx++) {
        long i = (idx / embedding_blocks);
        long k = (idx % embedding_blocks);
#endif
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.o_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][0], &t_Wd_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], intermediate_blocks);
          else
            tpps_main.o_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][0], &t_Wd_a[0][k*MLP_BLOCKSIZE], &tmp[0], intermediate_blocks);
          tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        // }
      }
#else
      });
#endif
      if(token_len_re != 0){       //edge case or decode case
#ifdef USE_TBB
        tbb::parallel_for(0L, embedding_blocks, [&tpps_edge, &t_Out_a, &t_Wd_a, &t_inter_a, token_len_q, token_len_re, embedding_blocks, intermediate_blocks, b_vnni, scale]
        (long k) {
#else
        #pragma omp parallel for
        for (long k = 0L; k < embedding_blocks; k++) {
#endif
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[0][k*MLP_BLOCKSIZE*2], &tmp[0], intermediate_blocks);
          else
            tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[0][k*MLP_BLOCKSIZE], &tmp[0], intermediate_blocks);
          tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        }
#else
        });
#endif
      }
    }
  }

  // auto end_time = std::chrono::high_resolution_clock::now(); // End timing
  // auto ffn_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  // return ffn_time;
}

// Read Processor ID using inline assembly
uint64_t read_processor_id() {
#if defined(__x86_64__) || defined(_M_X64)
    uint64_t rpid;
    __asm__ volatile ("rdpid %0" : "=r"(rpid));
    return rpid;
#else
    // Fallback for non-x86 architectures
    return 0;
#endif
}

template<typename T>
void ffn_compute_blocked(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1) {

  // std::cout << "Running with flat layout: \n";
  auto t_In_a = GetVLAPtr<T>(t_In.get(), {embedding_dim});
  auto t_Out_a = GetVLAPtr<T>(t_Out.get(), {embedding_dim});

  auto t_Wg_a = GetVLAPtr<T>(t_Wg.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});   // [n2, n1, h1, h2]
  auto t_Wu_a = GetVLAPtr<T>(t_Wu.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
  auto t_Wd_a = GetVLAPtr<T>(t_Wd.get(), {intermediate_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

  long token_len_q = (token_len / MLP_BLOCKSIZE)*MLP_BLOCKSIZE;
  long token_len_re = (token_len % MLP_BLOCKSIZE);

  std::unique_ptr<T[]> t_inter (new (std::align_val_t(64)) T[token_len * intermediate_dim]);
  auto t_inter_a = GetVLAPtr<T>(t_inter.get(), {intermediate_dim});

  long embedding_blocks = embedding_dim/MLP_BLOCKSIZE;
  long intermediate_blocks = intermediate_dim/MLP_BLOCKSIZE;
  bool gate_up;

  gate_up = (embedding_dim > intermediate_dim)? true : false;

  if(gate_flag && gate_up){
#ifdef USE_TBB
      tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE)*(intermediate_blocks), [&tpps_main, &t_In_a, &t_Wg_a, &t_Wu_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q]
      (long idx) {
        long i = (idx / (intermediate_blocks));
        long k = (idx % (intermediate_blocks));
#else
      // #pragma omp parallel for collapse(2)
      // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
      //   for (long k = 0; k < intermediate_blocks; k++) {
      #pragma omp parallel for
      for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*intermediate_blocks; idx++) {
        long i = (idx / (intermediate_blocks));
        long k = (idx % (intermediate_blocks));
#endif
          // printf("Core %d is processing idx=%d, i=%d, k=%d \n", read_processor_id(), idx, i, k);
          // print_list.push_back("Core " + std::to_string(read_processor_id()) + " is processing idx=" + std::to_string(idx) + ", i=" + std::to_string(i) + ", k=" + std::to_string(k) + "\n");
          LIBXSMM_ALIGNED(float tmp_g[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          LIBXSMM_ALIGNED(float tmp_u[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
          tpps_main.silu_tpp(&tmp_g[0], &tmp_g[0]);
          tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
          tpps_main.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        // }
      }
#else
      });
#endif

      if(token_len_re != 0){   //edge case or decode case
#ifdef USE_TBB
        tbb::parallel_for(0L, intermediate_blocks, [&tpps_edge, &t_In_a, &t_Wg_a, &t_Wu_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q, token_len_re]
          (long k) {
#else
        #pragma omp parallel for
        for (long k = 0L; k < intermediate_blocks; k++) {
#endif
          LIBXSMM_ALIGNED(float tmp_g[token_len_re * MLP_BLOCKSIZE], 64);
          LIBXSMM_ALIGNED(float tmp_u[token_len_re * MLP_BLOCKSIZE], 64);
          tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
          tpps_edge.silu_tpp(&tmp_g[0], &tmp_g[0]);
          tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
          tpps_edge.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        }
#else
        });
#endif
      }
  } else {    // Normal gate and up computation
    {
      // Gate computation
      {
#ifdef USE_TBB
        tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE)*(intermediate_blocks), [&tpps_main, &t_In_a, &t_Wg_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q]
        (long idx) {
        long i = (idx / (intermediate_blocks));
        long k = (idx % (intermediate_blocks));
#else
        // #pragma omp parallel for collapse(2)
        // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
        //   for (long k = 0; k < intermediate_blocks; k++) {
        #pragma omp parallel for
        for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*intermediate_blocks; idx++) {
          long i = (idx / (intermediate_blocks));
          long k = (idx % (intermediate_blocks));
#endif
            LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[k][0][0][0], &tmp[0], embedding_blocks);
            tpps_main.silu_tpp(&tmp[0], &tmp[0]);
            tpps_main.downconvert_inter_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
          // }
        }
#else
        });
#endif
        if(token_len_re != 0){   //edge case or decode case
#ifdef USE_TBB
        tbb::parallel_for(0L, intermediate_blocks, [&tpps_edge, &t_In_a, &t_Wg_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q, token_len_re]
          (long k) {
#else
          #pragma omp parallel for
          for (long k = 0L; k < intermediate_blocks; k++) {
#endif
            LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k][0][0][0], &tmp[0], embedding_blocks);
            tpps_edge.silu_tpp(&tmp[0], &tmp[0]);
            tpps_edge.downconvert_inter_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
          }
#else
          });
#endif
        }
      }
    }

    if(gate_flag){
      // Up computation
      {
#ifdef USE_TBB
        tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE)*(intermediate_blocks), [&tpps_main, &t_In_a, &t_Wu_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q]
        (long idx) {
        long i = (idx / (intermediate_blocks));
        long k = (idx % (intermediate_blocks));
#else
        // #pragma omp parallel for collapse(2)
        // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
        //   for (long k = 0; k < intermediate_blocks; k++) {
        #pragma omp parallel for
        for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*intermediate_blocks; idx++) {
          long i = (idx / (intermediate_blocks));
          long k = (idx % (intermediate_blocks));
#endif
            LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[k][0][0][0], &tmp[0], embedding_blocks);
            tpps_main.mul_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
          // }
        }
#else
        });
#endif
        if(token_len_re != 0){       //edge case or decode case
#ifdef USE_TBB
          tbb::parallel_for(0L, (intermediate_blocks), [&tpps_edge, &t_In_a, &t_Wu_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q, token_len_re]
          (long k) {
#else
          #pragma omp parallel for
          for (long k = 0L; k < intermediate_blocks; k++) {
#endif
            LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k][0][0][0], &tmp[0], embedding_blocks);
            tpps_edge.mul_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
          }
#else
          });
#endif
        }
      }
    }
  }

  {
    // Down computation
    {
#ifdef USE_TBB
      tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE)*(embedding_blocks), [&tpps_main, &t_Out_a, &t_Wd_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q, scale]
      (long idx) {
        long i = (idx / (embedding_blocks));
        long k = (idx % (embedding_blocks));
#else
      // #pragma omp parallel for collapse(2)
      // for (long i = 0; i < (token_len_q/MLP_BLOCKSIZE); i++) {
      //   for (long k = 0; k < embedding_blocks; k++) {
      #pragma omp parallel for
      for (long idx = 0L; idx < (token_len_q/MLP_BLOCKSIZE)*embedding_blocks; idx++) {
        long i = (idx / (embedding_blocks));
        long k = (idx % (embedding_blocks));
#endif
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          tpps_main.o_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][0], &t_Wd_a[k][0][0][0], &tmp[0], intermediate_blocks);
          tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        // }
      }
#else
      });
#endif
      
      if(token_len_re != 0){       //edge case or decode case
#ifdef USE_TBB
        tbb::parallel_for(0L, (embedding_blocks), [&tpps_edge, &t_Out_a, &t_Wd_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q, token_len_re, scale]
          (long k) {
#else
        #pragma omp parallel for
        for (long k = 0L; k < embedding_blocks; k++) {
#endif
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[k][0][0][0], &tmp[0], intermediate_blocks);
          tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][k*MLP_BLOCKSIZE]);
#ifndef USE_TBB
        }
#else
        });
#endif
      }
    }
  }

  // auto end_time = std::chrono::high_resolution_clock::now(); // End timing
  // auto ffn_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  // return ffn_time;

}

template<typename T>
void flops_and_bandwidth(
  long int time,
  long token_len,
  long embedding_dim,
  long intermediate_dim,
  long num_iter,
  long num_layer,
  long num_expert,
  bool gate_flag) {

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);

  auto flops = (double)4.0 * token_len * embedding_dim * intermediate_dim;
  if (gate_flag)
    flops += ((double)2.0 * token_len * embedding_dim *intermediate_dim) + (token_len * intermediate_dim);
  flops = flops / ((double)1e9 * time_ms);

  auto bw = ((double)2.0 * token_len * embedding_dim) + ((double) token_len * intermediate_dim) + ((double)2.0 * embedding_dim * intermediate_dim);
  if (gate_flag)
    bw += (intermediate_dim * embedding_dim);

  bw = (bw * sizeof(T)) / ((double)1e6 * time_ms);

  // print time
  printf("Time taken for ffn_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", time_ms, flops, bw);
}


template<typename T>
void correctness_checking(std::vector<std::unique_ptr<T[]>>& t_Out, 
                            std::vector<std::unique_ptr<T[]>>& t_In,
                            std::vector<std::unique_ptr<T[]>>& t_Wg,
                            std::vector<std::unique_ptr<T[]>>& t_Wu,
                            std::vector<std::unique_ptr<T[]>>& t_Wd,
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1, long num_layer=1, long num_expert=1) {

  std::cout<< "Starting Correctness Check" << "\n";                      
  std::vector<FFNTPPs<T>> ffn_flat_tpp_set = create_ffn_tpp_set<T>(0, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  std::vector<FFNTPPs<T>> ffn_blocked_tpp_set = create_ffn_tpp_set<T>(1, embedding_dim, intermediate_dim, gate_flag, b_vnni);

  std::vector<std::unique_ptr<T[]>> t2_Out(num_layer), t2_Wg(num_layer*num_expert), t2_Wu(num_layer*num_expert), t2_Wd(num_layer*num_expert);
  for(int l = 0; l < num_layer; l++){
    t2_Out[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);
#ifdef USE_TBB
    tbb::parallel_for(0L, num_expert, [&](long e) {
#else
    #pragma omp parallel for
    for(long e = 0; e < num_expert; e++){
#endif
      t2_Wg[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim*intermediate_dim]);
      t2_Wu[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim*intermediate_dim]);
      t2_Wd[l*num_expert + e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim*embedding_dim]);

      std::memcpy(t2_Wg[l*num_expert + e].get(), t_Wg[l*num_expert + e].get(), sizeof(T)*embedding_dim*intermediate_dim);
      std::memcpy(t2_Wu[l*num_expert + e].get(), t_Wu[l*num_expert + e].get(), sizeof(T)*embedding_dim*intermediate_dim);
      std::memcpy(t2_Wd[l*num_expert + e].get(), t_Wd[l*num_expert + e].get(), sizeof(T)*intermediate_dim*embedding_dim);
#ifndef USE_TBB
    }
#else
    });
#endif
  }
  flat_to_blocked_weights<T>(t2_Wg, t2_Wu, t2_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);

  if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
    flat_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);
    blocked_weight_vnni_transform<T>(t2_Wg, t2_Wu, t2_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);
  }

  ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_flat_tpp_set[0], ffn_flat_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
  ffn_compute_blocked<T>(t2_Out[0], t_In[0], t2_Wg[0], t2_Wu[0], t2_Wd[0], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);

  auto upconvert_tpp = SCOPEIT((ConvertTPP<T, float>(embedding_dim)), EW_ZERO);
  for(int l=0; l < num_layer; l++){
    for(int i=0; i < token_len; i++){
      float t_Out_float[embedding_dim], t2_Out_float[embedding_dim];
      upconvert_tpp(&t_Out[l][i*embedding_dim], &t_Out_float[0]);
      upconvert_tpp(&t2_Out[l][i*embedding_dim], &t2_Out_float[0]);
      float tol;
      if(std::is_same<T, bfloat16>::value)
        tol = 1e-2;
      else
        tol = 1e-5;
      for(int j=0; j < embedding_dim; j++){
        if(std::abs((t_Out_float[j] - t2_Out_float[j])/t_Out_float[j]) > tol){
          std::cout << "Layer 0: Mismatch at index " << (i*embedding_dim + j) << ": " << (float)t_Out[0][j] << " vs " << (float)t2_Out[0][j] << "\n";
          exit(0);
        }
      }
    }
  }
  std::cout<< "Correctness Check done" << "\n";
}


int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <b_vnni> <blocked> <seq_len> <num_layer> <num_iter> <embedding_dim> <intermediate_dim> <gate_flag> <correctness_check>" << std::endl;
    return 1;
  }
  int num_threads = omp_get_max_threads();
#ifdef USE_TBB
  // Define stack size (e.g., 8 MB)
  // const std::size_t stack_size = 32 * 1024 * 1024; // bytes
  // tbb::global_control global_limit(tbb::global_control::thread_stack_size, stack_size);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
#endif

  long batch_size = std::stoi(argv[1]);
  long seq_len = std::stoi(argv[2]);
  long token_len = batch_size * seq_len;
  bool b_vnni = std::stoi(argv[3]);
  bool blocked = std::stoi(argv[4]);
  long num_layer = std::stoi(argv[5]);
  long num_iter = std::stoi(argv[6]);
  long embedding_dim = std::stoi(argv[7]);
  long intermediate_dim = std::stoi(argv[8]);
  long num_expert = std::stoi(argv[9]);
  long num_expert_per_token = std::stoi(argv[10]);
  bool gate_flag = std::stoi(argv[11]);
  bool correctness_check = std::stoi(argv[12]);

  std::vector<std::unique_ptr<T[]>> t_In(num_layer), t_Out(num_layer), t_Wg(num_layer*num_expert), t_Wu(num_layer*num_expert), t_Wd(num_layer*num_expert);
  activation_allocate_and_initialize<T>(t_Out, t_In, token_len, embedding_dim, intermediate_dim, num_layer);
  flat_weight_allocate_and_initialize<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);

  std::vector<FFNTPPs<T>> ffn_tpp_set = create_ffn_tpp_set<T>(blocked, embedding_dim, intermediate_dim, gate_flag, b_vnni);

  if(correctness_check){
    correctness_checking(t_Out, t_In, t_Wg, t_Wu, t_Wd, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  }

  // Timing
  long int time = 0;
  if(blocked){
    std::cout << "Running with blocked weight layout \n";
    flat_to_blocked_weights<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);
    if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
      blocked_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);
    }
    if (num_expert > 1) {  // MoE FFN
      std::cout << "Number of Experts: " << num_expert << "\n";
      std::cout << "Number of Experts per token: " << num_expert_per_token << "\n";
      std::vector<std::unique_ptr<T[]>> t_split_In(num_expert);
      std::vector<std::unique_ptr<T[]>> t_split_Out(num_expert);
      std::vector<long> expert_token_counts(num_expert, 0);
      std::vector<std::vector<long>> token_to_expert(token_len);
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      for(int i = 0; i < num_iter; i++) {
        for(int l = 1; l < num_layer; l++) {
          // Instead of clearing and resizing, just reset the unique_ptrs
          for (auto& ptr : t_split_In) ptr.reset();
          for (auto& ptr : t_split_Out) ptr.reset();
          for (auto& count : expert_token_counts) count = 0;
          for (auto& vec : token_to_expert) vec.clear();
          moe_split_layer_input<T>(t_In[l-1],
                      t_split_In,
                      t_split_Out,
                      expert_token_counts,
                      token_to_expert,
                      token_len, embedding_dim, num_expert, num_expert_per_token);

#ifdef USE_TBB
          tbb::parallel_for(0L, num_expert, [&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_counts, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni]
          (long e) {
#else
          for(long e = 0; e < num_expert; e++){
#endif
            if (expert_token_counts[e] != 0){   // some tokens for this expert
// #ifdef USE_TBB
//               tbb::task_arena nested; // Limit to 8 threads per expert
//               nested.execute([&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_counts, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni, e]() {
// #endif
                ffn_compute_blocked<T>(t_split_Out[e], t_split_In[e], t_Wg[l*num_expert + e], t_Wu[l*num_expert + e], t_Wd[l*num_expert + e], ffn_tpp_set[0], ffn_tpp_set[expert_token_counts[e] % MLP_BLOCKSIZE], expert_token_counts[e], embedding_dim, intermediate_dim, gate_flag, b_vnni);
// #ifdef USE_TBB
//               });
// #endif
            }
#ifndef USE_TBB
          }
#else
          });
#endif
          moe_combine_layer_output<T>(t_Out[l], t_split_Out, token_to_expert, token_len, embedding_dim, num_expert);
        }
      }
      auto end_time = std::chrono::high_resolution_clock::now(); // End timing
      time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    } else {

      // Warm up
      ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      for(int l = 1; l < num_layer; l++) {
        ffn_compute_blocked<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      }
      
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      for(int i = 0; i < num_iter; i++) {
        ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        for(int l = 1; l < num_layer; l++) {
          ffn_compute_blocked<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        }
      }

      auto end_time = std::chrono::high_resolution_clock::now(); // End timing
      time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }
  } else {
    std::cout << "Running with flat weight layout \n";
    if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
      flat_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, num_expert);
    }

    // Warm up
    ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    for(int l = 1; l < num_layer; l++) {
      ffn_compute_flat<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
    for(int i = 0; i < num_iter; i++) {
      ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      for(int l = 1; l < num_layer; l++) {
        ffn_compute_flat<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // End timing
    time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  }

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);
  printf("Total Compute Time taken: %0.2f s \n", time / ((double)1e6));
  flops_and_bandwidth<T>(time, token_len, embedding_dim, intermediate_dim, num_iter, num_layer, num_expert, gate_flag);

  return 0;
}

// auto t_In = torch.randn({B_t, S_t, embedding_dim});
// auto t_Wg = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wu = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wd = t_In.randn({4*embedding_dim, embedding_dim});

// auto t_I = i_gemm(SiluPostOp(), t_In, t_Wg, t_null);
// t_I = i_gemm(MulPostOp(t_I), t_In, t_Wu, t_null);
// auto t_Out = o_gemm(AddScalePostOp(t_res, scale), t_I, t_Wd, t_null);