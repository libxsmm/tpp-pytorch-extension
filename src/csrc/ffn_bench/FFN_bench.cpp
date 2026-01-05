#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <system_error>
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
  #include <oneapi/tbb/parallel_for.h>
  #include <oneapi/tbb/blocked_range.h>
  #include <oneapi/tbb/global_control.h>
  #include <oneapi/tbb/concurrent_vector.h>
  #include <oneapi/tbb/flow_graph.h>
  #include <oneapi/tbb/queuing_mutex.h>
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
                          //  long* expert_token_counts,
                          std::vector<std::vector<long>>& expert_token_ids,
  std::vector<std::vector<long>>& token_to_expert,
  long token_len, long embedding_dim, long num_expert, long num_expert_per_token) {

  std::random_device rd;
  std::mt19937 gen(rd());
  
  // // Zipf's Law distribution
  // std::vector<double> probs(num_expert, 0.0);
  // for (int k=1; k <= num_expert; k++) {
  //   probs[k-1] = 1.0 / std::pow((double) k, 1.5);
  // } 
  // auto sum = std::accumulate(probs.begin(), probs.end()-1, 0.0);
  // for (auto& p : probs) {
  //   p = p / sum;
  // }
  // std::discrete_distribution<int> dist(probs.begin(), probs.end());

  std::uniform_int_distribution<long> dist(0, num_expert - 1);
  // #pragma omp parallel for reduction(+:expert_token_counts[:num_expert])
  for (long t = 0; t < token_len; t++){
    // randomly select num_experts_per_token from num_experts for each token
    for (long e = 0; e < num_expert_per_token; e++) {
      long assigned_expert;
      bool unique = false;
      while (!unique) {
        assigned_expert = dist(gen);
        if (std::find(token_to_expert[t].begin(), token_to_expert[t].end(), assigned_expert) == token_to_expert[t].end()) {
          unique = true;
        }
      }
      token_to_expert[t][e] = assigned_expert;
      // expert_token_counts[assigned_expert]++;
      expert_token_ids[assigned_expert].push_back(t);
    }

    // //round robin assignment
    // for (long e = 0; e < num_expert_per_token; e++) {
    //   long assigned_expert = (t*num_expert_per_token + e) % num_expert;
    //   // token_to_expert[t].push_back(assigned_expert);
    //   token_to_expert[t][e] = assigned_expert;
    //   expert_token_counts[assigned_expert]++;
    // }
  }

#ifdef USE_TBB
  tbb::parallel_for(0L, num_expert, [&](long e) {
#else
  #pragma omp parallel for
  for (long e = 0; e < num_expert; e++){
#endif
    // allocate input buffer for each expert
    t_split_In[e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[expert_token_ids[e].size() * embedding_dim]);
    t_split_Out[e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[expert_token_ids[e].size() * embedding_dim]);
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
#ifndef USE_TBB
  }
#else
  });
#endif
}

template<typename T>
void moe_combine_layer_output(std::unique_ptr<T[]>& t_Out,
                           std::vector<std::unique_ptr<T[]>>& t_split_Out,
  std::vector<std::vector<long>>& token_to_expert,
  long token_len, long embedding_dim, long num_expert) {

  auto add_embed_tpp = AddTPP<T, float, float>(embedding_dim);
  auto convert_embed_tpp = ConvertTPP<float, T>(embedding_dim);
  std::vector<long> expert_offsets(num_expert, 0);

  for (long t = 0; t < token_len; t++){
    std::vector<float> t_Out_float(embedding_dim, 0.0f);
    for (auto& expert : token_to_expert[t]) {
      // accumulate outputs from different experts
      add_embed_tpp(&t_split_Out[expert][expert_offsets[expert]],
                    t_Out_float.data(),
                    t_Out_float.data());
      // move expert output pointer
      expert_offsets[expert] += embedding_dim;
    }
    // copy accumulated output to final output buffer
    convert_embed_tpp(t_Out_float.data(), &t_Out[t * embedding_dim]);
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
        std::uniform_real_distribution<float> dist(0.0f, 0.009f);

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
        std::uniform_real_distribution<float> dist(0.0f, 0.009f);

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

class src_body {
    const long my_limit;
    long my_next_value;
public:
    src_body(long l) : my_limit(l), my_next_value(0) {}
    int operator()( tbb::flow_control& fc ) {
        if ( my_next_value < my_limit ) {
            // std::this_thread::sleep_for(std::chrono::microseconds(10));
            return my_next_value++;
        } else {
            fc.stop();
            return int();
        }
    }
};

template<typename T>
void ffn_compute_dataflow(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1) {

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
  bool gate_up = true;
  // gate_up = (embedding_dim > intermediate_dim)? true : false;

  auto zero_tpp_main = SetZeroTPP<float>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);
  auto zero_tpp_edge = SetZeroTPP<float>(token_len_re, MLP_BLOCKSIZE, embedding_dim);
  auto convert_tpp_main = ConvertTPP<float, T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim, embedding_dim);
  auto convert_tpp_edge = ConvertTPP<float, T>(token_len_re, MLP_BLOCKSIZE, embedding_dim, embedding_dim);
  auto o_single_main_gemm_tpp = BrgemmTPP<
            T,
            float>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 1, 1, intermediate_dim, MLP_BLOCKSIZE, embedding_dim, 1.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  auto o_single_edge_gemm_tpp = BrgemmTPP<
            T,
            float>(token_len_re, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 1, 1, intermediate_dim, MLP_BLOCKSIZE, embedding_dim, 1.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  
  if(gate_flag && gate_up){
      int N = 8;
      tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE), [&tpps_main, &t_In_a, &t_Wg_a, &t_Wu_a, &t_Wd_a, &t_Out_a, &t_inter_a, &zero_tpp_main, &o_single_main_gemm_tpp, &convert_tpp_main, 
        embedding_blocks, intermediate_blocks, token_len_q, N, scale](long i) {

        std::unique_ptr<float[]> t_Out_float (new (std::align_val_t(64)) float[MLP_BLOCKSIZE * (embedding_blocks * MLP_BLOCKSIZE)]);
        auto t_Out_float_a = GetVLAPtr<float>(t_Out_float.get(), {(embedding_blocks * MLP_BLOCKSIZE)});
        // set t_Out_a to zero
        for (long j = 0; j < embedding_blocks; j++) {
          zero_tpp_main(&t_Out_float_a[0][j*MLP_BLOCKSIZE]);
        }

        tbb::queuing_mutex mtx[N];
        tbb::flow::graph g;
        tbb::flow::input_node<long> src( g, src_body(intermediate_blocks) );

        tbb::flow::function_node<long, long> up_node(g, tbb::flow::unlimited,
        [&tpps_main, &t_In_a, &t_Wg_a, &t_Wu_a, &t_inter_a, embedding_blocks, intermediate_blocks, i, scale](const long &k){
            LIBXSMM_ALIGNED(float tmp_g[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            LIBXSMM_ALIGNED(float tmp_u[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
            tpps_main.silu_tpp(&tmp_g[0], &tmp_g[0]);
            tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
            tpps_main.scale_tpp(&tmp_u[0], &tmp_u[0], scale);
            tpps_main.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
            return k;
        });

        std::vector<tbb::flow::function_node<long>> down_nodes;

        for (int n = 0; n < N; n++) {
          down_nodes.emplace_back(g, tbb::flow::unlimited, 
            [&t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_main_gemm_tpp, &mtx,
              N, embedding_blocks, intermediate_blocks, token_len_q, n, i](const long &k){
              long j_start = n * (embedding_blocks / N);
              long j_end = (n + 1) * (embedding_blocks / N);
              tbb::queuing_mutex::scoped_lock lock(mtx[n]);
              for (long j = j_start; j < j_end; j++) {
                  o_single_main_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE], &t_Wd_a[j][k][0][0], &t_Out_float_a[0][j*MLP_BLOCKSIZE], 1);
              }
          });
        }

        // tbb::flow::function_node<long> down_node(g, 1, 
        //   [&tpps_main, &t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_main_gemm_tpp, 
        //     embedding_blocks, intermediate_blocks, i](const long &k){
        //     // for (long j = 0L; j < embedding_blocks; j++) {
        //     tbb::parallel_for(tbb::blocked_range<long>(0L, embedding_blocks),
        //     [&tpps_main, &t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_main_gemm_tpp, 
        //       embedding_blocks, intermediate_blocks, i, k](const tbb::blocked_range<long>& r) {
        //       for (long j = r.begin(); j != r.end(); j++){
        //         o_single_main_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE], &t_Wd_a[j][k][0][0], &t_Out_float_a[0][j*MLP_BLOCKSIZE], 1);
        //       }
        //     });
        //     // }
        // });

        tbb::flow::make_edge(src, up_node);
        // tbb::flow::make_edge(up_node, down_node);
        for (int n = 0; n < N; n++) {
          tbb::flow::make_edge(up_node, down_nodes[n]);
        }
        src.activate();
        g.wait_for_all();

        for (long j = 0; j < embedding_blocks; j++) {
          convert_tpp_main(&t_Out_float_a[0][j*MLP_BLOCKSIZE], &t_Out_a[i*MLP_BLOCKSIZE][j*MLP_BLOCKSIZE]);
        }
        // for (long k = 0; k < intermediate_blocks; k++) {
        //   LIBXSMM_ALIGNED(float tmp_g[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
        //   LIBXSMM_ALIGNED(float tmp_u[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
        //   tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
        //   tpps_main.silu_tpp(&tmp_g[0], &tmp_g[0]);
        //   tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
        //   tpps_main.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
        // }

        // for (long k = 0; k < embedding_blocks; k++) {
        //   LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
        //   tpps_main.o_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][0], &t_Wd_a[k][0][0][0], &tmp[0], intermediate_blocks);
        //   tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
        //   tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
        // }
      });

      if(token_len_re != 0){   //edge case or decode case
        std::unique_ptr<float[]> t_Out_float (new (std::align_val_t(64)) float[token_len_re * embedding_dim]);
        auto t_Out_float_a = GetVLAPtr<float>(t_Out_float.get(), {embedding_dim});
        for (long j = 0; j < embedding_blocks; j++) {
          zero_tpp_edge(&t_Out_float_a[0][j*MLP_BLOCKSIZE]);
        }

        tbb::queuing_mutex mtx[N];
        tbb::flow::graph g;
        tbb::flow::function_node<long, long> up_node(g, tbb::flow::unlimited,
        [&tpps_edge, &t_In_a, &t_Wg_a, &t_Wu_a, &t_inter_a,
          embedding_blocks, intermediate_blocks, token_len_q, token_len_re, scale](const long &k){
            LIBXSMM_ALIGNED(float tmp_g[token_len_re * MLP_BLOCKSIZE], 64);
            LIBXSMM_ALIGNED(float tmp_u[token_len_re * MLP_BLOCKSIZE], 64);
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
            tpps_edge.silu_tpp(&tmp_g[0], &tmp_g[0]);
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
            tpps_edge.scale_tpp(&tmp_u[0], &tmp_u[0], scale);
            tpps_edge.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
            return k;
        });

        std::vector<tbb::flow::function_node<long>> down_nodes;

        for (int n = 0; n < N; n++) {
          down_nodes.emplace_back(g, tbb::flow::unlimited, 
            [&t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_edge_gemm_tpp, &mtx,
              N, embedding_blocks, intermediate_blocks, token_len_q, n](const long &k){
              long j_start = n * (embedding_blocks / N);
              long j_end = (n + 1) * (embedding_blocks / N);
              tbb::queuing_mutex::scoped_lock lock(mtx[n]);
              for (long j = j_start; j < j_end; j++) {
                  o_single_edge_gemm_tpp(&t_inter_a[token_len_q][k*MLP_BLOCKSIZE], &t_Wd_a[j][k][0][0], &t_Out_float_a[0][j*MLP_BLOCKSIZE], 1);
              }
          });
        }

        // tbb::flow::function_node<long> down_node(g, 1, 
        //   [&tpps_edge, &t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_edge_gemm_tpp, &mtx,
        //     embedding_blocks, intermediate_blocks, token_len_q](const long &k){
        //     tbb::queuing_mutex::scoped_lock lock(mtx);
        //     for (long j = 0L; j < embedding_blocks; j++) {
        //     // tbb::parallel_for(tbb::blocked_range<long>(0L, embedding_blocks), 
        //     //   [&tpps_edge, &t_Out_float_a, &t_Wd_a, &t_inter_a, &o_single_edge_gemm_tpp, 
        //     //     embedding_blocks, intermediate_blocks, token_len_q, k](const tbb::blocked_range<long>& r) {
        //     //     for (long j = r.begin(); j != r.end(); j++) {
        //     //       // tbb::queuing_mutex::scoped_lock lock(mtx);
        //           o_single_edge_gemm_tpp(&t_inter_a[token_len_q][k*MLP_BLOCKSIZE], &t_Wd_a[j][k][0][0], &t_Out_float_a[0][j*MLP_BLOCKSIZE], 1);
        //     //     }
        //     // });
        //     }
        // }); 

        // tbb::flow::make_edge(up_node, down_node);
        for (int n = 0; n < N; n++) {
            tbb::flow::make_edge(up_node, down_nodes[n]);
        }
        tbb::flow::input_node<long> src( g, src_body(intermediate_blocks) );
        tbb::flow::make_edge(src, up_node);
        src.activate();
        g.wait_for_all();

        // copy back to t_Out_a
        for (long j = 0; j < embedding_blocks; j++) {
          convert_tpp_edge(&t_Out_float_a[0][j*MLP_BLOCKSIZE], &t_Out_a[token_len_q][j*MLP_BLOCKSIZE]);
        }

        // for (long k = 0; k < intermediate_blocks; k++) {
        //   LIBXSMM_ALIGNED(float tmp_g[token_len_re * MLP_BLOCKSIZE], 64);
        //   LIBXSMM_ALIGNED(float tmp_u[token_len_re * MLP_BLOCKSIZE], 64);
        //   tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k][0][0][0], &tmp_g[0], embedding_blocks);
        //   tpps_edge.silu_tpp(&tmp_g[0], &tmp_g[0]);
        //   tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k][0][0][0], &tmp_u[0], embedding_blocks);
        //   tpps_edge.gateup_mul_tpp(&tmp_u[0], &tmp_g[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
        // }

        // for (long j = 0L; j < embedding_blocks; j++) {
        //   LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
        //   tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[j][0][0][0], &tmp[0], intermediate_blocks);
        //   tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
        //   tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][j*MLP_BLOCKSIZE]);
        // }
      }
  } else {    // Normal gate and up computation
    // {
    //   {
    //     tbb::parallel_for(0L, (token_len_q/MLP_BLOCKSIZE), [&tpps_main, &t_In_a, &t_Wg_a, &t_inter_a, embedding_blocks, intermediate_blocks, token_len_q]
    //     (long i) {
    //       for (long k = 0; k < intermediate_blocks; k++) {
    //         LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
    //         tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wg_a[k][0][0][0], &tmp[0], embedding_blocks);
    //         tpps_main.silu_tpp(&tmp[0], &tmp[0]);
    //         tpps_main.downconvert_inter_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
    //       }

    //       if(gate){
    //         for (long k = 0; k < intermediate_blocks; k++) {
    //           LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
    //           tpps_main.i_gemm_tpp(&t_In_a[i*MLP_BLOCKSIZE][0], &t_Wu_a[k][0][0][0], &tmp[0], embedding_blocks);
    //           tpps_main.mul_tpp(&tmp[0], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE], &t_inter_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
    //         }
    //       }

    //       for (long k = 0; k < embedding_blocks; k++) {         // Down computation
    //         LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
    //         tpps_main.o_gemm_tpp(&t_inter_a[i*MLP_BLOCKSIZE][0], &t_Wd_a[k][0][0][0], &tmp[0], intermediate_blocks);
    //         tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
    //         tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i*MLP_BLOCKSIZE][k*MLP_BLOCKSIZE]);
    //       }
    //     });

    //     if(token_len_re != 0){   //edge case or decode case
    //       for (long k = 0; k < intermediate_blocks; k++) {
    //         LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
    //         tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k][0][0][0], &tmp[0], embedding_blocks);
    //         tpps_edge.silu_tpp(&tmp[0], &tmp[0]);
    //         tpps_edge.downconvert_inter_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
    //       }

    //       if (gate_flag){
    //         for (long k = 0L; k < intermediate_blocks; k++) {
    //           LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
    //           tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k][0][0][0], &tmp[0], embedding_blocks);
    //           tpps_edge.mul_tpp(&tmp[0], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE], &t_inter_a[token_len_q][k*MLP_BLOCKSIZE]);
    //         }
    //       }

    //       for (long k = 0L; k < embedding_blocks; k++) {         // Down computation
    //         LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
    //         tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[k][0][0][0], &tmp[0], intermediate_blocks);
    //         tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
    //         tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][k*MLP_BLOCKSIZE]);
    //       }
    //     }
    //   }
    // }
  }
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
  long num_expert_per_token,
  bool gate_flag,
  double average_active_experts) {

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);

  auto flops = (double)4.0 * token_len * embedding_dim * intermediate_dim * num_expert_per_token;
  if (gate_flag)
    flops += (((double)2.0 * token_len * embedding_dim *intermediate_dim) + (token_len * intermediate_dim)) * num_expert_per_token;
  flops = flops / ((double)1e9 * time_ms);

  auto bw = ((double)2.0 * token_len * embedding_dim) + ((double) token_len * intermediate_dim) + ((double)2.0 * embedding_dim * intermediate_dim);
  if (gate_flag)
    bw += (intermediate_dim * embedding_dim);

  bw = ((bw * sizeof(T)) * (average_active_experts)) / ((double)1e6 * time_ms);

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
                            bool gate_flag, bool b_vnni, long num_expert, long num_expert_per_token, long num_layer=1, float scale=0.1) {

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

  if (num_expert > 1) {
    std::vector<std::unique_ptr<T[]>> t_split_In(num_expert);
    std::vector<std::unique_ptr<T[]>> t_split_Out(num_expert);
    std::vector<std::unique_ptr<T[]>> t2_split_Out(num_expert);
    // std::vector<long> expert_token_counts(num_expert, 0);
    // long* expert_token_counts = new long[num_expert];
    // std::vector<std::vector<long>> token_to_expert(token_len);
    std::vector<std::vector<long>> expert_token_ids(num_expert);
    std::vector<std::vector<long>> token_to_expert(token_len, std::vector<long>(num_expert_per_token, -1));

    for(int l = 0; l < num_layer; l++) {
      // Instead of clearing and resizing, just reset the unique_ptrs
      for (auto& ptr : t_split_In) ptr.reset();
      for (auto& ptr : t_split_Out) ptr.reset();
      // for (long i = 0; i < num_expert; i++) expert_token_counts[i] = 0;
      for (long i = 0; i < num_expert; i++) expert_token_ids[i].clear();
      for (auto& vec : token_to_expert) std::fill(vec.begin(), vec.end(), -1);
      if (l == 0){
        moe_split_layer_input<T>(t_In[0],
                    t_split_In,
                    t_split_Out,
                    expert_token_ids,
                    token_to_expert,
                    token_len, embedding_dim, num_expert, num_expert_per_token);
      } else {
        moe_split_layer_input<T>(t_Out[l-1],
                    t_split_In,
                    t_split_Out,
                    expert_token_ids,
                    token_to_expert,
                    token_len, embedding_dim, num_expert, num_expert_per_token);
      }

      // make another copy of t_split_Out for blocked layout
      for(long e = 0; e < num_expert; e++){
        t2_split_Out[e] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[expert_token_ids[e].size() * embedding_dim]);
        std::memcpy(t2_split_Out[e].get(), t_split_Out[e].get(), sizeof(T)*expert_token_ids[e].size() *embedding_dim);
      }

  #ifdef USE_TBB
      tbb::parallel_for(0L, num_expert, [&ffn_flat_tpp_set, &ffn_blocked_tpp_set, &t_split_Out, &t2_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &t2_Wg, &t2_Wu, &t2_Wd, &expert_token_ids, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni]
      (long e) {
  #else
      for(long e = 0; e < num_expert; e++){          // no omp parallel for here to avoid too many threads
  #endif
        if (expert_token_ids[e].size() != 0){   // some tokens for this expert
          ffn_compute_flat<T>(t_split_Out[e], t_split_In[e], t_Wg[0*num_expert + e], t_Wu[0*num_expert + e], t_Wd[0*num_expert + e], ffn_flat_tpp_set[0], ffn_flat_tpp_set[expert_token_ids[e].size() % MLP_BLOCKSIZE], expert_token_ids[e].size(), embedding_dim, intermediate_dim, gate_flag, b_vnni);
          ffn_compute_blocked<T>(t2_split_Out[e], t_split_In[e], t2_Wg[0*num_expert + e], t2_Wu[0*num_expert + e], t2_Wd[0*num_expert + e], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[expert_token_ids[e].size() % MLP_BLOCKSIZE], expert_token_ids[e].size(), embedding_dim, intermediate_dim, gate_flag, b_vnni);
        }
  #ifndef USE_TBB
      }
  #else
      });
  #endif
      moe_combine_layer_output<T>(t_Out[l], t_split_Out, token_to_expert, token_len, embedding_dim, num_expert);
      moe_combine_layer_output<T>(t2_Out[l], t2_split_Out, token_to_expert, token_len, embedding_dim, num_expert);
    }
    // delete [] expert_token_counts;
  } else {
    for (long l=0; l < num_layer; l++){
      if (l == 0){
        ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_flat_tpp_set[0], ffn_flat_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
        // ffn_compute_blocked<T>(t2_Out[0], t_In[0], t2_Wg[0], t2_Wu[0], t2_Wd[0], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
        ffn_compute_dataflow<T>(t2_Out[0], t_In[0], t2_Wg[0], t2_Wu[0], t2_Wd[0], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
      } else {
        ffn_compute_flat<T>(t_Out[l], t_Out[l-1], t_Wg[0], t_Wu[0], t_Wd[0], ffn_flat_tpp_set[0], ffn_flat_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
        // ffn_compute_blocked<T>(t2_Out[l], t2_Out[l-1], t2_Wg[0], t2_Wu[0], t2_Wd[0], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
        ffn_compute_dataflow<T>(t2_Out[l], t2_Out[l-1], t2_Wg[0], t2_Wu[0], t2_Wd[0], ffn_blocked_tpp_set[0], ffn_blocked_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
      }
    }
  }

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
        float diff = std::abs(t_Out_float[j] - t2_Out_float[j]);
        float largest = std::max(std::abs(t_Out_float[j]), std::abs(t2_Out_float[j]));
        // if(std::abs((t_Out_float[j] - t2_Out_float[j])/t_Out_float[j]) > tol){
        if (diff > (largest*tol)){
          std::cout << "Layer 0: Mismatch at index " << (i*embedding_dim + j) << ": " << (float)t_Out[0][i*embedding_dim + j] << " vs " << (float)t2_Out[0][i*embedding_dim + j] << "\n";
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
#else
  // omp_set_max_active_levels(2);
  // int upper_level_threads = std::sqrt(num_threads);
  // int lower_level_threads = num_threads / upper_level_threads;
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
    correctness_checking(t_Out, t_In, t_Wg, t_Wu, t_Wd, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, num_expert, num_expert_per_token, num_layer);
  }

  // Timing
  long int time = 0;
  double average_active_experts = 0.0; 
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
      // std::vector<long> expert_token_counts(num_expert, 0);
      // long* expert_token_counts = new long[num_expert];
      // std::vector<std::vector<long>> token_to_expert(token_len);
      std::vector<std::vector<long>> expert_token_ids(num_expert);
      std::vector<std::vector<long>> token_to_expert(token_len, std::vector<long>(num_expert_per_token, -1));
      auto cold_start_time = std::chrono::high_resolution_clock::now();
      auto cold_end_time = std::chrono::high_resolution_clock::now();
      long int split_time = 0, merge_time = 0;
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      for(int i = 0; i < num_iter + 1; i++) {
        if (i == 0)
          cold_start_time = std::chrono::high_resolution_clock::now(); // Start timing for cold run
        for(int l = 0; l < num_layer; l++) {
          // Instead of clearing and resizing, just reset the unique_ptrs
          for (auto& ptr : t_split_In) ptr.reset();
          for (auto& ptr : t_split_Out) ptr.reset();
          // for (long i = 0; i < num_expert; i++) expert_token_counts[i] = 0;
          for (long i = 0; i < num_expert; i++) expert_token_ids[i].clear();
          #pragma omp parallel for
          for (auto& vec : token_to_expert) std::fill(vec.begin(), vec.end(), -1);
          auto split_start = std::chrono::high_resolution_clock::now();
          if (l == 0) {
            moe_split_layer_input<T>(t_In[0],
                      t_split_In,
                      t_split_Out,
                      expert_token_ids,
                      token_to_expert,
                      token_len, embedding_dim, num_expert, num_expert_per_token);
          } else {
            moe_split_layer_input<T>(t_Out[l-1],
                        t_split_In,
                        t_split_Out,
                        expert_token_ids,
                        token_to_expert,
                        token_len, embedding_dim, num_expert, num_expert_per_token);
          }
          auto split_end =  std::chrono::high_resolution_clock::now();
          split_time += std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start).count();

          std::vector<long> active_experts;
          for(long e = 0; e < num_expert; e++){
            if (expert_token_ids[e].size() != 0){
              active_experts.push_back(e);
            }
          }

#ifdef USE_TBB
          // tbb::parallel_for(0L, num_expert, [&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_counts, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni]
          // (long e) {
          tbb::parallel_for(0L, (long)active_experts.size(), [&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_ids, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni, &active_experts]
          (long idx) {
            long e = active_experts[idx];
#else
          #pragma omp parallel for schedule(dynamic)
          for (auto& e : active_experts){
          // for(long e = 0; e < num_expert; e++){
#endif
            // if (expert_token_counts[e] != 0){   // some tokens for this expert
// #ifdef USE_TBB
//               tbb::task_arena nested(8); // Limit to 8 threads per expert
//               nested.execute([&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_counts, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni, e]() {
// #endif
                ffn_compute_blocked<T>(t_split_Out[e], t_split_In[e], t_Wg[l*num_expert + e], t_Wu[l*num_expert + e], t_Wd[l*num_expert + e], ffn_tpp_set[0], ffn_tpp_set[expert_token_ids[e].size() % MLP_BLOCKSIZE], expert_token_ids[e].size(), embedding_dim, intermediate_dim, gate_flag, b_vnni);
// #ifdef USE_TBB
//               });
// #endif
            // }
#ifndef USE_TBB
          }
#else
          });
#endif
          auto merge_start = std::chrono::high_resolution_clock::now();
          moe_combine_layer_output<T>(t_Out[l], t_split_Out, token_to_expert, token_len, embedding_dim, num_expert);
          auto merge_end =  std::chrono::high_resolution_clock::now();
          merge_time += std::chrono::duration_cast<std::chrono::microseconds>(merge_end - merge_start).count();
          if (i > 0)
            average_active_experts += active_experts.size();
          else{
            split_time = 0;
            merge_time = 0;
          }
        }
        if (i == 0)
          cold_end_time = std::chrono::high_resolution_clock::now(); // End timing for cold run
      }
      // delete [] expert_token_counts;
      auto end_time = std::chrono::high_resolution_clock::now(); // End timing
      time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
      auto cold_time = std::chrono::duration_cast<std::chrono::microseconds>(cold_end_time - cold_start_time).count();
      time = time - cold_time - split_time - merge_time; // exclude cold run time
    } else {

      // Warm up
      for(int l = 0; l < num_layer; l++) {
        if(l==0){
          // ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
          ffn_compute_dataflow<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        } else {
          // ffn_compute_blocked<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
          ffn_compute_dataflow<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        }
      }
      
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      for(int i = 0; i < num_iter; i++) {
        for(int l = 0; l < num_layer; l++) {
          if (l==0){
            // ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
            ffn_compute_dataflow<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
          } else {
            // ffn_compute_blocked<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
            ffn_compute_dataflow<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
          }
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

    if (num_expert > 1) {  // MoE FFN
      std::cout << "Number of Experts: " << num_expert << "\n";
      std::cout << "Number of Experts per token: " << num_expert_per_token << "\n";
      std::vector<std::unique_ptr<T[]>> t_split_In(num_expert);
      std::vector<std::unique_ptr<T[]>> t_split_Out(num_expert);
      // std::vector<long> expert_token_counts(num_expert, 0);
      // long* expert_token_counts = new long[num_expert];
      // std::vector<std::vector<long>> token_to_expert(token_len);
      std::vector<std::vector<long>> expert_token_ids(num_expert);
      std::vector<std::vector<long>> token_to_expert(token_len, std::vector<long>(num_expert_per_token, -1));
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      auto cold_start_time = std::chrono::high_resolution_clock::now();
      auto cold_end_time = std::chrono::high_resolution_clock::now();
      for(int i = 0; i < num_iter + 1; i++) {
        if (i == 0)
          cold_start_time = std::chrono::high_resolution_clock::now(); // Start timing for cold run
        for(int l = 0; l < num_layer; l++) {
          // Instead of clearing and resizing, just reset the unique_ptrs
          for (auto& ptr : t_split_In) ptr.reset();
          for (auto& ptr : t_split_Out) ptr.reset();
          // for (long i = 0; i < num_expert; i++) expert_token_counts[i] = 0;
          for (long i = 0; i < num_expert; i++) expert_token_ids[i].clear();
          #pragma omp parallel for
          for (auto& vec : token_to_expert) std::fill(vec.begin(), vec.end(), -1);
          if (l == 0) {
            moe_split_layer_input<T>(t_In[0],
                      t_split_In,
                      t_split_Out,
                      expert_token_ids,
                      token_to_expert,
                      token_len, embedding_dim, num_expert, num_expert_per_token);
          } else {
            moe_split_layer_input<T>(t_Out[l-1],
                        t_split_In,
                        t_split_Out,
                        expert_token_ids,
                        token_to_expert,
                        token_len, embedding_dim, num_expert, num_expert_per_token);
          }

#ifdef USE_TBB
          tbb::parallel_for(0L, num_expert, [&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_ids, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni]
          (long e) {
#else
          for(long e = 0; e < num_expert; e++){
#endif
            if (expert_token_ids[e].size() != 0){   // some tokens for this expert
// #ifdef USE_TBB
//               tbb::task_arena nested; // Limit to 8 threads per expert
//               nested.execute([&ffn_tpp_set, &t_split_Out, &t_split_In, &t_Wg, &t_Wu, &t_Wd, &expert_token_counts, l, num_expert, embedding_dim, intermediate_dim, gate_flag, b_vnni, e]() {
// #endif
                ffn_compute_flat<T>(t_split_Out[e], t_split_In[e], t_Wg[l*num_expert + e], t_Wu[l*num_expert + e], t_Wd[l*num_expert + e], ffn_tpp_set[0], ffn_tpp_set[expert_token_ids[e].size() % MLP_BLOCKSIZE], expert_token_ids[e].size(), embedding_dim, intermediate_dim, gate_flag, b_vnni);
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
        if (i == 0)
          cold_end_time = std::chrono::high_resolution_clock::now(); // End timing for cold run
      }
      auto end_time = std::chrono::high_resolution_clock::now(); // End timing
      time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
      auto cold_time = std::chrono::duration_cast<std::chrono::microseconds>(cold_end_time - cold_start_time).count();
      time -= cold_time; // exclude cold run time
    } else {
      // Warm up
      for(int l = 0; l < num_layer; l++) {
        if (l==0)
          ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        else
          ffn_compute_flat<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      }
      
      auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
      for(int i = 0; i < num_iter; i++) {
        for(int l = 0; l < num_layer; l++) {
          if (l==0)
            ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
          else
            ffn_compute_flat<T>(t_Out[l], t_Out[l-1], t_Wg[l], t_Wu[l], t_Wd[l], ffn_tpp_set[0], ffn_tpp_set[token_len % MLP_BLOCKSIZE], token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
        }
      }

      auto end_time = std::chrono::high_resolution_clock::now(); // End timing
      time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }
  }

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);
  printf("Total Compute Time taken: %0.2f s \n", time / ((double)1e6));
  if (num_expert > 1) {
    average_active_experts = average_active_experts / (num_iter * num_layer);
    std::cout << "Average Active Experts: " << average_active_experts << "\n";
  }
  else
    average_active_experts = 1.0;
  flops_and_bandwidth<T>(time, token_len, embedding_dim, intermediate_dim, num_iter, num_layer, num_expert, num_expert_per_token, gate_flag, average_active_experts);

  return 0;
}

// auto t_In = torch.randn({B_t, S_t, embedding_dim});
// auto t_Wg = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wu = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wd = t_In.randn({4*embedding_dim, embedding_dim});

// auto t_I = i_gemm(SiluPostOp(), t_In, t_Wg, t_null);
// t_I = i_gemm(MulPostOp(t_I), t_In, t_Wu, t_null);
// auto t_Out = o_gemm(AddScalePostOp(t_res, scale), t_I, t_Wd, t_null);