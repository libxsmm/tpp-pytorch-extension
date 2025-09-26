#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <tuple>
#include <chrono>
#include <memory>
#include <cstring>

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


template<typename Tin, typename Tout> using BrgemmTPPtype = decltype((BrgemmTPP<Tin, Tout>()));
template<typename T> using SiLUFwdTPPtype = decltype(SiLUFwdTPP<T>());
template<typename Tin, typename Tout, typename Tin2> using Mul2TPPtype = decltype(Mul2TPP<Tin, Tout, Tin2>());
template<typename T> using ScaleTPPtype = decltype(ScaleTPP<T, T>());
template<typename T> using TransTPPtype = decltype(XformExtTPP<T>());
template<typename Tin, typename Tout> using ConvertTPPtype = decltype(ConvertTPP<Tin, Tout>());
template<typename T>
struct FFNTPPs {
  BrgemmTPPtype<T, float> i_gemm_tpp;
  SiLUFwdTPPtype<float> silu_tpp;
  Mul2TPPtype<float, T, T> mul_tpp;
  BrgemmTPPtype<T, float> o_gemm_tpp;
  ScaleTPPtype<float> scale_tpp;
  ConvertTPPtype<float, T> downconvert_inter_tpp;
  ConvertTPPtype<float, T> downconvert_embed_tpp;
  TransTPPtype<T> i_vnni_tpp;
  TransTPPtype<T> o_vnni_tpp;
};

template<typename T>
void activation_allocate_and_initialize(std::vector<std::unique_ptr<T[]>>& t_Out,
                               std::vector<std::unique_ptr<T[]>>& t_In,
  long token_len, long embedding_dim, long intermediate_dim, long num_layer) {

  auto downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(embedding_dim)), EW_ZERO);

  for (int l = 0; l < num_layer; l++) {
    t_In[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);   // allocate an dynamic array at 64 bit boundary and make unique pointer 
    t_Out[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);

    for (int i = 0; i < token_len; ++i) {
      float t_In_tmp[embedding_dim], t_Out_tmp[embedding_dim];

      #pragma omp parallel
      {
        int j;
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (j = 0; j < embedding_dim; j++){
          t_In_tmp[j] = static_cast<float>(rand_r(&myseed) % 10)*0.1; /// RAND_MAX;
          t_Out_tmp[j] = static_cast<float>(rand_r(&myseed) % 10)*0.1; /// RAND_MAX;
        }
      }
      downconvert_embed_tpp(t_In_tmp, &t_In[l][i*embedding_dim]);
      downconvert_embed_tpp(t_Out_tmp, &t_Out[l][i*embedding_dim]);
    }
  }
}


template<typename T>
void flat_weight_allocate_and_initialize(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
  long embedding_dim, long intermediate_dim, long num_layer) {

  auto downconvert_embed_tpp = SCOPEIT((ConvertTPP<float, T>(embedding_dim)), EW_ZERO);
  auto downconvert_inter_tpp = SCOPEIT((ConvertTPP<float, T>(intermediate_dim)), EW_ZERO);

  for (int l = 0; l < num_layer; l++) {
    t_Wg[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    t_Wu[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    t_Wd[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

    for(int i=0; i < intermediate_dim; i++){
      float t_Wd_tmp[embedding_dim];
      #pragma omp parallel
      {
        int j;
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (j = 0; j < embedding_dim; j++){
          t_Wd_tmp[j] = static_cast<float>(rand_r(&myseed) % 10)*0.1; /// RAND_MAX;
        }
      }
      downconvert_embed_tpp(t_Wd_tmp, &t_Wd[l][i*embedding_dim]);
    }
// when debugging
    
    for (int i = 0; i < embedding_dim; ++i) {
      float t_Wg_tmp[intermediate_dim], t_Wu_tmp[intermediate_dim];
      #pragma omp parallel
      {
        int j;
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (j = 0; j < intermediate_dim; j++){
          t_Wg_tmp[j] = static_cast<float>(rand_r(&myseed) % 10)*0.1; /// RAND_MAX;
          t_Wu_tmp[j] = static_cast<float>(rand_r(&myseed) % 10)*0.1; /// RAND_MAX;
        }
      }
      downconvert_inter_tpp(t_Wg_tmp, &t_Wg[l][i*intermediate_dim]);
      downconvert_inter_tpp(t_Wu_tmp, &t_Wu[l][i*intermediate_dim]);
    }
  }
}

template<typename T>
void flat_to_blocked_weights(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
  long embedding_dim, long intermediate_dim, long num_layer) {

    for(int l = 0; l < num_layer; l++) {
      auto t_Wg_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wu_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
      auto t_Wd_tmp = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

      for(int n1 = 0; n1 < embedding_dim/MLP_BLOCKSIZE; n1++) {
        for(int h1 = 0; h1 < MLP_BLOCKSIZE; h1++) {
          for(int n2 = 0; n2 < intermediate_dim/MLP_BLOCKSIZE; n2++) {
            for(int h2 = 0; h2 < MLP_BLOCKSIZE; h2++) {
              t_Wg_tmp[n2 * (embedding_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]     //[n2, n1, h1, h2]
                = t_Wg[l][n1 * (MLP_BLOCKSIZE * intermediate_dim) + h1 * (intermediate_dim) + n2 * (MLP_BLOCKSIZE) + h2];             //[n1, h1, n2, h2]

              t_Wu_tmp[n2 * (embedding_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]
                = t_Wu[l][n1 * (MLP_BLOCKSIZE * intermediate_dim) + h1 * (intermediate_dim) + n2 * (MLP_BLOCKSIZE) + h2];
            }
          }
        }
      }

      for(int n1 = 0; n1 < intermediate_dim/MLP_BLOCKSIZE; n1++) {
        for(int h1 = 0; h1 < MLP_BLOCKSIZE; h1++) {
          for(int n2 = 0; n2 < embedding_dim/MLP_BLOCKSIZE; n2++) {
            for(int h2 = 0; h2 < MLP_BLOCKSIZE; h2++) {
              t_Wd_tmp[n2 * (intermediate_dim * MLP_BLOCKSIZE) + n1 * (MLP_BLOCKSIZE * MLP_BLOCKSIZE) + h1 * (MLP_BLOCKSIZE) + h2]
                = t_Wd[l][n1 * MLP_BLOCKSIZE * embedding_dim + h1 * (embedding_dim) + n2 * (MLP_BLOCKSIZE) + h2];
            }
          }
        }
      }

      t_Wg[l].swap(t_Wg_tmp);
      t_Wu[l].swap(t_Wu_tmp);
      t_Wd[l].swap(t_Wd_tmp);
    }
}

template<typename T>
FFNTPPs<T> create_flat_ffn_tpps(long M_dim, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni){
  FFNTPPs<T> tpps;
  tpps.i_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  if(b_vnni && std::is_same<T, bfloat16>::value){
    tpps.i_vnni_tpp = XformExtTPP<T>(
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          intermediate_dim,
          intermediate_dim,
          XformTPP::XFORM_N2V_TPP);
  }
  
  tpps.silu_tpp = SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE);
  tpps.downconvert_inter_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.mul_tpp = Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim);
  tpps.o_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  if (b_vnni && std::is_same<T, bfloat16>::value){
    tpps.o_vnni_tpp = XformExtTPP<T>(
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          embedding_dim,
          embedding_dim,
          XformTPP::XFORM_N2V_TPP);
  }
  tpps.scale_tpp = ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE);
  tpps.downconvert_embed_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);

  return tpps;
}


template<typename T>
FFNTPPs<T> create_blocked_ffn_tpps(long M_dim, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni){
  FFNTPPs<T> tpps;
  tpps.i_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, embedding_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  if(b_vnni && std::is_same<T, bfloat16>::value){
    tpps.i_vnni_tpp = XformExtTPP<T>(
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          XformTPP::XFORM_N2V_TPP);
  }
  
  tpps.silu_tpp = SiLUFwdTPP<float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE);
  tpps.downconvert_inter_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.mul_tpp = Mul2TPP<float, T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim);
  tpps.o_gemm_tpp = BrgemmTPP<
            T,
            float>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*MLP_BLOCKSIZE, intermediate_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, 0.0, 0, 1, (b_vnni && std::is_same<T, bfloat16>::value));
  if (b_vnni && std::is_same<T, bfloat16>::value){
    tpps.o_vnni_tpp = XformExtTPP<T>(
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          MLP_BLOCKSIZE,
          XformTPP::XFORM_N2V_TPP);
  }
  tpps.scale_tpp = ScaleTPP<float, float>(M_dim * MLP_BLOCKSIZE);
  tpps.downconvert_embed_tpp = ConvertTPP<float, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);

  return tpps;
}

template<typename T>
void flat_weight_vnni_transform(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
                            long embedding_dim, long intermediate_dim, long num_layer,
                            FFNTPPs<T>& tpps_main) {

  for (int l = 0; l < num_layer; l++) {
    
    auto t_Wg_a = GetVLAPtr<T>(t_Wg[l].get(), {intermediate_dim});
    auto t_Wu_a = GetVLAPtr<T>(t_Wu[l].get(), {intermediate_dim});
    auto t_Wd_a = GetVLAPtr<T>(t_Wd[l].get(), {embedding_dim});

    auto t_Wg_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    auto t_Wu_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    auto t_Wd_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

    auto t_Wg_vnni_a = GetVLAPtr<T>(t_Wg_vnni.get(), {intermediate_dim});
    auto t_Wu_vnni_a = GetVLAPtr<T>(t_Wu_vnni.get(), {intermediate_dim});
    auto t_Wd_vnni_a = GetVLAPtr<T>(t_Wd_vnni.get(), {embedding_dim});

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < embedding_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < intermediate_dim; j += MLP_BLOCKSIZE){
        tpps_main.i_vnni_tpp(&t_Wg_a[i][j], &t_Wg_vnni_a[i][2*j]);
      }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < embedding_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < intermediate_dim; j += MLP_BLOCKSIZE){
        tpps_main.i_vnni_tpp(&t_Wu_a[i][j], &t_Wu_vnni_a[i][2*j]);
      }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < intermediate_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < embedding_dim; j += MLP_BLOCKSIZE){
        tpps_main.o_vnni_tpp(&t_Wd_a[i][j], &t_Wd_vnni_a[i][2*j]);
      }
    }

    // copy t_Wg_vnni contents t_Wg[l]
    t_Wg[l].swap(t_Wg_vnni);
    t_Wu[l].swap(t_Wu_vnni);
    t_Wd[l].swap(t_Wd_vnni);
  }
}


template<typename T>
void blocked_weight_vnni_transform(std::vector<std::unique_ptr<T[]>>& t_Wg,
                             std::vector<std::unique_ptr<T[]>>& t_Wu,
                             std::vector<std::unique_ptr<T[]>>& t_Wd,
                            long embedding_dim, long intermediate_dim, long num_layer,
                            FFNTPPs<T>& tpps_main) {

  for (int l = 0; l < num_layer; l++) {
    
    auto t_Wg_a = GetVLAPtr<T>(t_Wg[l].get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
    auto t_Wu_a = GetVLAPtr<T>(t_Wu[l].get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
    auto t_Wd_a = GetVLAPtr<T>(t_Wd[l].get(), {intermediate_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

    auto t_Wg_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    auto t_Wu_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]);
    auto t_Wd_vnni = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]);

    auto t_Wg_vnni_a = GetVLAPtr<T>(t_Wg_vnni.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
    auto t_Wu_vnni_a = GetVLAPtr<T>(t_Wu_vnni.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
    auto t_Wd_vnni_a = GetVLAPtr<T>(t_Wd_vnni.get(), {intermediate_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < embedding_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < intermediate_dim; j += MLP_BLOCKSIZE){
        tpps_main.i_vnni_tpp(&t_Wg_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0], &t_Wg_vnni_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0]);
      }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < embedding_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < intermediate_dim; j += MLP_BLOCKSIZE){
        tpps_main.i_vnni_tpp(&t_Wu_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0], &t_Wu_vnni_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0]);
      }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < intermediate_dim; i += MLP_BLOCKSIZE){
      for(int j = 0; j < embedding_dim; j += MLP_BLOCKSIZE){
        tpps_main.o_vnni_tpp(&t_Wd_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0], &t_Wd_vnni_a[j/MLP_BLOCKSIZE][i/MLP_BLOCKSIZE][0][0]);
      }
    }

    // copy t_Wg_vnni contents t_Wg[l]
    t_Wg[l].swap(t_Wg_vnni);
    t_Wu[l].swap(t_Wu_vnni);
    t_Wd[l].swap(t_Wd_vnni);
  }
}


template<typename T>
long int ffn_compute_flat(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1) {


  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  auto t_In_a = GetVLAPtr<T>(t_In.get(), {embedding_dim});
  auto t_Out_a = GetVLAPtr<T>(t_Out.get(), {embedding_dim});
  auto t_Wg_a = GetVLAPtr<T>(t_Wg.get(), {intermediate_dim});
  auto t_Wu_a = GetVLAPtr<T>(t_Wu.get(), {intermediate_dim});
  auto t_Wd_a = GetVLAPtr<T>(t_Wd.get(), {embedding_dim});

  std::unique_ptr<T[]> t_inter (new (std::align_val_t(64)) T[token_len * intermediate_dim]);
  auto t_inter_a = GetVLAPtr<T>(t_inter.get(), {intermediate_dim});

  long token_len_q = (token_len / MLP_BLOCKSIZE)*MLP_BLOCKSIZE;
  long token_len_re = (token_len % MLP_BLOCKSIZE);
  {
    // Gate computation
    {

      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wg_a[0][k*2], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          else
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wg_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_main.silu_tpp(&tmp[0], &tmp[0]);
          tpps_main.downconvert_inter_tpp(&tmp[0], &t_inter_a[i][k]);
        }
      }
      if(token_len_re != 0){   //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[0][k*2], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          else
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_edge.silu_tpp(&tmp[0], &tmp[0]);
          tpps_edge.downconvert_inter_tpp(&tmp[0], &t_inter_a[token_len_q][k]);
        }
      }
    }
  }


  if(gate_flag){
    // Up computation
    {
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wu_a[0][k*2], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          else
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wu_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_main.mul_tpp(&tmp[0], &t_inter_a[i][k], &t_inter_a[i][k]);
        }
      }
      if(token_len_re != 0){       //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[0][k*2], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          else
            tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_edge.mul_tpp(&tmp[0], &t_inter_a[token_len_q][k], &t_inter_a[token_len_q][k]);
        }
      }
    }
  }

  {
    // Down computation
    {
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_main.o_gemm_tpp(&t_inter_a[i][0], &t_Wd_a[0][k*2], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          else
            tpps_main.o_gemm_tpp(&t_inter_a[i][0], &t_Wd_a[0][k], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i][k]);
        }
      }
      if(token_len_re != 0){       //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          if(b_vnni && std::is_same<T, bfloat16>::value)
            tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[0][k*2], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          else
            tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[0][k], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][k]);
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // End timing
  auto ffn_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  return ffn_time;
}


template<typename T>
long int ffn_compute_blocked(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool b_vnni, float scale=0.1) {

  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  // std::cout << "Running with flat layout: \n";
  auto t_In_a = GetVLAPtr<T>(t_In.get(), {embedding_dim});
  auto t_Out_a = GetVLAPtr<T>(t_Out.get(), {embedding_dim});

  auto t_Wg_a = GetVLAPtr<T>(t_Wg.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});   // [n2, n1, h1, h2]
  auto t_Wu_a = GetVLAPtr<T>(t_Wu.get(), {embedding_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});
  auto t_Wd_a = GetVLAPtr<T>(t_Wd.get(), {intermediate_dim/MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE});

  std::unique_ptr<T[]> t_inter (new (std::align_val_t(64)) T[token_len * intermediate_dim]);
  auto t_inter_a = GetVLAPtr<T>(t_inter.get(), {intermediate_dim});

  long token_len_q = (token_len / MLP_BLOCKSIZE)*MLP_BLOCKSIZE;
  long token_len_re = (token_len % MLP_BLOCKSIZE);

  {
    // Gate computation
    {

      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wg_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_main.silu_tpp(&tmp[0], &tmp[0]);
          tpps_main.downconvert_inter_tpp(&tmp[0], &t_inter_a[i][k]);
        }
      }
      if(token_len_re != 0){   //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wg_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_edge.silu_tpp(&tmp[0], &tmp[0]);
          tpps_edge.downconvert_inter_tpp(&tmp[0], &t_inter_a[token_len_q][k]);
        }
      }
    }
  }

  if(gate_flag){
    // Up computation
    {
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wu_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_main.mul_tpp(&tmp[0], &t_inter_a[i][k], &t_inter_a[i][k]);
        }
      }
      if(token_len_re != 0){       //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          tpps_edge.i_gemm_tpp(&t_In_a[token_len_q][0], &t_Wu_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
          tpps_edge.mul_tpp(&tmp[0], &t_inter_a[token_len_q][k], &t_inter_a[token_len_q][k]);
        }
      }
    }
  }

  {
    // Down computation
    {
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < token_len_q; i += MLP_BLOCKSIZE) {
        for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
          tpps_main.o_gemm_tpp(&t_inter_a[i][0], &t_Wd_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_main.downconvert_embed_tpp(&tmp[0], &t_Out_a[i][k]);
        }
      }
      if(token_len_re != 0){       //edge case or decode case
        #pragma omp parallel for
        for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[token_len_re * MLP_BLOCKSIZE], 64);
          tpps_edge.o_gemm_tpp(&t_inter_a[token_len_q][0], &t_Wd_a[k/MLP_BLOCKSIZE][0][0][0], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
          tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
          tpps_edge.downconvert_embed_tpp(&tmp[0], &t_Out_a[token_len_q][k]);
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // End timing
  auto ffn_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  return ffn_time;

}

template<typename T>
void flops_and_bandwidth(
  long int time,
  long token_len,
  long embedding_dim,
  long intermediate_dim,
  long num_iter,
  long num_layer,
  bool gate_flag) {

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);

  auto flops = (double)4.0 * token_len * embedding_dim * intermediate_dim;
  if (gate_flag)
    flops += ((double)2.0 * token_len * embedding_dim *intermediate_dim) + (token_len * intermediate_dim);
  flops = flops / ((double)1e9 * time_ms);

  auto bw = ((double)2.0 * token_len * embedding_dim) + ((double)2.0 * embedding_dim * intermediate_dim);
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
                            bool gate_flag, bool b_vnni, float scale=0.1, long num_layer=1) {

  std::cout<< "Starting Correctness Check" << "\n";                       
  FFNTPPs<T> tpps_main_flat, tpps_edge_flat;
  FFNTPPs<T> tpps_main_blocked, tpps_edge_blocked;

  std::vector<std::unique_ptr<T[]>> t2_Out(num_layer), t2_Wg(num_layer), t2_Wu(num_layer), t2_Wd(num_layer);
  for(int l = 0; l < num_layer; l++){
    t2_Out[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[token_len * embedding_dim]);
    t2_Wg[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim*intermediate_dim]);
    t2_Wu[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[embedding_dim*intermediate_dim]);
    t2_Wd[l] = std::unique_ptr<T[]> (new (std::align_val_t(64)) T[intermediate_dim*embedding_dim]);

    std::memcpy(t2_Wg[l].get(), t_Wg[l].get(), sizeof(T)*embedding_dim*intermediate_dim);
    std::memcpy(t2_Wu[l].get(), t_Wu[l].get(), sizeof(T)*embedding_dim*intermediate_dim);
    std::memcpy(t2_Wd[l].get(), t_Wd[l].get(), sizeof(T)*intermediate_dim*embedding_dim);
  }
  flat_to_blocked_weights<T>(t2_Wg, t2_Wu, t2_Wd, embedding_dim, intermediate_dim, num_layer);

  if((token_len / MLP_BLOCKSIZE) != 0){      // No need to create main tpp if token_len < MLP_BLOCKSIZE
    tpps_main_flat = create_flat_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    tpps_main_blocked = create_blocked_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  }
  if ((token_len % MLP_BLOCKSIZE) != 0){     // No need to create edge tpp if token_len is multiple of MLP_BLOCKSIZE
    tpps_edge_flat = create_flat_ffn_tpps<T>(token_len % MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    tpps_edge_blocked = create_blocked_ffn_tpps<T>(token_len % MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  }

  if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
    flat_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, tpps_main_flat);
    blocked_weight_vnni_transform<T>(t2_Wg, t2_Wu, t2_Wd, embedding_dim, intermediate_dim, num_layer, tpps_main_blocked);
  }

  ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main_flat, tpps_edge_flat, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);
  ffn_compute_blocked<T>(t2_Out[0], t_In[0], t2_Wg[0], t2_Wu[0], t2_Wd[0], tpps_main_blocked, tpps_edge_blocked, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni, scale);

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
        if(std::abs(t_Out_float[j] - t2_Out_float[j]) > tol){
          std::cout << "Layer 0: Mismatch at index " << i << ": " << (float)t_Out[0][i] << " vs " << (float)t2_Out[0][i] << std::endl;
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

  long batch_size = std::stoi(argv[1]);
  long seq_len = std::stoi(argv[2]);
  long token_len = batch_size * seq_len;
  bool b_vnni = std::stoi(argv[3]);
  bool blocked = std::stoi(argv[4]);
  long num_layer = std::stoi(argv[5]);
  long num_iter = std::stoi(argv[6]);
  long embedding_dim = std::stoi(argv[7]);
  long intermediate_dim = std::stoi(argv[8]);
  bool gate_flag = std::stoi(argv[9]);
  bool correctness_check = std::stoi(argv[10]);

  std::vector<std::unique_ptr<T[]>> t_In(num_layer), t_Out(num_layer), t_Wg(num_layer), t_Wu(num_layer), t_Wd(num_layer);
  activation_allocate_and_initialize<T>(t_Out, t_In, token_len, embedding_dim, intermediate_dim, num_layer);
  flat_weight_allocate_and_initialize<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer);

  if(correctness_check){
    correctness_checking(t_Out, t_In, t_Wg, t_Wu, t_Wd, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  }

  // Timing
  long int time = 0;
  if(blocked){
    std::cout << "Running with blocked weight layout \n";
    FFNTPPs<T> tpps_main, tpps_edge;
    flat_to_blocked_weights<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer);

    if((token_len / MLP_BLOCKSIZE) != 0)      // No need to create main tpp if token_len < MLP_BLOCKSIZE
      tpps_main = create_blocked_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    if ((token_len % MLP_BLOCKSIZE) != 0)     // No need to create edge tpp if token_len is multiple of MLP_BLOCKSIZE
      tpps_edge = create_blocked_ffn_tpps<T>(token_len % MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);

    if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
      blocked_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, tpps_main);
    }

    // Warm up
    ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    for(int l = 1; l < num_layer; l++) {
      ffn_compute_blocked<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    }
    
    for(int i = 0; i < num_iter; i++) {
      time += ffn_compute_blocked<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      for(int l = 1; l < num_layer; l++) {
        time += ffn_compute_blocked<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      }
    }

  } else {
    std::cout << "Running with flat weight layout \n";
    FFNTPPs<T> tpps_main, tpps_edge;
    if((token_len / MLP_BLOCKSIZE) != 0)      // No need to create main tpp if token_len < MLP_BLOCKSIZE
      tpps_main = create_flat_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    if ((token_len % MLP_BLOCKSIZE) != 0)     // No need to create edge tpp if token_len is multiple of MLP_BLOCKSIZE
      tpps_edge = create_flat_ffn_tpps<T>(token_len % MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);

    if(b_vnni && std::is_same<T, bfloat16>::value) {        // take VNNI transform when b_vnni is True
      flat_weight_vnni_transform<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer, tpps_main);
    }

    // Warm up
    ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    for(int l = 1; l < num_layer; l++) {
      ffn_compute_flat<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
    }
    
    for(int i = 0; i < num_iter; i++) {
      time += ffn_compute_flat<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      for(int l = 1; l < num_layer; l++) {
        time += ffn_compute_flat<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, b_vnni);
      }
    }
  }

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);
  printf("Time taken for ffn_gemm: %0.2f ms \n", time_ms);
  flops_and_bandwidth<T>(time, token_len, embedding_dim, intermediate_dim, num_iter, num_layer, gate_flag);

  return 0;
}

// auto t_In = torch.randn({B_t, S_t, embedding_dim});
// auto t_Wg = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wu = t_In.randn({embedding_dim, 4*embedding_dim});
// auto t_Wd = t_In.randn({4*embedding_dim, embedding_dim});

// auto t_I = i_gemm(SiluPostOp(), t_In, t_Wg, t_null);
// t_I = i_gemm(MulPostOp(t_I), t_In, t_Wu, t_null);
// auto t_Out = o_gemm(AddScalePostOp(t_res, scale), t_I, t_Wd, t_null);