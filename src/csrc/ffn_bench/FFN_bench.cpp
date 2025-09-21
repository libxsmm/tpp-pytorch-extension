#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <tuple>
#include <chrono>
#include <memory>

// #include <ATen/record_function.h>
// #include <torch/csrc/autograd/VariableTypeUtils.h>
// #include <torch/extension.h>

#include "ext_tpp.h"
#include "timing.h"
#include "xsmm_functors.h"
// #include "fused_gemm.h"
// #include "qtypes.h"

using namespace tpp;

#define MLP_BLOCKSIZE 32


template<typename T> using BrgemmTPPtype = decltype((BrgemmTPP<T, T>()));
template<typename T> using SiLUFwdTPPtype = decltype(SiLUFwdTPP<T>());
template<typename T> using Mul2TPPtype = decltype(Mul2TPP<T, T>());
template<typename T> using ScaleTPPtype = decltype(ScaleTPP<T, T>());
template<typename T> using CpyTPPtype = decltype(CpyTPP<T>());
template<typename T>
struct FFNTPPs {
  BrgemmTPPtype<T> i_gemm_tpp;
  SiLUFwdTPPtype<T> silu_tpp;
  Mul2TPPtype<T> mul_tpp;
  BrgemmTPPtype<T> o_gemm_tpp;
  ScaleTPPtype<T> scale_tpp;
  CpyTPPtype<T> copy_tpp;
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
void weight_allocate_and_initialize(std::vector<std::unique_ptr<T[]>>& t_Wg,
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
FFNTPPs<T> create_flat_ffn_tpps(long M_dim, long embedding_dim, long intermediate_dim, bool gate_flag, bool b_vnni, float scale=1.0){
  FFNTPPs<T> tpps;
  tpps.i_gemm_tpp = BrgemmTPP<
            T,
            T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1);
  tpps.silu_tpp = SiLUFwdTPP<T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim);
  tpps.mul_tpp = Mul2TPP<T, T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim);
  tpps.o_gemm_tpp = BrgemmTPP<
            T,
            T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1);
  tpps.scale_tpp = ScaleTPP<T, T>(M_dim * MLP_BLOCKSIZE);
  tpps.copy_tpp = CpyTPP<T>(M_dim, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim);

  return tpps;
}

template<typename T>
long int expert_compute(const std::unique_ptr<T[]>& t_Out, 
                            const std::unique_ptr<T[]>& t_In,
                            const std::unique_ptr<T[]>& t_Wg,
                            const std::unique_ptr<T[]>& t_Wu,
                            const std::unique_ptr<T[]>& t_Wd,
                            FFNTPPs<T>& tpps_main,
                            FFNTPPs<T>& tpps_edge, 
                            long token_len, long embedding_dim, long intermediate_dim, 
                            bool gate_flag, bool blocked, bool b_vnni, float scale=1.0) {


auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

if (blocked){
  std::cout << "waiting for the blocked implementation: \n";
  return 0;
} else {
    // std::cout << "Running with flat layout: \n";
    auto t_In_a = GetVLAPtr<T>(t_In.get(), {embedding_dim});
    auto t_Out_a = GetVLAPtr<T>(t_Out.get(), {embedding_dim});
    auto t_Wg_a = GetVLAPtr<T>(t_Wg.get(), {intermediate_dim});
    auto t_Wu_a = GetVLAPtr<T>(t_Wu.get(), {intermediate_dim});
    auto t_Wd_a = GetVLAPtr<T>(t_Wd.get(), {embedding_dim});

    std::unique_ptr<T[]> t_inter (new (std::align_val_t(64)) T[token_len * intermediate_dim]);
    auto t_inter_a = GetVLAPtr<T>(t_inter.get(), {intermediate_dim});

    // if(b_vnni) {
    //   std::unique_ptr<T[]> ffn_Wg_vnni (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]); 
    //   auto ffn_Wg_vnni_a = GetVLAPtr<T>(ffn_Wg_vnni.get(), {intermediate_dim});

    //   std::unique_ptr<T[]> ffn_Wu_vnni (new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]); 
    //   auto ffn_Wu_vnni_a = GetVLAPtr<T>(ffn_Wu_vnni.get(), {intermediate_dim});

    //   std::unique_ptr<T[]> ffn_Wd_vnni (new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]); 
    //   auto ffn_Wd_vnni_a = GetVLAPtr<T>(ffn_Wd_vnni.get(), {embedding_dim});
    // }

    // auto i_gemm_main_tpp = SCOPEITGEMM(
    //     (BrgemmTPP<
    //         T,
    //         T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1)));

    // auto silu_main_tpp = SCOPEIT(SiLUFwdTPP<T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim), ACT);
    // auto mul_main_tpp = SCOPEIT((Mul2TPP<T, T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim)), EW_MUL);
    // auto o_gemm_main_tpp = SCOPEITGEMM(
    //     (BrgemmTPP<
    //         T,
    //         T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1)));
    // auto scale_main_tpp = SCOPEIT((ScaleTPP<T, T>(MLP_BLOCKSIZE * MLP_BLOCKSIZE)), EW_SCL);
    // auto copy_main_tpp = SCOPEIT((CpyTPP<T>(MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim)), EW_COPY);

    // auto i_gemm_edge_tpp = SCOPEITGEMM(
    //     (BrgemmTPP<
    //         T,
    //         T>((token_len % MLP_BLOCKSIZE), MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*intermediate_dim, embedding_dim, intermediate_dim, MLP_BLOCKSIZE, 0.0, 0, 1)));

    // auto silu_edge_tpp = SCOPEIT(SiLUFwdTPP<T>((token_len % MLP_BLOCKSIZE), MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim), ACT);
    // auto mul_edge_tpp = SCOPEIT((Mul2TPP<T, T>((token_len % MLP_BLOCKSIZE), MLP_BLOCKSIZE, MLP_BLOCKSIZE, intermediate_dim, intermediate_dim)), EW_MUL);
    // auto o_gemm_edge_tpp = SCOPEITGEMM(
    //     (BrgemmTPP<
    //         T,
    //         T>((token_len % MLP_BLOCKSIZE), MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE, MLP_BLOCKSIZE*embedding_dim, intermediate_dim, embedding_dim, MLP_BLOCKSIZE, 0.0, 0, 1)));
    
    // auto scale_edge_tpp = SCOPEIT((ScaleTPP<T, T>((token_len % MLP_BLOCKSIZE) * MLP_BLOCKSIZE)), EW_SCL);
    // auto copy_edge_tpp = SCOPEIT((CpyTPP<T>((token_len % MLP_BLOCKSIZE), MLP_BLOCKSIZE, MLP_BLOCKSIZE, embedding_dim)), EW_COPY);

    {
      // Gate computation
      {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < (token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE; i += MLP_BLOCKSIZE) {
          for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wg_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
            tpps_main.silu_tpp(&tmp[0], &t_inter_a[i][k]);
          }
        }
        if((token_len % MLP_BLOCKSIZE) != 0){   //edge case or decode case
          #pragma omp parallel for
          for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[(token_len % MLP_BLOCKSIZE) * MLP_BLOCKSIZE], 64);
            tpps_edge.i_gemm_tpp(&t_In_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][0], &t_Wg_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
            tpps_edge.silu_tpp(&tmp[0], &t_inter_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][k]);
          }
        }
      }
    }


    {
      // Up computation
      {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < (token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE; i += MLP_BLOCKSIZE) {
          for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.i_gemm_tpp(&t_In_a[i][0], &t_Wu_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
            tpps_main.mul_tpp(&tmp[0], &t_inter_a[i][k], &t_inter_a[i][k]);
          }
        }
        if((token_len % MLP_BLOCKSIZE) != 0){       //edge case or decode case
          #pragma omp parallel for
          for (int k = 0; k < intermediate_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[(token_len % MLP_BLOCKSIZE) * MLP_BLOCKSIZE], 64);
            tpps_edge.i_gemm_tpp(&t_In_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][0], &t_Wg_a[0][k], &tmp[0], embedding_dim/MLP_BLOCKSIZE);
            tpps_edge.mul_tpp(&tmp[0], &t_inter_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][k], &t_inter_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][k]);
          }
        }
      }
    }

    {
      // Down computation
      {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < (token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE; i += MLP_BLOCKSIZE) {
          for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[MLP_BLOCKSIZE * MLP_BLOCKSIZE], 64);
            tpps_main.o_gemm_tpp(&t_inter_a[i][0], &t_Wd_a[0][k], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
            tpps_main.scale_tpp(&tmp[0], &tmp[0], scale);
            tpps_main.copy_tpp(&tmp[0], &t_Out_a[i][k]);
          }
        }
        if((token_len % MLP_BLOCKSIZE) != 0){       //edge case or decode case
          #pragma omp parallel for
          for (int k = 0; k < embedding_dim; k += MLP_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp[(token_len % MLP_BLOCKSIZE) * MLP_BLOCKSIZE], 64);
            tpps_edge.o_gemm_tpp(&t_inter_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][0], &t_Wd_a[0][k], &tmp[0], intermediate_dim/MLP_BLOCKSIZE);
            tpps_edge.scale_tpp(&tmp[0], &tmp[0], scale);
            tpps_edge.copy_tpp(&tmp[0], &t_Out_a[(token_len/MLP_BLOCKSIZE)*MLP_BLOCKSIZE][k]);
          }
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


int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <BF16> <num_layer> <num_iter> <embedding_dim> <intermediate_dim> <gate_flag> <blocked> <b_vnni>" << std::endl;
    return 1;
  }

  long batch_size = std::stoi(argv[1]);
  long seq_len = std::stoi(argv[2]);
  long token_len = batch_size * seq_len;
  long bf16_flag = std::stoi(argv[3]);
  long num_layer = std::stoi(argv[4]);
  long num_iter = std::stoi(argv[5]);
  long embedding_dim = std::stoi(argv[6]);
  long intermediate_dim = std::stoi(argv[7]);
  bool gate_flag = std::stoi(argv[8]);
  bool blocked = std::stoi(argv[9]);
  bool b_vnni = std::stoi(argv[10]);
  
  typedef float T;
  std::vector<std::unique_ptr<T[]>> t_In(num_layer), t_Out(num_layer), t_Wg(num_layer), t_Wu(num_layer), t_Wd(num_layer);
  activation_allocate_and_initialize<T>(t_Out, t_In, token_len, embedding_dim, intermediate_dim, num_layer);
  weight_allocate_and_initialize<T>(t_Wg, t_Wu, t_Wd, embedding_dim, intermediate_dim, num_layer);

  FFNTPPs<T> tpps_main, tpps_edge;
  if((token_len / MLP_BLOCKSIZE) != 0)      // No need to create main tpp if token_len < MLP_BLOCKSIZE
    tpps_main = create_flat_ffn_tpps<T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);
  if ((token_len % MLP_BLOCKSIZE) != 0)     // No need to create edge tpp if token_len is multiple of MLP_BLOCKSIZE
    tpps_edge = create_flat_ffn_tpps<T>(token_len % MLP_BLOCKSIZE, embedding_dim, intermediate_dim, gate_flag, b_vnni);

  long int time = 0;
  for(int i = 0; i < num_iter; i++) {
    time += expert_compute<T>(t_Out[0], t_In[0], t_Wg[0], t_Wu[0], t_Wd[0], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, blocked, b_vnni);
    for(int l = 1; l < num_layer; l++) {
      time += expert_compute<T>(t_Out[l], t_In[l-1], t_Wg[l], t_Wu[l], t_Wd[l], tpps_main, tpps_edge, token_len, embedding_dim, intermediate_dim, gate_flag, blocked, b_vnni);
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