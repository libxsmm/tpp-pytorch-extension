#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <tuple>
#include <chrono>

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

// REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
// REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");
// REGISTER_LOCAL_SCOPE(lnorm, "lnorm");

// auto i_gemm = GemmCaller<T>(SCOPE_ARG(i_gemm));
// auto o_gemm = GemmCaller<T>(SCOPE_ARG(o_gemm));

int B_t = 64;
int S_t = 2048;
int N_t = 64;
int H_t = 128;
int HS_t = N_t*H_t;
int intermediate_dim = 4 * HS_t;


template<typename T>
std::tuple<T**, T**, T**, T**, T**> allocate_and_initialize(long batch_size, long seq_len, long embedding_dim, long intermediate_dim, long num_layer) {
  
  T** t_HS = new T*[num_layer];
  T** t_Out = new T*[num_layer];
  T** t_Wg = new T*[num_layer];
  T** t_Wu = new T*[num_layer];
  T** t_Wd = new T*[num_layer];

  for (int l = 0; l < num_layer; l++) {
    t_HS[l] = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
    t_Out[l] = new (std::align_val_t(64)) T[batch_size * seq_len * embedding_dim];
    t_Wg[l] = new (std::align_val_t(64)) T[embedding_dim * intermediate_dim];
    t_Wu[l] = new (std::align_val_t(64)) T[embedding_dim * intermediate_dim];
    t_Wd[l] = new (std::align_val_t(64)) T[intermediate_dim * embedding_dim];

    for (int i = 0; i < batch_size*seq_len*embedding_dim; ++i) {
      t_HS[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
      t_Out[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
    }

    for (int i = 0; i < embedding_dim * intermediate_dim; ++i) {
      t_Wg[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
      t_Wu[l][i] = static_cast<T>(rand() % 10)*0.1; /// RAND_MAX;
    }
  }

  return std::make_tuple(t_HS, t_Wg, t_Wu, t_Wd, t_Out);
}

template<typename T>
long int declare_and_compute(T* t_HS, T* t_Wg, T* t_Wu, T* t_Wd, T* t_Out, long batch_size, long seq_len, long embedding_dim, long intermediate_dim, bool gate_flag) {

  auto t_HS_a = GetVLAPtr<T>(t_HS, {seq_len, embedding_dim});
  auto t_Out_a = GetVLAPtr<T>(t_Out, {seq_len, embedding_dim});
  auto t_Wg_a = GetVLAPtr<T>(t_Wg, {intermediate_dim});
  auto t_Wu_a = GetVLAPtr<T>(t_Wu, {intermediate_dim});
  auto t_Wd_a = GetVLAPtr<T>(t_Wd, {embedding_dim});

  T* ffn_Wg_vnni = new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]; 
  auto ffn_Wg_vnni_a = GetVLAPtr<T>(ffn_Wg_vnni, {intermediate_dim});

  T* ffn_Wu_vnni = new (std::align_val_t(64)) T[embedding_dim * intermediate_dim]; 
  auto ffn_Wu_vnni_a = GetVLAPtr<T>(ffn_Wu_vnni, {intermediate_dim});

  T* ffn_Wd_vnni = new (std::align_val_t(64)) T[intermediate_dim * embedding_dim]; 
  auto ffn_Wd_vnni_a = GetVLAPtr<T>(ffn_Wd_vnni, {embedding_dim});

  long lda = embedding_dim;
  long ldb = intermediate_dim; 
  long ldc = intermediate_dim;

  auto i_gemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<
          T,
          T>(MLP_BLOCKSIZE, intermediate_dim, embedding_dim, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));

  auto silu_tpp = SCOPEIT(SiLUFwdTPP<T>(MLP_BLOCKSIZE, intermediate_dim), EW_MUL);
  auto gelu_tpp = SCOPEIT(GeluFwdTPP<T>(MLP_BLOCKSIZE, intermediate_dim), EW_MUL);
  auto mul_tpp = SCOPEIT((MulTPP<T, T>(MLP_BLOCKSIZE * intermediate_dim)), EW_MUL);

  lda = intermediate_dim;
  ldb = embedding_dim;
  ldc = embedding_dim;

  auto o_gemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<
          T,
          T>(MLP_BLOCKSIZE, embedding_dim, intermediate_dim, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));

  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  {
    {
  #pragma omp parallel for collapse(2)
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j += MLP_BLOCKSIZE) {
          T* tmp_u = new (std::align_val_t(64)) T[MLP_BLOCKSIZE * intermediate_dim];
          i_gemm_tpp(&t_HS_a[i][j][0], &t_Wu_a[0][0], &tmp_u[0], 1);
          if (gate_flag) {
            T* tmp_g = new (std::align_val_t(64)) T[MLP_BLOCKSIZE * intermediate_dim];
            i_gemm_tpp(&t_HS_a[i][j][0], &t_Wg_a[0][0], &tmp_g[0], 1);
            silu_tpp(&tmp_g[0], &tmp_g[0]);
            mul_tpp(&tmp_g[0], &tmp_u[0], &tmp_u[0]);
            delete [] tmp_g;
          } else{
            gelu_tpp(&tmp_u[0], &tmp_u[0]);
          }
          o_gemm_tpp(&tmp_u[0], &t_Wd_a[0][0], &t_Out_a[i][j][0], 1);
          delete [] tmp_u;
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
  long batch_size,
  long seq_len,
  long embedding_dim,
  long intermediate_dim,
  long num_iter,
  long num_layer,
  bool gate_flag) {

  auto time_ms = time / ((double)1e3 * num_iter * num_layer);

  auto flops = (double)4.0 * batch_size * seq_len * embedding_dim * intermediate_dim;
  if (gate_flag)
    flops += ((double)2.0 * batch_size * seq_len * embedding_dim *intermediate_dim) + (batch_size * seq_len * intermediate_dim);
  flops = flops / ((double)1e9 * time_ms);

  auto bw = ((double)2.0 * batch_size * seq_len * embedding_dim) + ((double)2.0 * embedding_dim * intermediate_dim);
  if (gate_flag)
    bw += (intermediate_dim * embedding_dim);

  bw = (bw * sizeof(T)) / ((double)1e6 * time_ms);

  // print time
  printf("Time taken for ffn_gemm: %0.2f ms, TFLOPS = %0.2f TF/s, Bandwidth = %0.2f GB/s \n", time_ms, flops, bw);
}

template<typename T>
void deallocate(T** t_HS, T** t_Wg, T** t_Wu, T** t_Wd, T** t_Out, long num_layer) {
  for (int l = 0; l < num_layer; l++) {
    delete[] t_HS[l];
    delete[] t_Wg[l];
    delete[] t_Wu[l];
    delete[] t_Wd[l];
    delete[] t_Out[l];
  }
  delete[] t_HS;
  delete[] t_Wg;
  delete[] t_Wu;
  delete[] t_Wd;
  delete[] t_Out;
}

int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len> <BF16> <num_layer> <num_iter> <embedding_dim> <intermediate_dim> <gate_flag>" << std::endl;
    return 1;
  }

  long batch_size = std::stoi(argv[1]);
  long seq_len = std::stoi(argv[2]);
  long bf16_flag = std::stoi(argv[3]);
  long num_layer = std::stoi(argv[4]);
  long num_iter = std::stoi(argv[5]);
  long embedding_dim = std::stoi(argv[6]);
  long intermediate_dim = std::stoi(argv[7]);
  bool gate_flag = std::stoi(argv[8]);
  
  typedef float T;
  auto [t_HS, t_Wg, t_Wu, t_Wd, t_Out] = allocate_and_initialize<T>(batch_size, seq_len, embedding_dim, intermediate_dim, num_layer);

  long int time = 0;
  for(int i = 0; i < num_iter; i++) {
    for(int l = 0; l < num_layer; l++) {
      time += declare_and_compute<T>(t_HS[l], t_Wg[l], t_Wu[l], t_Wd[l], t_Out[l], batch_size, seq_len, embedding_dim, intermediate_dim, gate_flag);
    }
  }

  flops_and_bandwidth<T>(time, batch_size, seq_len, embedding_dim, intermediate_dim, num_iter, num_layer, gate_flag);

  // deallocate memory
  deallocate<T>(t_HS, t_Wg, t_Wu, t_Wd, t_Out, num_layer);

  return 0;
}

// auto t_HS = torch.randn({B_t, S_t, HS_t});

// auto t_Wg = t_HS.randn({HS_t, 4*HS_t});
// auto t_Wu = t_HS.randn({HS_t, 4*HS_t});
// auto t_Wd = t_HS.randn({4*HS_t, HS_t});

// auto t_I = i_gemm(SiluPostOp(), t_HS, t_Wg, t_null);
// t_I = i_gemm(MulPostOp(t_I), t_HS, t_Wu, t_null);
// auto t_Out = o_gemm(AddScalePostOp(t_res, scale), t_I, t_Wd, t_null);