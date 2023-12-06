/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Evangelos Georganas, Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
// #include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include <torch/csrc/distributed/c10d/comm.hpp>
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

static int sparse_type = env2int("SPARSE_KERNELS", 0);
static int measure_sparsity = env2int("MEASURE_SPARSITY", 0);
// #define S_FIRST_KVC

static long get_batch_dim_in_kv_cache() {
#ifdef S_FIRST_KVC
  return 1;
#else
  return 0;
#endif
}

static int my_rank = guess_mpi_rank();
static int my_size = 1;
static int large_cache_opt = false;
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static int SK_BLOCK_SIZE = env2int("SK_BLOCK_SIZE", 64);
static int KV_CACHE_INC_SIZE = env2int("KV_CACHE_INC_SIZE", 128);
static int SPMM_PACKED_BLOCK_SIZE = env2int("SPMM_PACKED_BLOCK_SIZE", 64);
int spmm_use_flat_acts = env2int("SPMM_FLAT_ACTIVATIONS", 0);
int layer = 0;

static const char* GEMM_LOOP_SCHEME =
    getenv("GEMM_LOOP_SCHEME") ? getenv("GEMM_LOOP_SCHEME") : "aCB";

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(pln_gemm, "pln_gemm");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(qkv_gemm_sparse, "qkv_spmm");
REGISTER_LOCAL_SCOPE(qkv_gemm_compressed, "qkv_gemm_bm");
REGISTER_LOCAL_SCOPE(mha, "mha");
REGISTER_LOCAL_SCOPE(ac_gemm1, "ac_gemm1");
REGISTER_LOCAL_SCOPE(ac_gemm2, "ac_gemm2");
REGISTER_LOCAL_SCOPE(ac_gemm21, "ac_gemm21");
REGISTER_LOCAL_SCOPE(ac_gemm22, "ac_gemm22");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(o_spmm, "o_spmm");
REGISTER_LOCAL_SCOPE(o_gemm_compressed, "o_gemm_bm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");
REGISTER_LOCAL_SCOPE(i_spmm, "i_spmm");
REGISTER_LOCAL_SCOPE(i_gemm_compressed, "i_gemm_bm");
REGISTER_LOCAL_SCOPE(lnorm, "lnorm");
REGISTER_LOCAL_SCOPE(rotary, "rotary");
REGISTER_LOCAL_SCOPE(reorder, "rorder");
REGISTER_LOCAL_SCOPE(allred, "allred");
REGISTER_LOCAL_SCOPE(barrier, "barrier");
REGISTER_LOCAL_SCOPE(concat, "concat");
REGISTER_LOCAL_SCOPE(fftkn, "fftkn");
REGISTER_LOCAL_SCOPE(k_trans, "k_trans");
REGISTER_LOCAL_SCOPE(pt_op, "pt_op");

template <typename T>
void trans_output_block(long S, long H, T* input, T* output) {
  auto xform = XformTPP::XFORM_XPOSE_TPP;
  if (sizeof(T) != 4) {
    xform = XformTPP::XFORM_XPOSE_N2V_TPP;
  }
  auto xform_tpp = SCOPEIT(
      XformExtTPP<T>(H, S, xform, true), (sizeof(T) == 4) ? XPOSE : VNNI);
  xform_tpp(input, output);
}

double measure_sparsity_in_blocked_nkkn_weight(at::Tensor& t_W) {
  double res = 0.0;
  if (t_W.dtype() == at::kFloat) {
    res = measure_sparsity_from_blocked_weight_tensor(
        GetVLAPtr<float>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()));
  } else {
    res = measure_sparsity_from_blocked_weight_tensor(
        GetVLAPtr<bfloat16>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()));
  }
  return res;
}

void create_bcsc_from_blocked_nkkn_weight(
    at::Tensor& t_W,
    tensor_bcsc_t* t_W_bcsc,
    int bcsc_bk,
    int bcsc_bn) {
  if (t_W.dtype() == at::kFloat) {
    create_bcsc_from_blocked_weight_tensor(
        GetVLAPtr<float>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()),
        bcsc_bk,
        bcsc_bn,
        omp_get_max_threads(),
        t_W_bcsc);
  } else {
    create_bcsc_from_blocked_weight_tensor(
        GetVLAPtr<bfloat16>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()),
        bcsc_bk,
        bcsc_bn,
        omp_get_max_threads(),
        t_W_bcsc);
  }
}

void create_compressed_from_blocked_nkkn_weight(
    at::Tensor& t_W,
    tensor_compressed_t* t_W_compressed) {
  if (t_W.dtype() == at::kFloat) {
    create_compressed_from_blocked_weight_tensor(
        GetVLAPtr<float>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()),
        t_W_compressed);
  } else {
    create_compressed_from_blocked_weight_tensor(
        GetVLAPtr<bfloat16>(t_W),
        t_W.sizes()[0],
        t_W.sizes()[1],
        t_W.sizes()[2],
        t_W.sizes()[3],
        get_vnni_block_size(t_W.dtype()),
        t_W_compressed);
  }
}
static c10::intrusive_ptr<c10d::ProcessGroup> process_group;

void set_pg(c10::intrusive_ptr<c10d::ProcessGroup> process_group_) {
  process_group = process_group_;
  my_size = process_group->getSize();
  my_rank = process_group->getRank();
  printf("Setting PG: my_size = %d  my_rank = %d\n", my_size, my_rank);
}

inline void allreduce(at::Tensor t_in) {
  RECORD_SCOPE(allred, {t_in});
  if (!process_group) {
    printf("Missing process group when using model parallel, use set_pg()\n");
    exit(1);
  }
#if 0
  {
    RECORD_SCOPE(barrier, {});
    process_group->barrier()->wait();
  }
#endif
  std::vector<at::Tensor> temp_vec = {t_in};
  process_group->allreduce(temp_vec)->wait();
}

inline at::Tensor allgather(at::Tensor t_in, std::vector<long>& split_sizes) {
  RECORD_SCOPE(allred, {t_in});
  if (!process_group) {
    printf("Missing process group when using model parallel, use set_pg()\n");
    exit(1);
  }
#if 0
  {
    RECORD_SCOPE(barrier, {});
    process_group->barrier()->wait();
  }
#endif
  std::vector<std::vector<at::Tensor>> ag_vec(1);
  auto sz = t_in.sizes().vec();
  auto dim = t_in.dim() - 1;
  TPP_ASSERT(
      split_sizes.size() == my_size,
      "Length of split vector doesn't match group size");
  for (int i = 0; i < my_size; i++) {
    c10::InferenceMode guard(false);
    sz[dim] = split_sizes[i];
    ag_vec[0].push_back(t_in.new_empty(sz));
  }
  std::vector<at::Tensor> temp_vec = {t_in};
  process_group->allgather(ag_vec, temp_vec)->wait();
  auto t_out = at::cat(ag_vec[0], -1);
  return t_out;
}

template <typename T>
inline at::Tensor kv_concat(
    at::Tensor t_in1,
    at::Tensor t_in2,
    int dim,
    at::Tensor t_beam_idx) {
  RECORD_SCOPE(concat, {t_in1, t_in2});
  bool indirect = t_beam_idx.numel() > 0;
  auto ndim = t_in1.dim();
  dim = dim >= 0 ? dim : dim + ndim;

  auto out_sizes = t_in1.sizes().vec();
  out_sizes[dim] += t_in2.size(dim);
  if (indirect)
    out_sizes[0] = t_beam_idx.size(0);
  auto t_out = t_in1.new_empty(out_sizes);

  auto B = out_sizes[0];
  auto N = out_sizes[1];
  auto S = out_sizes[2];
  auto F = out_sizes[3];
  TPP_ASSERT(B == t_in2.size(0), "Batch size mismatch\n");
  auto BNS = B * N * S;
  auto S1 = t_in1.size(dim);
  auto S2 = t_in2.size(dim);

  // auto cpy_tpp = SCOPEIT(CpyTPP<T>(F), EW_COPY);
  auto cpy_tpp = CpyTPP<T>(F);

  auto in1 = GetVLAPtr<T>(t_in1, {N, S1, F});
  auto in2 = GetVLAPtr<T>(t_in2, {N, S2, F});
  auto out = GetVLAPtr<T>(t_out, {F});
  auto beam_idx = GetVLAPtr<long>(t_beam_idx);
  // std::cout << "t_beam_idx.dtype: " << t_beam_idx.dtype() << std::endl;
  // auto beam_idx = (long*)t_beam_idx.data_ptr();
  T* ptrs[BNS];
  int p = 0;
  for (int j = 0; j < B; j++) {
    int j1 = indirect ? beam_idx[j] : j;
    for (int k = 0; k < N; k++) {
      for (int i = 0; i < S1; i++) {
        ptrs[p++] = in1[j1][k][i];
      }
      for (int i = 0; i < S2; i++) {
        ptrs[p++] = in2[j][k][i];
      }
    }
  }
  TPP_ASSERT(p == BNS, "Unmatched p=%d and BNS=%ld\n", p, BNS);
  {
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int i = 0; i < BNS; i++) {
      cpy_tpp(ptrs[i], out[i]);
    }
  }
  return t_out;
}

template <typename T>
inline void apply_rotary_pos_emb_gptj(
    at::Tensor t_in,
    at::Tensor t_emb_pos,
    at::Tensor t_pos,
    long N,
    long H) {
  RECORD_SCOPE(rotary, {t_in, t_emb_pos, t_pos});
  auto in_sizes = t_in.sizes(); // in[B][S][F]
  auto MP = t_emb_pos.size(0); // Max Pos
  auto HR = t_emb_pos.size(1); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto COFF = HR / 2;

  auto in = GetVLAPtr<T>(t_in, {S, N, H}); // [B][S][N][H]
  auto emb_pos = GetVLAPtr<float>(t_emb_pos, {HR}); // [MP][HR]
  auto pos = GetVLAPtr<long>(t_pos, {S}); // [MB][S]
  // printf("MP=%ld HR=%ld B=%ld S=%ld N=%ld H=%ld\n", MP, HR, B, S, N, H);
  // std::cout << "pos: " << t_pos.sizes() << std::endl;
  // std::cout << "emb_pos: " << t_emb_pos.sizes() << std::endl;

  {
    RECORD_OMP_TIME();

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
            float in0 = in[b][s][n][h];
            float in1 = in[b][s][n][h + 1];
            int p = pos[b][s];
            if (p >= MP)
              continue;
            // TPP_ASSERT(p < MP, "Invalid idx: %d (max %ld)\n", p, MP);
            float sin = emb_pos[p][h2];
            float cos = emb_pos[p][COFF + h2];
            float out0 = in0 * cos - in1 * sin;
            float out1 = in1 * cos + in0 * sin;
            in[b][s][n][h] = out0;
            in[b][s][n][h + 1] = out1;
            // if (b == 1 && s < 2 && n == 0 && h < 8) {
            //   printf("%d %d %d %d  %d: %g %g    %g %g\n", b, s, n, h, p, in0,
            //   out0, in1, out1);
            // }
          }
        }
      }
    }
  }
}

template <typename T>
inline void apply_rotary_pos_emb_llama(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    long N,
    long H) {
  RECORD_SCOPE(rotary, {t_in, t_emb_pos, t_pos});
  auto in_sizes = t_in.sizes(); // in[B][S][F]
  auto MP = t_emb_pos.size(1); // Max Pos
  auto HR = t_emb_pos.size(2); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto COFF = HR / 2;

  auto in = GetVLAPtr<T>(t_in, {S, N, H}); // [B][S][N][H]
  auto emb_pos = GetVLAPtr<float>(t_emb_pos, {MP, HR}); // [MP][HR]
  auto pos = GetVLAPtr<long>(t_pos, {S}); // [MB][S]
  // printf("MP=%ld HR=%ld B=%ld S=%ld N=%ld H=%ld\n", MP, HR, B, S, N, H);

  {
    RECORD_OMP_TIME();

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          for (int h2 = 0; h2 < HR / 2; h2++) {
            float in0 = in[b][s][n][h2];
            float in1 = in[b][s][n][COFF + h2];
            int p = pos[b][s];
            float cos = emb_pos[0][p][h2];
            float sin = emb_pos[1][p][h2];
            float out0 = in0 * cos - in1 * sin;
            float out1 = in1 * cos + in0 * sin;
            in[b][s][n][h2] = out0;
            in[b][s][n][COFF + h2] = out1;
          }
        }
      }
    }
  }
}

template <typename T, typename LT = T>
inline void lyr_norm(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_beta,
    at::Tensor t_out,
    float eps) {
  RECORD_SCOPE(lnorm, {t_in, t_gamma, t_beta});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto K = in_sizes[2];

  auto in = GetVLAPtr<T>(t_in, {K});
  auto gamma = GetVLAPtr<LT>(t_gamma);
  auto beta = GetVLAPtr<LT>(t_beta);
  auto out = GetVLAPtr<T>(t_out, {K});

  auto layer_norm_fwd_tpp =
      SCOPEIT((LayerNormFwdTPP<T, LT>(1, 1, K, eps)), LAYER_NORM);

  {
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int b = 0; b < BS; b++) {
      layer_norm_fwd_tpp(in[b], gamma, beta, nullptr, nullptr, out[b]);
    }
  }
}

template <typename T, typename LT = T>
inline at::Tensor lyr_norm(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_beta,
    float eps) {
  auto t_out = at::empty_like(t_in);
  lyr_norm<T, LT>(t_in, t_gamma, t_beta, t_out, eps);
  return t_out;
}

template <typename T, typename LT = T>
inline void rms_norm(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_out,
    float eps) {
  RECORD_SCOPE(lnorm, {t_in, t_gamma});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto K = in_sizes[2];

  auto in = GetVLAPtr<T>(t_in, {K});
  auto gamma = GetVLAPtr<LT>(t_gamma);
  auto out = GetVLAPtr<T>(t_out, {K});

  auto rms_norm_fwd_tpp =
      SCOPEIT((RMSNormFwdTPP<T, LT>(1, 1, K, eps)), LAYER_NORM);

  {
    RECORD_OMP_TIME();

#pragma omp parallel for
    for (int b = 0; b < BS; b++) {
      rms_norm_fwd_tpp(in[b], gamma, nullptr, out[b]);
    }
  }
}

template <typename T, typename LT = T>
inline at::Tensor llama_rms_norm(at::Tensor t_in, at::Tensor t_wt, float eps) {
  RECORD_SCOPE(lnorm, {t_in, t_wt});

  // auto orig_dt = t_in.dtype();
  // auto t_var = t_in.to(at::kFloat).pow(2).mean(-1, true);
  // auto t_tmp = t_in * at::rsqrt(t_var + eps);
  // auto ret = t_wt * t_tmp;
  // ret = ret.to(orig_dt);
  // return ret;

  auto t_out = at::empty_like(t_in);
  rms_norm<T, LT>(t_in, t_wt, t_out, eps);
  return t_out;
}

template <typename T>
inline void fc_plain(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(pln_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  t_in = t_in.contiguous();
  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_OMP_TIME();
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_plain(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_plain<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline at::Tensor wt_tensor_for_first_token(at::Tensor t) {
  RECORD_SCOPE(fftkn, {t});
  auto dim = t.dim();
  if (dim < 5)
    return t;
  auto sizes = t.sizes();
  constexpr long RBS = 4;
  auto K1 = sizes[0];
  if (K1 % RBS != 0)
    return t;
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = sizes[4];
#if 0
  auto t_new = t.view({K1/RBS, RBS, C1, C2, K2, C3}).permute({0, 2, 3, 1, 4, 5}).contiguous().view({K1/RBS, C1, C2, RBS*K2, C3});
#else
  auto t_new = t.new_empty({K1 / RBS, C1, C2, RBS * K2, C3});
  auto in = GetVLAPtr<T>(t, {RBS, C1, C2, K2 * C3});
  auto out = GetVLAPtr<T>(t_new, {C1, C2, RBS, K2 * C3});

#if 1
  auto cpy_tpp =
      SCOPEIT(CpyTPP<T>(C2, K2 * C3, K2 * C3, RBS * K2 * C3), EW_COPY);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < K1 / RBS; i++) {
    for (int j = 0; j < C1; j++) {
      for (int k = 0; k < RBS; k++) {
        cpy_tpp(in[i][k][j][0], out[i][j][0][k]);
      }
    }
  }
#else
  auto cpy_tpp =
      SCOPEIT(CpyTPP<T>(RBS, K2 * C3, C1 * C2 * K2 * C3, K2 * C3), EW_COPY);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < K1 / RBS; i++) {
    for (int j = 0; j < C1; j++) {
      for (int k = 0; k < C2; k++) {
        cpy_tpp(in[i][0][j][k], out[i][j][k][0]);
      }
    }
  }
#endif

#endif
  return t_new;
}

template <typename T>
inline void fc_mul(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(o_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto mul_tpp = SCOPEIT((MulTPP<T, T>(BSb, Hk, K, K)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<T, T>(rem, Hk, K, K)), EW_MUL);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp_rem(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_mul(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto t_out = at::empty_like(t_in1);
  fc_mul<T>(t_in, t_in1, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_mul_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(o_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);

  auto gemm_tpp = SCOPEITGEMM((GemmTPP<T, T>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, T>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  auto mul_tpp = SCOPEIT((MulTPP<T, T>(BSb, Hk, K, K)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<T, T>(rem, Hk, K, K)), EW_MUL);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp_rem(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_mul_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias) {
  auto t_out = at::empty_like(t_in1);
  fc_mul_compressed<T>(t_in, t_in1, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_add_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  RECORD_SCOPE(o_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp(in1[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp_rem(in1[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_add_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    float scale) {
  auto t_out = at::empty_like(t_in1);
  fc_add_scale<T>(t_in, t_in1, t_wt, t_bias, t_out, scale);
  return t_out;
}

template <typename T>
inline void fc_add_scale_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  RECORD_SCOPE(o_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);

  auto gemm_tpp = SCOPEITGEMM((GemmTPP<T, T>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, T>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp(in1[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp_rem(in1[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_add_scale_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    float scale) {
  auto t_out = at::empty_like(t_in1);
  fc_add_scale_compressed<T>(t_in, t_in1, t_wt, t_bias, t_out, scale);
  return t_out;
}

template <typename T>
inline void fc_add2_scale_sparse(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  RECORD_SCOPE(o_spmm, {t_in});
  auto in_sizes = t_in.sizes();
  auto out_sizes = t_out.sizes();
  auto BS = out_sizes[0] * out_sizes[1];
  auto C = (spmm_use_flat_acts > 0) ? in_sizes[2]
                                    : in_sizes[1] * in_sizes[2] * in_sizes[4];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;
  auto BSb = (spmm_use_flat_acts > 0) ? LIBXSMM_MIN(BS, SPMM_PACKED_BLOCK_SIZE)
                                      : in_sizes[3];
  auto BS_blocks =
      (spmm_use_flat_acts > 0) ? (BS + BSb - 1) / BSb : in_sizes[0];
  auto in = GetVLAPtr<T>(t_in, {BSb * C});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});
  auto rem = (BS % BSb == 0) ? BSb : (BS % BSb);
  bool with_bias = (t_bias.numel() > 0);

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);

  auto copy_bias_row_tpp = SCOPEIT(CpyBiasRowTPP<T>(Hk, BSb, BSb), BIAS);
  auto copy_bias_row_tpp_rem = SCOPEIT(CpyBiasRowTPP<T>(Hk, rem, BSb), BIAS);

  auto zero_tpp = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);
  auto zero_tpp_rem = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);

  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);

  auto trans_out_tpp = SCOPEIT(
      XformExtTPP<T>(Hk, BSb, BSb, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);
  auto trans_out_tpp_rem = SCOPEIT(
      XformExtTPP<T>(Hk, rem, rem, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);

  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
      BSb,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));

  auto spmm_tpp_rem = SCOPEITGEMM((SpmmTPP<T, T>(
      rem,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));

  {
    RECORD_OMP_TIME();
    auto loop_scheme = large_cache_opt ? "BA" : "Ba";
    auto ogemm_loop = ThreadedLoop<2>(
        {LoopSpecs{BS_blocks}, LoopSpecs{t_wt.n_blocks}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int s1 = ind[0], i_n = ind[1];
          T tmp_block[BSb * Hk];
          int cur_n_blocks =
              (t_wt.Nblocks_offsets[i_n + 1] - t_wt.Nblocks_offsets[i_n]) /
              t_wt.bcsc_bn;
          for (int l_block = 0; l_block < cur_n_blocks;
               l_block += t_wt.bcsc_blocks_in_bn) {
            int nk = (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) / Hk;
            T* dst = (spmm_use_flat_acts > 0) ? out[s1 * BSb][nk] : tmp_block;
            if (with_bias) {
              if (spmm_use_flat_acts > 0) {
                if (s1 == BS_blocks - 1) {
                  copy_bias_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_tpp(bias[nk], dst);
                }
              } else {
                if (s1 == BS_blocks - 1) {
                  copy_bias_row_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_row_tpp(bias[nk], dst);
                }
              }
            } else {
              if (s1 == BS_blocks - 1) {
                zero_tpp_rem(dst);
              } else {
                zero_tpp(dst);
              }
            }
            if (spmm_use_flat_acts == 0 || rem == BSb || s1 < BS_blocks - 1) {
              spmm_tpp(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  true);
            } else {
              spmm_tpp_rem(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  false);
              spmm_tpp.config();
            }
            if (s1 == BS_blocks - 1) {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp_rem(dst, out[s1 * BSb][nk]);
              add_tpp_rem(
                  out[s1 * BSb][nk], in1[s1 * BSb][nk], out[s1 * BSb][nk]);
              sadd_tpp_rem(in2[s1 * BSb][nk], out[s1 * BSb][nk], scale);
            } else {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp(dst, out[s1 * BSb][nk]);
              add_tpp(out[s1 * BSb][nk], in1[s1 * BSb][nk], out[s1 * BSb][nk]);
              sadd_tpp(in2[s1 * BSb][nk], out[s1 * BSb][nk], scale);
            }
          }
        },
        [&]() {
          TimerStart();
          spmm_tpp.config();
        },
        [&]() {
          spmm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_add2_scale_sparse(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias,
    float scale) {
  auto t_out = at::empty_like(t_in1);
  fc_add2_scale_sparse<T>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  return t_out;
}

template <typename T>
inline void fc_add2_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  RECORD_SCOPE(o_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp(in2[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp_rem(in2[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_add2_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    at::Tensor t_wt,
    at::Tensor t_bias,
    float scale) {
  auto t_out = at::empty_like(t_in1);
  fc_add2_scale<T>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  return t_out;
}

template <typename T>
inline void fc_add2_scale_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  RECORD_SCOPE(o_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);

  auto gemm_tpp = SCOPEITGEMM((GemmTPP<T, T>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, T>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp(in2[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp_rem(in2[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_add2_scale_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    float scale) {
  auto t_out = at::empty_like(t_in1);
  fc_add2_scale_compressed<T>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  return t_out;
}

template <typename T>
inline void fc_gelu_sparse(
    at::Tensor t_in,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_spmm, {t_in});
  auto in_sizes = t_in.sizes();
  auto out_sizes = t_out.sizes();
  auto BS = out_sizes[0] * out_sizes[1];
  auto C = (spmm_use_flat_acts > 0) ? in_sizes[2]
                                    : in_sizes[1] * in_sizes[2] * in_sizes[4];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;
  auto BSb = (spmm_use_flat_acts > 0) ? LIBXSMM_MIN(BS, SPMM_PACKED_BLOCK_SIZE)
                                      : in_sizes[3];
  auto BS_blocks =
      (spmm_use_flat_acts > 0) ? (BS + BSb - 1) / BSb : in_sizes[0];
  auto in = GetVLAPtr<T>(t_in, {BSb * C});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto rem = (BS % BSb == 0) ? BSb : (BS % BSb);
  bool with_bias = (t_bias.numel() > 0);

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);

  auto copy_bias_row_tpp = SCOPEIT(CpyBiasRowTPP<T>(Hk, BSb, BSb), BIAS);
  auto copy_bias_row_tpp_rem = SCOPEIT(CpyBiasRowTPP<T>(Hk, rem, BSb), BIAS);

  auto zero_tpp = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);
  auto zero_tpp_rem = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);

  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);

  auto trans_out_tpp = SCOPEIT(
      XformExtTPP<T>(Hk, BSb, BSb, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);
  auto trans_out_tpp_rem = SCOPEIT(
      XformExtTPP<T>(Hk, rem, rem, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);

  auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
      BSb,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));

  auto spmm_tpp_rem = SCOPEITGEMM((SpmmTPP<T, T>(
      rem,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));
  {
    RECORD_OMP_TIME();
    auto loop_scheme = large_cache_opt ? "BA" : "Ba";
    auto igemm_loop = ThreadedLoop<2>(
        {LoopSpecs{BS_blocks}, LoopSpecs{t_wt.n_blocks}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int s1 = ind[0], i_n = ind[1];
          T tmp_block[BSb * Hk];
          int cur_n_blocks =
              (t_wt.Nblocks_offsets[i_n + 1] - t_wt.Nblocks_offsets[i_n]) /
              t_wt.bcsc_bn;
          for (int l_block = 0; l_block < cur_n_blocks;
               l_block += t_wt.bcsc_blocks_in_bn) {
            int nk = (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) / Hk;
            T* dst = (spmm_use_flat_acts > 0) ? out[s1 * BSb][nk] : tmp_block;
            if (with_bias) {
              if (spmm_use_flat_acts > 0) {
                if (s1 == BS_blocks - 1) {
                  copy_bias_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_tpp(bias[nk], dst);
                }
              } else {
                if (s1 == BS_blocks - 1) {
                  copy_bias_row_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_row_tpp(bias[nk], dst);
                }
              }
            } else {
              if (s1 == BS_blocks - 1) {
                zero_tpp_rem(dst);
              } else {
                zero_tpp(dst);
              }
            }
            if (spmm_use_flat_acts == 0 || rem == BSb || s1 < BS_blocks - 1) {
              spmm_tpp(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  true);
            } else {
              spmm_tpp_rem(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  false);
              spmm_tpp.config();
            }
            if (s1 == BS_blocks - 1) {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp_rem(dst, out[s1 * BSb][nk]);
              gelu_fwd_tpp_rem(out[s1 * BSb][nk], out[s1 * BSb][nk]);
            } else {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp(dst, out[s1 * BSb][nk]);
              gelu_fwd_tpp(out[s1 * BSb][nk], out[s1 * BSb][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          spmm_tpp.config();
        },
        [&]() {
          spmm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_gelu_sparse(
    at::IntArrayRef t_in_sizes,
    at::Tensor t_in,
    tensor_bcsc_t& t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in_sizes.vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);
  fc_gelu_sparse<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_gelu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_gelu(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_gelu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_gelu_compressed(
    at::Tensor t_in,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);
  auto gemm_tpp = SCOPEITGEMM((GemmTPP<T, T>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, T>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_gelu_compressed(
    at::Tensor t_in,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);
  fc_gelu_compressed<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_silu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_silu(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_silu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_silu_compressed(
    at::Tensor t_in,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);

  auto gemm_tpp = SCOPEITGEMM((GemmTPP<T, T>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, T>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_silu_compressed(
    at::Tensor t_in,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);
  fc_silu_compressed<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_relu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(i_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<T>(BSb, Hk, K, K, false), ACT);
  auto relu_fwd_tpp_rem = SCOPEIT(ReLUFwdTPP<T>(rem, Hk, K, K, false), ACT);

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T>
inline at::Tensor fc_relu(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_relu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tout = T>
inline void qkv_gemm_sparse(
    at::Tensor t_in,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(qkv_gemm_sparse, {t_in});
  auto in_sizes = t_in.sizes();
  auto out_sizes = t_out.sizes();
  auto BS = out_sizes[0] * out_sizes[1];
  auto C = (spmm_use_flat_acts > 0) ? in_sizes[2]
                                    : in_sizes[1] * in_sizes[2] * in_sizes[4];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;
  auto BSb = (spmm_use_flat_acts > 0) ? LIBXSMM_MIN(BS, SPMM_PACKED_BLOCK_SIZE)
                                      : in_sizes[3];
  auto BS_blocks =
      (spmm_use_flat_acts > 0) ? (BS + BSb - 1) / BSb : in_sizes[0];
  auto in = GetVLAPtr<T>(t_in, {BSb * C});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});
  auto rem = (BS % BSb == 0) ? BSb : (BS % BSb);
  bool with_bias = (t_bias.numel() > 0);

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto copy_bias_row_tpp = SCOPEIT(CpyBiasRowTPP<T>(Hk, BSb, BSb), BIAS);
  auto copy_bias_row_tpp_rem = SCOPEIT(CpyBiasRowTPP<T>(Hk, rem, BSb), BIAS);
  auto zero_tpp = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);
  auto zero_tpp_rem = (spmm_use_flat_acts > 0)
      ? SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO)
      : SCOPEIT(SetZeroTPP<T>(Hk, BSb, BSb), EW_ZERO);
  auto trans_out_tpp = SCOPEIT(
      XformExtTPP<T>(Hk, BSb, BSb, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);
  auto trans_out_tpp_rem = SCOPEIT(
      XformExtTPP<T>(Hk, rem, rem, Hk, BSb, K, XformTPP::XFORM_XPOSE_TPP, true),
      XPOSE);

  auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
      BSb,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));

  auto spmm_tpp_rem = SCOPEITGEMM((SpmmTPP<T, T>(
      rem,
      Hk,
      Nc * Hc,
      t_wt.bcsc_bk,
      t_wt.bcsc_bn,
      (spmm_use_flat_acts > 0) ? C : Nc * Hc,
      (spmm_use_flat_acts > 0) ? K : Hk,
      1.0,
      spmm_use_flat_acts)));

  {
    RECORD_OMP_TIME();
    auto loop_scheme = large_cache_opt ? "BA" : "Ba";
    auto gemm_loop = ThreadedLoop<2>(
        {LoopSpecs{BS_blocks}, LoopSpecs{t_wt.n_blocks}}, loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int s1 = ind[0], i_n = ind[1];
          T tmp_block[BSb * Hk];
          int cur_n_blocks =
              (t_wt.Nblocks_offsets[i_n + 1] - t_wt.Nblocks_offsets[i_n]) /
              t_wt.bcsc_bn;
          for (int l_block = 0; l_block < cur_n_blocks;
               l_block += t_wt.bcsc_blocks_in_bn) {
            int nk = (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) / Hk;
            T* dst = (spmm_use_flat_acts > 0) ? out[s1 * BSb][nk] : tmp_block;
            if (with_bias) {
              if (spmm_use_flat_acts > 0) {
                if (s1 == BS_blocks - 1) {
                  copy_bias_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_tpp(bias[nk], dst);
                }
              } else {
                if (s1 == BS_blocks - 1) {
                  copy_bias_row_tpp_rem(bias[nk], dst);
                } else {
                  copy_bias_row_tpp(bias[nk], dst);
                }
              }
            } else {
              if (s1 == BS_blocks - 1) {
                zero_tpp_rem(dst);
              } else {
                zero_tpp(dst);
              }
            }
            if (spmm_use_flat_acts == 0 || rem == BSb || s1 < BS_blocks - 1) {
              spmm_tpp(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  true);
            } else {
              spmm_tpp_rem(
                  in[s1],
                  t_wt,
                  t_wt.bcsc_blocks_in_bn,
                  (t_wt.Nblocks_offsets[i_n] + l_block * t_wt.bcsc_bn) /
                      t_wt.bcsc_bn,
                  dst,
                  false);
              spmm_tpp.config();
            }
            if (s1 == BS_blocks - 1) {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp_rem(dst, out[s1 * BSb][nk]);
            } else {
              if (spmm_use_flat_acts == 0)
                trans_out_tpp(dst, out[s1 * BSb][nk]);
            }
          }
        },
        [&]() {
          TimerStart();
          spmm_tpp.config();
        },
        [&]() {
          spmm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T, typename Tout = T>
inline void qkv_gemm(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(qkv_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % BSb;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, Tout>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, Tout>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto gemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() {
          TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T, typename Tout = T>
inline void qkv_gemm_compressed(
    at::Tensor t_in,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  RECORD_SCOPE(qkv_gemm_compressed, {t_in});
  auto in_sizes = t_in.sizes();
  auto out_sizes = t_out.sizes();
  auto BS = out_sizes[0] * out_sizes[1];
#if 0
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
#endif
  auto C = in_sizes[2];
  auto Nc = t_wt.sizes[1];
  auto Hc = C / Nc;
  auto Nk = t_wt.sizes[0];
  auto Hk = t_wt.sizes[3];
  auto K = Nk * Hk;

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % BSb;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);

  auto gemm_tpp =
      SCOPEITGEMM((GemmTPP<T, Tout>(BSb, Hk, C, C, Hk, K, 1.0, 0, 1)));
  auto gemm_tpp_rem =
      SCOPEITGEMM((GemmTPP<T, Tout>(rem, Hk, C, C, Hk, K, 1.0, 0, 1)));

  {
    RECORD_OMP_TIME();
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto gemm_loop = ThreadedLoop<3>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0, BS, BSb}, LoopSpecs{Nk}},
        loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            gemm_tpp(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                true);
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            gemm_tpp_rem(
                in[s1][nc],
                *((T**)t_wt.data_ptrs + nk),
                out[s1][nk],
                (char*)t_wt.bitmap + nk * C * (Hk / 8),
                false);
            gemm_tpp.config();
          }
        },
        [&]() {
          TimerStart();
          gemm_tpp.config();
        },
        [&]() {
          gemm_tpp.release();
          TimerEnd();
        });
  }
}

template <typename T, typename Tout = T>
inline at::Tensor qkv_gemm(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];
  auto t_out = t_in.new_empty(sizes);
  qkv_gemm<T, Tout>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tout = T>
inline at::Tensor qkv_gemm_compressed(
    at::Tensor t_in,
    tensor_compressed_t& t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);
  qkv_gemm_compressed<T, Tout>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tout = T>
inline at::Tensor qkv_gemm_sparse(
    at::IntArrayRef t_in_sizes,
    at::Tensor t_in,
    tensor_bcsc_t& t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in_sizes.vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);
  qkv_gemm_sparse<T, Tout>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tv>
struct AttnKernels {
  SCOPEIT_DECL(BrgemmTPP<T, float>) a_gemm_tpp;
  SCOPEIT_DECL(ScaleTPP<float, float>) scale_tpp;
  SCOPEIT_DECL(AddBiasTPP<T>) add_mask_tpp;
  SCOPEIT_DECL(AddTPP<T, float, float>) add_2dmask_tpp;
  SCOPEIT_DECL(VarSoftMaxFwdTPP<float, Tv>) softmax_fwd_tpp;
  SCOPEIT_DECL(BrgemmTPP<Tv, Tv>) c_gemm_tpp;
  SCOPEIT_DECL(ConvertTPP<Tv, T>) cvt_tpp;
  SCOPEIT_DECL(XformExtTPP<T>) xform_tpp;
  SCOPEIT_DECL(XformExtTPP<T>) vnni_tpp;
  SCOPEIT_DECL(SoftMaxFixUpTPP<T>) softmax_fixup;

  AttnKernels(
      long Sqb,
      long Skb,
      long H,
      int pad,
      int kl_in_vnni,
      int vl_in_vnni) {
    // printf("Sqb: %ld, Skb: %ld, H: %ld, psd: %d, kl: %d, vl: %d\n", Sqb, Skb,
    // H, pad, kl_in_vnni, vl_in_vnni);
    if (Sqb == 0)
      Sqb = 1; // hack for unused kernels to not generate 0 size kernels
    if (Skb == 0)
      Skb = 2; // hack for unused kernels to not generate 0 size kernels
    // [Sqb, H] * [H, Skb] = [Sqb, Skb]
    a_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
        Sqb, Skb, H, H, H * Skb, H, Skb, Skb, 0.0, 0, 1, kl_in_vnni)));
    // [Sqb, Skb]
    scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sqb * Skb)), EW_SCL);
    add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sqb, Skb), EW_ADD);
    add_2dmask_tpp = SCOPEIT((AddTPP<T, float, float>(Sqb, Skb)), EW_ADD);
    softmax_fwd_tpp = SCOPEIT((VarSoftMaxFwdTPP<float, Tv>(Sqb, Skb)), SOFTMAX);
    softmax_fixup = SCOPEIT((SoftMaxFixUpTPP<T>(Sqb, H)), EW_RCP);
    // [Sqb, Skb] * [Skb, H] = tmp[Sqb, H]
    c_gemm_tpp = SCOPEITGEMM((BrgemmTPP<Tv, Tv>(
        Sqb, H, Skb, Sqb * Skb, Skb * H, Skb, H, H, 0.0, 0, 1, vl_in_vnni)));
    // [Sqb, H] --> [Sqb, H]
    cvt_tpp = SCOPEIT((ConvertTPP<Tv, T>(Sqb, H, H, H)), EW_COPY);
    auto xform = XformTPP::XFORM_XPOSE_TPP;
    if (!std::is_same<T, float>::value && kl_in_vnni) {
      xform = XformTPP::XFORM_XPOSE_N2V_TPP;
    }
    // [Skb-pad, H] --> [H, Skb]
    xform_tpp = SCOPEIT(
        XformExtTPP<T>(Skb - pad, H, H, Skb, H, Skb, xform, true), XPOSE);
    if (vl_in_vnni != 0)
      vnni_tpp = SCOPEIT(
          XformExtTPP<T>(
              Skb - pad, H, Skb, H, H, H, XformTPP::XFORM_N2V_TPP, true),
          VNNI);
  }
};

template <typename T>
inline at::Tensor attn(
    at::Tensor t_QL,
    at::Tensor t_KL,
    at::Tensor t_AM,
    at::Tensor t_VL,
    at::Tensor t_KL_cache,
    at::Tensor t_VL_cache,
    VLAPtr<long, 1, long>& beam_idx,
    long offset) {
  RECORD_SCOPE(ac_gemm2, {t_QL, t_KL});
  auto t_CL = at::empty_like(t_QL);
  auto sizes = t_QL.sizes();
  long B = sizes[0];
  long Nq = sizes[1];
  long Sq = sizes[2];
  long H = sizes[3];
  float one_by_sqrt_H = 1.0 / sqrtf(H);
  auto ksizes = t_KL.sizes();
  long Sk = ksizes[2];
  long Nkv = ksizes[1];
  long Nq_per_kv = (Nq == Nkv ? 1 : Nq / Nkv);
  // printf("Sq = %ld, Sk = %ld\n", Sq, Sk);
  // std::cout << "QL: " << t_QL.sizes() << std::endl;
  // std::cout << "KL: " << t_KL.sizes() << std::endl;
  TPP_ASSERT(
      Sq == 1 && Sk == 1,
      "Sq (%ld) and Sk (%ld) must be 1, offset (%ld)\n",
      Sq,
      Sk,
      offset);
  auto FSk = offset + Sk;
  auto FSk_aligned = (FSk + 0x3FL) & ~0x3FL;
#ifdef S_FIRST_KVC
  auto CSk = t_KL_cache.size(0);
#else
  auto CSk = t_KL_cache.size(2);
#endif
  // printf("CSk = %d, FSk = %d\n", (int)CSk, (int)FSk);
  const bool am_valid = (t_AM.numel() > 0);

  auto QL = GetVLAPtr<T>(t_QL, {Nq, Sq, H});
  auto KL = GetVLAPtr<T>(t_KL, {Nkv, Sk, H});
  auto VL = GetVLAPtr<T>(t_VL, {Nkv, Sk, H});
  auto CL = GetVLAPtr<T>(t_CL, {Nq, Sq, H});
  auto AM = GetVLAPtr<T>(t_AM, {FSk});
#ifdef S_FIRST_KVC
  auto KL_C = GetVLAPtr<T>(t_KL_cache, {B, Nkv, H});
  auto VL_C = GetVLAPtr<T>(t_VL_cache, {B, Nkv, H});
#else
  auto KL_C = GetVLAPtr<T>(t_KL_cache, {Nkv, CSk, H});
  auto VL_C = GetVLAPtr<T>(t_VL_cache, {Nkv, CSk, H});
#endif

  // Removing SCOPEIT due to very high overhead of timing these
  // auto dot_tpp = SCOPEIT((MulReduceTPP<T,T,float>(1, H)), EW_MUL);
  // auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(H)), EW_ADD);
  // auto cpy_tpp = SCOPEIT(CpyTPP<T>(H), EW_COPY);
  // auto zero_tpp = SCOPEIT(SetZeroTPP<T>(H), EW_ZERO);
  auto dot_tpp = MulReduceTPP<float, T, float>(1, H);
  auto scale_add_tpp = ScaleAddTPP<T, float>(H);
  auto cpy_tpp = CpyTPP<T>(H);
  auto cvt_f2b_tpp = ConvertTPP<float, T>(H);
  auto cvt_b2f_tpp = ConvertTPP<T, float>(H);
  auto zero_tpp = SetZeroTPP<float>(H);
  auto softmax_fwd_tpp =
      SCOPEIT((SoftMaxFwdTPP<float, float>(1, 1, FSk_aligned)), SOFTMAX);
  auto nThreads = omp_get_max_threads();
  float tasks_per_thread = ((float)B * Nq) / nThreads;
  auto thr_eff = tasks_per_thread / ceilf(tasks_per_thread);
  if (FSk <= 256 || thr_eff >= 0.75) {
    RECORD_OMP_TIME();
    {
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        // auto t00 = getTime();
        TimerStart();
#pragma omp for collapse(2) nowait
        for (int nq = 0; nq < Nq; nq++) {
          for (int b = 0; b < B; b++) {
            float AS[FSk_aligned];
            int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
            // float *AS = GAS[tid]; //FSk];
            // auto t0 = getTime();
            {
              ScopedTimer t_(
                  BRGEMM,
                  2 * FSk * H,
                  (FSk + H) * sizeof(T) + H * FSk * sizeof(float));
              float tmp_QL[H];
              cvt_b2f_tpp(QL[b][nq][0], tmp_QL);
              for (int sk = 0; sk < FSk; sk++) {
                AS[sk] = 0.0f;
                if (sk < offset) {
                  int bid = beam_idx[b][sk];
                  // printf("b: %d n: %d sk: %d  bid = %d\n", b, n, sk, bid);
#ifdef S_FIRST_KVC
                  dot_tpp(tmp_QL, KL_C[sk][bid][nkv], &AS[sk]);
#else
                  dot_tpp(tmp_QL, KL_C[bid][nkv][sk], &AS[sk]);
#endif
                } else {
                  // printf("b: %d n: %d sk: %d \n", b, n, sk);
                  dot_tpp(tmp_QL, KL[b][nkv][0], &AS[sk]);
#ifdef S_FIRST_KVC
                  cpy_tpp(KL[b][nkv][0], KL_C[sk][b][nkv]);
#else
                  cpy_tpp(KL[b][nkv][0], KL_C[b][nkv][sk]);
#endif
                }
                AS[sk] *= one_by_sqrt_H;
                if (am_valid) {
                  AS[sk] += AM[b][sk];
                }
              }
              for (int sk = FSk; sk < FSk_aligned; sk++) {
                // pad AS to align for softmax
                AS[sk] = -1e9f;
              }
            }
            // auto t1 = getTime();
            softmax_fwd_tpp(AS, AS);
            // auto t2 = getTime();
            // printf("post softmax b: %d n: %d\n", b, n);
            {
              float tmp_CL[H];
              ScopedTimer t_(
                  BRGEMM,
                  2 * FSk * H,
                  (FSk + H) * sizeof(T) + H * FSk * sizeof(float));
              zero_tpp(tmp_CL);
              for (int sk = 0; sk < FSk; sk++) {
                // printf("bmm2: b: %d n: %d sk: %d \n", b, n, sk);
                // if (b == 0&& n == 0) printf("AS[%d]: %g\n", sk, AS[sk]);
                if (sk < offset) {
                  int bid = beam_idx[b][sk];
#ifdef S_FIRST_KVC
                  scale_add_tpp(VL_C[sk][bid][nkv], tmp_CL, AS[sk]);
#else
                  scale_add_tpp(VL_C[bid][nkv][sk], tmp_CL, AS[sk]);
#endif
                } else {
                  scale_add_tpp(VL[b][nkv][0], tmp_CL, AS[sk]);
#ifdef S_FIRST_KVC
                  cpy_tpp(VL[b][nkv][0], VL_C[sk][b][nkv]);
#else
                  cpy_tpp(VL[b][nkv][0], VL_C[b][nkv][sk]);
#endif
                }
              }
              cvt_f2b_tpp(tmp_CL, CL[b][nq][0]);
            }
            // auto t3 = getTime();
            // if (tid == 0) printf("MHA: bns= %d %d %ld  %10g %10g %10g
            // %10g\n", b, n, FSk, (t1-t0)*1e6, (t2-t1)*1e6, (t3-t2)*1e6,
            // (t3-t0)*1e6);
          }
        }
        TimerEnd();
        // auto t01 = getTime();
        // if (tid == 0) printf("MHA: s= %ld  %10g\n", FSk, (t01-t00)*1e6);
      }
    }
  } else {
    auto t_AS = t_QL.new_empty({B, Nq, FSk_aligned}, at::kFloat);
    // auto t_XL = t_QL.new_empty({B, N, H}, at::kFloat);
    auto t_XL = t_QL.to(at::kFloat);
    auto XL = GetVLAPtr<float>(t_XL, {Nq, H});
    auto AS = GetVLAPtr<float>(t_AS, {Nq, FSk_aligned});

    {
      {
        RECORD_SCOPE(ac_gemm21, {});
        RECORD_OMP_TIME();
#pragma omp parallel
        {
          TimerStart();
#pragma omp for collapse(3)
          for (int nq = 0; nq < Nq; nq++) {
            for (int b = 0; b < B; b++) {
              for (int sk = 0; sk < FSk; sk++) {
                int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
                AS[b][nq][sk] = 0.0f;
                if (sk < offset) {
                  int bid = beam_idx[b][sk];
                  // printf("b: %d n: %d sk: %d  bid = %d\n", b, n, sk, bid);
#ifdef S_FIRST_KVC
                  dot_tpp(XL[b][nq], KL_C[sk][bid][nkv], &AS[b][nq][sk]);
#else
                  dot_tpp(XL[b][nq], KL_C[bid][nkv][sk], &AS[b][nq][sk]);
#endif
                } else {
                  // printf("b: %d n: %d sk: %d \n", b, n, sk);
                  dot_tpp(XL[b][nq], KL[b][nkv][0], &AS[b][nq][sk]);
#ifdef S_FIRST_KVC
                  cpy_tpp(KL[b][nkv][0], KL_C[sk][b][nkv]);
#else
                  cpy_tpp(KL[b][nkv][0], KL_C[b][nkv][sk]);
#endif
                }
                AS[b][nq][sk] *= one_by_sqrt_H;
                if (am_valid) {
                  AS[b][nq][sk] += AM[b][sk];
                }
              }
            }
          }
          TimerEnd();
        }
      }
      {
        RECORD_SCOPE(ac_gemm22, {});
        RECORD_OMP_TIME();
#pragma omp parallel
        {
          TimerStart();
#pragma omp for collapse(2)
          for (int nq = 0; nq < Nq; nq++) {
            for (int b = 0; b < B; b++) {
              int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
              for (int sk = FSk; sk < FSk_aligned; sk++) {
                // pad AS to align for softmax
                AS[b][nq][sk] = -1e9f;
              }
              softmax_fwd_tpp(AS[b][nq], AS[b][nq]);
              zero_tpp(XL[b][nq]);
              for (int sk = 0; sk < FSk; sk++) {
                if (sk < offset) {
                  int bid = beam_idx[b][sk];
#ifdef S_FIRST_KVC
                  scale_add_tpp(VL_C[sk][bid][nkv], XL[b][nq], AS[b][nq][sk]);
#else
                  scale_add_tpp(VL_C[bid][nkv][sk], XL[b][nq], AS[b][nq][sk]);
#endif
                } else {
                  scale_add_tpp(VL[b][nkv][0], XL[b][nq], AS[b][nq][sk]);
#ifdef S_FIRST_KVC
                  cpy_tpp(VL[b][nkv][0], VL_C[sk][b][nkv]);
#else
                  cpy_tpp(VL[b][nkv][0], VL_C[b][nkv][sk]);
#endif
                }
              }
            }
          }
          TimerEnd();
        }
      }
    }
    t_CL = t_XL.to(t_CL.dtype());
  }
  return t_CL;
}

template <typename T, typename Tv>
inline at::Tensor attn(
    at::Tensor t_QL,
    at::Tensor t_KL,
    at::Tensor t_AM,
    at::Tensor t_VL) {
  RECORD_SCOPE(ac_gemm1, {t_QL, t_KL});
  auto t_CL = at::empty_like(t_QL);
  auto sizes = t_QL.sizes();
  long B = sizes[0];
  long Nq = sizes[1];
  long Sq = sizes[2];
  long H = sizes[3];
  float one_by_sqrt_H = 1.0 / sqrtf(H);
  auto ksizes = t_KL.sizes();
  long Sk = ksizes[2];
  long Nkv = ksizes[1];
  const long Nq_per_kv = (Nq == Nkv ? 1 : Nq / Nkv);
  // printf("Nq = %ld, Nkv = %ld, Nq_per_kv = %ld\n", Nq, Nkv, Nq_per_kv);
  // fflush(stdout);
  long offset = Sk - Sq;
  constexpr long Sqb = 64;
  long qrem = Sq % Sqb;
  bool inline_trans = ((Sq + Sqb - 1) / Sqb == 1);
  bool am_is_2d = t_AM.size(2) != 1;

  int vl_in_vnni = 1; //(Sk % 2 == 0 ? 1 : 0);
  const long VBS = (vl_in_vnni ? get_vnni_block_size<T>() : 1);
  long Sk_pad = (Sk + VBS - 1) & ~(VBS - 1);
  const long Skb = (!inline_trans ? 2048 : SK_BLOCK_SIZE);
  long krem = Sk % Skb;
  int pad = Sk_pad - Sk;

  auto t_KL_TV = t_KL.new_empty({B, Nkv, Sk_pad, H});
  auto t_VL_V = t_VL;
  if (VBS != 1) {
    t_VL_V = t_VL.new_empty({B, Nkv, Sk_pad, H});
  }
  if (Sk != Sk_pad) {
    // TPP_ASSERT(am_is_2d == false, "2D AM not supported yet\n");
    if (!am_is_2d) {
      auto t_tmp = t_AM.new_empty({B, pad});
      t_tmp.fill_(-10000.0);
      t_AM = at::cat({t_AM.view({B, -1}), t_tmp}, -1);
    } else {
      auto t_tmp = t_AM.new_empty({B, 1, Sq, pad});
      t_tmp.fill_(-10000.0);
      t_AM = at::cat({t_AM, t_tmp}, -1);
    }
  }
  auto QL = GetVLAPtr<T>(t_QL, {Nq, Sq, H});
  auto KL = GetVLAPtr<T>(t_KL, {Nkv, Sk, H});
  auto KL_TV = GetVLAPtr<T>(t_KL_TV, {Nkv, Sk_pad, H});
  auto VL = GetVLAPtr<Tv>(t_VL, {Nkv, Sk, H});
  auto VL_V = GetVLAPtr<Tv>(t_VL_V, {Nkv, Sk_pad, H});
  auto CL = GetVLAPtr<T>(t_CL, {Nq, Sq, H});
  auto AM = GetVLAPtr<T>(t_AM, {Sk_pad});
  auto AM2 = GetVLAPtr<T>(t_AM, {Sq, Sk_pad});
  int kl_in_vnni = 1;

  AttnKernels<T, Tv> attn_kern[4] = {
      AttnKernels<T, Tv>(Sqb, Skb, H, 0, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(Sqb, krem + pad, H, pad, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(qrem, Skb, H, 0, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(qrem, krem + pad, H, pad, kl_in_vnni, vl_in_vnni),
  };

  if (!inline_trans) {
    RECORD_SCOPE(k_trans, {t_QL, t_KL});
#pragma omp parallel for collapse(3)
    for (int n = 0; n < Nkv; n++) {
      for (int b = 0; b < B; b++) {
        for (int sk = 0; sk < Sk; sk += Skb) {
          int kid = (sk + Skb > Sk) ? 1 : 0;
          attn_kern[kid].xform_tpp(KL[b][n][sk], KL_TV[b][n][sk]);
          if (VBS != 1)
            attn_kern[kid].vnni_tpp(VL[b][n][sk], VL_V[b][n][sk]);
        }
      }
    }
  }

  {
    RECORD_OMP_TIME();
    {
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int nq = 0; nq < Nq; nq++) {
          for (int sq = 0; sq < Sq; sq += Sqb) {
            int nkv = nq / Nq_per_kv;
            long qbs = (Sq - sq >= Sqb ? Sqb : Sq - sq);
            int qid = (sq + Sqb > Sq) ? 1 : 0;
            float omax[qbs], osum[qbs], cmax[qbs], csum[qbs];
            for (int sk = 0; sk < Sk; sk += Skb) {
              long kbs = (Sk - sk >= Skb ? Skb : Sk_pad - sk);
              int kid = qid * 2 + ((sk + Skb > Sk) ? 1 : 0);
              auto& ak = attn_kern[kid];
              float AS[qbs][kbs];
              Tv AST[qbs][kbs];
              T* k_ptr = KL_TV[b][nkv][sk];
              T k_tmp[kbs * H];
              if (inline_trans) {
                // ak.xform_tpp(KL[b][n][sk], KL_TV[b][n][sk]);
                ak.xform_tpp(KL[b][nkv][sk], k_tmp);
                k_ptr = k_tmp;
              }
              ak.a_gemm_tpp(QL[b][nq][sq], k_ptr, AS[0], 1);
              for (int sq1 = 0; sq1 < qbs; sq1++) {
                auto qval = sq + sq1 + offset;
                for (int sk1 = qval + 1; sk1 < sk + kbs; sk1++) {
                  AS[sq1][sk1 - sk] = -1e9f;
                }
              }
              ak.scale_tpp(AS[0], AS[0], one_by_sqrt_H);
              if (t_AM.numel() != 0) {
                if (am_is_2d)
                  ak.add_2dmask_tpp(&AM2[b][sq][sk], AS[0], AS[0]);
                else
                  ak.add_mask_tpp(&AM[b][sk], AS[0]);
              }
              float *pmax, *psum;
              if (sk == 0) {
                pmax = omax;
                psum = osum;
              } else {
                pmax = cmax;
                psum = csum;
              }
              ak.softmax_fwd_tpp(1, AS[0], AST[0], pmax, psum);
              // for (int xx=0;xx<kbs;xx++)
              // if (b == 0 && n == 0 && Sq == 1) printf("AS[%d]: %g\n", sk+xx,
              // (float)AST[0][xx]);
              Tv tmp[qbs * H];
              Tv* v_ptr = VL_V[b][nkv][sk];
              Tv v_tmp[kbs * H];
              if (inline_trans && VBS != 1) {
                // ak.vnni_tpp(VL[b][n][sk], VL_V[b][n][sk]);
                ak.vnni_tpp(VL[b][nkv][sk], v_tmp);
                v_ptr = v_tmp;
              }
              ak.c_gemm_tpp(AST[0], v_ptr, tmp, 1);
              if (sk == 0) {
                ak.cvt_tpp(tmp, CL[b][nq][sq]);
              } else {
                ak.softmax_fixup(tmp, CL[b][nq][sq], cmax, csum, omax, osum);
              }
            }
          }
        }
      }
    }
  }
  return t_CL;
}

template <typename cls>
struct LLMBlock : torch::CustomClassHolder {
 public:
  std::string name;
  at::Tensor t_dummy;
  at::Tensor t_dummy_int;
  caffe2::TypeMeta dt;
  caffe2::TypeMeta ldt;

  LLMBlock(std::string name, at::Tensor& t, at::Tensor& lt)
      : name(name),
        t_dummy(t.new_empty({0})),
        t_dummy_int(t.new_empty({0}, at::kLong)),
        dt(t.dtype()),
        ldt(lt.dtype()) {}

  std::vector<at::Tensor> forward_common(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    GlobalPass _gp(FWD);
    RECORD_FUNCTION(name, std::vector<c10::IValue>());
    std::vector<at::Tensor> ret;
    auto self = static_cast<cls*>(this);

    if (dt == at::kFloat && ldt == at::kFloat) {
      ret = self->template _forward<float, float>(t_inp, t_cache, use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
      ret = self->template _forward<bfloat16, float>(t_inp, t_cache, use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
      ret = self->template _forward<bfloat16, bfloat16>(
          t_inp, t_cache, use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
      ret = self->template _forward<bfloat8, float>(t_inp, t_cache, use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kBFloat16) {
      ret =
          self->template _forward<bfloat8, bfloat16>(t_inp, t_cache, use_cache);
    } else {
      std::cout << "Types: " << dt << "  " << ldt << std::endl;
      TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
    }
    // printf("Returning Layer \n");

    return ret;
  }

  template <typename T>
  std::vector<at::Tensor> self_mha(
      at::Tensor t_QL,
      at::Tensor t_KL,
      at::Tensor t_VL,
      at::Tensor t_am,
      std::vector<at::Tensor>& t_cache) {
    RECORD_SCOPE(mha, {t_QL, t_KL});
    auto self = static_cast<cls*>(this);
    auto t_key_past = this->t_dummy;
    auto t_value_past = this->t_dummy;
    auto t_beam_idx = this->t_dummy_int;
    auto t_offset = this->t_dummy_int;
    auto B = t_QL.size(0);
    auto S = t_QL.size(1);
    // auto N = self->N;
    auto H = self->H;
    at::Tensor t_CL;
    long offset = 0;
    int csz = t_cache.size();
    if (csz > 0)
      t_key_past = t_cache[0];
    if (csz > 1)
      t_value_past = t_cache[1];
    if (csz > 2)
      t_beam_idx = t_cache[2].to(at::kLong);
    if (csz > 3) {
      t_offset = t_cache[3];
      offset = t_offset.item<long>();
      TPP_ASSERT(
          csz == 6, "Updated indirect kv_cache tuple should be of length 6\n");
      t_key_past = t_cache[4];
      t_value_past = t_cache[5];
    } else if (csz > 0) {
      offset = t_key_past.size(2);
    }
    // TPP_ASSERT(S == 1, "S must be 1");

    t_QL = t_QL.view({B, S, -1, H}).permute({0, 2, 1, 3}).contiguous();
    t_KL = t_KL.view({B, S, -1, H}).permute({0, 2, 1, 3}).contiguous();
    t_VL = t_VL.view({B, S, -1, H}).permute({0, 2, 1, 3}).contiguous();
    auto Nq = t_QL.size(1);
    auto Nkv = t_KL.size(1);
    // printf("%s:%d Nq = %ld, Nkv = %ld\n", __func__, __LINE__, Nq, Nkv);
    // TPP_ASSERT(Nq == Nkv, "Nq and Nkv are not equal\n");

    if (csz < 4) {
      if (t_key_past.numel() > 0) {
        t_KL = kv_concat<T>(t_key_past, t_KL, 2, t_beam_idx);
      }
      if (t_value_past.numel() > 0) {
        t_VL = kv_concat<T>(t_value_past, t_VL, 2, t_beam_idx);
      }
      // std::cout << "1 t_KL.shape: " << t_KL.sizes() << std::endl;

      t_CL = attn<T, T>(t_QL, t_KL, t_am, t_VL);
      t_CL = t_CL.view({B, Nq, S, H})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({B, S, Nq * H});

      return {t_CL, t_KL, t_VL};

    } else if (offset == 0) {
      // std::cout << "2 t_KL.shape: " << t_KL.sizes() << std::endl;
      t_CL = attn<T, T>(t_QL, t_KL, t_am, t_VL);
      auto capacity = S + KV_CACHE_INC_SIZE;
#ifdef S_FIRST_KVC
      t_key_past = t_KL.new_zeros({capacity, B, Nkv, H});
      t_value_past = t_VL.new_zeros({capacity, B, Nkv, H});
#else
      t_key_past = t_KL.new_zeros({B, Nkv, capacity, H});
      t_value_past = t_VL.new_zeros({B, Nkv, capacity, H});
#endif
      // t_beam_idx = t_beam_idx.new_zeros({capacity, B});
      t_beam_idx =
          at::arange(B).unsqueeze(0).expand({capacity, B}).contiguous();
      // if (my_rank == 0) std::cout << "t_beam_idx: " << t_beam_idx.sizes() <<
      // std::endl;
      t_offset = t_offset + S;
#ifdef S_FIRST_KVC
      t_key_past.slice(0, 0, S, 1).copy_(t_KL.permute({2, 0, 1, 3}));
      t_value_past.slice(0, 0, S, 1).copy_(t_VL.permute({2, 0, 1, 3}));
#else
      t_key_past.slice(2, 0, S, 1).copy_(t_KL);
      t_value_past.slice(2, 0, S, 1).copy_(t_VL);
#endif
      t_CL = t_CL.view({B, Nq, S, H})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({B, S, Nq * H});
      return {t_CL, t_KL, t_VL, t_beam_idx, t_offset, t_key_past, t_value_past};
      // printf("old offset = %d, new_offset = %ld\n", offset,
      // t_offset.item<long>());
    } else {
#ifdef S_FIRST_KVC
      auto capacity = t_key_past.size(0);
#else
      auto capacity = t_key_past.size(2);
#endif
      if (capacity <= offset) {
        printf(
            "Warning: Reallocating kv cache, consider increasing KV_CACHE_INC_SIZE (%d)\n",
            KV_CACHE_INC_SIZE);
        auto new_capacity = offset + KV_CACHE_INC_SIZE;
#ifdef S_FIRST_KVC
        auto t_key_past_new = t_key_past.new_empty({new_capacity, B, Nkv, H});
        t_key_past_new.slice(0, 0, offset, 1).copy_(t_key_past);
        t_key_past = t_key_past_new;

        auto t_value_past_new =
            t_value_past.new_empty({new_capacity, B, Nkv, H});
        t_value_past_new.slice(0, 0, offset, 1).copy_(t_value_past);
        t_value_past = t_value_past_new;
#else
        auto t_key_past_new = t_key_past.new_empty({B, Nkv, new_capacity, H});
        t_key_past_new.slice(2, 0, offset, 1).copy_(t_key_past);
        t_key_past = t_key_past_new;

        auto t_value_past_new =
            t_value_past.new_empty({B, Nkv, new_capacity, H});
        t_value_past_new.slice(2, 0, offset, 1).copy_(t_value_past);
        t_value_past = t_value_past_new;
#endif

        auto t_beam_idx_new =
            at::arange(B).unsqueeze(0).expand({new_capacity, B}).contiguous();
        t_beam_idx_new.slice(0, 0, offset, 1).copy_(t_beam_idx);
        t_beam_idx = t_beam_idx_new;
      }

      // if (my_rank == 0) std::cout << "t_beam_idx2: " << t_beam_idx.sizes() <<
      // std::endl; std::cout << "t_key_past.shape:" << t_key_past.sizes() <<
      // std::endl; std::cout << "t_beam_idx.shape:" << t_beam_idx.sizes() <<
      // std::endl; std::cout << "t_offset:" << t_offset << std::endl; std::cout
      // << "B: " << B << " offset:" << offset << std::endl;
      auto t_new_beam_idx = t_beam_idx.new_empty({B, offset});
      auto beam_idx = GetVLAPtr<long>(t_new_beam_idx, {offset});
      auto b_ptr = GetVLAPtr<long>(t_beam_idx, {B});
      for (auto i = 0; i < B; i++) {
        beam_idx[i][offset - 1] = b_ptr[offset - 1][i];
        for (auto j = offset - 2; j >= 0;
             j--) { // for the token of input, the target beam is alwarys 0
          beam_idx[i][j] = b_ptr[j][beam_idx[i][j + 1]];
        }
      }
      t_CL = attn<T>(
          t_QL, t_KL, t_am, t_VL, t_key_past, t_value_past, beam_idx, offset);
      t_CL = t_CL.view({B, Nq, S, H})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({B, S, Nq * H});
      t_offset = t_offset + 1;
      S = t_offset.item<long>();
#ifdef S_FIRST_KVC
      t_KL = t_key_past.slice(0, 0, S, 1).permute({1, 2, 0, 3});
      t_VL = t_value_past.slice(0, 0, S, 1).permute({1, 2, 0, 3});
#else
      t_KL = t_key_past.slice(2, 0, S, 1);
      t_VL = t_value_past.slice(2, 0, S, 1);
#endif
      // printf("old offset = %d, new_offset = %ld\n", offset,
      // t_offset.item<long>());
      // std::cout << "t_key_past = " << t_key_past.sizes() << std::endl;
      return {t_CL, t_KL, t_VL, t_beam_idx, t_offset, t_key_past, t_value_past};
    }
  }
};

struct GPTJBlock : LLMBlock<GPTJBlock> {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wi, t_Wo;
  tensor_bcsc_t t_Wi_bcsc = {0}, t_Wo_bcsc = {0}, t_Wq_bcsc = {0},
                t_Wk_bcsc = {0}, t_Wv_bcsc = {0}, t_Wp_bcsc = {0};
  tensor_compressed_t t_Wq_compressed = {0}, t_Wk_compressed = {0},
                      t_Wv_compressed = {0}, t_Wp_compressed = {0},
                      t_Wi_compressed = {0}, t_Wo_compressed = {0};
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G, t_B;
  at::Tensor t_EP; // embed_positions
  float eps;
  long N, H;
  long max_positions, rotary_dim;

  GPTJBlock(
      std::vector<at::Tensor> params,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("gptj_fwd", params[2], params[0]),
        eps(eps),
        H(H),
        max_positions(max_positions),
        rotary_dim(rotary_dim) {
    int i = 0;
    t_G = params[i++]; // ln_gamma
    t_B = params[i++]; // ln_beta

    t_Wq = params[i++]; // q_proj
    t_Wk = params[i++]; // k_proj
    t_Wv = params[i++]; // v_proj
    t_Wp = params[i++]; // out_proj

    t_Wi = params[i++]; // fc_in
    t_Bi = params[i++];

    t_Wo = params[i++]; // fc_out
    t_Bo = params[i++];

    t_EP = params[i++]; // embed_positions

    if (sparse_type > 0) {
      update_sparse_copy();
    }

    N = t_Wq.size(0) * t_Wq.size(3) / H;
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " N=" << N << " H=" << H
                << std::endl;
    }
  }

  ~GPTJBlock() {
    if (sparse_type == 1) {
      libxsmm_free(t_Wi_bcsc.data);
      libxsmm_free(t_Wi_bcsc.rowidx);
      libxsmm_free(t_Wi_bcsc.colptr);
      libxsmm_free(t_Wi_bcsc.Nblocks_offsets);
      libxsmm_free(t_Wq_bcsc.data);
      libxsmm_free(t_Wq_bcsc.rowidx);
      libxsmm_free(t_Wq_bcsc.colptr);
      libxsmm_free(t_Wq_bcsc.Nblocks_offsets);
      libxsmm_free(t_Wk_bcsc.data);
      libxsmm_free(t_Wk_bcsc.rowidx);
      libxsmm_free(t_Wk_bcsc.colptr);
      libxsmm_free(t_Wk_bcsc.Nblocks_offsets);
      libxsmm_free(t_Wv_bcsc.data);
      libxsmm_free(t_Wv_bcsc.rowidx);
      libxsmm_free(t_Wv_bcsc.colptr);
      libxsmm_free(t_Wv_bcsc.Nblocks_offsets);
      libxsmm_free(t_Wp_bcsc.data);
      libxsmm_free(t_Wp_bcsc.rowidx);
      libxsmm_free(t_Wp_bcsc.colptr);
      libxsmm_free(t_Wp_bcsc.Nblocks_offsets);
      libxsmm_free(t_Wo_bcsc.data);
      libxsmm_free(t_Wo_bcsc.rowidx);
      libxsmm_free(t_Wo_bcsc.colptr);
      libxsmm_free(t_Wo_bcsc.Nblocks_offsets);
    } else if (sparse_type == 2) {
#if 0
      libxsmm_free(t_Wo_compressed.data);
      libxsmm_free(t_Wo_compressed.column_offsets);
      libxsmm_free(t_Wo_compressed.bitmap);
      libxsmm_free(t_Wi_compressed.data);
      libxsmm_free(t_Wi_compressed.column_offsets);
      libxsmm_free(t_Wi_compressed.bitmap);
      libxsmm_free(t_Wq_compressed.data);
      libxsmm_free(t_Wq_compressed.column_offsets);
      libxsmm_free(t_Wq_compressed.bitmap);
      libxsmm_free(t_Wk_compressed.data);
      libxsmm_free(t_Wk_compressed.column_offsets);
      libxsmm_free(t_Wk_compressed.bitmap);
      libxsmm_free(t_Wv_compressed.data);
      libxsmm_free(t_Wv_compressed.column_offsets);
      libxsmm_free(t_Wv_compressed.bitmap);
      libxsmm_free(t_Wp_compressed.data);
      libxsmm_free(t_Wp_compressed.column_offsets);
      libxsmm_free(t_Wp_compressed.bitmap);
#endif
    } else {
    }
  }

  void update_sparse_copy() {
    auto bcsc_bk = env2int("BK", 4);
    auto bcsc_bn = env2int("BN", 4);
    double sparsity = 0.0;
    int is_spr = (libxsmm_cpuid(NULL) == LIBXSMM_X86_AVX512_SPR) ? 1 * 0 : 0;
    int bcsc_bk_use, bcsc_bn_use;
    double sparse_threshold = 1.0 * env2int("SPTHRES", 40);

    /* Create BCSC sparse tensors */
    if (sparse_type == 1) {
      /* Create BCSC sparse tensors */
      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wi);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wi, &t_Wi_bcsc, bcsc_bk_use, bcsc_bn_use);

      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wq);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wq, &t_Wq_bcsc, bcsc_bk_use, bcsc_bn_use);

      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wk);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wk, &t_Wk_bcsc, bcsc_bk_use, bcsc_bn_use);

      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wv);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wv, &t_Wv_bcsc, bcsc_bk_use, bcsc_bn_use);

      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wp);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wp, &t_Wp_bcsc, bcsc_bk_use, bcsc_bn_use);

      // sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wo);
      bcsc_bk_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bk;
      bcsc_bn_use = (sparsity <= sparse_threshold && is_spr > 0) ? 32 : bcsc_bn;
      create_bcsc_from_blocked_nkkn_weight(
          t_Wo, &t_Wo_bcsc, bcsc_bk_use, bcsc_bn_use);
    } else if (sparse_type == 2) {
      create_compressed_from_blocked_nkkn_weight(t_Wi, &t_Wi_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wq, &t_Wq_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wk, &t_Wk_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wv, &t_Wv_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wp, &t_Wp_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wo, &t_Wo_compressed);
    } else {
    }
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) {
    return this->forward_common(t_inp, t_cache, use_cache);
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];
    auto t_pid = t_inp[2];
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];

    float scale = 1.0 / my_size;

    if (B * S / 64 > 4)
      large_cache_opt = true;
    else
      large_cache_opt = false;

    if (sparse_type == 0) {
      auto t_null = t_HS.new_empty({0});
      auto t_res = t_HS;
      t_HS = lyr_norm<T, LT>(t_HS, t_G, t_B, eps);
      auto t_QL = qkv_gemm<T>(t_HS, t_Wq, t_null);
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);
      auto t_KL = qkv_gemm<T>(t_HS, t_Wk, t_null);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);
      auto t_VL = qkv_gemm<T>(t_HS, t_Wv, t_null);
      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);
      auto t_CL = outputs[0];
      auto t_SO = qkv_gemm<T>(t_CL, t_Wp, t_null);
      auto t_I = fc_gelu<T>(t_HS, t_Wi, t_Bi);
      auto t_Out = fc_add2_scale<T>(t_I, t_SO, t_res, t_Wo, t_Bo, scale);
      if (my_size > 1) {
        allreduce(t_Out);
      }
      outputs[0] = t_Out;
      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else if (sparse_type == 1) {
      auto t_null = t_HS.new_empty({0});
      auto t_res = t_HS;
      t_HS = lyr_norm<T, LT>(t_HS, t_G, t_B, eps);
      auto N_HS = sizes[0] * sizes[1];
      auto C_HS = sizes[2];
      auto bc_HS = C_HS / t_Wi_bcsc.sizes[1];
      auto bn_HS =
          (N_HS <= SPMM_PACKED_BLOCK_SIZE) ? N_HS : SPMM_PACKED_BLOCK_SIZE;
      auto t_HS_use = (spmm_use_flat_acts > 0)
          ? t_HS
          : act_tensor_trans_n2v_flat(N_HS, C_HS, bn_HS, bc_HS, t_HS);
      auto t_QL = qkv_gemm_sparse<T>(t_HS.sizes(), t_HS_use, t_Wq_bcsc, t_null);
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);
      auto t_KL = qkv_gemm_sparse<T>(t_HS.sizes(), t_HS_use, t_Wk_bcsc, t_null);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);
      auto t_VL = qkv_gemm_sparse<T>(t_HS.sizes(), t_HS_use, t_Wv_bcsc, t_null);
      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);
      auto t_CL = outputs[0];
      auto sizesCL = t_CL.sizes();
      auto N_CL = sizesCL[0] * sizesCL[1];
      auto C_CL = sizesCL[2];
      auto bc_CL = C_CL / t_Wp_bcsc.sizes[1];
      auto bn_CL =
          (N_CL <= SPMM_PACKED_BLOCK_SIZE) ? N_CL : SPMM_PACKED_BLOCK_SIZE;
      auto t_CL_use = (spmm_use_flat_acts > 0)
          ? t_CL
          : act_tensor_trans_n2v_flat(N_CL, C_CL, bn_CL, bc_CL, t_CL);
      auto t_SO = qkv_gemm_sparse<T>(t_CL.sizes(), t_CL_use, t_Wp_bcsc, t_null);
      auto t_I = fc_gelu_sparse<T>(t_HS.sizes(), t_HS_use, t_Wi_bcsc, t_Bi);
      auto sizesI = t_I.sizes();
      auto N_I = sizesI[0] * sizesI[1];
      auto C_I = sizesI[2];
      auto bc_I = C_I / t_Wo_bcsc.sizes[1];
      auto bn_I =
          (N_I <= SPMM_PACKED_BLOCK_SIZE) ? N_I : SPMM_PACKED_BLOCK_SIZE;
      auto t_I_use = (spmm_use_flat_acts > 0)
          ? t_I
          : act_tensor_trans_n2v_flat(N_I, C_I, bn_I, bc_I, t_I);
      auto t_Out =
          fc_add2_scale_sparse<T>(t_I_use, t_SO, t_res, t_Wo_bcsc, t_Bo, scale);
      if (my_size > 1) {
        allreduce(t_Out);
      }
      outputs[0] = t_Out;
      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else if (sparse_type == 2) {
      auto t_null = t_HS.new_empty({0});
      auto t_res = t_HS;
      t_HS = lyr_norm<T, LT>(t_HS, t_G, t_B, eps);
      auto t_QL = qkv_gemm_compressed<T>(t_HS, t_Wq_compressed, t_null);
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);
      auto t_KL = qkv_gemm_compressed<T>(t_HS, t_Wk_compressed, t_null);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);
      auto t_VL = qkv_gemm_compressed<T>(t_HS, t_Wv_compressed, t_null);
      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);
      auto t_CL = outputs[0];
      auto t_SO = qkv_gemm_compressed<T>(t_CL, t_Wp_compressed, t_null);
      auto t_I = fc_gelu_compressed<T>(t_HS, t_Wi_compressed, t_Bi);
      auto t_Out = fc_add2_scale_compressed<T>(
          t_I, t_SO, t_res, t_Wo_compressed, t_Bo, scale);
      if (my_size > 1) {
        allreduce(t_Out);
      }
      outputs[0] = t_Out;
      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else {
      TPP_ASSERT(false, "Should not reach here\n");
      return {};
    }
  }
};

struct OPTDecoderLayer : LLMBlock<OPTDecoderLayer> {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp; // wt and bias for attn
  at::Tensor t_Bq, t_Bk, t_Bv, t_Bp;
  at::Tensor t_Wi, t_Wo; // wt and bias for fc1 and fc2
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G1, t_B1; // Gamma and Beta for attention layernorm
  at::Tensor t_G2, t_B2; // Gamma and Beta for MLP layernorm
  float eps1, eps2;
  long N, H;
  bool do_layer_norm_before;

  OPTDecoderLayer(
      std::vector<at::Tensor> params,
      double eps1,
      double eps2,
      long H,
      bool do_layer_norm_before)
      : LLMBlock("opt_fwd", params[4], params[0]),
        eps1(eps1),
        eps2(eps2),
        H(H),
        do_layer_norm_before(do_layer_norm_before) {
    int i = 0;
    t_G1 = params[i++]; // ln_gamma, lnorm before attention
    t_B1 = params[i++]; // ln_beta
    t_G2 = params[i++]; // ln_gamma, lnorm before mlp
    t_B2 = params[i++]; // ln_beta

    t_Wq = params[i++]; // q_proj
    t_Bq = params[i++];
    t_Wk = params[i++]; // k_proj
    t_Bk = params[i++];
    t_Wv = params[i++]; // v_proj
    t_Bv = params[i++];
    t_Wp = params[i++]; // out_proj
    t_Bp = params[i++];

    t_Wi = params[i++]; // fc1
    t_Bi = params[i++];
    t_Wo = params[i++]; // fc2
    t_Bo = params[i++];

    N = t_Wq.size(0) * t_Wq.size(3) / H;
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " N=" << N << " H=" << H
                << std::endl;
    }
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) {
    return this->forward_common(t_inp, t_cache, use_cache);
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];

    float scale = 1.0 / my_size;

    if (B * S / 64 > 4)
      large_cache_opt = true;
    else
      large_cache_opt = false;

    auto t_null = t_HS.new_empty({0}); // at::Tensor().to(t_HS.dtype());

    auto t_res = t_HS;
    if (do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G1, t_B1, eps1);
    }

    auto t_QL = qkv_gemm<T>(t_HS, t_Wq, t_Bq);
    auto t_KL = qkv_gemm<T>(t_HS, t_Wk, t_Bk);
    auto t_VL = qkv_gemm<T>(t_HS, t_Wv, t_Bv);

    auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

    auto t_CL = outputs[0];

    t_HS = fc_add_scale<T>(t_CL, t_res, t_Wp, t_Bp, scale);
    if (my_size > 1) {
      allreduce(t_HS);
    }

    if (!do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G1, t_B1, eps1);
    }

    t_res = t_HS;

    if (do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G2, t_B2, eps2);
    }

    t_HS = fc_relu<T>(t_HS, t_Wi, t_Bi);
    t_HS = fc_add_scale<T>(t_HS, t_res, t_Wo, t_Bo, scale);

    if (my_size > 1) {
      allreduce(t_HS);
    }

    if (!do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G2, t_B2, eps2);
    }

    outputs[0] = t_HS;

    if (use_cache) {
      return outputs;
    } else {
      return {t_HS};
    }
  }
};

struct LlamaDecoderLayer : LLMBlock<LlamaDecoderLayer> {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wg, t_Wu, t_Wd;
  tensor_compressed_t t_Wq_compressed = {0}, t_Wk_compressed = {0},
                      t_Wv_compressed = {0}, t_Wp_compressed = {0},
                      t_Wg_compressed = {0}, t_Wu_compressed = {0},
                      t_Wd_compressed = {0};
  at::Tensor t_Gi, t_Gpa;
  at::Tensor t_EP; // embed_positions
  float eps;
  long Nq, Nkv, H;
  long max_positions, rotary_dim;

  LlamaDecoderLayer(
      std::vector<at::Tensor> params,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("llama_fwd", params[1], params[0]),
        eps(eps),
        H(H),
        max_positions(max_positions),
        rotary_dim(rotary_dim) {
    int i = 0;
    t_Gi = params[i++]; // input_ln_gamma

    t_Wq = params[i++]; // q_proj
    t_Wk = params[i++]; // k_proj
    t_Wv = params[i++]; // v_proj
    t_Wp = params[i++]; // out_proj

    t_Gpa = params[i++]; // post_attention_ln_gamma

    t_Wg = params[i++]; // fc_gate
    t_Wu = params[i++]; // fc_up
    t_Wd = params[i++]; // fc_down

    t_EP = params[i++]; // embed_positions

    if (sparse_type > 0) {
      update_sparse_copy();
    }

    Nq = t_Wq.size(0) * t_Wq.size(3) / H;
    Nkv = t_Wk.size(0) * t_Wk.size(3) / H;
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " Nq=" << Nq << " Nkv=" << Nkv
                << " H=" << H << std::endl;
      if (sparse_type > 0 && measure_sparsity > 0) {
        printf("########################\n");
        printf("Layer %d sparsity stats:\n", layer);
        printf(
            "Wq sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wq_compressed.nnz) /
                    (1.0 * t_Wq_compressed.n_dense_elts));
        printf(
            "Wk sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wk_compressed.nnz) /
                    (1.0 * t_Wk_compressed.n_dense_elts));
        printf(
            "Wv sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wv_compressed.nnz) /
                    (1.0 * t_Wv_compressed.n_dense_elts));
        printf(
            "Wp sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wp_compressed.nnz) /
                    (1.0 * t_Wp_compressed.n_dense_elts));
        printf(
            "Wg sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wg_compressed.nnz) /
                    (1.0 * t_Wg_compressed.n_dense_elts));
        printf(
            "Wu sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wu_compressed.nnz) /
                    (1.0 * t_Wu_compressed.n_dense_elts));
        printf(
            "Wd sparsity %.5g %%\n",
            100.0 -
                (100.0 * t_Wd_compressed.nnz) /
                    (1.0 * t_Wd_compressed.n_dense_elts));
        layer++;
        printf("########################\n");
      }
    }
  }

  ~LlamaDecoderLayer() {
    if (sparse_type == 2) {
#if 0
      libxsmm_free(t_Wg_compressed.data);
      libxsmm_free(t_Wg_compressed.column_offsets);
      libxsmm_free(t_Wg_compressed.bitmap);
      libxsmm_free(t_Wu_compressed.data);
      libxsmm_free(t_Wu_compressed.column_offsets);
      libxsmm_free(t_Wu_compressed.bitmap);
      libxsmm_free(t_Wd_compressed.data);
      libxsmm_free(t_Wd_compressed.column_offsets);
      libxsmm_free(t_Wd_compressed.bitmap);
      libxsmm_free(t_Wq_compressed.data);
      libxsmm_free(t_Wq_compressed.column_offsets);
      libxsmm_free(t_Wq_compressed.bitmap);
      libxsmm_free(t_Wk_compressed.data);
      libxsmm_free(t_Wk_compressed.column_offsets);
      libxsmm_free(t_Wk_compressed.bitmap);
      libxsmm_free(t_Wv_compressed.data);
      libxsmm_free(t_Wv_compressed.column_offsets);
      libxsmm_free(t_Wv_compressed.bitmap);
      libxsmm_free(t_Wp_compressed.data);
      libxsmm_free(t_Wp_compressed.column_offsets);
      libxsmm_free(t_Wp_compressed.bitmap);
#endif
    } else {
    }
  }

  void update_sparse_copy() {
    if (sparse_type == 2) {
      create_compressed_from_blocked_nkkn_weight(t_Wq, &t_Wq_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wk, &t_Wk_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wv, &t_Wv_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wg, &t_Wg_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wp, &t_Wp_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wu, &t_Wu_compressed);
      create_compressed_from_blocked_nkkn_weight(t_Wd, &t_Wd_compressed);
    } else {
    }
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) {
    return this->forward_common(t_inp, t_cache, use_cache);
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];
    auto t_pid = t_inp[2];
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];

    float scale = 1.0 / my_size;

    if (B * S / 64 > 4)
      large_cache_opt = true;
    else
      large_cache_opt = false;

    if (sparse_type == 0 || B * S > 4) {
      auto t_null = t_HS.new_empty({0});
      auto t_res = t_HS;
      t_HS = llama_rms_norm<T, LT>(t_HS, t_Gi, eps);
      auto t_QL = qkv_gemm<T>(t_HS, t_Wq, t_null);
      apply_rotary_pos_emb_llama<T>(t_QL, t_EP, t_pid, Nq, H);
      auto t_KL = qkv_gemm<T>(t_HS, t_Wk, t_null);
      apply_rotary_pos_emb_llama<T>(t_KL, t_EP, t_pid, Nkv, H);
      auto t_VL = qkv_gemm<T>(t_HS, t_Wv, t_null);
      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);
      auto t_CL = outputs[0];
      auto t_SO = fc_add_scale<T>(t_CL, t_res, t_Wp, t_null, scale);
      if (my_size > 1) {
        allreduce(t_SO);
      }
      t_res = t_SO;
      t_HS = llama_rms_norm<T, LT>(t_SO, t_Gpa, eps);
      auto t_I = fc_silu<T>(t_HS, t_Wg, t_null);
      t_I = fc_mul<T>(t_HS, t_I, t_Wu, t_null);
      auto t_Out = fc_add_scale<T>(t_I, t_res, t_Wd, t_null, scale);
      if (my_size > 1) {
        allreduce(t_Out);
      }
      outputs[0] = t_Out;
      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else if (sparse_type == 2) {
      auto t_null = t_HS.new_empty({0});
      auto t_res = t_HS;
      t_HS = llama_rms_norm<T, LT>(t_HS, t_Gi, eps);
      auto t_QL = qkv_gemm_compressed<T>(t_HS, t_Wq_compressed, t_null);
      apply_rotary_pos_emb_llama<T>(t_QL, t_EP, t_pid, Nq, H);
      auto t_KL = qkv_gemm_compressed<T>(t_HS, t_Wk_compressed, t_null);
      apply_rotary_pos_emb_llama<T>(t_KL, t_EP, t_pid, Nkv, H);
      auto t_VL = qkv_gemm_compressed<T>(t_HS, t_Wv_compressed, t_null);
      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);
      auto t_CL = outputs[0];
      auto t_SO = fc_add_scale_compressed<T>(
          t_CL, t_res, t_Wp_compressed, t_null, scale);
      if (my_size > 1) {
        allreduce(t_SO);
      }
      t_res = t_SO;
      t_HS = llama_rms_norm<T, LT>(t_SO, t_Gpa, eps);
      auto t_I = fc_silu_compressed<T>(t_HS, t_Wg_compressed, t_null);
      t_I = fc_mul_compressed<T>(t_HS, t_I, t_Wu_compressed, t_null);
      auto t_Out = fc_add_scale_compressed<T>(
          t_I, t_res, t_Wd_compressed, t_null, scale);
      if (my_size > 1) {
        allreduce(t_Out);
      }
      outputs[0] = t_Out;
      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else {
      TPP_ASSERT(false, "Should not reach here\n");
      return {};
    }
  }
};

static void apply_rotary_pos_emb_gptj_wrap(
    at::Tensor t_in,
    at::Tensor t_emb_pos,
    at::Tensor t_pos,
    long N,
    long H) {
  GlobalPass _gp(FWD);

  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    apply_rotary_pos_emb_gptj<float>(t_in, t_emb_pos, t_pos, N, H);
  } else if (dt == at::kBFloat16) {
    apply_rotary_pos_emb_gptj<bfloat16>(t_in, t_emb_pos, t_pos, N, H);
  } else if (dt == at::kBFloat8) {
    apply_rotary_pos_emb_gptj<bfloat8>(t_in, t_emb_pos, t_pos, N, H);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
}

static at::Tensor lyr_norm_wrap(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_beta,
    double eps) {
  GlobalPass _gp(FWD);
  auto dt = t_in.dtype();
  auto ldt = t_gamma.dtype();
  auto t_out = at::empty_like(t_in);

  if (dt == at::kFloat && ldt == at::kFloat) {
    lyr_norm<float, float>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
    lyr_norm<bfloat16, float>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
    lyr_norm<bfloat16, bfloat16>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
    lyr_norm<bfloat8, float>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (dt == at::kBFloat8 && ldt == at::kBFloat8) {
    lyr_norm<bfloat8, bfloat8>(t_in, t_gamma, t_beta, t_out, eps);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_plain_wrap(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    long parallel_dim,
    std::vector<long> split_sizes) {
  GlobalPass _gp(FWD);
  if (parallel_dim == 1) {
    // t_in = t_in.chunk(my_size, -1)[my_rank].contiguous();
    t_in = t_in.split(split_sizes, -1)[my_rank].contiguous();
  }

  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  // std::cout << "YYY " << t_out.dtype() << "  " << t_in.dtype() << std::endl;
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_plain<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_plain<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    fc_plain<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  if (my_size > 1) {
    if (parallel_dim == 0) {
      t_out = allgather(t_out, split_sizes);
    } else if (parallel_dim == 1) {
      allreduce(t_out);
    }
  }
  return t_out;
}

static at::Tensor fc_add2_scale_wrap_compressed(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias,
    double scale) {
  GlobalPass _gp(FWD);
  auto t_out = at::empty_like(t_in1);
  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    fc_add2_scale_compressed<float>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    fc_add2_scale_compressed<bfloat16>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat8) {
    // fc_add2_scale<bfloat8>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_add2_scale_wrap_sparse(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias,
    double scale) {
  GlobalPass _gp(FWD);
  auto t_out = at::empty_like(t_in1);
  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    fc_add2_scale_sparse<float>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    fc_add2_scale_sparse<bfloat16>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat8) {
    // fc_add2_scale<bfloat8>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_add2_scale_wrap(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    at::Tensor t_wt,
    at::Tensor t_bias,
    double scale) {
  GlobalPass _gp(FWD);
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_add2_scale<float>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    fc_add2_scale<bfloat16>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat8) {
    fc_add2_scale<bfloat8>(t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_gelu_wrap_sparse(
    at::Tensor t_in,
    const tensor_bcsc_t& t_wt,
    at::Tensor t_bias) {
  GlobalPass _gp(FWD);
  auto sizes = t_in.sizes().vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);

  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    fc_gelu_sparse<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_gelu_sparse<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    // fc_gelu<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_gelu_wrap_compressed(
    at::Tensor t_in,
    const tensor_compressed_t& t_wt,
    at::Tensor t_bias) {
  GlobalPass _gp(FWD);
  auto sizes = t_in.sizes().vec();
  sizes[2] = t_wt.sizes[0] * t_wt.sizes[3];
  auto t_out = t_in.new_empty(sizes);

  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    fc_gelu_compressed<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_gelu_compressed<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    // fc_gelu<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

static at::Tensor fc_gelu_wrap(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  GlobalPass _gp(FWD);
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_gelu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_gelu<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    fc_gelu<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

REGISTER_SUBMODULE(_fused_llm_infer, m) {
  m.def("layer_norm", &lyr_norm_wrap, "TPP layer norm");
  m.def("fc_gelu", &fc_gelu_wrap, "TPP fc_gelu");
  m.def("fc_add2_scale", &fc_add2_scale_wrap, "TPP fc_add2_scale");
  m.def("fc_gelu_sparse", &fc_gelu_wrap_sparse, "TPP fc_gelu_sparse");
  m.def(
      "fc_add2_scale_sparse",
      &fc_add2_scale_wrap_sparse,
      "TPP fc_add2_scale_sparse");
  m.def(
      "fc_gelu_compressed", &fc_gelu_wrap_compressed, "TPP fc_gelu_compressed");
  m.def(
      "fc_add2_scale_compressed",
      &fc_add2_scale_wrap_compressed,
      "TPP fc_add2_compressed");
  m.def("fc_plain", &fc_plain_wrap, "TPP fc_plain");
  m.def("set_pg", &set_pg);
  m.def("get_batch_dim_in_kv_cache", &get_batch_dim_in_kv_cache);
  m.def(
      "apply_rotary_pos_emb_gptj",
      &apply_rotary_pos_emb_gptj_wrap,
      "TPP apply_rotary_pos_emb_gptj");
  py::class_<GPTJBlock>(m, "GPTJBlock")
      .def(py::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  py::class_<OPTDecoderLayer>(m, "OPTDecoderLayer")
      .def(py::init<std::vector<at::Tensor>, double, double, long, bool>())
      .def("forward", &OPTDecoderLayer::forward);
  py::class_<LlamaDecoderLayer>(m, "LlamaDecoderLayer")
      .def(py::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &LlamaDecoderLayer::forward);
}

TORCH_LIBRARY(tpp_llm, m) {
  m.def("layer_norm", &lyr_norm_wrap);
  // m.def("fc_gelu", &fc_gelu_wrap);
  // m.def("fc_add2_scale", &fc_add2_scale_wrap);
  m.def("fc_plain", &fc_plain_wrap);
  m.def("set_pg", &set_pg);
  m.def("get_batch_dim_in_kv_cache", &get_batch_dim_in_kv_cache);
  m.class_<GPTJBlock>("GPTJBlock")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &GPTJBlock::forward)
      .def("update_sparse_copy", &GPTJBlock::update_sparse_copy);
  m.class_<OPTDecoderLayer>("OPTDecoderLayer")
      .def(torch::init<std::vector<at::Tensor>, double, double, long, bool>())
      .def("forward", &OPTDecoderLayer::forward);
  m.class_<LlamaDecoderLayer>("LlamaDecoderLayer")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &LlamaDecoderLayer::forward)
      .def("update_sparse_copy", &LlamaDecoderLayer::update_sparse_copy);
}
