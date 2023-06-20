/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
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
#include "timing.h"
#include "xsmm_functors.h"
#include <torch/csrc/distributed/c10d/comm.hpp>

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();
static int my_size = 1;
static int large_cache_opt = false;
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static int SK_BLOCK_SIZE = env2int("SK_BLOCK_SIZE", 64);
static const char *GEMM_LOOP_SCHEME = getenv("GEMM_LOOP_SCHEME") ? getenv("GEMM_LOOP_SCHEME") : "aCB";

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(pln_gemm, "pln_gemm");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(ac_gemm, "ac_gemm");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");
REGISTER_LOCAL_SCOPE(lnorm, "lnorm");
REGISTER_LOCAL_SCOPE(rotary, "rotary");
REGISTER_LOCAL_SCOPE(reorder, "rorder");
REGISTER_LOCAL_SCOPE(allred, "allred");
REGISTER_LOCAL_SCOPE(concat, "concat");
REGISTER_LOCAL_SCOPE(fftkn, "fftkn");
REGISTER_LOCAL_SCOPE(k_trans, "k_trans");
REGISTER_LOCAL_SCOPE(attn1, "attn");

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
  std::vector<at::Tensor> temp_vec = {t_in};
  process_group->allreduce(temp_vec)->wait();
}

inline at::Tensor allgather(at::Tensor t_in) {
  RECORD_SCOPE(allred, {t_in});
  if (!process_group) {
    printf("Missing process group when using model parallel, use set_pg()\n");
    exit(1);
  }
  std::vector<std::vector<at::Tensor>> ag_vec(1);
  for (int i = 0; i < my_size; i++) {
    c10::InferenceMode guard(false);
    ag_vec[0].push_back(at::empty_like(t_in));
  }
  std::vector<at::Tensor> temp_vec = {t_in};
  process_group->allgather(ag_vec, temp_vec)->wait();
  auto t_out = at::cat(ag_vec[0], -1);
  return t_out;
}

template <typename T>
inline at::Tensor kv_concat(at::Tensor t_in1, at::Tensor t_in2, int dim, at::Tensor t_beam_idx) {
  auto ndim =  t_in1.dim();
  dim = dim >= 0 ? dim : dim+ndim;

  auto out_sizes = t_in1.sizes().vec();
  out_sizes[dim] += t_in2.size(dim);
  auto t_out = t_in1.new_empty(out_sizes);

  auto B = out_sizes[0];
  auto N = out_sizes[1];
  auto S = out_sizes[2];
  auto F = out_sizes[3];
  auto BNS = B * N * S;
  auto S1 = t_in1.size(dim);
  auto S2 = t_in2.size(dim);
  bool indirect = t_beam_idx.numel() > 0;

  //auto cpy_tpp = SCOPEIT(CpyTPP<T>(F), EW_COPY);
  auto cpy_tpp = CpyTPP<T>(F);

  auto in1 = GetVLAPtr<T>(t_in1, {N, S1, F});
  auto in2 = GetVLAPtr<T>(t_in2, {N, S2, F});
  auto out = GetVLAPtr<T>(t_out, {F});
  auto beam_idx = GetVLAPtr<long>(t_beam_idx);
  //std::cout << "t_beam_idx.dtype: " << t_beam_idx.dtype() << std::endl;
  //auto beam_idx = (long*)t_beam_idx.data_ptr();
  T *ptrs[BNS];
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
    RECORD_SCOPE(concat, {t_in1, t_in2});
#pragma omp parallel for
    for (int i = 0; i < BNS; i++) {
      cpy_tpp(ptrs[i], out[i]);
    }
  }
  return t_out;
}

template <typename T>
inline void apply_rotary_pos_emb(
    at::Tensor t_in,
    at::Tensor t_emb_pos,
    at::Tensor t_pos,
    long N,
    long H) {
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
    RECORD_SCOPE(rotary, {t_in, t_emb_pos, t_pos});

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
            float in0 = in[b][s][n][h];
            float in1 = in[b][s][n][h + 1];
            int p = pos[b][s];
	    if (p >= MP) continue;
	    //TPP_ASSERT(p < MP, "Invalid idx: %d (max %ld)\n", p, MP);
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

template <typename T, typename LT = T>
inline void lyr_norm(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_beta,
    at::Tensor t_out,
    float eps) {
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
    RECORD_SCOPE(lnorm, {t_in, t_gamma, t_beta});

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
inline at::Tensor llama_rms_norm(
    at::Tensor t_in,
    at::Tensor t_wt,
    float eps) {

  auto orig_dt = t_in.dtype();
  auto t_var = t_in.to(at::kFloat).pow(2).mean(-1, true);
  auto t_tmp = t_in * at::rsqrt(t_var + eps);
  auto ret = t_wt * t_tmp;
  ret = ret.to(orig_dt);

  return ret;
}

template <typename T>
inline void fc_plain(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
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

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(pln_gemm, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
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

template<typename T>
inline at::Tensor wt_tensor_for_first_token(at::Tensor t) {
  RECORD_SCOPE(fftkn, {t});
  auto dim = t.dim();
  if (dim < 5) return t;
  auto sizes = t.sizes();
  constexpr long RBS = 2;
  auto K1 = sizes[0];
  if (K1 % RBS != 0) return t;
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = sizes[4];
#if 0
  auto t_new = t.view({K1/RBS, RBS, C1, C2, K2, C3}).permute({0, 2, 3, 1, 4, 5}).contiguous().view({K1/RBS, C1, C2, RBS*K2, C3});
#else
  auto t_new = t.new_empty({K1/RBS, C1, C2, RBS*K2, C3});
  auto in = GetVLAPtr<T>(t, {RBS, C1, C2, K2 * C3});
  auto out = GetVLAPtr<T>(t_new, {C1, C2, RBS, K2 * C3});

#if 1
  auto cpy_tpp = SCOPEIT(CpyTPP<T>(C2, K2*C3, K2*C3, RBS*K2*C3), EW_COPY);

#pragma omp parallel for collapse (2)
  for (int i = 0; i < K1/RBS; i++) {
    for (int j = 0; j < C1; j++) {
      for (int k = 0; k < RBS; k++) {
        cpy_tpp(in[i][k][j][0], out[i][j][0][k]);
      }
    }
  }
#else
  auto cpy_tpp = SCOPEIT(CpyTPP<T>(RBS, K2*C3, C1*C2*K2*C3, K2*C3), EW_COPY);

#pragma omp parallel for collapse (2)
  for (int i = 0; i < K1/RBS; i++) {
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
inline void fc_add_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(o_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
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
inline void fc_add2_scale(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_in2,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(o_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
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
inline void fc_gelu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline at::Tensor fc_gelu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_gelu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_silu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline at::Tensor fc_silu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_silu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T>
inline void fc_relu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
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
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline at::Tensor fc_relu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  fc_relu<T>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tout = T>
inline void qkv_gemm(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias, at::Tensor t_out) {
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
  if (large_cache_opt) Ncb = NCB_BLOCK_SIZE;

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
    RECORD_SCOPE(qkv_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto gemm_loop = ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
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
            brgemm_tpp_rem(
                in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline at::Tensor qkv_gemm(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];
  auto t_out = t_in.new_empty(sizes);
  qkv_gemm<T, Tout>(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename T, typename Tv>
struct AttnKernels
{

  SCOPEIT_DECL(BrgemmTPP<T, float>) a_gemm_tpp;
  SCOPEIT_DECL(ScaleTPP<float, float>) scale_tpp;
  SCOPEIT_DECL(AddBiasTPP<T>) add_mask_tpp;
  SCOPEIT_DECL(AddTPP<T,float,float>) add_2dmask_tpp;
  SCOPEIT_DECL(VarSoftMaxFwdTPP<float, Tv>) softmax_fwd_tpp;
  SCOPEIT_DECL(BrgemmTPP<Tv, Tv>) c_gemm_tpp;
  SCOPEIT_DECL(ConvertTPP<Tv, T>) cvt_tpp;
  SCOPEIT_DECL(XformExtTPP<T>) xform_tpp;
  SCOPEIT_DECL(XformExtTPP<T>) vnni_tpp;
  SCOPEIT_DECL(SoftMaxFixUpTPP<T>) softmax_fixup;

  AttnKernels(long Sqb, long Skb, long H, int pad, int kl_in_vnni, int vl_in_vnni) {
    //printf("Sqb: %ld, Skb: %ld, H: %ld, psd: %d, kl: %d, vl: %d\n", Sqb, Skb, H, pad, kl_in_vnni, vl_in_vnni);
    if (Sqb == 0) Sqb = 1; // hack for unused kernels to not generate 0 size kernels
    if (Skb == 0) Skb = 2; // hack for unused kernels to not generate 0 size kernels
    // [Sqb, H] * [H, Skb] = [Sqb, Skb]
    a_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
            Sqb, Skb, H, H, H * Skb, H, Skb, Skb, 0.0, 0, 1, kl_in_vnni)));
    // [Sqb, Skb]
    scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sqb * Skb)), EW_SCL);
    add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sqb, Skb), EW_ADD);
    add_2dmask_tpp = SCOPEIT((AddTPP<T,float,float>(Sqb, Skb)), EW_ADD);
    softmax_fwd_tpp =
      SCOPEIT((VarSoftMaxFwdTPP<float, Tv>(Sqb, Skb)), SOFTMAX);
    softmax_fixup =
      SCOPEIT((SoftMaxFixUpTPP<T>(Sqb, H)), EW_RCP);
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
    xform_tpp =
      SCOPEIT(XformExtTPP<T>(Skb-pad, H, H, Skb, H, Skb, xform, true), XPOSE);
    if (vl_in_vnni != 0)
      vnni_tpp =
        SCOPEIT(XformExtTPP<T>(Skb-pad, H, Skb, H, H, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  }
};

template <typename T, typename Tv>
inline at::Tensor attn(
    at::Tensor t_QL,
    at::Tensor t_KL,
    at::Tensor t_AM,
    at::Tensor t_VL) {
  RECORD_SCOPE(attn1, {t_QL, t_KL});
  auto t_CL = at::empty_like(t_QL);
  auto sizes = t_QL.sizes();
  long B = sizes[0];
  long N = sizes[1];
  long Sq = sizes[2];
  long H = sizes[3];
  float one_by_sqrt_H = 1.0 / sqrtf(H);
  auto ksizes = t_KL.sizes();
  long Sk = ksizes[2];
  long offset = Sk - Sq;
  constexpr long Sqb = 64;
  long qrem = Sq % Sqb;
  bool inline_trans = ((Sq+Sqb-1) / Sqb == 1);
  bool am_is_2d = t_AM.size(2) != 1;

  int vl_in_vnni = 1; //(Sk % 2 == 0 ? 1 : 0);
  const long VBS = (vl_in_vnni ? get_vnni_block_size<T>() : 1);
  long Sk_pad = (Sk + VBS - 1) & ~(VBS - 1);
  const long Skb = (!inline_trans ? 512 : SK_BLOCK_SIZE); 
  long krem = Sk % Skb;
  int pad = Sk_pad - Sk;

  auto t_KL_TV = t_KL.new_empty({B, N, Sk_pad, H});
  auto t_VL_V = t_VL;
  if (VBS != 1) {
    t_VL_V = t_VL.new_empty({B, N, Sk_pad, H});
  }
  if (Sk != Sk_pad) {
    TPP_ASSERT(am_is_2d == false, "2D AM not supported yet\n");
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
  auto QL = GetVLAPtr<T>(t_QL, {N, Sq, H});
  auto KL = GetVLAPtr<T>(t_KL, {N, Sk, H});
  auto KL_TV = GetVLAPtr<T>(t_KL_TV, {N, Sk_pad, H});
  auto VL = GetVLAPtr<Tv>(t_VL, {N, Sk, H});
  auto VL_V = GetVLAPtr<Tv>(t_VL_V, {N, Sk_pad, H});
  auto CL = GetVLAPtr<T>(t_CL, {N, Sq, H});
  auto AM = GetVLAPtr<T>(t_AM, {Sk_pad});
  auto AM2 = GetVLAPtr<T>(t_AM, {Sq, Sk_pad});
  int kl_in_vnni = 1;


  AttnKernels<T, Tv> attn_kern[4] =  {
    AttnKernels<T,Tv>(Sqb, Skb, H, 0, kl_in_vnni, vl_in_vnni),
    AttnKernels<T,Tv>(Sqb, krem+pad, H, pad, kl_in_vnni, vl_in_vnni),
    AttnKernels<T,Tv>(qrem, Skb, H, 0, kl_in_vnni, vl_in_vnni),
    AttnKernels<T,Tv>(qrem, krem+pad, H, pad, kl_in_vnni, vl_in_vnni),
  };

  if (!inline_trans) {
    RECORD_SCOPE(k_trans, {t_QL, t_KL});
#pragma omp parallel for collapse(3)
    for (int n = 0; n < N; n++) {
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
    RECORD_SCOPE(ac_gemm, {t_QL, t_KL});
    {
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int sq = 0; sq < Sq; sq += Sqb) {
            long qbs = (Sq - sq >= Sqb ? Sqb : Sq - sq);
            int qid = (sq + Sqb > Sq) ? 1 : 0;
            float omax[qbs], osum[qbs], cmax[qbs], csum[qbs];
            for (int sk = 0; sk < Sk; sk += Skb) {
              long kbs = (Sk - sk >= Skb ? Skb : Sk_pad - sk);
              int kid = qid * 2 + ((sk + Skb > Sk) ? 1 : 0);
              auto &ak = attn_kern[kid];
              float AS[qbs][kbs];
              Tv AST[qbs][kbs];
              T*k_ptr = KL_TV[b][n][sk];
              T k_tmp[kbs*H];
              if (inline_trans) {
                //ak.xform_tpp(KL[b][n][sk], KL_TV[b][n][sk]);
                ak.xform_tpp(KL[b][n][sk], k_tmp);
                k_ptr = k_tmp;
              }
              ak.a_gemm_tpp(QL[b][n][sq], k_ptr, AS[0], 1);
              for (int sq1 = 0; sq1 < qbs; sq1++) {
                auto qval = sq + sq1 + offset;
                for (int sk1 = qval + 1; sk1 < sk+kbs; sk1++) {
                  AS[sq1][sk1-sk] = -1e9;
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
              Tv tmp[qbs * H];
              Tv *v_ptr = VL_V[b][n][sk];
              Tv v_tmp[kbs*H];
              if (inline_trans && VBS != 1) {
                //ak.vnni_tpp(VL[b][n][sk], VL_V[b][n][sk]);
                ak.vnni_tpp(VL[b][n][sk], v_tmp);
                v_ptr = v_tmp;
              }
              ak.c_gemm_tpp(AST[0], v_ptr, tmp, 1);
              if (sk == 0) {
                ak.cvt_tpp(tmp, CL[b][n][sq]);
              } else {
                ak.softmax_fixup(tmp, CL[b][n][sq], cmax, csum, omax, osum);
              }
            }
          }
        }
      }
    }
  }
  return t_CL;
}


struct GPTJBlock : torch::CustomClassHolder {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wi, t_Wo;
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G, t_B;
  at::Tensor t_EP; // embed_positions
  float eps, one_by_sqrt_H;
  long N, H;
  long mp_size; // model_paralle size
  long max_positions, rotary_dim;

  GPTJBlock(
      std::vector<at::Tensor> params,
      double eps,
      long N,
      long H,
      long max_positions,
      long rotary_dim)
      : eps(eps),
        N(N),
        H(H),
        max_positions(max_positions),
        rotary_dim(rotary_dim) {
    int i = 0;
    t_G =  params[i++]; // ln_gamma
    t_B =  params[i++]; // ln_beta

    t_Wq = params[i++]; // q_proj
    t_Wk = params[i++]; // k_proj
    t_Wv = params[i++]; // v_proj
    t_Wp = params[i++]; // out_proj

    t_Wi = params[i++]; // fc_in
    t_Bi = params[i++];

    t_Wo = params[i++]; // fc_out
    t_Bo = params[i++];

    t_EP = params[i++]; // embed_positions

    one_by_sqrt_H = 1.0 / sqrt(H);
    mp_size = t_Wp.size(0) / t_Wq.size(0);
    if (my_rank == 0) {
      std::cout << "mp_size=" << mp_size << " N=" << N << " H=" << H << std::endl;
    }
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      at::Tensor t_HS,
      at::Tensor t_key_past,
      at::Tensor t_value_past,
      at::Tensor t_beam_idx,
      at::Tensor t_am,
      at::Tensor t_pid,
      bool use_cache) {
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];

    float scale = 1.0 / mp_size;

    if (B*S / 64 > 4)
      large_cache_opt = true;
    else
      large_cache_opt = false;

    auto t_null = t_HS.new_empty({0});
    auto t_res = t_HS;
    t_HS = lyr_norm<T, LT>(t_HS, t_G, t_B, eps);
    auto t_KL = qkv_gemm<T>(t_HS, t_Wk, t_null);
    apply_rotary_pos_emb<T>(t_KL, t_EP, t_pid, N, H);
    t_KL = t_KL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();
    if (t_key_past.numel() > 0) {
      t_KL = kv_concat<T>(t_key_past, t_KL, 2, t_beam_idx);
    }
    auto t_VL = qkv_gemm<T>(t_HS, t_Wv, t_null);
    t_VL = t_VL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();
    if (t_value_past.numel() > 0) {
      t_VL = kv_concat<T>(t_value_past, t_VL, 2, t_beam_idx);
    }

    auto t_QL = qkv_gemm<T>(t_HS, t_Wq, t_null);
    apply_rotary_pos_emb<T>(t_QL, t_EP, t_pid, N, H);
    t_QL = t_QL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();

    auto t_CL = attn<T, T>(t_QL, t_KL, t_am, t_VL);
    t_CL = t_CL.view({B, N, S, H}).permute({0, 2, 1, 3}).contiguous().view({B, S, N * H});
    auto t_SO = qkv_gemm<T>(t_CL, t_Wp, t_null);
    auto t_I = fc_gelu<T>(t_HS, t_Wi, t_Bi);
    auto t_Out = fc_add2_scale<T>(t_I, t_SO, t_res, t_Wo, t_Bo, scale);
    if (mp_size > 1) {
      allreduce(t_Out);
    }

    if (use_cache) {
      return {t_Out, t_KL, t_VL};
    } else {
      return {t_Out, t_null, t_null};
    }
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      bool use_cache) {
    GlobalPass _gp(FWD);
    RECORD_FUNCTION("gptj_fwd", std::vector<c10::IValue>());
    auto t_HS = t_inp[0];
    auto t_key_past = t_inp[1];
    auto t_value_past = t_inp[2];
    auto t_beam_idx = t_inp[3];
    auto t_AM = t_inp[4];
    auto t_pid = t_inp[5];
    auto dt = this->t_Wq.dtype();
    auto ldt = this->t_G.dtype();
    std::vector<at::Tensor> ret;

    // printf("Layer %d\n", i++);
    if (dt == at::kFloat && ldt == at::kFloat) {
      ret = this->_forward<float, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          t_pid,
          use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
      ret = this->_forward<bfloat16, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          t_pid,
          use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat16, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          t_pid,
          use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
      ret = this->_forward<bfloat8, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          t_pid,
          use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat8, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          t_pid,
          use_cache);
    } else {
      std::cout << "Types: " << dt << "  " << ldt << std::endl;
      TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
    }
    // printf("Returning Layer \n");

    return ret;
  }
};

struct OPTDecoderLayer : torch::CustomClassHolder {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp; // wt and bias for attn
  at::Tensor t_Bq, t_Bk, t_Bv, t_Bp;
  at::Tensor t_Wi, t_Wo; // wt and bias for fc1 and fc2
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G1, t_B1; // Gamma and Beta for attention layernorm
  at::Tensor t_G2, t_B2; // Gamma and Beta for MLP layernorm 
  float eps1, eps2;
  long N, H;
  long mp_size; // model_paralle size

  OPTDecoderLayer(
      std::vector<at::Tensor> params,
      double eps1,
      double eps2,
      long N,
      long H)
      : eps1(eps1),
        eps2(eps2),
        N(N),
        H(H) {
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

    mp_size = t_Wp.size(0) / t_Wq.size(0);
    if (my_rank == 0) {
      std::cout << "mp_size=" << mp_size << " N=" << N << " H=" << H << std::endl;
    }
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      at::Tensor& t_HS,
      at::Tensor& t_key_past,
      at::Tensor& t_value_past,
      at::Tensor t_beam_idx,
      at::Tensor& t_am,
      bool do_layer_norm_before,
      bool use_cache) {
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];

    float scale = 1.0 / mp_size;

    if (B*S / 64 > 4)
      large_cache_opt = true;
    else
      large_cache_opt = false;

    auto t_null = t_HS.new_empty({0}); // at::Tensor().to(t_HS.dtype());

		auto t_res = t_HS;
    if (do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G1, t_B1, eps1);
    }

    auto t_KL = qkv_gemm<T>(t_HS, t_Wk, t_Bk);
    t_KL = t_KL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();
    if (t_key_past.numel() > 0) {
      t_KL = kv_concat<T>(t_key_past, t_KL, 2, t_beam_idx);
    }
    auto t_VL = qkv_gemm<T>(t_HS, t_Wv, t_Bv);
    t_VL = t_VL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();
    if (t_value_past.numel() > 0) {
      t_VL = kv_concat<T>(t_value_past, t_VL, 2, t_beam_idx);
    }

    auto t_QL = qkv_gemm<T>(t_HS, t_Wq, t_Bq);
    t_QL = t_QL.view({B, S, N, H}).permute({0, 2, 1, 3}).contiguous();

    auto t_CL = attn<T, T>(t_QL, t_KL, t_am, t_VL);
    t_CL = t_CL.view({B, N, S, H}).permute({0, 2, 1, 3}).contiguous().view({B, S, N * H});

    t_HS = fc_add_scale<T>(t_CL, t_res, t_Wp, t_Bp, scale);
    if (mp_size > 1) {
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

    if (mp_size > 1) {
      allreduce(t_HS);
    }

    if (!do_layer_norm_before) {
      t_HS = lyr_norm<T, LT>(t_HS, t_G2, t_B2, eps2);
    }

    if (use_cache) {
      return {t_HS, t_KL, t_VL};
    } else {
      return {t_HS, t_null, t_null};
    }
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      bool do_layer_norm_before,
      bool use_cache) {
    GlobalPass _gp(FWD);
    RECORD_FUNCTION("opt_fwd", std::vector<c10::IValue>());
    auto t_HS = t_inp[0];
    auto t_key_past = t_inp[1];
    auto t_value_past = t_inp[2];
    auto t_beam_idx = t_inp[3];
    auto t_AM = t_inp[4];
    auto dt = this->t_Wq.dtype();
    auto ldt = this->t_G1.dtype();
    std::vector<at::Tensor> ret;

    // printf("Layer %d\n", i++);
    if (dt == at::kFloat && ldt == at::kFloat) {
      ret = this->_forward<float, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          do_layer_norm_before,
          use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
      ret = this->_forward<bfloat16, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          do_layer_norm_before,
          use_cache);
    } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat16, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          do_layer_norm_before,
          use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
      ret = this->_forward<bfloat8, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          do_layer_norm_before,
          use_cache);
    } else if (dt == at::kBFloat8 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat8, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_beam_idx,
          t_AM,
          do_layer_norm_before,
          use_cache);
    } else {
      std::cout << "Types: " << dt << "  " << ldt << std::endl;
      TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
    }

    return ret;
  }
};

static void apply_rotary_pos_emb_wrap(
    at::Tensor t_in,
    at::Tensor t_emb_pos,
    at::Tensor t_pos,
    long N,
    long H) {
  GlobalPass _gp(FWD);

  auto dt = t_in.dtype();
  if (dt == at::kFloat) {
    apply_rotary_pos_emb<float>(t_in, t_emb_pos, t_pos, N, H);
  } else if (dt == at::kBFloat16) {
    apply_rotary_pos_emb<bfloat16>(t_in, t_emb_pos, t_pos, N, H);
  } else if (dt == at::kBFloat8) {
    apply_rotary_pos_emb<bfloat8>(t_in, t_emb_pos, t_pos, N, H);
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
    long parallel_dim) {
  GlobalPass _gp(FWD);
  if (parallel_dim == 1) {
    t_in = t_in.chunk(my_size, -1)[my_rank].contiguous();
  }
	
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  //std::cout << "YYY " << t_out.dtype() << "  " << t_in.dtype() << std::endl;
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
      t_out = allgather(t_out);
    } else if (parallel_dim == 1) {
      allreduce(t_out);
    }
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
  m.def("fc_plain", &fc_plain_wrap, "TPP fc_plain");
  m.def("set_pg", &set_pg);
  m.def(
      "apply_rotary_pos_emb",
      &apply_rotary_pos_emb_wrap,
      "TPP apply_rotary_pos_emb");
  py::class_<GPTJBlock>(m, "GPTJBlock")
      .def(py::init<std::vector<at::Tensor>, double, long, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  py::class_<OPTDecoderLayer>(m, "OPTDecoderLayer")
      .def(py::init<std::vector<at::Tensor>, double, double, long, long>())
      .def("forward", &OPTDecoderLayer::forward);
}

TORCH_LIBRARY(tpp_llm, m) {
  m.def("layer_norm", &lyr_norm_wrap);
  m.def("fc_gelu", &fc_gelu_wrap);
  m.def("fc_add2_scale", &fc_add2_scale_wrap);
  m.def("fc_plain", &fc_plain_wrap);
  m.def("set_pg", &set_pg);
  m.class_<GPTJBlock>("GPTJBlock")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  m.class_<OPTDecoderLayer>("OPTDecoderLayer")
      .def(torch::init<std::vector<at::Tensor>, double, double, long, long>())
      .def("forward", &OPTDecoderLayer::forward);
}
