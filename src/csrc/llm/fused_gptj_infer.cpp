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

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();
static int large_cache_opt = false;
static int use_at_vnni = false; // env2int("USE_AT_VNNI");

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(ac_gemm, "ac_gemm");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");
REGISTER_LOCAL_SCOPE(lnorm, "lnorm");
REGISTER_LOCAL_SCOPE(rotary, "rotary");

template <typename T>
inline void apply_rotary_pos_emb(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
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

static void apply_rotary_pos_emb_wrap(
    at::Tensor t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
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

template <typename T, typename LT = T>
inline void lyr_norm(
    at::Tensor& t_in,
    at::Tensor& t_gamma,
    at::Tensor& t_beta,
    at::Tensor& t_out,
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

static at::Tensor lyr_norm_wrap(
    at::Tensor& t_in,
    at::Tensor& t_gamma,
    at::Tensor& t_beta,
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

template <typename T>
inline void fc_plain(
    at::Tensor t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
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
    RECORD_SCOPE(qkv_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? "acB" : "aCb";
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
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

static at::Tensor fc_plain_wrap(
    at::Tensor t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  GlobalPass _gp(FWD);
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
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
  return t_out;
}

template <typename T>
inline void fc_out(
    at::Tensor t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
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
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_SCOPE(o_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? "acB" : "aCb";
    auto ogemm_loop = ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], out[s1][nk]);
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
              add_tpp(out[s1][nk], in2[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              copy_bias_tpp_rem(bias[nk], out[s1][nk]);
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[s1][nk], in1[s1][nk], out[s1][nk]);
              add_tpp_rem(out[s1][nk], in2[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

static at::Tensor fc_out_wrap(
    at::Tensor t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  GlobalPass _gp(FWD);
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_out<float>(t_in, t_in1, t_in2, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_out<bfloat16>(t_in, t_in1, t_in2, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    fc_out<bfloat8>(t_in, t_in1, t_in2, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

template <typename T>
inline void fc_in(
    at::Tensor t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
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

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? "acB" : "aCb";
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], out[s1][nk]);
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              copy_bias_tpp_rem(bias[nk], out[s1][nk]);
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

static at::Tensor fc_in_wrap(
    at::Tensor t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  GlobalPass _gp(FWD);
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_in<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_in<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    fc_in<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

struct GPTJBlock : torch::CustomClassHolder {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wi, t_Wo;
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G, t_B;
  at::Tensor t_EP; // embed_positions
  float eps, one_by_sqrt_H;
  long K1, K2, C1, C2, N, H, H1, H2;
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

    one_by_sqrt_H = 1.0 / sqrt(H);
    auto sizes = t_Wq.sizes();
    // std::cout << "t_Wq.sizes" << sizes << std::endl;
    K1 = sizes[0];
    C1 = sizes[1];
    K2 = sizes[3];
    C2 = sizes.size() == 4 ? sizes[2] : sizes[2] * sizes[4];
    H2 = K2;
    H1 = H / H2;
    std::cout << "H2=" << H2 << " H=" << H << " H1=" << H1 << std::endl;
  }

  template <typename T, typename Tout = T>
  inline void qkv_gemm(at::Tensor t_in, at::Tensor& t_wt, at::Tensor& t_out) {
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
    auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

    auto Ncb = Nc;
    auto BSb = 64L;
    auto rem = BS % 64;

    auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
    auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
    auto brgemm_tpp = SCOPEITGEMM(
        (BrgemmTPP<T, Tout>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
    auto brgemm_tpp_rem = SCOPEITGEMM(
        (BrgemmTPP<T, Tout>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

    {
      RECORD_SCOPE(qkv_gemm, {t_in, t_wt_V});
      // auto loop_scheme = large_cache_opt ? "acB" : "aBC";
      auto loop_scheme = large_cache_opt ? "acB" : "aCb";
      auto gemm_loop = ThreadedLoop<3>(
          {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
            bool is_rem = (s1 + BSb > BS);
            if (!is_rem) {
              if (nc == 0) {
                zero_tpp(out[s1][nk]);
              }
              brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            } else {
              if (nc == 0) {
                zero_tpp_rem(out[s1][nk]);
              }
              brgemm_tpp_rem(
                  in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            }
          },
          [&]() { brgemm_tpp.config(); },
          [&]() { brgemm_tpp.release(); });
    }
  }

  template <typename T, typename Tv>
  inline void attn(
      at::Tensor& t_QL,
      at::Tensor& t_KL,
      at::Tensor& t_AM,
      at::Tensor& t_VL,
      at::Tensor& t_CL) {
    auto sizes = t_QL.sizes();
    long B = sizes[0];
    long Sq = sizes[1];
    auto ksizes = t_KL.sizes();
    long Sk = ksizes[1];
    long offset = Sk - Sq;
    // printf("B=%ld S1=%ld S2=%ld H1=%ld H2=%ld N=%ld\n", B, S1, S2, H1, H2,
    // N);
    // printf("B=%ld Sq1=%ld Sq2=%ld N=%ld H=%ld, H1=%ld H2=%ld Sk1=%ld Sk2=%ld
    // offset=%ld\n", B, Sq1, Sq2, N, H, H1, H2, Sk1, Sk2, offset);
    // printf("B=%ld Sq=%ld N=%ld H=%ld, H1=%ld H2=%ld Sk=%ld offset=%ld\n", B,
    // Sq, N, H, H1, H2, Sk, offset);
    auto t_KL_TV = at::empty_like(t_KL);
    auto QL = GetVLAPtr<T>(t_QL, {Sq, N, H1, H2});
    auto KL = GetVLAPtr<T>(t_KL, {Sk, N, H1, H2});
    auto KL_TV = GetVLAPtr<T>(t_KL_TV, {N, H1, H2, Sk});
    auto VL = GetVLAPtr<Tv>(t_VL, {Sk, N, H1, H2});
    auto CL = GetVLAPtr<T>(t_CL, {Sq, N, H1, H2});
    auto AM = GetVLAPtr<T>(t_AM, {Sk});
    auto a_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
        Sq, Sk, H2, H2, H2 * Sk, N * H, Sk, Sk, 0.0, 0, H1)));
    auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sq * Sk)), EW_SCL);
    auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sq, Sk), EW_ADD);
    auto softmax_fwd_tpp =
        SCOPEIT((SoftMaxFwdTPP<float, Tv>(1, Sq, Sk)), SOFTMAX);
    auto c_gemm_tpp = SCOPEITGEMM((BrgemmTPP<Tv, Tv>(
        Sq, H2, Sk, Sq * Sk, Sk * N * H, Sk, N * H, H2, 0.0, 0, 1)));
    auto cvt_tpp = SCOPEIT((ConvertTPP<Tv, T>(Sq, H2, H2, N * H)), EW_COPY);
    auto cpy_tpp = SCOPEIT(CpyTPP<T>(Sk, H2, N * H, H2), EW_COPY);
    auto xform = XformTPP::XFORM_XPOSE_TPP;
    if (!std::is_same<T, float>::value) {
      xform = XformTPP::XFORM_XPOSE_N2V_TPP;
    }
    auto xform_tpp =
        SCOPEIT(XformExtTPP<T>(Sk, H2, H2, Sk, N * H, Sk, xform, true), XPOSE);

    {
      RECORD_SCOPE(ac_gemm, {t_QL, t_KL});
      {
#pragma omp parallel for collapse(2)
        for (int n = 0; n < N; n++) {
          for (int b = 0; b < B; b++) {
            float AS[Sq][Sk];
            Tv AST[Sq][Sk];
            for (int h1 = 0; h1 < H1; h1++) {
              // T tmp[Sk * H2];
              // cpy_tpp(KL[b][0][n][h1], tmp);
              xform_tpp(KL[b][0][n][h1], KL_TV[b][n][h1][0]);
            }
            a_gemm_tpp(QL[b][0][n][0], KL_TV[b][n][0][0], AS[0], H1);
            for (int sq = 0; sq < Sq; sq++) {
              auto qval = sq + offset;
              for (int sk = qval + 1; sk < Sk; sk++) {
                AS[sq][sk] = -1e9;
              }
            }
            scale_tpp(AS[0], AS[0], one_by_sqrt_H);
            if (t_AM.numel() != 0)
              add_mask_tpp(AM[b], AS[0]);
            softmax_fwd_tpp(AS[0], AST[0]);
            for (int h1 = 0; h1 < H1; h1++) {
              Tv tmp[Sq * H2];
              c_gemm_tpp(AST[0], VL[b][0][n][h1], tmp, 1);
              cvt_tpp(tmp, CL[b][0][n][h1]);
            }
          }
        }
      }
    }
  }

  template <typename T, typename LT = T>
  std::vector<at::Tensor> _forward(
      at::Tensor& t_HS,
      at::Tensor& t_key_past,
      at::Tensor& t_value_past,
      at::Tensor& t_am,
      at::Tensor& t_pid,
      at::Tensor t_Out,
      bool use_cache,
      std::vector<at::Tensor> t_scratch) {
    auto sizes = t_HS.sizes();
    auto B = sizes[0];
    auto S = sizes[1];
    auto t_QL = at::empty_like(t_HS); // t_scratch[0];
    auto t_KL = at::empty_like(t_QL); // t_scratch[0];
    auto t_VL = at::empty_like(t_QL, at::kFloat); // t_scratch[0];
    // auto t_KL_TV = at::empty_like(t_KL); // t_scratch[1];
    // auto t_VL_V = at::empty_like(t_VL); // t_scratch[2];
    auto t_CL = at::empty_like(t_QL); // t_scratch[3];
    auto t_SO = at::empty_like(t_HS); // t_scratch[4];
    auto t_I = t_HS.new_empty({B, S, t_Wi.size(0) * t_Wi.size(3)});
    // auto t_I = at::empty_like(t_HS); // t_scratch[5];
    auto t_HS_qkv = at::empty_like(t_HS);

    // std::cout << "HS: " << t_HS.sizes() << std::endl;
    // std::cout << "use_cche: " << use_cache << " t_key_past.numel: " <<
    // t_key_past.numel() << std::endl;
    auto t_null = t_HS.new_empty({0}); // at::Tensor().to(t_HS.dtype());
    lyr_norm<T, LT>(t_HS, t_G, t_B, t_HS_qkv, eps);
    // std::cout << "HS_qkv: " << t_HS_qkv.sizes() << std::endl;
    qkv_gemm<T>(t_HS_qkv, t_Wk, t_KL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    apply_rotary_pos_emb<T>(t_KL, t_EP, t_pid, N, H);
    if (t_key_past.numel() > 0) {
      // std::cout << "t_key_past: " << t_key_past.sizes() << std::endl;
      // std::cout << "t_KL: " << t_KL.sizes() << std::endl;
      t_KL = at::cat({t_key_past, t_KL}, -2);
    }
    // printf("reached at %s:%d\n", __func__, __LINE__);
    // std::cout << t_KL.sizes() << std::endl;
    qkv_gemm<T, float>(t_HS_qkv, t_Wv, t_VL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    if (t_value_past.numel() > 0) {
      t_VL = at::cat({t_value_past, t_VL}, -2);
    }
    // printf("reached at %s:%d\n", __func__, __LINE__);
    qkv_gemm<T>(t_HS_qkv, t_Wq, t_QL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    apply_rotary_pos_emb<T>(t_QL, t_EP, t_pid, N, H);

    attn<T, float>(t_QL, t_KL, t_am, t_VL, t_CL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    qkv_gemm<T>(t_CL, t_Wp, t_SO);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    fc_in<T>(t_HS_qkv, t_Wi, t_Bi, t_I);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    fc_out<T>(t_I, t_SO, t_HS, t_Wo, t_Bo, t_Out);
    // printf("reached at %s:%d\n", __func__, __LINE__);

    if (use_cache) {
      return {t_Out, t_KL, t_VL};
    } else {
      return {t_Out, t_null, t_null};
    }
  }
  std::vector<at::Tensor> get_scratch(at::Tensor& t_HS) {
    std::vector<at::Tensor> ret;
    return ret;
  }

  std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      bool use_cache) {
    GlobalPass _gp(FWD);
    auto t_HS = t_inp[0];
    auto t_key_past = t_inp[1];
    auto t_value_past = t_inp[2];
    auto t_AM = t_inp[3];
    auto t_pid = t_inp[4];
    auto t_scratch = get_scratch(t_HS);
    auto dt = this->t_Wq.dtype();
    auto ldt = this->t_G.dtype();
    std::vector<at::Tensor> ret;
    auto t_Out = at::empty_like(t_HS);

    // printf("Layer %d\n", i++);
    if (dt == at::kFloat && ldt == at::kFloat) {
      ret = this->_forward<float, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_AM,
          t_pid,
          t_Out,
          use_cache,
          t_scratch);
    } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
      ret = this->_forward<bfloat16, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_AM,
          t_pid,
          t_Out,
          use_cache,
          t_scratch);
    } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat16, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_AM,
          t_pid,
          t_Out,
          use_cache,
          t_scratch);
    } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
      ret = this->_forward<bfloat8, float>(
          t_HS,
          t_key_past,
          t_value_past,
          t_AM,
          t_pid,
          t_Out,
          use_cache,
          t_scratch);
    } else if (dt == at::kBFloat8 && ldt == at::kBFloat16) {
      ret = this->_forward<bfloat8, bfloat16>(
          t_HS,
          t_key_past,
          t_value_past,
          t_AM,
          t_pid,
          t_Out,
          use_cache,
          t_scratch);
    } else {
      std::cout << "Types: " << dt << "  " << ldt << std::endl;
      TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
    }
    // printf("Returning Layer \n");

    return ret;
  }
};

REGISTER_SUBMODULE(_fused_gptj_infer, m) {
  m.def("layer_norm", &lyr_norm_wrap, "TPP layer norm");
  m.def("fc_in", &fc_in_wrap, "TPP fc_in");
  m.def("fc_out", &fc_out_wrap, "TPP fc_out");
  m.def("fc_plain", &fc_plain_wrap, "TPP fc_plain");
  m.def(
      "apply_rotary_pos_emb",
      &apply_rotary_pos_emb_wrap,
      "TPP apply_rotary_pos_emb");
  py::class_<GPTJBlock>(m, "GPTJBlock")
      .def(py::init<std::vector<at::Tensor>, double, long, long, long, long>())
      //.def("setMasks", &BertEncoder::setMasks)
      .def("forward", &GPTJBlock::forward);
}

TORCH_LIBRARY(tpp_gptj, m) {
  m.def("layer_norm", &lyr_norm_wrap);
  m.class_<GPTJBlock>("GPTJBlock")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long, long>())
      .def("forward", &GPTJBlock::forward);
}
