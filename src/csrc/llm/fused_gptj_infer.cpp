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

template <typename T, typename LT = T>
inline void lyr_norm(
    at::Tensor& t_in,
    at::Tensor& t_gamma,
    at::Tensor& t_beta,
    at::Tensor& t_out,
    float eps) {
  auto in_sizes = t_in.sizes();
  auto Nk = in_sizes[0];
  auto BS = in_sizes[1] * in_sizes[2];
  auto Hk = in_sizes[3];

  auto in = GetVLAPtr<T>(t_in, {BS, Hk});
  auto gamma = GetVLAPtr<LT>(t_gamma, {Hk});
  auto beta = GetVLAPtr<LT>(t_beta, {Hk});
  auto out = GetVLAPtr<T>(t_out, {BS, Hk});

  auto layer_norm_fwd_tpp =
      SCOPEIT((LayerNormFwdTPP<T, LT>(Nk, BS, Hk, eps)), LAYER_NORM);

  {
    RECORD_SCOPE(lnorm, {t_in, t_gamma, t_beta});

      layer_norm_fwd_tpp(
          in[0][0], gamma[0], beta[0], nullptr, nullptr, out[0][0]);
  }
}

static at::Tensor lyr_norm_wrap(
    at::Tensor& t_in,
    at::Tensor& t_gamma,
    at::Tensor& t_beta,
    float eps) {
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
  } else {
    TPP_ASSERT(0, "Should not come here\n");
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
  auto Nc = in_sizes[0];
  auto BS = in_sizes[1] * in_sizes[2];
  auto Hc = in_sizes[3];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {BS, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {BS, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {BS, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {BS, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      BSb,
      Hk,
      Hc,
      BS * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      rem,
      Hk,
      Hc,
      BS * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      Ncb)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb * Hk)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem * Hk)), EW_ADD);

  {
    RECORD_SCOPE(o_gemm, {t_in, t_wt_V});
    //auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto loop_scheme = large_cache_opt ? "acB" : "aCb";
    auto ogemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
	  bool is_rem = (s1 + BSb > BS);
	  if (!is_rem) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], out[nk][s1]);
            }
            brgemm_tpp(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[nk][s1], in1[nk][s1], out[nk][s1]);
              add_tpp(out[nk][s1], in2[nk][s1], out[nk][s1]);
            }
	  } else {
            if (nc == 0) {
              copy_bias_tpp_rem(bias[nk], out[nk][s1]);
            }
            brgemm_tpp_rem(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, false);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[nk][s1], in1[nk][s1], out[nk][s1]);
              add_tpp_rem(out[nk][s1], in2[nk][s1], out[nk][s1]);
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
    TPP_ASSERT(0, "Should not come here\n");
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
  auto Nc = in_sizes[0];
  auto BS = in_sizes[1] * in_sizes[2];
  auto Hc = in_sizes[3];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {BS, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {BS, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      BSb,
      Hk,
      Hc,
      BS * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      rem,
      Hk,
      Hc,
      BS * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      Ncb)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb * Hk), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem * Hk), ACT);

  {
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    //auto loop_scheme = large_cache_opt ? "acB" : "aBC";
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
              copy_bias_tpp(bias[nk], out[nk][s1]);
            }
            brgemm_tpp(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[nk][s1], out[nk][s1]);
            }
	  } else {
            if (nc == 0) {
              copy_bias_tpp_rem(bias[nk], out[nk][s1]);
            }
            brgemm_tpp_rem(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, false);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[nk][s1], out[nk][s1]);
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
  sizes[0] = wt_sizes[0];
  sizes[3] = wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    fc_in<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    fc_in<bfloat16>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat8) {
    fc_in<bfloat8>(t_in, t_wt, t_bias, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here\n");
  }
  return t_out;
}

class GPTJBlock {
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
      float eps,
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

  template <typename T, typename Tout=T>
  inline void qkv_gemm(
    at::Tensor t_in,
    at::Tensor& t_wt,
    at::Tensor& t_out) {
    auto in_sizes = t_in.sizes();
    auto wt_sizes = t_wt.sizes();
    auto Nc = in_sizes[0];
    auto BS = in_sizes[1] * in_sizes[2];
    auto Hc = in_sizes[3];

    auto Nk = wt_sizes[0];
    auto Hk = wt_sizes[3];

    auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

    auto in = GetVLAPtr<T>(t_in, {BS, Hc});
    auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
    auto out = GetVLAPtr<Tout>(t_out, {BS, Hk});

    auto Ncb = Nc;
    auto BSb = 64L;
    auto rem = BS % 64;

    auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk), EW_ZERO);
    auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk), EW_ZERO);
    auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, Tout>(
        BSb,
        Hk,
        Hc,
        BS * Hc,
        Hk * Hc,
        1.0,
        XformTPP::XFORM_NONE_TPP,
        0,
        Ncb)));
    auto brgemm_tpp_rem = SCOPEITGEMM((BrgemmExtTPP<T, Tout>(
        rem,
        Hk,
        Hc,
        BS * Hc,
        Hk * Hc,
        1.0,
        XformTPP::XFORM_NONE_TPP,
        0,
        Ncb)));

    {
      RECORD_SCOPE(qkv_gemm, {t_in, t_wt_V});
      //auto loop_scheme = large_cache_opt ? "acB" : "aBC";
      auto loop_scheme = large_cache_opt ? "acB" : "aCb";
      auto gemm_loop =
          ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
            bool is_rem = (s1 + BSb > BS);
            if (!is_rem) {
              if (nc == 0) {
                zero_tpp(out[nk][s1]);
              }
              brgemm_tpp(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, true);
            } else {
              if (nc == 0) {
                zero_tpp_rem(out[nk][s1]);
              }
              brgemm_tpp_rem(in[nc][s1], wt_V[nk][nc], out[nk][s1], count, false);
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
    long B = sizes[1];
    long Sq = sizes[2];
    long H2 = sizes[3];
    auto ksizes = t_KL.sizes();
    long Sk = ksizes[2];
    long offset = Sk - Sq;
    auto H1 = H / H2;
    // printf("B=%ld S1=%ld S2=%ld H1=%ld H2=%ld N=%ld\n", B, S1, S2, H1, H2,
    // N);
    //printf("B=%ld Sq1=%ld Sq2=%ld N=%ld H=%ld, H1=%ld H2=%ld Sk1=%ld Sk2=%ld offset=%ld\n", B, Sq1, Sq2, N, H, H1, H2, Sk1, Sk2, offset);
    auto QL = GetVLAPtr<T>(t_QL, {H1, B, Sq * H2});
    auto KL = GetVLAPtr<T>(t_KL, {H1, B, Sk * H2});
    auto VL = GetVLAPtr<Tv>(t_VL, {H1, B, Sk * H2});
    auto CL = GetVLAPtr<T>(t_CL, {H1, B, Sq * H2});
    auto AM = GetVLAPtr<T>(t_AM);
    auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
        Sq,
        Sk,
        H2,
        B * Sq * H2,
        B * H2 * Sk,
        0.0,
        XformTPP::XFORM_NONE_TPP,
        0,
        H1)));
    auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sq * Sk)), EW_SCL);
    auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sq, Sk), EW_ADD);
    auto softmax_fwd_tpp =
        SCOPEIT((SoftMaxFwdTPP<float, Tv>(1, Sq, Sk)), SOFTMAX);
    auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<Tv, Tv>(
        Sq,
        H2,
        Sk,
        Sq * Sk,
        Sk * H2,
        0.0,
        XformTPP::XFORM_NONE_TPP,
        0,
        1)));
    auto cvt_tpp = SCOPEIT((ConvertTPP<Tv,T>(Sq,H2)), EW_COPY);
    auto cpy_tpp = SCOPEIT(CpyTPP<T>(Sk,H2), EW_COPY);
    auto xform = XformTPP::XFORM_XPOSE_TPP;
    if (!std::is_same<T, float>::value) {
      xform = XformTPP::XFORM_XPOSE_N2V_TPP;
    }
    auto xform_tpp =
      SCOPEIT(XformExtTPP<T>(Sk, H2, xform, true), XPOSE);

    {
      RECORD_SCOPE(ac_gemm, {t_QL, t_KL});
      {
#pragma omp parallel for collapse(2)
        for (int n = 0; n < N; n++) {
          for (int b = 0; b < B; b++) {
            float AS[Sq][Sk];
            Tv AST[Sq][Sk];
            for (int h1 = 0; h1 < H1; h1++) {
              T tmp[Sk * H2];
              cpy_tpp(KL[n][h1][b], tmp);
              xform_tpp(tmp, KL[n][h1][b]);
            }
            a_gemm_tpp(
                QL[n][0][b], KL[n][0][b], AS[0], H1);
            for (int sq = 0; sq < Sq; sq++) {
              auto qval = sq + offset;
              for (int sk = qval + 1; sk < Sk; sk++) {
                AS[sq][sk] = -1e9;
              }
            }
            scale_tpp(AS[0], AS[0], one_by_sqrt_H);
            if (t_AM.numel() != 0)
              add_mask_tpp(AM, AS[0]);
            softmax_fwd_tpp(AS[0], AST[0]);
            for (int h1 = 0; h1 < H1; h1++) {
              if (std::is_same<Tv, T>::value) {
                c_gemm_tpp(
                    AST[0], VL[n][h1][b], (Tv*)CL[n][h1][b], 1);
              } else {
                Tv tmp[Sq * H2];
                c_gemm_tpp(
                    AST[0], VL[n][h1][b], tmp, 1);
                cvt_tpp(tmp, CL[n][h1][b]);
              }
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
    auto B = sizes[1];
    auto S = sizes[2];
    auto t_QL = at::empty_like(t_HS); // t_scratch[0];
    auto t_KL = at::empty_like(t_QL); // t_scratch[0];
    auto t_VL = at::empty_like(t_QL, at::kFloat); // t_scratch[0];
    //auto t_KL_TV = at::empty_like(t_KL); // t_scratch[1];
    //auto t_VL_V = at::empty_like(t_VL); // t_scratch[2];
    auto t_CL = at::empty_like(t_QL); // t_scratch[3];
    auto t_SO = at::empty_like(t_HS); // t_scratch[4];
    auto t_I = t_HS.new_empty({t_Wi.size(0), B, S, t_Wi.size(3)});
    //auto t_I = at::empty_like(t_HS); // t_scratch[5];
    auto t_HS_qkv = at::empty_like(t_HS);

    //std::cout << "HS: " << t_HS.sizes() << std::endl;
    //std::cout << "use_cche: " << use_cache << " t_key_past.numel: " << t_key_past.numel() << std::endl;
    auto t_null = t_HS.new_empty({0}); //at::Tensor().to(t_HS.dtype());
    lyr_norm<T, LT>(t_HS, t_G, t_B, t_HS_qkv, eps);
    //std::cout << "HS_qkv: " << t_HS_qkv.sizes() << std::endl;
    qkv_gemm<T>(t_HS_qkv, t_Wk, t_KL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    if (t_key_past.numel() > 0) {
      // std::cout << "t_key_past: " << t_key_past.sizes() << std::endl;
      // std::cout << "t_KL: " << t_KL.sizes() << std::endl;
      t_KL = at::cat({t_key_past, t_KL}, -2);
    }
    // printf("reached at %s:%d\n", __func__, __LINE__);
    // std::cout << t_KL.sizes() << std::endl;
    qkv_gemm<T,float>(t_HS_qkv, t_Wv, t_VL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    if (t_value_past.numel() > 0) {
      t_VL = at::cat({t_value_past, t_VL}, -2);
    }
    // printf("reached at %s:%d\n", __func__, __LINE__);
    qkv_gemm<T>(t_HS_qkv, t_Wq, t_QL);
    // printf("reached at %s:%d\n", __func__, __LINE__);

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
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
    // printf("Returning Layer \n");

    return ret;
  }
};


REGISTER_SUBMODULE(_fused_gptj_infer, m) {
  m.def("layer_norm", &lyr_norm_wrap, "TPP layer norm");
  m.def("fc_in", &fc_in_wrap, "TPP fc_in");
  m.def("fc_out", &fc_out_wrap, "TPP fc_oit");
  py::class_<GPTJBlock>(m, "GPTJBlock")
      .def(py::init<std::vector<at::Tensor>, float, long, long, long, long>())
      //.def("setMasks", &BertEncoder::setMasks)
      .def("forward", &GPTJBlock::forward);
}
