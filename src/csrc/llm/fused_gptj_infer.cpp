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

template <typename T, bool transpose = false, bool vnni = false>
inline void blocked_gemm(
    at::Tensor& t_HS,
    at::Tensor& t_W,
    at::Tensor& t_B,
    at::Tensor& t_Out) {
  auto sizes = t_HS.sizes();
  long S1 = sizes[0] * sizes[1];
  long C1 = sizes[2];
  long S2 = sizes[3];
  long C2 = sizes[4];
  auto wt_sizes = t_W.sizes();
  auto K1 = wt_sizes[0];
  auto K2 = wt_sizes[3];

  TPP_ASSERT(sizes.size() == 5, "Incorrect dims");
  bool dt_low_prec = (t_HS.dtype() != at::kFloat);
  bool with_bias = t_B.numel() != 0;
  auto W = GetVLAPtr<T>(t_W, {C1, C2 * K2});
  auto B = GetVLAPtr<T>(t_B, {K2});
  auto Out = GetVLAPtr<T>(t_Out, {K1, S2 * K2});
  auto HS = GetVLAPtr<T>(t_HS, {C1, S2 * C2});

  auto xform = XformTPP::XFORM_XPOSE_TPP;
  if (vnni && dt_low_prec) {
    if (transpose) {
      xform = XformTPP::XFORM_XPOSE_N2V_TPP;
    } else {
      xform = XformTPP::XFORM_N2V_TPP;
    }
  } else if (transpose) {
    xform = XformTPP::XFORM_XPOSE_TPP;
  }
  auto xform_tpp =
      SCOPEIT(XformExtTPP<T>(S2, K2, xform, true), transpose ? XPOSE : VNNI);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, K2), BIAS);
  auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      K2,
      C2,
      S2 * C2,
      C2 * K2,
      with_bias ? 1.0 : 0.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      C1)));
  auto loop_scheme = large_cache_opt ? "bA" : "AB";
  auto qkv_loop = ThreadedLoop<2>({{S1}, {K1}}, loop_scheme);
  {
    RECORD_SCOPE(qkv_gemm, {t_HS, t_W});
    qkv_loop(
        [&](int* ind) {
          int s1 = ind[0], nk = ind[1];
          T tmp[S2 * K2];
          T* tmpp = (transpose || (vnni && dt_low_prec)) ? tmp : Out[s1][nk];
          if (with_bias)
            copy_bias_tpp(B[nk], tmpp);
          qkv_gemm_tpp(HS[s1][0], W[nk][0], tmpp, C1, true);
          if (transpose || (vnni && dt_low_prec)) {
            xform_tpp(tmpp, Out[s1][nk]);
          }
        },
        [&]() { qkv_gemm_tpp.config(); },
        [&]() { qkv_gemm_tpp.release(); });
  }
}

static at::Tensor blocked_gemm_wrap(
    at::Tensor& t_HS,
    at::Tensor& t_W,
    at::Tensor& t_B) {
  GlobalPass _gp(FWD);
  auto dt = t_HS.dtype();
  auto sizes = t_HS.sizes().vec();
  auto wt_sizes = t_W.sizes();
  sizes[2] = wt_sizes[0];
  sizes[4] = wt_sizes[3];
  auto t_out = t_HS.new_empty(sizes);

  if (dt == at::kFloat) {
    blocked_gemm<float>(t_HS, t_W, t_B, t_out);
  } else if (dt == at::kBFloat16) {
    blocked_gemm<bfloat16>(t_HS, t_W, t_B, t_out);
  } else if (dt == at::kBFloat8) {
    blocked_gemm<bfloat8>(t_HS, t_W, t_B, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here\n");
  }
  return t_out;
}

template <typename T, typename LT = T>
inline void lyr_norm(
    at::Tensor& t_in,
    at::Tensor& t_gamma,
    at::Tensor& t_beta,
    at::Tensor& t_out,
    float eps) {
  auto in_sizes = t_in.sizes();
  auto S1 = in_sizes[0] * in_sizes[1];
  auto Nk = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hk = in_sizes[4];

  auto in = GetVLAPtr<T>(t_in, {Nk, S2 * Hk});
  auto gamma = GetVLAPtr<LT>(t_gamma, {Hk});
  auto beta = GetVLAPtr<LT>(t_beta, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, S2 * Hk});

  auto layer_norm_fwd_tpp =
      SCOPEIT((LayerNormFwdTPP<T, LT>(Nk, S2, Hk, eps)), LAYER_NORM);

  {
    RECORD_SCOPE(lnorm, {t_in, t_gamma, t_beta});

#pragma omp parallel for
    for (int s1 = 0; s1 < S1; s1++) {
      layer_norm_fwd_tpp(
          in[s1][0], gamma[0], beta[0], nullptr, nullptr, out[s1][0]);
    }
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
  auto S1 = in_sizes[0] * in_sizes[1];
  auto Nc = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hc = in_sizes[4];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  const bool use_at_vnni_local = t_in.dtype() != at::kFloat && use_at_vnni;
  if (use_at_vnni_local) {
    t_in = act_tensor_trans_n2v(S1, Nc, S2, Hc, t_in);
  }

  auto in = GetVLAPtr<T>(t_in, {Nc, S2 * Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, S2 * Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, S2 * Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, S2 * Hk});

  auto Ncb = 8;

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, Hk), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      Hk,
      Hc,
      S2 * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      use_at_vnni_local ? 1 : 0,
      Ncb)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(S2 * Hk)), EW_ADD);

  {
    RECORD_SCOPE(o_gemm, {t_in, t_wt});
    auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto ogemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {S1}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          if (nc == 0) {
            copy_bias_tpp(bias[nk], out[s1][nk]);
          }
          brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          if (!(nc + Ncb < Nc)) { // last nc iter
            add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
            add_tpp(out[s1][nk], in2[s1][nk], out[s1][nk]);
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
  auto S1 = in_sizes[0] * in_sizes[1];
  auto Nc = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hc = in_sizes[4];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);
  const bool use_at_vnni_local = t_in.dtype() != at::kFloat && use_at_vnni;
  if (use_at_vnni_local) {
    t_in = act_tensor_trans_n2v(S1, Nc, S2, Hc, t_in);
  }

  auto Ncb = 8;

  // Create TPPs
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, Hk), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      Hk,
      Hc,
      S2 * Hc,
      Hk * Hc,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      use_at_vnni_local ? 1 : 0,
      Ncb)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(S2 * Hk), ACT);

  auto in = GetVLAPtr<T>(t_in, {Nc, S2 * Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, S2 * Hk});

  {
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? "acB" : "aBC";
    auto gemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {S1}, {Nk}}, loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = (nc + Ncb < Nc) ? Ncb : Nc - nc;

          if (nc == 0) {
            copy_bias_tpp(bias[nk], out[s1][nk]);
          }
          brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          if (!(nc + Ncb < Nc)) { // last iter
            gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
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
  sizes[2] = wt_sizes[0];
  sizes[4] = wt_sizes[3];

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

  template <typename T, bool transpose, bool vnni, typename Tout>
  inline void qkv_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    auto sizes = t_HS.sizes();
    long S1 = sizes[0] * sizes[1];
    long C1 = sizes[2];
    long S2 = sizes[3];
    long C2 = sizes[4];
    if (use_at_vnni_local) {
      TPP_ASSERT(sizes.size() == 6, "Incorrect dims");
      S2 = sizes[4];
      C2 = sizes[3] * sizes[5];
    } else {
      TPP_ASSERT(sizes.size() == 5, "Incorrect dims");
    }
    bool dt_low_prec = (t_Out.dtype() != at::kFloat);
    bool with_bias = t_B.numel() != 0;
    auto W = GetVLAPtr<T>(t_W, {C1, C2 * K2});
    auto B = GetVLAPtr<T>(t_B, {K2});
    auto Out = GetVLAPtr<Tout>(t_Out, {K1, S2 * K2});
    auto HS = GetVLAPtr<T>(t_HS, {C1, S2 * C2});

    auto xform = XformTPP::XFORM_XPOSE_TPP;
    if (vnni && dt_low_prec) {
      if (transpose) {
        xform = XformTPP::XFORM_XPOSE_N2V_TPP;
      } else {
        xform = XformTPP::XFORM_N2V_TPP;
      }
    } else if (transpose) {
      xform = XformTPP::XFORM_XPOSE_TPP;
    }
    auto xform_tpp =
        SCOPEIT(XformExtTPP<Tout>(S2, K2, xform, true), transpose ? XPOSE : VNNI);
    auto copy_bias_tpp = SCOPEIT((CpyBiasTPP<T, Tout>(S2, K2)), BIAS);
    auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, Tout>(
        S2,
        K2,
        C2,
        S2 * C2,
        C2 * K2,
        with_bias ? 1.0 : 0.0,
        XformTPP::XFORM_NONE_TPP,
        use_at_vnni_local ? 1 : 0,
        C1)));
    auto loop_scheme = large_cache_opt ? "bA" : "AB";
    auto qkv_loop = ThreadedLoop<2>({{S1}, {K1}}, loop_scheme);
    {
      RECORD_SCOPE(qkv_gemm, {t_HS, t_W});
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];
            Tout tmp[S2 * K2];
            Tout* tmpp = (transpose || (vnni && dt_low_prec)) ? tmp : Out[s1][nk];
            if (with_bias)
              copy_bias_tpp(B[nk], tmpp);
            qkv_gemm_tpp(HS[s1][0], W[nk][0], tmpp, C1, true);
            if (transpose || (vnni && dt_low_prec)) {
              xform_tpp(tmpp, Out[s1][nk]);
            }
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
    }
  }

  template <typename T, typename Tout=T>
  inline void q_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    if (use_at_vnni_local) {
      qkv_gemm<T, true, true, Tout>(t_HS, t_W, t_B, t_Out);
    } else {
      qkv_gemm<T, false, false, Tout>(t_HS, t_W, t_B, t_Out);
    }
  }

  template <typename T, typename Tout=T>
  inline void k_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, true, true, Tout>(t_HS, t_W, t_B, t_Out);
  }

  template <typename T, typename Tout>
  inline void v_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, false, true, Tout>(t_HS, t_W, t_B, t_Out);
  }

  template <typename T>
  inline void p_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, false, false, T>(t_HS, t_W, t_B, t_Out);
  }

  template <typename T, typename Tv>
  inline void attn(
      at::Tensor& t_QL,
      at::Tensor& t_KL_TV,
      at::Tensor& t_AM,
      at::Tensor& t_VL_V,
      at::Tensor& t_CL) {
    auto sizes = t_QL.sizes();
    long B = sizes[0];
    long Sq1 = sizes[1];
    long Sq2 = sizes[3];
    long H2 = sizes[4];
    auto ksizes = t_KL_TV.sizes();
    long Sk1 = ksizes[1];
    long Sk2 = ksizes[3];
    long offset = Sk1 * Sk2 - Sq1 * Sq2;
    const bool use_at_vnni_local = t_QL.dtype() != at::kFloat && use_at_vnni;
    // auto H1 = F1 / N;
    auto H1 = H / H2;
    // printf("B=%ld S1=%ld S2=%ld H1=%ld H2=%ld N=%ld\n", B, S1, S2, H1, H2,
    // N);
    //printf("B=%ld Sq1=%ld Sq2=%ld N=%ld H=%ld, H1=%ld H2=%ld Sk1=%ld Sk2=%ld offset=%ld\n", B, Sq1, Sq2, N, H, H1, H2, Sk1, Sk2, offset);
    auto QL = GetVLAPtr<T>(t_QL, {Sq1, N, H1, Sq2 * H2});
    auto KL_TV = GetVLAPtr<T>(t_KL_TV, {Sk1, N, H1, H2 * Sk2});
    auto VL_V = GetVLAPtr<Tv>(t_VL_V, {Sk1, N, H1, Sk2 * H2});
    auto CL = GetVLAPtr<T>(t_CL, {Sq1, N, H1, Sq2 * H2});
    auto AM = GetVLAPtr<T>(t_AM, {Sk2});
    auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
        Sq2,
        Sk2,
        H2,
        Sq2 * H2,
        H2 * Sk2,
        0.0,
        XformTPP::XFORM_NONE_TPP,
        use_at_vnni_local ? 1 : 0,
        H1)));
    auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sq2 * Sk2)), EW_SCL);
    auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sq2, Sk2), EW_ADD);
    auto softmax_fwd_tpp =
        SCOPEIT((SoftMaxFwdTPP<float, Tv>(Sk1, Sq2, Sk2)), SOFTMAX);
    auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<Tv, Tv>(
        Sq2,
        H2,
        Sk2,
        Sq2 * Sk2,
        N * Sk2 * H,
        0.0,
        XformTPP::XFORM_NONE_TPP,
        0,
        Sk1)));
    auto cvt_tpp = SCOPEIT((ConvertTPP<Tv,T>(Sq2,H2)), EW_COPY);

    {
      RECORD_SCOPE(ac_gemm, {t_QL, t_KL_TV});
      {
#pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
          for (int n = 0; n < N; n++) {
            for (int sq1 = 0; sq1 < Sq1; sq1++) {
              float AS[Sk1][Sq2][Sk2];
              Tv AST[Sk1][Sq2][Sk2];
              for (int sk1 = 0; sk1 < Sk1; sk1++) {
                a_gemm_tpp(
                    QL[b][sq1][n][0], KL_TV[b][sk1][n][0], AS[sk1][0], H1);
                for (int sq2 = 0; sq2 < Sq2; sq2++) {
                  auto qval = sq1 * Sq2 + sq2 + offset - sk1 * Sk2;
                  for (int sk2 = qval + 1; sk2 < Sk2; sk2++) {
                    AS[sk1][sq2][sk2] = -1e9;
                  }
                }
                scale_tpp(AS[sk1][0], AS[sk1][0], one_by_sqrt_H);
                if (t_AM.numel() != 0)
                  add_mask_tpp(AM[sk1], AS[sk1][0]);
              }
              softmax_fwd_tpp(AS[0][0], AST[0][0]);
              for (int h1 = 0; h1 < H1; h1++) {
                if (std::is_same<Tv, T>::value) {
                  c_gemm_tpp(
                      AST[0][0], VL_V[b][0][n][h1], (Tv*)CL[b][sq1][n][h1], Sk1);
                } else {
                  Tv tmp[Sq2 * H2];
                  c_gemm_tpp(
                      AST[0][0], VL_V[b][0][n][h1], tmp, Sk1);
                  cvt_tpp(tmp, CL[b][sq1][n][h1]);
                }
              }
            }
          }
        }
      }
    }
  }

  at::Tensor cvt_to_trans_vnni(at::Tensor t) {
    auto sizes = t.sizes();
    std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
    return t;
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
    auto S1 = sizes[1];
    auto S2 = sizes[3];
    auto t_QL = t_HS.new_empty({B, S1, K1, S2, K2});
    auto t_KL = at::empty_like(t_QL); // t_scratch[0];
    auto t_VL = at::empty_like(t_QL, at::kFloat); // t_scratch[0];
    auto t_KL_TV = at::empty_like(t_KL); // t_scratch[1];
    auto t_VL_V = at::empty_like(t_VL); // t_scratch[2];
    auto t_CL = at::empty_like(t_QL); // t_scratch[3];
    auto t_SO = at::empty_like(t_HS); // t_scratch[4];
    auto t_I = t_HS.new_empty({B, S1, t_Wi.size(0), S2, t_Wi.size(3)});
    //auto t_I = at::empty_like(t_HS); // t_scratch[5];
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    auto t_HS_qkv = at::empty_like(t_HS);

    //std::cout << "HS: " << t_HS.sizes() << std::endl;
    //std::cout << "use_cche: " << use_cache << " t_key_past.numel: " << t_key_past.numel() << std::endl;
    auto t_null = t_HS.new_empty({0}); //at::Tensor().to(t_HS.dtype());
    lyr_norm<T, LT>(t_HS, t_G, t_B, t_HS_qkv, eps);
    //std::cout << "HS_qkv: " << t_HS_qkv.sizes() << std::endl;
    k_gemm<T>(t_HS_qkv, t_Wk, t_null, t_KL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    if (t_key_past.numel() > 0) {
      //std::cout << "t_key_past: " << t_key_past.sizes() << std::endl;
      //std::cout << "t_KL: " << t_KL.sizes() << std::endl;
      t_KL = at::cat({t_key_past, t_KL}, -2);
    }
    t_KL_TV = t_KL;
    // printf("reached at %s:%d\n", __func__, __LINE__);
    //std::cout << t_KL.sizes() << std::endl;
    v_gemm<T,float>(t_HS_qkv, t_Wv, t_null, t_VL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    if (t_value_past.numel() > 0) {
      t_VL = at::cat({t_value_past, t_VL}, -2);
    }
    t_VL_V = t_VL;
    // printf("reached at %s:%d\n", __func__, __LINE__);
    q_gemm<T>(t_HS_qkv, t_Wq, t_null, t_QL);
    // printf("reached at %s:%d\n", __func__, __LINE__);

    attn<T, float>(t_QL, t_KL_TV, t_am, t_VL_V, t_CL);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    p_gemm<T>(t_CL, t_Wp, t_null, t_SO);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    fc_in<T>(t_HS_qkv, t_Wi, t_Bi, t_I);
    // printf("reached at %s:%d\n", __func__, __LINE__);
    fc_out<T>(t_SO, t_I, t_HS, t_Wo, t_Bo, t_Out);
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

#if 0
class BertEncoder {
 public:
  float eps;
  long N, HS, IS, H;

  std::vector<BertEncoderLayer*> layers;
  BertEncoder(
      std::vector<std::vector<at::Tensor>> params,
      float eps,
      long N,
      long HS,
      long IS)
      : eps(eps), N(N), HS(HS), IS(IS) {
    H = HS / N;
    auto nLayers = params.size();
    layers.reserve(nLayers);
    for (int i = 0; i < nLayers; i++) {
      layers.push_back(new BertEncoderLayer(params[i], eps, N, H));
    }
  }

  ~BertEncoder() {
    for (int i = 0; i < layers.size(); i++) {
      delete layers[i];
      layers[i] = nullptr;
    }
  }

#if 0
    std::vector<at::Tensor>  generate_masks(at::Tensor t_mask) {
      auto sizes = t_mask.sizes();
      auto B = sizes[0];
      auto S = sizes[3];
      auto S1 = 1, S2 = S;
      int nnz[B];
      auto attention_mask = t_mask.view({B, S}).clone();
      long BS = 0;
      for (i = 0; i < B; i++) {
        int nz = S;
        for (j = 0; j < S; j++) {
          if (am[i][j] + 10000 == 0) {
            nz = ((j + S2) / S2) * S2;
            break;
          }
        }
        nnz[i] = nz;
        BS += nz;
      }
      S1 = BS / S2;
      auto seq_offsets = t_mask.new_empty({B+1}, at::kLong);
      auto seq_sqr_offsets = t_mask.new_empty({B+1}, at::kLong);
      auto bmap = t_mask.new_empty({S1}, at::kLong);
      auto attention_mask = t_mask.new_empty({BS}, at::kFloat);
      for (i = 0; i < B; i++) {
        seq_offsets[i]
        for (j = 0; j < nnj[i]; j++) {
          attention_mask[k] = mask[i][j];
        }
      }


      auto nnz = (((attention_mask + 10000).count_nonzero(-1) + (S2 - 1)) / S2) * S2;
      auto nnz1 = nnz.unsqueeze

    }
#endif
  std::vector<at::Tensor> get_scratch(at::Tensor& t_HS) {
    auto sizes1 = t_HS.sizes();
    auto sizes2 = t_HS.sizes().vec();
    auto F1 = sizes1[1];
    auto I1 = F1 * IS / HS;
    sizes2[1] = I1;
    std::vector<at::Tensor> ret;
    ret.push_back(t_HS.new_empty(sizes1)); // QL
    ret.push_back(t_HS.new_empty(sizes1)); // KL
    ret.push_back(t_HS.new_empty(sizes1)); // VL
    ret.push_back(t_HS.new_empty(sizes1)); // CL
    ret.push_back(t_HS.new_empty(sizes1)); // SO
    ret.push_back(t_HS.new_empty(sizes2)); // I

    return ret;
  }

  at::Tensor forward(at::Tensor& t_HS, std::vector<at::Tensor>& t_masks) {
    GlobalPass _gp(FWD);
    auto t_scratch = get_scratch(t_HS);
    auto dt = layers[0]->t_Wq.dtype();
    auto ldt = layers[0]->t_Go.dtype();

    for (auto l : layers) {
      // printf("Layer %d\n", i++);
      if (dt == at::kFloat && ldt == at::kFloat) {
        t_HS = l->forward<float, float>(t_HS, t_masks, t_HS, t_scratch);
      } else if (dt == at::kBFloat16 && ldt == at::kFloat) {
        t_HS = l->forward<bfloat16, float>(t_HS, t_masks, t_HS, t_scratch);
      } else if (dt == at::kBFloat16 && ldt == at::kBFloat16) {
        t_HS = l->forward<bfloat16, bfloat16>(t_HS, t_masks, t_HS, t_scratch);
      } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
        t_HS = l->forward<bfloat8, float>(t_HS, t_masks, t_HS, t_scratch);
      } else {
        TPP_ASSERT(0, "Should not come here\n");
      }
    }
    // printf("Returning Layer \n");

    return t_HS;
  }
};
#endif

REGISTER_SUBMODULE(_fused_gptj_infer, m) {
  m.def("layer_norm", &lyr_norm_wrap, "TPP layer norm");
  m.def("fc_in", &fc_in_wrap, "TPP fc_in");
  m.def("fc_out", &fc_out_wrap, "TPP fc_oit");
  py::class_<GPTJBlock>(m, "GPTJBlock")
      .def(py::init<std::vector<at::Tensor>, float, long, long, long, long>())
      //.def("setMasks", &BertEncoder::setMasks)
      .def("forward", &GPTJBlock::forward);
}
