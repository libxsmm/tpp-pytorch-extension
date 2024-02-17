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
static int use_at_vnni = env2int("USE_AT_VNNI");

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(ac_gemm, "ac_gemm");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");

class BertEncoderLayer {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv;
  at::Tensor t_Bq, t_Bk, t_Bv;
  at::Tensor t_Wso, t_Wi, t_Wo;
  at::Tensor t_Bso, t_Bi, t_Bo;
  at::Tensor t_Gso, t_Go, t_Beta_so, t_Beta_o;
  float eps, one_by_sqrt_H;
  long N, H, H1, F2;
  // bool large_cache_opt = false;

  BertEncoderLayer(std::vector<at::Tensor>& params, float eps, long N, long H)
      : eps(eps), N(N), H(H) {
    int i = 0;
    t_Wq = params[i++];
    t_Wk = params[i++];
    t_Wv = params[i++];
    t_Bq = params[i++];
    t_Bk = params[i++];
    t_Bv = params[i++];

    t_Wso = params[i++];
    t_Bso = params[i++];
    t_Gso = params[i++];
    t_Beta_so = params[i++];

    t_Wi = params[i++];
    t_Bi = params[i++];

    t_Wo = params[i++];
    t_Bo = params[i++];
    t_Go = params[i++];
    t_Beta_o = params[i++];

    one_by_sqrt_H = 1.0 / sqrt(H);
    auto sizes = t_Wq.sizes();
    // std::cout << "t_Wq.sizes" << sizes << std::endl;
    F2 = sizes[3];
    H1 = H / F2;
    // std::cout << "F2=" << F2 << " H=" << H << " H1=" << H1 << std::endl;
  }

  template <typename T, bool transpose, bool vnni>
  inline void qkv_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    auto sizes = t_HS.sizes();
    long S1 = sizes[0];
    long F1 = sizes[1];
    long S2 = sizes[2];
    long F2 = sizes[3];
    if (use_at_vnni_local) {
      TPP_ASSERT(sizes.size() == 5, "Incorrect dims");
      S2 = sizes[3];
      F2 = sizes[2] * sizes[4];
    } else {
      TPP_ASSERT(sizes.size() == 4, "Incorrect dims");
    }
    bool dt_low_prec = (t_HS.dtype() != at::kFloat);
    auto W = GetVLAPtr<T>(t_W, {F1, F2 * F2});
    auto B = GetVLAPtr<T>(t_B, {F2});
    auto Out = GetVLAPtr<T>(t_Out, {F1, S2 * F2});
    auto HS = GetVLAPtr<T>(t_HS, {F1, S2 * F2});

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
        SCOPEIT(XformExtTPP<T>(S2, F2, xform, true), transpose ? XPOSE : VNNI);
    auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, F2), BIAS);
    auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
        S2,
        F2,
        F2,
        S2 * F2,
        F2 * F2,
        1.0,
        XformTPP::XFORM_NONE_TPP,
        use_at_vnni_local ? 1 : 0,
        F1)));
    auto loop_scheme = large_cache_opt ? "bA" : "AB";
    auto qkv_loop =
        ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{F1}}, loop_scheme);
    {
      RECORD_SCOPE(qkv_gemm, {t_HS, t_W});
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];
            T tmp[S2 * F2];
            T* tmpp = (transpose || (vnni && dt_low_prec)) ? tmp : Out[s1][nk];
            copy_bias_tpp(B[nk], tmpp);
            qkv_gemm_tpp(HS[s1][0], W[nk][0], tmpp, F1, true);
            if (transpose || (vnni && dt_low_prec)) {
              xform_tpp(tmpp, Out[s1][nk]);
            }
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
    }
  }

  template <typename T>
  inline void q_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    if (use_at_vnni_local) {
      qkv_gemm<T, true, true>(t_HS, t_W, t_B, t_Out);
    } else {
      qkv_gemm<T, false, false>(t_HS, t_W, t_B, t_Out);
    }
  }

  template <typename T>
  inline void k_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, true, true>(t_HS, t_W, t_B, t_Out);
  }

  template <typename T>
  inline void v_gemm(
      at::Tensor& t_HS,
      at::Tensor& t_W,
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, false, true>(t_HS, t_W, t_B, t_Out);
  }

  template <typename T>
  inline void self_attn(
      at::Tensor& t_QL,
      at::Tensor& t_KL_TV,
      std::vector<at::Tensor>& t_masks,
      at::Tensor& t_VL_V,
      at::Tensor& t_CL) {
    auto& t_AM = t_masks[0];
    auto& t_offs = t_masks[1];
    auto& t_bmap = t_masks[2];
    auto sizes = t_QL.sizes();
#if 0
    long B = t_offs.sizes()[0] - 1;
#endif
    long S1 = sizes[0];
    long S2 = sizes[2];
    long F2 = sizes[3];
    auto H2 = F2;
    const bool use_at_vnni_local = t_QL.dtype() != at::kFloat && use_at_vnni;
    // auto H1 = F1 / N;
    // auto H1 = H / H2;
    // printf("B=%ld S1=%ld S2=%ld H1=%ld H2=%ld N=%ld\n", B, S1, S2, H1, H2,
    // N);
    auto QL = GetVLAPtr<T>(t_QL, {N, H1, S2 * H2});
    auto KL_TV = GetVLAPtr<T>(t_KL_TV, {N, H1, H2 * S2});
    auto VL_V = GetVLAPtr<T>(t_VL_V, {N, H1, S2 * H2});
    auto CL = GetVLAPtr<T>(t_CL, {N, H1, S2 * H2});
    auto AM = GetVLAPtr<T>(t_AM, {S2});
    auto bmap = GetVLAPtr<long>(t_bmap);
    auto offs = GetVLAPtr<long>(t_offs);
    auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
        S2,
        S2,
        H2,
        S2 * H2,
        H2 * S2,
        0.0,
        XformTPP::XFORM_NONE_TPP,
        use_at_vnni_local ? 1 : 0,
        H1)));
    auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(S2 * S2)), EW_SCL);
    auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(S2, S2), EW_ADD);
    auto softmax_fwd_tpp =
        SCOPEIT((VarSoftMaxFwdTPP<float, T>(S2, S2)), SOFTMAX);
    auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
        S2, H2, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, 1)));

    {
      RECORD_SCOPE(ac_gemm, {t_QL, t_KL_TV});
      {
#if 1
#pragma omp parallel for collapse(2) schedule(static, 1)
        for (int s11 = 0; s11 < S1; s11++) {
          for (int n = 0; n < N; n++) {
            long b = bmap[s11];
            long start = offs[b];
            long end = offs[b + 1];
            long len = end - start;
            float AS[len][S2][S2];
            T AST[len][S2][S2];
            for (int s21 = start; s21 < end; s21++) {
              long ls21 = s21 - start;
              a_gemm_tpp(QL[s11][n][0], KL_TV[s21][n][0], AS[ls21][0], H1);
              scale_tpp(AS[ls21][0], AS[ls21][0], one_by_sqrt_H);
              if (t_AM.numel() != 0)
                add_mask_tpp(AM[s21], AS[ls21][0]);
            }
            softmax_fwd_tpp(len, AS[0][0], AST[0][0]);
            for (int h1 = 0; h1 < H1; h1++) {
              c_gemm_tpp(AST[0][0], VL_V[start][n][h1], CL[s11][n][h1], len);
            }
          }
        }
#else
#pragma omp parallel for collapse(2) schedule(static, 1)
        for (int b = 0; b < B; b++) {
          for (int n = 0; n < N; n++) {
            long start = offs[b];
            long end = offs[b + 1];
            long len = end - start;
            for (int s11 = start; s11 < end; s11++) {
              float AS[len][S2][S2];
              T AST[len][S2][S2];
              for (int s21 = start; s21 < end; s21++) {
                long ls21 = s21 - start;
                a_gemm_tpp(QL[s11][n][0], KL_TV[s21][n][0], AS[ls21][0], H1);
                scale_tpp(AS[ls21][0], AS[ls21][0], one_by_sqrt_H);
                if (t_AM.numel() != 0)
                  add_mask_tpp(AM[s21], AS[ls21][0]);
              }
              softmax_fwd_tpp(len, AS[0][0], AST[0][0]);
              for (int h1 = 0; h1 < H1; h1++) {
                c_gemm_tpp(AST[0][0], VL_V[start][n][h1], CL[s11][n][h1], len);
              }
            }
          }
        }
#endif
      }
    }
  }

  template <typename T, typename LT = T>
  inline void output(
      at::Tensor t_in,
      at::Tensor& t_in2,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      at::Tensor& t_gamma,
      at::Tensor& t_beta,
      at::Tensor& t_out) {
    auto in_sizes = t_in.sizes();
    auto wt_sizes = t_wt.sizes();
    auto S1 = in_sizes[0];
    auto Nc = in_sizes[1];
    auto S2 = in_sizes[2];
    auto Hc = in_sizes[3];

    auto Nk = wt_sizes[0];
    auto Hk = wt_sizes[3];

    auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

    const bool use_at_vnni_local = t_in.dtype() != at::kFloat && use_at_vnni;
    if (use_at_vnni_local) {
      t_in = act_tensor_trans_n2v(S1, Nc, S2, Hc, t_in);
    }

    auto in = GetVLAPtr<T>(t_in, {Nc, S2 * Hc});
    auto in2 = GetVLAPtr<T>(t_in2, {Nk, S2 * Hk});
    auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
    auto bias = GetVLAPtr<T>(t_bias, {Hk});
    auto gamma = GetVLAPtr<LT>(t_gamma, {Hk});
    auto beta = GetVLAPtr<LT>(t_beta, {Hk});
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
    auto layer_norm_fwd_tpp =
        SCOPEIT((LayerNormFwdTPP<T, LT>(Nk, S2, Hk, eps)), LAYER_NORM);

    {
      RECORD_SCOPE(o_gemm, {t_in, t_wt});
      auto loop_scheme = large_cache_opt ? "acB" : "aBC";
      auto ogemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{S1}, LoopSpecs{Nk}},
          loop_scheme);
      bool parallelized_on_nk =
          large_cache_opt ? false : true; // ogemm_loop.is_parallel(2);
      ogemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
            if (nc == 0) {
              copy_bias_tpp(bias[nk], out[s1][nk]);
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in2[s1][nk], out[s1][nk]);
              if (!parallelized_on_nk && nk == Nk - 1) {
                layer_norm_fwd_tpp(
                    out[s1][0],
                    gamma[0],
                    beta[0],
                    nullptr,
                    nullptr,
                    out[s1][0]);
              }
            }
          },
          [&]() { brgemm_tpp.config(); },
          [&]() { brgemm_tpp.release(); });

      if (parallelized_on_nk) {
#pragma omp parallel for
        for (int s1 = 0; s1 < S1; s1++) {
          layer_norm_fwd_tpp(
              out[s1][0], gamma[0], beta[0], nullptr, nullptr, out[s1][0]);
        }
      }
    }
  }

  template <typename T>
  inline void intermediate(
      at::Tensor t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      at::Tensor& t_out) {
    auto in_sizes = t_in.sizes();
    auto wt_sizes = t_wt.sizes();
    auto S1 = in_sizes[0];
    auto Nc = in_sizes[1];
    auto S2 = in_sizes[2];
    auto Hc = in_sizes[3];

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
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{S1}, LoopSpecs{Nk}},
          loop_scheme);
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

  template <typename T, typename LT = T>
  at::Tensor& forward(
      at::Tensor& t_HS,
      std::vector<at::Tensor>& t_masks,
      at::Tensor& t_Out,
      std::vector<at::Tensor>& t_scratch) {
    auto& t_QL = t_scratch[0];
    auto& t_KL_TV = t_scratch[1];
    auto& t_VL_V = t_scratch[2];
    auto& t_CL = t_scratch[3];
    auto& t_SO = t_scratch[4];
    auto& t_I = t_scratch[5];
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    auto t_HS_qkv = t_HS;

    if (use_at_vnni_local) {
      auto sizes = t_HS.sizes();
      long S1 = sizes[0];
      long F1 = sizes[1];
      long S2 = sizes[2];
      long F2 = sizes[3];
      t_HS_qkv = act_tensor_trans_n2v(S1, F1, S2, F2, t_HS);
    }

    q_gemm<T>(t_HS_qkv, t_Wq, t_Bq, t_QL);
    k_gemm<T>(t_HS_qkv, t_Wk, t_Bk, t_KL_TV);
    v_gemm<T>(t_HS_qkv, t_Wv, t_Bv, t_VL_V);

    self_attn<T>(t_QL, t_KL_TV, t_masks, t_VL_V, t_CL);
    output<T, LT>(t_CL, t_HS, t_Wso, t_Bso, t_Gso, t_Beta_so, t_SO);
    intermediate<T>(t_SO, t_Wi, t_Bi, t_I);
    output<T, LT>(t_I, t_SO, t_Wo, t_Bo, t_Go, t_Beta_o, t_Out);

    return t_Out;
  }
};

class __attribute__((visibility("hidden"))) BertEncoder {
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
    int nLayers = params.size();
    layers.reserve(nLayers);
    for (int i = 0; i < nLayers; i++) {
      layers.push_back(new BertEncoderLayer(params[i], eps, N, H));
    }
  }

  ~BertEncoder() {
    for (int i = 0; i < (int)layers.size(); i++) {
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
#ifdef PYTORCH_SUPPORTS_FLOAT8
      } else if (dt == at::kBFloat8 && ldt == at::kFloat) {
        t_HS = l->forward<bfloat8, float>(t_HS, t_masks, t_HS, t_scratch);
      } else if (dt == at::kHFloat8 && ldt == at::kFloat) {
        t_HS = l->forward<hfloat8, float>(t_HS, t_masks, t_HS, t_scratch);
#endif
      } else {
        TPP_ASSERT(0, "Should not come here\n");
      }
    }
    // printf("Returning Layer \n");

    return t_HS;
  }
};

REGISTER_SUBMODULE(_fused_bert_infer, m) {
  py::class_<BertEncoder>(m, "BertEncoder")
      .def(py::init<
           std::vector<std::vector<at::Tensor>>,
           float,
           long,
           long,
           long>())
      //.def("setMasks", &BertEncoder::setMasks)
      .def("forward", &BertEncoder::forward);
}
