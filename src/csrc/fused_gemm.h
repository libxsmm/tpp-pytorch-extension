/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#pragma once

#include <ATen/record_function.h>
#include <torch/extension.h>

#include "ext_tpp.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include "qtypes.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;

extern int TPP_VERBOSE;
static int CK_BLOCK_SIZE = env2int("CK_BLOCK_SIZE", 64);
static int BSB_BLOCK_SIZE = env2int("BSB_BLOCK_SIZE", 64);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static int MAX_HC_SIZE = env2int("MAX_HC_SIZE", 128);
static const char* GEMM_LOOP_SCHEME_REUSE =
    getenv("GEMM_LOOP_SCHEME_REUSE") ? getenv("GEMM_LOOP_SCHEME_REUSE") : "aCB";
static const char* GEMM_LOOP_SCHEME_STREAMING =
    getenv("GEMM_LOOP_SCHEME_STREAMING") ? getenv("GEMM_LOOP_SCHEME_STREAMING")
                                         : "aCb";

REGISTER_LOCAL_SCOPE(pln_gemm, "pln_gemm");
REGISTER_LOCAL_SCOPE(gemm, "gemm");

template <typename T, typename TOUT>
class TppFlatLinearBase {
 public:
  using Tin = T;
  using Tout = TOUT;
  using Tbias =
      typename std::conditional<std::is_same_v<Tin, int8_t>, Tout, Tin>::type;

 protected:
  static constexpr int nOutputShapes = 4;
  long Nc, Hc[2], Nk, Hk[2], Ncb[2], BS, BSb[2];
  bool weight_reuse, Atrans, Btrans;
  long C, K, aligned_K;
  SCOPEIT_DECL(CpyBiasTPP<Tbias, Tout>) copy_bias_tpp[2][2];
  SCOPEIT_DECL(SetZeroTPP<Tout>) zero_tpp[2][2];

  std::string loop_scheme;
  std::function<void(const VLAPtr<T, 2, long>&, long, long)>
      postOpCBs[nOutputShapes];

 public:
  TppFlatLinearBase(
      at::Tensor t_in,
      at::Tensor t_wt,
      at::Tensor t_bias,
      long _Hc = 0,
      long _Hk = 0) {
    std::tie(
        Nc,
        Hc[0],
        Hc[1],
        Nk,
        Hk[0],
        Hk[1],
        Ncb[0],
        BS,
        BSb[0],
        BSb[1],
        weight_reuse,
        Atrans,
        Btrans) = getBlockingParams(t_in, t_wt, t_bias, _Hc, _Hk);
    C = Nc * Hc[0] + Hc[1];
    K = Nk * Hk[0] + Hk[1];
    aligned_K = Hk[1] == 0 ? K : K + Hk[0];
    if (Ncb[0] > 0) {
      Ncb[1] = Nc % Ncb[0];
    } else {
      Ncb[1] = 0;
    }

    for (int b = 0; b < 2; b++) {
      if (BSb[b] == 0)
        continue;
      for (int k = 0; k < 2; k++) {
        if (Hk[k] == 0)
          continue;
        copy_bias_tpp[b][k] =
            SCOPEIT((CpyBiasTPP<Tbias, Tout>(BSb[b], Hk[k], K)), BIAS);
        zero_tpp[b][k] = SCOPEIT(SetZeroTPP<Tout>(BSb[b], Hk[k], K), EW_ZERO);
      }
    }
  }
  template <typename T1 = Tout>
  VLAPtr<T1, 2, long> getOutputVLAPtr(at::Tensor t) {
    return GetVLAPtr<T1>(t, {K, 1});
  }
  int numOutputShapes() {
    return nOutputShapes;
  }
  std::tuple<long, long, long> getOutputShape(int i) {
    int b = i / 2;
    int k = i % 2;
    if (i < nOutputShapes)
      return std::make_tuple(BSb[b], Hk[k], K);
    else
      TPP_ASSERT(false, "Invalid Index");
  }

  void setPostOpCB(
      int i,
      const std::function<void(const VLAPtr<Tout, 2, long>&, long, long)>& f) {
    TPP_ASSERT(i < nOutputShapes, "Invalid Index");
    postOpCBs[i] = f;
  }

  at::Tensor new_empty(at::Tensor t_in) {
    auto sizes = t_in.sizes().vec();
    auto dim = t_in.dim();
    sizes[dim - 1] = K;
    return t_in.new_empty(sizes, c10::CppTypeToScalarType<Tout>::value);
  }

  static std::tuple<
      long,
      long,
      long,
      long,
      long,
      long,
      long,
      long,
      long,
      long,
      bool,
      bool,
      bool>
  getBlockingParams(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      long _Hc,
      long _Hk) {
    long Nc, Nk, Hc[2], Hk[2], Ncb, BSb[2];
    // auto in_sizes = t_in.sizes();
    auto wt_sizes = t_wt.sizes();
    // auto in_dim = t_in.dim();
    TORCH_CHECK(t_in.is_contiguous() || t_in.t().is_contiguous());
    TORCH_CHECK(wt_sizes.size() == 2);
    TORCH_CHECK(t_wt.is_contiguous() || t_wt.t().is_contiguous());
    bool Atrans = t_in.is_contiguous() ? false : true;
    bool Btrans = t_wt.is_contiguous() ? false : true;
    auto C = wt_sizes[0];
    auto K = wt_sizes[1];
    auto BS = t_in.numel() / C;
    const int default_block_size = CK_BLOCK_SIZE;

    if (_Hc == 0)
      _Hc = C <= default_block_size ? C : default_block_size;
    if (_Hk == 0)
      _Hk = K <= default_block_size ? K : default_block_size;
    Hc[0] = _Hc;
    Hc[1] = C % Hc[0];
    Nc = C / Hc[0];

    Hk[0] = _Hk;
    Hk[1] = K % Hk[0];
    Nk = K / Hk[0];

    Ncb = Nc;
    BSb[0] = BS <= BSB_BLOCK_SIZE ? BS : BSB_BLOCK_SIZE;
    BSb[1] = BS % BSb[0];
    auto nBS = BS / BSb[0];
    bool weight_reuse = nBS > 4;
    if (weight_reuse)
      Ncb = NCB_BLOCK_SIZE;
    return std::make_tuple(
        Nc,
        Hc[0],
        Hc[1],
        Nk,
        Hk[0],
        Hk[1],
        Ncb,
        BS,
        BSb[0],
        BSb[1],
        weight_reuse,
        Atrans,
        Btrans);
  }

  static inline const std::string extra(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return "";
  }

  void operator()(
      at::Tensor t_in,
      at::Tensor t_wt_V,
      at::Tensor t_bias,
      at::Tensor t_out) {
    TPP_ASSERT(false, "Should not come here\n");
  }

  static TppFlatLinearBase<Tin, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      long _Hc = 0,
      long _Hk = 0) {
    TPP_ASSERT(false, "Should not come here\n");
    return TppFlatLinearBase<Tin, Tout>(t_in, t_wt, t_bias, _Hc, _Hk);
  }

 protected:
  template <typename GemmT>
  static GemmT _get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      long _Hc = 0,
      long _Hk = 0) {
    static ska::flat_hash_map<std::string, GemmT*> gemm_cache;
    long Nc, Hc[2], Nk, Hk[2], Ncb, BS, BSb[2];
    bool weight_reuse, Atrans, Btrans;
    std::tie(
        Nc,
        Hc[0],
        Hc[1],
        Nk,
        Hk[0],
        Hk[1],
        Ncb,
        BS,
        BSb[0],
        BSb[1],
        weight_reuse,
        Atrans,
        Btrans) = getBlockingParams(t_in, t_wt, t_bias, _Hc, _Hk);
    auto extra = GemmT::extra(t_in, t_wt, t_bias);
    char hash[200] = "";
    snprintf(
        hash,
        199,
        "gemm_Nc%ld_Hc0:%ld_Hc1:%ld_Nk%ld_Hk0:%ld_Hk1:%ld_BS%ld_Bsb0:%ld_BSb1:%ld_Ncb%ld_wr%d_at%d_bt%d_%s",
        Nc,
        Hc[0],
        Hc[1],
        Nk,
        Hk[0],
        Hk[1],
        Atrans ? BS : 0,
        BSb[0],
        BSb[1],
        Ncb,
        weight_reuse ? 1 : 0,
        Atrans ? 1 : 0,
        Btrans ? 1 : 0,
        extra.c_str());
    auto search = gemm_cache.find(hash);
    GemmT* gemm = NULL;
    if (search != gemm_cache.end())
      gemm = search->second;
    if (gemm == NULL) {
      gemm = new GemmT(t_in, t_wt, t_bias, _Hc, _Hk);
      gemm_cache[hash] = gemm;
      if (TPP_VERBOSE > 0) {
        printf("%s:\nHash: %s\n", get_class_name<GemmT>().c_str(), hash);
      }
    }
    return *gemm;
  }
};

template <typename T, typename TW, typename TOUT = T>
class TppFlatLinear : public TppFlatLinearBase<T, TOUT> {
 public:
  using Tin = T;
  using Tout = TOUT;
  using Tw = TW;
  using Base = TppFlatLinearBase<Tin, Tout>;
  using Tbias = typename Base::Tbias;
  using Base::Atrans;
  using Base::BS;
  using Base::BSb;
  using Base::Btrans;
  using Base::C;
  using Base::Hc;
  using Base::Hk;
  using Base::K;
  using Base::Nc;
  using Base::Ncb;
  using Base::Nk;
  using Base::loop_scheme;
  using Base::postOpCBs;
  using Base::weight_reuse;

 protected:
  SCOPEIT_DECL(BrgemmTPP<T, Tout, Tw>) brgemm_tpp[2][2][2];

 public:
  TppFlatLinear(
      at::Tensor t_in,
      at::Tensor t_wt,
      at::Tensor t_bias,
      long _Hc = 0,
      long _Hk = 0)
      : TppFlatLinearBase<Tin, Tout>(t_in, t_wt, t_bias, _Hc, _Hk) {
    int b_vnni = 0;
    int a_vnni = 0;
    if (t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockMxFP) {
      if (t_wt.dtype() == at::kQUInt4x2) {
        // b_vnni = 4;
        throw std::runtime_error("FlatGemm Wt: at::kQUInt4x2 not supported\n");
        printf("FlatGemm Using Wt: at::kQUInt4x2\n");
      } else {
        TPP_ASSERT(false, "Unsupported qtype\n");
      }
    }

    for (int b = 0; b < 2; b++) {
      if (BSb[b] == 0)
        continue;
      for (int k = 0; k < 2; k++) {
        if (Hk[k] == 0)
          continue;
        for (int c = 0; c < 2; c++) {
          if (Hc[c] == 0)
            continue;
          auto str_a = Atrans ? Hc[0] * BS : Hc[0];
          auto str_b = Btrans ? Hc[0] : Hc[0] * K;
          auto lda = Atrans ? BS : C;
          auto ldb = Btrans ? C : K;
          auto ldc = K;
          auto beta = 1.0f;
          int a_trans = Atrans ? 1 : 0;
          int b_trans = Btrans ? 1 : 0;
          int uh = Ncb[0];

          // printf(
          //     "MNK=%ld %ld %ld, str= %ld %ld, ld= %ld %ld %ld\n",
          //     BSb[b],
          //     Hk[k],
          //     Hc[c],
          //     str_a,
          //     str_b,
          //     lda,
          //     ldb,
          //     ldc);
          brgemm_tpp[b][k][c] = SCOPEITGEMM((BrgemmTPP<T, Tout, Tw>(
              BSb[b],
              Hk[k],
              Hc[c],
              str_a,
              str_b,
              lda,
              ldb,
              ldc,
              beta,
              a_trans,
              uh,
              b_vnni,
              b_trans,
              a_vnni)));
        }
      }
    }

    loop_scheme =
        weight_reuse ? GEMM_LOOP_SCHEME_REUSE : GEMM_LOOP_SCHEME_STREAMING;
  }
  std::function<void(int, int, int)> stepFunc(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias,
      at::Tensor& t_out,
      long BS) {
    auto bias = GetVLAPtr<Tbias>(t_bias, {1});
    auto out = GetVLAPtr<Tout>(t_out, {K, 1});
    bool with_bias = (t_bias.numel() > 0);
    if (!t_wt.is_quantized()) {
      if (!Atrans && !Btrans) {
        auto in = GetVLAPtr<T>(t_in, {C, 1});
        auto wt = GetVLAPtr<Tw>(t_wt, {K, 1});
        auto func = [&, in, wt, bias, out, BS, with_bias ](int c, int s, int k)
            __attribute__((always_inline)) {
          auto rb = s + BSb[0] > BS ? 1 : 0;
          auto rc = c + Ncb[0] * Hc[0] > C ? 1 : 0;
          auto rk = k + Hk[0] > K ? 1 : 0;
          auto count = Ncb[rc];
          bool no_tilecfg = !(rb || rk);
          if (c == 0) {
            if (with_bias) {
              this->copy_bias_tpp[rb][rk](bias[k], out[s][k]);
            } else {
              this->zero_tpp[rb][rk](out[s][k]);
            }
          }
          if (count > 0) {
            brgemm_tpp[rb][rk][0](
                in[s][c], wt[c][k], out[s][k], count, no_tilecfg);
          }
          if (rc == 1 && Hc[1] > 0) {
            // Adjust c to account for brgemm call above
            c += count * Hc[0];
            brgemm_tpp[rb][rk][1](in[s][c], wt[c][k], out[s][k], 1, false);
          }
          if (!(c + Ncb[0] * Hc[0] < C)) { // last c iter
            if (postOpCBs[rb * 2 + rk])
              postOpCBs[rb * 2 + rk](out, s, k);
          }
        };
        return func;
      } else if (!Atrans && Btrans) {
        auto in = GetVLAPtr<T>(t_in, {C, 1});
        auto wt = GetVLAPtr<Tw>(t_wt.t(), {C, 1});
        auto func = [&, in, wt, bias, out, BS, with_bias ](int c, int s, int k)
            __attribute__((always_inline)) {
          auto rb = s + BSb[0] > BS ? 1 : 0;
          auto rc = c + Ncb[0] * Hc[0] > C ? 1 : 0;
          auto rk = k + Hk[0] > K ? 1 : 0;
          auto count = Ncb[rc];
          bool no_tilecfg = !(rb || rk);
          if (c == 0) {
            if (with_bias) {
              this->copy_bias_tpp[rb][rk](bias[k], out[s][k]);
            } else {
              this->zero_tpp[rb][rk](out[s][k]);
            }
          }
          if (count > 0) {
            brgemm_tpp[rb][rk][0](
                in[s][c], wt[k][c], out[s][k], count, no_tilecfg);
          }
          if (rc == 1 && Hc[1] > 0) {
            c += count * Hc[0];
            brgemm_tpp[rb][rk][1](in[s][c], wt[k][c], out[s][k], 1, false);
          }
          if (!(c + Ncb[0] * Hc[0] < C)) { // last c iter
            if (postOpCBs[rb * 2 + rk])
              postOpCBs[rb * 2 + rk](out, s, k);
          }
        };
        return func;
      } else if (Atrans && !Btrans) {
        auto in = GetVLAPtr<T>(t_in.t(), {BS, 1});
        auto wt = GetVLAPtr<Tw>(t_wt, {K, 1});
        auto func = [&, in, wt, bias, out, BS, with_bias ](int c, int s, int k)
            __attribute__((always_inline)) {
          auto rb = s + BSb[0] > BS ? 1 : 0;
          auto rc = c + Ncb[0] * Hc[0] > C ? 1 : 0;
          auto rk = k + Hk[0] > K ? 1 : 0;
          auto count = Ncb[rc];
          bool no_tilecfg = !(rb || rk);
          if (c == 0) {
            if (with_bias) {
              this->copy_bias_tpp[rb][rk](bias[k], out[s][k]);
            } else {
              this->zero_tpp[rb][rk](out[s][k]);
            }
          }
          if (count > 0) {
            brgemm_tpp[rb][rk][0](
                in[c][s], wt[c][k], out[s][k], count, no_tilecfg);
          }
          if (rc == 1 && Hc[1] > 0) {
            c += count * Hc[0];
            brgemm_tpp[rb][rk][1](in[c][s], wt[c][k], out[s][k], 1, false);
          }
          if (!(c + Ncb[0] * Hc[0] < C)) { // last c iter
            if (postOpCBs[rb * 2 + rk])
              postOpCBs[rb * 2 + rk](out, s, k);
          }
        };
        return func;
      } else {
        auto in = GetVLAPtr<T>(t_in.t(), {BS, 1});
        auto wt = GetVLAPtr<Tw>(t_wt.t(), {C, 1});
        auto func = [&, in, wt, bias, out, BS, with_bias ](int c, int s, int k)
            __attribute__((always_inline)) {
          auto rb = s + BSb[0] > BS ? 1 : 0;
          auto rc = c + Ncb[0] * Hc[0] > C ? 1 : 0;
          auto rk = k + Hk[0] > K ? 1 : 0;
          auto count = Ncb[rc];
          bool no_tilecfg = !(rb || rk);
          if (c == 0) {
            if (with_bias) {
              this->copy_bias_tpp[rb][rk](bias[k], out[s][k]);
            } else {
              this->zero_tpp[rb][rk](out[s][k]);
            }
          }
          if (count > 0) {
            brgemm_tpp[rb][rk][0](
                in[c][s], wt[k][c], out[s][k], count, no_tilecfg);
          }
          if (rc == 1 && Hc[1] > 0) {
            c += count * Hc[0];
            brgemm_tpp[rb][rk][1](in[c][s], wt[k][c], out[s][k], 1, false);
          }
          if (!(c + Ncb[0] * Hc[0] < C)) { // last c iter
            if (postOpCBs[rb * 2 + rk])
              postOpCBs[rb * 2 + rk](out, s, k);
          }
        };
        return func;
      }
    } else {
      throw std::runtime_error("QTensor not supported yet by TppFlatLinear\n");
    }
  }

  void operator()(
      at::Tensor t_in,
      at::Tensor t_wt_V,
      at::Tensor t_bias,
      at::Tensor t_out) {
    // t_in = t_in.contiguous();

    auto BS = t_in.numel() / this->C;
    auto func = stepFunc(t_in, t_wt_V, t_bias, t_out, BS);
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0L, C, Ncb[0] * Hc[0], false},
           LoopSpecs{0L, BS, BSb[0]},
           LoopSpecs{0L, K, Hk[0]}},
          loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int c = ind[0], s = ind[1], k = ind[2];
            func(c, s, k);
          },
          [&]() {
            TimerStart();
            brgemm_tpp[0][0][0].config();
          },
          [&]() {
            brgemm_tpp[0][0][0].release();
            TimerEnd();
          });
    }
  }

  static void fused_gemm(
      std::vector<TppFlatLinear<T, Tw, Tout>>& gemms,
      at::Tensor& t_in,
      std::vector<at::Tensor>& t_wt,
      std::vector<at::Tensor>& t_bias,
      std::vector<at::Tensor>& t_out) {
    int n_gemms = gemms.size();

    // Check if gemms are fusable
    long Hc = gemms[0].Hc[0];
    long Ncb = gemms[0].Ncb[0];
    long BSb = gemms[0].BSb[0];
    long Hk0 = gemms[0].Hk[0];
    long totalN = gemms[0].aligned_K;
    bool fusable = true;
    for (int i = 1; i < n_gemms; i++) {
      auto& g = gemms[i];
      if (g.Hc[0] != Hc || g.Ncb[0] != Ncb || g.BSb[0] != BSb ||
          g.Hk[0] != Hk) {
        fusable = false;
        break;
      }
      totalN += g.aligned_K;
    }

    if (!fusable) {
      for (int i = 0; i < n_gemms; i++) {
        gemms[i](t_in, t_wt[i], t_bias[i], t_out[i]);
      }
    } else {
      auto C = gemms[0].C;
      auto BS = t_in.numel() / C;
      auto loop_scheme = gemms[0].loop_scheme;
      std::vector<std::function<void(int, int, int)>> funcs;
      for (int i = 0; i < n_gemms; i++) {
        auto& g = gemms[i];
        funcs.push_back(g.stepFunc(t_in, t_wt[i], t_bias[i], t_out[i], BS));
      }
      {
        RECORD_OMP_TIME();
        auto gemm_loop = ThreadedLoop<3>(
            {LoopSpecs{0L, C, Ncb * Hc, false},
             LoopSpecs{0L, BS, BSb},
             LoopSpecs{0L, totalN, Hk}},
            loop_scheme);
        gemm_loop(
            [&](int* ind) {
              int c = ind[0], s = ind[1], k = ind[2];
              int i = 0;
              while (k >= gemms[i].aligned_K) {
                k -= gemms[i].aligned_K;
                i++;
              }
              funcs[i](c, s, k);
            },
            [&]() {
              TimerStart();
              gemms[0].brgemm_tpp[0][0][0].config();
            },
            [&]() {
              gemms[0].brgemm_tpp[0][0][0].release();
              TimerEnd();
            });
      }
    }
  }

  static TppFlatLinear<T, Tw, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return Base::template _get<TppFlatLinear<T, Tw, Tout>>(t_in, t_wt, t_bias);
  }
};

template <typename T, typename TOUT>
class TppBlockedLinearWBase {
 public:
  using Tin = T;
  using Tout = TOUT;
  using Tbias =
      typename std::conditional<std::is_same_v<Tin, int8_t>, Tout, Tin>::type;

 protected:
  static constexpr int nOutputShapes = 2;
  long Nc, Hc, Nk, Hk, Ncb, BSb, rem;
  bool weight_reuse;
  long C, K;
  SCOPEIT_DECL(CpyBiasTPP<Tbias, Tout>) copy_bias_tpp, copy_bias_tpp_rem;
  SCOPEIT_DECL(SetZeroTPP<Tout>) zero_tpp, zero_tpp_rem;

  std::string loop_scheme;
  std::function<void(const VLAPtr<Tout, 2, long>&, long, long)>
      postOpCBs[nOutputShapes];

 public:
  TppBlockedLinearWBase(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
    std::tie(Nc, Hc, Nk, Hk, Ncb, BSb, rem, weight_reuse) =
        getBlockingParams(t_in, t_wt, t_bias);
    C = Nc * Hc;
    K = Nk * Hk;

    copy_bias_tpp = SCOPEIT((CpyBiasTPP<Tbias, Tout>(BSb, Hk, K)), BIAS);
    copy_bias_tpp_rem = SCOPEIT((CpyBiasTPP<Tbias, Tout>(rem, Hk, K)), BIAS);
    zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
    zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  }
  template <typename T1 = Tout>
  VLAPtr<T1, 2, long> getOutputVLAPtr(at::Tensor t) {
    return GetVLAPtr<T1>(t, {Nk, Hk});
  }
  int numOutputShapes() {
    return nOutputShapes;
  }
  std::tuple<long, long, long> getOutputShape(int i) {
    if (i == 0)
      return std::make_tuple(BSb, Hk, K);
    else if (i == 1)
      return std::make_tuple(rem, Hk, K);
    else
      TPP_ASSERT(false, "Invalid Index");
  }

  void setPostOpCB(
      int i,
      const std::function<void(const VLAPtr<Tout, 2, long>&, long, long)>& f) {
    TPP_ASSERT(i < nOutputShapes, "Invalid Index");
    postOpCBs[i] = f;
  }

  at::Tensor new_empty(at::Tensor t_in) {
    auto sizes = t_in.sizes().vec();
    auto dim = t_in.dim();
    sizes[dim - 1] = K;
    return t_in.new_empty(sizes, c10::CppTypeToScalarType<Tout>::value);
  }

  static std::tuple<long, long, long, long, long, long, long, bool>
  getBlockingParams(at::Tensor& t_in, at::Tensor& t_wt, at::Tensor& t_bias) {
    long Nc, Nk, Hc, Hk, Ncb, BSb, rem;
    auto in_sizes = t_in.sizes();
    auto wt_sizes = t_wt.sizes();
    auto in_dim = t_in.dim();
    auto C = in_sizes[in_dim - 1];
    auto BS = t_in.numel() / C;

    Nc = wt_sizes[1];
    Hc = C / Nc;
    if (Hc > MAX_HC_SIZE && Hc % MAX_HC_SIZE == 0) {
      Hc = MAX_HC_SIZE;
      Nc = C / Hc;
    }
    Nk = wt_sizes[0];
    Hk = wt_sizes[3];

    Ncb = Nc;
    BSb = BSB_BLOCK_SIZE;
    rem = BS % BSb;
    auto nBS = BS / BSb;
    bool weight_reuse = nBS > 4;
    if (weight_reuse)
      Ncb = NCB_BLOCK_SIZE;
    return std::make_tuple(Nc, Hc, Nk, Hk, Ncb, BSb, rem, weight_reuse);
  }

  static inline const std::string extra(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return "";
  }

  void operator()(
      at::Tensor t_in,
      at::Tensor t_wt_V,
      at::Tensor t_bias,
      at::Tensor t_out) {
    TPP_ASSERT(false, "Should not come here\n");
  }

  static TppBlockedLinearWBase<Tin, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    TPP_ASSERT(false, "Should not come here\n");
    return TppBlockedLinearWBase<Tin, Tout>(t_in, t_wt, t_bias);
  }

 protected:
  template <typename GemmT>
  static GemmT _get(at::Tensor& t_in, at::Tensor& t_wt, at::Tensor& t_bias) {
    static ska::flat_hash_map<std::string, GemmT*> gemm_cache;
    long Nc, Hc, Nk, Hk, Ncb, BSb, rem;
    bool weight_reuse;
    std::tie(Nc, Hc, Nk, Hk, Ncb, BSb, rem, weight_reuse) =
        getBlockingParams(t_in, t_wt, t_bias);
    auto extra = GemmT::extra(t_in, t_wt, t_bias);
    char hash[200] = "";
    snprintf(
        hash,
        199,
        "gemm_Nc%ld_Hc%ld_Nk%ld_Hk%ld_Bsb%ld_rem%ld_Ncb%ld_wr%d_%s",
        Nc,
        Hc,
        Nk,
        Hk,
        BSb,
        rem,
        Ncb,
        weight_reuse ? 1 : 0,
        extra.c_str());
    auto search = gemm_cache.find(hash);
    GemmT* gemm = NULL;
    if (search != gemm_cache.end())
      gemm = search->second;
    if (gemm == NULL) {
      gemm = new GemmT(t_in, t_wt, t_bias);
      gemm_cache[hash] = gemm;
      if (TPP_VERBOSE > 0) {
        printf("%s:\nHash: %s\n", get_class_name<GemmT>().c_str(), hash);
      }
    }
    return *gemm;
  }
};

template <typename T, typename TW, typename TOUT = T>
class TppBlockedLinearW : public TppBlockedLinearWBase<T, TOUT> {
 public:
  using Tin = T;
  using Tout = TOUT;
  using Tw = TW;
  using Base = TppBlockedLinearWBase<Tin, Tout>;
  using Tbias = typename Base::Tbias;
  using Base::BSb;
  using Base::C;
  using Base::Hc;
  using Base::Hk;
  using Base::K;
  using Base::Nc;
  using Base::Ncb;
  using Base::Nk;
  using Base::loop_scheme;
  using Base::postOpCBs;
  using Base::rem;
  using Base::weight_reuse;

 protected:
  SCOPEIT_DECL(BrgemmTPP<T, Tout, Tw>) brgemm_tpp, brgemm_tpp_rem;

 public:
  TppBlockedLinearW(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias)
      : TppBlockedLinearWBase<Tin, Tout>(t_in, t_wt, t_bias) {
    int b_vnni = 1;
    // printf("Using class: %s\n", get_class_name<decltype(*this)>().c_str());
    if (t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockMxFP) {
      if (t_wt.dtype() == at::kQUInt4x2) {
        b_vnni = t_wt.size(-1);
      } else {
        TPP_ASSERT(false, "Unsupported qtype\n");
      }
    }

    brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout, Tw>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
    brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout, Tw>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));

    loop_scheme =
        weight_reuse ? GEMM_LOOP_SCHEME_REUSE : GEMM_LOOP_SCHEME_STREAMING;
  }
  std::function<void(int, int, int)> stepFunc(
      at::Tensor& t_in,
      at::Tensor& t_wt_V,
      at::Tensor& t_bias,
      at::Tensor& t_out,
      long BS) {
    auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
    auto bias = GetVLAPtr<Tbias>(t_bias, {Hk});
    auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});
    bool with_bias = (t_bias.numel() > 0);
    if (!t_wt_V.is_quantized()) {
      auto wt_V = GetVLAPtr<Tw>(t_wt_V, {Nc, Hc * Hk});
      auto func = [&, in, wt_V, bias, out, BS, with_bias ](
          int nc, int s1, int nk) __attribute__((always_inline)) {
        auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
        bool is_rem = (s1 + BSb > BS);
        if (!is_rem) {
          if (nc == 0) {
            if (with_bias) {
              this->copy_bias_tpp(bias[nk], out[s1][nk]);
            } else {
              this->zero_tpp(out[s1][nk]);
            }
          }
          brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          if (!(nc + Ncb < Nc)) { // last nc iter
            if (postOpCBs[0])
              postOpCBs[0](out, s1, nk);
          }
        } else {
          if (nc == 0) {
            if (with_bias) {
              this->copy_bias_tpp_rem(bias[nk], out[s1][nk]);
            } else {
              this->zero_tpp_rem(out[s1][nk]);
            }
          }
          brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
          if (!(nc + Ncb < Nc)) { // last nc iter
            if (postOpCBs[1])
              postOpCBs[1](out, s1, nk);
          }
        }
      };
      return func;
    } else {
      if (t_wt_V.qscheme() == at::kPerBlockMxFP) {
        // We are using kQUInt4x2 for MXFP4 and uint8_t for scale

        TPP_ASSERT((std::is_same_v<Tw, uint8_t>), "MXFP use uint8_t\n");
        auto quantizer = at::get_qtensorimpl(t_wt_V)->quantizer();
        auto mxfp_quantizer =
            static_cast<at::PerBlockMxFPQuantizer*>(quantizer.get());
        auto pack_size = mxfp_quantizer->pack_size();
        auto block_size = mxfp_quantizer->block_size();
        auto wt_V = GetVLAPtr<Tw>(t_wt_V, {Nc, (Hc * Hk) / pack_size});
        auto t_scl = mxfp_quantizer->scales();
        auto scl = GetVLAPtr<Tw>(t_scl, {Nc, (Hc * Hk) / block_size});
        at::Tensor t_i_scl = t_in.new_empty({0}, at::kFloat);
        bool is_input_quantized = t_in.is_quantized();
        if (is_input_quantized) {
          t_i_scl = q_per_block_scales(t_in);
          TPP_ASSERT(
              t_in.qscheme() == at::kPerBlockAffine,
              "Unsupported input qscheme\n");
        }
        auto i_scl = GetVLAPtr<float>(t_i_scl, {Nc});
        auto func = [&, in, wt_V, scl, i_scl, bias, out, BS, with_bias, is_input_quantized ](
            int nc, int s1, int nk) __attribute__((always_inline)) {
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          float* i_scl_ptr = nullptr;
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                this->copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                this->zero_tpp(out[s1][nk]);
              }
            }
            if (is_input_quantized) {
              i_scl_ptr = &i_scl[s1][nc];
            }
            brgemm_tpp(
                in[s1][nc],
                i_scl_ptr,
                wt_V[nk][nc],
                scl[nk][nc],
                out[s1][nk],
                count,
                true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              if (postOpCBs[0])
                postOpCBs[0](out, s1, nk);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                this->copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                this->zero_tpp_rem(out[s1][nk]);
              }
            }
            if (is_input_quantized) {
              i_scl_ptr = &i_scl[s1][nc];
            }
            brgemm_tpp_rem(
                in[s1][nc],
                i_scl_ptr,
                wt_V[nk][nc],
                scl[nk][nc],
                out[s1][nk],
                count,
                false);
            if (!(nc + Ncb < Nc)) { // last nc iter
              if (postOpCBs[1])
                postOpCBs[1](out, s1, nk);
            }
          }
        };
        return func;
      } else {
        TPP_ASSERT(false, "Unsupported qscheme 1\n");
      }
    }
  }

  void operator()(
      at::Tensor t_in,
      at::Tensor t_wt_V,
      at::Tensor t_bias,
      at::Tensor t_out) {
    t_in = t_in.contiguous();

    auto BS = t_in.numel() / this->C;
    auto t_qin = t_in;
    if (std::is_same<Tw, uint8_t>::value && std::is_same<Tin, int8_t>::value) {
      TPP_ASSERT(
          t_wt_V.size(-1) == 8, "Unsupported vnni packing for weights\n");
      if (t_in.is_quantized()) {
        TORCH_CHECK(t_qin.is_contiguous());
        TORCH_CHECK(t_qin.dtype() == at::kQInt8);
      } else {
        t_in = t_in.contiguous();
        t_qin =
            quantize_int8sym(t_in, q_per_block_block_size(t_wt_V), -1, false);
        auto scl = q_per_block_scales(t_qin);
        scl.mul_(6.0 / 127.0);
      }
    }
    auto func = stepFunc(t_qin, t_wt_V, t_bias, t_out, BS);
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
          loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            func(nc, s1, nk);
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

  static void fused_gemm(
      std::vector<TppBlockedLinearW<T, Tw, Tout>>& gemms,
      at::Tensor& t_in,
      std::vector<at::Tensor>& t_wt_V,
      std::vector<at::Tensor>& t_bias,
      std::vector<at::Tensor>& t_out) {
    int n_gemms = gemms.size();
    long totalN = 0;
    auto BS = t_in.numel() / gemms[0].C;
    long Nc = gemms[0].Nc;
    long Ncb = gemms[0].Ncb;
    long BSb = gemms[0].BSb;
    auto loop_scheme = gemms[0].loop_scheme;
    std::vector<std::function<void(int, int, int)>> funcs;
    auto t_qin = t_in;
    if (std::is_same<Tw, uint8_t>::value && std::is_same<Tin, int8_t>::value) {
      // TPP_ASSERT(t_wt_V[0].size(-1) == 8, "Unsupported vnni packing for
      // weights\n");
      if (t_in.is_quantized()) {
        TORCH_CHECK(t_qin.is_contiguous());
        TORCH_CHECK(t_qin.dtype() == at::kQInt8);
      } else {
        t_in = t_in.contiguous();
        t_qin = quantize_int8sym(
            t_in, q_per_block_block_size(t_wt_V[0]), -1, false);
        auto scl = q_per_block_scales(t_qin);
        scl.mul_(6.0 / 127.0);
      }
    }
    for (int i = 0; i < n_gemms; i++) {
      auto& g = gemms[i];
      funcs.push_back(g.stepFunc(t_qin, t_wt_V[i], t_bias[i], t_out[i], BS));
      totalN += g.Nk;
      TPP_ASSERT(
          g.Nc == Nc && g.Ncb == Ncb && g.BSb == BSb,
          "Fused QKV weight block mismatch\n");
    }
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false},
           LoopSpecs{0L, BS, BSb},
           LoopSpecs{totalN}},
          loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            int i = 0;
            while (nk >= gemms[i].Nk) {
              nk -= gemms[i].Nk;
              i++;
            }
            funcs[i](nc, s1, nk);
          },
          [&]() {
            TimerStart();
            gemms[0].brgemm_tpp.config();
          },
          [&]() {
            gemms[0].brgemm_tpp.release();
            TimerEnd();
          });
    }
  }

  static TppBlockedLinearW<T, Tw, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return Base::template _get<TppBlockedLinearW<T, Tw, Tout>>(
        t_in, t_wt, t_bias);
  }
};

template <typename T, typename TW, typename TOUT = T>
class TppBlockedQInt8LinearW : public TppBlockedLinearWBase<T, TOUT> {
 public:
  using Tin = T;
  using Tout = TOUT;
  using Tw = TW;
  using Base = TppBlockedLinearWBase<Tin, Tout>;
  using Tbias = typename Base::Tbias;
  using Base::BSb;
  using Base::C;
  using Base::Hc;
  using Base::Hk;
  using Base::K;
  using Base::Nc;
  using Base::Ncb;
  using Base::Nk;
  using Base::loop_scheme;
  using Base::postOpCBs;
  using Base::rem;
  using Base::weight_reuse;

 protected:
  using BrTin = int8_t;
  using BrTw = int8_t;
  using BrTout = int32_t;
  SCOPEIT_DECL(BrgemmTPP<BrTin, BrTout, BrTw>) brgemm_tpp, brgemm_tpp_rem;
  SCOPEIT_DECL(DequantTPP<BrTout, Tout, float>) dequant_acc, dequant_acc_rem;
  long block_size = 0;
  long n_Hc_blocks = 1;
  long ScNc = 1;

 public:
  TppBlockedQInt8LinearW(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias)
      : TppBlockedLinearWBase<Tin, Tout>(t_in, t_wt, t_bias) {
    int b_vnni = 1;
    if (t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockAffine) {
      if (t_wt.dtype() == at::kQInt8) {
        // b_vnni = 2;
        block_size = q_per_block_block_size(t_wt);
        TPP_ASSERT(block_size % Hc == 0, "Block size mismatch\n");
        n_Hc_blocks = block_size / Hc;
        ScNc = Nc / n_Hc_blocks;
        printf(
            "Using weight type at::kQInt8 q_bs: %ld, Hc: %ld\n",
            block_size,
            Hc);
      } else {
        TPP_ASSERT(false, "Unsupported qtype\n");
      }
    }

    brgemm_tpp = SCOPEIT((BrgemmTPP<BrTin, BrTout, BrTw>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
    brgemm_tpp_rem = SCOPEIT((BrgemmTPP<BrTin, BrTout, BrTw>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
    dequant_acc = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(BSb, Hk, Hk, K, ScNc)), EW_RCP);
    dequant_acc_rem = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(rem, Hk, Hk, K, ScNc)), EW_RCP);

    loop_scheme =
        weight_reuse ? GEMM_LOOP_SCHEME_REUSE : GEMM_LOOP_SCHEME_STREAMING;
  }

  static inline const std::string extra(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    TORCH_CHECK(t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockAffine);
    long block_size = q_per_block_block_size(t_wt);
    char ehash[20];
    snprintf(ehash, 19, "bs%ld", block_size);
    return std::string(ehash);
  }

  std::function<void(int, int, int)> stepFunc(
      at::Tensor& t_in,
      at::Tensor& t_wt_V,
      at::Tensor& t_bias,
      at::Tensor& t_out,
      long BS) {
    auto in = GetVLAPtr<BrTin>(t_in, {Nc, Hc});
    auto bias = GetVLAPtr<Tbias>(t_bias, {Hk});
    auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});
    bool with_bias = (t_bias.numel() > 0);
    TPP_ASSERT(
        t_in.is_quantized() && t_in.qscheme() == at::kPerBlockAffine,
        "Input not quantized\n");
    TPP_ASSERT(
        t_wt_V.is_quantized() && t_wt_V.qscheme() == at::kPerBlockAffine,
        "Weight not quantized\n");
    auto constexpr pack_size = 1;
    auto wt_V = GetVLAPtr<BrTw>(t_wt_V, {Nc, (Hc * Hk) / pack_size});
    auto t_w_scl = q_per_block_scales(t_wt_V);
    auto t_i_scl = q_per_block_scales(t_in);
    auto w_scl = GetVLAPtr<float>(t_w_scl, {ScNc, Hk});
    auto i_scl = GetVLAPtr<float>(t_i_scl, {ScNc});
    auto func = [&, in, wt_V, w_scl, i_scl, bias, out, BS, with_bias ](
        int nc, int s1, int nk) __attribute__((always_inline)) {
      auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
      bool is_rem = (s1 + BSb > BS);
      if (!is_rem) {
        BrTout tmp_out[BSb * Hk];
        if (nc == 0) {
          if (with_bias) {
            this->copy_bias_tpp(bias[nk], out[s1][nk]);
          } else {
            this->zero_tpp(out[s1][nk]);
          }
        }
        for (int c = 0; c < count; c += n_Hc_blocks) {
          brgemm_tpp(
              in[s1][nc + c], wt_V[nk][nc + c], tmp_out, n_Hc_blocks, true);

          dequant_acc(
              tmp_out,
              out[s1][nk],
              &i_scl[s1][(nc + c) / n_Hc_blocks],
              w_scl[nk][(nc + c) / n_Hc_blocks]);
        }
        if (!(nc + Ncb < Nc)) { // last nc iter
          if (postOpCBs[0])
            postOpCBs[0](out, s1, nk);
        }
      } else {
        BrTout tmp_out[rem * Hk];
        if (nc == 0) {
          if (with_bias) {
            this->copy_bias_tpp_rem(bias[nk], out[s1][nk]);
          } else {
            this->zero_tpp_rem(out[s1][nk]);
          }
        }
        for (int c = 0; c < count; c++) {
          brgemm_tpp_rem(in[s1][nc + c], wt_V[nk][nc + c], tmp_out, 1, false);

          dequant_acc_rem(
              tmp_out,
              out[s1][nk],
              &i_scl[s1][(nc + c) / n_Hc_blocks],
              w_scl[nk][(nc + c) / n_Hc_blocks]);
        }
        if (!(nc + Ncb < Nc)) { // last nc iter
          if (postOpCBs[1])
            postOpCBs[1](out, s1, nk);
        }
      }
    };
    return func;
  }

  void operator()(
      at::Tensor t_in,
      at::Tensor t_wt_V,
      at::Tensor t_bias,
      at::Tensor t_out) {
    auto BS = t_in.numel() / this->C;
    auto t_qin = t_in;
    if (t_in.is_quantized()) {
      TORCH_CHECK(t_qin.is_contiguous());
      TORCH_CHECK(t_qin.dtype() == at::kQInt8);
    } else {
      t_in = t_in.contiguous();
      t_qin = quantize_int8sym(t_in, block_size, -1, false);
    }
    auto func = stepFunc(t_qin, t_wt_V, t_bias, t_out, BS);
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}, LoopSpecs{Nk}},
          loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            func(nc, s1, nk);
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

  static void fused_gemm(
      std::vector<TppBlockedQInt8LinearW<T, Tw, Tout>>& gemms,
      at::Tensor& t_in,
      std::vector<at::Tensor>& t_wt_V,
      std::vector<at::Tensor>& t_bias,
      std::vector<at::Tensor>& t_out) {
    int n_gemms = gemms.size();
    long totalN = 0;
    auto BS = t_in.numel() / gemms[0].C;
    long Nc = gemms[0].Nc;
    // long Hc = gemms[0].Hc;
    long Ncb = gemms[0].Ncb;
    long BSb = gemms[0].BSb;
    long block_size = gemms[0].block_size;
    auto loop_scheme = gemms[0].loop_scheme;

    auto t_qin = t_in;
    if (t_in.is_quantized()) {
      TORCH_CHECK(t_qin.is_contiguous());
      TORCH_CHECK(t_qin.dtype() == at::kQInt8);
    } else {
      t_in = t_in.contiguous();
      t_qin = quantize_int8sym(t_in, block_size, -1, false);
    }
    std::vector<std::function<void(int, int, int)>> funcs;
    for (int i = 0; i < n_gemms; i++) {
      auto& g = gemms[i];
      funcs.push_back(g.stepFunc(t_qin, t_wt_V[i], t_bias[i], t_out[i], BS));
      totalN += g.Nk;
      TPP_ASSERT(
          g.Nc == Nc && g.Ncb == Ncb && g.BSb == BSb,
          "Fused QKV weight block mismatch\n");
    }
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false},
           LoopSpecs{0L, BS, BSb},
           LoopSpecs{totalN}},
          loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            int i = 0;
            while (nk >= gemms[i].Nk) {
              nk -= gemms[i].Nk;
              i++;
            }
            funcs[i](nc, s1, nk);
          },
          [&]() {
            TimerStart();
            gemms[0].brgemm_tpp.config();
          },
          [&]() {
            gemms[0].brgemm_tpp.release();
            TimerEnd();
          });
    }
  }

  static TppBlockedQInt8LinearW<T, Tw, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return Base::template _get<TppBlockedQInt8LinearW<T, Tw, Tout>>(
        t_in, t_wt, t_bias);
  }
};

class NullPostOp {
 public:
  template <typename GemmT>
  constexpr void operator()(GemmT& gemm) {}
};

class GeluPostOp {
 public:
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto gelu_tpp = SCOPEIT(GeluFwdTPP<T>(M, N, LD, LD), ACT);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            gelu_tpp(out[x][y], out[x][y]);
          });
    }
  }
};

class SiluPostOp {
 public:
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto silu_tpp = SCOPEIT(SiLUFwdTPP<T>(M, N, LD, LD), ACT);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            silu_tpp(out[x][y], out[x][y]);
          });
    }
  }
};

class ReluPostOp {
 public:
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto relu_tpp = SCOPEIT(ReLUFwdTPP<T>(M, N, LD, LD, false), ACT);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            relu_tpp(out[x][y], out[x][y]);
          });
    }
  }
};

class MulPostOp {
 public:
  MulPostOp(at::Tensor t_in) : t_in(t_in) {}
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto in = gemm.getOutputVLAPtr(t_in);
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto mul_tpp = SCOPEIT((MulTPP<T, T>(M, N, LD, LD)), EW_MUL);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            mul_tpp(in[x][y], out[x][y], out[x][y]);
          });
    }
  }

 private:
  at::Tensor t_in;
};

class AddScalePostOp {
 public:
  AddScalePostOp(at::Tensor t_in, float scale) : t_in(t_in), scale(scale) {}
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto in = gemm.getOutputVLAPtr(t_in);
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(M, N, LD, LD)), EW_ADD);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            sadd_tpp(in[x][y], out[x][y], scale);
          });
    }
  }

 private:
  at::Tensor t_in;
  float scale;
};

class Add2ScalePostOp {
 public:
  Add2ScalePostOp(at::Tensor t_in1, at::Tensor t_in2, float scale)
      : t_in1(t_in1), t_in2(t_in2), scale(scale) {}
  template <typename GemmT>
  void operator()(GemmT& gemm) {
    using T = typename GemmT::Tout;
    auto in1 = gemm.getOutputVLAPtr(t_in1);
    auto in2 = gemm.getOutputVLAPtr(t_in2);
    auto num_shapes = gemm.numOutputShapes();
    for (int i = 0; i < num_shapes; i++) {
      long M, N, LD;
      std::tie(M, N, LD) = gemm.getOutputShape(i);
      if (M == 0 || N == 0)
        continue;
      auto add_tpp = SCOPEIT((AddTPP<T, T>(M, N, LD, LD)), EW_ADD);
      auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(M, N, LD, LD)), EW_ADD);
      gemm.setPostOpCB(
          i, [=](const VLAPtr<T, 2, long>& out, long x, long y) mutable {
            add_tpp(out[x][y], in1[x][y], out[x][y]);
            sadd_tpp(in2[x][y], out[x][y], scale);
          });
    }
  }

 private:
  at::Tensor t_in1, t_in2;
  float scale;
};

template <typename GemmT, typename CB>
inline at::Tensor dispatch_gemm(
    CB& cb,
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    c10::optional<at::Tensor> t_out_ = {}) {
  auto gemm = GemmT::get(t_in, t_wt, t_bias);
  cb(gemm);
  at::Tensor t_out;
  if (t_out_) {
    t_out = t_out_.value();
  } else {
    t_out = gemm.new_empty(t_in);
  }
  gemm(t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename Tin, typename Tout, typename CB>
inline at::Tensor call_gemm_with_post_op(
    CB cb,
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    c10::optional<at::Tensor> t_out_ = {}) {
  // Check and redispatch with specialized type
  if (t_wt.is_quantized()) {
    if (t_wt.qscheme() == at::kPerBlockMxFP) {
      if (t_wt.dtype() == at::kQUInt4x2) {
        if (t_wt.size(-1) == 2) {
          return dispatch_gemm<TppBlockedLinearW<Tin, uint8_t, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
        } else {
          return dispatch_gemm<TppBlockedLinearW<int8_t, uint8_t, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
        }
      } else {
        TPP_ASSERT(false, "Unsupported qdtype\n");
      }
    } else if (t_wt.qscheme() == at::kPerBlockAffine) {
      if (t_wt.dtype() == at::kQInt8) {
        return dispatch_gemm<TppBlockedQInt8LinearW<Tin, uint8_t, Tout>, CB>(
            cb, t_in, t_wt, t_bias);
      } else {
        TPP_ASSERT(false, "Unsupported qdtype\n");
      }
    } else {
      TPP_ASSERT(false, "Unsupported qscheme 2\n");
    }
  } else if (t_wt.is_sparse_csr()) {
  } else {
    auto dtype = t_wt.scalar_type();
    if (t_wt.dim() > 2) {
      switch (dtype) {
        case at::kFloat:
          return dispatch_gemm<TppBlockedLinearW<Tin, float, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kBFloat16:
          return dispatch_gemm<TppBlockedLinearW<Tin, bfloat16, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kHalf:
          return dispatch_gemm<TppBlockedLinearW<Tin, half, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kHFloat8:
          return dispatch_gemm<TppBlockedLinearW<Tin, hfloat8, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kBFloat8:
          return dispatch_gemm<TppBlockedLinearW<Tin, bfloat8, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        default:
          TPP_ASSERT(false, "Unsupported dtype\n");
      }
    } else {
      switch (dtype) {
        case at::kFloat:
          return dispatch_gemm<TppFlatLinear<Tin, float, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kBFloat16:
          return dispatch_gemm<TppFlatLinear<Tin, bfloat16, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kHalf:
          return dispatch_gemm<TppFlatLinear<Tin, half, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kHFloat8:
          return dispatch_gemm<TppFlatLinear<Tin, hfloat8, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        case at::kBFloat8:
          return dispatch_gemm<TppFlatLinear<Tin, bfloat8, Tout>, CB>(
              cb, t_in, t_wt, t_bias);
          break;
        default:
          TPP_ASSERT(false, "Unsupported dtype \n");
      }
    }
  }
  return at::Tensor();
}

template <typename Tin, typename Tout = Tin>
inline at::Tensor call_gemm(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    c10::optional<at::Tensor> t_out_ = {}) {
  return call_gemm_with_post_op<Tin, Tout>(
      NullPostOp(), t_in, t_wt, t_bias, t_out_);
}

template <typename Tin, typename Tout = Tin>
struct GemmCaller {
  GemmCaller() : GemmCaller(SCOPE_ARG(gemm)) {}
  GemmCaller(int scopeId, const char* scopeName)
      : scopeId(scopeId), scopeName(scopeName) {}

  at::Tensor operator()(
      at::Tensor in,
      at::Tensor wt,
      at::Tensor bias,
      c10::optional<at::Tensor> out = c10::nullopt) {
    GlobalScope gs_(scopeId);
    RECORD_FUNCTION(scopeName, std::vector<c10::IValue>({in, wt}));
    return call_gemm_with_post_op<Tin, Tout>(NullPostOp(), in, wt, bias, out);
  }

  template <typename CB>
  at::Tensor operator()(
      CB cb,
      at::Tensor in,
      at::Tensor wt,
      at::Tensor bias,
      c10::optional<at::Tensor> out = c10::nullopt) {
    GlobalScope gs_(scopeId);
    RECORD_FUNCTION(scopeName, std::vector<c10::IValue>({in, wt}));
    return call_gemm_with_post_op<Tin, Tout>(cb, in, wt, bias, out);
  }

 protected:
  const int scopeId;
  const char* scopeName;
};

template <typename T>
inline at::Tensor fc_plain(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    c10::optional<at::Tensor> t_out = {}) {
  auto pln_gemm = GemmCaller<T>(SCOPE_ARG(pln_gemm));
  return pln_gemm(t_in, t_wt, t_bias, t_out);
}
