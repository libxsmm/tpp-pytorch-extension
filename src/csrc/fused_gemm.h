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
// #include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include "ext_tpp.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include "qtypes.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;

static int BSB_BLOCK_SIZE = env2int("BSB_BLOCK_SIZE", 64);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static const char* GEMM_LOOP_SCHEME_REUSE =
    getenv("GEMM_LOOP_SCHEME_REUSE") ? getenv("GEMM_LOOP_SCHEME_REUSE") : "aCB";
static const char* GEMM_LOOP_SCHEME_STREAMING =
    getenv("GEMM_LOOP_SCHEME_STREAMING") ? getenv("GEMM_LOOP_SCHEME_STREAMING")
                                         : "aCb";
static const int QINT8_EMU = env2int("QINT8_EMU", 0);
static const int QINT8_BLOCK_SIZE = env2int("QINT8_BLOCK_SIZE", 256);

REGISTER_LOCAL_SCOPE(pln_gemm, "pln_gemm");
REGISTER_LOCAL_SCOPE(gemm, "gemm");
REGISTER_LOCAL_SCOPE(quant, "quant");

template <typename T, typename TOUT>
class TppBlockedLinearWBase {
 public:
  using Tin = T;
  using Tout = TOUT;

 protected:
  static constexpr int nOutputShapes = 2;
  long Nc, Hc, Nk, Hk, Ncb, BSb, rem;
  bool weight_reuse;
  long C, K;
  SCOPEIT_DECL(CpyBiasTPP<T, Tout>) copy_bias_tpp, copy_bias_tpp_rem;
  SCOPEIT_DECL(SetZeroTPP<Tout>) zero_tpp, zero_tpp_rem;

  std::string loop_scheme;
  std::function<void(const VLAPtr<T, 2, long>&, long, long)>
      postOpCBs[nOutputShapes];

 public:
  TppBlockedLinearWBase(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
    std::tie(Nc, Hc, Nk, Hk, Ncb, BSb, rem, weight_reuse) =
        getBlockingParams(t_in, t_wt, t_bias);
    C = Nc * Hc;
    K = Nk * Hk;

    copy_bias_tpp = SCOPEIT((CpyBiasTPP<T, Tout>(BSb, Hk, K)), BIAS);
    copy_bias_tpp_rem = SCOPEIT((CpyBiasTPP<T, Tout>(rem, Hk, K)), BIAS);
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
    char hash[200] = "";
    snprintf(
        hash,
        199,
        "gemm_Nc%ld_Hc%ld_Nk%ld_Hk%ld_Bsb%ld_rem%ld_Ncb%ld_wr%d",
        Nc,
        Hc,
        Nk,
        Hk,
        BSb,
        rem,
        Ncb,
        weight_reuse ? 1 : 0);
    auto search = gemm_cache.find(hash);
    GemmT* gemm = NULL;
    if (search != gemm_cache.end())
      gemm = search->second;
    if (gemm == NULL) {
      gemm = new GemmT(t_in, t_wt, t_bias);
      gemm_cache[hash] = gemm;
      // printf("Hash: %s\n", hash);
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
    if (t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockMxFP) {
      if (t_wt.dtype() == at::kQUInt4x2) {
        b_vnni = 2;
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
    auto bias = GetVLAPtr<T>(t_bias, {Hk});
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
        auto func = [&, in, wt_V, scl, bias, out, BS, with_bias ](
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
            brgemm_tpp(
                in[s1][nc],
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
            brgemm_tpp_rem(
                in[s1][nc],
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
    auto func = stepFunc(t_in, t_wt_V, t_bias, t_out, BS);
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
    for (int i = 0; i < n_gemms; i++) {
      auto& g = gemms[i];
      funcs.push_back(g.stepFunc(t_in, t_wt_V[i], t_bias[i], t_out[i], BS));
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

 public:
  TppBlockedQInt8LinearW(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias)
      : TppBlockedLinearWBase<Tin, Tout>(t_in, t_wt, t_bias) {
    int b_vnni = 1;
    if (t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockAffine) {
      if (t_wt.dtype() == at::kQInt8) {
        // b_vnni = 2;
        printf("Using weight type at::kQInt8\n");
      } else {
        TPP_ASSERT(false, "Unsupported qtype\n");
      }
    }

    brgemm_tpp = SCOPEITGEMM((BrgemmTPP<BrTin, BrTout, BrTw>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
    brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<BrTin, BrTout, BrTw>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
    dequant_acc =
        SCOPEIT((DequantTPP<BrTout, Tout, float>(BSb, Hk, Hk, K, Nc)), EW_RCP);
    dequant_acc_rem =
        SCOPEIT((DequantTPP<BrTout, Tout, float>(rem, Hk, Hk, K, Nc)), EW_RCP);

    loop_scheme =
        weight_reuse ? GEMM_LOOP_SCHEME_REUSE : GEMM_LOOP_SCHEME_STREAMING;
  }
  std::function<void(int, int, int)> stepFunc(
      at::Tensor& t_in,
      at::Tensor& t_wt_V,
      at::Tensor& t_bias,
      at::Tensor& t_out,
      long BS) {
    auto in = GetVLAPtr<BrTin>(t_in, {Nc, Hc});
    auto bias = GetVLAPtr<T>(t_bias, {Hk});
    auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});
    bool with_bias = (t_bias.numel() > 0);
    TPP_ASSERT(
        t_in.is_quantized() && t_in.qscheme() == at::kPerBlockAffine,
        "Input not quantized\n");
    TPP_ASSERT(
        t_wt_V.is_quantized() && t_wt_V.qscheme() == at::kPerBlockAffine,
        "Weight not quantized\n");
    auto quantizer = at::get_qtensorimpl(t_wt_V)->quantizer();
    auto w_quantizer =
        static_cast<at::PerBlockAffineQuantizer*>(quantizer.get());
    quantizer = at::get_qtensorimpl(t_in)->quantizer();
    auto i_quantizer =
        static_cast<at::PerBlockAffineQuantizer*>(quantizer.get());
    auto pack_size = w_quantizer->pack_size();
    auto block_size = w_quantizer->block_size();
    TPP_ASSERT(block_size == Hc, "Block size mismatch\n");
    auto wt_V = GetVLAPtr<BrTw>(t_wt_V, {Nc, (Hc * Hk) / pack_size});
    auto t_w_scl = w_quantizer->scales();
    auto t_i_scl = i_quantizer->scales();
    auto w_scl = GetVLAPtr<float>(t_w_scl, {Nc, Hk /* *(Hc / block_size)*/});
    auto i_scl = GetVLAPtr<float>(t_i_scl, {Nc /*, Hc / block_size*/});
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
        for (int c = 0; c < count; c++) {
          brgemm_tpp(in[s1][nc + c], wt_V[nk][nc + c], tmp_out, 1, true);

          dequant_acc(
              tmp_out, out[s1][nk], &i_scl[s1][nc + c], w_scl[nk][nc + c]);
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
              tmp_out, out[s1][nk], &i_scl[s1][nc + c], w_scl[nk][nc + c]);
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
    t_in = t_in.contiguous();
    auto BS = t_in.numel() / this->C;
    auto t_qin = quantize_int8sym(t_in, Hc, 2, false);
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
    long Hc = gemms[0].Hc;
    long Ncb = gemms[0].Ncb;
    long BSb = gemms[0].BSb;
    auto loop_scheme = gemms[0].loop_scheme;

    auto t_qin = quantize_int8sym(t_in, Hc, 2, false);
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
        return dispatch_gemm<TppBlockedLinearW<Tin, uint8_t, Tout>, CB>(
            cb, t_in, t_wt, t_bias);
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

inline at::Tensor fix_vnni(at::Tensor t) {
  auto dtype = t.scalar_type();
  auto dims = t.dim();
  TPP_ASSERT(dims >= 4, "Invalid shape, dims = %ld\n", dims);
  auto sizes = t.sizes();
  auto K1 = sizes[0];
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = dims == 5 ? sizes[4] : 1L;
  if (dtype == at::kBFloat16) {
    if (C3 == 2) {
      return t;
    } else if (C3 == 4) {
      return t.view({K1, C1, C2, K2, 2, 2})
          .permute({0, 1, 2, 4, 3, 5})
          .contiguous()
          .view({K1, C1, C2 * 2, K2, 2});
    } else if (C3 == 1) {
      TPP_ASSERT(
          C2 % 2 == 0, "Shape incompatible for VNNI2 layout, C2 = %ld\n", C2);
      return t.view({K1, C1, C2 / 2, 2, K2})
          .permute({0, 1, 2, 4, 3})
          .contiguous();
    } else {
      TPP_ASSERT(false, "Shape incompatible for VNNI2 layout\n");
    }
  } else if (dtype == at::kFloat) {
    if (C3 == 1 and dims == 4) {
      return t;
    } else if (C3 == 1) {
      return t.view({K1, C1, C2, K2});
    } else {
      return t.permute({0, 1, 2, 4, 3})
          .contiguous()
          .view({K1, C1, C2 * C3, K2});
    }
  } else {
    TPP_ASSERT(false, "Unknown dtype for FIX_VNNI layout\n");
  }
  return t;
}

inline at::Tensor remap_and_quantize_mxfp4(at::Tensor t) {
  RECORD_SCOPE(quant, {t});
  auto dim = t.dim();
  if (dim == 5) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C3 = sizes[4];
    if (K2 < 32 && 32 % K2 == 0) {
      int RBS = 32 / K2;
      TPP_ASSERT(K1 % RBS == 0, "Shape not compatible for MXFP4\n");
      t = t.view({K1 / RBS, RBS, C1, C2, K2, C3})
              .permute({0, 2, 3, 1, 4, 5})
              .contiguous()
              .view({K1 / RBS, C1, C2, RBS * K2, C3});
    }
  } else if (dim == 4) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    TPP_ASSERT(C2 % 2 == 0, "Shape not compatible for VNNI2\n");
    if (K2 < 32 && 32 % K2 == 0) {
      int RBS = 32 / K2;
      TPP_ASSERT(K1 % RBS == 0, "Shape not compatible for MXFP4\n");
      t = t.view({K1 / RBS, RBS, C1, C2 / 2, 2L, K2})
              .permute({0, 2, 3, 1, 5, 4})
              .contiguous()
              .view({K1 / RBS, C1, C2 / 2, RBS * K2, 2});
    } else {
      t = t.view({K1, C1, C2 / 2, 2L, K2})
              .permute({0, 1, 2, 4, 3})
              .contiguous()
              .view({K1, C1, C2 / 2, K2, 2});
    }
  }
  auto ret = quantize_mxfp4(t, 32, 2, true);
  return ret;
}

inline at::Tensor remap_and_quantize_qint8(at::Tensor t) {
  RECORD_SCOPE(quant, {t});
  auto dim = t.dim();
  long block_size = 0;
  if (dim == 5) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C3 = sizes[4];
    auto C = C1 * C2 * C3;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE / C3;
      C1 = C / QINT8_BLOCK_SIZE;
    }
    if (C3 != 4) {
      int RBS = 4 / C3;
      TPP_ASSERT(C2 % RBS == 0, "Shape not compatible for VNNI4\n");
      t = t.view({K1, C1, C2 / RBS, RBS, K2, C3})
              .permute({0, 1, 2, 4, 3, 5})
              .contiguous()
              .view({K1, C1, C2 / RBS, K2, RBS * C3});
    }
    block_size = C2 * C3;
  } else if (dim == 4) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C = C1 * C2;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE;
      C1 = C / QINT8_BLOCK_SIZE;
    }
    TPP_ASSERT(C2 % 4 == 0, "Shape not compatible for VNNI4\n");
    t = t.view({K1, C1, C2 / 4, 4, K2})
            .permute({0, 1, 2, 4, 3})
            .contiguous()
            .view({K1, C1, C2 / 4, K2, 4});
    block_size = C2;
  }
  auto ret = quantize_int8sym(t, block_size, 2, true);
  std::cout << "remap_and_quantize_qint8: " << ret.sizes()
            << " dt: " << ret.dtype() << std::endl;
  if (QINT8_EMU == 1) {
    ret = fix_vnni(ret.dequantize().to(t.dtype()));
  }
  return ret;
}
