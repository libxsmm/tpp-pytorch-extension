/******************************************************************************
 * Copyright (c) 2025 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include <torch/csrc/distributed/c10d/comm.hpp>
#include "fused_gemm.h"
#include "qtypes.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "shm_coll.h"
#include "tensor_helper.h"

#include "attn.h"

static int my_rank = guess_mpi_rank();
static int SK_BLOCK_SIZE = env2int("SK_BLOCK_SIZE", 128);
static const bool USE_FLASH = env2int("USE_FLASH", 1);

int key_layout_id = KVLayout::kBNSH;
// int key_layout_id = KVLayout::kSBNH;
int value_layout_id = KVLayout::kBNSH;

REGISTER_LOCAL_SCOPE(ac_gemm1, "ac_gemm1");

template <typename F>
std::vector<int> get_SBlocks(long S, long S2, F func) {
  std::vector<int> blocks;
  long s = 0;
  blocks.push_back(s);
  while (s < S) {
    auto k = func(s);
    auto k2 = k % S2;
    long x = 1;
    for (; x < S2; x++) {
      if (s + x >= S)
        break;
      auto kx = func(s + x);
      auto k2x = kx % S2;
      if (k2x != k2 + x)
        break;
    }
    s += x;
    blocks.push_back(s);
  }
#if 0
  int nbs = blocks.size();
  printf("nbs1 = Blocks[%d] = \n", nbs-1);
  for (auto ns = 0; ns < nbs-1; ns++) {
    printf(" %3d (%2d)", blocks[ns], blocks[ns + 1] - blocks[ns]);
    if (ns % 10 == 9)
      printf("\n");
  }
  printf("\nEND\n");
#endif

  return blocks;
}

template <typename Ts, typename Tp>
struct KVLayout_SBNH : public KVLayout {
  // [S1, S2, B, N, H]
  // SCOPEIT_DECL(ConvertTPP<Ts, Tp>) cvt_tpp;
  ConvertTPP<Ts, Tp> cvt_tpp;

  KVLayout_SBNH(long B, long N, long H, long S2) : KVLayout(B, N, H, S2, 2, 0) {
    Hp = 1;
    Np = H;
    Bp = N * H;
    S2p = B * Bp;
    S1p = S2 * S2p;
    in_vnni = 0;
    VBS = get_vnni_block_size<Tp>();
    // cvt_tpp = SCOPEIT((ConvertTPP<Ts, Tp>(H)), EW_COPY);
    cvt_tpp = ConvertTPP<Ts, Tp>(H);
  }
  int get_layout() const override {
    return kSBNH;
  }
  at::Tensor new_empty(const at::TensorOptions& options) const override {
    return at::empty({0, S2, B, N, H}, options);
  }
  const KVMeta get_meta() const {
    return KVMeta(S1p, S2p, Bp, Np, Hp, in_vnni);
  }
  void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) override {
    auto S = t_states.size(-2);
    long Ss, Bs, Ns;
    auto s_strides = t_states.strides();
    Bs = s_strides[0];
    Ns = s_strides[1];
    Ss = s_strides[2];
    // Hs = s_strides[3]; must be 1

    auto states = GetVLAPtrFromStrides<Ts>(t_states, {Bs, Ns, Ss}); // H contig
    auto past =
        GetVLAPtrFromStrides<Tp>(t_past, {S1p, S2p, Bp, Np}); // H contig
#pragma omp parallel for collapse(3) if (S > 1)
    for (auto i = 0; i < B; i++) {
      for (auto j = 0; j < N; j++) {
        for (auto s = 0; s < S; s++) {
          auto k = func(s);
          auto k1 = k / S2;
          auto k2 = k % S2;
          cvt_tpp(states[i][j][s], past[k1][k2][i][j]);
        }
      }
    }
  }
};

template <typename Ts, typename Tp>
struct KVLayout_BNHS : public KVLayout {
  // [S1, B, N, H, S2]
  // SCOPEIT_DECL(ConvertTPP<Ts, Tp>) cvt_tpp;
  XformExtTPP<Ts> trans_tpp_H, trans_tpp_NH;

  KVLayout_BNHS(long B, long N, long H, long S2) : KVLayout(B, N, H, S2, 1, 0) {
    S2p = 1;
    Hp = S2;
    Np = H * Hp;
    Bp = N * Np;
    S1p = B * Bp;
    in_vnni = 0;
    VBS = get_vnni_block_size<Tp>();
    auto xform = XformTPP::XFORM_XPOSE_TPP;
    trans_tpp_H = XformExtTPP<Ts>(S2, H, H, S2, H, S2, xform, true);
    trans_tpp_NH = XformExtTPP<Ts>(S2, H, H, S2, N * H, S2, xform, true);
  }
  int get_layout() const override {
    return kBNHS;
  }
  at::Tensor new_empty(const at::TensorOptions& options) const override {
    return at::empty({0, B, N, H, S2}, options);
  }
  const KVMeta get_meta() const {
    return KVMeta(S1p, S2p, Bp, Np, Hp, in_vnni);
  }
  void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) override {
    auto S = t_states.size(-2);
    long Ss, Bs, Ns;
    auto s_strides = t_states.strides();
    Bs = s_strides[0];
    Ns = s_strides[1];
    Ss = s_strides[2];
    // Hs = s_strides[3]; must be 1

    auto states = GetVLAPtrFromStrides<Ts>(t_states, {Bs, Ns, Ss}); // H contig
    auto past =
        GetVLAPtrFromStrides<Tp>(t_past, {S1p, Bp, Np, Hp}); // S2 contig

#if 0
    for (auto s = 0; s < S; s++) {
      auto k = func(s);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto h = 0; h < H; h++) {
            past[k1][i][j][h][k2] = states[i][j][s][h];
          }
        }
      }
    }
#else
    if (S == 1) {
      auto k = func(0);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto h = 0; h < H; h++) {
            past[k1][i][j][h][k2] = states[i][j][0][h];
          }
        }
      }
    } else {
      auto bs = get_SBlocks(S, S2, func);
      int nbs = bs.size();
      XformExtTPP<Ts> trans_tpp;
      if (Ss == H) {
        trans_tpp = trans_tpp_H;
      } else if (Ss == N * H) {
        trans_tpp = trans_tpp_NH;
      } else {
        auto xform = XformTPP::XFORM_XPOSE_TPP;
        trans_tpp = XformExtTPP<Ts>(S2, H, H, S2, Ss, S2, xform, true);
      }

      auto nTasks = B * N * (nbs - 1);
#pragma omp parallel for collapse(3) if (nTasks > 4)
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto ns = 0; ns < nbs - 1; ns++) {
            auto ss = bs[ns];
            auto ls = bs[ns + 1] - ss;
            auto k = func(ss);
            auto k1 = k / S2;
            if (ls == S2) {
              trans_tpp(states[i][j][ss], past[k1][i][j][0]);
            } else {
              auto k2 = k % S2;
              for (auto h = 0; h < H; h++) {
                for (auto x = 0; x < ls; x++) {
                  past[k1][i][j][h][k2 + x] = states[i][j][ss + x][h];
                }
              }
            }
          }
        }
      }
    }
#endif
  }
};

template <typename Ts, typename Tp>
struct KVLayout_vBNHS : public KVLayout {
  // [S1, B, N, H, S2]
  // SCOPEIT_DECL(ConvertTPP<Ts, Tp>) cvt_tpp;
  XformExtTPP<Ts> trans_tpp_H, trans_tpp_NH;

  KVLayout_vBNHS(long B, long N, long H, long S2)
      : KVLayout(B, N, H, S2, 1, 0) {
    S2p = 1;
    Hp = S2;
    Np = H * Hp;
    Bp = N * Np;
    S1p = B * Bp;
    in_vnni = 1;
    VBS = get_vnni_block_size<Tp>();
    auto xform = XformTPP::XFORM_XPOSE_N2V_TPP;
    trans_tpp_H = XformExtTPP<Ts>(S2, H, H, S2, H, S2, xform, true);
    trans_tpp_NH = XformExtTPP<Ts>(S2, H, H, S2, N * H, S2, xform, true);
  }
  int get_layout() const override {
    return kvBNHS;
  }
  at::Tensor new_empty(const at::TensorOptions& options) const override {
    return at::empty({0, B, N, H / VBS, S2, VBS}, options);
  }
  const KVMeta get_meta() const {
    return KVMeta(S1p, S2p, Bp, Np, Hp, in_vnni);
  }
  void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) override {
    auto S = t_states.size(-2);
    long Ss, Bs, Ns;
    auto s_strides = t_states.strides();
    Bs = s_strides[0];
    Ns = s_strides[1];
    Ss = s_strides[2];
    // Hs = s_strides[3]; must be 1

    auto states = GetVLAPtrFromStrides<Ts>(t_states, {Bs, Ns, Ss}); // H contig
    auto past = GetVLAPtrFromStrides<Tp>(t_past, {S1p, Bp, Np}); // S2 contig

#if 0
    for (auto s = 0; s < S; s++) {
      auto k = func(s);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto h = 0; h < H; h++) {
            auto h2 = h % VBS;
            auto h1 = h - h2;
            auto dst = past[k1][i][j];
            dst[h1 * S2 + k2 * VBS + h2] = states[i][j][s][h];
          }
        }
      }
    }
#else
    if (S == 1) {
      auto k = func(0);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto h = 0; h < H; h++) {
            auto h2 = h % VBS;
            auto h1 = h - h2;
            auto dst = past[k1][i][j];
            dst[h1 * S2 + k2 * VBS + h2] = states[i][j][0][h];
          }
        }
      }
    } else {
      auto bs = get_SBlocks(S, S2, func);
      int nbs = bs.size();
      XformExtTPP<Ts> trans_tpp;
      if (Ss == H) {
        trans_tpp = trans_tpp_H;
      } else if (Ss == N * H) {
        trans_tpp = trans_tpp_NH;
      } else {
        auto xform = XformTPP::XFORM_XPOSE_N2V_TPP;
        trans_tpp = XformExtTPP<Ts>(S2, H, H, S2, Ss, S2, xform, true);
      }

      auto nTasks = B * N * (nbs - 1);
#pragma omp parallel for collapse(3) if (nTasks > 4)
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto ns = 0; ns < nbs - 1; ns++) {
            auto ss = bs[ns];
            auto ls = bs[ns + 1] - ss;
            auto k = func(ss);
            auto k1 = k / S2;
            if (ls == S2) {
              trans_tpp(states[i][j][ss], past[k1][i][j]);
            } else {
              auto k2 = k % S2;
              auto dst = past[k1][i][j];
              dst += k2 * VBS;
              for (auto h = 0; h < H; h++) {
                auto h2 = h % VBS;
                auto h1 = h - h2;
                for (auto x = 0; x < ls; x++) {
                  dst[h1 * S2 + x * VBS + h2] = states[i][j][ss + x][h];
                  // past[k1][i][j][h][k2 + x] = states[i][j][ss + x][h];
                }
              }
            }
          }
        }
      }
    }
#endif
  }
};

template <typename Ts, typename Tp>
struct KVLayout_BNSH : public KVLayout {
  // [S1, B, N, S2, H]
  ConvertTPP<Ts, Tp> cvt_tpp_1, cvt_tpp_H, cvt_tpp_NH;
  // XformExtTPP<Ts> trans_tpp_H, trans_tpp_NH;

  KVLayout_BNSH(long B, long N, long H, long S2) : KVLayout(B, N, H, S2, 1, 0) {
    Hp = 1;
    S2p = H;
    Np = S2 * S2p;
    Bp = N * Np;
    S1p = B * Bp;
    in_vnni = 0;
    VBS = get_vnni_block_size<Tp>();
    cvt_tpp_1 = ConvertTPP<Ts, Tp>(H);
    cvt_tpp_H = ConvertTPP<Ts, Tp>(S2, H, H, H);
    cvt_tpp_NH = ConvertTPP<Ts, Tp>(S2, H, N * H, H);
    // auto xform = XformTPP::XFORM_XPOSE_TPP;
    // trans_tpp_H = XformExtTPP<Ts>(S2, H, H, S2, H, S2, xform, true);
    // trans_tpp_NH = XformExtTPP<Ts>(S2, H, H, S2, N*H, S2, xform, true);
  }
  int get_layout() const override {
    return kBNSH;
  }
  at::Tensor new_empty(const at::TensorOptions& options) const override {
    return at::empty({0, B, N, S2, H}, options);
  }
  const KVMeta get_meta() const {
    return KVMeta(S1p, S2p, Bp, Np, Hp, in_vnni);
  }
  void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) override {
    auto S = t_states.size(-2);
    long Ss, Bs, Ns;
    auto s_strides = t_states.strides();
    Bs = s_strides[0];
    Ns = s_strides[1];
    Ss = s_strides[2];
    // Hs = s_strides[3]; must be 1

    auto states = GetVLAPtrFromStrides<Ts>(t_states, {Bs, Ns, Ss}); // H contig
    auto past =
        GetVLAPtrFromStrides<Tp>(t_past, {S1p, Bp, Np, S2p}); // H contig

#if 0
    for (auto s = 0; s < S; s++) {
      auto k = func(s);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto h = 0; h < H; h++) {
            past[k1][i][j][k2][h] = states[i][j][s][h];
          }
        }
      }
    }
#else
    if (S == 1) {
      auto k = func(0);
      auto k1 = k / S2;
      auto k2 = k % S2;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          cvt_tpp_1(states[i][j][0], past[k1][i][j][k2]);
        }
      }
    } else {
      auto bs = get_SBlocks(S, S2, func);
      int nbs = bs.size();
      ConvertTPP<Ts, Tp> cvt_tpp;
      // XformExtTPP<Ts> trans_tpp;
      if (Ss == H) {
        cvt_tpp = cvt_tpp_H;
      } else if (Ss == N * H) {
        cvt_tpp = cvt_tpp_NH;
      } else {
        cvt_tpp = ConvertTPP<Ts, Tp>(S2, H, Ss, H);
      }

      auto nTasks = B * N * (nbs - 1);
#pragma omp parallel for collapse(3) if (nTasks > 4)
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto ns = 0; ns < nbs - 1; ns++) {
            auto ss = bs[ns];
            auto ls = bs[ns + 1] - ss;
            auto k = func(ss);
            auto k1 = k / S2;
            if (ls == S2) {
              cvt_tpp(states[i][j][ss], past[k1][i][j][0]);
            } else {
              auto k2 = k % S2;
              for (auto x = 0; x < ls; x++) {
                cvt_tpp_1(states[i][j][ss + x], past[k1][i][j][k2 + x]);
              }
            }
          }
        }
      }
    }
#endif
  }
};

template <typename Ts, typename Tp>
struct KVLayout_vBNSH : public KVLayout {
  // [S1, B, N, S2/VBS, H, VBS]
  // SCOPEIT_DECL(ConvertTPP<Ts, Tp>) cvt_tpp;
  ConvertTPP<Ts, Tp> cvt_tpp;
  XformExtTPP<Ts> vnni_tpp_H, vnni_tpp_NH;

  KVLayout_vBNSH(long B, long N, long H, long S2)
      : KVLayout(B, N, H, S2, 1, 0) {
    Hp = 1;
    S2p = H;
    Np = S2 * S2p;
    Bp = N * Np;
    S1p = B * Bp;
    in_vnni = 1;
    VBS = get_vnni_block_size<Tp>();
    cvt_tpp = ConvertTPP<Ts, Tp>(H, 1, 1, VBS);
    auto xform = XformTPP::XFORM_N2V_TPP;
    vnni_tpp_H = XformExtTPP<Ts>(S2, H, S2, H, H, H, xform, true);
    vnni_tpp_NH = XformExtTPP<Ts>(S2, H, S2, H, N * H, H, xform, true);
  }
  int get_layout() const override {
    return kvBNSH;
  }
  at::Tensor new_empty(const at::TensorOptions& options) const override {
    return at::empty({0, B, N, S2 / VBS, H, VBS}, options);
  }
  const KVMeta get_meta() const {
    return KVMeta(S1p, S2p, Bp, Np, Hp, in_vnni);
  }
  void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) override {
    auto S = t_states.size(-2);
    long Ss, Bs, Ns;
    auto s_strides = t_states.strides();
    Bs = s_strides[0];
    Ns = s_strides[1];
    Ss = s_strides[2];
    // Hs = s_strides[3]; must be 1

    auto states = GetVLAPtrFromStrides<Ts>(t_states, {Bs, Ns, Ss}); // H contig
    auto past =
        GetVLAPtrFromStrides<Tp>(t_past, {S1p, Bp, Np}); // [S2, H] in vnni

#if 0
    for (auto s = 0; s < S; s++) {
      auto k = func(s);
      auto k1 = k / S2;
      auto k2 = k % S2;
      auto k22 = k2 % VBS;
      auto k21 = k2 - k22;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          auto dst = past[k1][i][j] + k21 * H + k22;
          for (auto h = 0; h < H; h++) {
            dst[h * VBS] = states[i][j][s][h];
          }
        }
      }
    }
#else
    if (S == 1) {
      auto k = func(0);
      auto k1 = k / S2;
      auto k2 = k % S2;
      auto k22 = k2 % VBS;
      auto k21 = k2 - k22;
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          auto dst = past[k1][i][j] + k21 * H + k22;
          for (auto h = 0; h < H; h++) {
            dst[h * VBS] = states[i][j][0][h];
          }
        }
      }
    } else {
      auto bs = get_SBlocks(S, S2, func);
      int nbs = bs.size();

      XformExtTPP<Ts> vnni_tpp;
      if (Ss == H) {
        vnni_tpp = vnni_tpp_H;
      } else if (Ss == N * H) {
        vnni_tpp = vnni_tpp_NH;
      } else {
        auto xform = XformTPP::XFORM_N2V_TPP;
        vnni_tpp = XformExtTPP<Ts>(S2, H, S2, H, Ss, H, xform, true);
      }

      auto nTasks = B * N * (nbs - 1);
#pragma omp parallel for collapse(3) if (nTasks > 4)
      for (auto i = 0; i < B; i++) {
        for (auto j = 0; j < N; j++) {
          for (auto ns = 0; ns < nbs - 1; ns++) {
            auto ss = bs[ns];
            auto ls = bs[ns + 1] - ss;
            auto k = func(ss);
            auto k1 = k / S2;
            if (ls == S2) {
              vnni_tpp(states[i][j][ss], past[k1][i][j]);
            } else {
              auto k2 = k % S2;
              auto dst = past[k1][i][j];
              for (auto h = 0; h < H; h++) {
                for (auto x = 0; x < ls; x++) {
                  auto k22 = (k2 + x) % VBS;
                  auto k21 = k2 + x - k22;
                  dst[k21 * H + h * VBS + k22] = states[i][j][ss + x][h];
                  // past[k1][i][j][k2 + x][h] = states[i][j][ss + x][h];
                }
              }
            }
          }
        }
      }
    }
#endif
  }
};

template <typename T>
KVLayout* get_kv_layout_typed(int layout, long B, long N, long H, long S2) {
  switch (layout) {
    case KVLayout::kSBNH:
      return new KVLayout_SBNH<T, T>(B, N, H, S2);
    case KVLayout::kBNHS:
      return new KVLayout_BNHS<T, T>(B, N, H, S2);
    case KVLayout::kvBNHS:
      return new KVLayout_vBNHS<T, T>(B, N, H, S2);
    case KVLayout::kBNSH:
      return new KVLayout_BNSH<T, T>(B, N, H, S2);
    case KVLayout::kvBNSH:
      return new KVLayout_vBNSH<T, T>(B, N, H, S2);
    default:
      TPP_ASSERT(false, "Unsupported layout %d\n", layout);
  }
  return nullptr;
}

KVLayout* get_kv_layout(
    int layout,
    long B,
    long N,
    long H,
    long S2,
    c10::ScalarType cache_type) {
  switch (cache_type) {
    case c10::kBFloat16:
      return get_kv_layout_typed<at::BFloat16>(layout, B, N, H, S2);
    case c10::kFloat:
      return get_kv_layout_typed<float>(layout, B, N, H, S2);
    default:
      TPP_ASSERT(false, "Unsupported cache type %d\n", (int)cache_type);
  }
  return nullptr;
}

#if 0
template <typename T, typename Tc>
struct AttnKernelsNew {
  SCOPEIT_DECL(BrgemmTPP<T, float, Tc>) a_gemm_tpp;
  SCOPEIT_DECL(ScaleTPP<float, float>) scale_tpp;
  SCOPEIT_DECL(AddBiasTPP<T>) add_mask_tpp;
  SCOPEIT_DECL(VarSoftMaxFwdTPP<float, T>) softmax_fwd_tpp;
  SCOPEIT_DECL(BrgemmTPP<T, T, Tc>) c_gemm_tpp;
  SCOPEIT_DECL(ConvertTPP<T, T>) cvt_tpp;
  SCOPEIT_DECL(SoftMaxFixUpTPP<T>) softmax_fixup;
  SCOPEIT_DECL(SoftMaxFlashScaleTPP<T>) softmax_scale;

  AttnKernelsNew(long Sqb, long Skb, long H, const KVCacheMeta& cm) {
    if (Sqb == 0)
      return;
    // [Sqb, H] * [H, Skb] = [Sqb, Skb]
    auto k_ld = cm.kl_needs_trans ? cm.key.s2_str : cm.key.h_str;
    a_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float, Tc>(
        Sqb,
        Skb,
        H,
        H,
        cm.key.s1_str,
        H,
        k_ld,
        Skb,
        0.0,
        0,
        1,
        cm.key.in_vnni,
        cm.kl_needs_trans)));
    // [Sqb, Skb]
    scale_tpp = SCOPEIT((ScaleTPP<float, float>(Sqb * Skb)), EW_SCL);
    add_mask_tpp = SCOPEIT(AddBiasTPP<T>(Sqb, Skb), EW_ADD);
    softmax_fwd_tpp =
        SCOPEIT((VarSoftMaxFwdTPP<float, T>(Sqb, Skb, USE_FLASH)), SOFTMAX);
    softmax_fixup = SCOPEIT((SoftMaxFixUpTPP<T>(Sqb, H, USE_FLASH)), EW_RCP);
    softmax_scale =
        SCOPEIT((SoftMaxFlashScaleTPP<T>(Sqb, H, USE_FLASH)), EW_RCP);
    // [nSk, Sqb, Skb] * [nSk, Skb, H] = tmp[Sqb, H]
    c_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T, Tc>(
        Sqb,
        H,
        Skb,
        Sqb * Skb,
        cm.value.s1_str,
        Skb,
        cm.value.s2_str,
        H,
        0.0,
        0,
        1,
        cm.value.in_vnni)));
    // [Sqb, H] --> [Sqb, H]
    cvt_tpp = SCOPEIT((ConvertTPP<T, T>(Sqb, H, H, H)), EW_COPY);
  }
};
#else
template <typename T, typename Tc>
struct AttnKernelsNew {
  BrgemmTPP<T, float, Tc> a_gemm_tpp;
  ScaleTPP<float, float> scale_tpp;
  AddBiasTPP<T> add_mask_tpp;
  VarSoftMaxFwdTPP<float, T> softmax_fwd_tpp;
  BrgemmTPP<T, T, Tc> c_gemm_tpp;
  ConvertTPP<T, T> cvt_tpp;
  SoftMaxFixUpTPP<T> softmax_fixup;
  SoftMaxFlashScaleTPP<T> softmax_scale;

  AttnKernelsNew(
      long Sqb,
      long Skb,
      long H,
      const KVCacheMeta& cm,
      bool isBeamSearch) {
    if (Sqb == 0)
      return;
    if (isBeamSearch) {
      TPP_ASSERT(
          cm.key.h_str == 1 && cm.value.h_str == 1,
          "Beam search only supports h_str=1");
      TPP_ASSERT(
          cm.key.in_vnni == 0 && cm.value.in_vnni == 0,
          "Beam search only supports in_vnni=0");
    }
    // [Sqb, H] * [H, Skb] = [Sqb, Skb]
    auto k_ld =
        cm.kl_needs_trans ? (isBeamSearch ? H : cm.key.s2_str) : cm.key.h_str;
    a_gemm_tpp = BrgemmTPP<T, float, Tc>(
        Sqb,
        Skb,
        H,
        H,
        cm.key.s1_str,
        H,
        k_ld,
        Skb,
        0.0,
        0,
        1,
        cm.key.in_vnni,
        cm.kl_needs_trans);
    // [Sqb, Skb]
    scale_tpp = ScaleTPP<float, float>(Sqb * Skb);
    add_mask_tpp = AddBiasTPP<T>(Sqb, Skb);
    softmax_fwd_tpp = VarSoftMaxFwdTPP<float, T>(Sqb, Skb, USE_FLASH);
    softmax_fixup = SoftMaxFixUpTPP<T>(Sqb, H, USE_FLASH);
    softmax_scale = SoftMaxFlashScaleTPP<T>(Sqb, H, USE_FLASH);
    // [nSk, Sqb, Skb] * [nSk, Skb, H] = tmp[Sqb, H]
    c_gemm_tpp = BrgemmTPP<T, T, Tc>(
        Sqb,
        H,
        Skb,
        Sqb * Skb,
        isBeamSearch ? Skb * H : cm.value.s1_str,
        Skb,
        isBeamSearch ? H : cm.value.s2_str,
        H,
        0.0,
        0,
        1,
        cm.value.in_vnni);
    // [Sqb, H] --> [Sqb, H]
    cvt_tpp = ConvertTPP<T, T>(Sqb, H, H, H);
  }
};
#endif

template <typename T, typename Tc>
AttnKernelsNew<T, Tc>& GetAttnKernelsNew(
    long Sqb,
    long Skb,
    long H,
    const KVCacheMeta& cm,
    bool isBeamSearch) {
  static ska::flat_hash_map<std::string, AttnKernelsNew<T, Tc>*> attn_kernels;
  // Below key assumes static Skb, H and cm
  std::string key = std::to_string(Sqb) + "_" + std::to_string(H) + "_" +
      cm.hash_str + "_" + std::to_string(isBeamSearch);
  AttnKernelsNew<T, Tc>* kernel = nullptr;
  auto search = attn_kernels.find(key);
  if (search != attn_kernels.end()) {
    kernel = search->second;
  }
  if (kernel == nullptr) {
    kernel = new AttnKernelsNew<T, Tc>(Sqb, Skb, H, cm, isBeamSearch);
    attn_kernels[key] = kernel;
  }
  if (kernel == nullptr) {
    printf("Error: AttnKernelsNew is nullptr\n");
    exit(1);
  }
  return *kernel;
}

template <typename T, typename Tc = T>
inline at::Tensor attn(
    at::Tensor t_QL, // [B, N, Sq, R, H]
    std::vector<at::Tensor> t_KV,
    at::Tensor t_AM,
    const KVCacheMeta& cm,
    long key_len,
    bool isCausal = true) {
  RECORD_SCOPE(ac_gemm1, {t_QL, t_KV[0], t_KV[1]});
  auto t_CL = at::empty_like(t_QL);
  auto sizes = t_QL.sizes();
  long B = sizes[0];
  long N = sizes[1]; // = Nkv
  long Sq = sizes[2];
  long R = sizes[3]; // = Nq / Nkv
  long H = sizes[4];
  long offset = key_len - Sq;
  float one_by_sqrt_H = 1.0 / sqrtf(H);
  auto t_KL = t_KV[0];
  auto t_VL = t_KV[1];
  bool isBeamSearch = t_KV.size() == 3;
  at::Tensor t_beam_idx;
  if (isBeamSearch) {
    t_beam_idx = t_KV[2];
  } else {
    t_beam_idx = at::empty({0}, t_QL.options().dtype(at::kLong));
  }

  // KL: [NT, B, N, H, S2]
  // VL: [NT, B, N, S2, H]
  long S = t_AM.size(1);
  long S2 = cm.S2;
  long S1 = S / S2;
  // We interprete t_QL as [B, N, Sq, R, H]
  const long Sqb = (R <= 32 ? 64 / R : 1);
  long qrem = Sq % Sqb;

  const long Skb = (SK_BLOCK_SIZE + S2 - 1) / S2;

  auto s1k_str = cm.key.s1_str;
  auto s1v_str = cm.value.s1_str;
  auto bk_str = cm.key.b_str;
  auto bv_str = cm.value.b_str;
  auto nk_str = cm.key.n_str;
  auto nv_str = cm.value.n_str;

  // auto QL = GetVLAPtr<T>(t_QL,  {B, S1, S2, N, H});
  // auto QL1 = GetVLAPtr<T>(t_QL,  {B, S1q, S2, N, H});
  // auto QK1 = GetVLAPtr<T>(t_QL,  {B, N, S1k, H, S2});

  auto QL = GetVLAPtr<T>(t_QL, {N, Sq, R, H});
  // auto KL = GetVLAPtr<T>(t_QL,  {B, N, H * S2});
  auto KL = GetVLAPtrFromStrides<Tc>(t_KL, {s1k_str, bk_str, nk_str});
  auto VL = GetVLAPtrFromStrides<Tc>(t_VL, {s1v_str, bv_str, nv_str});
  auto CL = GetVLAPtr<T>(t_CL, {N, Sq, R, H});
  auto AM = GetVLAPtr<T>(t_AM, {S1, S2});
  auto beam_idx = GetVLAPtr<long>(t_beam_idx, {S1, S2});

  if (isBeamSearch) {
    // std::cout << "beam idx: " << t_beam_idx.sizes() << std::endl;
    // std::cout << "S1: " << S1 << " S2: " << S2 << std::endl;
    TPP_ASSERT(t_beam_idx.size(1) == S1 * S2, "Invalid beam_idx size\n");
  }

  static bool print_flag = true;
  if (print_flag) {
    if (my_rank == 0)
      printf("BNSqRH: %ld %ld %ld %ld %ld\n", B, N, Sq, R, H);
    printf("S S1 S2: %ld %ld %ld\n", S, S1, S2);
    printf(
        "Attn: Sqb = %ld Skb = %ld qrem = %ld use_flash = %s\n",
        Sqb,
        Skb,
        qrem,
        USE_FLASH ? "True" : "False");
    cm.print();
    print_flag = false;
  }

  AttnKernelsNew<T, Tc> attn_kern[2] = {
      GetAttnKernelsNew<T, Tc>(Sqb * R, S2, H, cm, isBeamSearch),
      GetAttnKernelsNew<T, Tc>(qrem * R, S2, H, cm, isBeamSearch),
  };
  {
    RECORD_OMP_TIME();
    {
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int sq = 0; sq < Sq; sq += Sqb) {
            long qbs = (Sq - sq >= Sqb ? Sqb : Sq - sq);
            long qbsr = qbs * R;
            int qid = (sq + Sqb > Sq) ? 1 : 0;
            auto& ak = attn_kern[qid];
            auto q_ptr = QL[b][n][sq][0];
            auto c_ptr = CL[b][n][sq][0];
            float omax[qbsr], osum[qbsr], cmax[qbsr], csum[qbsr];
            auto S1End = (!isCausal ? S1 : (sq + qbs + offset + S2 - 1) / S2);
            // auto S1End = S1;
            for (int s1 = 0; s1 < S1End; s1 += Skb) {
              long kbs = std::min(Skb, S1End - s1);
              float AS[kbs][qbsr * S2];
              T AST[kbs][qbsr * S2];
              ak.a_gemm_tpp.config();
              for (int k = 0; k < kbs; k++) {
                // printf("s1 = %d, k = %d b = %d n = %d kbs = %d qbs = %d S1End
                // = %d\n", s1, k, b, n, kbs, qbs, S1End);
                Tc* k_ptr = KL[s1 + k][b][n];
                Tc k_buf[S2 * H];
                if (isBeamSearch) {
                  auto p_idx = beam_idx[b][s1 + k];
                  for (int s2 = 0; s2 < S2; s2++) {
                    auto b_idx = p_idx[s2];
                    memcpy(
                        k_buf + s2 * H,
                        KL[s1 + k][b_idx][n] + s2 * cm.key.s2_str,
                        H * sizeof(Tc));
                  }
                  k_ptr = k_buf;
                }
                ak.a_gemm_tpp(q_ptr, k_ptr, AS[k], 1, true);
                ak.scale_tpp(AS[k], AS[k], one_by_sqrt_H);
                ak.add_mask_tpp(AM[b][s1 + k], AS[k]);
              }
              ak.a_gemm_tpp.release();
              if (isCausal) {
                auto k1end = s1 + kbs;
                for (int sq1 = 0; sq1 < qbs; sq1++) {
                  auto kstart = sq + sq1 + offset + 1;
                  auto k1start = kstart / S2;
                  auto k2start = kstart % S2;
                  for (int k1 = k1start; k1 < k1end; k1++) {
                    auto lk2start = k1 == k1start ? k2start : 0;
                    for (int r = 0; r < R; r++) {
                      for (int k2 = lk2start; k2 < S2; k2++) {
                        // AS[k1 - s1][sq1][r][k2] = -1e9f;
                        AS[k1 - s1][sq1 * S2 * R + r * S2 + k2] = -1e9f;
                      }
                    }
                  }
                }
              }
              if (s1 == 0) {
                ak.softmax_fwd_tpp(kbs, AS[0], AST[0], omax, osum, nullptr);
              } else {
                ak.softmax_fwd_tpp(kbs, AS[0], AST[0], cmax, csum, omax);
              }
              Tc v_buf[kbs * S2 * H];
              T tmp[qbs * R * H];
              Tc* v_ptr = VL[s1][b][n];
              if (isBeamSearch) {
                for (int k = 0; k < kbs; k++) {
                  auto p_idx = beam_idx[b][s1 + k];
                  for (int s2 = 0; s2 < S2; s2++) {
                    auto b_idx = p_idx[s2];
                    Tc* p_tmp = VL[s1 + k][b_idx][n] + s2 * cm.value.s2_str;
                    memcpy(v_buf + k * S2 * H + s2 * H, p_tmp, H * sizeof(Tc));
                  }
                }
                v_ptr = v_buf;
              }
              ak.c_gemm_tpp(AST[0], v_ptr, tmp, kbs);
              if (s1 == 0) {
                ak.cvt_tpp(tmp, c_ptr);
              } else {
                ak.softmax_fixup(tmp, c_ptr, cmax, csum, omax, osum);
              }
            }
            ak.softmax_scale(c_ptr, osum);
          }
        }
      }
    }
  }
  return t_CL;
}

template <typename T>
at::Tensor attn_wrapper_impl(
    at::Tensor& t_QL,
    at::Tensor& t_KL,
    at::Tensor& t_VL,
    at::Tensor& t_AM,
    c10::intrusive_ptr<TppCache> cache,
    long layer_idx,
    bool isCausal) {
  auto t_KV = cache->update(t_KL, t_VL, layer_idx);
  auto key_len = cache->get_seq_length(layer_idx);
  auto cm = cache->get_kv_cache_meta();
  at::Tensor t_CL = attn<T>(t_QL, t_KV, t_AM, cm, key_len, isCausal);
  return t_CL;
}

template at::Tensor attn_wrapper_impl<float>(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    c10::intrusive_ptr<TppCache>,
    long,
    bool);
template at::Tensor attn_wrapper_impl<bfloat16>(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    c10::intrusive_ptr<TppCache>,
    long,
    bool);
template at::Tensor attn_wrapper_impl<half>(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    c10::intrusive_ptr<TppCache>,
    long,
    bool);
template at::Tensor attn_wrapper_impl<bfloat8>(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    c10::intrusive_ptr<TppCache>,
    long,
    bool);
template at::Tensor attn_wrapper_impl<hfloat8>(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    c10::intrusive_ptr<TppCache>,
    long,
    bool);

at::Tensor attn_wrapper(
    at::Tensor& t_QL,
    at::Tensor& t_KL,
    at::Tensor& t_VL,
    at::Tensor& t_AM,
    // c10::intrusive_ptr<TppCache> cache,
    TppCache& cache1,
    long layer_idx,
    bool isCausal) {
  c10::intrusive_ptr<TppCache> cache = c10::make_intrusive<TppCache>(cache1);
  if (t_QL.dtype() == at::kFloat) {
    return attn_wrapper_impl<float>(
        t_QL, t_KL, t_VL, t_AM, cache, layer_idx, isCausal);
  } else if (t_QL.dtype() == at::kBFloat16) {
    return attn_wrapper_impl<bfloat16>(
        t_QL, t_KL, t_VL, t_AM, cache, layer_idx, isCausal);
  } else if (t_QL.dtype() == at::kHalf) {
    return attn_wrapper_impl<half>(
        t_QL, t_KL, t_VL, t_AM, cache, layer_idx, isCausal);
  } else if (t_QL.dtype() == at::kBFloat8) {
    return attn_wrapper_impl<bfloat8>(
        t_QL, t_KL, t_VL, t_AM, cache, layer_idx, isCausal);
  } else if (t_QL.dtype() == at::kHFloat8) {
    return attn_wrapper_impl<hfloat8>(
        t_QL, t_KL, t_VL, t_AM, cache, layer_idx, isCausal);
  } else {
    TPP_ASSERT(0, "Unsupported data type at %s:%d\n", __func__, __LINE__);
  }
  return at::Tensor();
}

#if 0
// Old attention kernels

REGISTER_LOCAL_SCOPE(ac_gemm1, "ac_gemm1");
REGISTER_LOCAL_SCOPE(ac_gemm2, "ac_gemm2");
REGISTER_LOCAL_SCOPE(ac_gemm21, "ac_gemm21");
REGISTER_LOCAL_SCOPE(ac_gemm22, "ac_gemm22");

#define S_FIRST_KVC
#define PER_THREAD_COPY

static long get_batch_dim_in_kv_cache() {
#ifdef S_FIRST_KVC
  return 1;
#else
  return 0;
#endif
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
  SCOPEIT_DECL(SoftMaxFlashScaleTPP<T>) softmax_scale;

  AttnKernels(
      long Sqb,
      long Skb,
      long Sk_pad,
      long H,
      int pad,
      int kl_in_vnni,
      int vl_in_vnni) {
    // printf("Sqb: %ld, Skb: %ld, H: %ld, psd: %d, kl: %d, vl: %d\n", Sqb,
    // Skb, H, pad, kl_in_vnni, vl_in_vnni);
    if (Sk_pad < Skb)
      Sk_pad = Skb;
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
    add_2dmask_tpp =
        SCOPEIT((AddTPP<T, float, float>(Sqb, Skb, Sk_pad, Skb, Skb)), EW_ADD);
    softmax_fwd_tpp =
        SCOPEIT((VarSoftMaxFwdTPP<float, Tv>(Sqb, Skb, USE_FLASH)), SOFTMAX);
    softmax_fixup = SCOPEIT((SoftMaxFixUpTPP<T>(Sqb, H, USE_FLASH)), EW_RCP);
    softmax_scale =
        SCOPEIT((SoftMaxFlashScaleTPP<T>(Sqb, H, USE_FLASH)), EW_RCP);
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


template <typename T, typename Tv>
inline at::Tensor attn(
    at::Tensor t_QL,
    at::Tensor t_KL,
    at::Tensor t_AM,
    at::Tensor t_VL,
    bool isCausal = true) {
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
  const bool am_valid = (t_AM.numel() > 0);
  bool am_is_2d = am_valid && t_AM.size(2) != 1;

  int vl_in_vnni = 1; //(Sk % 2 == 0 ? 1 : 0);
  const long VBS = (vl_in_vnni ? get_vnni_block_size<T>() : 1);
  long Sk_pad = (Sk + VBS - 1) & ~(VBS - 1);
  // const long Skb = (!inline_trans ? 2048 : SK_BLOCK_SIZE);
  const long Skb = SK_BLOCK_SIZE;
  long krem = Sk % Skb;
  int pad = Sk_pad - Sk;

  auto t_KL_TV = t_KL.new_empty({B, Nkv, Sk_pad, H});
  auto t_VL_V = t_VL;
  if (VBS != 1) {
    t_VL_V = t_VL.new_empty({B, Nkv, Sk_pad, H});
  }
  if (am_valid && Sk != Sk_pad) {
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
  static bool print_flag = true;
  if (print_flag) {
    if (my_rank == 0)
      printf(
          "Attn: Sqb = %ld Skb = %ld use_flash = %s\n",
          Sqb,
          Skb,
          USE_FLASH ? "True" : "False");
    print_flag = false;
  }

  AttnKernels<T, Tv> attn_kern[4] = {
      AttnKernels<T, Tv>(Sqb, Skb, Sk_pad, H, 0, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(
          Sqb, krem + pad, Sk_pad, H, pad, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(qrem, Skb, Sk_pad, H, 0, kl_in_vnni, vl_in_vnni),
      AttnKernels<T, Tv>(
          qrem, krem + pad, Sk_pad, H, pad, kl_in_vnni, vl_in_vnni),
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
              if (!am_is_2d && isCausal) {
                for (int sq1 = 0; sq1 < qbs; sq1++) {
                  auto qval = sq + sq1 + offset;
                  for (int sk1 = qval + 1; sk1 < sk + kbs; sk1++) {
                    AS[sq1][sk1 - sk] = -1e9f;
                  }
                }
              }
              ak.scale_tpp(AS[0], AS[0], one_by_sqrt_H);
              if (am_valid) {
                if (am_is_2d)
                  ak.add_2dmask_tpp(&AM2[b][sq][sk], AS[0], AS[0]);
                else
                  ak.add_mask_tpp(&AM[b][sk], AS[0]);
              }
              if (sk == 0) {
                ak.softmax_fwd_tpp(1, AS[0], AST[0], omax, osum, nullptr);
              } else {
                ak.softmax_fwd_tpp(1, AS[0], AST[0], cmax, csum, omax);
              }
              // for (int xx=0;xx<kbs;xx++)
              // if (b == 0 && n == 0 && Sq == 1) printf("AS[%d]: %g\n",
              // sk+xx, (float)AST[0][xx]);
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
            attn_kern[qid * 2].softmax_scale(CL[b][nq][sq], osum);
          }
        }
      }
    }
  }
  return t_CL;
}

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
#if defined(__AVX512F__) && defined(S_FIRST_KVC)
  constexpr long FSk_BS = 16L;
#else
  constexpr long FSk_BS = 64L;
#endif
  auto FSk_aligned = (FSk + (FSk_BS - 1)) & ~(FSk_BS - 1);
#ifdef S_FIRST_KVC
  // auto CSk = t_KL_cache.size(0);
#else
  // auto CSk = t_KL_cache.size(2);
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

#ifdef __AVX512F__
#pragma message "Using AVX512 attn"
  TPP_ASSERT(H % 16 == 0, "Head size must be multiple of 16\n");
#if 1
#ifdef S_FIRST_KVC
  const int nh = H / 16;
  const int nbFSk = FSk_aligned / FSk_BS;
  auto t_AS = t_QL.new_empty({nbFSk, B, Nq, FSk_BS}, at::kFloat);
  auto AS = GetVLAPtr<float>(t_AS, {B, Nq, FSk_BS});
#ifndef PER_THREAD_COPY
  auto t_tmpCL = t_QL.new_empty({nbFSk, B, Nq, H}, at::kFloat);
  auto tmpCL = GetVLAPtr<float>(t_tmpCL, {B, Nq, H});
#else
  const int nThreads = omp_get_max_threads();
  auto t_tmpCL = t_QL.new_empty({nThreads, B, Nq, H}, at::kFloat);
  auto tmpCL = GetVLAPtr<float>(t_tmpCL, {B, Nq, H});
  auto t_accFlags = t_QL.new_zeros({nThreads, B, Nq}, at::kByte);
  auto accFlags = GetVLAPtr<uint8_t>(t_accFlags, {B, Nq});
#endif
  {
    RECORD_OMP_TIME();
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int nkv = 0; nkv < Nkv; nkv++) {
#ifdef S_FIRST_KVC
        memcpy(KL_C[FSk - 1][b][nkv], KL[b][nkv][0], H * sizeof(T));
        memcpy(VL_C[FSk - 1][b][nkv], VL[b][nkv][0], H * sizeof(T));
#else
        memcpy(KL_C[b][nkv][FSk - 1], KL[b][nkv][0], H * sizeof(T));
        memcpy(VL_C[b][nkv][FSk - 1], VL[b][nkv][0], H * sizeof(T));
#endif
      }
    }
#pragma omp parallel for collapse(3)
    for (int sk1 = 0; sk1 < nbFSk; sk1++) {
      for (int nq = 0; nq < Nq; nq++) {
        for (int b = 0; b < B; b++) {
          int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
          __m512 vql[nh];
          for (int h = 0; h < nh; h++) {
            vql[h] = _mm512_loadu_ps_auto(QL[b][nq][0] + h * 16);
          }
          int sk_off = sk1 * FSk_BS;
          for (int sk2 = 0; sk2 < FSk_BS; sk2++) {
            int sk = sk_off + sk2;
            if (sk < FSk) {
              int bid = beam_idx[b][sk];
              __m512 vas = _mm512_setzero_ps();
              for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
                auto vklc = _mm512_loadu_ps_auto(KL_C[sk][bid][nkv] + h * 16);
#else
                auto vklc = _mm512_loadu_ps_auto(KL_C[bid][nkv][sk] + h * 16);
#endif
                vas = _mm512_fmadd_ps(vql[h], vklc, vas);
              }
              float as = _mm512_reduce_add_ps(vas);
              as *= one_by_sqrt_H;
              if (am_valid) {
                as += AM[b][sk];
              }
              AS[sk1][b][nq][sk2] = as;
            } else {
              AS[sk1][b][nq][sk2] = -1e10;
            }
          }
        }
      }
    }
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int nq = 0; nq < Nq; nq++) {
        __m512 vmax = _mm512_set1_ps(-1e20);
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[sk1][b][nq];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps_auto(ASP + sk2));
          }
        }
        float max = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max);
        __m512 vsum = _mm512_setzero_ps();
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[sk1][b][nq];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(
                _mm512_sub_ps(_mm512_loadu_ps_auto(ASP + sk2), vmax));
            _mm512_storeu_ps(ASP + sk2, vz);
            vsum = _mm512_add_ps(vsum, vz);
          }
        }
        float sum = _mm512_reduce_add_ps(vsum);
        sum = 1.0 / sum;
        vsum = _mm512_set1_ps(sum);
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[sk1][b][nq];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            auto vmul = _mm512_mul_ps(_mm512_loadu_ps_auto(ASP + sk2), vsum);
            _mm512_storeu_ps(ASP + sk2, vmul);
          }
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int sk1 = 0; sk1 < nbFSk; sk1++) {
      for (int nq = 0; nq < Nq; nq++) {
        for (int b = 0; b < B; b++) {
#ifdef PER_THREAD_COPY
          int tid = omp_get_thread_num();
#endif
          int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
          __m512 vql[nh];
          for (int h = 0; h < nh; h++) {
#ifdef PER_THREAD_COPY
            if (accFlags[tid][b][nq] == 0) {
              vql[h] = _mm512_setzero_ps();
            } else {
              vql[h] = _mm512_loadu_ps_auto(tmpCL[tid][b][nq] + h * 16);
            }
#else
            vql[h] = _mm512_setzero_ps();
#endif
          }
          int sk_off = sk1 * FSk_BS;
          float* ASP = AS[sk1][b][nq];
          for (int sk2 = 0; sk2 < FSk_BS; sk2++) {
            int sk = sk_off + sk2;
            if (sk < FSk) {
              int bid = beam_idx[b][sk];
              __m512 vas = _mm512_set1_ps(ASP[sk2]);
              for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
                auto vvlc = _mm512_loadu_ps_auto(VL_C[sk][bid][nkv] + h * 16);
#else
                auto vvlc = _mm512_loadu_ps_auto(VL_C[bid][nkv][sk] + h * 16);
#endif
                vql[h] = _mm512_fmadd_ps(vvlc, vas, vql[h]);
              }
            }
          }
          for (int h = 0; h < nh; h++) {
#ifndef PER_THREAD_COPY
            _mm512_storeu_ps_auto(tmpCL[sk1][b][nq] + h * 16, vql[h]);
#else
            _mm512_storeu_ps_auto(tmpCL[tid][b][nq] + h * 16, vql[h]);
#endif
          }
#ifdef PER_THREAD_COPY
          accFlags[tid][b][nq] = 1;
#endif
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int nq = 0; nq < Nq; nq++) {
        for (int h = 0; h < nh; h++) {
          auto vec = _mm512_setzero_ps();
#ifndef PER_THREAD_COPY
          for (int sk1 = 0; sk1 < nbFSk; sk1++) {
            vec = _mm512_add_ps(
                vec, _mm512_loadu_ps_auto(&tmpCL[sk1][b][nq][h * 16]));
          }
#else
          for (int tid = 0; tid < nThreads; tid++) {
            if (accFlags[tid][b][nq] == 0)
              continue;
            vec = _mm512_add_ps(
                vec, _mm512_loadu_ps_auto(&tmpCL[tid][b][nq][h * 16]));
          }
#endif
          _mm512_storeu_ps_auto(&CL[b][nq][0][h * 16], vec);
        }
      }
    }
  }
#else // S_FIRST_KVC is not define
  const int nh = H / 16;
  const int nbFSk = FSk_aligned / FSk_BS;
  auto t_AS = t_QL.new_empty({B, Nq, nbFSk, FSk_BS}, at::kFloat);
  auto t_tmpCL = t_QL.new_empty({B, Nq, nbFSk, H}, at::kFloat);
  auto AS = GetVLAPtr<float>(t_AS, {Nq, nbFSk, FSk_BS});
  auto tmpCL = GetVLAPtr<float>(t_tmpCL, {Nq, nbFSk, H});
  {
    RECORD_OMP_TIME();
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int nkv = 0; nkv < Nkv; nkv++) {
#ifdef S_FIRST_KVC
        memcpy(KL_C[FSk - 1][b][nkv], KL[b][nkv][0], H * sizeof(T));
        memcpy(VL_C[FSk - 1][b][nkv], VL[b][nkv][0], H * sizeof(T));
#else
        memcpy(KL_C[b][nkv][FSk - 1], KL[b][nkv][0], H * sizeof(T));
        memcpy(VL_C[b][nkv][FSk - 1], VL[b][nkv][0], H * sizeof(T));
#endif
      }
    }
#pragma omp parallel for collapse(3)
    for (int nq = 0; nq < Nq; nq++) {
      for (int b = 0; b < B; b++) {
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
          __m512 vql[nh];
          for (int h = 0; h < nh; h++) {
            vql[h] = _mm512_loadu_ps_auto(QL[b][nq][0] + h * 16);
          }
          int sk_off = sk1 * FSk_BS;
          for (int sk2 = 0; sk2 < FSk_BS; sk2++) {
            int sk = sk_off + sk2;
            if (sk < FSk) {
              int bid = beam_idx[b][sk];
              __m512 vas = _mm512_setzero_ps();
              for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
                auto vklc = _mm512_loadu_ps_auto(KL_C[sk][bid][nkv] + h * 16);
#else
                auto vklc = _mm512_loadu_ps_auto(KL_C[bid][nkv][sk] + h * 16);
#endif
                vas = _mm512_fmadd_ps(vql[h], vklc, vas);
              }
              float as = _mm512_reduce_add_ps(vas);
              as *= one_by_sqrt_H;
              if (am_valid) {
                as += AM[b][sk];
              }
              AS[b][nq][sk1][sk2] = as;
            } else {
              AS[b][nq][sk1][sk2] = -1e10;
            }
          }
        }
      }
    }
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int nq = 0; nq < Nq; nq++) {
        __m512 vmax = _mm512_set1_ps(-1e20);
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[b][nq][sk1];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps_auto(ASP + sk2));
          }
        }
        float max = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max);
        __m512 vsum = _mm512_setzero_ps();
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[b][nq][sk1];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(
                _mm512_sub_ps(_mm512_loadu_ps_auto(ASP + sk2), vmax));
            _mm512_storeu_ps(ASP + sk2, vz);
            vsum = _mm512_add_ps(vsum, vz);
          }
        }
        float sum = _mm512_reduce_add_ps(vsum);
        sum = 1.0 / sum;
        vsum = _mm512_set1_ps(sum);
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[b][nq][sk1];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            auto vmul = _mm512_mul_ps(_mm512_loadu_ps_auto(ASP + sk2), vsum);
            _mm512_storeu_ps(ASP + sk2, vmul);
          }
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int nq = 0; nq < Nq; nq++) {
      for (int b = 0; b < B; b++) {
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
          __m512 vql[nh];
          for (int h = 0; h < nh; h++) {
            vql[h] = _mm512_setzero_ps();
          }
          int sk_off = sk1 * FSk_BS;
          float* ASP = AS[b][nq][sk1];
          for (int sk2 = 0; sk2 < FSk_BS; sk2++) {
            int sk = sk_off + sk2;
            if (sk < FSk) {
              int bid = beam_idx[b][sk];
              __m512 vas = _mm512_set1_ps(ASP[sk2]);
              for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
                auto vvlc = _mm512_loadu_ps_auto(VL_C[sk][bid][nkv] + h * 16);
#else
                auto vvlc = _mm512_loadu_ps_auto(VL_C[bid][nkv][sk] + h * 16);
#endif
                vql[h] = _mm512_fmadd_ps(vvlc, vas, vql[h]);
              }
            }
          }
          for (int h = 0; h < nh; h++) {
            _mm512_storeu_ps_auto(tmpCL[b][nq][sk1] + h * 16, vql[h]);
          }
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int nq = 0; nq < Nq; nq++) {
      for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
          auto vec = _mm512_setzero_ps();
          for (int sk1 = 0; sk1 < nbFSk; sk1++) {
            vec = _mm512_add_ps(
                vec, _mm512_loadu_ps_auto(&tmpCL[b][nq][sk1][h * 16]));
          }
          _mm512_storeu_ps_auto(&CL[b][nq][0][h * 16], vec);
        }
      }
    }
  }
#endif
#else
  const int nh = H / 16;
  RECORD_OMP_TIME();
#pragma omp parallel
  {
    TimerStart();
#pragma omp for collapse(2) nowait
    for (int nq = 0; nq < Nq; nq++) {
      for (int b = 0; b < B; b++) {
        LIBXSMM_ALIGNED(float AS[FSk_aligned], FSk_BS);
        int nkv = Nq_per_kv == 1 ? nq : nq / Nq_per_kv;
        {
          ScopedTimer t_(BRGEMM, 2 * FSk * H);
#ifdef S_FIRST_KVC
          memcpy(KL_C[FSk - 1][b][nkv], KL[b][nkv][0], H * sizeof(T));
          memcpy(VL_C[FSk - 1][b][nkv], VL[b][nkv][0], H * sizeof(T));
#else
          memcpy(KL_C[b][nkv][FSk - 1], KL[b][nkv][0], H * sizeof(T));
          memcpy(VL_C[b][nkv][FSk - 1], VL[b][nkv][0], H * sizeof(T));
#endif
          __m512 vql[nh];
          for (int h = 0; h < nh; h++) {
            vql[h] = _mm512_loadu_ps_auto(QL[b][nq][0] + h * 16);
          }
          float max = -1e20;
          int sk;
          for (sk = 0; sk < FSk; sk++) {
            int bid = beam_idx[b][sk];
            __m512 vas = _mm512_setzero_ps();
            for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
              auto vklc = _mm512_loadu_ps_auto(KL_C[sk][bid][nkv] + h * 16);
#else
              auto vklc = _mm512_loadu_ps_auto(KL_C[bid][nkv][sk] + h * 16);
#endif
              vas = _mm512_fmadd_ps(vql[h], vklc, vas);
            }
            float as = _mm512_reduce_add_ps(vas);
            as *= one_by_sqrt_H;
            if (am_valid) {
              as += AM[b][sk];
            }
            max = std::max(max, as);
            AS[sk] = as;
          }
          __m512 vmax = _mm512_set1_ps(max);
          __m512 vsum = _mm512_setzero_ps();
          for (sk = 0; sk < ALIGNDOWN(FSk, 16); sk += 16) {
            __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(
                _mm512_sub_ps(_mm512_loadu_ps_auto(AS + sk), vmax));
            _mm512_storeu_ps(AS + sk, vz);
            vsum = _mm512_add_ps(vsum, vz);
          }
          if (sk < FSk) {
            int rem = FSk - sk;
            __mmask16 mask = (1 << rem) - 1;
            __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(
                _mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, AS + sk), vmax));
            _mm512_mask_storeu_ps(AS + sk, mask, vz);
            vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
          }
          float sum = _mm512_reduce_add_ps(vsum);
          sum = 1.0 / sum;
          for (int h = 0; h < nh; h++) {
            vql[h] = _mm512_setzero_ps();
          }
          for (sk = 0; sk < FSk; sk++) {
            int bid = beam_idx[b][sk];
            __m512 vas = _mm512_set1_ps(AS[sk] * sum);
            for (int h = 0; h < nh; h++) {
#ifdef S_FIRST_KVC
              auto vvlc = _mm512_loadu_ps_auto(VL_C[sk][bid][nkv] + h * 16);
#else
              auto vvlc = _mm512_loadu_ps_auto(VL_C[bid][nkv][sk] + h * 16);
#endif
              vql[h] = _mm512_fmadd_ps(vvlc, vas, vql[h]);
            }
          }
          for (int h = 0; h < nh; h++) {
            _mm512_storeu_ps_auto(CL[b][nq][0] + h * 16, vql[h]);
          }
        }
      }
    }
    TimerEnd();
  }
#endif
#else
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
        // int tid = omp_get_thread_num();
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
              ScopedTimer t_(BRGEMM, 2 * FSk * H);
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
              ScopedTimer t_(BRGEMM, 2 * FSk * H);
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
#endif
  return t_CL;
}

#endif
REGISTER_SUBMODULE(_tpp_attn, m) {
  py::class_<TppCache>(m, "TppCache")
      .def(py::init<>())
      .def("update", &TppCache::update)
      .def("get_seq_length", &TppCache::get_seq_length)
      .def("size", &TppCache::size)
      .def("get", &TppCache::get)
      .def("get_capacity", &TppCache::get_capacity)
      //.def("get_kv_cache_meta", &TppCache::get_kv_cache_meta)
      .def("align_and_invert_mask", &TppCache::align_and_invert_mask)
      .def("crop", &TppCache::crop)
      .def("reorder_cache", &TppCache::reorder_cache);
  m.def("attn", &attn_wrapper);
}

TORCH_LIBRARY_FRAGMENT(tpp_llm, m) {
  m.class_<TppCache>("TppCache")
      .def(torch::init<>())
      .def("update", &TppCache::update)
      .def("get_seq_length", &TppCache::get_seq_length)
      .def("size", &TppCache::size)
      .def("get", &TppCache::get)
      .def("get_capacity", &TppCache::get_capacity)
      //.def("get_kv_cache_meta", &TppCache::get_kv_cache_meta)
      .def("align_and_invert_mask", &TppCache::align_and_invert_mask)
      .def("crop", &TppCache::crop)
      .def("reorder_cache", &TppCache::reorder_cache);
}
