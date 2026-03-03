/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Evangelos Georganas (Intel Corp.)
 ******************************************************************************/

#ifndef SFC_CA_GEMM_TPP_H
#define SFC_CA_GEMM_TPP_H

#pragma once

#include <ATen/record_function.h>
#include <torch/extension.h>

#include "ext_tpp.h"
#include "timing.h"
#include "xsmm_functors.h"

#include <omp.h>
#include <map>
#include <vector>

// Include SFC utilities and performance model headers
#include "sfc_ca_gemm/sfc_utils.h"

extern "C" {
#include "sfc_ca_gemm/knn_model.h"
#include "sfc_ca_gemm/roofline_predictor.h"
}

using namespace tpp;

// Environment variable to control SFC CA GEMM usage
// 0 = disabled, 1 = enabled with roofline model, 2 = enabled with kNN model
static int USE_SFC_CA_GEMM = env2int("USE_SFC_CA_GEMM", 0);

REGISTER_LOCAL_SCOPE(sfc_gemm, "sfc_gemm");

// ============================================================================
// Global scratch buffer for SFC CA GEMM (2 GB, allocated once, reused)
// ============================================================================
#define SFC_CA_GEMM_SCRATCH_SIZE (2048UL * 1024UL * 1024UL)  // 2 GB

static void* sfc_ca_gemm_global_scratch = nullptr;

inline void* get_sfc_ca_gemm_scratch() {
  if (sfc_ca_gemm_global_scratch == nullptr) {
    sfc_ca_gemm_global_scratch =
        libxsmm_aligned_malloc(SFC_CA_GEMM_SCRATCH_SIZE, 2097152);
  }
  return sfc_ca_gemm_global_scratch;
}

// ============================================================================
// SFC Index Map utilities (from sfc_ca_gemm.hpp, self-contained here)
// ============================================================================
inline unsigned int sfc_ca_gemm_fill_sfc_index_map(
    unsigned char** sfc_index_map,
    unsigned int Mb,
    unsigned int Nb) {
  long long i, n_tasks = (long long)Mb * Nb;
  int m_id, n_id;
  if (Mb < 256 && Nb < 256) {
    unsigned char* map;
    *sfc_index_map = (unsigned char*)libxsmm_aligned_malloc(
        2 * Mb * Nb * sizeof(unsigned char), 2097152);
    map = (unsigned char*)*sfc_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy(&m_id, &n_id, i, Mb, Nb);
      map[2 * i + 0] = (unsigned char)m_id;
      map[2 * i + 1] = (unsigned char)n_id;
    }
    return 1;
  } else if (Mb < 65536 && Nb < 65536) {
    unsigned short* map;
    *sfc_index_map = (unsigned char*)libxsmm_aligned_malloc(
        2 * Mb * Nb * sizeof(unsigned short), 2097152);
    map = (unsigned short*)*sfc_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy(&m_id, &n_id, i, Mb, Nb);
      map[2 * i + 0] = (unsigned short)m_id;
      map[2 * i + 1] = (unsigned short)n_id;
    }
    return 2;
  } else {
    unsigned int* map;
    *sfc_index_map = (unsigned char*)libxsmm_aligned_malloc(
        2 * Mb * Nb * sizeof(unsigned int), 2097152);
    map = (unsigned int*)*sfc_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy(&m_id, &n_id, i, Mb, Nb);
      map[2 * i + 0] = (unsigned int)m_id;
      map[2 * i + 1] = (unsigned int)n_id;
    }
    return 4;
  }
}

inline void sfc_ca_gemm_extract_indices_from_sfc(
    int* i_m,
    int* i_n,
    unsigned char* sfc_index_map,
    int sfc_index,
    unsigned int index_tsize) {
  if (index_tsize == 1) {
    unsigned char* map = (unsigned char*)sfc_index_map;
    *i_m = (int)map[2 * sfc_index + 0];
    *i_n = (int)map[2 * sfc_index + 1];
  } else if (index_tsize == 2) {
    unsigned short* map = (unsigned short*)sfc_index_map;
    *i_m = (int)map[2 * sfc_index + 0];
    *i_n = (int)map[2 * sfc_index + 1];
  } else {
    unsigned int* map = (unsigned int*)sfc_index_map;
    *i_m = (int)map[2 * sfc_index + 0];
    *i_n = (int)map[2 * sfc_index + 1];
  }
}

// ============================================================================
// Global SFC index map cache: keyed by (Mb, Nb), allocated once and reused
// across all GEMM instances that share the same (Mb, Nb) block grid.
// ============================================================================
struct SfcIndexMapEntry {
  unsigned char* sfc_index_map;
  unsigned int index_tsize;
};

static std::map<std::pair<unsigned int, unsigned int>, SfcIndexMapEntry>
    sfc_index_map_cache;

inline SfcIndexMapEntry get_or_create_sfc_index_map(
    unsigned int Mb,
    unsigned int Nb) {
  auto key = std::make_pair(Mb, Nb);
  auto it = sfc_index_map_cache.find(key);
  if (it != sfc_index_map_cache.end()) {
    return it->second;
  }
  SfcIndexMapEntry entry;
  entry.index_tsize =
      sfc_ca_gemm_fill_sfc_index_map(&entry.sfc_index_map, Mb, Nb);
  sfc_index_map_cache[key] = entry;
  return entry;
}

// ============================================================================
// SFC CA GEMM Configuration
// ============================================================================

// Configuration struct for SFC CA GEMM - holds precomputed kernels and maps
struct SfcCaGemmConfig {
  long M, N, K;
  long Mb, Nb, Kb;
  long bm, bn, bk;
  long K_layers;
  long brcount;
  long Kb_per_layer;
  long Kb_last_layer;
  long K_rounds_per_layer;

  // SFC index map
  unsigned char* sfc_index_map;
  unsigned int index_tsize;

  // LIBXSMM kernels
  libxsmm_gemmfunction brgemm_kernel;
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_tilecfgfunction tileconfig_kernel;
  libxsmm_tilecfgfunction tilerelease_kernel;
  libxsmm_meltwfunction_binary l_add_kernel;       // full-block add (bn x bm, stride N) for K_layers == 2
  libxsmm_meltwfunction_binary l_add_kernel_row;   // per-row add (bn x 1) for K_layers > 2
  libxsmm_meltwfunction_unary l_reduce_kernel;     // per-row reduce (bn elements, n_out_copies sources)

  SfcCaGemmConfig()
      : M(0), N(0), K(0), Mb(0), Nb(0), Kb(0),
        bm(0), bn(0), bk(0), K_layers(1), brcount(1),
        Kb_per_layer(0), Kb_last_layer(0), K_rounds_per_layer(0),
        sfc_index_map(nullptr), index_tsize(0),
        brgemm_kernel(nullptr), zero_kernel(nullptr),
        tileconfig_kernel(nullptr), tilerelease_kernel(nullptr),
        l_add_kernel(nullptr), l_add_kernel_row(nullptr),
        l_reduce_kernel(nullptr) {}

  ~SfcCaGemmConfig() {
  }
};

// ============================================================================
// Helper: get libxsmm dtype from C++ type
// ============================================================================
template <typename T>
inline libxsmm_datatype sfc_get_libxsmm_dtype() {
  if constexpr (std::is_same_v<T, float>)
    return LIBXSMM_DATATYPE_F32;
  else if constexpr (std::is_same_v<T, bfloat16>)
    return LIBXSMM_DATATYPE_BF16;
  else if constexpr (std::is_same_v<T, half>)
    return LIBXSMM_DATATYPE_F16;
#ifdef PYTORCH_SUPPORTS_FLOAT8
  else if constexpr (std::is_same_v<T, bfloat8>)
    return LIBXSMM_DATATYPE_BF8;
  else if constexpr (std::is_same_v<T, hfloat8>)
    return LIBXSMM_DATATYPE_HF8;
#endif
  else
    return LIBXSMM_DATATYPE_BF16;  // default
}

// ============================================================================
// Setup SFC CA GEMM config
// ============================================================================
template <typename T, typename Tout = T>
SfcCaGemmConfig* setup_sfc_ca_gemm_config(
    long M,
    long N,
    long K,
    long bm,
    long bn,
    long bk) {
  auto* config = new SfcCaGemmConfig();

  // Calculate blocking parameters
  long Mb = M / bm;
  long Nb = N / bn;
  long Kb = K / bk;

  // Predict optimal kbf and K_layers using the performance model
  int predicted_kbf = 1;
  int predicted_K_layers = 1;

  // For GEMV-like cases (M <= 32), skip the performance model and use
  // kbf=1, K_layers=1.
  if (M <= 32) {
    predicted_kbf = 1;
    predicted_K_layers = 1;
  } else if (USE_SFC_CA_GEMM == 1) {
    // Use roofline model
    predict_config_roofline_simple(M, N, K, &predicted_kbf, &predicted_K_layers);
  } else if (USE_SFC_CA_GEMM == 2) {
    // Use kNN model
    predict_config_knn((int)M, (int)N, (int)K, &predicted_kbf, &predicted_K_layers);
  } else {
    predicted_kbf = 1;
    predicted_K_layers = 1;
  }

  long kbf = predicted_kbf;
  long K_layers = predicted_K_layers;

  //printf("SFC CA GEMM: Inquired GEMM %ld x %ld x %ld and recommend (K_layers, kbf) = (%ld, %ld)\n",
  //       M, N, K, K_layers, kbf);

  long Kb_per_layer = (Kb + K_layers - 1) / K_layers;
  long brcount = (Kb_per_layer + kbf - 1) / kbf;
  long K_rounds_per_layer = (Kb_per_layer + brcount - 1) / brcount;
  long Kb_last_layer = Kb - (K_layers - 1) * Kb_per_layer;

  // Store parameters
  config->M = M;
  config->N = N;
  config->K = K;
  config->Mb = Mb;
  config->Nb = Nb;
  config->Kb = Kb;
  config->bm = bm;
  config->bn = bn;
  config->bk = bk;
  config->K_layers = K_layers;
  config->brcount = brcount;
  config->Kb_per_layer = Kb_per_layer;
  config->Kb_last_layer = Kb_last_layer;
  config->K_rounds_per_layer = K_rounds_per_layer;

  // Get or create SFC index map from global cache (reused across GEMMs)
  auto sfc_map_entry = get_or_create_sfc_index_map(Mb, Nb);
  unsigned char* sfc_index_map = sfc_map_entry.sfc_index_map;
  unsigned int index_tsize = sfc_map_entry.index_tsize;
  config->sfc_index_map = sfc_index_map;
  config->index_tsize = index_tsize;

  // Setup LIBXSMM TPP kernels
  auto dtype = sfc_get_libxsmm_dtype<T>();
  auto dtype_out = sfc_get_libxsmm_dtype<Tout>();
  auto dtype_comp = LIBXSMM_DATATYPE_F32;

  auto l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') |
      LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG |
      LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG |
      LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
      LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');

  // For flat layouts:
  //   A (activations): [M][K] with lda = K (full contraction dim)
  //   B (weights): blocked [Nb][Kb][bk*bn] with ldb = bn
  //   C (output): [M][N] with ldc = N (full output feature dim)
  auto l_shape = libxsmm_create_gemm_shape(
      bm, bn, bk, K, bk, N, dtype, dtype, dtype_out, dtype_comp);
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  // stride_a = bk * sizeof(T): consecutive K-blocks in flat A are bk apart
  // stride_b = bk * bn * sizeof(T): consecutive K-blocks in blocked weights
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config(
      LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
      bk * sizeof(T),
      bk * bn * sizeof(T),
      brcount);

  if (K_rounds_per_layer == 1)
    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  // Zero kernel: zero a bm x bn block in flat C layout (stride N between rows)
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(
      bn, bm, N, N, dtype_out, dtype_out, dtype_comp);
  config->zero_kernel = libxsmm_dispatch_meltw_unary(
      LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  config->tileconfig_kernel = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tc_flags);
  config->tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tr_flags);
  config->brgemm_kernel =
      libxsmm_dispatch_brgemm(l_shape, l_flags, l_prefetch_flags, l_brconfig);

  // Add kernel: element-wise add of bm x bn blocks in flat layout (for K_layers == 2)
  auto l_binary_shape = libxsmm_create_meltw_binary_shape(
      bn, bm, N, N, N, dtype_out, dtype_out, dtype_out, dtype_comp);
  config->l_add_kernel = libxsmm_dispatch_meltw_binary(
      LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape,
      LIBXSMM_MELTW_FLAG_BINARY_NONE);

  // Per-row add kernel: add bn elements, one row at a time (for K_layers > 2)
  // When doing per-row reduction, reduce_scratch is contiguous bn elements,
  // while the output row has stride N. With n=1, ld values don't affect striding.
  auto l_binary_shape_row = libxsmm_create_meltw_binary_shape(
      bn, 1, N, N, N, dtype_out, dtype_out, dtype_out, dtype_comp);
  config->l_add_kernel_row = libxsmm_dispatch_meltw_binary(
      LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape_row,
      LIBXSMM_MELTW_FLAG_BINARY_NONE);

  // Reduce kernel: per-row column-reduce across n_out_copies K-layer partial outputs
  // For flat C layout [M][N], each block's rows are N apart (non-contiguous).
  // We reduce one row (bn elements) at a time, with stride M*N between copies.
  int n_out_copies = std::max(1L, K_layers - 1);
  auto l_reduce_shape = libxsmm_create_meltw_unary_shape(
      bn, n_out_copies, M * N, bn, dtype_out, dtype_out, dtype_comp);
  config->l_reduce_kernel = libxsmm_dispatch_meltw_unary(
      LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_reduce_shape,
      LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);

  return config;
}

// ============================================================================
// SFC CA Blocked Linear Weight class
// ============================================================================
// This class integrates the SFC CA GEMM approach into the TPP extension's
// blocked linear GEMM hierarchy. It inherits from TppBlockedLinearWBase
// and overrides the operator() to use SFC-ordered cache-aware GEMM.

template <typename T, typename TW, typename TOUT = T>
class SfcCaBlockedLinearW : public TppBlockedLinearWBase<T, TOUT> {
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
  // Standard brgemm TPPs (beta=1 for accumulation across K-rounds)
  SCOPEIT_DECL(BrgemmTPP<T, Tout, Tw>) brgemm_tpp, brgemm_tpp_rem;

  // SFC CA GEMM config (lazy initialized)
  mutable SfcCaGemmConfig* sfc_config = nullptr;

 public:
  SfcCaBlockedLinearW(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias)
      : TppBlockedLinearWBase<Tin, Tout>(t_in, t_wt, t_bias) {
    int b_vnni = 1;

    // beta=1.0: accumulate across multiple K-rounds (explicit zero at i_k==0)
    brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, Tout, Tw>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));
    brgemm_tpp_rem = SCOPEITGEMM((BrgemmTPP<T, Tout, Tw>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb, b_vnni)));

    loop_scheme =
        weight_reuse ? GEMM_LOOP_SCHEME_REUSE : GEMM_LOOP_SCHEME_STREAMING;
  }

  // Get or create the SFC CA GEMM config for given dimensions
  SfcCaGemmConfig* get_sfc_config(long BS) const {
    if (sfc_config && sfc_config->M == BS && sfc_config->N == K &&
        sfc_config->K == C) {
      return sfc_config;
    }
    // Create new config
    // In SFC CA GEMM terms:
    //   M_sfc = BS (batch size - rows)
    //   N_sfc = K (output features - columns)
    //   K_sfc = C (input/contraction features)
    //   bm_sfc = BSb (batch block size, now 32 when SFC enabled)
    //   bn_sfc = Hk (output feature block size, constrained by weight layout)
    //   bk_sfc = Hc (input feature block size, constrained by weight layout)
    long bm_sfc = BSb;
    long bn_sfc = Hk;
    long bk_sfc = Hc;

    // Ensure blocking divides evenly
    while (BS % bm_sfc != 0 && bm_sfc > 1)
      bm_sfc--;
    while (K % bn_sfc != 0 && bn_sfc > 1)
      bn_sfc--;
    while (C % bk_sfc != 0 && bk_sfc > 1)
      bk_sfc--;

    // Only use SFC CA GEMM if blocking is clean
    if (BS % bm_sfc != 0 || K % bn_sfc != 0 || C % bk_sfc != 0) {
      return nullptr;
    }

    sfc_config = setup_sfc_ca_gemm_config<T, Tout>(BS, K, C, bm_sfc, bn_sfc, bk_sfc);
    return sfc_config;
  }

  // Check if SFC CA GEMM can be used for this configuration
  bool can_use_sfc(long BS) const {
    if (USE_SFC_CA_GEMM == 0)
      return false;
    if (rem != 0)
      return false;  // Need clean batch blocking for SFC

    // Need BS divisible by BSb (now 32 when SFC enabled)
    if (BS % BSb != 0)
      return false;
    if (K % Hk != 0)
      return false;
    if (C % Hc != 0)
      return false;

    // Need at least 2x2 blocks for SFC to be beneficial
    if (BS / BSb < 2 || K / Hk < 2)
      return false;

    return true;
  }

  // Step function for fallback to standard GEMM
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
    auto wt_V = GetVLAPtr<Tw>(t_wt_V, {Nc, Hc * Hk});
    auto func = [&, in, wt_V, bias, out, BS, with_bias](
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
        if (!(nc + Ncb < Nc)) {
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
        if (!(nc + Ncb < Nc)) {
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

    // Record GEMM-level bytes for bandwidth reporting
    record_gemm_bytes((long)BS * C * sizeof(T) + (long)C * K * sizeof(Tw) + (long)BS * K * sizeof(Tout));

    // Try to use SFC CA GEMM
    if (can_use_sfc(BS) && !t_wt_V.is_quantized()) {
      auto cfg = get_sfc_config(BS);
      if (cfg) {
        auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
        auto bias_ptr = GetVLAPtr<Tbias>(t_bias, {Hk});
        auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});
        auto wt_V = GetVLAPtr<Tw>(t_wt_V, {Nc, Hc * Hk});
        bool with_bias = (t_bias.numel() > 0);

        long Mb_sfc = BS / BSb;
        long Nb_sfc = Nk;  // K / Hk

        // Get K_layers parameters from the config (predicted by performance model)
        long K_layers = cfg->K_layers;
        long brcount = cfg->brcount;
        long Kb_per_layer = cfg->Kb_per_layer;
        long Kb_last_layer = cfg->Kb_last_layer;
        long K_rounds_per_layer = cfg->K_rounds_per_layer;

        // Use SFC index map from config (cached globally per Mb x Nb combo).
        unsigned char* sfc_map = cfg->sfc_index_map;
        unsigned int idx_tsize = cfg->index_tsize;

        // Global scratch buffer (allocated once, reused across all GEMMs).
        // Layer 0 writes directly to output. Layers 1..K_layers-1 write
        // to scratch_C[i_k_layer - 1], each of size BS * K elements.
        Tout** scratch_C = nullptr;
        std::vector<Tout*> scratch_layers;
        if (K_layers > 1) {
          char* scratch_base = (char*)get_sfc_ca_gemm_scratch();
          scratch_layers.resize(K_layers - 1, nullptr);
          for (long i = 0; i < K_layers - 1; i++) {
            scratch_layers[i] = (Tout*)(scratch_base +
                (size_t)i * (size_t)BS * (size_t)K * sizeof(Tout));
          }
          scratch_C = scratch_layers.data();
        }

        {
          RECORD_OMP_TIME();
#pragma omp parallel
          {
            brgemm_tpp.config();

            // Outer loop: walk through K-blocks within a single layer
            for (long i_k = 0; i_k < Kb_per_layer; i_k += brcount) {
              // Inner loop: fused (M-block, N-block, K-layer) using SFC ordering
              // for the M x N part, with all K_layers in lockstep
#pragma omp for nowait
              for (long i_sfc = 0; i_sfc < Mb_sfc * Nb_sfc * K_layers; i_sfc++) {
                int i_mb, i_nk;
                long i_k_layer = i_sfc / (Mb_sfc * Nb_sfc);
                sfc_ca_gemm_extract_indices_from_sfc(
                    &i_mb, &i_nk, sfc_map, i_sfc % (Mb_sfc * Nb_sfc), idx_tsize);

                // Compute brcount for this iteration (handle boundary)
                long brcount_use = brcount;
                if (i_k_layer == K_layers - 1) {
                  if (i_k + brcount > Kb_last_layer)
                    brcount_use = Kb_last_layer - i_k;
                } else {
                  if (i_k + brcount > Kb_per_layer)
                    brcount_use = Kb_per_layer - i_k;
                }
                if (brcount_use <= 0)
                  continue;

                long s1 = i_mb * BSb;
                long nk = i_nk;
                // nc = offset within the K dimension for this K-layer + i_k
                long nc = i_k + i_k_layer * Kb_per_layer;

                // Determine C target: layer 0 -> output, layers 1+ -> scratch
                Tout* c_ptr;
                if (i_k_layer == 0) {
                  c_ptr = out[s1][nk];
                } else {
                  c_ptr = scratch_C[i_k_layer - 1] +
                      (long)s1 * K + (long)nk * Hk;
                }

                // Zero/bias on first K iteration within this layer
                if (i_k == 0) {
                  if (i_k_layer == 0 && with_bias) {
                    this->copy_bias_tpp(bias_ptr[nk], c_ptr);
                  } else {
                    this->zero_tpp(c_ptr);
                  }
                }

                brgemm_tpp(in[s1][nc], wt_V[nk][nc], c_ptr, brcount_use, true);

                // For K_layers == 1: fuse postOp inline after the last BRGEMM
                if (K_layers == 1 && !(i_k + brcount < Kb_per_layer)) {
                  if (postOpCBs[0])
                    postOpCBs[0](out, s1, nk);
                }
              }
            }

            brgemm_tpp.release();
          }
        }

        // K_layers > 1: reduce partial outputs and fuse postOps in same loop
        if (K_layers > 1) {
          auto reduce_kernel = cfg->l_reduce_kernel;
          auto add_kernel = cfg->l_add_kernel;
          auto add_kernel_row = cfg->l_add_kernel_row;

#pragma omp parallel for
          for (long i_red = 0; i_red < Mb_sfc * Nb_sfc; i_red++) {
            int i_mb, i_nk;
            sfc_ca_gemm_extract_indices_from_sfc(
                &i_mb, &i_nk, sfc_map, i_red, idx_tsize);

            long s1 = i_mb * BSb;
            long nk = i_nk;

            if (K_layers == 2) {
              libxsmm_meltw_binary_param add_param;
              memset(&add_param, 0, sizeof(add_param));
              add_param.in0.primary = (void*)(
                  scratch_C[0] + (long)s1 * K + (long)nk * Hk);
              add_param.in1.primary = (void*)(out[s1][nk]);
              add_param.out.primary = (void*)(out[s1][nk]);
              add_kernel(&add_param);
            } else {
              // K_layers > 2: per-row reduce + add
              Tout reduce_scratch_buf[Hk];
              for (long l_row = 0; l_row < BSb; l_row++) {
                libxsmm_meltw_unary_param reduce_param;
                libxsmm_meltw_binary_param add_param;
                memset(&reduce_param, 0, sizeof(reduce_param));
                memset(&add_param, 0, sizeof(add_param));

                reduce_param.in.primary = (void*)(
                    scratch_C[0] + (long)(s1 + l_row) * K + (long)nk * Hk);
                reduce_param.out.primary = (void*)reduce_scratch_buf;
                reduce_kernel(&reduce_param);

                Tout* out_row = out[s1 + l_row][nk];
                add_param.in0.primary = (void*)reduce_scratch_buf;
                add_param.in1.primary = (void*)out_row;
                add_param.out.primary = (void*)out_row;
                add_kernel_row(&add_param);
              }
            }

            // Fuse postOp right after reduction for this (s1, nk) block
            if (postOpCBs[0])
              postOpCBs[0](out, s1, nk);
          }
        }

        return;
      }
    }

    // Fallback: standard blocked GEMM
    auto func = stepFunc(t_in, t_wt_V, t_bias, t_out, BS);
    {
      RECORD_OMP_TIME();
      auto gemm_loop = ThreadedLoop<3>(
          {LoopSpecs{0, Nc, Ncb, false},
           LoopSpecs{0L, BS, BSb},
           LoopSpecs{Nk}},
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
      std::vector<SfcCaBlockedLinearW<T, Tw, Tout>>& gemms,
      at::Tensor& t_in,
      std::vector<at::Tensor>& t_wt_V,
      std::vector<at::Tensor>& t_bias,
      std::vector<at::Tensor>& t_out) {
    int n_gemms = gemms.size();

    // For fused QKV, each GEMM may have different output sizes.
    // We run them sequentially with SFC ordering if possible,
    // or fall back to the standard fused approach.
    auto BS = t_in.numel() / gemms[0].C;

    // Check if all gemms can use SFC
    bool all_sfc = true;
    for (int i = 0; i < n_gemms; i++) {
      if (!gemms[i].can_use_sfc(BS) || t_wt_V[i].is_quantized()) {
        all_sfc = false;
        break;
      }
    }

    if (all_sfc) {
      // Run each GEMM individually with SFC ordering
      for (int i = 0; i < n_gemms; i++) {
        gemms[i](t_in, t_wt_V[i], t_bias[i], t_out[i]);
      }
    } else {
      // Fallback: fused approach like TppBlockedLinearW
      long totalN = 0;
      long Nc = gemms[0].Nc;
      long Ncb = gemms[0].Ncb;
      long BSb = gemms[0].BSb;
      auto loop_scheme = gemms[0].loop_scheme;
      std::vector<std::function<void(int, int, int)>> funcs;
      for (int i = 0; i < n_gemms; i++) {
        auto& g = gemms[i];
        funcs.push_back(
            g.stepFunc(t_in, t_wt_V[i], t_bias[i], t_out[i], BS));
        totalN += g.Nk;
        TPP_ASSERT(
            g.Nc == Nc && g.Ncb == Ncb && g.BSb == BSb,
            "Fused QKV weight block mismatch\n");
      }
      // Record GEMM-level bytes for all fused GEMMs
      {
        long fused_bytes = (long)BS * gemms[0].C * sizeof(T);
        for (int i = 0; i < n_gemms; i++) {
          fused_bytes += (long)gemms[i].C * gemms[i].K * sizeof(Tw);
          fused_bytes += (long)BS * gemms[i].K * sizeof(Tout);
        }
        record_gemm_bytes(fused_bytes);
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
  }

  static SfcCaBlockedLinearW<T, Tw, Tout> get(
      at::Tensor& t_in,
      at::Tensor& t_wt,
      at::Tensor& t_bias) {
    return Base::template _get<SfcCaBlockedLinearW<T, Tw, Tout>>(
        t_in, t_wt, t_bias);
  }
};

#endif  // SFC_CA_GEMM_TPP_H
