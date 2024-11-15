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
#include <torch/csrc/distributed/c10d/comm.hpp>
#include "fused_gemm.h"
#include "qtypes.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "shm_coll.h"
#include "tensor_helper.h"

#define S_FIRST_KVC
#define PER_THREAD_COPY

static long get_batch_dim_in_kv_cache() {
#ifdef S_FIRST_KVC
  return 1;
#else
  return 0;
#endif
}

static int my_rank = guess_mpi_rank();
static int my_size = 1;
static int TPP_CACHE_REMAPPED_WEIGHTS =
    env2int("TPP_CACHE_REMAPPED_WEIGHTS", 1);
static int FUSED_QKV_GEMM = env2int("FUSED_QKV_GEMM", 2);
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int SK_BLOCK_SIZE = env2int("SK_BLOCK_SIZE", 128);
static int KV_CACHE_INC_SIZE = env2int("KV_CACHE_INC_SIZE", 128);
static int USE_SHM_ALLREDUCE = env2int("USE_SHM_ALLREDUCE", -1);
static const int USE_MXFP4 = env2int("USE_MXFP4", 0);
static const int USE_MXFP4_FT = env2int("USE_MXFP4_FT", 0);
static const int USE_QINT8 = env2int("USE_QINT8", 0);
static const int USE_QINT8_FT = env2int("USE_QINT8_FT", 0);
static const bool USE_FLASH = env2int("USE_FLASH", 1);

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(fqkv_gemm, "fqkv_gemm");
REGISTER_LOCAL_SCOPE(proj_gemm, "proj_gemm");
REGISTER_LOCAL_SCOPE(mha, "mha");
REGISTER_LOCAL_SCOPE(ac_gemm1, "ac_gemm1");
REGISTER_LOCAL_SCOPE(ac_gemm2, "ac_gemm2");
REGISTER_LOCAL_SCOPE(ac_gemm21, "ac_gemm21");
REGISTER_LOCAL_SCOPE(ac_gemm22, "ac_gemm22");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");
REGISTER_LOCAL_SCOPE(lnorm, "lnorm");
REGISTER_LOCAL_SCOPE(rotary, "rotary");
REGISTER_LOCAL_SCOPE(reorder, "rorder");
REGISTER_LOCAL_SCOPE(allred, "allred");
REGISTER_LOCAL_SCOPE(barrier, "barrier");
REGISTER_LOCAL_SCOPE(concat, "concat");
REGISTER_LOCAL_SCOPE(fftkn, "fftkn");
REGISTER_LOCAL_SCOPE(k_trans, "k_trans");
REGISTER_LOCAL_SCOPE(pt_op, "pt_op");

static c10::intrusive_ptr<c10d::ProcessGroup> process_group;

void set_pg(c10::intrusive_ptr<c10d::ProcessGroup> process_group_) {
  process_group = process_group_;
  my_size = process_group->getSize();
  my_rank = process_group->getRank();
  if (my_size == 1)
    return;
  auto my_local_size = env2int("MPI_LOCALNRANKS", 0);
  if (my_local_size > 0 && my_local_size == my_size) {
    if (USE_SHM_ALLREDUCE == -1) {
      USE_SHM_ALLREDUCE = 1;
      // if (my_rank == 0) printf("Single node distributed run detected,
      // enabling SHM_ALLREDUCE\n");
    } else if (USE_SHM_ALLREDUCE == 0) {
      // if (my_rank == 0) printf("Single node distributed run detected, but
      // USE_SHM_ALLREDUCE set to 0, so not using SHM_ALLREDUCE\n");
    } else {
      USE_SHM_ALLREDUCE = 1;
      // if (my_rank == 0) printf("Single node distributed run detected, using
      // SHM_ALLREDUCE\n");
    }
  } else if (
      my_local_size > 0 &&
      (USE_SHM_ALLREDUCE != -1 || USE_SHM_ALLREDUCE != 0)) {
    USE_SHM_ALLREDUCE = 0;
    // if (my_rank == 0) printf("Multi-node distributed run detected, disabling
    // SHM_ALLREDUCE\n");
  }
  printf(
      "Setting PG: my_size = %d  my_rank = %d  my_local_size = %d SHM_ALLREDUCE = %d\n",
      my_size,
      my_rank,
      my_local_size,
      USE_SHM_ALLREDUCE);
}

static inline void allreduce(at::Tensor t_in) {
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
  if (USE_SHM_ALLREDUCE != 1) {
    std::vector<at::Tensor> temp_vec = {t_in};
    process_group->allreduce(temp_vec)->wait();
  } else {
    shm_allreduce(t_in, process_group);
  }
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
      (int)split_sizes.size() == my_size,
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
  auto B_pos = t_pos.size(0);
  auto S = in_sizes[1];
  auto COFF = HR / 2;

  auto in = GetVLAPtr<T>(t_in, {S, N, H}); // [B][S][N][H]
  auto emb_pos = GetVLAPtr<float>(t_emb_pos, {HR}); // [MP][HR]
  auto pos = GetVLAPtr<long>(t_pos, {S}); // [MB][S]
  TPP_ASSERT(B_pos == 1 || B_pos == B, "position_ids shape not compatible\n");
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
            int b_pos = (B_pos == 1 ? 0 : b);
            int p = pos[b_pos][s];
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
  auto B_pos = t_pos.size(0);
  auto COFF = HR / 2;

  auto in = GetVLAPtr<T>(t_in, {S, N, H}); // [B][S][N][H]
  auto emb_pos = GetVLAPtr<float>(t_emb_pos, {MP, HR}); // [MP][HR]
  auto pos = GetVLAPtr<long>(t_pos, {S}); // [MB][S]
  TPP_ASSERT(B_pos == 1 || B_pos == B, "position_ids shape not compatible\n");
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
            int b_pos = (B_pos == 1 ? 0 : b);
            int p = pos[b_pos][s];
            if (p >= MP)
              continue;
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
  auto K = t_in.size(-1);
  auto BS = t_in.numel() / K;

  auto in = GetVLAPtr<T>(t_in, {K});
  auto gamma = GetVLAPtr<LT>(t_gamma);
  auto beta = GetVLAPtr<LT>(t_beta);
  auto out = GetVLAPtr<T>(t_out, {K});

  // auto layer_norm_fwd_tpp =
  //    SCOPEIT((LayerNormFwdTPP<T, LT>(1, 1, K, eps)), LAYER_NORM);
  auto layer_norm_fwd_tpp = LayerNormFwdTPP<T, LT>(1, 1, K, eps);

  {
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int b = 0; b < BS; b++) {
      layer_norm_fwd_tpp(in[b], gamma, beta, nullptr, nullptr, out[b]);
    }
  }
}

template <typename T>
inline at::Tensor lyr_norm(
    at::Tensor t_in,
    at::Tensor t_gamma,
    at::Tensor t_beta,
    float eps) {
  auto t_out = at::empty_like(t_in);
  auto ldt = t_gamma.dtype();
  if (ldt == t_in.dtype()) {
    lyr_norm<T, T>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (ldt == at::kFloat) {
    lyr_norm<T, float>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (ldt == at::kBFloat16) {
    lyr_norm<T, bfloat16>(t_in, t_gamma, t_beta, t_out, eps);
  } else if (ldt == at::kHalf) {
    lyr_norm<T, half>(t_in, t_gamma, t_beta, t_out, eps);
  } else {
    std::cout << "gamma dtype: " << ldt << std::endl;
    TPP_ASSERT(false, "lyr_norm: unsupported gamma dtype\n");
  }

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

template <typename T>
inline at::Tensor llama_rms_norm(at::Tensor t_in, at::Tensor t_wt, float eps) {
  // RECORD_SCOPE(lnorm, {t_in, t_wt});

  // auto orig_dt = t_in.dtype();
  // auto t_var = t_in.to(at::kFloat).pow(2).mean(-1, true);
  // auto t_tmp = t_in * at::rsqrt(t_var + eps);
  // auto ret = t_wt * t_tmp;
  // ret = ret.to(orig_dt);
  // return ret;

  auto t_out = at::empty_like(t_in);
  auto ldt = t_wt.dtype();
  if (ldt == t_in.dtype()) {
    rms_norm<T, T>(t_in, t_wt, t_out, eps);
  } else if (ldt == at::kFloat) {
    rms_norm<T, float>(t_in, t_wt, t_out, eps);
  } else if (ldt == at::kBFloat16) {
    rms_norm<T, bfloat16>(t_in, t_wt, t_out, eps);
  } else if (ldt == at::kHalf) {
    rms_norm<T, half>(t_in, t_wt, t_out, eps);
  } else {
    std::cout << "gamma dtype: " << ldt << std::endl;
    TPP_ASSERT(false, "llama_rms_norm: unsupported gamma dtype\n");
  }

  return t_out;
}

template <typename T>
inline at::Tensor wt_tensor_for_first_token(at::Tensor t) {
  RECORD_SCOPE(fftkn, {t});
  auto dim = t.dim();

  if (dim < 5) {
    if (USE_QINT8_FT == 1)
      return remap_and_quantize_qint8(t);
    else if (USE_MXFP4_FT == 1)
      return remap_and_quantize_mxfp4(t);
    else
      return t;
  }
  auto sizes = t.sizes();
  auto K1 = sizes[0];
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = sizes[4];
  if (K2 >= 64)
    return t;
  long RBS = 4;
  if (64 % K2 == 0)
    RBS = 64 / K2;
  if (K1 % RBS != 0)
    return t;
#if 0
  auto t_new = t.view({K1/RBS, RBS, C1, C2, K2, C3}).permute({0, 2, 3, 1, 4, 5}).contiguous().view({K1/RBS, C1, C2, RBS*K2, C3});
#else
  auto t_new = t.new_empty({K1 / RBS, C1, C2, RBS * K2, C3});
  auto in = GetVLAPtr<T>(t, {RBS, C1, C2, K2 * C3});
  auto out = GetVLAPtr<T>(t_new, {C1, C2, RBS, K2 * C3});

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
#endif
  if (USE_QINT8_FT == 1)
    return remap_and_quantize_qint8(t_new);
  else if (USE_MXFP4_FT == 1)
    return remap_and_quantize_mxfp4(t_new);
  else
    return t_new;
}

template <typename GemmT>
inline std::vector<at::Tensor> fused_qkv_gemm_spl(
    at::Tensor t_in,
    std::vector<at::Tensor> t_wt,
    std::vector<at::Tensor> t_bias) {
  RECORD_SCOPE(fqkv_gemm, {t_in, t_wt[0]});
  t_in = t_in.contiguous();
  int n_gemms = t_wt.size();
  std::vector<at::Tensor> t_out;
  std::vector<GemmT> gemms;
  for (int i = 0; i < n_gemms; i++) {
    gemms.push_back(GemmT::get(t_in, t_wt[i], t_bias[i]));
    t_out.push_back(gemms[i].new_empty(t_in));
  }
  if (n_gemms == 4) {
    GeluPostOp()(gemms[3]);
  }

  GemmT::fused_gemm(gemms, t_in, t_wt, t_bias, t_out);
  return t_out;
}

template <typename Tin, typename Tout = Tin>
inline std::vector<at::Tensor> fused_qkv_gemm(
    at::Tensor t_in,
    std::vector<at::Tensor> t_wts,
    std::vector<at::Tensor> t_bias) {
  auto& t_wt = t_wts[0];
  // Check and redispatch with specialized type
  if (t_wt.is_quantized()) {
    if (t_wt.qscheme() == at::kPerBlockMxFP) {
      if (t_wt.dtype() == at::kQUInt4x2) {
        if (t_wt.size(-1) == 2) {
          return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, uint8_t, Tout>>(
              t_in, t_wts, t_bias);
        } else {
          return fused_qkv_gemm_spl<TppBlockedLinearW<int8_t, uint8_t, Tout>>(
              t_in, t_wts, t_bias);
        }
      } else {
        TPP_ASSERT(false, "Unsupported qdtype\n");
      }
    } else if (t_wt.qscheme() == at::kPerBlockAffine) {
      if (t_wt.dtype() == at::kQInt8) {
        return fused_qkv_gemm_spl<TppBlockedQInt8LinearW<Tin, uint8_t, Tout>>(
            t_in, t_wts, t_bias);
      } else {
        TPP_ASSERT(false, "Unsupported qdtype\n");
      }
    } else {
      TPP_ASSERT(false, "Unsupported qscheme 3\n");
    }
  } else if (t_wt.is_sparse_csr()) {
    TPP_ASSERT(false, "Sparse Tensor Types not supported yet\n");
  } else {
    auto dtype = t_wt.scalar_type();
    switch (dtype) {
      case at::kFloat:
        return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, float, Tout>>(
            t_in, t_wts, t_bias);
        break;
      case at::kBFloat16:
        return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, bfloat16, Tout>>(
            t_in, t_wts, t_bias);
        break;
      case at::kHalf:
        return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, half, Tout>>(
            t_in, t_wts, t_bias);
      case at::kHFloat8:
        return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, hfloat8, Tout>>(
            t_in, t_wts, t_bias);
        break;
      case at::kBFloat8:
        return fused_qkv_gemm_spl<TppBlockedLinearW<Tin, bfloat8, Tout>>(
            t_in, t_wts, t_bias);
        break;
      default:
        TPP_ASSERT(false, "Unsupported dtype\n");
    }
  }

  return {at::Tensor()};
}

template <typename T, typename Tv>
struct AttnKernels {
  SCOPEIT_DECL(BrgemmTPP<T, float>) a_gemm_tpp;
  SCOPEIT_DECL(ScaleTPP<float, float>) scale_tpp;
  SCOPEIT_DECL(AddBiasTPP<T>) add_mask_tpp;
  SCOPEIT_DECL(AddTPP<T, float, float>) add_2dmask_tpp;
  SCOPEIT_REF_DECL(VarSoftMaxFwdTPP<float, Tv>) softmax_fwd_tpp;
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
        SCOPEIT_REF((VarSoftMaxFwdTPP<float, Tv>(Sqb, Skb, USE_FLASH)), SOFTMAX);
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
        float max = _mm512_reduce_max_ps(_mm512_updown_convert(vmax, MAX_SUB_DTYPE, 0));
        vmax = _mm512_set1_ps(max);
        __m512 vsum = _mm512_setzero_ps();
        __attribute__((aligned(64))) float expm[16];
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[sk1][b][nq];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            __m512 vs = _mm512_sub_ps(_mm512_updown_convert(_mm512_loadu_ps_auto(ASP + sk2), MAX_SUB_DTYPE, 0), vmax);

            _mm512_store_ps(expm, vs);

            for(int e=0; e<16; e++)
              expm[e] = expf(expm[e]);
            __m512 vz = _mm512_load_ps(expm);

            vz = _mm512_updown_convert(vz, EXP_DTYPE, 0);

            _mm512_storeu_ps(ASP + sk2, vz);
            vsum = _mm512_updown_convert(_mm512_add_ps(vsum, vz), ACC_DTYPE, 0);
          }
        }
        float sum = _mm512_reduce_add_ps(_mm512_updown_convert(vsum, ACC_DTYPE, 0));
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
        float max = _mm512_reduce_max_ps(_mm512_updown_convert(vmax, MAX_SUB_DTYPE, 0));
        vmax = _mm512_set1_ps(max);
        __m512 vsum = _mm512_setzero_ps();
        __attribute__((aligned(64))) float expm[16];
        for (int sk1 = 0; sk1 < nbFSk; sk1++) {
          float* ASP = AS[b][nq][sk1];
          for (int sk2 = 0; sk2 < FSk_BS; sk2 += 16) {
            __m512 vs = _mm512_sub_ps(_mm512_updown_convert(_mm512_loadu_ps_auto(ASP + sk2), MAX_SUB_DTYPE, 0), vmax);

            _mm512_store_ps(expm, vs);

            for(int e=0; e<16; e++)
              expm[e] = expf(expm[e]);
            __m512 vz = _mm512_load_ps(expm);

            vz = _mm512_updown_convert(vz, EXP_DTYPE, 0);

            _mm512_storeu_ps(ASP + sk2, vz);
            vsum = _mm512_updown_convert(_mm512_add_ps(vsum, vz), ACC_DTYPE, 0);
          }
        }
        float sum = _mm512_reduce_add_ps(_mm512_updown_convert(vsum, ACC_DTYPE, 0));
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

at::Tensor remap_indices(at::Tensor t_beam_idx, at::Tensor t_offset) {
  long B = t_beam_idx.size(1);
  long offset = t_offset.item<long>();
  auto t_new_beam_idx = t_beam_idx.new_empty({B, offset + 1});
  auto beam_idx = GetVLAPtr<long>(t_new_beam_idx, {offset + 1});
  auto b_ptr = GetVLAPtr<long>(t_beam_idx, {B});
  for (auto i = 0; i < B; i++) {
    beam_idx[i][offset] = i;
    beam_idx[i][offset - 1] = b_ptr[offset - 1][i];
    for (auto j = offset - 2; j >= 0;
         j--) { // for the token of input, the target beam is alwarys 0
      beam_idx[i][j] = b_ptr[j][beam_idx[i][j + 1]];
    }
  }
  return t_new_beam_idx;
}

struct __attribute__((visibility("hidden"))) LLMBlock
    : torch::CustomClassHolder {
 public:
  std::string name;
  long H;

  LLMBlock(std::string name, long H) : name(name), H(H) {}

  bool check_weight_reuse(at::Tensor& t_in) {
    long BS = t_in.numel() / t_in.size(-1);
    if (BS >= FT_OPT_SIZE) {
      return true;
    }
    return false;
  }

  virtual std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) = 0;

  template <typename cls>
  std::vector<at::Tensor> forward_common(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    GlobalPass _gp(FWD);
    RECORD_FUNCTION(name, std::vector<c10::IValue>());
    std::vector<at::Tensor> ret;
    auto self = static_cast<cls*>(this);
    caffe2::TypeMeta dt_in = t_inp[0].dtype();

    if (dt_in == at::kFloat) {
      ret = self->template _forward<float>(t_inp, t_cache, use_cache);
    } else if (dt_in == at::kBFloat16) {
      ret = self->template _forward<bfloat16>(t_inp, t_cache, use_cache);
    } else if (dt_in == at::kHalf) {
      ret = self->template _forward<half>(t_inp, t_cache, use_cache);
#ifdef PYTORCH_SUPPORTS_FLOAT8
    } else if (dt_in == at::kBFloat8) {
      ret = self->template _forward<bfloat8>(t_inp, t_cache, use_cache);
    } else if (dt_in == at::kHFloat8) {
      ret = self->template _forward<hfloat8>(t_inp, t_cache, use_cache);
#endif
    } else {
      std::cout << "Input Type: " << dt_in << std::endl;
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
    auto t_dummy = t_KL.new_empty({0});
    auto t_dummy_int = t_KL.new_empty({0}, at::kLong);
    auto t_key_past = t_dummy;
    auto t_value_past = t_dummy;
    auto t_beam_idx = t_dummy_int;
    auto t_offset = t_dummy_int;
    auto B = t_QL.size(0);
    auto S = t_QL.size(1);
    // auto N = self->N;
    auto H = this->H;
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
          csz >= 6,
          "Updated indirect kv_cache tuple should be of minimum length 6\n");
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
      // if (my_rank == 0) std::cout << "t_beam_idx: " << t_beam_idx.sizes()
      // << std::endl;
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

      // if (my_rank == 0) std::cout << "t_beam_idx2: " <<
      // t_beam_idx.sizes() << std::endl; std::cout << "t_key_past.shape:"
      // << t_key_past.sizes() << std::endl; std::cout <<
      // "t_beam_idx.shape:" << t_beam_idx.sizes() << std::endl; std::cout
      // << "t_offset:" << t_offset << std::endl; std::cout
      // << "B: " << B << " offset:" << offset << std::endl;

      at::Tensor t_new_beam_idx;
      if (csz > 6) {
        t_new_beam_idx = t_cache[6];
      } else {
        t_new_beam_idx = t_beam_idx.new_empty({B, offset + 1});
      }
      auto beam_idx = GetVLAPtr<long>(t_new_beam_idx, {offset + 1});
      if (csz <= 6) {
        auto b_ptr = GetVLAPtr<long>(t_beam_idx, {B});
        for (auto i = 0; i < B; i++) {
          beam_idx[i][offset] = i;
          beam_idx[i][offset - 1] = b_ptr[offset - 1][i];
          for (auto j = offset - 2; j >= 0;
               j--) { // for the token of input, the target beam is alwarys 0
            beam_idx[i][j] = b_ptr[j][beam_idx[i][j + 1]];
          }
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

void print_diff(const std::string& tag, at::Tensor a, at::Tensor b) {
  a = a.flatten().to(at::kFloat);
  b = b.flatten().to(at::kFloat);
  long N = a.numel();
  auto ap = a.data_ptr<float>();
  auto bp = b.data_ptr<float>();
  float max_diff = 0;
  std::cout << tag << "  sz: " << N << std::endl;
  for (long i = 0; i < N; i++) {
    auto diff = fabsf(ap[i] - bp[i]);
    if (diff > max_diff) {
      printf("%5ld: r: %12.8g  q: %12.8g  d: %12.8g\n", i, ap[i], bp[i], diff);
      max_diff = diff;
    }
  }
  std::cout << tag << "  end " << std::endl;
}

struct __attribute__((visibility("hidden"))) GPTJBlock : LLMBlock {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wi, t_Wo;
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G, t_B;
  at::Tensor t_EP; // embed_positions
  // cached for first token
  at::Tensor t_Wq_1, t_Wk_1, t_Wv_1, t_Wp_1;
  at::Tensor t_Wi_1, t_Wo_1;
  bool first_token_remapped = false;
  float eps;
  long N, H;
  long max_positions, rotary_dim;

  GPTJBlock(
      std::vector<at::Tensor> params,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("gptj_fwd", H),
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

    if (USE_MXFP4) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_mxfp4(t_Wq);
      t_Wk = remap_and_quantize_mxfp4(t_Wk);
      t_Wv = remap_and_quantize_mxfp4(t_Wv);
      t_Wp = remap_and_quantize_mxfp4(t_Wp);
      t_Wi = remap_and_quantize_mxfp4(t_Wi);
      t_Wo = remap_and_quantize_mxfp4(t_Wo);
    } else if (USE_QINT8) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_qint8(t_Wq);
      t_Wk = remap_and_quantize_qint8(t_Wk);
      t_Wv = remap_and_quantize_qint8(t_Wv);
      t_Wp = remap_and_quantize_qint8(t_Wp);
      t_Wi = remap_and_quantize_qint8(t_Wi);
      t_Wo = remap_and_quantize_qint8(t_Wo);
    }

    N = t_Wq.size(0) * t_Wq.size(3) / H;
    auto dt = t_Wq.dtype();
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " N=" << N << " H=" << H
                << " wt dt=" << dt << " 1st wt dt=" << t_Wq_1.dtype()
                << std::endl;
    }
  }

  template <typename Tw>
  void remap_for_first_token() {
    auto dtype = c10::CppTypeToScalarType<Tw>::value;
    t_Wq_1 = wt_tensor_for_first_token<Tw>(t_Wq.to(dtype));
    t_Wk_1 = wt_tensor_for_first_token<Tw>(t_Wk.to(dtype));
    t_Wv_1 = wt_tensor_for_first_token<Tw>(t_Wv.to(dtype));
    t_Wp_1 = wt_tensor_for_first_token<Tw>(t_Wp.to(dtype));
    t_Wi_1 = wt_tensor_for_first_token<Tw>(t_Wi.to(dtype));
    t_Wo_1 = wt_tensor_for_first_token<Tw>(t_Wo.to(dtype));
    first_token_remapped = true;
  }

  virtual std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) override {
    return this->template forward_common<GPTJBlock>(t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];
    auto t_pid = t_inp[2];

    bool weight_reuse = check_weight_reuse(t_HS);

    float scale = 1.0 / my_size;

    auto t_Wq = this->t_Wq;
    auto t_Wk = this->t_Wk;
    auto t_Wv = this->t_Wv;
    auto t_Wp = this->t_Wp;
    auto t_Wi = this->t_Wi;
    auto t_Wo = this->t_Wo;

    if (weight_reuse && TPP_CACHE_REMAPPED_WEIGHTS) {
      if (!first_token_remapped)
        remap_for_first_token<T>();

      t_Wq = this->t_Wq_1;
      t_Wk = this->t_Wk_1;
      t_Wv = this->t_Wv_1;
      t_Wp = this->t_Wp_1;
      t_Wi = this->t_Wi_1;
      t_Wo = this->t_Wo_1;
    }

    auto t_null = t_HS.new_empty({0});
    auto t_res = t_HS;
    t_HS = lyr_norm<T>(t_HS, t_G, t_B, eps);
    auto qkv_gemm = GemmCaller<T, T>(SCOPE_ARG(qkv_gemm));
    auto proj_gemm = GemmCaller<T, T>(SCOPE_ARG(proj_gemm));
    auto i_gemm = GemmCaller<T>(SCOPE_ARG(i_gemm));
    auto o_gemm = GemmCaller<T>(SCOPE_ARG(o_gemm));

    if (FUSED_QKV_GEMM == 0) {
      auto t_QL = qkv_gemm(t_HS, t_Wq, t_null);
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);

      auto t_KL = qkv_gemm(t_HS, t_Wk, t_null);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);

      auto t_VL = qkv_gemm(t_HS, t_Wv, t_null);

      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

      auto t_CL = outputs[0];
      auto t_SO = proj_gemm(t_CL, t_Wp, t_null);
      auto t_I = i_gemm(GeluPostOp(), t_HS, t_Wi, t_Bi);
      auto t_Out = o_gemm(Add2ScalePostOp(t_SO, t_res, scale), t_I, t_Wo, t_Bo);
      if (my_size > 1) {
        allreduce(t_Out);
      }

      outputs[0] = t_Out;

      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    } else if (FUSED_QKV_GEMM == 1) {
      auto t_qkv_outs =
          fused_qkv_gemm<T>(t_HS, {t_Wq, t_Wk, t_Wv}, {t_null, t_null, t_null});
      auto t_QL = t_qkv_outs[0];
      auto t_KL = t_qkv_outs[1];
      auto t_VL = t_qkv_outs[2];
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);

      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

      auto t_CL = outputs[0];
      auto t_SO = proj_gemm(t_CL, t_Wp, t_null);
      auto t_I = i_gemm(GeluPostOp(), t_HS, t_Wi, t_Bi);
      auto t_Out = o_gemm(Add2ScalePostOp(t_SO, t_res, scale), t_I, t_Wo, t_Bo);
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
      auto t_qkv_outs = fused_qkv_gemm<T>(
          t_HS, {t_Wq, t_Wk, t_Wv, t_Wi}, {t_null, t_null, t_null, t_Bi});
      auto t_QL = t_qkv_outs[0];
      auto t_KL = t_qkv_outs[1];
      auto t_VL = t_qkv_outs[2];
      auto t_I = t_qkv_outs[3];
      apply_rotary_pos_emb_gptj<T>(t_QL, t_EP, t_pid, N, H);
      apply_rotary_pos_emb_gptj<T>(t_KL, t_EP, t_pid, N, H);

      auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

      auto t_CL = outputs[0];
      auto t_SO = proj_gemm(t_CL, t_Wp, t_null);
      auto t_Out = o_gemm(Add2ScalePostOp(t_SO, t_res, scale), t_I, t_Wo, t_Bo);
      if (my_size > 1) {
        allreduce(t_Out);
      }

      outputs[0] = t_Out;

      if (use_cache) {
        return outputs;
      } else {
        return {t_Out};
      }
    }
  }
};

struct __attribute__((visibility("hidden"))) OPTDecoderLayer : LLMBlock {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp; // wt and bias for attn
  at::Tensor t_Bq, t_Bk, t_Bv, t_Bp;
  at::Tensor t_Wi, t_Wo; // wt and bias for fc1 and fc2
  at::Tensor t_Bi, t_Bo;
  at::Tensor t_G1, t_B1; // Gamma and Beta for attention layernorm
  at::Tensor t_G2, t_B2; // Gamma and Beta for MLP layernorm
  at::Tensor t_Wq_1, t_Wk_1, t_Wv_1, t_Wp_1;
  at::Tensor t_Wi_1, t_Wo_1;
  bool first_token_remapped = false;
  float eps1, eps2;
  long N, H;
  bool do_layer_norm_before;

  OPTDecoderLayer(
      std::vector<at::Tensor> params,
      double eps1,
      double eps2,
      long H,
      bool do_layer_norm_before)
      : LLMBlock("opt_fwd", H),
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

    if (USE_MXFP4) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_mxfp4(t_Wq);
      t_Wk = remap_and_quantize_mxfp4(t_Wk);
      t_Wv = remap_and_quantize_mxfp4(t_Wv);
      t_Wp = remap_and_quantize_mxfp4(t_Wp);
      t_Wi = remap_and_quantize_mxfp4(t_Wi);
      t_Wo = remap_and_quantize_mxfp4(t_Wo);
    }

    else if (USE_QINT8) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_qint8(t_Wq);
      t_Wk = remap_and_quantize_qint8(t_Wk);
      t_Wv = remap_and_quantize_qint8(t_Wv);
      t_Wp = remap_and_quantize_qint8(t_Wp);
      t_Wi = remap_and_quantize_qint8(t_Wi);
      t_Wo = remap_and_quantize_qint8(t_Wo);
    }

    N = t_Wq.size(0) * t_Wq.size(3) / H;
    auto dt = t_Wq.dtype();
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " N=" << N << " H=" << H
                << " wt dt=" << dt

                << std::endl;
    }
  }

  template <typename Tw>
  void remap_for_first_token() {
    auto dtype = c10::CppTypeToScalarType<Tw>::value;
    t_Wq_1 = wt_tensor_for_first_token<Tw>(t_Wq.to(dtype));
    t_Wk_1 = wt_tensor_for_first_token<Tw>(t_Wk.to(dtype));
    t_Wv_1 = wt_tensor_for_first_token<Tw>(t_Wv.to(dtype));
    t_Wp_1 = wt_tensor_for_first_token<Tw>(t_Wp.to(dtype));
    t_Wi_1 = wt_tensor_for_first_token<Tw>(t_Wi.to(dtype));
    t_Wo_1 = wt_tensor_for_first_token<Tw>(t_Wo.to(dtype));
    first_token_remapped = true;
  }

  virtual std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) override {
    return this->template forward_common<OPTDecoderLayer>(
        t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];

    bool weight_reuse = check_weight_reuse(t_HS);

    float scale = 1.0 / my_size;

    auto t_Wq = this->t_Wq;
    auto t_Wk = this->t_Wk;
    auto t_Wv = this->t_Wv;
    auto t_Wp = this->t_Wp;
    auto t_Wi = this->t_Wi;
    auto t_Wo = this->t_Wo;

    if (weight_reuse && TPP_CACHE_REMAPPED_WEIGHTS) {
      if (!first_token_remapped)
        remap_for_first_token<T>();

      t_Wq = this->t_Wq_1;
      t_Wk = this->t_Wk_1;
      t_Wv = this->t_Wv_1;
      t_Wp = this->t_Wp_1;
      t_Wi = this->t_Wi_1;
      t_Wo = this->t_Wo_1;
    }

    auto t_null = t_HS.new_empty({0}); // at::Tensor().to(t_HS.dtype());

    auto t_res = t_HS;
    if (do_layer_norm_before) {
      t_HS = lyr_norm<T>(t_HS, t_G1, t_B1, eps1);
    }
    auto qkv_gemm = GemmCaller<T>(SCOPE_ARG(qkv_gemm));
    auto proj_gemm = GemmCaller<T>(SCOPE_ARG(proj_gemm));
    auto i_gemm = GemmCaller<T>(SCOPE_ARG(i_gemm));
    auto o_gemm = GemmCaller<T>(SCOPE_ARG(o_gemm));

    at::Tensor t_QL, t_KL, t_VL;
    if (FUSED_QKV_GEMM == 0) {
      t_QL = qkv_gemm(t_HS, t_Wq, t_Bq);
      t_KL = qkv_gemm(t_HS, t_Wk, t_Bk);
      t_VL = qkv_gemm(t_HS, t_Wv, t_Bv);
    } else {
      auto t_qkv_outs =
          fused_qkv_gemm<T>(t_HS, {t_Wq, t_Wk, t_Wv}, {t_Bq, t_Bk, t_Bv});
      t_QL = t_qkv_outs[0];
      t_KL = t_qkv_outs[1];
      t_VL = t_qkv_outs[2];
    }

    auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

    auto t_CL = outputs[0];
    t_HS = proj_gemm(AddScalePostOp(t_res, scale), t_CL, t_Wp, t_Bp);

    if (my_size > 1) {
      allreduce(t_HS);
    }

    if (!do_layer_norm_before) {
      t_HS = lyr_norm<T>(t_HS, t_G1, t_B1, eps1);
    }

    t_res = t_HS;

    if (do_layer_norm_before) {
      t_HS = lyr_norm<T>(t_HS, t_G2, t_B2, eps2);
    }

    t_HS = i_gemm(ReluPostOp(), t_HS, t_Wi, t_Bi);
    t_HS = o_gemm(AddScalePostOp(t_res, scale), t_HS, t_Wo, t_Bo);

    if (my_size > 1) {
      allreduce(t_HS);
    }

    if (!do_layer_norm_before) {
      t_HS = lyr_norm<T>(t_HS, t_G2, t_B2, eps2);
    }

    outputs[0] = t_HS;

    if (use_cache) {
      return outputs;
    } else {
      return {t_HS};
    }
  }
};

struct __attribute__((visibility("hidden"))) LlamaDecoderLayer : LLMBlock {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wg, t_Wu, t_Wd;
  at::Tensor t_Gi, t_Gpa;
  at::Tensor t_EP; // embed_positions
  at::Tensor t_Wq_1, t_Wk_1, t_Wv_1, t_Wp_1;
  at::Tensor t_Wg_1, t_Wu_1, t_Wd_1;
  bool first_token_remapped = false;
  float eps;
  long Nq, Nkv, H;
  long max_positions, rotary_dim;

  LlamaDecoderLayer(
      std::vector<at::Tensor> params,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("llama_fwd", H),
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

    if (USE_MXFP4) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_mxfp4(t_Wq);
      t_Wk = remap_and_quantize_mxfp4(t_Wk);
      t_Wv = remap_and_quantize_mxfp4(t_Wv);
      t_Wp = remap_and_quantize_mxfp4(t_Wp);
      t_Wg = remap_and_quantize_mxfp4(t_Wg);
      t_Wu = remap_and_quantize_mxfp4(t_Wu);
      t_Wd = remap_and_quantize_mxfp4(t_Wd);
    }

    else if (USE_QINT8) {
      if (t_Wq.dtype() == at::kBFloat16) {
        remap_for_first_token<bfloat16>();
      } else {
        remap_for_first_token<float>();
      }
      t_Wq = remap_and_quantize_qint8(t_Wq);
      t_Wk = remap_and_quantize_qint8(t_Wk);
      t_Wv = remap_and_quantize_qint8(t_Wv);
      t_Wp = remap_and_quantize_qint8(t_Wp);
      t_Wg = remap_and_quantize_qint8(t_Wg);
      t_Wu = remap_and_quantize_qint8(t_Wu);
      t_Wd = remap_and_quantize_qint8(t_Wd);
    }

    Nq = t_Wq.size(0) * t_Wq.size(3) / H;
    Nkv = t_Wk.size(0) * t_Wk.size(3) / H;
    auto dt = t_Wq.dtype();
    if (my_rank == 0) {
      std::cout << "my_size=" << my_size << " Nq=" << Nq << " Nkv=" << Nkv
                << " wt dt=" << dt << " H=" << H << std::endl;
    }
  }

  template <typename Tw>
  void remap_for_first_token() {
    auto dtype = c10::CppTypeToScalarType<Tw>::value;
    t_Wq_1 = wt_tensor_for_first_token<Tw>(t_Wq.to(dtype));
    t_Wk_1 = wt_tensor_for_first_token<Tw>(t_Wk.to(dtype));
    t_Wv_1 = wt_tensor_for_first_token<Tw>(t_Wv.to(dtype));
    t_Wp_1 = wt_tensor_for_first_token<Tw>(t_Wp.to(dtype));
    t_Wg_1 = wt_tensor_for_first_token<Tw>(t_Wg.to(dtype));
    t_Wu_1 = wt_tensor_for_first_token<Tw>(t_Wu.to(dtype));
    t_Wd_1 = wt_tensor_for_first_token<Tw>(t_Wd.to(dtype));
    first_token_remapped = true;
  }

  virtual std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      std::vector<at::Tensor> t_cache,
      bool use_cache) override {
    return this->template forward_common<LlamaDecoderLayer>(
        t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      std::vector<at::Tensor>& t_cache,
      bool use_cache) {
    auto t_HS = t_inp[0];
    RECORD_SCOPE(pt_op, {t_HS});
    auto t_am = t_inp[1];
    auto t_pid = t_inp[2];

    bool weight_reuse = check_weight_reuse(t_HS);

    float scale = 1.0 / my_size;

    auto t_Wq = this->t_Wq;
    auto t_Wk = this->t_Wk;
    auto t_Wv = this->t_Wv;
    auto t_Wp = this->t_Wp;
    auto t_Wg = this->t_Wg;
    auto t_Wu = this->t_Wu;
    auto t_Wd = this->t_Wd;

    if (weight_reuse && TPP_CACHE_REMAPPED_WEIGHTS) {
      if (!first_token_remapped)
        remap_for_first_token<T>();

      t_Wq = this->t_Wq_1;
      t_Wk = this->t_Wk_1;
      t_Wv = this->t_Wv_1;
      t_Wp = this->t_Wp_1;
      t_Wg = this->t_Wg_1;
      t_Wu = this->t_Wu_1;
      t_Wd = this->t_Wd_1;
    }

    auto t_null = t_HS.new_empty({0});
    auto t_res = t_HS;
    t_HS = llama_rms_norm<T>(t_HS, t_Gi, eps);

    auto qkv_gemm = GemmCaller<T>(SCOPE_ARG(qkv_gemm));
    auto proj_gemm = GemmCaller<T>(SCOPE_ARG(proj_gemm));
    auto i_gemm = GemmCaller<T>(SCOPE_ARG(i_gemm));
    auto o_gemm = GemmCaller<T>(SCOPE_ARG(o_gemm));

    at::Tensor t_QL, t_KL, t_VL;
    if (FUSED_QKV_GEMM == 0) {
      t_QL = qkv_gemm(t_HS, t_Wq, t_null);
      apply_rotary_pos_emb_llama<T>(t_QL, t_EP, t_pid, Nq, H);

      t_KL = qkv_gemm(t_HS, t_Wk, t_null);
      apply_rotary_pos_emb_llama<T>(t_KL, t_EP, t_pid, Nkv, H);

      t_VL = qkv_gemm(t_HS, t_Wv, t_null);
    } else {
      auto t_qkv_outs =
          fused_qkv_gemm<T>(t_HS, {t_Wq, t_Wk, t_Wv}, {t_null, t_null, t_null});
      t_QL = t_qkv_outs[0];
      t_KL = t_qkv_outs[1];
      t_VL = t_qkv_outs[2];
      apply_rotary_pos_emb_llama<T>(t_QL, t_EP, t_pid, Nq, H);
      apply_rotary_pos_emb_llama<T>(t_KL, t_EP, t_pid, Nkv, H);
    }

    auto outputs = self_mha<T>(t_QL, t_KL, t_VL, t_am, t_cache);

    auto t_CL = outputs[0];

    auto t_SO = proj_gemm(AddScalePostOp(t_res, scale), t_CL, t_Wp, t_null);

    if (my_size > 1) {
      allreduce(t_SO);
    }

    t_res = t_SO;

    t_HS = llama_rms_norm<T>(t_SO, t_Gpa, eps);

    auto t_I = i_gemm(SiluPostOp(), t_HS, t_Wg, t_null);
    t_I = i_gemm(MulPostOp(t_I), t_HS, t_Wu, t_null);
    auto t_Out = o_gemm(AddScalePostOp(t_res, scale), t_I, t_Wd, t_null);

    if (my_size > 1) {
      allreduce(t_Out);
    }

    outputs[0] = t_Out;

    if (use_cache) {
      return outputs;
    } else {
      return {t_Out};
    }
  }
};

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

  at::Tensor t_out;

  auto dt_in = t_in.dtype();
  if (dt_in == at::kQInt8) {
    if (t_bias.dtype().itemsize() > 0) {
      dt_in = t_bias.dtype();
    } else {
      dt_in = c10::scalarTypeToTypeMeta(at::kFloat);
    }
  }

  if (dt_in == at::kFloat) {
    typedef float T;
    t_out = fc_plain<T>(t_in, t_wt, t_bias);
  } else if (dt_in == at::kBFloat16) {
    typedef bfloat16 T;
    t_out = fc_plain<T>(t_in, t_wt, t_bias);
  } else if (dt_in == at::kHalf) {
    typedef half T;
    t_out = fc_plain<T>(t_in, t_wt, t_bias);
#ifdef PYTORCH_SUPPORTS_FLOAT8
  } else if (dt_in == at::kBFloat8) {
    typedef bfloat8 T;
    t_out = fc_plain<T>(t_in, t_wt, t_bias);
  } else if (dt_in == at::kHFloat8) {
    typedef hfloat8 T;
    t_out = fc_plain<T>(t_in, t_wt, t_bias);
#endif
  } else {
    std::cout << "dtypes: input: " << dt_in << std::endl;
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

REGISTER_SUBMODULE(_fused_llm_infer, m) {
  m.def("fc_plain", &fc_plain_wrap, "TPP fc_plain");
  m.def("set_pg", &set_pg);
  m.def("allreduce", &allreduce);
  m.def("remap_indices", &remap_indices);
  m.def("get_batch_dim_in_kv_cache", &get_batch_dim_in_kv_cache);
  py::class_<LLMBlock>(m, "LLMBlock").def("forward", &LLMBlock::forward);
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
  m.def("fc_plain", &fc_plain_wrap);
  m.def("set_pg", &set_pg);
  m.def("allreduce", &allreduce);
  m.def("remap_indices", &remap_indices);
  m.def("get_batch_dim_in_kv_cache", &get_batch_dim_in_kv_cache);
  m.class_<LLMBlock>("LLMBlock").def("forward", &LLMBlock::forward);
  m.class_<GPTJBlock>("GPTJBlock")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  m.class_<OPTDecoderLayer>("OPTDecoderLayer")
      .def(torch::init<std::vector<at::Tensor>, double, double, long, bool>())
      .def("forward", &OPTDecoderLayer::forward);
  m.class_<LlamaDecoderLayer>("LlamaDecoderLayer")
      .def(torch::init<std::vector<at::Tensor>, double, long, long, long>())
      .def("forward", &LlamaDecoderLayer::forward);
}
