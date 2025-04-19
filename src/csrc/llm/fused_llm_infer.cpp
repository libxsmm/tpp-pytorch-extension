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
#include "attn.h"
#include "shm_coll.h"
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();
static int my_size = 1;
static int TPP_CACHE_REMAPPED_WEIGHTS =
    env2int("TPP_CACHE_REMAPPED_WEIGHTS", 1);
static int FUSED_QKV_GEMM = env2int("FUSED_QKV_GEMM", 2);
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int USE_SHM_ALLREDUCE = env2int("USE_SHM_ALLREDUCE", -1);
static const int USE_MXFP4 = env2int("USE_MXFP4", 0);
static const int USE_MXFP4_FT = env2int("USE_MXFP4_FT", 0);
static const int USE_QINT8 = env2int("USE_QINT8", 0);
static const int USE_QINT8_FT = env2int("USE_QINT8_FT", 0);

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(fqkv_gemm, "fqkv_gemm");
REGISTER_LOCAL_SCOPE(proj_gemm, "proj_gemm");
REGISTER_LOCAL_SCOPE(mha, "mha");
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

namespace tpp_models {
struct __attribute__((visibility("hidden"))) LLMBlock
    : torch::CustomClassHolder {
 public:
  std::string name;
  long layer_idx;
  long H;

  LLMBlock(std::string name, long layer_idx, long H)
      : name(name), layer_idx(layer_idx), H(H) {}

  bool check_weight_reuse(at::Tensor& t_in) {
    long BS = t_in.numel() / t_in.size(-1);
    if (BS >= FT_OPT_SIZE) {
      return true;
    }
    return false;
  }

  virtual std::vector<at::Tensor> forward(
      std::vector<at::Tensor> t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
      bool use_cache) = 0;

  template <typename cls>
  std::vector<at::Tensor> forward_common(
      std::vector<at::Tensor>& t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
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
      c10::intrusive_ptr<TppCache> t_cache) {
    RECORD_SCOPE(mha, {t_QL, t_KL});
    auto q_sizes = t_QL.sizes();
    auto B = t_QL.size(0);
    auto S = t_QL.size(1);
    auto H = this->H;

    t_KL = t_KL.view({B, S, -1, H}).permute({0, 2, 1, 3}); // .contiguous();
    t_VL = t_VL.view({B, S, -1, H}).permute({0, 2, 1, 3}); // .contiguous();
    auto N = t_KL.size(1);
    t_QL = t_QL.view({B, S, N, -1, H}).permute({0, 2, 1, 3, 4}).contiguous();
    auto t_CL =
        attn_wrapper_impl<T>(t_QL, t_KL, t_VL, t_am, t_cache, layer_idx);
    t_CL = t_CL.permute({0, 2, 1, 3, 4}).contiguous().view(q_sizes);
    return {t_CL};
  }
};

#if 0
static void print_diff(const std::string& tag, at::Tensor a, at::Tensor b) {
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
#endif

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
      long layer_idx,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("gptj_fwd", layer_idx, H),
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

    if (t_Wq.is_quantized()) {
      t_Wq_1 = t_Wq;
      t_Wk_1 = t_Wk;
      t_Wv_1 = t_Wv;
      t_Wp_1 = t_Wp;
      t_Wi_1 = t_Wi;
      t_Wo_1 = t_Wo;
      first_token_remapped = true;
    } else {
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
      c10::intrusive_ptr<TppCache> t_cache,
      bool use_cache) override {
    return this->template forward_common<GPTJBlock>(t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
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
      long layer_idx,
      double eps1,
      double eps2,
      long H,
      bool do_layer_norm_before)
      : LLMBlock("opt_fwd", layer_idx, H),
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

    if (t_Wq.is_quantized()) {
      t_Wq_1 = t_Wq;
      t_Wk_1 = t_Wk;
      t_Wv_1 = t_Wv;
      t_Wp_1 = t_Wp;
      t_Wi_1 = t_Wi;
      t_Wo_1 = t_Wo;
      first_token_remapped = true;
    } else {
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
      c10::intrusive_ptr<TppCache> t_cache,
      bool use_cache) override {
    return this->template forward_common<OPTDecoderLayer>(
        t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
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
      long layer_idx,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("llama_fwd", layer_idx, H),
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

    if (t_Wq.is_quantized()) {
      t_Wq_1 = t_Wq;
      t_Wk_1 = t_Wk;
      t_Wv_1 = t_Wv;
      t_Wp_1 = t_Wp;
      t_Wg_1 = t_Wg;
      t_Wu_1 = t_Wu;
      t_Wd_1 = t_Wd;
      first_token_remapped = true;
    } else {
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
      c10::intrusive_ptr<TppCache> t_cache,
      bool use_cache) override {
    return this->template forward_common<LlamaDecoderLayer>(
        t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
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

struct __attribute__((visibility("hidden"))) Qwen2DecoderLayer : LLMBlock {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv, t_Wp;
  at::Tensor t_Wg, t_Wu, t_Wd;
  at::Tensor t_Gi, t_Gpa;
  at::Tensor t_Bq, t_Bk, t_Bv;
  at::Tensor t_EP; // embed_positions
  at::Tensor t_Wq_1, t_Wk_1, t_Wv_1, t_Wp_1;
  at::Tensor t_Wg_1, t_Wu_1, t_Wd_1;
  bool first_token_remapped = false;
  float eps;
  long Nq, Nkv, H;
  long max_positions, rotary_dim;

  Qwen2DecoderLayer(
      std::vector<at::Tensor> params,
      long layer_idx,
      double eps,
      long H,
      long max_positions,
      long rotary_dim)
      : LLMBlock("llama_fwd", layer_idx, H),
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

    t_Bq = params[i++]; // q_proj
    t_Bk = params[i++]; // k_proj
    t_Bv = params[i++]; // v_proj

    t_EP = params[i++]; // embed_positions

    if (t_Wq.is_quantized()) {
      t_Wq_1 = t_Wq;
      t_Wk_1 = t_Wk;
      t_Wv_1 = t_Wv;
      t_Wp_1 = t_Wp;
      t_Wg_1 = t_Wg;
      t_Wu_1 = t_Wu;
      t_Wd_1 = t_Wd;
      first_token_remapped = true;
    } else {
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
      c10::intrusive_ptr<TppCache> t_cache,
      bool use_cache) override {
    return this->template forward_common<Qwen2DecoderLayer>(
        t_inp, t_cache, use_cache);
  }

  template <typename T>
  std::vector<at::Tensor> _forward(
      std::vector<at::Tensor>& t_inp,
      c10::intrusive_ptr<TppCache> t_cache,
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
      t_QL = qkv_gemm(t_HS, t_Wq, t_Bq);
      apply_rotary_pos_emb_llama<T>(t_QL, t_EP, t_pid, Nq, H);

      t_KL = qkv_gemm(t_HS, t_Wk, t_Bk);
      apply_rotary_pos_emb_llama<T>(t_KL, t_EP, t_pid, Nkv, H);

      t_VL = qkv_gemm(t_HS, t_Wv, t_Bv);
    } else {
      auto t_qkv_outs =
          fused_qkv_gemm<T>(t_HS, {t_Wq, t_Wk, t_Wv}, {t_Bq, t_Bk, t_Bv});
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

} // namespace tpp_models
using namespace tpp_models;
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
  py::class_<LLMBlock>(m, "LLMBlock", py::module_local())
      .def("forward", &LLMBlock::forward);
  py::class_<GPTJBlock>(m, "GPTJBlock", py::module_local())
      .def(py::init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  py::class_<OPTDecoderLayer>(m, "OPTDecoderLayer", py::module_local())
      .def(
          py::init<std::vector<at::Tensor>, long, double, double, long, bool>())
      .def("forward", &OPTDecoderLayer::forward);
  py::class_<LlamaDecoderLayer>(m, "LlamaDecoderLayer", py::module_local())
      .def(py::init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &LlamaDecoderLayer::forward);
  py::class_<Qwen2DecoderLayer>(m, "Qwen2DecoderLayer", py::module_local())
      .def(py::init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &Qwen2DecoderLayer::forward);
}

TORCH_LIBRARY(tpp_llm, m) {
  m.def("fc_plain", &fc_plain_wrap);
  m.def("set_pg", &set_pg);
  m.def("allreduce", &allreduce);
  m.class_<LLMBlock>("LLMBlock").def("forward", &LLMBlock::forward);
  m.class_<GPTJBlock>("GPTJBlock")
      .def(torch::
               init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &GPTJBlock::forward);
  m.class_<OPTDecoderLayer>("OPTDecoderLayer")
      .def(
          torch::
              init<std::vector<at::Tensor>, long, double, double, long, bool>())
      .def("forward", &OPTDecoderLayer::forward);
  m.class_<LlamaDecoderLayer>("LlamaDecoderLayer")
      .def(torch::
               init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &LlamaDecoderLayer::forward);
  m.class_<Qwen2DecoderLayer>("Qwen2DecoderLayer")
      .def(torch::
               init<std::vector<at::Tensor>, long, double, long, long, long>())
      .def("forward", &Qwen2DecoderLayer::forward);
}
