/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"
#include "fused_gemm.h"
#include "qtypes.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(o_mlp_attn, "o_mlp_attn");
REGISTER_SCOPE(o_mlp, "o_mlp");
REGISTER_SCOPE(o_attn, "o_attn");
REGISTER_SCOPE(gather_mlp, "gather_mlp");
REGISTER_SCOPE(mlp_attn, "mlp_attn");
REGISTER_SCOPE(mlp, "mlp");
REGISTER_SCOPE(attn, "attn");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################

// T = int8_t
template <typename T, typename Tidx, typename Tout>
inline void gather_fc_plain_quant(
    at::Tensor t_in,
    at::Tensor t_idx,
    at::Tensor t_wt,
    at::Tensor t_out) {
  RECORD_SCOPE(pln_gemm, {t_in, t_idx, t_wt});
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto BS = t_idx.size(0);
  auto C = in_sizes[1];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto Ncb = Nc;
  auto BSb = 32L;
  auto rem = BS % 32;

  static_assert(std::is_same<T, int8_t>::value, "T must be int8_t\n");
  typedef int32_t BrTout;
  TPP_ASSERT(
      t_in.is_quantized() && t_in.qscheme() == at::kPerBlockAffine,
      "Input not quantized\n");
  TPP_ASSERT(
      t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockAffine,
      "Weight not quantized\n");
  auto constexpr pack_size = 1;
  long block_size = q_per_block_block_size(t_wt);
  TPP_ASSERT(block_size % Hc == 0, "Block size mismatch\n");
  long n_Hc_blocks = block_size / Hc;
  long ScNc = Nc / n_Hc_blocks;
  auto wt_V = GetVLAPtr<T>(t_wt, {Nc, (Hc * Hk) / pack_size});
  auto t_w_scf = q_per_block_scales(t_wt);
  auto t_i_scf = q_per_block_scales(t_in);
  auto w_scf = GetVLAPtr<float>(t_w_scf, {ScNc, Hk});
  auto i_scf = GetVLAPtr<float>(t_i_scf, {ScNc});
  auto b_vnni = 1;

  auto in = GetVLAPtr<T>(t_in, {C});
  auto idx = GetVLAPtr<Tidx>(t_idx);
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(BSb, C, C, C)), ROW_GT);
  auto gather_tpp_rem = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(rem, C, C, C)), ROW_GT);
  auto gather_scf_tpp = SCOPEIT((EmbeddingFwdTPP<float, Tidx, float>(BSb, ScNc, ScNc, ScNc)), ROW_GT);
  auto gather_scf_tpp_rem = SCOPEIT((EmbeddingFwdTPP<float, Tidx, float>(rem, ScNc, ScNc, ScNc)), ROW_GT);
  // auto brgemm_tpp = SCOPEIT(
  //     (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  // auto brgemm_tpp_rem = SCOPEIT(
  //     (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<T, BrTout, T>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
  auto brgemm_tpp_rem = SCOPEIT((BrgemmTPP<T, BrTout, T>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
  auto dequant_acc = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(BSb, Hk, Hk, K, ScNc)), EW_RCP);
  auto dequant_acc_rem = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(rem, Hk, Hk, K, ScNc)), EW_RCP);

  {
    RECORD_OMP_TIME();
    auto loop_scheme = "A";
    auto ogemm_loop = ThreadedLoop<1>(
        {LoopSpecs{0L, BS, BSb}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int s1 = ind[0];
          auto count = Nc;
          T tmp[BSb][Nc][Hc];
          float tmp_scf[BSb][ScNc];
          BrTout tmp_out[BSb * Hk];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            gather_tpp(in[0], &idx[s1], tmp[0][0]);
            gather_scf_tpp(i_scf[0], &idx[s1], tmp_scf[0]);
            for(int nk=0; nk < Nk; nk++) {
              zero_tpp(out[s1][nk]);
              for (int c = 0; c < Nc; c += n_Hc_blocks) {
                brgemm_tpp(tmp[0][c], wt_V[nk][c], tmp_out, n_Hc_blocks, true);
                dequant_acc(
                    tmp_out,
                    out[s1][nk],
                    &tmp_scf[0][c / n_Hc_blocks],
                    w_scf[nk][c / n_Hc_blocks]);
              }
            }
          } else {
            gather_tpp_rem(in[0], &idx[s1], tmp[0][0]);
            gather_scf_tpp_rem(i_scf[0], &idx[s1], tmp_scf[0]);
            for(int nk=0; nk < Nk; nk++) {
              zero_tpp_rem(out[s1][nk]);
              for (int c = 0; c < Nc; c += n_Hc_blocks) {
                brgemm_tpp_rem(tmp[0][c], wt_V[nk][c], tmp_out, n_Hc_blocks, false);
                dequant_acc_rem(
                    tmp_out,
                    out[s1][nk],
                    &tmp_scf[0][c / n_Hc_blocks],
                    w_scf[nk][c / n_Hc_blocks]);
              }
            }
          }
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

template <typename T, typename Tidx>
inline void gather_fc_plain(
    at::Tensor t_in,
    at::Tensor t_idx,
    at::Tensor t_wt,
    at::Tensor t_out) {
  RECORD_SCOPE(pln_gemm, {t_in, t_wt});
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto BS = t_idx.size(0);
  auto C = in_sizes[1];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto Ncb = Nc;
  auto BSb = 32L;
  auto rem = BS % 32;

  auto in = GetVLAPtr<T>(t_in, {C});
  auto idx = GetVLAPtr<Tidx>(t_idx);
  auto wt_V = GetVLAPtr<T>(t_wt, {Nc, Hc * Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(BSb, C, C, C)), ROW_GT);
  auto gather_tpp_rem = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(rem, C, C, C)), ROW_GT);
  auto brgemm_tpp = SCOPEIT(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEIT(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_OMP_TIME();
    auto loop_scheme = "aB";
    auto ogemm_loop = ThreadedLoop<2>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          T tmp[BSb][Nc][Hc];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            gather_tpp(in[0], &idx[s1], tmp[0][0]);
            for(int nk=0; nk < Nk; nk++) {
              if (nc == 0)
                zero_tpp(out[s1][nk]);
              brgemm_tpp(tmp[0][0], wt_V[nk][nc], out[s1][nk], count, true);
            }
          } else {
            gather_tpp_rem(in[0], &idx[s1], tmp[0][0]);
            for(int nk=0; nk < Nk; nk++) {
               if (nc == 0) {
                 zero_tpp_rem(out[s1][nk]);
               }
               brgemm_tpp_rem(tmp[0][0], wt_V[nk][nc], out[s1][nk], count, false);
            }
          }
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

static at::Tensor gather_mlp(
    std::vector<at::Tensor> inp) {
  GlobalPass _gp(FWD);

  at::Tensor t_in = inp[0];
  at::Tensor t_idx = inp[1];
  at::Tensor t_wt = inp[2];

  auto wt_sizes = t_wt.sizes();
  auto K = wt_sizes[0] * wt_sizes[3];
  auto BS = t_idx.size(0);
  auto out_dt = t_in.is_quantized() ? at::kBFloat16 : t_in.scalar_type();
  auto t_out = t_in.new_empty({BS, K}, out_dt);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    gather_fc_plain<float, long>(t_in, t_idx, t_wt, t_out);
  } else if (dt == at::kBFloat16) {
    gather_fc_plain<bfloat16, long>(t_in, t_idx, t_wt, t_out);
  } else if (dt == at::kBFloat8) {
    gather_fc_plain<bfloat8, long>(t_in, t_idx, t_wt, t_out);
  } else if (dt == at::kQInt8) {
    gather_fc_plain_quant<int8_t, long, bfloat16>(t_in, t_idx, t_wt, t_out);
  } else {
    TPP_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

std::vector<at::Tensor> gather_mlp_attn(
  long align,
  int add_bias,
  int use_qint8gemm,
  std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
#include "gather_mlp_attn_qint8.h"
  }

std::vector<at::Tensor> mlp_attn(
    long align,
    int add_bias,
    int use_bf_or_fp16,
    int use_qint8_gemm,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);

  if(!use_qint8_gemm) {
    auto dact = -1;
    if(inputs[0].dtype() == at::kChar) dact=0;
    else if(inputs[0].dtype() == at::kFloat8_e4m3fn) dact=1;
    else if(inputs[0].dtype() == at::kFloat8_e5m2) dact=2;
    else if(inputs[0].dtype() == at::kBFloat16) dact=3;

    assert(inputs[2].dtype() == at::kBFloat16);
    assert(inputs[3].dtype() == at::kBFloat16);

    if(dact==0) {
      typedef int8_t Tact;
#include "mlp_attn_scf.h"
    }
    else if(dact==1) {
      typedef hfloat8 Tact;
#include "mlp_attn.h"
    }
    else if(dact==2) {
      typedef bfloat8 Tact;
#include "mlp_attn.h"
    }
    else if(dact==3) {
      typedef bfloat16 Tact;
#include "mlp_attn.h"
    }
    else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
    }
  }
  else {
    if(use_bf_or_fp16==0) {
      typedef bfloat16 Tact;
#include "mlp_attn_qint8.h"
    }
    else if(use_bf_or_fp16==1) {
      typedef half Tact;
#include "mlp_attn_qint8.h"
    }
    else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
    }
  }
}

at::Tensor mlp(long align, int add_bias, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);

  auto dact=-1;
  if(inputs[0].dtype() == at::kFloat) dact=0;
  else if(inputs[0].dtype() == at::kBFloat16) dact=1;
  else if(inputs[0].dtype() == at::kHalf) dact=2;

  auto dwt = -1;
  if (inputs[1].dtype() == at::kFloat)
    dwt = 0;
  else if (inputs[1].dtype() == at::kBFloat16)
    dwt = 1;
  else if (inputs[2].dtype() == at::kHalf)
    dwt = 2;

  if (dact == 0) {
    typedef float Tact;
    if (dwt == 0) {
      typedef float Tprm;
#include "mlp.h"
    } else if (dwt == 1) {
      typedef bfloat16 Tprm;
#include "mlp.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  }
  else if (dact == 1) {
    typedef bfloat16 Tact;
    if (dwt == 1) {
      typedef bfloat16 Tprm;
#include "mlp.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  }
  else if (dact == 2) {
    typedef half Tact;
    if (dwt == 2) {
      typedef half Tprm;
#include "mlp.h"
    } else {
      TPP_ASSERT(
          0, "%s:%d Unsupported type for parameters\n", __FILE__, __LINE__);
    }
  }
  else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

at::Tensor attn(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);

  auto dt = inputs[0].dtype();
  if(dt == at::kFloat) {
    typedef float T;
#include "attn.h"
  }
  else if(dt == at::kBFloat16) {
    typedef bfloat16 T;
#include "attn.h"
  }
  else if(dt == at::kBFloat8) {
    typedef bfloat8 T;
#include "attn.h"
  }
  else {
    TPP_ASSERT(
        0, "%s:%d Unsupported type for activations\n", __FILE__, __LINE__);
  }
}

REGISTER_SUBMODULE(_fused_gat_inf, m) {
  m.def("gather_mlp", &gather_mlp, "TPP fused Gather with MLP");
  m.def("gather_mlp_attn", &gather_mlp_attn, "TPP fused Gather with MLP-Attention");
  m.def("mlp_attn", &mlp_attn, "TPP fused MLP-Attention");
  m.def("mlp", &mlp, "TPP fused MLP");
  m.def("attn", &attn, "TPP Attn");
}
