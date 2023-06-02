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
int n_Layers = 0;
int current_layer = 0;
int cur_packing_layer_id = 0;
int is_self_output = 0;
unsigned int nnz_qkv = 0;
unsigned int dense_qkv = 0;
unsigned int nnz_intermediate = 0;
unsigned int dense_intermediate = 0;
unsigned int nnz_output = 0;
unsigned int dense_output = 0;
unsigned int nnz_selfoutput = 0;
unsigned int dense_selfoutput = 0;
unsigned int batch_execution = 0;

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(qkv_gemm, "qkv_gemm");
REGISTER_LOCAL_SCOPE(ac_gemm, "ac_gemm");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");

template <typename T>
void trans_output_block( long S, long H, T* input, T* output) {
  auto xform = XformTPP::XFORM_XPOSE_TPP;
  if (sizeof(T) != 4) {
    xform = XformTPP::XFORM_XPOSE_N2V_TPP;
  } 
  auto xform_tpp = SCOPEIT(XformExtTPP<T>(H, S, xform, true), (sizeof(T) == 4) ? XPOSE : VNNI);
  xform_tpp(input, output);
}

double measure_sparsity_in_blocked_nkkn_weight(at::Tensor& t_W) {
  double res = 0.0;
  if (t_W.dtype() == at::kFloat) {
    res = measure_sparsity_from_blocked_weight_tensor( GetVLAPtr<float>(t_W), t_W.sizes()[0], t_W.sizes()[1], t_W.sizes()[2], t_W.sizes()[3], get_vnni_block_size(t_W.dtype()));
  } else {
    res = measure_sparsity_from_blocked_weight_tensor( GetVLAPtr<bfloat16>(t_W), t_W.sizes()[0], t_W.sizes()[1], t_W.sizes()[2], t_W.sizes()[3], get_vnni_block_size(t_W.dtype()));
  }
  return res;
}

void create_bcsc_from_blocked_nkkn_weight(at::Tensor& t_W, tensor_bcsc_t *t_W_bcsc, int bcsc_bk, int bcsc_bn) {
  if (t_W.dtype() == at::kFloat) {
    create_bcsc_from_blocked_weight_tensor( GetVLAPtr<float>(t_W), t_W.sizes()[0], t_W.sizes()[1], t_W.sizes()[2], t_W.sizes()[3], get_vnni_block_size(t_W.dtype()), bcsc_bk, bcsc_bn, omp_get_max_threads(), t_W_bcsc);
  } else {
    create_bcsc_from_blocked_weight_tensor( GetVLAPtr<bfloat16>(t_W), t_W.sizes()[0], t_W.sizes()[1], t_W.sizes()[2], t_W.sizes()[3], get_vnni_block_size(t_W.dtype()), bcsc_bk, bcsc_bn, omp_get_max_threads(), t_W_bcsc);
  }
}

class BertEncoderLayer {
 public:
  at::Tensor t_Wq, t_Wk, t_Wv;
  tensor_bcsc_t t_Wq_bcsc, t_Wk_bcsc, t_Wv_bcsc;
  at::Tensor t_Bq, t_Bk, t_Bv;
  at::Tensor t_Wso, t_Wi, t_Wo;
  tensor_bcsc_t t_Wso_bcsc, t_Wi_bcsc, t_Wo_bcsc;
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
   
    auto bcsc_bk = env2int("BK", 4);
    auto bcsc_bn = env2int("BN", 4);
    double sparsity = 0.0;
    int is_spr = (libxsmm_cpuid(NULL) == LIBXSMM_X86_AVX512_SPR) ? 1 : 0;
    int bcsc_bk_use, bcsc_bn_use;
    double sparse_threshold = 1.0*env2int("SPTHRES", 40);

    /* Create BCSC sparse tensors */
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wi);
    printf("Wi in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity );
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;
    create_bcsc_from_blocked_nkkn_weight(t_Wi, &t_Wi_bcsc, bcsc_bk_use, bcsc_bn_use);
    nnz_intermediate += t_Wi_bcsc.nnz;
    dense_intermediate += t_Wi_bcsc.n_dense_elts;
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wo);
    printf("Wo in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity );
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;   
    create_bcsc_from_blocked_nkkn_weight(t_Wo, &t_Wo_bcsc, bcsc_bk_use, bcsc_bn_use);
    nnz_output += t_Wo_bcsc.nnz;
    dense_output += t_Wo_bcsc.n_dense_elts;
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wso);  
    printf("Wso in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity );
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;
    create_bcsc_from_blocked_nkkn_weight(t_Wso, &t_Wso_bcsc, bcsc_bk_use, bcsc_bn_use);
    nnz_selfoutput += t_Wso_bcsc.nnz;
    dense_selfoutput += t_Wso_bcsc.n_dense_elts;
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wq);
    printf("Wq in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity ); 
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;
    create_bcsc_from_blocked_nkkn_weight(t_Wq, &t_Wq_bcsc, bcsc_bk_use, bcsc_bn_use);
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wk);   
    printf("Wk in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity ); 
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;
    create_bcsc_from_blocked_nkkn_weight(t_Wk, &t_Wk_bcsc, bcsc_bk_use, bcsc_bn_use);
    sparsity = measure_sparsity_in_blocked_nkkn_weight(t_Wv);  
    printf("Wv in layer %d has sparsity %.3g\n", cur_packing_layer_id, sparsity);
    bcsc_bk_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bk : 32;
    bcsc_bn_use = (sparsity > sparse_threshold && is_spr > 0) ? bcsc_bn : 32;
    create_bcsc_from_blocked_nkkn_weight(t_Wv, &t_Wv_bcsc, bcsc_bk_use, bcsc_bn_use);
    nnz_qkv += t_Wq_bcsc.nnz + t_Wk_bcsc.nnz + t_Wv_bcsc.nnz;
    dense_qkv += t_Wq_bcsc.n_dense_elts + t_Wk_bcsc.n_dense_elts + t_Wv_bcsc.n_dense_elts;
    cur_packing_layer_id++;

    // std::cout << "F2=" << F2 << " H=" << H << " H1=" << H1 << std::endl;
  }

  template <typename T, bool transpose, bool vnni>
  inline void qkv_gemm(
      at::Tensor& t_HS,
      tensor_bcsc_t& t_wt_bcsc,    
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    auto sizes = t_HS.sizes();
    long S1 = sizes[0];
    long F1 = sizes[1];
    long S2, F2;
    if (t_HS.dtype() == at::kFloat) {
      S2 = sizes[2];
      F2 = sizes[3];
    } else {
      auto dims = sizes.size();
      if (dims == 5) {
        S2 = sizes[3];
        F2 = sizes[2]*sizes[4];
      } else {
        S2 = sizes[2];
        F2 = sizes[3];
      }
    }
    bool dt_low_prec = (t_HS.dtype() != at::kFloat);
    auto B = GetVLAPtr<T>(t_B, {F2});
    auto Out = GetVLAPtr<T>(t_Out, {F1, S2 * F2});
    auto HS = GetVLAPtr<T>(t_HS, {F1 * S2 * F2});
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

    // Create TPPs
    auto copy_bias_tpp = SCOPEIT(CpyBiasRowTPP<T>(S2, F2), BIAS);
    auto trans_out_tpp = SCOPEIT(XformExtTPP<T>(S2, F2, XformTPP::XFORM_XPOSE_TPP, true), XPOSE);
    auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
        S2,
        F2,
        F1*F2,
        t_wt_bcsc.bcsc_bk,
        t_wt_bcsc.bcsc_bn,
        F1*F2,
        -1,
        F2,
        1.0,
        0)));
    auto loop_scheme = large_cache_opt ? "bA" : "AB";
    auto qkv_loop = ThreadedLoop<2>({{S1}, {t_wt_bcsc.n_blocks}}, loop_scheme);
    {
      RECORD_SCOPE(qkv_gemm, {t_HS});
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], i_n = ind[1];
            T tmp_block[S2*F2];
            T tmp[S2*F2];
            int cur_n_blocks = (t_wt_bcsc.Nblocks_offsets[i_n+1] - t_wt_bcsc.Nblocks_offsets[i_n])/t_wt_bcsc.bcsc_bn;
            for (int l_block = 0; l_block < cur_n_blocks; l_block += t_wt_bcsc.bcsc_blocks_in_bn) {
              int nk = (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/S2;
              T* tmpp = (transpose || (vnni && dt_low_prec)) ? tmp : Out[s1][nk];
              copy_bias_tpp(B[nk], tmp_block);
              spmm_tpp(HS[s1], 
                       t_wt_bcsc, t_wt_bcsc.bcsc_blocks_in_bn, (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/t_wt_bcsc.bcsc_bn, 
                       tmp_block, true);
              trans_out_tpp( tmp_block, tmpp );
              if (transpose || (vnni && dt_low_prec)) {
                xform_tpp(tmpp, Out[s1][nk]);
              }    
            }
          },
          [&]() { spmm_tpp.config(); },
          [&]() { spmm_tpp.release(); });
    }
  }

  template <typename T>
  inline void q_gemm(
      at::Tensor& t_HS,
      tensor_bcsc_t& t_wt_bcsc,    
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    const bool use_at_vnni_local = t_HS.dtype() != at::kFloat && use_at_vnni;
    if (use_at_vnni_local) {
      qkv_gemm<T, true, true>(t_HS, t_wt_bcsc, t_B, t_Out);
    } else {
      qkv_gemm<T, false, false>(t_HS, t_wt_bcsc, t_B, t_Out);
    }
  }

  template <typename T>
  inline void k_gemm(
      at::Tensor& t_HS,
      tensor_bcsc_t& t_wt_bcsc,  
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, true, true>(t_HS, t_wt_bcsc, t_B, t_Out);
  }

  template <typename T>
  inline void v_gemm(
      at::Tensor& t_HS,
      tensor_bcsc_t& t_wt_bcsc,    
      at::Tensor& t_B,
      at::Tensor& t_Out) {
    qkv_gemm<T, false, true>(t_HS, t_wt_bcsc, t_B, t_Out);
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
    //long B = t_offs.sizes()[0] - 1;
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
            T tmp_block[H2*S2];
            for (int s21 = start; s21 < end; s21++) {
              long ls21 = s21 - start;
              a_gemm_tpp(QL[s11][n][0], KL_TV[s21][n][0], AS[ls21][0], H1);
              scale_tpp(AS[ls21][0], AS[ls21][0], one_by_sqrt_H);
              if (t_AM.numel() != 0)
                add_mask_tpp(AM[s21], AS[ls21][0]);
            }
            softmax_fwd_tpp(len, AS[0][0], AST[0][0]);
            for (int h1 = 0; h1 < H1; h1++) {
              c_gemm_tpp(AST[0][0], VL_V[start][n][h1], tmp_block, len);
              trans_output_block<T>( S2, H2, tmp_block, CL[s11][n][h1]);
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
      tensor_bcsc_t& t_wt_bcsc,  
      at::Tensor& t_bias,
      at::Tensor& t_gamma,
      at::Tensor& t_beta,
      at::Tensor& t_out,
      at::Tensor& t_out_trans) {
    auto in_sizes = t_in.sizes();
    auto S1 = in_sizes[0];
    auto Nc = in_sizes[1];
    auto S2 = in_sizes[2];
    auto Hc = in_sizes[3];
    auto Nk = t_wt_bcsc.sizes[0];
    auto Hk = t_wt_bcsc.sizes[3];
    auto in = GetVLAPtr<T>(t_in, {Nc * S2 * Hc});
    auto in2 = GetVLAPtr<T>(t_in2, {Nk, S2 * Hk});
    auto bias = GetVLAPtr<T>(t_bias, {Hk});
    auto gamma = GetVLAPtr<LT>(t_gamma, {Hk});
    auto beta = GetVLAPtr<LT>(t_beta, {Hk});
    auto out = GetVLAPtr<T>(t_out, {Nk, S2 * Hk});
    auto out_tr = GetVLAPtr<T>(t_out_trans, {Nk, S2 * Hk});  
    // Create TPPs
    auto copy_bias_tpp = SCOPEIT(CpyBiasRowTPP<T>(S2, Hk), BIAS);
    auto trans_out_tpp = SCOPEIT(XformExtTPP<T>(S2, Hk, XformTPP::XFORM_XPOSE_TPP, true), XPOSE);
    auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
        S2,
        Hk,
        Nc*Hc,
        t_wt_bcsc.bcsc_bk,
        t_wt_bcsc.bcsc_bn,
        Nc*Hc,
        -1,
        Hk,
        1.0,
        0)));
    auto add_tpp = SCOPEIT((AddTPP<T, T>(S2 * Hk)), EW_ADD);
    auto layer_norm_fwd_tpp = SCOPEIT((LayerNormFwdTPP<T, LT>(Nk, S2, Hk, eps)), LAYER_NORM);
    {
      RECORD_SCOPE(o_gemm, {t_in});
      auto loop_scheme = large_cache_opt ? "bA" : "AB";
      auto ogemm_loop =
          ThreadedLoop<2>({{S1}, {t_wt_bcsc.n_blocks}}, loop_scheme);
      bool parallelized_on_nk =
          large_cache_opt ? false : true; // ogemm_loop.is_parallel(2);
      ogemm_loop(

          [&](int* ind) {
            int s1 = ind[0], i_n = ind[1];
            T tmp_block[S2*Hk];
            int cur_n_blocks = (t_wt_bcsc.Nblocks_offsets[i_n+1] - t_wt_bcsc.Nblocks_offsets[i_n])/t_wt_bcsc.bcsc_bn;
            for (int l_block = 0; l_block < cur_n_blocks; l_block += t_wt_bcsc.bcsc_blocks_in_bn) {
              int nk = (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/S2;
              copy_bias_tpp(bias[nk], tmp_block);
              spmm_tpp(in[s1], 
                       t_wt_bcsc, t_wt_bcsc.bcsc_blocks_in_bn, (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/t_wt_bcsc.bcsc_bn, 
                       tmp_block, true);
              trans_out_tpp( tmp_block, out[s1][nk] );                
              add_tpp(out[s1][nk], in2[s1][nk], out[s1][nk]);
              if (!parallelized_on_nk && nk == Nk - 1) {
                layer_norm_fwd_tpp(
                    out[s1][0],
                    gamma[0],
                    beta[0],
                    nullptr,
                    nullptr,
                    out[s1][0]);
                if (is_self_output == 1 || current_layer < n_Layers-1) {             
                  for (int l_nk = 0; l_nk < Nk; l_nk++) {
                    trans_output_block<T>( S2, Hk, out[s1][l_nk], out_tr[s1][l_nk]);
                  }
                }
              }
            }
          },
          [&]() { spmm_tpp.config(); },
          [&]() { spmm_tpp.release(); });
      
      if (parallelized_on_nk) {
#pragma omp parallel for
        for (int s1 = 0; s1 < S1; s1++) {
          layer_norm_fwd_tpp(
              out[s1][0], gamma[0], beta[0], nullptr, nullptr, out[s1][0]);
          if (is_self_output == 1 || current_layer < n_Layers-1) {             
            for (int l_nk = 0; l_nk < Nk; l_nk++) {
              trans_output_block<T>( S2, Hk, out[s1][l_nk], out_tr[s1][l_nk]);
            }
          }
        }
      }
    }
  }

  template <typename T>
  inline void intermediate(
      at::Tensor t_in,
      tensor_bcsc_t& t_wt_bcsc,
      at::Tensor& t_bias,
      at::Tensor& t_out) {
    auto in_sizes = t_in.sizes();
    auto S1 = in_sizes[0];
    auto Nc = in_sizes[1];
    auto S2 = in_sizes[2];
    auto Hc = in_sizes[3];
    auto Nk = t_wt_bcsc.sizes[0];
    auto Hk = t_wt_bcsc.sizes[3];
    // Create TPPs
    auto copy_bias_tpp = SCOPEIT(CpyBiasRowTPP<T>(S2, Hk), BIAS);
    auto norm2v_out_tpp = SCOPEIT(XformExtTPP<T>(S2, Hk, XformTPP::XFORM_N2V_TPP, true), VNNI);
    auto spmm_tpp = SCOPEITGEMM((SpmmTPP<T, T>(
        S2,
        Hk,
        Nc*Hc,
        t_wt_bcsc.bcsc_bk,
        t_wt_bcsc.bcsc_bn,
        Nc*Hc,
        -1,
        Hk,
        1.0,
        0)));
    auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(S2 * Hk), ACT);
    auto in = GetVLAPtr<T>(t_in, {Nc * S2 * Hc});
    auto bias = GetVLAPtr<T>(t_bias, {Hk});
    auto out = GetVLAPtr<T>(t_out, {Nk, S2 * Hk});
    {
      RECORD_SCOPE(i_gemm, {t_in});
      auto loop_scheme = large_cache_opt ? "bA" : "AB";
      auto gemm_loop =
          ThreadedLoop<2>({{S1}, {t_wt_bcsc.n_blocks}}, loop_scheme);
      gemm_loop(
          [&](int* ind) {
            int s1 = ind[0], i_n = ind[1];
            T tmp_block[S2*Hk];
            int cur_n_blocks = (t_wt_bcsc.Nblocks_offsets[i_n+1] - t_wt_bcsc.Nblocks_offsets[i_n])/t_wt_bcsc.bcsc_bn;
            for (int l_block = 0; l_block < cur_n_blocks; l_block += t_wt_bcsc.bcsc_blocks_in_bn) {
              int nk = (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/S2;
              T *dst = (t_out.dtype() == at::kFloat) ? out[s1][nk] : tmp_block;
              copy_bias_tpp(bias[nk], dst);
              spmm_tpp(in[s1], 
                       t_wt_bcsc, t_wt_bcsc.bcsc_blocks_in_bn, (t_wt_bcsc.Nblocks_offsets[i_n] + l_block * t_wt_bcsc.bcsc_bn)/t_wt_bcsc.bcsc_bn, 
                       dst, true);
              if (t_out.dtype() != at::kFloat) {
                norm2v_out_tpp( dst, out[s1][nk] );
              }
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          },
          [&]() { spmm_tpp.config(); },
          [&]() { spmm_tpp.release(); });
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
    auto& t_SO_trans = t_scratch[6];
    auto& t_Out_trans = t_scratch[6];
    auto t_HS_qkv = t_HS;
    auto sizes = t_HS.sizes();
    long S1 = sizes[0];
    long F1 = sizes[1];
    long S2 = sizes[2];
    long F2 = sizes[3];

    if (current_layer == 0) {
      if (t_HS_qkv.dtype() == at::kFloat) {
        t_HS_qkv = act_tensor_trans(S1, F1, S2, F2, t_HS);
      } else {
        t_HS_qkv = act_tensor_trans_n2v(S1, F1, S2, F2, t_HS);  
      }
    } else {
      t_HS_qkv = t_Out_trans;
    }
    q_gemm<T>(t_HS_qkv, t_Wq_bcsc, t_Bq, t_QL);
    k_gemm<T>(t_HS_qkv, t_Wk_bcsc, t_Bk, t_KL_TV);
    v_gemm<T>(t_HS_qkv, t_Wv_bcsc, t_Bv, t_VL_V);
    self_attn<T>(t_QL, t_KL_TV, t_masks, t_VL_V, t_CL);
    is_self_output = 1;
    output<T, LT>(t_CL, t_HS, t_Wso_bcsc, t_Bso, t_Gso, t_Beta_so, t_SO, t_SO_trans);
    intermediate<T>(t_SO_trans, t_Wi_bcsc, t_Bi, t_I);
    is_self_output = 0;
    output<T, LT>(t_I, t_SO, t_Wo_bcsc, t_Bo, t_Go, t_Beta_o, t_Out, t_Out_trans);
    
    return t_Out;
  }
};

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
    n_Layers = nLayers;
    layers.reserve(nLayers);
    for (long unsigned int i = 0; i < nLayers; i++) {
      layers.push_back(new BertEncoderLayer(params[i], eps, N, H));
    }
  }

  ~BertEncoder() {
    for (long unsigned int i = 0; i < layers.size(); i++) {
      libxsmm_free( layers[i]->t_Wi_bcsc.data);
      libxsmm_free( layers[i]->t_Wi_bcsc.rowidx);
      libxsmm_free( layers[i]->t_Wi_bcsc.colptr);
      libxsmm_free( layers[i]->t_Wi_bcsc.Nblocks_offsets);
      //printf("N logical columns are %d with %d nnz\n", layers[i]->t_Wi_bcsc.n_blocks,  layers[i]->t_Wi_bcsc.nnz );
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
    ret.push_back(t_HS.new_empty(sizes1)); // SO_trans, O_trans
    
    return ret;
  }

  at::Tensor forward(at::Tensor& t_HS, std::vector<at::Tensor>& t_masks) {
    GlobalPass _gp(FWD);
    auto t_scratch = get_scratch(t_HS);
    auto dt = layers[0]->t_Wq.dtype();
    auto ldt = layers[0]->t_Go.dtype();

    current_layer = 0;
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
      current_layer++;
    }

    if (batch_execution == 0) {
      printf("\nQKV sparsity: %.5g %%\n",(1.0-(double)nnz_qkv/(double)dense_qkv)*100.0);
      printf("Intermediate sparsity: %.5g %%\n",(1.0-(double)nnz_intermediate/(double)dense_intermediate)*100.0);
      printf("Self-output sparsity: %.5g %%\n", (1.0-(double)nnz_selfoutput/(double)dense_selfoutput)*100.0);
      printf("Output sparsity: %.5g %%\n", (1.0-(double)nnz_output/(double)dense_output)*100.0);
      batch_execution = 1;
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
