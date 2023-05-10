/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "utils.h"

template<typename DType>
void create_bcsc_from_blocked_weight_tensor(DType *dense_wt_ptr, int Nb, int Kb, int bk, int bn, int v, int bcsc_bk, int bcsc_bn, int N_target_blocks, tensor_bcsc_t *sparse_wt) {
  int nnz = 0, l_zero_block = 0;
  int N = Nb*bn;
  int K = Kb*bk*v;
  unsigned int l_val_idx = 0;
  unsigned int l_nz_block_id = 0;
  unsigned int *l_colptr;
  unsigned int *l_rowidx;
  DType        *l_data, val;
  unsigned int *Nblocks_offsets = (unsigned int*)libxsmm_aligned_malloc(sizeof(unsigned int) * N_target_blocks, 64);
  int l_i, l_j, l_di, l_dj, l_ui, l_uj, l_ii, l_jj, total_nnz_processed, nnz_entries_per_block, all_done = 0, n_blocks = 0;
  LIBXSMM_VLA_DECL(5, DType, l_wt_dense, dense_wt_ptr, Kb, bk, bn, v);
  int l_blocking_step = (bn + bcsc_bn - 1)/bcsc_bn;

  if (bn % bcsc_bn != 0) {
    printf("WARNING: end up with assymetric logical blocks...\n");
  }

  /* First pass to count number of non-zeros */
  for ( l_i = 0; l_i < N/bcsc_bn; l_i++ ) {
    for ( l_j = 0; l_j < K/bcsc_bk; l_j++ ) {
      l_ui = l_i * bcsc_bn;
      l_uj = l_j * bcsc_bk;
      l_zero_block = 1;
      /* Scan block to see if there are non-zeros */
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_ii = l_ui + l_di;
          l_jj = l_uj + l_dj;
          val = LIBXSMM_VLA_ACCESS(5, l_wt_dense, l_ii/bn, l_jj/(bk*v), (l_jj%(bk*v))/v, l_ii%bn, (l_jj%(bk*v))%v, Kb, bk, bn, v);
          if ((val != 0) && (val != -0)) {
            l_zero_block = 0;
          }
        }
      }
      if (l_zero_block == 0) {
        nnz += bcsc_bk * bcsc_bn;
      }
    }
  }

  /* Allocate BCSC structures */
  l_colptr  = (unsigned int*) libxsmm_aligned_malloc( (N/bcsc_bn+1)*sizeof(unsigned int), 64 );
  l_rowidx  = (unsigned int*) libxsmm_aligned_malloc( nnz/(bcsc_bk*bcsc_bn)*sizeof(unsigned int),   64 );
  l_data    = (DType*) libxsmm_aligned_malloc( nnz*sizeof(DType), 64 );

  /* Second pass to create BCSC */
  l_nz_block_id = 0;
  l_colptr[N/bcsc_bn] = nnz/(bcsc_bk*bcsc_bn);
  for ( l_i = 0; l_i < N/bcsc_bn; l_i++ ) {
    l_colptr[l_i] = l_nz_block_id;
    for ( l_j = 0; l_j < K/bcsc_bk; l_j++ ) {
      l_ui = l_i * bcsc_bn;
      l_uj = l_j * bcsc_bk;
      l_zero_block = 1;
      /* Scan block to see if there are non-zeros */
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_ii = l_ui + l_di;
          l_jj = l_uj + l_dj;
          val = LIBXSMM_VLA_ACCESS(5, l_wt_dense, l_ii/bn, l_jj/(bk*v), (l_jj%(bk*v))/v, l_ii%bn, (l_jj%(bk*v))%v, Kb, bk, bn, v);
          if ((val != 0) && (val != -0)) {
            l_zero_block = 0;
          }
        }
      }
      if (l_zero_block == 0) {
        l_rowidx[l_nz_block_id] = l_j;
        for (l_di = 0; l_di < bcsc_bn; l_di++) {
          for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
            l_ii = l_ui + l_di;
            l_jj = l_uj + l_dj;
            val = LIBXSMM_VLA_ACCESS(5, l_wt_dense, l_ii/bn, l_jj/(bk*v), (l_jj%(bk*v))/v, l_ii%bn, (l_jj%(bk*v))%v, Kb, bk, bn, v);
            l_data[l_val_idx] = val;
            l_val_idx++;
          }
        }
        l_nz_block_id++;  
      }
    }
  }

  /* Convert the BCSC to be in VNNI4T if need be */
  if ((v == 4) && (sizeof(DType) == 2) && (bcsc_bk > v)) {
    l_di = 0;
    l_dj = 0;  
    for ( l_i = 0; l_i < nnz/(bcsc_bk*bcsc_bn); l_i++) {
      DType tmp_block[bcsc_bk*bcsc_bn];
      memcpy(tmp_block, &l_data[l_i*(bcsc_bk*bcsc_bn)], (bcsc_bk*bcsc_bn)*sizeof(DType));
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_data[l_i*(bcsc_bk*bcsc_bn) + (l_dj/v) * (bcsc_bn * v) + l_di * v + l_dj % v] = tmp_block[l_di * bcsc_bk + l_dj];    
        }
      }
    }
  }

  /* Logically partition the sparse BCSC matrix  */
  total_nnz_processed = 0;
  nnz_entries_per_block = (nnz+N_target_blocks-1)/N_target_blocks;
  all_done = 0;
  l_i = 0;
  l_j = 0;
  Nblocks_offsets[0] = 0;
  while (all_done == 0) {
    int nnz_so_far = 0;
    while ((nnz_so_far < nnz_entries_per_block) && (l_j < N/bcsc_bn)) {
      if (l_j + l_blocking_step <= N/bcsc_bn) {
        nnz_so_far += (l_colptr[l_j+l_blocking_step] - l_colptr[l_j]) * bcsc_bk * bcsc_bn;
        l_j += l_blocking_step;
      } else {
        /* Should not happen  */
      }
    }
    total_nnz_processed += nnz_so_far;
    l_i++;
    if (total_nnz_processed < nnz) {
      Nblocks_offsets[l_i] = l_j*bcsc_bn;
      if (l_j >= N/bcsc_bn) {
        all_done = 1; 
      }
    } else {
      Nblocks_offsets[l_i] = N;
      all_done = 1; 
    }
  }
  n_blocks = l_i;

  sparse_wt->data = (void*)l_data;
  sparse_wt->colptr = l_colptr;
  sparse_wt->rowidx = l_rowidx;
  sparse_wt->Nblocks_offsets = Nblocks_offsets;
  sparse_wt->n_blocks = n_blocks;
  sparse_wt->nnz = nnz;
  sparse_wt->bcsc_bn = bcsc_bn;
  sparse_wt->bcsc_bk = bcsc_bk;
  sparse_wt->bcsc_blocks_in_bn = l_blocking_step;
}

template <typename T>
inline at::Tensor wt_tensor_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  return input.view({Nk, Nc, Hc/BS, BS, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto Hcp2 = (Hc + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hcp2, Hk, BS});
  auto out = GetVLAPtr<T>(output, {Hcp2 * Hk * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto n2v_tpp = SCOPEIT(
      XformExtTPP<T>(Hc, Hk, Hcp2 * BS, Hk, XformTPP::XFORM_N2V_TPP), VNNI);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/BS, BS}).permute({0, 1, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_n2v_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/BS, BS}).permute({1, 0, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nc, Nk, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Nk, Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Nc, Hc * Hk});
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
  for (int nk = 0; nk < Nk; nk++) {
    for (int nc = 0; nc < Nc; nc++) {
      trans_n2v_tpp(in[nk][nc], out[nc][nk]);
    }
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_v2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/BS, Hk/BS, BS, BS}).permute({0, 1, 3, 2, 5, 4}).contiguous().view({Nk, Nc, Hk/BS, Hc, BS});
#else
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_v2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_v2v_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/BS, Hk/BS, BS, BS}).permute({1, 0, 3, 2, 5, 4}).contiguous().view({Nc, Nk, Hk/BS, Hc, BS});
#else
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nc, Nk, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Nk, Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Nc, Hc * Hk});
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
  for (int nk = 0; nk < Nk; nk++) {
    for (int nc = 0; nc < Nc; nc++) {
      trans_v2v_tpp(in[nk][nc], out[nc][nk]);
    }
  }
  return output;
#endif
}

USING_SCOPE(w_vnni);

inline at::Tensor wt_tensor_for_fwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_vnni, {input});
  if (input.dtype() != at::kFloat) {
    if (input.dim() == 5) {
      return input;
    } else {
      if (input.dtype() == at::kBFloat16) {
        return wt_tensor_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
      } else if (input.dtype() == at::kBFloat8) {
        return wt_tensor_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
      } else {
        TPP_ASSERT(false, "Unsupported datatype!");
      }
    }
  } else {
    return input;
  }
}

USING_SCOPE(w_xpose);

inline at::Tensor wt_tensor_for_bwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kBFloat8) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kFloat) {
#if 0
    return input.permute({0, 1, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    auto out = GetVLAPtr<float>(output, {Hk * Hc});
    auto in = GetVLAPtr<float>(input, {Hc * Hk});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < Nk * Nc; n++) {
      trans_tpp(in[n], out[n]);
    }
    return output;
#endif
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
}

inline at::Tensor wt_tensor_for_bwd_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v_compact<bfloat16>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v_compact<bfloat16>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kBFloat8) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v_compact<bfloat8>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v_compact<bfloat8>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kFloat) {
#if 0
    return input.permute({1, 0, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    auto out = GetVLAPtr<float>(output, {Nk, Hk * Hc});
    auto in = GetVLAPtr<float>(input, {Nc, Hc * Hk});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int nk = 0; nk < Nk; nk++) {
      for (int nc = 0; nc < Nc; nc++) {
        trans_tpp(in[nk][nc], out[nc][nk]);
      }
    }
    return output;
#endif
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
}

USING_SCOPE(a_xpose);

inline at::Tensor act_tensor_trans(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, H, S2});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < B * S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < B * S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

inline at::Tensor act_tensor_trans(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 3, 2}).contiguous();
#else
  auto output = input.new_empty({S1, N, H, S2});
  if (input.dtype() == at::kFloat) {
    auto out = GetVLAPtr<float>(output, {H * S2});
    auto in = GetVLAPtr<float>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

inline at::Tensor act_tensor_trans_compact(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({1, 0, 3, 2}).contiguous();
#else
  auto output = input.new_empty({N, S1, H, S2});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {N, H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          trans_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {N, H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          trans_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

USING_SCOPE(a_vnni);

inline at::Tensor act_tensor_n2v(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({B, S1, N, S2 / BS, BS, H})
      .permute({0, 1, 2, 3, 5, 4})
      .contiguous();
#else
  auto output = input.new_empty({B, S1, N, S2 / BS, H, BS});
  auto out = GetVLAPtr<bfloat16>(output, {H * S2});
  auto in = GetVLAPtr<bfloat16>(input, {H * S2});
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < B * S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor act_tensor_n2v(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({S1, N, S2 / BS, BS, H})
      .permute({0, 1, 2, 4, 3})
      .contiguous();
#else
  auto output = input.new_empty({S1, N, S2 / BS, H, BS});
  auto out = GetVLAPtr<bfloat16>(output, {H * S2});
  auto in = GetVLAPtr<bfloat16>(input, {H * S2});
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor act_tensor_n2v_compact(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 0
  return input.view({S1, N, S2/BS, BS, H}).permute({1,0,2,4,3}).contiguous();
#else
  auto output = input.new_empty({N, S1, S2 / BS, H, BS});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {N, H * S2});
    auto n2v_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          n2v_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {N, H * S2});
    auto n2v_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          n2v_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }

  return output;
#endif
}

inline at::Tensor get_padded_activation_for_vnni(at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  auto dtype = input.dtype();
  if (dtype == at::kFloat)
    return input;
  const int align = get_vnni_block_size(dtype);
  auto sizes = input.sizes();
  int ndims = input.dim();
  TPP_ASSERT(ndims >= 2, "Invalid shape\n");
  auto C = sizes[ndims - 1];
  int pad = C % align;
  if (pad == 0)
    return input;
  std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
  new_sizes[ndims - 1] = align - pad;
  auto output = at::cat({input, input.new_zeros(new_sizes)}, ndims - 1);
  return output;
}

inline at::Tensor act_tensor_trans_n2v(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 0
  return input.view({S1, N, S2, H / BS, BS})
      .permute({0, 1, 3, 2, 4})
      .contiguous();
#else
  auto output = input.new_empty({S1, N, H / BS, S2, BS});
  auto out = GetVLAPtr<bfloat16>(output, {H * S2});
  auto in = GetVLAPtr<bfloat16>(input, {H * S2});
  auto xpose_n2v_tpp = SCOPEIT(
      XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < S1 * N; n++) {
      xpose_n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

USING_SCOPE(zero);

inline void tensor_set_zero(long N, long sz, at::Tensor& input) {
#if 0
  input.zero_();
#else
  RECORD_SCOPE(zero, {input});
  if (input.dtype() == at::kFloat) {
    auto in = GetVLAPtr<float>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else if (input.dtype() == at::kBFloat16) {
    auto in = GetVLAPtr<bfloat16>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat16>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else {
    auto in = GetVLAPtr<bfloat8>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat8>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  }
#endif
}

template <typename T>
class LToPBlockAccessMapper {
 public:
  LToPBlockAccessMapper(long M, long N) : M(M), N(N) {}
  long operator()(long i, long j) {
    if (std::is_same<T, float>()) {
      return i * N + j;
    } else {
      return j * M + i;
    }
  }

 private:
  long M, N;
};

template <typename Tin, typename Tout>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    Tout** ptrs,
    Tin* buf,
    bool accumulate = false) {
  ScopedTimer _t(EW_RED);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}

#endif // _TENSOR_HELPER_H_
