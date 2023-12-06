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

template <typename DType>
int l_ullcompare(const void* a, const void* b) {
  const unsigned long long aull = *(const unsigned long long*)a;
  const unsigned long long bull = *(const unsigned long long*)b;
  if (aull < bull) {
    return -1;
  } else if (aull > bull) {
    return 1;
  } else {
    return 0;
  }
}

template <typename DType>
void l_shuffle_array(unsigned long long* array, int n) {
  if (n > 1) {
    int i;
    for (i = 0; i < n - 1; i++) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

template <typename DType>
int l_is_dense_grid_point(
    unsigned long long grid_point_id,
    int n_dense_grid_points,
    unsigned long long* grid_point_array) {
  unsigned long long key = grid_point_id;
  unsigned long long* found_ptr = (unsigned long long*)bsearch(
      &key,
      grid_point_array,
      n_dense_grid_points,
      sizeof(unsigned long long),
      l_ullcompare<DType>);
  return ((found_ptr == NULL) ? 0 : 1);
}

template <typename DType>
void sparsify_weight_tensor(
    DType* dense_wt_ptr,
    int Nb,
    int Kb,
    int bk,
    int bn,
    int v,
    int sparse_block_bk,
    int sparse_block_bn,
    double sparse_frac) {
  int N = Nb * bn;
  int K = Kb * bk * v;
  LIBXSMM_VLA_DECL(5, DType, l_wt_dense, dense_wt_ptr, Kb, bk, bn, v);
  unsigned long long i;
  long long l_i, l_j, in, ik;
  if (sparse_block_bn == 1 && sparse_block_bk == 1) {
    for (l_i = 0; l_i < N / sparse_block_bn; l_i++) {
      for (l_j = 0; l_j < K / sparse_block_bk; l_j++) {
        float tmp = (float)libxsmm_rng_f64();
        if (tmp <= sparse_frac) {
          long long l_ui = l_i * sparse_block_bn;
          long long l_uj = l_j * sparse_block_bk;
          long long l_di = 0, l_dj = 0;
          for (l_di = 0; l_di < sparse_block_bn; l_di++) {
            for (l_dj = 0; l_dj < sparse_block_bk; l_dj++) {
              in = l_ui + l_di;
              ik = l_uj + l_dj;
              LIBXSMM_VLA_ACCESS(
                  5,
                  l_wt_dense,
                  in / bn,
                  ik / (bk * v),
                  (ik % (bk * v)) / v,
                  in % bn,
                  (ik % (bk * v)) % v,
                  Kb,
                  bk,
                  bn,
                  v) = (DType)0;
            }
          }
        }
      }
    }
  } else {
    unsigned long long n_grid_points =
        (N / sparse_block_bn) * (K / sparse_block_bk);
    unsigned long long* grid_point_array =
        (unsigned long long*)malloc(n_grid_points * sizeof(unsigned long long));
    unsigned long long n_dense_grid_points =
        (unsigned long long)((double)(1.0 - sparse_frac) * n_grid_points);
    for (i = 0; i < n_grid_points; i++) {
      grid_point_array[i] = (unsigned long long)i;
    }
    /* Pemute array of n grid points and consider densifying on the ones with id
     * <= n_dense_grid_points */
    l_shuffle_array<DType>(grid_point_array, n_grid_points);
    qsort(
        grid_point_array,
        n_dense_grid_points,
        sizeof(unsigned long long),
        l_ullcompare<DType>);
    for (l_i = 0; l_i < N / sparse_block_bn; l_i++) {
      for (l_j = 0; l_j < K / sparse_block_bk; l_j++) {
        if (l_is_dense_grid_point<DType>(
                l_i * (K / sparse_block_bk) + l_j,
                n_dense_grid_points,
                grid_point_array) == 0) {
          long long l_ui = l_i * sparse_block_bn;
          long long l_uj = l_j * sparse_block_bk;
          long long l_di = 0, l_dj = 0;
          for (l_di = 0; l_di < sparse_block_bn; l_di++) {
            for (l_dj = 0; l_dj < sparse_block_bk; l_dj++) {
              in = l_ui + l_di;
              ik = l_uj + l_dj;
              LIBXSMM_VLA_ACCESS(
                  5,
                  l_wt_dense,
                  in / bn,
                  ik / (bk * v),
                  (ik % (bk * v)) / v,
                  in % bn,
                  (ik % (bk * v)) % v,
                  Kb,
                  bk,
                  bn,
                  v) = (DType)0;
            }
          }
        }
      }
    }
    free(grid_point_array);
  }
}

template <typename DType>
double measure_sparsity_from_blocked_weight_tensor(
    DType* dense_wt_ptr,
    int Nb,
    int Kb,
    int bk,
    int bn,
    int v) {
  unsigned int n_entries = 0;
  unsigned int nnz = 0;
  double res = 0.0;
  int N = Nb * bn;
  int K = Kb * bk * v;
  int l_i, l_j;
  DType val;
  LIBXSMM_VLA_DECL(5, DType, l_wt_dense, dense_wt_ptr, Kb, bk, bn, v);

  for (l_i = 0; l_i < N; l_i++) {
    for (l_j = 0; l_j < K; l_j++) {
      val = LIBXSMM_VLA_ACCESS(
          5,
          l_wt_dense,
          l_i / bn,
          l_j / (bk * v),
          (l_j % (bk * v)) / v,
          l_i % bn,
          (l_j % (bk * v)) % v,
          Kb,
          bk,
          bn,
          v);
      if ((val != 0) && (val != -0)) {
        nnz++;
      }
      n_entries++;
    }
  }

  res = 100.0 * (1.0 - ((double)1.0 * nnz) / ((double)n_entries));
  return res;
}

template <typename DType>
void create_compressed_from_blocked_weight_tensor(
    DType* dense_wt_ptr,
    int Nb,
    int Kb,
    int bk,
    int bn,
    int v,
    tensor_compressed_t* compressed_wt) {
  long long nnz = 0;
  int N = Nb * bn;
  int K = Kb * bk * v;
  int l_i = 0, l_j = 0, l_ii = 0, l_jj = 0, l_v = 0, l_c = 0, l_cur = 0,
      l_bm = 0;
  DType val;
  LIBXSMM_VLA_DECL(5, DType, l_wt_dense, dense_wt_ptr, Kb, bk, bn, v);
  auto sparse_pct = env2int("SPFRAC", 0);
  double sparse_frac = (double)(1.0 * sparse_pct / 100.0);
  long long* column_offsets = NULL;
  DType** data_ptrs = NULL;
  unsigned int* bitmap = NULL;
  unsigned short* bitmap_array;
  int is_spr = (libxsmm_cpuid(NULL) == LIBXSMM_X86_AVX512_SPR) ? 1 : 0;

  /* Sparsify tensors for benchmarking purposes */
  if (sparse_pct > 0) {
    sparsify_weight_tensor<DType>(
        dense_wt_ptr, Nb, Kb, bk, bn, v, 1, 1, sparse_frac);
  }

  /* First pass to count number of non-zeros */
  for (l_i = 0; l_i < Nb * Kb * bk * bn * v; l_i++) {
    val = dense_wt_ptr[l_i];
    if ((val != 0) && (val != -0)) {
      nnz++;
    }
  }

  /* Allocate compressedC structures */
  // if (compressed_wt->data)
  //  libxsmm_free(compressed_wt->data);
  if (compressed_wt->bitmap)
    libxsmm_free(compressed_wt->bitmap);
  // if (compressed_wt->column_offsets)
  //  libxsmm_free(compressed_wt->column_offsets);

  bitmap = (unsigned int*)libxsmm_aligned_malloc((N * K) / 8, 64);
  column_offsets =
      (long long*)libxsmm_aligned_malloc(Nb * sizeof(long long), 64);
  // compressed_weight = (DType*)libxsmm_aligned_malloc(nnz * sizeof(DType),
  // 64);
  data_ptrs = (DType**)libxsmm_aligned_malloc(Nb * sizeof(DType*), 64);

  bitmap_array = (unsigned short*)bitmap;

  //#pragma omp parallel for
  //  for (l_i = 0; l_i < nnz; l_i++) {
  //    compressed_weight[l_i] = (DType)1;
  //  }

#pragma omp parallel for
  for (l_i = 0; l_i < (N * K) / 16; l_i++) {
    bitmap_array[l_i] = (unsigned short)0;
  }

  /* First pass to precompute column offsets... */
  if (v == 1 || is_spr > 0) {
#pragma omp parallel for
    for (l_i = 0; l_i < Nb; l_i++) {
      long long cur_count = 0, _l_j = 0, _l_cur = 0;
      cur_count = 0;
      for (_l_j = 0; _l_j < Kb; _l_j++) {
        DType* tmp = (DType*)&LIBXSMM_VLA_ACCESS(
            5, l_wt_dense, l_i, _l_j, 0, 0, 0, Kb, bk, bn, v);
        for (_l_cur = 0; _l_cur < bn * bk * v; _l_cur += 16) {
          int l_b = 0;
          for (l_b = 0; l_b < 16; l_b++) {
            DType cur_data = tmp[_l_cur + l_b];
            if ((cur_data != 0) && (cur_data != -0)) {
              cur_count++;
            }
          }
        }
      }
      column_offsets[l_i] = cur_count;
    }
  } else {
#pragma omp parallel for
    for (l_i = 0; l_i < Nb; l_i++) {
      long long cur_count = 0, _l_j = 0, _l_jj = 0, _l_ii = 0, _l_v = 0, _l_cur = 0;
      cur_count = 0;
      for (_l_j = 0; _l_j < Kb; _l_j++) {
        /* Undo vnni format to tmp buf */
        DType tmp[bn * bk * v];
        for (_l_jj = 0; _l_jj < bk; _l_jj++) {
          for (_l_ii = 0; _l_ii < bn; _l_ii++) {
            for (_l_v = 0; _l_v < v; _l_v++) {
              tmp[(_l_jj * v + _l_v) * bn + _l_ii] = LIBXSMM_VLA_ACCESS(
                  5, l_wt_dense, l_i, _l_j, _l_jj, _l_ii, _l_v, Kb, bk, bn, v);
            }
          }
        }
        for (_l_cur = 0; _l_cur < bn * bk * v; _l_cur += 16) {
          int l_b = 0;
          for (l_b = 0; l_b < 16; l_b++) {
            DType cur_data = tmp[_l_cur + l_b];
            if ((cur_data != 0) && (cur_data != -0)) {
              cur_count++;
            }
          }
        }
      }
      column_offsets[l_i] = cur_count;
    }
  }

  /* Now allocate memory using the just computed buffer sizes */
  /* We do Nb allocations and we partiotion then with rations 24(numa1) and 76
   * pct(numa0) */
  l_c = 0;
  long long total_so_far = 0;
  for (l_i = 0; l_i < Nb; l_i++) {
    long long cur_size = column_offsets[l_i];
    if (1.0 * (total_so_far + cur_size) <= 0.76 * nnz) {
      /* Allocate on numa node 0 */
      data_ptrs[l_i] =
          (DType*)libxsmm_aligned_malloc(cur_size * sizeof(DType), 64);
    } else {
      /* Allocate on numa node 1 */
      data_ptrs[l_i] =
          (DType*)libxsmm_aligned_malloc(cur_size * sizeof(DType), 64);
    }
    total_so_far += cur_size;
  }

  if (v == 1 || is_spr > 0) {
#pragma omp parallel for
    for (l_i = 0; l_i < Nb; l_i++) {
      DType* compressed_weight = data_ptrs[l_i];
      long long cur_count = 0, _l_j = 0, _l_cur = 0, _l_bm = 0;
      cur_count = 0;
      for (_l_j = 0; _l_j < Kb; _l_j++) {
        DType* tmp = (DType*)&LIBXSMM_VLA_ACCESS(
            5, l_wt_dense, l_i, _l_j, 0, 0, 0, Kb, bk, bn, v);
        for (_l_cur = 0; _l_cur < bn * bk * v; _l_cur += 16) {
          unsigned short mask = (unsigned short)0;
          int l_b = 0;
          for (l_b = 0; l_b < 16; l_b++) {
            DType cur_data = tmp[_l_cur + l_b];
            if ((cur_data != 0) && (cur_data != -0)) {
              mask |= (unsigned short)((unsigned short)1 << l_b);
              compressed_weight[cur_count] = cur_data;
              cur_count++;
            }
          }
          bitmap_array[_l_bm + l_i * (K * bn) / 16] = mask;
          _l_bm++;
        }
      }
    }
  } else {
#pragma omp parallel for
    for (l_i = 0; l_i < Nb; l_i++) {
      DType* compressed_weight = data_ptrs[l_i];
      long long cur_count = 0, _l_j = 0, _l_jj = 0, _l_ii = 0, _l_v = 0, _l_cur = 0, _l_bm = 0;
      cur_count = 0;
      for (_l_j = 0; _l_j < Kb; _l_j++) {
        /* Undo vnni format to tmp buf */
        DType tmp[bn * bk * v];
        for (_l_jj = 0; _l_jj < bk; _l_jj++) {
          for (_l_ii = 0; _l_ii < bn; _l_ii++) {
            for (_l_v = 0; _l_v < v; _l_v++) {
              tmp[(_l_jj * v + _l_v) * bn + _l_ii] = LIBXSMM_VLA_ACCESS(
                  5, l_wt_dense, l_i, _l_j, _l_jj, _l_ii, _l_v, Kb, bk, bn, v);
            }
          }
        }
        for (_l_cur = 0; _l_cur < bn * bk * v; _l_cur += 16) {
          unsigned short mask = (unsigned short)0;
          int l_b = 0;
          for (l_b = 0; l_b < 16; l_b++) {
            DType cur_data = tmp[_l_cur + l_b];
            if ((cur_data != 0) && (cur_data != -0)) {
              mask |= (unsigned short)((unsigned short)1 << l_b);
              compressed_weight[cur_count] = cur_data;
              cur_count++;
            }
          }
          bitmap_array[_l_bm + l_i * (K * bn) / 16] = mask;
          _l_bm++;
        }
      }
    }
  }

  compressed_wt->data = NULL; //(void*)compressed_weight;
  compressed_wt->bitmap = (char*)bitmap;
  compressed_wt->column_offsets = NULL; // (long long*)column_offsets;
  compressed_wt->data_ptrs = (void*)data_ptrs;
  compressed_wt->nnz = nnz;
  compressed_wt->n_dense_elts = N * K;
  compressed_wt->sizes[0] = Nb;
  compressed_wt->sizes[1] = Kb;
  compressed_wt->sizes[2] = bk;
  compressed_wt->sizes[3] = bn;
  compressed_wt->sizes[4] = v;
}

template <typename DType>
void create_bcsc_from_blocked_weight_tensor(
    DType* dense_wt_ptr,
    int Nb,
    int Kb,
    int bk,
    int bn,
    int v,
    int bcsc_bk,
    int bcsc_bn,
    int N_target_blocks,
    tensor_bcsc_t* sparse_wt) {
  int nnz = 0, l_zero_block = 0;
  int N = Nb * bn;
  int K = Kb * bk * v;
  unsigned int l_val_idx = 0;
  unsigned int l_nz_block_id = 0;
  unsigned int* l_colptr;
  unsigned int* l_rowidx;
  DType *l_data, val;
  unsigned int* Nblocks_offsets = (unsigned int*)libxsmm_aligned_malloc(
      sizeof(unsigned int) * N_target_blocks, 64);
  int l_i, l_j, l_di, l_dj, l_ui, l_uj, l_ii, l_jj, total_nnz_processed,
      nnz_entries_per_block, all_done = 0, n_blocks = 0;
  LIBXSMM_VLA_DECL(5, DType, l_wt_dense, dense_wt_ptr, Kb, bk, bn, v);
  int l_blocking_step = (bn + bcsc_bn - 1) / bcsc_bn;
  auto sparse_pct = env2int("SPFRAC", 0);
  double sparse_frac = (double)(1.0 * sparse_pct / 100.0);

  if (bn % bcsc_bn != 0) {
    printf(
        "WARNING: end up with assymetric logical blocks because bn is %d and bcsc_bn is %d...\n",
        bn,
        bcsc_bn);
  }

  /* Sparsify tensors for benchmarking purposes */
  if (sparse_pct > 0) {
    sparsify_weight_tensor<DType>(
        dense_wt_ptr, Nb, Kb, bk, bn, v, bcsc_bk, bcsc_bn, sparse_frac);
  }

  /* First pass to count number of non-zeros */
  for (l_i = 0; l_i < N / bcsc_bn; l_i++) {
    for (l_j = 0; l_j < K / bcsc_bk; l_j++) {
      l_ui = l_i * bcsc_bn;
      l_uj = l_j * bcsc_bk;
      l_zero_block = 1;
      /* Scan block to see if there are non-zeros */
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_ii = l_ui + l_di;
          l_jj = l_uj + l_dj;
          val = LIBXSMM_VLA_ACCESS(
              5,
              l_wt_dense,
              l_ii / bn,
              l_jj / (bk * v),
              (l_jj % (bk * v)) / v,
              l_ii % bn,
              (l_jj % (bk * v)) % v,
              Kb,
              bk,
              bn,
              v);
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
  if (sparse_wt->data)
    libxsmm_free(sparse_wt->data);
  if (sparse_wt->colptr)
    libxsmm_free(sparse_wt->colptr);
  if (sparse_wt->rowidx)
    libxsmm_free(sparse_wt->rowidx);
  if (sparse_wt->Nblocks_offsets)
    libxsmm_free(sparse_wt->Nblocks_offsets);

  l_colptr = (unsigned int*)libxsmm_aligned_malloc(
      (N / bcsc_bn + 1) * sizeof(unsigned int), 64);
  l_rowidx = (unsigned int*)libxsmm_aligned_malloc(
      nnz / (bcsc_bk * bcsc_bn) * sizeof(unsigned int), 64);
  l_data = (DType*)libxsmm_aligned_malloc(nnz * sizeof(DType), 64);

#pragma omp parallel for
  for (l_i = 0; l_i < nnz; l_i++) {
    l_data[l_i] = (DType)1;
  }

  /* Second pass to create BCSC */
  l_nz_block_id = 0;
  l_colptr[N / bcsc_bn] = nnz / (bcsc_bk * bcsc_bn);
  for (l_i = 0; l_i < N / bcsc_bn; l_i++) {
    l_colptr[l_i] = l_nz_block_id;
    for (l_j = 0; l_j < K / bcsc_bk; l_j++) {
      l_ui = l_i * bcsc_bn;
      l_uj = l_j * bcsc_bk;
      l_zero_block = 1;
      /* Scan block to see if there are non-zeros */
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_ii = l_ui + l_di;
          l_jj = l_uj + l_dj;
          val = LIBXSMM_VLA_ACCESS(
              5,
              l_wt_dense,
              l_ii / bn,
              l_jj / (bk * v),
              (l_jj % (bk * v)) / v,
              l_ii % bn,
              (l_jj % (bk * v)) % v,
              Kb,
              bk,
              bn,
              v);
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
            val = LIBXSMM_VLA_ACCESS(
                5,
                l_wt_dense,
                l_ii / bn,
                l_jj / (bk * v),
                (l_jj % (bk * v)) / v,
                l_ii % bn,
                (l_jj % (bk * v)) % v,
                Kb,
                bk,
                bn,
                v);
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
    for (l_i = 0; l_i < nnz / (bcsc_bk * bcsc_bn); l_i++) {
      DType tmp_block[bcsc_bk * bcsc_bn];
      memcpy(
          tmp_block,
          &l_data[l_i * (bcsc_bk * bcsc_bn)],
          (bcsc_bk * bcsc_bn) * sizeof(DType));
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_data
              [l_i * (bcsc_bk * bcsc_bn) + (l_dj / v) * (bcsc_bn * v) +
               l_di * v + l_dj % v] = tmp_block[l_di * bcsc_bk + l_dj];
        }
      }
    }
  }

  /* Logically partition the sparse BCSC matrix  */
  total_nnz_processed = 0;
  nnz_entries_per_block = (nnz + N_target_blocks - 1) / N_target_blocks;
  all_done = 0;
  l_i = 0;
  l_j = 0;
  Nblocks_offsets[0] = 0;
  while (all_done == 0) {
    int nnz_so_far = 0;
    while ((nnz_so_far < nnz_entries_per_block) && (l_j < N / bcsc_bn)) {
      if (l_j + l_blocking_step <= N / bcsc_bn) {
        nnz_so_far += (l_colptr[l_j + l_blocking_step] - l_colptr[l_j]) *
            bcsc_bk * bcsc_bn;
        l_j += l_blocking_step;
      } else {
        /* Should not happen  */
      }
    }
    total_nnz_processed += nnz_so_far;
    l_i++;
    if (total_nnz_processed < nnz) {
      Nblocks_offsets[l_i] = l_j * bcsc_bn;
      if (l_j >= N / bcsc_bn) {
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
  sparse_wt->n_dense_elts = N * K;
  sparse_wt->bcsc_bn = bcsc_bn;
  sparse_wt->bcsc_bk = bcsc_bk;
  sparse_wt->bcsc_blocks_in_bn = l_blocking_step;
  sparse_wt->sizes[0] = Nb;
  sparse_wt->sizes[1] = Kb;
  sparse_wt->sizes[2] = bk;
  sparse_wt->sizes[3] = bn;
  sparse_wt->sizes[4] = v;
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

inline at::Tensor act_tensor_trans_n2v_flat(
    long N,
    long C,
    long bn,
    long bc,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(bc % BS == 0, "Uneven number for S2\n");
  long Nup = ((N + bn - 1) / bn) * bn;
  long Nblocks = Nup / bn;
  auto output = input.new_empty({Nblocks, C / bc, bc / BS, bn, BS});
  auto rem = (N % bn == 0) ? bn : (N % bn);

  auto out = GetVLAPtr<bfloat16>(output, {C / bc, bn * bc});
  auto in = GetVLAPtr<bfloat16>(input, {C / bc, bc});

  auto xpose_n2v_tpp = SCOPEIT(
      XformExtTPP<bfloat16>(
          bn, bc, bc, bn, C, bn, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      VNNI);
  auto xpose_n2v_tpp_rem = SCOPEIT(
      XformExtTPP<bfloat16>(
          rem, bc, bc, rem, C, bn, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      VNNI);

  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int nc = 0; nc < C / bc; nc++) {
      for (int nb = 0; nb < Nblocks; nb++) {
        if (nb == Nblocks - 1) {
          xpose_n2v_tpp_rem(in[nb * bn][nc], out[nb][nc]);
        } else {
          xpose_n2v_tpp(in[nb * bn][nc], out[nb][nc]);
        }
      }
    }
  }
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
