/*!
 *  Copyright (c) 2021 Intel Corporation
 * \file array/cpu/spmm.h
 * \brief SPMM CPU kernel function header.
 * \author Sanchit Misra <sanchit.misra@intel.com>,
 *         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
 *         Vasimuddin Md <vasimuddin.md@intel.com>,
 *         Sasikanth Avancha <sasikanth.avancha@intel.com>
 */
#ifndef DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_
#define DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dmlc/logging.h>
#include <algorithm>

#if !defined(_WIN32)
#ifdef USE_LIBXSMM
#include <libxsmm_source.h>
#include <unistd.h>
#include <libxsmm.h>
#include "xsmm_functors.h"
#ifdef DEBUG
#include <x86intrin.h>
#endif  // DEBUG
#include <dmlc/omp.h>

#define NUM_BLOCKS_PER_THREAD 20
#define BLOCKING_HEURISTIC_PARAM 500
#define MIN_WORK_HEURISTIC_PARAM 1000000
#define MAX_COL_BLOCK_HI 64 
#define MAX_COL_BLOCK_LO 32 

namespace dgl {
namespace aten {
namespace cpu {

template <typename IdType, typename DType>
struct CSRMatrixInternal {
  IdType num_rows;
  IdType num_cols;
  IdType *indptr;
  IdType *indices;
  DType *data;
};

int32_t GetLLCSize() {
  int32_t cache_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
  if (cache_size < 0) cache_size = DGL_CPU_LLC_SIZE;
  return cache_size;
}

/*!
 * \brief Tile the CSR matrix to roughly make sure that the column tiles and
 *        corresponding neighbor features fit into LLC and the row tiles
 *        are assigned to OMP threads.
 * \param csr The Csr matrix.
 * \param block_csr_array The array containing csr matrices of all blocks.
 * \param num_M_blocks Number of blocks to create along the rows of adjacency matrix.
 * \param num_K_blocks Number of blocks to create along the columns of adjacency matrix.
 * \param M_block_size block size along the rows of adjacency matrix.
 * \param K_block_size block size along the columns of adjacency matrix.
 * \param use_lhs Whether to use lhs.
 * \param use_rhs Whether to use rhs.
 */
template <typename IdType>
inline void SpMMCreateBlocks(
    const CSRMatrix& csr,
    CSRMatrixInternal<IdType, IdType> *block_csr_array,
    IdType num_M_blocks,
    IdType num_K_blocks,
    IdType M_block_size,
    IdType K_block_size,
    bool use_lhs, bool use_rhs) {

  const IdType M = csr.num_rows;
  const IdType K = csr.num_cols;
  IdType* indptr = csr.indptr.Ptr<IdType>();
  IdType* indices = csr.indices.Ptr<IdType>();
  IdType* edges = csr.data.Ptr<IdType>();
  CHECK_NOTNULL(indptr);
  if (use_lhs)
    CHECK_NOTNULL(indices);
  if (use_rhs)
    CHECK_NOTNULL(edges);

  if (num_K_blocks > 1) {
    IdType *indptr_block_buf = reinterpret_cast<IdType *>(aligned_alloc(64,
                                                             (M_block_size + 1) * num_M_blocks *
                                                             num_K_blocks * sizeof(IdType)));
    IdType *indices_block_buf = NULL;
	if(use_lhs)
	  indices_block_buf = reinterpret_cast<IdType *>(aligned_alloc(64,
                                                              indptr[M] * sizeof(IdType)));
    IdType *edges_block_buf = NULL;
	if(use_rhs)
	  edges_block_buf = reinterpret_cast<IdType *>(aligned_alloc(64,
                                                            indptr[M] * sizeof(IdType)));

#pragma omp parallel
    {
      IdType *my_cur_col_id = reinterpret_cast<IdType *>(aligned_alloc(64, 2 * M_block_size *
                                                                          sizeof(IdType)));

#pragma omp for
      for (IdType m = 0; m < num_M_blocks; m++) {
        const IdType M_start = m * M_block_size;
        const IdType M_end = std::min((m + 1) * M_block_size, M);
        const IdType nnz = indptr[M_end] - indptr[M_start];

        IdType cur_indices_id = 0;
        IdType *my_indices_block_buf, *my_edges_block_buf;
        if (use_lhs)
          my_indices_block_buf = indices_block_buf + indptr[M_start];
        if (use_rhs)
          my_edges_block_buf = edges_block_buf + indptr[M_start];

        for (IdType i = M_start; i < M_end; i++) {
          my_cur_col_id[(i - M_start) * 2] = indptr[i];
          my_cur_col_id[(i - M_start) * 2 + 1] = indptr[i + 1];
        }
        for (IdType k = 0; k < num_K_blocks; k++) {
          const IdType K_start = k * K_block_size;
          const IdType K_end = std::min((k + 1) * K_block_size, K);
          CSRMatrixInternal<IdType, IdType> cur_csr;
          cur_csr.num_rows = M_end - M_start;
          cur_csr.num_cols = K_end - K_start;
          // Create csr_ij
          IdType *cur_csr_indptr = indptr_block_buf + (m * num_K_blocks + k) * (M_block_size + 1);
          IdType *cur_csr_indices = nullptr, *cur_csr_edges = nullptr;
          if (use_lhs)
            cur_csr_indices = my_indices_block_buf + cur_indices_id;
          if (use_rhs)
            cur_csr_edges = my_edges_block_buf + cur_indices_id;
          IdType cur_nnz = 0;
          for (IdType i = M_start; i < M_end; i++) {
            const IdType row_start = my_cur_col_id[(i - M_start) * 2];
            const IdType row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
            cur_csr_indptr[i - M_start] = cur_nnz;
            IdType eid;
            for (eid = row_start; eid < row_end; eid++) {
              const IdType src = indices[eid];
              const IdType edge = edges[eid];
              if (src >= K_end) {
                break;
              }
              CHECK_LT(cur_indices_id + cur_nnz, nnz);
              if (use_lhs)
                cur_csr_indices[cur_nnz] = src;
              if (use_rhs)
                cur_csr_edges[cur_nnz] = edge;
              cur_nnz++;
            }
            my_cur_col_id[(i - M_start) * 2] = eid;
          }
          cur_csr_indptr[cur_csr.num_rows] = cur_nnz;
          cur_indices_id += cur_nnz;
          cur_csr.indptr = cur_csr_indptr;
          if (use_lhs)
            cur_csr.indices = cur_csr_indices;
          if (use_rhs)
            cur_csr.data = cur_csr_edges;
          block_csr_array[m * num_K_blocks + k] = cur_csr;
        }
        CHECK_EQ(nnz, cur_indices_id);
      }
      free(my_cur_col_id);
    }
  } else {
#pragma omp for
    for (IdType m = 0; m < num_M_blocks; m++) {
      const IdType M_start = m * M_block_size;
      const IdType M_end = std::min((m + 1) * M_block_size, M);

      CSRMatrixInternal<IdType, IdType> cur_csr;
      cur_csr.num_rows = M_end - M_start;
      cur_csr.num_cols = K;
      cur_csr.indptr = indptr + M_start;
      cur_csr.indices = indices;
      cur_csr.data = edges;

      block_csr_array[m] = cur_csr;
    }
  }
}

/*!
 * \brief Free the tiled CSR matrix data.
 * \param block_csr_array The array containing csr matrices of all blocks.
 * \param num_M_blocks Number of blocks to create along the rows of adjacency matrix.
 * \param num_K_blocks Number of blocks to create along the columns of adjacency matrix.
 * \param use_lhs Whether to use lhs.
 * \param use_rhs Whether to use rhs.
 */
template <typename IdType>
inline void SpMMFreeBlocks(
    CSRMatrixInternal<IdType, IdType> *block_csr_array,
    IdType num_M_blocks, IdType num_K_blocks,
    bool use_lhs, bool use_rhs) {

  if (num_K_blocks > 1) {
    free(block_csr_array[0].indptr);
    if (use_lhs)
      free(block_csr_array[0].indices);
    if (use_rhs)
      free(block_csr_array[0].data);
  }
  free(block_csr_array);
}

/*!
 * \brief Optimized CPU kernel of SpMM-Sum/Max/Min on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes.
 * \param arge Arg-Min/Max on edges.
 * \note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op, typename Redop>
inline int SpMMRedopCsrOpt(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out,
    NDArray argu, NDArray arge) {

  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  CHECK_NOTNULL(indptr);
  const IdType total_nnz = indptr[M];
  if (M <= 0 || K <= 0 || N <= 0 || total_nnz <= 0) return 0;

  int32_t llc_size = GetLLCSize();

#ifdef DEBUG
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif  // DEBUG

  const bool has_idx = !IsNullArray(csr.data);

  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();
  DType* E = efeat.Ptr<DType>();

  IdType *argB=NULL, *argE=NULL;
  if (std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value) {
    argB = argu.Ptr<IdType>();
    argE = arge.Ptr<IdType>();
  }

  const int nthreads = omp_get_max_threads();
  int64_t dim = N;
  int64_t lhs_dim = Op::use_rhs ? bcast.lhs_len / bcast.rhs_len : bcast.lhs_len;
  int64_t rhs_dim = bcast.rhs_len;

  const double avg_degree = total_nnz * 1.0 / M;
  const double nnz_prob = avg_degree / K;

  IdType K_block_size = std::min((int64_t)K, (int64_t)(llc_size / (N * sizeof(DType) *
                                                       nnz_prob * BLOCKING_HEURISTIC_PARAM)));
  IdType M_block_size = M / (nthreads * NUM_BLOCKS_PER_THREAD);
  if (M_block_size == 0) M_block_size = 1;
  if (K_block_size == 0) K_block_size = 1;

  IdType num_M_blocks = (M + M_block_size - 1) / M_block_size;
  IdType num_K_blocks = (K + K_block_size - 1) / K_block_size;

#ifdef DEBUG
  if (std::is_same<Redop, op::Max<DType>>::value) {
    LOG(INFO) << "Redop = Max";
  } else if (std::is_same<Redop, op::Min<DType>>::value) {
    LOG(INFO) << "Redop = Min";
  } else if (std::is_same<Redop, op::Add<DType>>::value) {
    LOG(INFO) << "Redop = Add";
  }
  LOG(INFO) << "nthreads = " << nthreads << ", llc_size = " << llc_size;
  LOG(INFO) << "M = " << M << ", K = " << K << ", N = " << N;
  LOG(INFO) << "use_lhs = " << Op::use_lhs << ", use_rhs = " << Op::use_rhs;
  LOG(INFO) << "total_nnz = " << total_nnz << ", avg_degree = " << avg_degree;
  LOG(INFO) << "has_idx = " << has_idx;
  LOG(INFO) << "nnz_prob = " << nnz_prob;
  LOG(INFO) << "K_block_size = " << K_block_size << ", M_block_size = " << M_block_size;
  LOG(INFO) << "num_K_blocks = " << num_K_blocks << ", num_M_blocks = " << num_M_blocks;
  LOG(INFO) << "lhs_dim = " << lhs_dim << ", rhs_dim = " << rhs_dim << ", dim = " << dim;
#endif  // DEBUG

  auto ef_bcast_row_tpp = dgl_tpp::CpyBcastRowTPP<DType>(rhs_dim, lhs_dim);
  auto set_zero_tpp = dgl_tpp::SetZeroTPP<DType>(1, dim);

  auto copy_tpp = dgl_tpp::CpyTPP<DType>(1, dim);

  CSRMatrixInternal<IdType, IdType> *block_csr_array =
    (CSRMatrixInternal<IdType, IdType> *)aligned_alloc(64,
	sizeof(CSRMatrixInternal<IdType, IdType>) * num_M_blocks * num_K_blocks);

  SpMMCreateBlocks(csr, block_csr_array, num_M_blocks, num_K_blocks, M_block_size, K_block_size,
      Op::use_lhs, Op::use_rhs);

  // Op is CopyLhs 
  if(std::is_same<Op, op::CopyLhs<DType>>::value) {
    DType (*ufeat_mat)[dim] = (DType (*)[dim])B;

    // Reduction op is either Max or Min
    if(std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value) {
      auto reduce_max_idx_tpp = dgl_tpp::ReduceMaxRowsIdxTPP<DType, IdType>(dim);
      auto reduce_min_idx_tpp = dgl_tpp::ReduceMinRowsIdxTPP<DType, IdType>(dim);
#pragma omp parallel
      {
	DType (*ofeat_mat)[dim] = (DType (*)[dim])C;
	IdType (*argB_mat)[dim] = (IdType (*)[dim])argB;

	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;

	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType dst = i + M_start;
	      IdType *cid = &cur_csr.indices[row_start];
	      auto n = row_end - row_start;

	      if(std::is_same<Redop, op::Max<DType>>::value)
		reduce_max_idx_tpp(ufeat_mat[0], cid, n, ofeat_mat[dst], argB_mat[dst]);
	      else {
		reduce_min_idx_tpp(ufeat_mat[0], cid, n, ofeat_mat[dst], argB_mat[dst]);
	      }
	    }
	  }
	}
      }
    }
    else { // Reduction op is other than Max or Min
      auto reduce_add_idx_tpp = dgl_tpp::ReduceAddRowsIdxTPP<DType, IdType>(dim);
#pragma omp parallel
      {
	DType (*ofeat_mat)[dim] = (DType (*)[dim])C;

	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;

	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType *cid = &cur_csr.indices[row_start];
	      IdType dst = i + M_start;
	      auto n = row_end - row_start;

              reduce_add_idx_tpp(ufeat_mat[0], cid, n, ofeat_mat[dst]);
	    }
	  }
	}
      }
    }
  }
  // Op is CopyRhs
  else if(std::is_same<Op, op::CopyRhs<DType>>::value) {
    DType (*efeat_mat)[dim] = (DType (*)[dim])E;

    if(std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value) {
      auto reduce_max_idx_tpp = dgl_tpp::ReduceMaxRowsIdxTPP<DType, IdType>(dim);
      auto reduce_min_idx_tpp = dgl_tpp::ReduceMinRowsIdxTPP<DType, IdType>(dim);

#pragma omp parallel
      {
	IdType (*argE_mat)[rhs_dim] = (IdType (*)[rhs_dim])argE;
	DType (*ofeat_mat)[dim] = (DType (*)[dim])C;

	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;


	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType dst = i + M_start;
	      IdType *eid = has_idx ? &cur_csr.data[row_start] : &row_start;
	      auto n = row_end - row_start;


              if(std::is_same<Redop, op::Max<DType>>::value)
		reduce_max_idx_tpp(efeat_mat[0], eid, n, ofeat_mat[dst], argE_mat[dst]);
              else {
		reduce_min_idx_tpp(efeat_mat[0], eid, n, ofeat_mat[dst], argE_mat[dst]);
              }
	    }
	  }
	}
      }
    }
    else {
      auto reduce_add_idx_tpp = dgl_tpp::ReduceAddRowsIdxTPP<DType, IdType>(dim);
#pragma omp parallel
      {
	DType (*ofeat_mat)[dim] = (DType (*)[dim])C;

	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;


	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType *eid = has_idx ? &cur_csr.data[row_start] : &row_start;
	      IdType dst = i + M_start;
	      auto n = row_end - row_start;

              reduce_add_idx_tpp(efeat_mat[0], eid, n, ofeat_mat[dst]);
	    }
	  }
	}
      }
    }
  }
  else { //Binary + Unary
    if(std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value) {
      auto max_tpp = dgl_tpp::MaxTPP<DType>(1, dim);
      auto min_tpp = dgl_tpp::MinTPP<DType>(1, dim);
#pragma omp parallel
      {
	IdType (*argB_mat)[dim]     = (IdType (*)[dim])argB;
	IdType (*argE_mat)[dim] = (IdType (*)[dim])argE;

	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;
	    DType etmp[rhs_dim][lhs_dim];
	    DType oprev[dim]; 

	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType *eid = has_idx ? &cur_csr.data[row_start] : &row_start;
	      IdType *cid = &cur_csr.indices[row_start];
	      IdType dst = i + M_start;
	      auto n = row_end - row_start;

	      ef_bcast_row_tpp(&E[eid[0]*rhs_dim], etmp[0]);
              if(std::is_same<Redop, op::Max<DType>>::value)
                max_tpp(&B[cid[0]*dim], etmp[0], &C[dst*dim]);
              else
                min_tpp(&B[cid[0]*dim], etmp[0], &C[dst*dim]);

#pragma omp simd
              for(auto x=0; x<dim; x++) {
                argB_mat[dst][x] = cid[0];
                argE_mat[dst][x] = eid[0];
              }
              for(auto r=1; r<n; r++) {
                copy_tpp(&C[dst*dim], oprev);
                ef_bcast_row_tpp(&E[eid[r]*rhs_dim], etmp[0]);
                if(std::is_same<Redop, op::Max<DType>>::value)
                  max_tpp(&B[cid[0]*dim], etmp[0], &C[dst*dim]);
                else
                  min_tpp(&B[cid[0]*dim], etmp[0], &C[dst*dim]);
#pragma omp simd
                for(auto x=0; x<dim; x++) {
                  if(C[dst*dim+x] != oprev[x]) {
                    argB_mat[dst][x] = cid[r];
                    argE_mat[dst][x] = eid[r];
                  }
                }
              }
	    }
	  }
	}
      }
    }
    else {
      auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
      auto add_tpp = dgl_tpp::AddTPP<DType>(1, dim);
#pragma omp parallel
      {
	for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
	  for (IdType m = 0; m < num_M_blocks; m++) {
	    CSRMatrixInternal<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];
	    IdType M_start = m * M_block_size;

	    DType evec[dim], ovec[dim];
	    for (IdType i = 0; i < cur_csr.num_rows; i++) {
	      IdType row_start = cur_csr.indptr[i];
	      IdType row_end   = cur_csr.indptr[i + 1];
	      IdType *eid = has_idx ? &cur_csr.data[row_start] : &row_start;
	      IdType *cid = &cur_csr.indices[row_start];
	      IdType dst = i + M_start;
	      auto n = row_end - row_start;

              if(k==0)
                set_zero_tpp(&C[dst*dim]);
              for(auto r=0; r<n; r++) {
                ef_bcast_row_tpp(&E[eid[r]*rhs_dim], evec);
                mul_tpp(&B[cid[r]*dim], evec, ovec);
                add_tpp(&C[dst*dim], ovec, &C[dst*dim]);
              }
	    }
	  }
	}
      }
    }
  }
  SpMMFreeBlocks(block_csr_array, num_M_blocks, num_K_blocks, Op::use_lhs, Op::use_rhs);

  return 0;
}

/*!
 * \brief Optimized CPU kernel of SpMM-Sum on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op>
// void SpMMSumCsrLibxsmm(const BcastOff& bcast, const CSRMatrix& csr,
int SpMMSumCsrLibxsmm(const BcastOff& bcast, const CSRMatrix& csr,
                   NDArray ufeat, NDArray efeat, NDArray out) {
  NDArray dummy;
  return SpMMRedopCsrOpt<IdType, DType, Op, op::Add<DType>>(bcast, csr, ufeat, efeat, out, dummy, dummy);
}

/*!
 * \brief Optimized CPU kernel of SpMM-Min/Max on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes.
 * \param arge Arg-Min/Max on edges.
 * \note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
// void SpMMCmpCsrLibxsmm(const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat,
int SpMMCmpCsrLibxsmm(const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat,
                   NDArray efeat, NDArray out, NDArray argu, NDArray arge) {
  return SpMMRedopCsrOpt<IdType, DType, Op, Cmp>(bcast, csr, ufeat, efeat, out, argu, arge);
}

#if 1
template <typename IdType, typename DType, typename Op>
void Edge_softmax_csr_forward_libxsmm(const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat,
    NDArray efeat, NDArray out) {

  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  IdType* edges =
    has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const int64_t dim = bcast.out_len;
  DType* Eo = out.Ptr<DType>();
  DType* Ei = static_cast<DType*>(efeat->data);

  auto red_max_idx_tpp = dgl_tpp::ReduceMaxRowsIdxTPP<DType, IdType>(dim);
  auto max_tpp = dgl_tpp::MaxTPP<DType>(1, dim);
  auto recp_tpp = dgl_tpp::RecpTPP<DType>(1, dim);
  auto sub_tpp = dgl_tpp::SubTPP<DType, DType>(1, dim);
  auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
  auto exp_tpp = dgl_tpp::ExpTPP<DType, DType>(1, dim);
  auto add_tpp = dgl_tpp::AddTPP<DType>(1, dim);
  auto set_zero_tpp = dgl_tpp::SetZeroTPP<DType>(1, dim);

#pragma omp parallel for
  for(auto rid = 0; rid < csr.num_rows; rid++) {
    IdType row_start = indptr[rid];
    IdType row_end = indptr[rid+1];
    auto n = row_end - row_start;
    IdType* eid = has_idx ? &edges[row_start] : &row_start;
    DType emax[dim], esum[dim], eexp[n][dim];
    IdType arge[dim];

    red_max_idx_tpp(Ei, eid, n, emax, arge);

    set_zero_tpp(esum);
    for(auto r=0; r<n; r++) {
      sub_tpp(&Ei[eid[r]*dim], emax, eexp[r]);
      exp_tpp(eexp[r], eexp[r]);
      add_tpp(eexp[r], esum, esum);
    }
    recp_tpp(esum, esum);
    for(auto r=0; r<n; r++) {
      mul_tpp(eexp[r], esum, &Eo[eid[r]*dim]);
    }
  }
}

template <typename IdType, typename DType, typename Op>
void Edge_softmax_csr_backward_libxsmm(const BcastOff& bcast, const CSRMatrix& csr, NDArray out,
                NDArray sds, NDArray back_out) {

  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  IdType* edges =
    has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const int64_t dim = bcast.out_len;
  DType* W_out = static_cast<DType*>(out->data);
  DType* W_sds =  static_cast<DType*>(sds->data);
  DType* O = back_out.Ptr<DType>();

  auto red_add_idx_tpp = dgl_tpp::ReduceAddRowsIdxTPP<DType, IdType>(dim);
  auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
  auto sub_tpp = dgl_tpp::SubTPP<DType>(1, dim);
  auto set_zero_tpp = dgl_tpp::SetZeroTPP<DType>(1, dim);

#pragma omp parallel for
  for(auto rid = 0; rid < csr.num_rows; rid++) {
    IdType row_start = indptr[rid];
    IdType row_end = indptr[rid+1];
    auto n = row_end - row_start;
    IdType* eid = has_idx ? &edges[row_start] : &row_start;
    DType esum[dim], emul[dim];

    set_zero_tpp(esum);
    red_add_idx_tpp(W_sds, eid, n, esum);

    for(auto r=0; r<n; r++) {
      mul_tpp(&W_out[eid[r]*dim], esum, emul);
      sub_tpp(&W_sds[eid[r]*dim], emul, &O[eid[r]*dim]);
    }
  }
}
#endif

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // USE_LIBXSMM
#endif  // _WIN32

#endif  // DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_
