/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/sddmm.h
 * \brief SDDMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SDDMM_BLOCKING_LIBXSMM_H_
#define DGL_ARRAY_CPU_SDDMM_BLOCKING_LIBXSMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include "../selector.h"
#ifdef USE_LIBXSMM
#include <libxsmm_source.h>
#include <unistd.h>
#include "xsmm_functors.h"

/*!
 * \brief CPU kernel of g-SDDMM on Csr format using LIBXSMM.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op,
          int LhsTarget = 0, int RhsTarget = 2>
void SDDMMCsrLibxsmm(const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray lhs, NDArray rhs, NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  IdType* indices = csr.indices.Ptr<IdType>();
  IdType* edges = csr.data.Ptr<IdType>();
  DType* X = lhs.Ptr<DType>();
  DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len,
                reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();

  if(std::is_same<Op, dgl::aten::cpu::op::Add<DType>>::value) {
    auto add_tpp = dgl_tpp::AddTPP<DType>(1, dim);
#pragma omp parallel for
    for(IdType i = 0; i < csr.num_rows; i++) {
      IdType row_start = indptr[i];
      IdType row_end = indptr[i+1];

      auto n = row_end - row_start;

      IdType *eid = has_idx ? &edges[row_start] : &row_start;
      IdType *cid = &indices[row_start];

      for(auto r=0; r<n; r++) {
	add_tpp(&X[cid[r]*dim], &Y[cid[r]*dim], &O[eid[r]*dim]);
      }
    }
  }
  else if(std::is_same<Op, dgl::aten::cpu::op::Sub<DType>>::value) {
    auto sub_tpp = dgl_tpp::SubTPP<DType>(1, dim);
#pragma omp parallel for
    for(IdType i = 0; i < csr.num_rows; i++) {
      IdType row_start = indptr[i];
      IdType row_end = indptr[i+1];

      auto n = row_end - row_start;

      IdType *eid = has_idx ? &edges[row_start] : &row_start;
      IdType *cid = &indices[row_start];

      for(auto r=0; r<n; r++) {
	sub_tpp(&X[cid[r]*dim], &Y[cid[r]*dim], &O[eid[r]*dim]);
      }
    }
  }
  else if(std::is_same<Op, dgl::aten::cpu::op::Mul<DType>>::value) {
    auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
#pragma omp parallel for
    for(IdType i = 0; i < csr.num_rows; i++) {
      IdType row_start = indptr[i];
      IdType row_end = indptr[i+1];

      auto n = row_end - row_start;

      IdType *eid = has_idx ? &edges[row_start] : &row_start;
      IdType *cid = &indices[row_start];

      for(auto r=0; r<n; r++) {
	mul_tpp(&X[cid[r]*dim], &Y[cid[r]*dim], &O[eid[r]*dim]);
      }
    }
  }
  else if(std::is_same<Op, dgl::aten::cpu::op::Div<DType>>::value) {
    auto div_tpp = dgl_tpp::DivTPP<DType>(1, dim);
#pragma omp parallel for
    for(IdType i = 0; i < csr.num_rows; i++) {
      IdType row_start = indptr[i];
      IdType row_end = indptr[i+1];

      auto n = row_end - row_start;

      IdType *eid = has_idx ? &edges[row_start] : &row_start;
      IdType *cid = &indices[row_start];

      for(auto r=0; r<n; r++) {
	div_tpp(&X[cid[r]*dim], &Y[cid[r]*dim], &O[eid[r]*dim]);
      }
    }
  }
  else if(std::is_same<Op, dgl::aten::cpu::op::Dot<DType>>::value) {
    auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
    auto reduce_add_tpp = dgl_tpp::ReduceAddColsTPP<DType>(dim, reduce_size);
#pragma omp parallel for
    for(IdType i = 0; i < csr.num_rows; i++) {
      IdType row_start = indptr[i];
      IdType row_end = indptr[i+1];

      auto n = row_end - row_start;

      IdType *eid = has_idx ? &edges[row_start] : &row_start;
      IdType *cid = &indices[row_start];
      DType evec[lhs_dim];

      for(auto r=0; r<n; r++) {
	mul_tpp(&X[cid[r]*lhs_dim], &Y[cid[r]*rhs_dim], evec);
	reduce_add_tpp(evec, &O[eid[r]*dim]);
      }
    }
  }
}

/*!
 * \brief CPU kernel of g-SDDMM on Coo format using LIBXSMM.
 * \param bcast Broadcast information.
 * \param coo The COO matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 * \note it uses edge parallel strategy, different threads are responsible
 *       for the computation of different edges.
 */
template <typename IdType, typename DType, typename Op,
          int LhsTarget = 0, int RhsTarget = 2>
void SDDMMCooLibxsmm(const BcastOff& bcast,
              const COOMatrix& coo,
              NDArray lhs, NDArray rhs, NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  IdType* row = coo.row.Ptr<IdType>();
  IdType* col = coo.col.Ptr<IdType>();
  IdType* edges = has_idx ? coo.data.Ptr<IdType>() : NULL;
  DType* X = lhs.Ptr<DType>();
  DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len,
                reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();

  auto N = coo.row->shape[0];

  if(edges == NULL) {
    if(std::is_same<Op, dgl::aten::cpu::op::Add<DType>>::value) {
      auto add_tpp = dgl_tpp::AddTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	add_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[n*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Sub<DType>>::value) {
      auto sub_tpp = dgl_tpp::SubTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	sub_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[n*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Mul<DType>>::value) {
      auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	mul_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[n*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Div<DType>>::value) {
      auto div_tpp = dgl_tpp::DivTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	div_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[n*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Dot<DType>>::value){
      auto mul_tpp = dgl_tpp::MulTPP<DType>(1, lhs_dim);
      auto reduce_add_tpp = dgl_tpp::ReduceAddColsTPP<DType>(dim, reduce_size);

#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	DType fvec[lhs_dim];
	mul_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], fvec);
	reduce_add_tpp(fvec, &O[n*dim]);
      }
    }
  }
  else {

    if(std::is_same<Op, dgl::aten::cpu::op::Add<DType>>::value) {
      auto add_tpp = dgl_tpp::AddTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	add_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[edges[n]*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Sub<DType>>::value) {
      auto sub_tpp = dgl_tpp::SubTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	sub_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[edges[n]*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Mul<DType>>::value) {
      auto mul_tpp = dgl_tpp::MulTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	mul_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[edges[n]*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Div<DType>>::value) {
      auto div_tpp = dgl_tpp::DivTPP<DType>(1, dim);
#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	div_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], &O[edges[n]*dim]);
      }
    }
    else if(std::is_same<Op, dgl::aten::cpu::op::Dot<DType>>::value) {
      auto mul_tpp = dgl_tpp::MulTPP<DType>(1, lhs_dim);
      auto reduce_add_tpp = dgl_tpp::ReduceAddColsTPP<DType>(dim, reduce_size);

#pragma omp parallel for
      for(auto n = 0; n < N; n++) {
	DType fvec[lhs_dim];
	mul_tpp(&X[row[n]*lhs_dim], &Y[col[n]*rhs_dim], fvec);
	reduce_add_tpp(fvec, &O[edges[n]*dim]);
      }
    }
  }
}

#endif
#endif  // DGL_ARRAY_CPU_SDDMM_BLOCKING_LIBXSMM_H_
