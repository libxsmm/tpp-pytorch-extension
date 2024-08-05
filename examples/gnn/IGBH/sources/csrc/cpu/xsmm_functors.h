/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <dgl/runtime/bfloat16.h>
#include <string>
#include <unordered_map>

#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))

namespace dgl_tpp {
typedef BFloat16 bfloat16;
template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<long>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<double>() {
  return LIBXSMM_DATATYPE_F64;
}
template <>
inline libxsmm_datatype XsmmDtype<uint16_t>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no, bool print=false) {
  if (print) {
    libxsmm_meqn_tree_print(eqn_no);
    libxsmm_meqn_rpn_print(eqn_no);
  }
}

inline int meqn_push_arg(
    const libxsmm_blasint idx,
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint ld,
    const libxsmm_blasint in_pos,
    const libxsmm_blasint offs_in_pos,
    const libxsmm_datatype dtype) {
  // This "singular" type dictates that the arg is a regular tensor (and not a
  // set of tensors)
  libxsmm_matrix_arg_attributes arg_singular_attr =
      libxsmm_create_matrix_arg_attributes(
          LIBXSMM_MATRIX_ARG_TYPE_SINGULAR,
          LIBXSMM_MATRIX_ARG_SET_TYPE_NONE,
          0,
          0);
  // Arg metadata include equation id and pos in arg array at runtime
  libxsmm_meqn_arg_metadata arg_metadata =
      libxsmm_create_meqn_arg_metadata(idx, in_pos);
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, ld, dtype);
  return libxsmm_meqn_push_back_arg(
      arg_metadata, arg_shape, arg_singular_attr);
}

inline libxsmm_meqn_function meqn_dispatch(
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint* ldo,
    const libxsmm_datatype out_type,
    const unsigned int idx) {
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, *ldo, out_type);
  return libxsmm_dispatch_meqn(idx, arg_shape);
}

inline int meqn_push_unary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_unary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  // OP metadata include equation id and an integer dictating where the op
  // metadata at runtime (if any) are located in the op arg array. -1 dictates
  // there are no op metadata needed
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_unary_op(
      op_metadata, type, dtype, flags);
}
inline int meqn_push_binary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_binary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_binary_op(
      op_metadata, type, dtype, flags);
}

inline int meqn_push_ternary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_ternary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_ternary_op(
      op_metadata, type, dtype, flags);
}

class BaseTPP {
 public:
  void* get_kernel() {
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == "")
      hash = hash_str();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
      kernel = build_kernel();
      if (kernel == NULL) {
        fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
        exit(1);
      }
      //printf("TPP: %s @ %p\n", hash.c_str(), kernel);
      kernel_cache[hash] = kernel;
      //printf("Hash size = %ld\n", (long)kernel_cache.size());
    }
    return kernel;
  }
  // We should make hash_str() public
  std::string get_hash_str() {
    return hash_str();
  }

 protected:
  std::unordered_map<std::string, void*>& get_kernel_cache() {
    static std::unordered_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
  virtual std::string hash_str() = 0;
  virtual void* build_kernel() = 0;
  std::string hash = "";
  bool initialized = false;
};

class UnaryTPP : public BaseTPP {
 public:
  UnaryTPP() {}
  UnaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_unary_type type)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        dt_in(dt_in),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_unary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = NULL;
    unary_param.in.tertiary = NULL;
    unary_param.in.quaternary = NULL;
    unary_param.out.primary = out;
    unary_param.out.secondary = NULL;
    unary_param.out.tertiary = NULL;
    unary_param.out.quaternary = NULL;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = NULL;
    unary_param.in.tertiary = NULL;
    unary_param.in.quaternary = NULL;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    unary_param.out.tertiary = NULL;
    unary_param.out.quaternary = NULL;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.in.quaternary = NULL;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    unary_param.out.tertiary = NULL;
    unary_param.out.quaternary = NULL;
    kernel(&unary_param);
  }

  void operator()(
      void* in,
      void* in2,
      void* in3,
      void* op,
      void* op2,
      void* op3,
      void* out,
      void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.in.quaternary = NULL;
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    unary_param.out.tertiary = NULL;
    unary_param.out.quaternary = NULL;
    kernel(&unary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200]={""};
    snprintf(
        hash,
        200,
        "unary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi,
        ldo,
        dt_in,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
        cols, rows, ldi, ldo, dt_in, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_unary(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi = 0;
  libxsmm_blasint ldo = 0;
  libxsmm_datatype dt_in = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_out = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_compute = LIBXSMM_DATATYPE_F32;
  libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary kernel = NULL;
};

class BinaryTPP : public BaseTPP {
 public:
  BinaryTPP() {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : BinaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_binary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in0, void* in1, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out;
    kernel(&binary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "binary_r%d_c%d_i0%d_i1%d_o%d_di0%d_di1%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_binary_shape shape = libxsmm_create_meltw_binary_shape(
        cols, rows, ldi0, ldi1, ldo, dt_in0, dt_in1, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_binary(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_binary_type type;
  libxsmm_meltwfunction_binary kernel = NULL;
};

template <typename T>
class SetZeroTPP {
 public:
  SetZeroTPP() {}
  SetZeroTPP(int N) : SetZeroTPP(1, N) {}
  SetZeroTPP(int rows, int cols) : SetZeroTPP(rows, cols, cols) {}
  SetZeroTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_XOR) {}
  void operator()(T* buf) {
    kernel((void*)buf, (void*)buf);
  }
  void ref(T* buf) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        buf[i * ldo + j] = 0;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class ConvertTPP {
 public:
  ConvertTPP() {}
  ConvertTPP(int N) : ConvertTPP(1, N) {}
  ConvertTPP(int rows, int cols) : ConvertTPP(rows, cols, cols, cols) {}
  ConvertTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        init_done(true) {}
  void operator()(Tin* in, Tout* out) {
    if (!(XsmmDtype<Tin>() == LIBXSMM_DATATYPE_F32 &&
          XsmmDtype<Tout>() == LIBXSMM_DATATYPE_F32) ||
        ((void*)in != (void*)out))
      kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i * ldi + j];
      }
    }
  }
  bool initialized() {
    return init_done;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
  bool init_done = false;
};

template <typename T>
class CpyTPP {
 public:
  CpyTPP() {}
  CpyTPP(int N) : CpyTPP(1, N) {}
  CpyTPP(int rows, int cols) : CpyTPP(rows, cols, cols, cols) {}
  CpyTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBcastColTPP {
 public:
  CpyBcastColTPP() {}
  CpyBcastColTPP(int rows, int cols) : CpyBcastColTPP(rows, cols, cols) {}
  CpyBcastColTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            cols,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBcastRowTPP {
 public:
  CpyBcastRowTPP() {}
  CpyBcastRowTPP(int rows, int cols) : CpyBcastRowTPP(rows, cols, cols) {}
  CpyBcastRowTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class AddTPP {
 public:
  AddTPP() {}
  AddTPP(int N) : AddTPP(1, N) {}
  AddTPP(int rows, int cols) : AddTPP(rows, cols, cols, cols) {}
  AddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] + (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class SubTPP {
 public:
  SubTPP() {}
  SubTPP(int N) : SubTPP(1, N) {}
  SubTPP(int rows, int cols) : SubTPP(rows, cols, cols, cols) {}
  SubTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_SUB) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] - (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class SubBcastTPP {
 public:
  SubBcastTPP() {}
  SubBcastTPP(int N) : SubBcastTPP(1, N) {}
  SubBcastTPP(int rows, int cols) : SubBcastTPP(rows, cols, cols, cols) {}
  SubBcastTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_SUB) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = in0[r * ldi + c] - in1[c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulTPP {
 public:
  MulTPP() {}
  MulTPP(int N) : MulTPP(1, N) {}
  MulTPP(int rows, int cols) : MulTPP(rows, cols, cols, cols) {}
  MulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] * (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename T>
class MulBcastVecTPP {
 public:
  MulBcastVecTPP() {}
  MulBcastVecTPP(int rows, int cols) : MulBcastVecTPP(rows, cols, cols) {}
  MulBcastVecTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            ld,
            ld,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}

  void operator()(T* in1, T* in2, T* out) {
      kernel((void*)in1, (void*)in2, (void*)out);
  }
  void ref(T* in1, T* in2, T* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ld + c] = in1[c] * in2[r*ld+c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class DivTPP {
 public:
  DivTPP() {}
  DivTPP(int N) : DivTPP(1, N) {}
  DivTPP(int rows, int cols) : DivTPP(rows, cols, cols, cols) {}
  DivTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_DIV) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] / (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MaxTPP {
 public:
  MaxTPP() {}
  MaxTPP(int N) : MaxTPP(1, N) {}
  MaxTPP(int rows, int cols) : MaxTPP(rows, cols, cols, cols) {}
  MaxTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MAX) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] > (float)in1[r * ldi + c] ?
          (float)in0[r * ldi + c] : (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MinTPP {
 public:
  MinTPP() {}
  MinTPP(int N) : MinTPP(1, N) {}
  MinTPP(int rows, int cols) : MinTPP(rows, cols, cols, cols) {}
  MinTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MIN) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] < (float)in1[r * ldi + c] ?
          (float)in0[r * ldi + c] : (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout = Tin>
class ReduceMaxRowsIdxTPP {
 public:
  ReduceMaxRowsIdxTPP() {}
  ReduceMaxRowsIdxTPP(int E)
      : E(E),
        reduce(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
	    (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES :
	                         LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES) | 
	    LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP|LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) {}
  void operator()(Tin* in, Tind* idx, Tind n, Tout* out, Tind* argM) {
    reduce((void*)in, (void*)idx, (void*)&n, (void*)out, (void*)argM);
  }
  void ref(Tin* in, Tind* idx, Tind n, Tout* out, Tind* argM) {
    if(sizeof(Tin) == 4) {
      for (int c = 0; c < E; c++) {
	Tin max = 0xff800000;
	Tind am;
	for(int r=0; r<n; r++) {
	  if((float)in[idx[r] * E + c] > max) {
	    max = (float)in[idx[r] * E + c];
	    am = r;
	  }
	}
	out[c] = max;
	argM[c] = am;
      }
    }
    else if(sizeof(Tin) == 2) {
      for (int c = 0; c < E; c++) {
	Tin max = 0xFF80;
	Tind am;
	for(int r=0; r<n; r++) {
	  if((bfloat16)in[idx[r] * E + c] > max) {
	    max = (bfloat16)in[idx[r] * E + c];
	    am = r;
	  }
	}
	out[c] = max;
	argM[c] = am;
      }
    }
  }

 private:
  int E;
  UnaryTPP reduce;
};

template <typename Tin, typename Tind, typename Tout = Tin>
class ReduceMinRowsIdxTPP {
 public:
  ReduceMinRowsIdxTPP() {}
  ReduceMinRowsIdxTPP(int E)
      : E(E),
        reduce(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
	    (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : 
	                         LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES) | 
	    LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP|LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {}
  void operator()(Tin* in, Tind* idx, Tind n, Tout* out, Tind* argM) {
    reduce((void*)in, (void*)idx, (void*)&n, (void*)out, (void*)argM);
  }
  void ref(Tin* in, Tind* idx, Tind n, Tout* out, Tind* argM) {
    if(sizeof(Tin) == 4) {
      for (int c = 0; c < E; c++) {
	Tin min = 0x7f800000;
	Tind am;
	for(int r=0; r<n; r++) {
	  if((float)in[idx[r] * E + c] < min) {
	    min = (float)in[idx[r] * E + c];
	    am = r;
	  }
	}
	out[c] = min;
	argM[c] = am;
      }
    }
    else if(sizeof(Tin) == 2) {
      for (int c = 0; c < E; c++) {
	Tin min = 0x7F80;
	Tind am;
	for(int r=0; r<n; r++) {
	  if((bfloat16)in[idx[r] * E + c] < min) {
	    min = (bfloat16)in[idx[r] * E + c];
	    am = r;
	  }
	}
	out[c] = min;
	argM[c] = am;
      }
    }
  }

 private:
  int E;
  UnaryTPP reduce;
};

template <typename T>
class RecpTPP {
 public:
  RecpTPP() {}
  RecpTPP(int cols) : RecpTPP(1, cols, cols, cols) {}
  RecpTPP(int rows, int cols) : RecpTPP(rows, cols, cols, cols) {}
  RecpTPP(int rows, int cols, int ldi) : RecpTPP(rows, cols, ldi, cols) {}
  RecpTPP(int rows, int cols, int ldi, int ldo) 
      : rows(rows),
	cols(cols),
	ldi(ldi),
	ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
	out[r*ldo+c] = 1.0 / in[r*ldi+c];
  }

 private:
  int rows = 1;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout=Tin>
class GatherTPP {
 public:
  GatherTPP() {}
  GatherTPP(int rows, int cols, int ldi)
      : GatherTPP(rows, cols, ldi, ldi) {}
  GatherTPP(int rows, int cols)
      : GatherTPP(rows, cols, cols, cols) {}
  GatherTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tout>(),
            (LIBXSMM_MELTW_FLAG_UNARY_GS_COLS |
             (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_GATHER) {}
  void operator()(Tin* in0, Tind* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, NULL, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      auto ind = in1[r];
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = in0[ind * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class ScatterTPP {
 public:
  ScatterTPP() {}
  ScatterTPP(int rows, int cols, int ldi) : ScatterTPP(rows, cols, ldi, ldi) {}
  ScatterTPP(int rows, int cols) : ScatterTPP(rows, cols, cols, cols) {}
  ScatterTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tout>(),
            (LIBXSMM_MELTW_FLAG_UNARY_GS_COLS |
             (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_SCATTER) {}
  void operator()(Tin* in, Tind* out1, Tout* out) {
    kernel((void*)in, NULL, NULL, (void*)out, (void*)out1);
  }
  void ref(Tin* in, Tind* out1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      auto ind = out1[r];
      for (int c = 0; c < cols; c++) {
        out[ind * ldo + c] = in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ExpTPP {
  public:
    ExpTPP() {}
    ExpTPP(int rows, int cols) : ExpTPP(rows, cols, cols, cols) {}
    ExpTPP(int rows, int cols, int ldi) : ExpTPP(rows, cols, cols, ldi, ldi) {}
    ExpTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
	ldi(ldi),
	ldo(ldo),
	kernel(
	    rows,
	    cols,
	    ldi,
	    ldo,
	    XsmmDtype<Tin>(),
	    XsmmDtype<Tout>(),
	    LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
	    LIBXSMM_MELTW_TYPE_UNARY_EXP) {}

  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
	out[r*ldo + c] = std::exp((float)in[r*ldi + c]);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout = Tin>
class ReduceAddRowsIdxTPP {
 public:
  ReduceAddRowsIdxTPP() {}
  ReduceAddRowsIdxTPP(int E)
      : E(E),
        reduce(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES 
		    : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES) |  
	    LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tin* in, Tind* idx, Tind n, Tout* out) {
    reduce((void*)in, (void*)idx, (void*)&n, (void*)out, NULL);
  }
  void ref(Tin* in, Tind* idx, Tind n, Tout* out) {
    for (int c = 0; c < E; c++) {
      out[c] = 0;
      for(int r=0; r<n; r++) {
	out[c] += (float)in[idx[r] * E + c];
      }
    }
  }

 private:
  int E;
  UnaryTPP reduce;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddRowsTPP {
 public:
  ReduceAddRowsTPP() {}
  ReduceAddRowsTPP(int rows, int cols) : ReduceAddRowsTPP(rows, cols, cols, cols) {}
  ReduceAddRowsTPP(int rows, int cols, int ldi) : ReduceAddRowsTPP(rows, cols, ldi, cols) {}
  ReduceAddRowsTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS | LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, Tout* out) {
    reduce(in, out);
  }
  void ref(Tin* in, Tout* out) {
    for (int c = 0; c < cols; c++) {
      for (int r = 0; r < rows; r++) {
        if (c == 0)
          out[r] = 0;
        out[r] += in[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddColsTPP {
 public:
  ReduceAddColsTPP() {}
  ReduceAddColsTPP(int rows, int cols) : ReduceAddColsTPP(rows, cols, cols, cols) {}
  ReduceAddColsTPP(int rows, int cols, int ldi) : ReduceAddColsTPP(rows, cols, ldi, cols) {}
  ReduceAddColsTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, Tout* out) {
    reduce(in, out);
  }
  void ref(Tin* in, Tout* out) {
    for (int c = 0; c < cols; c++) {
      for (int r = 0; r < rows; r++) {
        if (c == 0)
          out[r] = 0;
        out[r] += in[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

template <typename Tin, typename Tout, typename Tind>
class EmbBagFwdTPP {
 public:
  EmbBagFwdTPP() {}
  EmbBagFwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (libxsmm_meltw_unary_flags)(
                sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                  : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tout* output, Tin* weight, Tind* input, int N) {
    unsigned long long _N = N;
    kernel((void*)weight, (void*)input, (void*)&_N, (void*)output, NULL);
  }
  void ref(Tout* output, Tin* weight, Tind* input, int N) {
    for (long v = 0; v < E; v++)
      output[v] = 0;
    for (long s = 0; s < N; s++) {
      auto ind = input[s];
      for (long v = 0; v < E; v++)
        output[v] += weight[ind * E + v];
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class EmbBagBwdTPP {
 public:
  EmbBagBwdTPP() {}
  EmbBagBwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tout>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {}
  void operator()(Tin* in, Tout* out, uint64_t N) {
    kernel((void*)in, NULL, NULL, (void*)&N, NULL, NULL, (void*)out, NULL);
  }
  void ref(Tin* in, Tout* out, uint64_t N) {
    for (uint64_t i = 0; i < N; i++) {
      for (int v = 0; v < E; v++) {
        out[i * E + v] = in[v];
      }
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};

template <typename T1, typename T2 = T1, typename T3 = T1>
class MulReduceTPP : public BaseTPP {
 public:
  MulReduceTPP() {}
  MulReduceTPP(int rows, int cols) : rows(rows), cols(cols) {
    kernel = (libxsmm_meqn_function)get_kernel();
    initialized = true;
  }

  void operator()(T1* in0, T2* in1, T3* out) {
    if (!initialized)
      return;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[2];
    arg_array[0].primary = (void*)in0;
    arg_array[1].primary = (void*)in1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;

    kernel(&eqn_param);
  }

  void ref(T1* in0, T2* in1, T3* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r] += (float)in0[r * cols + c] * (float)in1[r * cols + c];
      }
    }
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "mul_reduce_eqn_t%d_%d_%d_r%d_c%d",
        XsmmDtype<T1>(),
        XsmmDtype<T2>(),
        XsmmDtype<T3>(),
        rows, 
        cols);
    return std::string(hash);
  }
  void* build_kernel() override {
    auto dt1 = XsmmDtype<T1>();
    auto dt2 = XsmmDtype<T2>();
    auto dt3 = XsmmDtype<T3>();
    libxsmm_blasint ld = 1;
    libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
    meqn_push_unary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
        LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
        LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_MUL,
        LIBXSMM_MELTW_FLAG_BINARY_NONE,
        LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, cols, rows, cols, 0, 0, dt1);
    meqn_push_arg(my_eqn0, cols, rows, cols, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(1, rows, &ld, dt3, my_eqn0);
  }

 private:
  int rows = 0;
  int cols = 0;
  libxsmm_meqn_function kernel = NULL;
};

}; // namespace dgl_tpp

#endif // _XSMM_FUNCTORS_H_
