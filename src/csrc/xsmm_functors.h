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
#include <libxsmm_intrinsics_x86.h>
#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#else
#include <pytorch_extension_wrapper.h>
#endif
#include <float8.h>
#include <string>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#ifdef DEBUG_TRACE_TPP
extern int tpp_debug_trace;
#endif
#ifndef __x86_64__
#define __rdtsc() (0)
#endif

#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
extern long long hsh_key, hsh_ret;
namespace tpp {
typedef at::BFloat16 bfloat16;
typedef at::Half half;
#ifdef PYTORCH_SUPPORTS_FLOAT8
typedef at::Float8_e5m2 bfloat8;
typedef at::Float8_e4m3fn hfloat8;
#endif

inline float upconvert_to_float(float val) {
  return val;
}
inline float upconvert_to_float(bfloat16 val) {
  return (float)val;
}
inline float upconvert_to_float(half val) {
  return (float)val;
}
#ifdef PYTORCH_SUPPORTS_FLOAT8
inline float upconvert_to_float(bfloat8 val) {
  return (float)val;
}
inline float upconvert_to_float(hfloat8 val) {
  return (float)val;
}
#endif
template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<int64_t>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int32_t>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<half>() {
  return LIBXSMM_DATATYPE_F16;
}
template <>
inline libxsmm_datatype XsmmDtype<uint8_t>() {
  return LIBXSMM_DATATYPE_I8;
}
template <>
inline libxsmm_datatype XsmmDtype<int8_t>() {
  return LIBXSMM_DATATYPE_I8;
}
#ifdef PYTORCH_SUPPORTS_FLOAT8
template <>
inline libxsmm_datatype XsmmDtype<bfloat8>() {
  return LIBXSMM_DATATYPE_BF8;
}
template <>
inline libxsmm_datatype XsmmDtype<hfloat8>() {
  return LIBXSMM_DATATYPE_HF8;
}
#endif

#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto(float const* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, float const* mem_addr) {
  return _mm512_maskz_loadu_ps(k, mem_addr);
}
inline void _mm512_storeu_ps_auto(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
inline void _mm512_mask_storeu_ps_auto(float* mem_addr, __mmask16 k, __m512 a) {
  _mm512_mask_storeu_ps(mem_addr, k, a);
}

inline __m512 _mm512_loadu_ps_auto(half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(half* mem_addr, __m512 a) {
  _mm256_storeu_si256(
      (__m256i*)mem_addr,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void _mm512_mask_storeu_ps_auto(half* mem_addr, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16(
      (__m256i*)mem_addr,
      k,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m512 _mm512_convert_bf_ps(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a), 16));
}
inline __m256i _mm256_convert_ps_bf(__m512 a) {
  return _mm512_cvtepi32_epi16(
      _mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

inline __m512 _mm512_loadu_ps_auto(bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(
    __mmask16 k,
    bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat16* mem_addr, __m512 a) {
  _mm256_storeu_si256((__m256i*)mem_addr, _mm256_convert_ps_bf(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat16* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a));
}

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(
    __mmask16 k,
    bfloat16 const* hi,
    bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16* hi, bfloat16* lo, __m512 a) {
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256(
      (__m256i*)hi,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256(
      (__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(
    bfloat16* hi,
    bfloat16* lo,
    __mmask16 k,
    __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16(
      (__m256i*)hi,
      k,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16(
      (__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline __m512 _mm512_convert_bf8_ps(__m128i a) {
  return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8));
}
inline __m128i _mm_convert_ps_bf8(__m512 a) {
  return _mm256_cvtepi16_epi8(_mm256_srai_epi16(
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 8));
}
inline __m512 _mm512_convert_hf8_ps(__m128i a) {
  TPP_ASSERT(0, "_mm512_convert_hf8_ps not implemented\n");
  return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8));
}
inline __m128i _mm_convert_ps_hf8(__m512 a) {
  TPP_ASSERT(0, "_mm_convert_ps_hf8 not implemented\n");
  return _mm256_cvtepi16_epi8(_mm256_srai_epi16(
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 8));
}

inline __m512 _mm512_loadu_ps_auto(int const* mem_addr) {
  return _mm512_cvtepi32_ps(_mm512_loadu_epi32(mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, int const* mem_addr) {
  return _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(k, mem_addr));
}

#ifdef PYTORCH_SUPPORTS_FLOAT8
inline __m512 _mm512_loadu_ps_auto(bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(_mm_loadu_si128((__m128i const*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(
      _mm_maskz_loadu_epi8(k, (__m128i const*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat8* mem_addr, __m512 a) {
  _mm_storeu_si128((__m128i*)mem_addr, _mm_convert_ps_bf8(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat8* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm_mask_storeu_epi8((__m128i*)mem_addr, k, _mm_convert_ps_bf8(a));
}

inline __m512 _mm512_loadu_ps_auto(hfloat8 const* mem_addr) {
  return _mm512_convert_hf8_ps(_mm_loadu_si128((__m128i const*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, hfloat8 const* mem_addr) {
  return _mm512_convert_hf8_ps(
      _mm_maskz_loadu_epi8(k, (__m128i const*)mem_addr));
}

inline void _mm512_storeu_ps_auto(hfloat8* mem_addr, __m512 a) {
  _mm_storeu_si128((__m128i*)mem_addr, _mm_convert_ps_hf8(a));
}
inline void _mm512_mask_storeu_ps_auto(
    hfloat8* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm_mask_storeu_epi8((__m128i*)mem_addr, k, _mm_convert_ps_hf8(a));
}
#endif // PYTORCH_SUPPORTS_FLOAT8
#endif

inline libxsmm_datatype convert_dtype_pt2xsmm(at::ScalarType dtype) {
  static const std::map<at::ScalarType, libxsmm_datatype> pt2xsmmDtypes = {
      {at::kDouble, LIBXSMM_DATATYPE_F64},
      {at::kFloat, LIBXSMM_DATATYPE_F32},
      {at::kHalf, LIBXSMM_DATATYPE_F16},
      {at::kBFloat16, LIBXSMM_DATATYPE_BF16},
      {at::kByte, LIBXSMM_DATATYPE_U8},
      {at::kChar, LIBXSMM_DATATYPE_I8},
      {at::kShort, LIBXSMM_DATATYPE_I16},
      {at::kInt, LIBXSMM_DATATYPE_I32},
      {at::kLong, LIBXSMM_DATATYPE_I64},
#ifdef PYTORCH_SUPPORTS_FLOAT8
      {at::kFloat8_e5m2, LIBXSMM_DATATYPE_BF8},
      {at::kFloat8_e4m3fn, LIBXSMM_DATATYPE_HF8}
#endif
  };

  return pt2xsmmDtypes.at(dtype);
}

inline int xsmm_get_vnni_block_size(libxsmm_datatype dtype) {
  int bs = libxsmm_cpuid_dot_pack_factor(dtype);
  if (bs <= 0) {
    throw std::invalid_argument("Unsupported datatype");
  }
  return bs;
}

inline int get_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline int get_vnni_block_size(caffe2::TypeMeta dtype_) {
  at::ScalarType dtype = dtype_.toScalarType();
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

template <typename T>
inline int get_vnni_block_size() {
  auto xsmm_dtype = XsmmDtype<T>();
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no) {
  if (false) {
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
  return libxsmm_meqn_push_back_arg(arg_metadata, arg_shape, arg_singular_attr);
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
  return libxsmm_meqn_push_back_unary_op(op_metadata, type, dtype, flags);
}
inline int meqn_push_binary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_binary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_binary_op(op_metadata, type, dtype, flags);
}
inline int meqn_push_ternary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_ternary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_ternary_op(op_metadata, type, dtype, flags);
}

class BaseTPP {
 public:
  void* get_kernel() {
    auto t0 = __rdtsc();
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == "")
      hash = hash_str();
    auto t1 = __rdtsc();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
#ifdef DEBUG_TRACE_TPP
      if (tpp_debug_trace >= 1)
        printf("TPP: building %s\n", hash.c_str());
#endif
      kernel = build_kernel();
      if (kernel == NULL) {
        fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
        exit(1);
      }
#ifdef DEBUG_TRACE_TPP
      if (tpp_debug_trace >= 1)
        printf("TPP: built    %s @ %p\n", hash.c_str(), kernel);
#endif
      kernel_cache[hash] = kernel;
      // printf("Hash size = %ld\n", (long)kernel_cache.size());
    }
    auto t2 = __rdtsc();
    hsh_key += t1 - t0;
    hsh_ret += t2 - t1;
    // printf("%6lld  %6lld %6lld  get_kernel[%s]\n", t2-t0, (t1-t0), (t2-t1),
    // hash.c_str());
    return kernel;
  }
  // We should make hash_str() public
  std::string get_hash_str() {
    return hash_str();
  }

 protected:
#if 0
  std::unordered_map<std::string, void*>& get_kernel_cache() {
    static std::unordered_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
#else
  ska::flat_hash_map<std::string, void*>& get_kernel_cache() {
    static ska::flat_hash_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
#endif
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
    unary_param.out.primary = out;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
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
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
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

class TernaryTPP : public BaseTPP {
 public:
  TernaryTPP() {}
  TernaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_ternary_type type)
      : TernaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  TernaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldi2,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_in2,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_ternary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldi2(ldi2),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_in2(dt_in2),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_ternary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in0, void* in1, void* in2, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_ternary_param ternary_param;
    ternary_param.in0.primary = in0;
    ternary_param.in1.primary = in1;
    ternary_param.in2.primary = in2;
    ternary_param.out.primary = out;
    kernel(&ternary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "ternary_r%d_c%d_i0%d_i1%d_i2%d_o%d_di0%d_di1%d_di2%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi0,
        ldi1,
        ldi2,
        ldo,
        dt_in0,
        dt_in1,
        dt_in2,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_ternary_shape shape = libxsmm_create_meltw_ternary_shape(
        cols,
        rows,
        ldi0,
        ldi1,
        ldi2,
        ldo,
        dt_in0,
        dt_in1,
        dt_in2,
        dt_out,
        dt_compute);
    return (void*)libxsmm_dispatch_meltw_ternary(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldi2;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_in2;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_ternary_type type;
  libxsmm_meltwfunction_ternary kernel = NULL;
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

template <typename Tin, typename Tout, typename Tscale = float>
class QuantTPP {
 public:
  QuantTPP() {}
  QuantTPP(int N) : QuantTPP(1, N) {}
  QuantTPP(int rows, int cols) : QuantTPP(rows, cols, cols, cols) {}
  QuantTPP(int rows, int cols, int ldi, int ldo)
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
            LIBXSMM_MELTW_TYPE_UNARY_QUANT) {}
  void operator()(Tin* in, Tout* out, Tscale scale) {
    if constexpr (true || std::is_same_v<Tin, float>) {
      float fscale = (float)scale;
      kernel((void*)in, &fscale, nullptr, (void*)out, nullptr);
    } else {
      ref(in, out, scale);
    }
  }
  void ref(Tin* in, Tout* out, Tscale scale) {
    constexpr int max = std::is_signed<Tout>::value
        ? (1 << (8 * sizeof(Tout) - 1)) - 1
        : (1 << (8 * sizeof(Tout))) - 1;
    constexpr int min = std::is_signed<Tout>::value ? -max : 0;
    float fscale = (float)scale;
#ifdef __AVX512F__
    auto vfscale = _mm512_set1_ps(fscale);
    auto vmax = _mm512_set1_epi32(max);
    auto vmin = _mm512_set1_epi32(min);
    for (int j = 0; j < rows; j++) {
      int i = 0;
      for (i = 0; i < ALIGNDOWN(cols, 16); i += 16) {
        auto vin = _mm512_loadu_ps_auto(&in[j * ldi + i]);
        auto vq = _mm512_mul_ps(vin, vfscale);
        auto v_i32 = _mm512_cvt_roundps_epi32(
            vq, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        v_i32 = _mm512_max_epi32(_mm512_min_epi32(v_i32, vmax), vmin);
        if constexpr (sizeof(Tout) == 1) {
          auto res = _mm512_cvtepi32_epi8(v_i32);
          _mm_storeu_epi8(&out[j * ldo + i], res);
        } else if constexpr (sizeof(Tout) == 2) {
          auto res = _mm512_cvtepi32_epi16(v_i32);
          _mm256_storeu_epi16(&out[j * ldo + i], res);
        } else {
          _mm512_storeu_epi32(&out[j * ldo + i], v_i32);
        }
      }
      if (i < cols) {
        int rem = cols - i;
        __mmask16 mask = (1 << rem) - 1;
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
        auto vq = _mm512_mul_ps(vin, vfscale);
        auto v_i32 = _mm512_cvt_roundps_epi32(
            vq, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        v_i32 = _mm512_max_epi32(_mm512_min_epi32(v_i32, vmax), vmin);
        if constexpr (sizeof(Tout) == 1) {
          auto res = _mm512_cvtepi32_epi8(v_i32);
          _mm_mask_storeu_epi8(&out[j * ldo + i], mask, res);
        } else if constexpr (sizeof(Tout) == 2) {
          auto res = _mm512_cvtepi32_epi16(v_i32);
          _mm256_mask_storeu_epi16(&out[j * ldo + i], mask, res);
        } else {
          _mm512_mask_storeu_epi32(&out[j * ldo + i], mask, v_i32);
        }
      }
    }
#else
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int qval = nearbyintf((float)in[i * ldi + j] * fscale);
        qval = std::clamp(qval, min, max);
        out[i * ldo + j] = (Tout)qval;
      }
    }
#endif
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout, typename Tscale>
class DequantTPP : public BaseTPP {
 public:
  DequantTPP() {}
  DequantTPP(int N) : DequantTPP(1, N) {}
  DequantTPP(int rows, int cols) : DequantTPP(rows, cols, cols, cols) {}
  DequantTPP(int rows, int cols, int ldi, int ldo)
      : DequantTPP(rows, cols, ldi, ldo, 1) {}
  DequantTPP(int rows, int cols, int ldi, int ldo, int ldis)
      : rows(rows), cols(cols), ldi(ldi), ldo(ldo), ldis(ldis) {
    kernel = (libxsmm_meqn_function)get_kernel();
  }
  void operator()(Tin* in, Tout* out, Tscale* i_scl, Tscale* w_scl) {
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[4];
    arg_array[0].primary = (void*)w_scl;
    arg_array[1].primary = (void*)i_scl;
    arg_array[2].primary = (void*)in;
    arg_array[3].primary = (void*)out;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;
    kernel(&eqn_param);
  }
  void ref(Tin* in, Tout* out, Tscale* i_scl, Tscale* w_scl) {
#ifdef __AVX512F__
    for (int i = 0; i < rows; i++) {
      int j;
      float fi_scl = (float)i_scl[i * ldis];
      auto vfi_scl = _mm512_set1_ps(fi_scl);
      for (j = 0; j < ALIGNDOWN(cols, 16); j += 16) {
        auto vfw_scl = _mm512_loadu_ps_auto(&w_scl[j]);
        auto vfscl = _mm512_mul_ps(vfi_scl, vfw_scl);
        auto vin = _mm512_loadu_ps_auto(&in[i * ldi + j]);
        auto vout = _mm512_loadu_ps_auto(&out[i * ldo + j]);
        vout = _mm512_fmadd_ps(vfscl, vin, vout);
        _mm512_storeu_ps_auto(&out[i * ldo + j], vout);
      }
      if (j < cols) {
        int rem = cols - j;
        __mmask16 mask = (1 << rem) - 1;
        auto vfw_scl = _mm512_maskz_loadu_ps_auto(mask, &w_scl[j]);
        auto vfscl = _mm512_mul_ps(vfi_scl, vfw_scl);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[i * ldi + j]);
        auto vout = _mm512_maskz_loadu_ps_auto(mask, &out[i * ldo + j]);
        vout = _mm512_fmadd_ps(vfscl, vin, vout);
        _mm512_mask_storeu_ps_auto(&out[i * ldo + j], mask, vout);
      }
    }
#else
    for (int i = 0; i < rows; i++) {
      float fi_scl = (float)i_scl[i * ldis];
      for (int j = 0; j < cols; j++) {
        float fscl = fi_scl * (float)w_scl[j];
        out[i * ldo + j] += (Tout)in[i * ldi + j] * fscl;
      }
    }
#endif
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "dequant_eqn_t%d_%d_%d_r%d_c%d_ldi%d_ldo%d_ldsi%d",
        XsmmDtype<Tin>(),
        XsmmDtype<Tout>(),
        XsmmDtype<Tscale>(),
        rows,
        cols,
        ldi,
        ldo,
        ldis);
    return std::string(hash);
  }
  void* build_kernel() override {
    auto dt_in = XsmmDtype<Tin>();
    auto dt_out = XsmmDtype<Tout>();
    auto dt_scale = XsmmDtype<Tscale>();
    libxsmm_blasint ld = ldo;
    libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
    meqn_push_ternary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
        LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
        LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_MUL,
        LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0 |
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1,
        LIBXSMM_DATATYPE_F32);
    // TODO: FIX below eqn
    meqn_push_arg(my_eqn0, cols, 1, cols, 0, 0, dt_scale);
    meqn_push_arg(my_eqn0, 1, rows, ldis, 1, 0, dt_scale);
    meqn_push_arg(my_eqn0, cols, rows, ldi, 2, 0, dt_in);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 3, 0, dt_out);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(cols, rows, &ld, dt_out, my_eqn0);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  int ldis = 1;
  // int ldws = 1; must be 1
  libxsmm_meqn_function kernel = NULL;
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

template <typename T>
class PadTPP {
 public:
  PadTPP() {}
  PadTPP(int in_rows, int in_cols, int out_rows, int out_cols)
      : PadTPP(in_rows, in_cols, out_rows, out_cols, in_cols, out_cols) {}
  PadTPP(int in_rows, int in_cols, int out_rows, int out_cols, int ldi, int ldo)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        cpy(),
        zero() {
    if (out_rows > in_rows || out_cols > in_cols) {
      TPP_ASSERT(
          out_rows == in_rows || out_cols == in_cols,
          "PadTPP can pad only 1 dim at a time");
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, ldo);
      if (out_rows > in_rows) {
        zero = SetZeroTPP<T>(out_rows - in_rows, out_cols, ldo);
        zero_offset = in_rows * ldo;
      } else {
        zero = SetZeroTPP<T>(out_rows, out_cols - in_cols, ldo);
        zero_offset = in_cols;
      }
    }
  }
  void operator()(T* in, T* out) {
    cpy(in, out);
    zero(out);
  }
  void ref(T* in, T* out) {
    cpy.ref(in, out);
    zero.ref(out);
  }

 private:
  int in_rows = 0;
  int in_cols = 0;
  int out_rows = 0;
  int out_cols = 0;
  int ldi;
  int ldo;
  int zero_offset = 0;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <typename Tin, typename Tout = Tin>
class CpyBiasTPP {
 public:
  CpyBiasTPP() {}
  CpyBiasTPP(int rows, int cols) : CpyBiasTPP(rows, cols, cols) {}
  CpyBiasTPP(int rows, int cols, int ldo)
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
class CpyBcastTPP {
 public:
  CpyBcastTPP() {}
  CpyBcastTPP(int rows, int cols) : CpyBcastTPP(rows, cols, cols) {}
  CpyBcastTPP(int rows, int cols, int ldo)
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
template <typename T>
class AddBiasTPP {
 public:
  AddBiasTPP() {}
  AddBiasTPP(int rows, int cols) : AddBiasTPP(rows, cols, cols) {}
  AddBiasTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            cols,
            ld,
            ld,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(T* in, float* out) {
    kernel((void*)in, (void*)out, (void*)out);
  }
  void ref(T* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ld + c] += (float)in[c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin, typename Tin2 = Tin>
class AddTPP {
 public:
  AddTPP() {}
  AddTPP(int N) : AddTPP(1, N) {}
  AddTPP(int rows, int cols) : AddTPP(rows, cols, cols, cols) {}
  AddTPP(int rows, int cols, int ldi, int ldo)
      : AddTPP(rows, cols, ldi, ldi, ldo) {}
  AddTPP(int rows, int cols, int ldi0, int ldi1, int ldo)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi0,
            ldi1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin2>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(Tin* in0, Tin2* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin2* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi0 + c] + (float)in1[r * ldi1 + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi0;
  int ldi1;
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

template <typename Tin>
class GradBiasTPP {
 public:
  GradBiasTPP() {}
  GradBiasTPP(int rows, int cols) : GradBiasTPP(rows, cols, cols) {}
  GradBiasTPP(int rows, int cols, int ldi)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(cols) {}
  void operator()(Tin* in, float* out) {
    float tmp[cols];
    reduce((void*)in, (void*)tmp);
    add(tmp, out, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;

  UnaryTPP reduce;
  AddTPP<float, float> add;
};

// ############################# Mul & Reduction TPP
// #####################################

template <typename T1, typename T2 = T1, typename T3 = T1>
class MulReduceTPP : public BaseTPP {
 public:
  MulReduceTPP() {}
  MulReduceTPP(int N, int M) : N(N), M(M) {
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
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < M; c++) {
        out[r] += (float)in0[r * M + c] * (float)in1[r * M + c];
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
        N, //);
        M);
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
    // libxsmm_meqn_push_back_arg(my_eqn0, M, N, M, 0, 0, dt1);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_MUL,
        LIBXSMM_MELTW_FLAG_BINARY_NONE,
        LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, M, N, M, 0, 0, dt1);
    meqn_push_arg(my_eqn0, M, N, M, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(1, N, &ld, dt3, my_eqn0);
  }

 private:
  int N = 0;
  int M = 0;
  libxsmm_meqn_function kernel = NULL;
};
template <typename Tin, typename Tout = Tin>
class ReduceAddColTPP {
 public:
  ReduceAddColTPP() {}
  ReduceAddColTPP(int rows, int cols, int ldi, int ldo)
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
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, float* out) {
    reduce(in, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (r == 0)
          out[c] = 0;
        out[c] += (float)in[r * ldi + c];
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
class ReduceAddRowTPP {
 public:
  ReduceAddRowTPP() {}
  ReduceAddRowTPP(int rows, int cols, bool acc)
      : ReduceAddRowTPP(rows, cols, cols, acc) {}
  ReduceAddRowTPP(int rows, int cols, int ldi, bool acc)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        acc(acc),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(rows) {}
  void operator()(Tin* in, Tout* out) {
    if (acc) {
      Tout tmp[rows];
      reduce((void*)in, (void*)tmp);
      add(tmp, out, out);
    } else {
      reduce((void*)in, (void*)out);
    }
  }
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++) {
      if (!acc) {
        out[r] = 0;
      }
      for (int c = 0; c < cols; c++) {
        out[r] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  bool acc;
  UnaryTPP reduce;
  AddTPP<Tout, Tout> add;
};

// ############################# Broadcast & Multiplication TPP
// #####################################
template <typename Tin, typename Tin2, typename Tout = Tin>
class BCastMulTPP {
 public:
  BCastMulTPP() {}
  BCastMulTPP(int rows, int cols) : BCastMulTPP(rows, cols, cols, cols) {}
  BCastMulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin2>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MUL) // Multiplication
  {}
  void operator()(Tin* in0, Tin2* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin2* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] = (float)in0[r] * (float)in1[c * ldi + r];
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

// ############################# Broadcast & Multiplication Addition TPP
// #####################################
template <typename Tin, typename Tout>
class BCastMulAddTPP {
 public:
  BCastMulAddTPP() {}
  BCastMulAddTPP(int rows, int cols) : BCastMulAddTPP(rows, cols, cols, cols) {}
  BCastMulAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) // Multiplication
  {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }

  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] += (Tin)in0[r] * (Tin)in1[c * ldi + r];
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

template <typename Tin, typename Tout>
class ScaleTPP {
 public:
  ScaleTPP() {}
  ScaleTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    for (int i = 0; i < N; i++) {
      out[i] = (float)in[i] * (float)alpha;
    }
  }

 private:
  int N = 0;
  BinaryTPP kernel;
};

template <typename T, typename TN = float>
class Norm2TPP {
 public:
  Norm2TPP() {}
  Norm2TPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) {}
  void operator()(T* in, TN* sum) {
    float lsum = 0.0f;
    kernel((void*)in, (void*)&lsum);
    *sum += (TN)lsum;
  }
  void ref(T* in, TN* sum) {
    float lsum = 0.0f;
    for (int i = 0; i < N; i++) {
      lsum += (float)in[i] * (float)in[i];
    }
    *sum += (TN)lsum;
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpTPP {
 public:
  RecpTPP() {}
  RecpTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / in[i];
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpSqrtTPP {
 public:
  RecpSqrtTPP() {}
  RecpSqrtTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / sqrt(in[i]);
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulNormTPP {
 public:
  MulNormTPP() {}
  MulNormTPP(int rows, int cols) : MulNormTPP(rows, cols, cols, cols) {}
  MulNormTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1, // ldi0
            ldi, // ldi1
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel((void*)in, (void*)in2, (void*)out);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        out[r * ldo + c] = in[r] * in2[r * ldi + c];
  }

 private:
  int rows, cols;
  int ldi, ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleAddTPP {
 public:
  ScaleAddTPP() {}
  ScaleAddTPP(int N) : ScaleAddTPP(1, N) {}
  ScaleAddTPP(int rows, int cols) : ScaleAddTPP(rows, cols, cols, cols) {}
  ScaleAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<float>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) {}
  void operator()(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] += (float)in[i * ldi + j] * (float)alpha;
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

template <typename Tin, typename Tind, typename Tout>
class EmbeddingFwdTPP {
 public:
  EmbeddingFwdTPP() {}
  EmbeddingFwdTPP(int rows, int cols, int ldi)
      : EmbeddingFwdTPP(rows, cols, ldi, ldi) {}
  EmbeddingFwdTPP(int rows, int cols)
      : EmbeddingFwdTPP(rows, cols, cols, cols) {}
  EmbeddingFwdTPP(int rows, int cols, int ldi, int ldo)
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
class EmbeddingBwdTPP {
 public:
  EmbeddingBwdTPP() {}
  EmbeddingBwdTPP(int E)
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
  void operator()(Tin* in0, Tind* in1, Tout* out, int N) {
    unsigned long long _N = N;
    kernel((void*)in0, (void*)in1, (void*)&_N, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out, int N) {
    for (long v = 0; v < E; v++)
      out[v] = 0;
    for (long s = 0; s < N; s++) {
      auto ind = in1[s];
      for (long v = 0; v < E; v++)
        out[v] += in0[ind * E + v];
    }
  }

 private:
  int E = 0;
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
            LIBXSMM_DATATYPE_F32,
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

class XformTPP {
 public:
  XformTPP() {}
  XformTPP(
      libxsmm_blasint rows_i,
      libxsmm_blasint cols_i,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dtype,
      libxsmm_meltw_unary_type type)
      : rows(rows_i),
        cols(cols_i),
        ldi(ldi),
        ldo(ldo),
        dtype(dtype),
        type(type),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            dtype,
            dtype,
            dtype,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            type) {}
  void operator()(void* in, void* out) {
    kernel(in, out);
  }
  typedef enum XFORM_TYPE {
    XFORM_NONE_TPP = 0,
    XFORM_XPOSE_TPP = 1,
    XFORM_N2V_TPP = 2,
    XFORM_XPOSE_N2V_TPP = 3,
    XFORM_XPOSE_V2V_TPP = 4
  } XFORM_TYPE;

 private:
  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;
  libxsmm_meltw_unary_type type;
  UnaryTPP kernel;
};

template <typename T>
class XformExtTPP {
 public:
  XformExtTPP() {}
  XformExtTPP(
      /* rows and cols as for input tensor */
      int rows,
      int cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            rows,
            cols,
            (xtype == XformTPP::XFORM_N2V_TPP ? rows : cols),
            (xtype == XformTPP::XFORM_N2V_TPP ? cols : rows),
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            in_rows,
            in_cols,
            out_rows,
            out_cols,
            in_cols,
            out_cols,
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      int ldi,
      int ldo,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        xtype(xtype),
        dtype(XsmmDtype<T>()),
        kernel(),
        cvt(),
        cpy(),
        zero() {
    libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
    if (ignore_vnni_for_fp32 == false) {
      TPP_ASSERT(
          (xtype == XformTPP::XFORM_XPOSE_TPP || dtype != LIBXSMM_DATATYPE_F32),
          "Only Transpose Xofrm supportd for FP32 datatype, specified %d\n",
          (int)xtype);
    }
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_N2V_TPP) {
      in_rows_p = out_rows;
      in_cols_p = out_cols;
      TPP_ASSERT(in_rows_p % BS == 0, "N2VTPP: unaligned number of rows\n");
      if (BS == 1) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
      } else if (BS == 2) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
      } else if (BS == 4) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
      } else {
        TPP_ASSERT(false, "N2VTPP: unsupported packing size (%d)\n", BS);
      }
    } else {
      in_rows_p = out_cols;
      in_cols_p = out_rows;
      if (dtype != LIBXSMM_DATATYPE_F32) {
        if (xtype == XformTPP::XFORM_XPOSE_TPP) {
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
        } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
          // unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT;
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeN2VTPP: uneven number of cols\n");
        } else {
          if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T;
          } else {
            TPP_ASSERT(false, "V2VTPP: unsupported packing size (%d)\n", BS);
          }
          TPP_ASSERT(in_rows % BS == 0, "XposeV2VTPP: uneven number of rows\n");
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeV2VTPP: uneven number of cols\n");
        }
      } else {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
      }
    }
    TPP_ASSERT(
        (in_rows_p >= in_rows && in_cols_p >= in_cols),
        "Invalid output rows or cols value\n");
    TPP_ASSERT(
        in_rows_p == in_rows || in_cols_p == in_cols,
        "Padding can only be done in rows or cols\n");

    if (xtype != XformTPP::XFORM_XPOSE_N2V_TPP) {
      int ld = (in_rows_p != in_rows || in_cols_p != in_cols) ? in_cols_p : ldi;
      kernel = XformTPP(in_rows_p, in_cols_p, ld, ldo, dtype, unary_type);
    } else {
      // LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT not implemented so use
      // workaround...
      kernel = XformTPP(
          in_rows_p,
          in_cols_p / BS,
          ldi / BS,
          ldo,
          ((dtype == LIBXSMM_DATATYPE_BF16 && BS == 4) ||
           (dtype == LIBXSMM_DATATYPE_BF8 && BS == 8))
              ? LIBXSMM_DATATYPE_F64
              : LIBXSMM_DATATYPE_F32,
          unary_type);
    }

    if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows / BS, in_cols * BS, ldi * BS, in_cols_p * BS);
      zero = SetZeroTPP<T>(
          in_rows / BS, (in_cols_p - in_cols) * BS, in_cols_p * BS);
      zero_offset = in_cols * BS;
    } else if (/*(xtype == XformTPP::XFORM_N2V_TPP ||
         xtype == XformTPP::XFORM_XPOSE_TPP) &&*/
               in_rows_p != in_rows) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols);
      zero = SetZeroTPP<T>(in_rows_p - in_rows, in_cols);
      zero_offset = in_rows * in_cols;
    } else if (
        /*xtype == XformTPP::XFORM_XPOSE_N2V_TPP &&*/ in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols_p);
      zero = SetZeroTPP<T>(in_rows, in_cols_p - in_cols, in_cols_p);
      zero_offset = in_cols;
    }
    if (!std::is_same<T, float>::value)
      cvt = ConvertTPP<float, T>(in_rows, in_cols);
  }
  void operator()(T* in, T* out) {
    if (in != out) {
      if (in_rows_p != in_rows || in_cols_p != in_cols) {
        T tmp[in_rows_p * in_cols_p];
        cpy(in, tmp);
        zero(tmp + zero_offset);
        kernel((void*)tmp, (void*)out);
      } else {
        kernel((void*)in, (void*)out);
      }
    }
  }
  void ref(T* in, T* out) {
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < in_rows / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols && i * BS + k < out_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }

  template <
      typename T1 = T,
      std::enable_if_t<!std::is_same<T1, float>::value, bool> = true>
  void operator()(float* in, T1* out) {
    T1 tmp2[in_rows * in_cols];
    cvt(in, tmp2);
    if (in_rows_p != in_rows || in_cols_p != in_cols) {
      T tmp[in_rows_p * in_cols_p];
      cpy(tmp2, tmp);
      zero(tmp + zero_offset);
      kernel((void*)tmp, (void*)out);
    } else {
      kernel((void*)tmp2, (void*)out);
    }
  }
  template <
      typename T1 = T,
      std::enable_if_t<!std::is_same<T1, float>::value, bool> = true>
  void ref(float* in, T1* out) {
    auto BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < out_cols / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }
  void operator()(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }
  template <
      typename T1 = T,
      std::enable_if_t<!std::is_same<T1, float>::value, bool> = true>
  void operator()(int count, long str_in, long str_out, float* in, T1* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  template <
      typename T1 = T,
      std::enable_if_t<!std::is_same<T1, float>::value, bool> = true>
  void ref(int count, long str_in, long str_out, float* in, T1* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }

 private:
  libxsmm_blasint in_rows = 0;
  libxsmm_blasint in_cols = 0;
  libxsmm_blasint out_rows = 0;
  libxsmm_blasint out_cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int in_rows_p = 0;
  int in_cols_p = 0;
  XformTPP::XFORM_TYPE xtype;
  libxsmm_datatype dtype;
  int zero_offset = 0;
  XformTPP kernel;
  ConvertTPP<float, T> cvt;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <
    typename Tin,
    typename Tout,
    typename Tw = Tin,
    typename Tcomp = typename std::conditional<
        std::is_same_v<Tout, int> || std::is_same_v<Tin, int8_t>,
        int,
        float>::type>
class BrgemmTPP {
 public:
  BrgemmTPP() {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta = 1.0,
      int a_trans = 0,
      int unroll_hint = 0)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            unroll_hint) {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint,
      int b_vnni = 1,
      int b_trans = 0,
      int a_vnni = 1)
      : M(M),
        N(N),
        K(K),
        str_a(str_a),
        str_b(str_b),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        beta(beta),
        a_trans(a_trans),
        b_trans(b_trans),
        unroll_hint(unroll_hint),
        a_vnni(a_vnni),
        b_vnni(b_vnni),
        k_gemm_with_tc(this, 0),
        k_cfg(this, 1),
        k_rls(this, 2),
        k_gemm_no_tc(this, 3) {}
  void config(void* ptr_ = nullptr) {
    libxsmm_tilecfg_state* ptr = (libxsmm_tilecfg_state*)ptr_;
    if (!ptr && omp_get_thread_num() == 0)
      ptr = &l_tilestate;
    k_cfg(ptr);
  }
  void release(void* ptr_ = nullptr) {
    libxsmm_tilecfg_state* ptr = (libxsmm_tilecfg_state*)ptr_;
    if (!ptr && omp_get_thread_num() == 0)
      ptr = &l_tilestate;
    k_rls(ptr);
  }
  void operator()(
      Tin* A,
      Tw* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }
  void operator()(
      Tin* A,
      Tw* B,
      Tw* B_scales,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.a.tertiary = (void*)B_scales;
    gemm_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }
  void operator()(
      Tin* A,
      float* A_scales,
      Tw* B,
      Tw* B_scales,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.a.tertiary = (void*)B_scales;
    gemm_param.b.primary = (void*)A;
    gemm_param.b.tertiary = (void*)A_scales;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }
  void ref(
      Tin* A,
      Tw* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    // VNNI blocking is based on input type to allow weight only quantization
    auto dtype = XsmmDtype<Tin>();
    for (uint64_t c = 0; c < count; c++) {
      auto A_ = &A[c * str_a];
      auto B_ = &B[c * str_b];
      if (std::is_same<Tw, float>::value || b_vnni == 0) {
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            if (beta == 0.0 && c == 0)
              C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
              if (a_trans == 1) {
                C[i * ldc + j] +=
                    (float)A_[k * lda + i] * (float)B_[k * ldb + j];
              } else {
                C[i * ldc + j] +=
                    (float)A_[i * lda + k] * (float)B_[k * ldb + j];
              }
            }
          }
        }
      } else {
        const int BS = xsmm_get_vnni_block_size(dtype);
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            Tcomp sum =
                ((beta == 0.0 && c == 0) ? 0.0f : (Tcomp)C[i * ldc + j]);
            for (int k = 0; k < K / BS; k++) {
              for (int b = 0; b < BS; b++) {
                if (a_trans == 1) {
                  sum += (Tcomp)A_[k * lda * BS + i * BS + b] *
                      (Tcomp)B_[k * ldb * BS + j * BS + b];
                } else {
                  sum += (Tcomp)A_[i * lda + k * BS + b] *
                      (Tcomp)B_[k * ldb * BS + j * BS + b];
                }
              }
            }
            C[i * ldc + j] = (Tout)sum;
          }
        }
      }
    }
  }

  long flops() {
    return 2L * M * N * K;
  }

  class BrgemmKernel : public BaseTPP {
   public:
    BrgemmKernel() {}
    BrgemmKernel(BrgemmTPP* p, int config) : p(p), config(config) {
      auto dt_in = XsmmDtype<Tin>();
      auto dt_wt = XsmmDtype<Tw>();
      auto dt_out = XsmmDtype<Tout>();
      long type = -1;
      if (dt_in == LIBXSMM_DATATYPE_F32 && dt_wt == LIBXSMM_DATATYPE_F32) {
        TPP_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
        type = 0;
      } else if (dt_out == LIBXSMM_DATATYPE_F32) {
        type = 1;
      } else {
        type = 2;
      }
      // if (type != 0)
      //   TPP_ASSERT(
      //       p->a_trans == 0, "A Transpose supported only for FP32 BRGEMM\n");
      brgemm_type = type;
      kernel.gemm = (libxsmm_gemmfunction)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_gemm_param* gemm_param) {
      if (!initialized)
        return;
      kernel.gemm(gemm_param);
    }

    void operator()(libxsmm_tilecfg_state* state_ptr) {
      if (!initialized)
        return;
      kernel.tilecfg(state_ptr);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "brgemm_m%ld_n%ld_k%ld_a%ld_b%ld_t%ld_beta%d_at%d_uh%d_ld_a%ld_b%ld_c%ld_cfg%d_bv%d_dti%d_dtw%d_dto%d_dtc%d",
          p->M,
          p->N,
          p->K,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->unroll_hint,
          (long)p->lda,
          (long)p->ldb,
          (long)p->ldc,
          config,
          p->b_vnni,
          XsmmDtype<Tin>(),
          XsmmDtype<Tw>(),
          XsmmDtype<Tout>(),
          XsmmDtype<Tcomp>());
      return std::string(hash);
    }
    void* build_kernel() override {
      // float alpha = 1.0;
      libxsmm_gemm_shape l_shape;
      libxsmm_gemm_batch_reduce_config l_brconfig;
      libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      libxsmm_bitfield l_prefetch_flags = 0;
      libxsmm_xmmfunction l_test_jit = {NULL};

      if (p->a_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
      if (brgemm_type != 0) {
        if (p->b_vnni)
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
        if (p->b_vnni >= 2) {
          if (p->b_vnni == 2)
            l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
          else if (p->b_vnni == 8)
            l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI8_INTLV;
          else
            TPP_ASSERT(0, "Invalid B VNNI type\n");
          TPP_ASSERT(
              (std::is_same<Tw, uint8_t>::value),
              "MXFP4 must use uint8_t for weights\n");
        }
        if (p->a_trans == 1 && p->a_vnni == 1) {
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_B;
        }
      }
      if (p->beta == 0)
        l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
      if (p->b_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;

      // config = 0 - normal
      // config = 1 - no tile release
      // config = 2 - no tile config
      // config = 3 - brgemm with no tile config or release
      if (config == 1) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
      } else if (config == 2) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
      } else if (config == 3) {
        l_flags |=
            (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
             LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      }

      /* setting update GEMM struct */
      l_shape.m = p->N;
      l_shape.n = p->M;
      l_shape.k = p->K;
      l_shape.lda = p->ldb;
      l_shape.ldb = p->lda;
      l_shape.ldc = p->ldc;
      l_shape.a_in_type = XsmmDtype<Tw>();
      l_shape.b_in_type = XsmmDtype<Tin>();
      l_shape.out_type = XsmmDtype<Tout>();
      l_shape.comp_type = XsmmDtype<Tcomp>();

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
      l_brconfig.br_stride_a_hint = (p->b_vnni == 2 || p->b_vnni == 8)
          ? (p->str_b * sizeof(Tw)) / 2
          : p->str_b * sizeof(Tw);
      l_brconfig.br_stride_b_hint = p->str_a * sizeof(Tin);
      l_brconfig.br_unroll_hint = p->unroll_hint;

      if (config == 1 || config == 2) {
        l_test_jit.tilecfg = libxsmm_dispatch_tilecfg_gemm(l_shape, l_flags);
        return (void*)l_test_jit.tilecfg;
      } else {
        l_test_jit.gemm = libxsmm_dispatch_brgemm(
            l_shape, l_flags, l_prefetch_flags, l_brconfig);
        return (void*)l_test_jit.gemm;
      }
    }

   private:
    BrgemmTPP* p;
    int config;
    libxsmm_xmmfunction kernel;
    long brgemm_type = -1;
  };

 private:
  long M, N, K, str_a, str_b;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  float beta;
  int a_trans;
  int b_trans;
  long brgemm_type = -1;
  int unroll_hint;
  int a_vnni;
  int b_vnni;
  libxsmm_tilecfg_state l_tilestate;
  BrgemmKernel k_gemm_with_tc;
  BrgemmKernel k_cfg;
  BrgemmKernel k_rls;
  BrgemmKernel k_gemm_no_tc;
};

template <typename Tin, typename Tout = Tin>
class GeluFwdTPP {
 public:
  GeluFwdTPP() {}
  GeluFwdTPP(int N) : GeluFwdTPP(1, N) {}
  GeluFwdTPP(int M, int N) : GeluFwdTPP(M, N, N, N) {}
  GeluFwdTPP(int M, int N, int ldi, int ldo)
      : M(M),
        N(N),
        ldi(ldi),
        ldo(ldo),
        kernel(
            M,
            N,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_GELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
#ifdef __AVX512F__
    for (int j = 0; j < M; j++) {
      int i;
      for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
        auto vin = _mm512_loadu_ps_auto(&in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_storeu_ps_auto(&out[j * ldo + i], vout);
      }
      if (i < N) {
        int rem = N - i;
        __mmask16 mask = (1 << rem) - 1;
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_mask_storeu_ps_auto(&out[j * ldo + i], mask, vout);
      }
    }
#else
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        float x = in[j * ldi + i];
        out[j * ldo + i] = (erff(x / sqrtf(2.0)) + 1.0) * 0.5 * x;
      }
    }
#endif
  }

 private:
  int M = 0;
  int N = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename T1, typename T2 = T1, typename T3 = T1>
class GeluBwdTPP : public BaseTPP {
 public:
  GeluBwdTPP() {}
  GeluBwdTPP(int N) : N(N) {
    kernel = (libxsmm_meqn_function)get_kernel();
    initialized = true;
  }
  void operator()(T1* gout, T2* in, T3* gin) {
    if (!initialized)
      return;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[2];
    arg_array[0].primary = (void*)gout;
    arg_array[1].primary = (void*)in;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)gin;

    kernel(&eqn_param);
  }
  void ref(T1* gout, T2* in, T3* gin) {
#ifdef __AVX512F__
    int i;
    for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
      auto vgout = _mm512_loadu_ps_auto(&gout[i]);
      auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_storeu_ps_auto(&gin[i], vout);
    }
    if (i < N) {
      int rem = N - i;
      __mmask16 mask = (1 << rem) - 1;
      auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
      auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
    }
#else
    constexpr float PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
      float x = in[i];
      gin[i] = (float)gout[i] *
          (0.5 + 0.5 * erff(x / sqrtf(2.0)) +
           x / (sqrtf(2.0 * PI)) * expf(-0.5 * x * x));
    }
#endif
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "gelu_bwd_eqn_t%d_%d_%d_i%d",
        XsmmDtype<T1>(),
        XsmmDtype<T2>(),
        XsmmDtype<T3>(),
        N);
    return std::string(hash);
  }
  void* build_kernel() override {
    auto dt1 = XsmmDtype<T1>();
    auto dt2 = XsmmDtype<T2>();
    auto dt3 = XsmmDtype<T3>();
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, N, 1, N, 0, 0, dt1);
    meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV);
    meqn_push_arg(my_eqn0, N, 1, N, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(N, 1, &ld, dt3, my_eqn0);
  }

 private:
  int N = 0;
  libxsmm_meqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class SigmoidFwdTPP {
 public:
  SigmoidFwdTPP() {}
  SigmoidFwdTPP(int rows, int cols) : SigmoidFwdTPP(rows, cols, cols, cols) {}
  SigmoidFwdTPP(int rows, int cols, int ldi, int ldo)
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
            LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = 1. / (1. + exp(-in[i * ldi + j]));
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
class SigmoidBwdTPP {
 public:
  SigmoidBwdTPP() {}
  SigmoidBwdTPP(int rows, int cols) : SigmoidBwdTPP(rows, cols, cols, cols) {}
  SigmoidBwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        ksub(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_SUB),
        kmul(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in_grad_act, Tin* in_act, Tout* out) {
    Tin val = 1.0;
    ksub(&val, (void*)in_act, (void*)out);
    kmul((void*)out, (void*)in_act, (void*)out);
    kmul((void*)out, (void*)in_grad_act, (void*)out);
  }
  void ref(Tin* in_grad_act, Tin* in_act, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int outIndex = i * ldo + j;
        int inIndex = i * ldi + j;
        out[outIndex] =
            in_grad_act[inIndex] * (1.0 - in_act[inIndex]) * in_act[inIndex];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP ksub;
  BinaryTPP kmul;
};

template <typename Tin, typename Tout = Tin>
class ReLUFwdTPP {
 public:
  ReLUFwdTPP() {}
  ReLUFwdTPP(int N, bool bm) : ReLUFwdTPP(1, N, bm) {}
  ReLUFwdTPP(int rows, int cols, bool bm)
      : ReLUFwdTPP(rows, cols, cols, cols, bm) {}
  ReLUFwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
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
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ReLUBwdTPP {
 public:
  ReLUBwdTPP() {}
  ReLUBwdTPP(int N, bool bm) : ReLUBwdTPP(1, N, bm) {}
  ReLUBwdTPP(int rows, int cols, bool bm)
      : ReLUBwdTPP(rows, cols, cols, cols, bm) {}
  ReLUBwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        bm(bm),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  bool bm;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUFwdTPP {
 public:
  ELUFwdTPP() {}
  ELUFwdTPP(int N, float alpha) : ELUFwdTPP(1, N, alpha) {}
  ELUFwdTPP(int rows, int cols, float alpha)
      : ELUFwdTPP(rows, cols, cols, cols, alpha) {}
  ELUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in[i * ldi + j] > 0 ? in[i * ldi + j]
                                               : a * (exp(in[i * ldi + j]) - 1);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUBwdTPP {
 public:
  ELUBwdTPP() {}
  ELUBwdTPP(int N, float alpha) : ELUBwdTPP(1, N, alpha) {}
  ELUBwdTPP(int rows, int cols, float alpha)
      : ELUBwdTPP(rows, cols, cols, cols, alpha) {}
  ELUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel(in, in2, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in2[i * ldi + j] > 0
            ? in[i * ldi + j]
            : in[i * ldi + j] * in2[i * ldi + j] + a * in[i * ldi + j];
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUFwdTPP {
 public:
  LeakyReLUFwdTPP() {}
  LeakyReLUFwdTPP(int N, float alpha) : LeakyReLUFwdTPP(1, N, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, float alpha)
      : LeakyReLUFwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    float a = alpha;
    // std::cout << " op: " << out << " inp: "<< in << std::endl;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j] > 0 ? (Tout)in[i * ldi + j]
                                               : (Tout)(a * (in[i * ldi + j]));
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUBwdTPP {
 public:
  LeakyReLUBwdTPP() {}
  LeakyReLUBwdTPP(int N, float alpha) : LeakyReLUBwdTPP(1, N, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, float alpha)
      : LeakyReLUBwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, mask, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2, short* mask = NULL) {
    float a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        out[i * ldo + j] =
            in2[i * ldi + j] > 0 ? (Tout)grad_out : (Tout)(a * grad_out);
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename T>
class SiLUFwdTPP {
 public:
  SiLUFwdTPP() {}
  SiLUFwdTPP(int N) : SiLUFwdTPP(1, N) {}
  SiLUFwdTPP(int rows, int cols) : SiLUFwdTPP(rows, cols, cols, cols) {}
  SiLUFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        sigmoid(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_SIGMOID),
        mul(rows,
            cols,
            ldi,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    sigmoid((void*)in, (void*)sigout);
    mul((void*)in, (void*)sigout, (void*)out);
  }
  void ref(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        sigout[i * ldo + j] = 1. / (1. + exp(-in[i * ldi + j]));
        out[i * ldo + j] = in[i * ldi + j] * sigout[i * ldo + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP sigmoid;
  BinaryTPP mul;
};

template <typename Tin, typename Tout = Tin>
class SiLUBwdTPP : public BaseTPP {
 public:
  SiLUBwdTPP() {}
  SiLUBwdTPP(int N) : SiLUBwdTPP(1, N) {}
  SiLUBwdTPP(int rows, int cols) : SiLUBwdTPP(rows, cols, cols, cols) {}
  SiLUBwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows), cols(cols), ldi(ldi), ldo(ldo) {
    kernel = (libxsmm_meqn_function)get_kernel();
    initialized = true;
  }
  void operator()(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    if (!initialized)
      return;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    float one = 1.;
    arg_array[0].primary = (void*)in;
    arg_array[1].primary = (void*)in2;
    arg_array[2].primary = (void*)in3;
    arg_array[3].primary = (void*)&one;
    arg_array[4].primary = (void*)in2;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;

    kernel(&eqn_param);
  }
  void ref(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        float si = in2[i * ldi + j];
        float fout = in3[i * ldi + j];

        out[i] = grad_out * (si + fout * (1 - si));
      }
    }
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "silu_bwd_eqn_%d_%d", rows, cols);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 0, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 1, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_SUB,
        LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 3, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 4, 0, LIBXSMM_DATATYPE_F32);

    auto func0 = meqn_dispatch(cols, rows, &ldo, XsmmDtype<Tout>(), my_eqn0);
    return (void*)func0;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  libxsmm_meqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class DropOutFwdTPP {
 public:
  DropOutFwdTPP() {}
  DropOutFwdTPP(int N, float p) : DropOutFwdTPP(1, N, p) {}
  DropOutFwdTPP(int rows, int cols, float p)
      : DropOutFwdTPP(rows, cols, cols, cols, p) {}
  DropOutFwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {}
  void operator()(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }
  void ref(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class DropOutBwdTPP {
 public:
  DropOutBwdTPP() {}
  DropOutBwdTPP(int N, float p) : DropOutBwdTPP(1, N, p) {}
  DropOutBwdTPP(int rows, int cols, float p)
      : DropOutBwdTPP(rows, cols, cols, cols, p) {}
  DropOutBwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {}
  void operator()(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class SoftMaxFwdTPP {
 public:
  SoftMaxFwdTPP() {}
  SoftMaxFwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3), eqn1(S1, S2, S3) {}
  void operator()(Tin* in, Tout* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      eqn0(&in[s2 * S3], tmp);
      eqn1(tmp, &out[s2 * S3]);
    }
  }
  void ref(Tin* pinp, Tout* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              // __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS (_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              // __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS (_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      sum = 1.0 / sum;
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          _mm512_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          _mm512_mask_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      sum = 1.0 / sum;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = tmp[s1][s3] * sum;
          // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
          // s1, s2, s3, S2, S3), 1 );
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
        }
      }
    }
#endif
  }
  class Eqn0 : BaseTPP {
   public:
    Eqn0() {}
    Eqn0(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(Tin* in, float* out) {
      if (!initialized)
        return;
      libxsmm_meqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_fwd_eqn0_ti%d_to%d_S1%d_S2%d_S3%d",
          XsmmDtype<Tin>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_in = XsmmDtype<Tin>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
      meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP);
      meqn_push_binary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_BINARY_SUB,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(
          S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_meqn_function kernel = NULL;
  };

  class Eqn1 : BaseTPP {
   public:
    Eqn1() {}
    Eqn1(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(float* in, Tout* out) {
      if (!initialized)
        return;
      libxsmm_meqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_fwd_eqn1_ti%d_to%d_S1%d_S2%d_S3%d",
          LIBXSMM_DATATYPE_F32,
          XsmmDtype<Tout>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_out = XsmmDtype<Tout>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn1 = libxsmm_meqn_create();
      meqn_push_binary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_BINARY_MUL,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      /*debug_print_eqn_tree( my_eqn1 );*/
      return (void*)meqn_dispatch(S3, S1, &ld, dt_out, my_eqn1);
    }

   private:
    int S1, S2, S3;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn0 eqn0;
  Eqn1 eqn1;
};

template <typename T1, typename T2, typename T3>
class SoftMaxBwdTPP {
 public:
  SoftMaxBwdTPP() {}
  SoftMaxBwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3, 0), eqn1(S1, S2, S3, 1) {}
  void operator()(T1* gin, T2* gout, T3* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      libxsmm_meqn_param eqn_param;
      libxsmm_matrix_arg arg_array[2];
      arg_array[0].primary = (void*)&gout[s2 * S3];
      arg_array[1].primary = (void*)&out[s2 * S3];
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)tmp;
      eqn0(&eqn_param);

      arg_array[0].primary = (void*)tmp;
      eqn_param.output.primary = (void*)&gin[s2 * S3];
      eqn1(&eqn_param);
    }
  }
  void ref(T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_bwd_eqn%d_t1%d_t2%d_t3%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T2>(),
          XsmmDtype<T3>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_meqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, S1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, S1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_meqn_create();
#if 1
        meqn_push_ternary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_TERNARY_NMULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
#else
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
#endif
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, S1, &ld, dt_1, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn eqn0, eqn1;
};

template <typename Tin, typename Tout>
class VarSoftMaxFwdTPP {
 public:
  VarSoftMaxFwdTPP() {}
  VarSoftMaxFwdTPP(int S2, int S3, bool isFlash = false)
      : S2(S2),
        S3(S3),
        isFlash(isFlash),
        kmax(
            1,
            S3,
            S3,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX),
        ksub(
            1,
            S3,
            S3,
            1,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_SUB),
        kexp(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_EXP),
        ksum(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        kmul(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL),
        kcpy(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(
      int S1,
      Tin* in,
      Tout* out,
      float* max_buf = nullptr,
      float* sum_buf = nullptr,
      float* omax_buf = nullptr) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      float max = (omax_buf && isFlash) ? omax_buf[s2] : (float)in[s2 * S3];
      float sum = 0.0f;
      for (int s1 = 0; s1 < S1; s1++) {
        float rmax = 0;
        kmax(&in[s1 * S2 * S3 + s2 * S3], &rmax);
        if (max < rmax)
          max = rmax;
      }
      for (int s1 = 0; s1 < S1; s1++) {
        LIBXSMM_ALIGNED(float tmp2[S3], 64);
        ksub(&in[s1 * S2 * S3 + s2 * S3], &max, tmp2);
        kexp(tmp2, &tmp[s1 * S3]);
        float lsum;
        ksum(&tmp[s1 * S3], &lsum);
        sum += lsum;
      }
      if (max_buf)
        max_buf[s2] = max;
      if (sum_buf)
        sum_buf[s2] = sum;
      if (!isFlash) {
        sum = 1.0 / sum;
        for (int s1 = 0; s1 < S1; s1++) {
          kmul(&tmp[s1 * S3], &sum, &out[s1 * S2 * S3 + s2 * S3]);
        }
      } else {
        for (int s1 = 0; s1 < S1; s1++) {
          kcpy(&tmp[s1 * S3], &out[s1 * S2 * S3 + s2 * S3]);
        }
      }
    }
  }
  void ref(
      int S1,
      Tin* pinp,
      Tout* pout,
      float* max_buf = nullptr,
      float* sum_buf = nullptr,
      float* omax_buf = nullptr) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max = (omax_buf && isFlash)
          ? omax_buf[s2]
          : upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      if (max_buf)
        max_buf[s2] = max;
      if (sum_buf)
        sum_buf[s2] = sum;
      if (!isFlash) {
        sum = 1.0 / sum;
        vsum = _mm512_set1_ps(sum);
        for (s1 = 0; s1 < S1; s1++) {
          for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
            _mm512_storeu_ps_auto(
                &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
                _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
          }
          if (s3 < S3) {
            int rem = S3 - s3;
            __mmask16 mask = (1 << rem) - 1;
            _mm512_mask_storeu_ps_auto(
                &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
                mask,
                _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
          }
        }
      } else {
        for (s1 = 0; s1 < S1; s1++) {
          for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
            _mm512_storeu_ps_auto(
                &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
                _mm512_loadu_ps(&tmp[s1][s3]));
          }
          if (s3 < S3) {
            int rem = S3 - s3;
            __mmask16 mask = (1 << rem) - 1;
            _mm512_mask_storeu_ps_auto(
                &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
                mask,
                _mm512_maskz_loadu_ps(mask, &tmp[s1][s3]));
          }
        }
      }
    }
#else
    // #warning "Not using AVX512 path for VarSoftMax"
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max = (omax_buf && isFlash)
          ? omax_buf[s2]
          : upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      if (max_buf)
        max_buf[s2] = max;
      if (sum_buf)
        sum_buf[s2] = sum;
      if (!isFlash) {
        sum = 1.0 / sum;
        for (s1 = 0; s1 < S1; s1++) {
          for (s3 = 0; s3 < S3; s3++) {
            float cur = tmp[s1][s3] * sum;
            // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
            // s1, s2, s3, S2, S3), 1 );
            LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
          }
        }
      } else {
        for (s1 = 0; s1 < S1; s1++) {
          for (s3 = 0; s3 < S3; s3++) {
            LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = tmp[s1][s3];
          }
        }
      }
    }
#endif
  }

 private:
  int S2, S3;
  bool isFlash;
  UnaryTPP kmax;
  BinaryTPP ksub;
  UnaryTPP kexp;
  UnaryTPP ksum;
  BinaryTPP kmul;
  UnaryTPP kcpy;
};

template <typename T1, typename T2, typename T3>
class VarSoftMaxBwdTPP {
 public:
  VarSoftMaxBwdTPP() {}
  VarSoftMaxBwdTPP(int S2, int S3) : S2(S2), S3(S3), eqn0(S3, 0), eqn1(S3, 1) {}
  void operator()(int S1, T1* gin, T2* gout, T3* out) {
    long S23 = S2 * S3;
    for (int s2 = 0; s2 < S2; s2++) {
      float tmp = 0.0f;
      libxsmm_meqn_param eqn_param;
      libxsmm_matrix_arg arg_array[3];
      arg_array[2].primary = (void*)&tmp;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)&tmp;
      for (int s1 = 0; s1 < S1; s1++) {
        long ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn0(&eqn_param);
      }
      for (int s1 = 0; s1 < S1; s1++) {
        long ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn_param.output.primary = (void*)&gin[ind];
        eqn1(&eqn_param);
      }
    }
  }
  void ref(int S1, T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S3, int eqn_no) : S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "varsoftmax_bwd_eqn%d_t1%d_t2%d_t3%d_S3%d",
          eqn_no,
          XsmmDtype<T1>(),
          XsmmDtype<T2>(),
          XsmmDtype<T3>(),
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S3;
      libxsmm_meqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, 1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, 1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn3, S3, 1, ld, 1, 0, dt_3);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_SUB,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn3, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, 1, &ld, dt_1, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    int S3, eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S2, S3;
  Eqn eqn0, eqn1;
};

template <typename T>
class SoftMaxFixUpTPP {
 public:
  SoftMaxFixUpTPP() {}
  SoftMaxFixUpTPP(int S2, int S3, bool isFlash)
      : S2(S2), S3(S3), isFlash(isFlash), eqn(S3, isFlash) {}
  void operator()(
      T* cur,
      T* out,
      float* cmax,
      float* csum,
      float* omax,
      float* osum) {
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[4];
    eqn_param.inputs = arg_array;

    for (int s2 = 0; s2 < S2; s2++) {
      if (!isFlash) {
        float nmax = std::max(cmax[s2], omax[s2]);
        float oexp = expf(omax[s2] - nmax);
        float cexp = expf(cmax[s2] - nmax);
        float nsum = cexp * csum[s2] + oexp * osum[s2];
        float rsum = 1.0 / nsum;
        float oscale = osum[s2] * oexp * rsum;
        float cscale = csum[s2] * cexp * rsum;
        arg_array[0].primary = &out[s2 * S3];
        arg_array[1].primary = &oscale;
        arg_array[2].primary = &cur[s2 * S3];
        arg_array[3].primary = &cscale;
        eqn_param.output.primary = (void*)&out[s2 * S3];
        eqn(&eqn_param);
        omax[s2] = nmax;
        osum[s2] = nsum;
      } else {
        float nmax = std::max(cmax[s2], omax[s2]);
        TPP_ASSERT(
            nmax == cmax[s2],
            "Flash attention nmax (%g) must equal cmax (%g)\n",
            nmax,
            cmax[s2]);
        float oexp = exp(omax[s2] - nmax);
        // float cexp = exp(cmax[s2] - nmax); // must be 1.0
        float nsum = csum[s2] + oexp * osum[s2];
        arg_array[0].primary = &out[s2 * S3];
        arg_array[1].primary = &oexp;
        arg_array[2].primary = &cur[s2 * S3];
        eqn_param.output.primary = (void*)&out[s2 * S3];
        eqn(&eqn_param);
        omax[s2] = nmax;
        osum[s2] = nsum;
      }
    }
  }

  void ref(T* cur, T* out, float* cmax, float* csum, float* omax, float* osum) {
    for (int s2 = 0; s2 < S2; s2++) {
      if (!isFlash) {
        float nmax = std::max(cmax[s2], omax[s2]);
        float oexp = exp(omax[s2] - nmax);
        float cexp = exp(cmax[s2] - nmax);
        float nsum = cexp * csum[s2] + oexp * osum[s2];
        float rsum = 1.0 / nsum;
        float oscale = osum[s2] * oexp * rsum;
        float cscale = csum[s2] * cexp * rsum;
        for (int s3 = 0; s3 < S3; s3++) {
          out[s2 * S3 + s3] =
              oscale * out[s2 * S3 + s3] + cscale * cur[s2 * S3 + s3];
        }
        omax[s2] = nmax;
        osum[s2] = nsum;
      } else {
        float nmax = std::max(cmax[s2], omax[s2]);
        TPP_ASSERT(nmax == cmax[s2], "Flash attention nmax must equal cmax\n");
        float oexp = exp(omax[s2] - nmax);
        // float cexp = exp(cmax[s2] - nmax); // must be 1.0
        float nsum = csum[s2] + oexp * osum[s2];
        for (int s3 = 0; s3 < S3; s3++) {
          out[s2 * S3 + s3] = oexp * out[s2 * S3 + s3] + cur[s2 * S3 + s3];
        }
        omax[s2] = nmax;
        osum[s2] = nsum;
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S3, bool isFlash) : S3(S3), isFlash(isFlash) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_fixup_eqn_t1%d_S3%d_F%d",
          XsmmDtype<T>(),
          S3,
          isFlash ? 1 : 0);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint ld = S3;
      libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
      if (!isFlash) {
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_NONE);
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, S3, 1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, S3, 1, ld, 2, 0, in_dt);
        meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 3, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn0); // printf
        return (void*)meqn_dispatch(S3, 1, &ld, out_dt, my_eqn0);
      } else {
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, S3, 1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn0, S3, 1, ld, 2, 0, in_dt);
        debug_print_eqn_tree(my_eqn0); // printf
        return (void*)meqn_dispatch(S3, 1, &ld, out_dt, my_eqn0);
      }
    }

   private:
    int S3;
    bool isFlash;
    libxsmm_meqn_function kernel = NULL;
  };

 protected:
  int S2 = 0;
  int S3 = 0;
  bool isFlash;
  Eqn eqn;
};

template <typename T>
class SoftMaxFlashScaleTPP {
 public:
  SoftMaxFlashScaleTPP() {}
  SoftMaxFlashScaleTPP(int S2, int S3, bool isFlash)
      : S2(S2),
        S3(S3),
        isFlash(isFlash),
        div_kernel(
            S2,
            S3,
            S3,
            1,
            S3,
            XsmmDtype<T>(),
            XsmmDtype<float>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_DIV) {}

  void operator()(T* buf, float* osum) {
    if (!isFlash)
      return;
    div_kernel(buf, osum, buf);
  }

  void ref(T* buf, float* osum) {
    if (!isFlash)
      return;
    for (int s2 = 0; s2 < S2; s2++) {
      float isum = 1.0f / osum[s2];
      for (int s3 = 0; s3 < S3; s3++) {
        buf[s2 * S3 + s3] *= isum;
      }
    }
  }

 protected:
  int S2 = 0;
  int S3 = 0;
  bool isFlash;
  BinaryTPP div_kernel;
};

template <typename T, typename LT = T>
class RMSNormFwdTPP {
 public:
  RMSNormFwdTPP() {}
  RMSNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(T* inp, LT* gamma, float* var, T* out) {
    LIBXSMM_ALIGNED(float tmp[S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float v, s;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[3];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = (void*)gamma;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&v);
      v = v * c;
      // v = LIBXSMM_MAX(v, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      if (var)
        var[s2] = v;
      s = v;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, LT* pgamma, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    LIBXSMM_VLA_DECL(2, LT, gamma, pgamma, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      v = v * c;
      // v = LIBXSMM_MAX(v, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      float s = v;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "rmsnorm_fwd_eqn_t1%d_t2%d_S1%d_S2%d_S3%d",
          XsmmDtype<T>(),
          XsmmDtype<LT>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto bg_dt = XsmmDtype<LT>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
      meqn_push_binary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_BINARY_MUL,
          LIBXSMM_MELTW_FLAG_BINARY_NONE);
      meqn_push_binary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_BINARY_MUL,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_BINARY_NONE);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 2, 0, bg_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T, typename LT = T>
class LayerNormFwdTPP {
 public:
  LayerNormFwdTPP() {}
  LayerNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(
      T* inp,
      LT* gamma,
      LT* beta,
      float* mean,
      float* var,
      T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      if (mean)
        mean[s2] = m;
      if (var)
        var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, LT* pgamma, LT* pbeta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    LIBXSMM_VLA_DECL(2, LT, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, LT, beta, pbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      if (mean)
        mean[s2] = m;
      if (var)
        var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3);
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "layernorm_fwd_eqn_t1%d_t2%d_S1%d_S2%d_S3%d",
          XsmmDtype<T>(),
          XsmmDtype<LT>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto bg_dt = XsmmDtype<LT>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 3, 0, bg_dt);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 4, 0, bg_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T, typename LT = T>
class LayerNormBwdTPP {
 public:
  LayerNormBwdTPP() {}
  LayerNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      LT* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      LT* pgamma,
      T* pdin,
      float* pdgamma,
      float* pdbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    LIBXSMM_VLA_DECL(2, LT, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) +=
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "layernorm_bwd_eqn%d_t1%d_t2%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T>(),
          XsmmDtype<LT>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto bg_dt = XsmmDtype<LT>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_meqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_meqn_create();
        meqn_push_binary_op(
            my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 6, 0, bg_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_meqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, tmp_ld, 6, 0, bg_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, S3, S1, tmp_ld, 6, 0, bg_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        TPP_ASSERT(false, "LayerNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

template <typename T>
class GroupNormFwdTPP {
 public:
  GroupNormFwdTPP() {}
  GroupNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(T* inp, T* gamma, T* beta, float* mean, float* var, T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, T* gamma, T* beta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  gamma[s1] +
              beta[s1];
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "group_norm_fwd_eqn_t%d_S1%d_S2%d_S3%d",
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, S1, 1, 3, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, S1, 1, 4, 0, in_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T>
class GroupNormBwdTPP {
 public:
  GroupNormBwdTPP() {}
  GroupNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      T* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      T* gamma,
      T* pdin,
      float* dgamma,
      float* dbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          dgamma[s1] +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          dbeta[s1] += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1] *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1];
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a * gamma[s1] +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "group_norm_bwd_eqn%d_t%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_meqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_meqn_create();
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, 1, S1, 1, 6, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_meqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(
            my_eqn4,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0 |
                LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        TPP_ASSERT(false, "GroupNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

class SplitSGDTPP : public BaseTPP {
 public:
  SplitSGDTPP() {}
  SplitSGDTPP(int N) : N(N) {
    kernel = (libxsmm_meqn_function)get_kernel();
    initialized = true;
  }
  void operator()(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
    if (!initialized)
      return;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[4];
    arg_array[0].primary = (void*)lo;
    arg_array[1].primary = (void*)hi;
    arg_array[2].primary = (void*)&lr;
    arg_array[3].primary = (void*)grad;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)lo;
    auto offset = (long long)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)&offset;

    kernel(&eqn_param);
  }
  void ref(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
#ifndef __AVX512F__
    auto dwt = (libxsmm_bfloat16*)grad;
    auto out_hi = (libxsmm_bfloat16*)hi;
    auto out_lo = (libxsmm_bfloat16*)lo;
    for (int i = 0; i < N; i++) {
      union libxsmm_bfloat16_f32 bf16_hp;
      union libxsmm_bfloat16_f32 bf16_wt;
      bf16_wt.i[0] = 0;
      bf16_wt.i[1] = dwt[i];
      bf16_hp.i[0] = out_lo[i];
      bf16_hp.i[1] = out_hi[i];
      bf16_hp.f = bf16_wt.f * lr + bf16_hp.f;
      out_lo[i] = bf16_hp.i[0];
      out_hi[i] = bf16_hp.i[1];
    }
#else
    long sz = N;
    auto vlr = _mm512_set1_ps(lr);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "split_sgd_eqn_i%d", N);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_meqn_create();
    meqn_push_unary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_UNARY_UNZIP,
        LIBXSMM_MELTW_FLAG_UNARY_NONE,
        LIBXSMM_DATATYPE_IMPLICIT);
    meqn_push_ternary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
        LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
    /* This is the "gradient" weights   */
    meqn_push_arg(my_eqn0, N, 1, ld, 3, 0, LIBXSMM_DATATYPE_BF16);
    /* This is the scalar learning rate */
    meqn_push_arg(my_eqn0, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ZIP);
    /* This is the tensor with lo bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, LIBXSMM_DATATYPE_U16);
    /* This is the tensor with hi bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 1, 0, LIBXSMM_DATATYPE_U16);
    debug_print_eqn_tree(my_eqn0);
    auto func0 = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_U16, my_eqn0);
    return (void*)func0;
  }

 private:
  int N = 0;
  libxsmm_meqn_function kernel = NULL;
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

template <typename T>
class FusedAdamWTPP {
 public:
  FusedAdamWTPP() {}
  FusedAdamWTPP(
      int N,
      float beta1,
      float beta2,
      float weight_decay,
      float eps,
      bool use_adam = false)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        use_adam(use_adam),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2),
        eqn3(this, 3) {}
  void operator()(
      T* data,
      T* grad_,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    T wd_grad[N];
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    bool adam_style_weight_decay = weight_decay > 0.0 && use_adam;
    void* grad = adam_style_weight_decay ? (void*)wd_grad : (void*)grad_;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    eqn_param.inputs = arg_array;
    if (adam_style_weight_decay) {
      arg_array[0].primary = (void*)grad_;
      arg_array[4].primary = (void*)data;
      arg_array[6].primary = (void*)&weight_decay;
      eqn_param.output.primary = (void*)grad;
      eqn3(&eqn_param);
    }

    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)data;
    arg_array[5].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)data;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      auto avg_i = exp_avg[i];
      auto avg_sq_i = exp_avg_sq[i];
      auto grad_i = grad[i];
      auto data_i = data[i];
      if (weight_decay > 0.0 && use_adam)
        grad_i = grad_i + data_i * weight_decay;
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0 && !use_adam)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data[i] = data_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(weight_decay);
    auto vweight_decay_lr = _mm512_set1_ps(lr * weight_decay);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_loadu_ps(&data[i]);
      if (weight_decay > 0.0 && use_adam)
        grad_i = _mm512_add_ps(grad_i, _mm512_mul_ps(data_i, vweight_decay));
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0 && !use_adam)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay_lr));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_storeu_ps(&data[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
      if (weight_decay > 0.0 && use_adam)
        grad_i = _mm512_add_ps(grad_i, _mm512_mul_ps(data_i, vweight_decay));
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0 && !use_adam)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay_lr));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_storeu_ps(&data[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_adamw_eqn%d_t%d_n%d_wd%d_adam%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1),
          (p->use_adam ? 1 : 0));
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      libxsmm_meqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        int use_wd = (p->weight_decay == 0.0 && !p->use_adam) ? 0 : 1;
        // Equation for data_i (with decay)
        auto my_eqn2 = libxsmm_meqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn2, N, 1, ld, 4, 0, in_dt); // data_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else if (eqn_no == 3) {
        int use_wd = (p->weight_decay == 0.0 && p->use_adam) ? 0 : 1;
        if (!use_wd)
          return (void*)0xdeaddead;
        auto my_eqn3 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, N, 1, ld, 4, 0, in_dt); // data_i
        meqn_push_arg(
            my_eqn3, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32); // weight_decay
        meqn_push_arg(my_eqn3, N, 1, ld, 0, 0, in_dt); // grad_i
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedAdamWTPP* p;
    int eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  bool use_adam;
  Eqn eqn0, eqn1, eqn2, eqn3;
  friend class Eqn;
};

class FusedSplitAdamWTPP {
 public:
  typedef bfloat16 T;
  FusedSplitAdamWTPP() {}
  FusedSplitAdamWTPP(
      int N,
      float beta1,
      float beta2,
      float weight_decay,
      float eps,
      bool use_adam = false)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        use_adam(use_adam),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2),
        eqn3(this, 3) {}
  void operator()(
      T* hi,
      T* lo,
      T* grad_,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    T wd_grad[N];
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    bool adam_style_weight_decay = weight_decay > 0.0 && use_adam;
    void* grad = adam_style_weight_decay ? (void*)wd_grad : (void*)grad_;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;
    if (adam_style_weight_decay) {
      arg_array[0].primary = (void*)grad_;
      arg_array[4].primary = (void*)lo;
      arg_array[5].primary = (void*)hi;
      arg_array[7].primary = (void*)&weight_decay;
      eqn_param.output.primary = (void*)grad;
      eqn3(&eqn_param);
    }

    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)lo;
    arg_array[5].primary = (void*)hi;
    arg_array[6].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)lo;
    auto offset = (long long)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)&offset;
    eqn2(&eqn_param);
  }

  void ref(
      T* hi,
      T* lo,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      union libxsmm_bfloat16_f32 data_hp;
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      data_hp.i[0] = lo[i];
      data_hp.i[1] = hi[i];
      float data_i = data_hp.f;
      if (weight_decay > 0.0 && use_adam)
        grad_i = grad_i + data_i * weight_decay;

      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0 && !use_adam)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data_hp.f = data_i;
      lo[i] = data_hp.i[0];
      hi[i] = data_hp.i[1];
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(weight_decay);
    auto vweight_decay_lr = _mm512_set1_ps(lr * weight_decay);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      if (weight_decay > 0.0 && use_adam)
        grad_i = _mm512_add_ps(grad_i, _mm512_mul_ps(data_i, vweight_decay));
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0 && !use_adam)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay_lr));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      if (weight_decay > 0.0 && use_adam)
        grad_i = _mm512_add_ps(grad_i, _mm512_mul_ps(data_i, vweight_decay));
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0 && !use_adam)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay_lr));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedSplitAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_split_adamw_eqn%d_t%d_n%d_wd%d_adam%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1),
          (p->use_adam ? 1 : 0));
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      libxsmm_meqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for data_i (with decay)
        int use_wd = (p->weight_decay == 0.0 && !p->use_adam) ? 0 : 1;
        auto my_eqn2 = libxsmm_meqn_create();
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_UNZIP,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_DATATYPE_IMPLICIT);
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ZIP,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_DATATYPE_IMPLICIT);
        meqn_push_arg(
            my_eqn2, N, 1, ld, 4, 0, LIBXSMM_DATATYPE_U16); // data_i lo
        meqn_push_arg(
            my_eqn2, N, 1, ld, 5, 0, LIBXSMM_DATATYPE_U16); // data_i hi

        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_U16, my_eqn2);
      } else if (eqn_no == 3) {
        int use_wd = (p->weight_decay == 0.0 && p->use_adam) ? 0 : 1;
        if (!use_wd)
          return (void*)0xdeaddead;
        auto my_eqn3 = libxsmm_meqn_create();
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_ZIP,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_DATATYPE_IMPLICIT);
        meqn_push_arg(
            my_eqn3, N, 1, ld, 4, 0, LIBXSMM_DATATYPE_U16); // data_i lo
        meqn_push_arg(
            my_eqn3, N, 1, ld, 5, 0, LIBXSMM_DATATYPE_U16); // data_i hi
        meqn_push_arg(
            my_eqn3, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32); // weight_decay
        meqn_push_arg(my_eqn3, N, 1, ld, 0, 0, in_dt); // grad_i
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedSplitAdamWTPP* p;
    int eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  bool use_adam;
  Eqn eqn0, eqn1, eqn2, eqn3;
  friend class Eqn;
};

template <typename T>
class FusedAdamStepTPP {
 public:
  FusedAdamStepTPP() {}
  FusedAdamStepTPP(
      int N,
      float beta1,
      float beta2,
      float eps,
      bool use_weight_decay,
      bool use_bias_correction)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        eps(eps),
        use_weight_decay(use_weight_decay),
        use_bias_correction(use_bias_correction),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)data;
    arg_array[4].primary = (void*)&weight_decay;
    arg_array[5].primary = (void*)&exp_avg_scale;
    arg_array[6].primary = (void*)&exp_avg_sq_scale;
    eqn_param.output.primary = (void*)adam_step;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      if (use_bias_correction) {
        avg_i = avg_i * exp_avg_scale;
        avg_sq_i = avg_sq_i * exp_avg_sq_scale;
      }
      float denom = sqrtf(avg_sq_i) + eps;
      float adam_step_i = avg_i / denom;
      if (use_weight_decay) {
        float data_i = data[i];
        adam_step_i += data_i * weight_decay;
      }
      adam_step[i] = adam_step_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    // auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(weight_decay);
    auto vexp_avg_scale = _mm512_set1_ps(exp_avg_scale);
    auto vexp_avg_sq_scale = _mm512_set1_ps(exp_avg_sq_scale);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps_auto(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps_auto(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_storeu_ps_auto(&exp_avg[i], avg_i);
      _mm512_storeu_ps_auto(&exp_avg_sq[i], avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_loadu_ps_auto(&data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_storeu_ps_auto(&adam_step[i], adam_step_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_mask_storeu_ps_auto(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps_auto(&exp_avg_sq[i], mask, avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_maskz_loadu_ps_auto(mask, &data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_mask_storeu_ps_auto(&adam_step[i], mask, adam_step_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamStepTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_meqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_meqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_adam_step_eqn%d_t%d_n%d_wd%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          p->use_weight_decay);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->use_weight_decay;
      int use_bc = p->use_bias_correction;
      libxsmm_meqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_meqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for adam_step_i (with decay)
        auto my_eqn2 = libxsmm_meqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
          meqn_push_arg(my_eqn2, N, 1, ld, 3, 0, in_dt); // data_i
          // weight_decay
          meqn_push_arg(my_eqn2, 1, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32); // avg_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32); // avg_sq_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedAdamStepTPP* p;
    int eqn_no;
    libxsmm_meqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, eps;
  bool use_weight_decay, use_bias_correction;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

}; // namespace tpp

#endif // _XSMM_FUNCTORS_H_
