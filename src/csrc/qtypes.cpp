#include <ATen/ATen.h>
#include <ATen/ArrayRef.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/accumulate.h>

#include <cmath>
#include <utility>

#include "init.h"
#include "mxfp_quant.h"
#include "qtypes.h"
#include "timing.h"
#include "utils.h"
#include "vla.h"
#include "xsmm_functors.h"

static const int QINT8_EMU = env2int("QINT8_EMU", 0);
static const int QINT8_BLOCK_SIZE = env2int("QINT8_BLOCK_SIZE", 256);
static const int USE_MXFP4_INT8 = env2int("USE_MXFP4_INT8", 0);

REGISTER_LOCAL_SCOPE(quant, "quant");

using namespace tpp;
template <typename TIN>
struct MxFP4Quant {
  using Tin = TIN;
  using Tout = uint8_t;
  using Ts = uint8_t;

  static constexpr float lkp_tbl[] = {0.0,
                                      0.5,
                                      1.0,
                                      1.5,
                                      2.0,
                                      3.0,
                                      4.0,
                                      6.0,
                                      -0.0,
                                      -0.5,
                                      -1.0,
                                      -1.5,
                                      -2.0,
                                      -3.0,
                                      -4.0,
                                      -6.0};
  static constexpr bool flush_fp32_subnorms = true;
  static constexpr float elem_max_norm = 6.0;
  static constexpr int scale_bits = 8;
  static constexpr int elem_ebits = 2;
  static constexpr int elem_mbits = 3;
  static constexpr RoundingMode rounding_mode = RoundingMode::rd_even;
  bool flush_tile = false;
  float scale_fp = 0.0f;
  Ts scale;

  MxFP4Quant() : scale(0) {}
  MxFP4Quant(float max) {
    int shared_exp = (int)get_biased_exponent(max);
    flush_tile = (shared_exp == 0 && flush_fp32_subnorms);
    // Compute the shared scale
    scale_fp = mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm);

    scale = get_biased_exponent(scale_fp);
  }

  Ts get_scale() {
    return scale;
  }

  void set_scale(Ts sc) {
    scale = sc;
    scale_fp = get_fp_scale(scale);
  }

  inline unsigned int quantize(Tin val) {
    unsigned int qval = 0;
    float scaled_in = (flush_tile) ? 0 : (float)val / scale_fp;

    float scaled_out = quantize_elemwise(
        scaled_in,
        elem_mbits,
        elem_ebits,
        elem_max_norm,
        rounding_mode,
        true,
        true);
    if (scaled_out == 0.0) {
      qval = 0;
    } else if (scaled_out == 0.5) {
      qval = 1;
    } else if (scaled_out == 1.0) {
      qval = 2;
    } else if (scaled_out == 1.5) {
      qval = 3;
    } else if (scaled_out == 2.0) {
      qval = 4;
    } else if (scaled_out == 3.0) {
      qval = 5;
    } else if (scaled_out == 4.0) {
      qval = 6;
    } else if (scaled_out == 6.0) {
      qval = 7;
    } else if (scaled_out == -0.0) {
      qval = 8;
    } else if (scaled_out == -0.5) {
      qval = 9;
    } else if (scaled_out == -1.0) {
      qval = 10;
    } else if (scaled_out == -1.5) {
      qval = 11;
    } else if (scaled_out == -2.0) {
      qval = 12;
    } else if (scaled_out == -3.0) {
      qval = 13;
    } else if (scaled_out == -4.0) {
      qval = 14;
    } else if (scaled_out == -6.0) {
      qval = 15;
    }
    // printf("val: %g, scale_fp: %g, scaled_in: %g scaled_out: %g  qval: %u\n",
    // (float)val, scale_fp, scaled_in, scaled_out, qval);
    return qval;
  }

  static auto GetQuantizeTPP(int N) {
    return nullptr;
  }

  template <typename T_QTPP>
  inline void quantize(Tin* inp, Tout* outp, const T_QTPP& qtpp) {
    throw std::runtime_error("Not Implemented");
    // qtpp(inp, outp, iscale);
  }

  inline Tin dequantize(unsigned int qval) {
    float val = lkp_tbl[qval] * scale_fp;
    return (Tin)val;
  }
};

template <typename TIN>
struct Int8SymQuant {
  using Tin = TIN;
  using Tout = int8_t;
  using Ts = float;

  Ts scale;
  Ts iscale;

  Int8SymQuant() : scale(0), iscale(0) {}
  Int8SymQuant(float max) {
    scale = max / 127.0;
    iscale = (scale != 0) ? (1.0f / scale) : 0;
  }

  Ts get_scale() {
    return scale;
  }

  void set_scale(Ts sc) {
    scale = sc;
    iscale = (scale != 0) ? (1.0f / scale) : 0;
  }

  inline int quantize(Tin val) {
    int qval = 0;
    qval = nearbyintf((float)val * iscale);
    qval = std::clamp(qval, -127, 127);
    return qval;
  }

  static auto GetQuantizeTPP(int N) {
    return SCOPEIT((QuantTPP<Tin, Tout, Ts>(N)), EW_RSQRT);
  }

  template <typename T_QTPP>
  inline void quantize(Tin* inp, Tout* outp, T_QTPP& qtpp) {
    qtpp(inp, outp, iscale);
  }

  inline Tin dequantize(int qval) {
    Tin val = qval * scale;
    return val;
  }
};

template <typename TIN>
struct Int4SymQuant {
  using Tin = TIN;
  using Tout = uint8_t;
  using Ts = float;
  static constexpr int lkp_tbl[] =
      {0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};

  Ts scale;
  Ts iscale;

  Int4SymQuant() : scale(0), iscale(0) {}
  Int4SymQuant(float max) {
    scale = max / 7.0;
    iscale = (scale != 0) ? (1.0f / scale) : 0;
  }

  Ts get_scale() {
    return scale;
  }

  void set_scale(Ts sc) {
    scale = sc;
    iscale = (scale != 0) ? (1.0f / scale) : 0;
  }

  inline int quantize(Tin val) {
    int qval = 0;
    qval = nearbyintf((float)val * iscale);
    qval = std::clamp(qval, -8, 7);
    return qval;
  }

  static auto GetQuantizeTPP(int N) {
    return nullptr;
  }

  template <typename T_QTPP>
  inline void quantize(Tin* inp, Tout* outp, T_QTPP& qtpp) {
    // qtpp(inp, outp, iscale);
  }

  inline Tin dequantize(int qval) {
    Tin val = lkp_tbl[qval] * scale;
    return val;
  }
};

namespace at {

Tensor& dequantize_tensor_per_block_mxfp(
    const Tensor& qtensor,
    Tensor& rtensor,
    Tensor scales,
    const int64_t block_size,
    const int64_t axis) {
  TPP_ASSERT(false, "dequantize_tensor_per_block_mxfp: Not implemented\n");
  return rtensor;
}

QuantizerPtr make_per_block_mxfp_quantizer(
    const Tensor& in,
    int64_t block_size,
    int64_t axis,
    bool is_vnni,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerBlockMxFPQuantizer>(
      scalar_type, in, block_size, axis, is_vnni);
}

QuantizerPtr make_per_block_affine_quantizer(
    const Tensor& in,
    int64_t block_size,
    int64_t axis,
    bool is_vnni,
    bool has_zp,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerBlockAffineQuantizer>(
      scalar_type, in, block_size, axis, is_vnni, has_zp);
}

static int64_t get_sub_byte_tensor_size(
    IntArrayRef sizes,
    size_t dtype_itemsize,
    at::ScalarType t) {
  int64_t element_per_byte = get_elements_per_byte(t);
  ;
  // zero dim tensor
  if (sizes.empty()) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // Consider most inner dim as cols
  int64_t cols = sizes.at(sizes.size() - 1);
  int64_t bytes_per_row = cols * dtype_itemsize;
  // align qtensor most inner dim, compute ceil (bytes_per_row /
  // element_per_byte)
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) *
      at::ceil_div(bytes_per_row, element_per_byte);
}

inline Tensor new_qtensor(Tensor qval, QuantizerPtr quantizer) {
  auto options = qval.options().dtype(quantizer->scalar_type());
  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  // auto device = options.device();

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();

  auto tensor = detail::make_tensor<QTensorImpl>(
      c10::Storage(qval.storage()),
      at::DispatchKeySet(tensorDispatchKey),
      scalarTypeToTypeMeta(quantizer->scalar_type()),
      quantizer);
  get_qtensorimpl(tensor)->set_storage_offset(qval.storage_offset());
  get_qtensorimpl(tensor)->set_sizes_contiguous(qval.sizes());
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

inline Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  auto device = options.device();
  at::Allocator* allocator = nullptr;
  // TODO: why isn't this just using GetAllocator
  if (device.is_cuda()) {
    allocator = at::detail::getCUDAHooks().getCUDADeviceAllocator();
  } else if (device.is_cpu()) {
    allocator = at::getCPUAllocator();
  } else if (device.is_meta()) {
    allocator = GetAllocator(kMeta);
  } else {
    TORCH_INTERNAL_ASSERT(0, "unrecognized device for new_qtensor: ", device);
  }

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  auto dtype = options.dtype();
  auto scalar_type = typeMetaToScalarType(dtype);
  TORCH_CHECK(
      isQIntType(scalar_type),
      "ScalarType ",
      scalar_type,
      " is not supported in new_qtensor.");
  int64_t size_bytes =
      get_sub_byte_tensor_size(sizes, dtype.itemsize(), scalar_type);

  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor PerBlockMxFPQuantizer::quantize(const Tensor& rtensor) {
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());
  auto rtensor_contig =
      rtensor.expect_contiguous(rtensor.suggest_memory_format());
  if (scalar_type_ == kQUInt4x2) {
    if (rtensor.dtype() == kFloat) {
      this->quantize_symetric<MxFP4Quant<float>>(rtensor, scales_, qtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->quantize_symetric<MxFP4Quant<bfloat16>>(rtensor, scales_, qtensor);
    } else {
      TPP_ASSERT(false, "Unsupported input type for MXFP\n");
    }
  } else {
    TPP_ASSERT(false, "Unsupported MXFP datatype\n");
  }
  return qtensor;
}

Tensor PerBlockMxFPQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  return dequantize_out(rtensor, qtensor);
}

Tensor& PerBlockMxFPQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
          rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  const auto qtensor_contig = qtensor.contiguous();
  if (scalar_type_ == kQUInt4x2) {
    if (rtensor.dtype() == kFloat) {
      this->dequantize_symetric<MxFP4Quant<float>>(
          qtensor_contig, scales_, rtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->dequantize_symetric<MxFP4Quant<bfloat16>>(
          qtensor_contig, scales_, rtensor);
    } else {
      TPP_ASSERT(false, "Unsupported output type for MXFP\n");
    }
  } else {
    TPP_ASSERT(false, "Unsupported MXFP datatype\n");
  }
  return rtensor;
}

Tensor PerBlockAffineQuantizer::quantize(const Tensor& rtensor) {
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());
  auto rtensor_contig = rtensor.contiguous();
  if (scalar_type_ == kQInt8 && has_zp_ == false) {
    if (rtensor.dtype() == kFloat) {
      this->quantize_symetric<Int8SymQuant<float>>(
          rtensor_contig, scales_, qtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->quantize_symetric<Int8SymQuant<bfloat16>>(
          rtensor_contig, scales_, qtensor);
    } else {
      TPP_ASSERT(false, "Unsupported input type for Affine Quant\n");
    }
  } else if (scalar_type_ == kQUInt4x2 && has_zp_ == false) {
    if (rtensor.dtype() == kFloat) {
      this->quantize_symetric<Int4SymQuant<float>>(
          rtensor_contig, scales_, qtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->quantize_symetric<Int4SymQuant<bfloat16>>(
          rtensor_contig, scales_, qtensor);
    } else {
      TPP_ASSERT(false, "Unsupported input type for Affine Quant\n");
    }
  } else {
    TPP_ASSERT(false, "Unsupported Affine datatype\n");
  }
  return qtensor;
}

Tensor PerBlockAffineQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  dequantize_out(rtensor, qtensor);
  return rtensor;
}

Tensor& PerBlockAffineQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
          rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  const auto qtensor_contig = qtensor.contiguous();
  if (scalar_type_ == kQInt8) {
    if (rtensor.dtype() == kFloat) {
      this->dequantize_symetric<Int8SymQuant<float>>(
          qtensor_contig, scales_, rtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->dequantize_symetric<Int8SymQuant<bfloat16>>(
          qtensor_contig, scales_, rtensor);
    } else {
      TPP_ASSERT(false, "Unsupported output type for Per Block QInt8\n");
    }
  } else if (scalar_type_ == kQUInt4x2) {
    if (rtensor.dtype() == kFloat) {
      this->dequantize_symetric<Int4SymQuant<float>>(
          qtensor_contig, scales_, rtensor);
    } else if (rtensor.dtype() == kBFloat16) {
      this->dequantize_symetric<Int4SymQuant<bfloat16>>(
          qtensor_contig, scales_, rtensor);
    } else {
      TPP_ASSERT(false, "Unsupported output type for Per Block QInt8\n");
    }
  } else {
    TPP_ASSERT(false, "Unsupported QInt datatype\n");
  }
  return rtensor;
}

} // namespace at

at::Tensor quantize_mxfp_(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni,
    at::ScalarType dtype) {
  auto quantizer =
      at::make_per_block_mxfp_quantizer(self, block_size, axis, is_vnni, dtype);
  return quantizer->quantize(self);
}

at::Tensor quantize_mxfp(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni,
    py::object dtype_) {
  at::ScalarType dtype = torch::python::detail::py_object_to_dtype(dtype_);
  return quantize_mxfp_(self, block_size, axis, is_vnni, dtype);
}

at::Tensor quantize_mxfp4(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQUInt4x2;
  if (axis < 0)
    axis += self.dim();
  return quantize_mxfp_(self, block_size, axis, is_vnni, dtype);
}

at::Tensor quantize_int8sym(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQInt8;
  if (axis < 0)
    axis += self.dim();
  auto quantizer = at::make_per_block_affine_quantizer(
      self, block_size, axis, is_vnni, /*has_zp=*/false, dtype);
  return quantizer->quantize(self);
}

at::Tensor quantize_int4sym(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQUInt4x2;
  if (axis < 0)
    axis += self.dim();
  auto quantizer = at::make_per_block_affine_quantizer(
      self, block_size, axis, is_vnni, /*has_zp=*/false, dtype);
  return quantizer->quantize(self);
}

at::Tensor create_qtensor_int8sym(
    at::Tensor& val,
    at::Tensor& scales,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQInt8;
  TORCH_CHECK(val.dtype() == at::kChar && scales.dtype() == at::kFloat);
  auto quantizer = at::make_per_block_affine_quantizer(
      val, block_size, axis, is_vnni, /*has_zp=*/false, dtype);
  static_cast<at::PerBlockAffineQuantizer*>(quantizer.get())
      ->set_scales(scales);
  at::Tensor qtensor = at::new_qtensor(val, quantizer);
  return qtensor;
}

at::Tensor create_qtensor_int4sym(
    at::Tensor& val,
    at::Tensor& scales,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQUInt4x2;
  TORCH_CHECK(val.dtype() == at::kByte && scales.dtype() == at::kFloat);
  auto quantizer = at::make_per_block_affine_quantizer(
      val, block_size, axis, is_vnni, /*has_zp=*/false, dtype);
  static_cast<at::PerBlockAffineQuantizer*>(quantizer.get())
      ->set_scales(scales);
  at::Tensor qtensor = at::new_qtensor(val, quantizer);
  return qtensor;
}

at::Tensor q_per_block_scales(const at::Tensor& self) {
  auto quantizer = at::get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == at::kPerBlockMxFP ||
      quantizer->qscheme() == at::kPerBlockAffine);
  if (quantizer->qscheme() == at::kPerBlockMxFP) {
    return static_cast<at::PerBlockMxFPQuantizer*>(quantizer.get())->scales();
  } else {
    return static_cast<at::PerBlockAffineQuantizer*>(quantizer.get())->scales();
  }
}

int64_t q_per_block_block_size(const at::Tensor& self) {
  auto quantizer = at::get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == at::kPerBlockMxFP ||
      quantizer->qscheme() == at::kPerBlockAffine);
  return static_cast<at::PerBlockQuantizer*>(quantizer.get())->block_size();
}

int64_t q_per_block_axis(const at::Tensor& self) {
  auto quantizer = at::get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == at::kPerBlockMxFP ||
      quantizer->qscheme() == at::kPerBlockAffine);
  return static_cast<at::PerBlockQuantizer*>(quantizer.get())->axis();
}

inline at::Tensor fix_vnni(at::Tensor t) {
  auto dtype = t.scalar_type();
  auto dims = t.dim();
  TPP_ASSERT(dims >= 4, "Invalid shape, dims = %ld\n", dims);
  auto sizes = t.sizes();
  auto K1 = sizes[0];
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = dims == 5 ? sizes[4] : 1L;
  if (dtype == at::kBFloat16) {
    if (C3 == 2) {
      return t;
    } else if (C3 == 4) {
      return t.view({K1, C1, C2, K2, 2, 2})
          .permute({0, 1, 2, 4, 3, 5})
          .contiguous()
          .view({K1, C1, C2 * 2, K2, 2});
    } else if (C3 == 1) {
      TPP_ASSERT(
          C2 % 2 == 0, "Shape incompatible for VNNI2 layout, C2 = %ld\n", C2);
      return t.view({K1, C1, C2 / 2, 2, K2})
          .permute({0, 1, 2, 4, 3})
          .contiguous();
    } else {
      TPP_ASSERT(false, "Shape incompatible for VNNI2 layout\n");
    }
  } else if (dtype == at::kFloat) {
    if (C3 == 1 and dims == 4) {
      return t;
    } else if (C3 == 1) {
      return t.view({K1, C1, C2, K2});
    } else {
      return t.permute({0, 1, 2, 4, 3})
          .contiguous()
          .view({K1, C1, C2 * C3, K2});
    }
  } else {
    TPP_ASSERT(false, "Unknown dtype for FIX_VNNI layout\n");
  }
  return t;
}

inline at::Tensor remap_and_quantize_mxfp4(at::Tensor t) {
  RECORD_SCOPE(quant, {t});
  auto desired_K2 = 32L;
  auto VBS = 2L;
  if (USE_MXFP4_INT8) {
    VBS = 8L;
    desired_K2 = 8L;
  }
  auto dim = t.dim();
  if (dim == 5) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C3 = sizes[4];
    if (K2 < desired_K2 && desired_K2 % K2 == 0) {
      int RBS = desired_K2 / K2;
      TPP_ASSERT(K1 % RBS == 0, "Shape not compatible for MXFP4\n");
      t = t.view({K1 / RBS, RBS, C1, C2, K2, C3})
              .permute({0, 2, 3, 1, 4, 5})
              .contiguous()
              .view({K1 / RBS, C1, C2, RBS * K2, C3});
    } else if (K2 > desired_K2 && K2 % desired_K2 == 0) {
      int RBS = K2 / desired_K2;
      t = t.view({K1, C1, C2, RBS, desired_K2, C3})
              .permute({0, 3, 1, 2, 4, 5})
              .contiguous()
              .view({K1 * RBS, C1, C2, desired_K2, C3});
    }
  } else if (dim == 4) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    TPP_ASSERT(C2 % VBS == 0, "Shape not compatible for VNNI2\n");
    if (K2 < desired_K2 && desired_K2 % K2 == 0) {
      int RBS = desired_K2 / K2;
      TPP_ASSERT(K1 % RBS == 0, "Shape not compatible for MXFP4\n");
      t = t.view({K1 / RBS, RBS, C1, C2 / VBS, VBS, K2})
              .permute({0, 2, 3, 1, 5, 4})
              .contiguous()
              .view({K1 / RBS, C1, C2 / VBS, RBS * K2, VBS});
    } else if (K2 > desired_K2 && K2 % desired_K2 == 0) {
      int RBS = K2 / desired_K2;
      t = t.view({K1, C1, C2 / VBS, VBS, RBS, desired_K2})
              .permute({0, 4, 1, 2, 5, 3})
              .contiguous()
              .view({K1 * RBS, C1, C2 / VBS, desired_K2, VBS});
    } else {
      t = t.view({K1, C1, C2 / VBS, VBS, K2})
              .permute({0, 1, 2, 4, 3})
              .contiguous()
              .view({K1, C1, C2 / VBS, K2, VBS});
    }
  }
  if (VBS == 8) {
    auto sizes = t.sizes();
    if (sizes[4] < 8) {
      auto sizes = t.sizes();
      auto K1 = sizes[0];
      auto C1 = sizes[1];
      auto C2 = sizes[2];
      auto K2 = sizes[3];
      auto C3 = sizes[4];
      auto RBS = 8 / C3;
      TPP_ASSERT(C2 % RBS == 0, "Shape not compatible for VNNI8\n");
      t = t.view({K1, C1, C2 / RBS, RBS, K2, C3})
              .permute({0, 1, 2, 4, 3, 5})
              .contiguous()
              .view({K1, C1, C2 / RBS, K2, RBS * C3});
    }
    sizes = t.sizes();
    auto new_sizes = sizes.vec();
    new_sizes.push_back(4);
    new_sizes[4] = 2;
    t = t.view(new_sizes).permute({0, 1, 2, 3, 5, 4}).contiguous().view(sizes);
  }
  auto ret = quantize_mxfp4(t, 32, 2, true);
  return ret;
}

inline at::Tensor remap_and_quantize_qint8(at::Tensor t) {
  RECORD_SCOPE(quant, {t});
  auto dim = t.dim();
  long block_size = 0;
  if (dim == 5) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C3 = sizes[4];
    auto C = C1 * C2 * C3;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE / C3;
      C1 = C / QINT8_BLOCK_SIZE;
    } else {
      for (int QBS = 4096; QBS >= 64; QBS /= 2) {
        if (QINT8_BLOCK_SIZE > QBS && C % QBS == 0) {
          C2 = QBS / C3;
          C1 = C / QBS;
          break;
        }
      }
    }
    if (C3 != 4) {
      int RBS = 4 / C3;
      TPP_ASSERT(C2 % RBS == 0, "Shape not compatible for VNNI4\n");
      t = t.view({K1, C1, C2 / RBS, RBS, K2, C3})
              .permute({0, 1, 2, 4, 3, 5})
              .contiguous()
              .view({K1, C1, C2 / RBS, K2, RBS * C3});
    }
    block_size = C2 * C3;
  } else if (dim == 4) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C = C1 * C2;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE;
      C1 = C / QINT8_BLOCK_SIZE;
    } else {
      for (int QBS = 4096; QBS >= 64; QBS /= 2) {
        if (QINT8_BLOCK_SIZE > QBS && C % QBS == 0) {
          C2 = QBS;
          C1 = C / QBS;
          break;
        }
      }
    }
    TPP_ASSERT(C2 % 4 == 0, "Shape not compatible for VNNI4\n");
    t = t.view({K1, C1, C2 / 4, 4, K2})
            .permute({0, 1, 2, 4, 3})
            .contiguous()
            .view({K1, C1, C2 / 4, K2, 4});
    block_size = C2;
  }
  auto ret = quantize_int8sym(t, block_size, 2, true);
  std::cout << "remap_and_quantize_qint8: " << ret.sizes()
            << " dt: " << ret.dtype() << std::endl;
  if (QINT8_EMU == 1) {
    ret = fix_vnni(ret.dequantize().to(t.dtype()));
  }
  return ret;
}

inline at::Tensor remap_and_quantize_qint4(at::Tensor t) {
  RECORD_SCOPE(quant, {t});
  auto dim = t.dim();
  long block_size = 0;
  if (dim == 5) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C3 = sizes[4];
    auto C = C1 * C2 * C3;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE / C3;
      C1 = C / QINT8_BLOCK_SIZE;
    } else {
      for (int QBS = 4096; QBS >= 64; QBS /= 2) {
        if (QINT8_BLOCK_SIZE > QBS && C % QBS == 0) {
          C2 = QBS / C3;
          C1 = C / QBS;
          break;
        }
      }
    }
    if (C3 != 4) {
      int RBS = 4 / C3;
      TPP_ASSERT(C2 % RBS == 0, "Shape not compatible for VNNI4\n");
      t = t.view({K1, C1, C2 / RBS, RBS, K2, C3})
              .permute({0, 1, 2, 4, 3, 5})
              .contiguous()
              .view({K1, C1, C2 / RBS, K2, RBS * C3});
    }
    block_size = C2 * C3;
  } else if (dim == 4) {
    auto sizes = t.sizes();
    auto K1 = sizes[0];
    auto C1 = sizes[1];
    auto C2 = sizes[2];
    auto K2 = sizes[3];
    auto C = C1 * C2;
    if (C % QINT8_BLOCK_SIZE == 0) {
      C2 = QINT8_BLOCK_SIZE;
      C1 = C / QINT8_BLOCK_SIZE;
    } else {
      for (int QBS = 4096; QBS >= 64; QBS /= 2) {
        if (QINT8_BLOCK_SIZE > QBS && C % QBS == 0) {
          C2 = QBS;
          C1 = C / QBS;
          break;
        }
      }
    }
    TPP_ASSERT(C2 % 4 == 0, "Shape not compatible for VNNI4\n");
    t = t.view({K1, C1, C2 / 4, 4, K2})
            .permute({0, 1, 2, 4, 3})
            .contiguous()
            .view({K1, C1, C2 / 4, K2, 4});
    block_size = C2;
  }
  auto ret = quantize_int4sym(t, block_size, 2, true);
  std::cout << "remap_and_quantize_qint8: " << ret.sizes()
            << " dt: " << ret.dtype() << std::endl;
  if (QINT8_EMU == 1) {
    ret = fix_vnni(ret.dequantize().to(t.dtype()));
  }
  return ret;
}

REGISTER_SUBMODULE(_qtype, m) {
  m.def("quantize_mxfp", &quantize_mxfp);
  m.def("quantize_mxfp4", &quantize_mxfp4);
  m.def("quantize_int8sym", &quantize_int8sym);
  m.def("quantize_int4sym", &quantize_int4sym);
  m.def("create_qtensor_int8sym", &create_qtensor_int8sym);
  m.def("create_qtensor_int4sym", &create_qtensor_int4sym);
  m.def("q_per_block_scales", &q_per_block_scales);
  m.def("q_per_block_block_size", &q_per_block_block_size);
  m.def("q_per_block_axis", &q_per_block_axis);
  m.def("remap_and_quantize_mxfp4", &remap_and_quantize_mxfp4);
  m.def("remap_and_quantize_qint8", &remap_and_quantize_qint8);
  m.def("remap_and_quantize_qint4", &remap_and_quantize_qint4);
  m.def("fix_vnni", &fix_vnni);
}
