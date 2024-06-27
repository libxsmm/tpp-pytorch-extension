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
#include "utils.h"
#include "vla.h"

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
  inline Tin dequantize(int qval) {
    Tin val = qval * scale;
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
  } else if (device.is_privateuseone()) {
    allocator = GetAllocator(kPrivateUse1);
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
  return quantize_mxfp_(self, block_size, axis, is_vnni, dtype);
}

at::Tensor quantize_int8sym(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni) {
  at::ScalarType dtype = at::kQInt8;
  auto quantizer = at::make_per_block_affine_quantizer(
      self, block_size, axis, is_vnni, /*has_zp=*/false, dtype);
  return quantizer->quantize(self);
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

size_t q_get_ptr(const at::Tensor& self) {
  // return (size_t)self.data_ptr<uint8_t>();
  return (size_t)self.data_ptr();
}

REGISTER_SUBMODULE(_qtype, m) {
  m.def("quantize_mxfp", &quantize_mxfp);
  m.def("quantize_mxfp4", &quantize_mxfp4);
  m.def("quantize_int8sym", &quantize_int8sym);
  m.def("q_per_block_scales", &q_per_block_scales);
  m.def("q_per_block_block_size", &q_per_block_block_size);
  m.def("q_per_block_axis", &q_per_block_axis);
  m.def("q_get_ptr", &q_get_ptr);
}
