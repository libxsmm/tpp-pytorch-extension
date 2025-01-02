#pragma once

#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include "utils.h"
#include "vla.h"

namespace at {
// TODO: reusing unused QScheme here, fix it later
constexpr auto kPerBlockAffine = kPerTensorSymmetric;
constexpr auto kPerBlockMxFP = kPerChannelSymmetric;

inline int64_t get_elements_per_byte(at::ScalarType t) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t element_per_byte;
  switch (t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  return element_per_byte;
}

struct TORCH_API PerBlockQuantizer : public Quantizer {
  explicit PerBlockQuantizer(
      ScalarType scalar_type,
      IntArrayRef in_sizes,
      int64_t block_size,
      int64_t axis,
      bool is_vnni)
      : Quantizer(scalar_type),
        pack_size_(get_elements_per_byte(scalar_type)),
        block_size_(block_size),
        axis_(axis),
        is_vnni_(is_vnni) {
    scale_sizes_ = in_sizes.vec();
    vnni_pack_size_ = 1;
    if (is_vnni_) {
      vnni_pack_size_ = scale_sizes_.back();
      scale_sizes_.pop_back();
    } else {
      // Make sure axis is last dim
      TPP_ASSERT(
          axis == (int64_t)scale_sizes_.size() - 1,
          "Invalid axis %d\n",
          (int)axis);
      TPP_ASSERT(scale_sizes_[axis] % pack_size_ == 0, "Invalid dim size\n");
      vnni_pack_size_ = pack_size_;
      scale_sizes_[axis] /= pack_size_;
    }
    TPP_ASSERT(
        block_size % vnni_pack_size_ == 0,
        "Block size must be multiple of VNNI pack size\n");
    block_size_internal_ = block_size / vnni_pack_size_;
    TPP_ASSERT((int64_t)scale_sizes_.size() > axis, "Invalid axis\n");
    TPP_ASSERT(
        scale_sizes_[axis] % block_size_internal_ == 0,
        "Axis size must be multiple of block_size\n");
    scale_sizes_[axis] /= block_size_internal_;
    pre_ = c10::size_to_dim_(axis + 1, scale_sizes_);
    post_ = c10::size_from_dim_(axis + 1, scale_sizes_);
  }

  template <typename QCls>
  void quantize_affine(
      const at::Tensor& t_in,
      at::Tensor& t_scales,
      at::Tensor& t_zp,
      at::Tensor& t_out) {
    using Tin = typename QCls::Tin;
    using Tout = typename QCls::Tout;
    using Ts = typename QCls::Ts;
    using Tzp = typename QCls::Tzp;

    auto bs = this->block_size_internal_;
    auto vnni = vnni_pack_size_;
    auto packed_vnni = vnni_pack_size_ / pack_size_;
    auto in = GetVLAPtr<Tin>(t_in, {bs, post_, vnni});
    auto scales = GetVLAPtr<Ts>(t_scales, {post_});
    auto zps = GetVLAPtr<Tzp>(t_zp, {post_});
    auto out = GetVLAPtr<Tout>(t_out, {bs, post_, packed_vnni});

#pragma omp parallel for collapse(2)
    for (int i = 0; i < pre_; i++) {
      for (int j = 0; j < post_; j++) {
        float max = (float)in[i][0][j][0];
        float min = (float)in[i][0][j][0];
        for (int k = 0; k < bs; k++) {
          for (int v = 0; v < vnni; v++) {
            float val = (float)in[i][k][j][v];
            max = std::max(max, val);
            min = std::min(max, val);
          }
        }
        Ts scale;
        Tzp zp;
        auto qt = QCls(min, max);
        scales[i][j] = qt.get_scale();
        zps[i][j] = qt.zero_point();

        if (pack_size_ == 2) {
          for (int k = 0; k < bs; k++) {
            for (int v1 = 0; v1 < packed_vnni; v1++) {
              unsigned int packed_qval = 0;
              for (int v2 = 0; v2 < pack_size_; v2++) {
                int v = v1 * pack_size_ + v2;
                unsigned int qval = qt.quantize(in[i][k][j][v]);
                unsigned qval1 = ((qval & 0xF) << (v2 * 4));
                packed_qval |= qval1;
              }
              out[i][k][j][v1] = packed_qval;
            }
          }
        } else {
          for (int k = 0; k < bs; k++) {
            for (int v = 0; v < vnni; v++) {
              auto qval = qt.quantize(in[i][k][j][v]);
              out[i][k][j][v] = qval;
            }
          }
        }
      }
    }
  }

  template <typename QCls>
  void quantize_symetric(
      const at::Tensor& t_in,
      at::Tensor& t_scales,
      at::Tensor& t_out) {
    using Tin = typename QCls::Tin;
    using Tout = typename QCls::Tout;
    using Ts = typename QCls::Ts;

    auto bs = this->block_size_internal_;
    auto qtpp = QCls::GetQuantizeTPP(bs);
    auto vnni = vnni_pack_size_;
    auto packed_vnni = vnni_pack_size_ / pack_size_;
    auto in = GetVLAPtr<Tin>(t_in, {bs, post_, vnni});
    auto scales = GetVLAPtr<Ts>(t_scales, {post_});
    auto out = GetVLAPtr<Tout>(t_out, {bs, post_, packed_vnni});
    // std::cout << "tin: " << t_in.sizes() << std::endl;
    // std::cout << "bs: " << bs << " vnni: " << vnni << " pre_: " << pre_
    //           << " post_: " << post_ << " packed_vnni: " << packed_vnni
    //           << std::endl;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < pre_; i++) {
      for (int j = 0; j < post_; j++) {
        float max = fabsf((float)in[i][0][j][0]);
        for (int k = 0; k < bs; k++) {
          for (int v = 0; v < vnni; v++) {
            float val = fabsf((float)in[i][k][j][v]);
            max = std::max(max, val);
          }
        }
        auto qt = QCls(max);
        scales[i][j] = qt.get_scale();

        if (pack_size_ == 2) {
          for (int k = 0; k < bs; k++) {
            for (int v1 = 0; v1 < packed_vnni; v1++) {
              unsigned int packed_qval = 0;
              for (int v2 = 0; v2 < pack_size_; v2++) {
                int v = v1 * pack_size_ + v2;
                unsigned int qval = qt.quantize(in[i][k][j][v]);
                unsigned qval1 = ((qval & 0xF) << (v2 * 4));
                packed_qval |= qval1;
              }
              out[i][k][j][v1] = packed_qval;
            }
          }
        } else {
          if (vnni == 1 && post_ == 1) {
            qt.quantize(in[i][0][0], out[i][0][0], qtpp);
          } else {
            for (int k = 0; k < bs; k++) {
              for (int v = 0; v < vnni; v++) {
                auto qval = qt.quantize(in[i][k][j][v]);
                out[i][k][j][v] = qval;
              }
            }
          }
        }
      }
    }
  }

  template <typename QCls>
  void dequantize_symetric(
      const at::Tensor& t_in,
      at::Tensor& t_scales,
      at::Tensor& t_out) {
    using Tin = typename QCls::Tin;
    using Tout = typename QCls::Tout;
    using Ts = typename QCls::Ts;

    auto bs = this->block_size_internal_;
    auto vnni = vnni_pack_size_;
    auto packed_vnni = vnni_pack_size_ / pack_size_;
    auto in = GetVLAPtr<Tout>(t_in, {bs, post_, packed_vnni});
    auto scales = GetVLAPtr<Ts>(t_scales, {post_});
    auto out = GetVLAPtr<Tin>(t_out, {bs, post_, vnni});
    // std::cout << "tin: " << t_in.sizes() << std::endl;
    // std::cout << "bs: " << bs << " vnni: " << vnni << " pre_: " << pre_
    //           << " post_: " << post_ << " packed_vnni: " << packed_vnni
    //           << std::endl;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < pre_; i++) {
      for (int j = 0; j < post_; j++) {
        auto qt = QCls();
        qt.set_scale(scales[i][j]);

        if (pack_size_ == 2) {
          for (int k = 0; k < bs; k++) {
            for (int v1 = 0; v1 < packed_vnni; v1++) {
              unsigned int packed_qval = in[i][k][j][v1];
              for (int v2 = 0; v2 < pack_size_; v2++) {
                int v = v1 * pack_size_ + v2;
                unsigned int qval = (packed_qval >> (v2 * 4)) & 0xF;
                out[i][k][j][v] = qt.dequantize(qval);
              }
            }
          }
        } else {
          for (int k = 0; k < bs; k++) {
            for (int v = 0; v < vnni; v++) {
              auto val = qt.dequantize(in[i][k][j][v]);
              out[i][k][j][v] = val;
            }
          }
        }
      }
    }
  }

  IntArrayRef scale_sizes() const {
    return IntArrayRef(scale_sizes_.data(), scale_sizes_.size());
  }

  int64_t block_size_internal() const {
    return block_size_internal_;
  }

  int64_t vnni_pack_size() const {
    return vnni_pack_size_;
  }

  int64_t pack_size() const {
    return pack_size_;
  }

  int64_t block_size() const {
    return block_size_;
  }

  int64_t axis() const {
    return axis_;
  }

  bool is_vnni() const {
    return is_vnni_;
  }

 protected:
  std::vector<int64_t> scale_sizes_;
  int64_t block_size_internal_;
  int64_t vnni_pack_size_;
  int64_t pack_size_;
  int64_t pre_;
  int64_t post_;
  const int64_t block_size_;
  const int64_t axis_;
  const bool is_vnni_;
};

struct TORCH_API PerBlockMxFPQuantizer : public at::PerBlockQuantizer {
  explicit PerBlockMxFPQuantizer(
      ScalarType scalar_type,
      Tensor t_in,
      int64_t block_size,
      int64_t axis,
      bool is_vnni)
      : PerBlockQuantizer(scalar_type, t_in.sizes(), block_size, axis, is_vnni),
        block_size_(block_size),
        axis_(axis),
        is_vnni_(is_vnni) {
    scales_ = t_in.new_empty(this->scale_sizes(), kByte);
  }

  QScheme qscheme() const override {
    return kPerBlockMxFP;
  }

  Tensor scales() const {
    return scales_;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerBlockMxFP) {
      return false;
    }
    auto* other_per_block_mxfp =
        static_cast<PerBlockMxFPQuantizer*>(other.get());
    return scalar_type() == other_per_block_mxfp->scalar_type() &&
        scales().equal(other_per_block_mxfp->scales()) &&
        block_size() == other_per_block_mxfp->block_size() &&
        axis() == other_per_block_mxfp->axis() &&
        is_vnni() == other_per_block_mxfp->is_vnni();
  }

 protected:
  Tensor scales_;
  const int64_t block_size_;
  const int64_t axis_;
  const bool is_vnni_;
};

struct TORCH_API PerBlockAffineQuantizer : public at::PerBlockQuantizer {
  explicit PerBlockAffineQuantizer(
      ScalarType scalar_type,
      Tensor t_in,
      int64_t block_size,
      int64_t axis,
      bool is_vnni,
      bool use_zp)
      : PerBlockQuantizer(scalar_type, t_in.sizes(), block_size, axis, is_vnni),
        has_zp_(use_zp) {
    scales_ = t_in.new_empty(this->scale_sizes(), kFloat);
    if (has_zp_) {
      zero_points_ = t_in.new_empty(this->scale_sizes(), kFloat);
    }
  }

  QScheme qscheme() const override {
    return kPerBlockAffine;
  }

  Tensor scales() const {
    return scales_;
  }

  void set_scales(Tensor& scales) {
    scales_ = scales;
  }

  Tensor zero_points() const {
    return zero_points_;
  }
  bool has_zp() const {
    return has_zp_;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerBlockAffine) {
      return false;
    }
    auto* other_per_block_affine =
        static_cast<PerBlockAffineQuantizer*>(other.get());
    return scalar_type() == other_per_block_affine->scalar_type() &&
        scales().equal(other_per_block_affine->scales()) &&
        has_zp() == other_per_block_affine->has_zp() &&
        (has_zp() == false ||
         zero_points().equal(other_per_block_affine->zero_points())) &&
        block_size() == other_per_block_affine->block_size() &&
        axis() == other_per_block_affine->axis() &&
        is_vnni() == other_per_block_affine->is_vnni();
  }

 protected:
  Tensor scales_;
  Tensor zero_points_;
  bool has_zp_;
};

} // namespace at

int64_t q_per_block_block_size(const at::Tensor& self);

at::Tensor q_per_block_scales(const at::Tensor& self);

at::Tensor quantize_mxfp4(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni);

at::Tensor quantize_int8sym(
    const at::Tensor& self,
    int64_t block_size,
    int64_t axis,
    bool is_vnni);

at::Tensor remap_and_quantize_mxfp4(at::Tensor t);
at::Tensor remap_and_quantize_qint8(at::Tensor t);
