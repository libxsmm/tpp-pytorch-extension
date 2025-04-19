/******************************************************************************
 * Copyright (c) 2025 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include <torch/csrc/distributed/c10d/comm.hpp>
// #include "fused_gemm.h"
#include "qtypes.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "shm_coll.h"
#include "tensor_helper.h"

static int KV_CACHE_INC_SIZE = env2int("KV_CACHE_INC_SIZE", 128);

extern int key_layout_id;
extern int value_layout_id;

struct KVMeta {
  long s1_str;
  long s2_str;
  long b_str;
  long n_str;
  long h_str;
  int in_vnni;
  KVMeta() : s1_str(0), s2_str(0), b_str(0), n_str(0), h_str(0), in_vnni(0) {}
  KVMeta(
      long s1_str,
      long s2_str,
      long b_str,
      long n_str,
      long h_str,
      int in_vnni)
      : s1_str(s1_str),
        s2_str(s2_str),
        b_str(b_str),
        n_str(n_str),
        h_str(h_str),
        in_vnni(in_vnni) {}
};

struct KVCacheMeta {
  KVMeta key;
  KVMeta value;
  long S2;
  int kl_needs_trans;
  std::string hash_str;

  KVCacheMeta() : S2(0), kl_needs_trans(1) {}
  KVCacheMeta(const KVMeta& key, const KVMeta& value, long S2)
      : key(key),
        value(value),
        S2(S2),
        kl_needs_trans(key.s2_str != 1),
        hash_str(hash()) {}

  void print() const {
    printf(
        "Key: s1_str = %ld s2_str = %ld b_str = %ld n_str = %ld h_str = "
        "%ld in_vnni = %d\n",
        key.s1_str,
        key.s2_str,
        key.b_str,
        key.n_str,
        key.h_str,
        key.in_vnni);
    printf(
        "Value: s1_str = %ld s2_str = %ld b_str = %ld n_str = %ld h_str = "
        "%ld in_vnni = %d\n",
        value.s1_str,
        value.s2_str,
        value.b_str,
        value.n_str,
        value.h_str,
        value.in_vnni);
    printf("S2 = %ld kl_needs_trans = %d\n", S2, kl_needs_trans);
  }
  std::string hash() const {
    char buf[200];
    snprintf(
        buf,
        199,
        "ks1%ld_ks2%ld_kb%ld_kn%ld_kh%ld_kv%d_"
        "vs1%ld_vs2%ld_vb%ld_vn%ld_vh%ld_vv%d_knt%d_S2%ld",
        key.s1_str,
        key.s2_str,
        key.b_str,
        key.n_str,
        key.h_str,
        key.in_vnni,
        value.s1_str,
        value.s2_str,
        value.b_str,
        value.n_str,
        value.h_str,
        value.in_vnni,
        kl_needs_trans,
        S2);
    return std::string(buf);
  }
};

struct KVLayout {
  long B, N, H, S2;
  long S1p, S2p, Bp, Np, Hp;
  int in_vnni;
  int VBS;
  const long batch_dim;
  const long S1_dim;
  constexpr static int kSBNH = 0; // [S1, S2, B, N, H]
  constexpr static int kBNSH = 1; // [S1, B, N, S2, H]
  constexpr static int kBNHS = 2; // [S1, B, N, H, S2]
  constexpr static int kvBNSH = 3; // [S1, B, N, S2, H] with vnni
  constexpr static int kvBNHS = 4; // [S1, B, N, H, S2] with vnni
  KVLayout(long B, long N, long H, long S2, long batch_dim = 1, long S1_dim = 0)
      : B(B), N(N), H(H), S2(S2), batch_dim(batch_dim), S1_dim(S1_dim) {}
  virtual int get_layout() const = 0;
  long get_batch_dim() const {
    return batch_dim;
  }
  virtual at::Tensor new_empty(const at::TensorOptions& options) const = 0;
  virtual const KVMeta get_meta() const = 0;
  virtual void copy_states(
      at::Tensor& t_past,
      at::Tensor& t_states,
      const std::function<long(long)>& func) = 0;
  virtual at::Tensor extend(at::Tensor t, long new_S1) const {
    auto sizes = t.sizes();
    auto new_sizes = sizes.vec();
    new_sizes[S1_dim] = new_S1;
    auto t_new = t.new_zeros(new_sizes);
    if (t.numel() > 0) {
      t_new.slice(S1_dim, 0, t.size(S1_dim), 1).copy_(t);
    }
    return t_new;
  }
};

KVLayout* get_kv_layout(
    int layout,
    long B,
    long N,
    long H,
    long S2,
    c10::ScalarType cache_type);

struct LayerCache {
  at::Tensor t_key_past;
  at::Tensor t_value_past;
  long offset;
  long s1_blocks;
  // const
  KVLayout *key_layout, *value_layout;
  constexpr static long block_size = 32;

  LayerCache(KVLayout* key_layout, KVLayout* value_layout)
      : offset(0),
        s1_blocks(0),
        key_layout(key_layout),
        value_layout(value_layout) {}

  long get_capacity() {
    if (t_key_past.numel() == 0) {
      return 0;
    }

    return s1_blocks * block_size;
  }

  void extend_capacity(long new_capacity) {
    if (new_capacity <= get_capacity()) {
      return;
    }
    auto new_S1 = (new_capacity + block_size - 1) / block_size;
    t_key_past = key_layout->extend(t_key_past, new_S1);
    t_value_past = value_layout->extend(t_value_past, new_S1);
    s1_blocks = new_S1;
  }

  std::vector<at::Tensor> update(
      at::Tensor& t_key_states,
      at::Tensor& t_value_states) {
    auto S = t_key_states.size(-2);
    auto capacity = get_capacity();
    if (capacity == 0) {
      t_key_past = key_layout->new_empty(t_key_states.options());
      t_value_past = value_layout->new_empty(t_value_states.options());
    }
    if (capacity < offset + S) {
      if (capacity > 0)
        printf(
            "Warning: Reallocating kv cache, consider increasing KV_CACHE_INC_SIZE (%d)\n",
            KV_CACHE_INC_SIZE);
      auto new_capacity = offset + S + KV_CACHE_INC_SIZE;
      extend_capacity(new_capacity);
    }

    auto get_indices = [&](long s) { return offset + s; };

    key_layout->copy_states(t_key_past, t_key_states, get_indices);
    value_layout->copy_states(t_value_past, t_value_states, get_indices);
    offset += S;
    return {t_key_past, t_value_past};
  }

  void crop(long max_length) {
    if (offset > max_length)
      offset = max_length;
  }
};

struct __attribute__((visibility("hidden"))) TppCache
    : torch::CustomClassHolder {
 public:
  // using LayerCache = LayerCacheXYZ;

  std::vector<LayerCache> layers;
  at::Tensor t_beam_idx;
  at::Tensor t_beam_reorders;
  bool isBeamSearch;
  KVLayout* key_layout;
  KVLayout* value_layout;
  KVCacheMeta cm;
  bool layoutInitialized = false;

  TppCache() : isBeamSearch(false) {}

  void init_layout(at::Tensor& key_states) {
    auto B = key_states.size(0);
    auto N = key_states.size(1);
    auto H = key_states.size(-1);
    auto S2 = LayerCache::block_size;
    auto dtype = key_states.scalar_type();

    key_layout = get_kv_layout(key_layout_id, B, N, H, S2, dtype);
    value_layout = get_kv_layout(value_layout_id, B, N, H, S2, dtype);
    cm = KVCacheMeta(key_layout->get_meta(), value_layout->get_meta(), S2);
    layoutInitialized = true;
  }

  std::vector<at::Tensor> update(
      at::Tensor& key_states,
      at::Tensor& value_states,
      long layer_idx) {
    if (!layoutInitialized) {
      init_layout(key_states);
    }

    while (layer_idx >= (long)layers.size()) {
      layers.emplace_back(key_layout, value_layout);
    }
    auto& layer = layers[layer_idx];
    auto ret = layer.update(key_states, value_states);
    if (isBeamSearch) {
      ret.push_back(t_beam_idx);
    }
    return ret;
  }

  std::vector<at::Tensor> get(long layer_idx) {
    if (layer_idx >= (long)layers.size()) {
      return {};
    }
    auto& layer = layers[layer_idx];
    std::vector<at::Tensor> ret = {layer.t_key_past, layer.t_key_past};
    return ret;
  }

  long get_seq_length(long layer_idx = 0) {
    if (layer_idx >= (long)layers.size()) {
      return 0;
    }
    return layers[layer_idx].offset;
  }

  long get_capacity(long layer_idx = 0) {
    if (layer_idx >= (long)layers.size()) {
      return 0;
    }
    return layers[layer_idx].get_capacity();
  }

  void crop(long max_length) {
    for (auto i = 0; i < (int)layers.size(); i++) {
      layers[i].crop(max_length);
    }
  }

  long size() {
    return layers.size();
  }

  at::Tensor align_and_invert_mask(
      at::Tensor t_mask,
      double pad_value = -10000.0) {
    auto B = t_mask.size(0);
    auto S = t_mask.size(1);
    auto S2 = LayerCache::block_size;
    auto S1 = (S + S2 - 1) / S2;
    auto S_pad = S1 * S2;
    auto mask_pad = t_mask.new_empty({B, S_pad});
    mask_pad.fill_(pad_value);
    mask_pad.slice(1, 0, S, 1).masked_fill_(t_mask == 1, 0.0);
    return mask_pad;
  }

  const KVCacheMeta get_kv_cache_meta() {
    return cm;
  }
  void reorder_cache(at::Tensor t_beam_reorder) {
    isBeamSearch = true;
    // std::cout << "Reordering cache\n";
    // std::cout << "t_beam_reorder: " << t_beam_reorder.sizes() << std::endl;
    // std::cout << "t_beam_reorders: " << t_beam_reorders.sizes() << std::endl;
    // std::cout << "offset = " << layers[0].offset << std::endl;
    // std::cout << "layers[0].t_key_past: " << layers[0].t_key_past.sizes()
    //           << std::endl;
    // std::cout << "layers[0].t_value_past: " << layers[0].t_value_past.sizes()
    // << std::endl;
    auto B = t_beam_reorder.size(0);
    auto k_b_dim = key_layout->get_batch_dim();
    auto v_b_dim = value_layout->get_batch_dim();
    auto offset = layers[0].offset;
    if (t_beam_reorders.numel() == 0) {
      auto nTokens = layers[0].get_capacity();
      t_beam_reorders = t_beam_reorder.new_zeros({nTokens, B});
      t_beam_reorders.slice(0, 0, offset, 1).copy_(t_beam_reorder.unsqueeze(0));
      auto CB = layers[0].t_key_past.size(k_b_dim);
      if (CB != B) {
        TPP_ASSERT(
            B % CB == 0,
            "New batch (%ld) must be multiple of cached batch size (%ld)\n",
            B,
            CB);
        auto nBeams = B / CB;
        // std::cout << "Repeating cache B = " << B << " CB = " << CB << "
        // nBeams = " << nBeams << " \n"; std::cout << "k_b_dim = " << k_b_dim
        // << " v_b_dim = " << v_b_dim << std::endl; Reinitialize the layout to
        // reflect new batch size (B)
        at::Tensor t_dummy = layers[0].t_key_past.new_empty(
            {B, key_layout->N, 1, key_layout->H});
        init_layout(t_dummy);
        for (auto& l : layers) {
          l.t_key_past =
              l.t_key_past.repeat_interleave(nBeams, k_b_dim).contiguous();
          l.key_layout = key_layout;
          l.t_value_past =
              l.t_value_past.repeat_interleave(nBeams, v_b_dim).contiguous();
          l.value_layout = value_layout;
        }
      }
    } else {
      if (t_beam_reorders.size(0) < offset) {
        auto nTokens = layers[0].get_capacity();
        auto t_beam_reorders_new = t_beam_reorder.new_zeros({nTokens, B});
        t_beam_reorders_new.slice(0, 0, t_beam_reorders.size(0), 1)
            .copy_(t_beam_reorders);
        t_beam_reorders = t_beam_reorders_new;
      }
      t_beam_reorders[offset - 1] = t_beam_reorder;
    }
    // Align (offset + 1) to S2
    auto S2 = LayerCache::block_size;
    auto S1 = (offset + S2) / S2;
    auto S_pad = S1 * S2;
    if (t_beam_idx.numel() == 0 || t_beam_idx.size(1) != S_pad) {
      t_beam_idx = t_beam_reorders.new_zeros({B, S_pad});
    }
    auto beam_idx = GetVLAPtr<long>(t_beam_idx, {S_pad});
    auto b_ptr = GetVLAPtr<long>(t_beam_reorders, {B});
    for (auto i = 0; i < B; i++) {
      beam_idx[i][offset] = i;
      beam_idx[i][offset - 1] = b_ptr[offset - 1][i];
      for (auto j = offset - 2; j >= 0; j--) {
        beam_idx[i][j] = b_ptr[j][beam_idx[i][j + 1]];
      }
    }
    // std::cout << "After reorder:" << std::endl;
    // std::cout << "t_beam_reorders: " << t_beam_reorders.sizes() << std::endl;
    // std::cout << "t_beam_idx: " << t_beam_idx.sizes() << std::endl;
    // std::cout << "offset = " << layers[0].offset << std::endl;
    // std::cout << "layers[0].t_key_past: " << layers[0].t_key_past.sizes()
    //           << std::endl;
    // std::cout << "layers[0].t_value_past: " << layers[0].t_value_past.sizes()
    // << std::endl;
  }
};

template <typename T>
at::Tensor attn_wrapper_impl(
    at::Tensor& t_QL,
    at::Tensor& t_KL,
    at::Tensor& t_VL,
    at::Tensor& t_AM,
    c10::intrusive_ptr<TppCache> cache,
    long layer_idx,
    bool isCausal = true);

at::Tensor attn_wrapper(
    at::Tensor& t_QL,
    at::Tensor& t_KL,
    at::Tensor& t_VL,
    at::Tensor& t_AM,
    // c10::intrusive_ptr<TppCache> cache,
    TppCache& cache1,
    long layer_idx,
    bool isCausal = true);
