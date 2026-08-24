/******************************************************************************
 * Copyright (c) 2026 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>
#include "init.h"
#include "timing.h"

REGISTER_LOCAL_SCOPE(delta_step, "delta_step");
REGISTER_LOCAL_SCOPE(delta_layer, "delta_layer");
REGISTER_LOCAL_SCOPE(rmsnorm, "rmsnorm");
REGISTER_LOCAL_SCOPE(add_rmsnorm, "add_rmsnorm");
REGISTER_LOCAL_SCOPE(silu_mul, "silu_mul");
REGISTER_LOCAL_SCOPE(sigmoid_mul, "sigmoid_mul");

namespace {

// Measured: spreading even these small decode-shaped ops over the full team
// beats running them serially, so parallelize unconditionally by default.
static const long PAR_MIN_ELEMS = env2int("TPP_PAR_MIN_ELEMS", 0);

inline float siluf(float x) {
  return x / (1.f + expf(-x));
}

// Single-token gated DeltaNet recurrence, fused over the [K,V] state:
//   s1    = g * S
//   delta = (v - k @ s1) * beta
//   out   = q @ s1 + (q.k) * delta
//   S     = s1 + k^T delta
// Expressed via q@S and k@S so the state is streamed once to reduce and once
// to update, instead of once per elementwise term.
inline void delta_step_head(
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    float g,
    float beta,
    float* __restrict S,
    float* __restrict out,
    long K,
    long V,
    float* __restrict qs,
    float* __restrict ks,
    float* __restrict delta) {
  for (long j = 0; j < V; j++) {
    qs[j] = 0.f;
    ks[j] = 0.f;
  }
  float qk = 0.f;
  for (long i = 0; i < K; i++) {
    const float qi = q[i];
    const float ki = k[i];
    qk += qi * ki;
    const float* __restrict row = S + i * V;
#pragma omp simd
    for (long j = 0; j < V; j++) {
      const float sv = row[j];
      qs[j] += qi * sv;
      ks[j] += ki * sv;
    }
  }
#pragma omp simd
  for (long j = 0; j < V; j++) {
    const float d = (v[j] - g * ks[j]) * beta;
    delta[j] = d;
    out[j] = g * qs[j] + qk * d;
  }
  for (long i = 0; i < K; i++) {
    const float ki = k[i];
    float* __restrict row = S + i * V;
#pragma omp simd
    for (long j = 0; j < V; j++) {
      row[j] = g * row[j] + ki * delta[j];
    }
  }
}

at::Tensor gated_delta_step(
    at::Tensor t_q,
    at::Tensor t_k,
    at::Tensor t_v,
    at::Tensor t_g,
    at::Tensor t_beta,
    at::Tensor t_state) {
  RECORD_SCOPE(delta_step, {t_state});
  TORCH_CHECK(t_state.dim() == 3, "state must be [BH, K, V]");
  TORCH_CHECK(
      t_state.scalar_type() == at::kFloat, "gated_delta_step is fp32 only");
  TORCH_CHECK(t_state.is_contiguous(), "state must be contiguous");

  t_q = t_q.contiguous();
  t_k = t_k.contiguous();
  t_v = t_v.contiguous();
  t_g = t_g.contiguous();
  t_beta = t_beta.contiguous();

  const long BH = t_state.size(0);
  const long K = t_state.size(1);
  const long V = t_state.size(2);
  auto t_out = at::empty({BH, V}, t_state.options());

  const float* q_all = t_q.data_ptr<float>();
  const float* k_all = t_k.data_ptr<float>();
  const float* v_all = t_v.data_ptr<float>();
  const float* g_all = t_g.data_ptr<float>();
  const float* b_all = t_beta.data_ptr<float>();
  float* s_all = t_state.data_ptr<float>();
  float* o_all = t_out.data_ptr<float>();

#pragma omp parallel
  {
    std::vector<float> qs(V), ks(V), dl(V);
#pragma omp for
    for (long h = 0; h < BH; h++) {
      delta_step_head(
          q_all + h * K,
          k_all + h * K,
          v_all + h * V,
          g_all[h],
          b_all[h],
          s_all + h * K * V,
          o_all + h * V,
          K,
          V,
          qs.data(),
          ks.data(),
          dl.data());
    }
  }
  return t_out;
}

// Everything in a decode-step DeltaNet layer between in_proj and out_proj:
// causal conv1d state update + silu, qkv split, l2norm, gated delta recurrence
// and the gated RMSNorm. Collapses ~40 eager ops per layer into one call.
// causal_conv1d_update: append the new sample, shift the state, keep the last
// output position, then silu. Runs in the cache's own dtype to stay in place.
template <typename T>
void conv_update_impl(
    const float* __restrict qkv,
    T* __restrict cs,
    const float* __restrict w,
    float* __restrict mixed,
    long B,
    long conv_dim,
    long state_len,
    long kernel) {
#pragma omp parallel for
  for (long i = 0; i < B * conv_dim; i++) {
    const long b = i / conv_dim;
    const long c = i % conv_dim;
    T* __restrict st = cs + (b * conv_dim + c) * state_len;
    const float* __restrict wc = w + c * kernel;
    const float x = qkv[b * conv_dim + c];
    float acc = 0.f;
    for (long t = 0; t < kernel; t++) {
      const long idx = state_len + 1 - kernel + t;
      acc += ((idx < state_len) ? (float)st[idx] : x) * wc[t];
    }
    for (long t = 0; t < state_len - 1; t++)
      st[t] = st[t + 1];
    st[state_len - 1] = (T)x;
    mixed[b * conv_dim + c] = siluf(acc);
  }
}

at::Tensor gated_delta_layer(
    at::Tensor t_qkv, // [B, conv_dim] in_proj_qkv output
    at::Tensor t_conv_state, // [B, conv_dim, state_len] updated in place
    at::Tensor t_conv_w, // [conv_dim, kernel]
    at::Tensor t_z, // [B, HV*V]
    at::Tensor t_b, // [B, HV] pre-sigmoid
    at::Tensor t_a, // [B, HV] pre-softplus
    at::Tensor t_A_log, // [HV]
    at::Tensor t_dt_bias, // [HV]
    at::Tensor t_state, // [B*HV, K, V] fp32, updated in place
    at::Tensor t_norm_w, // [V]
    double eps,
    int64_t num_k_heads) {
  RECORD_SCOPE(delta_layer, {t_state});
  TORCH_CHECK(t_state.dim() == 3 && t_state.is_contiguous());
  TORCH_CHECK(t_state.scalar_type() == at::kFloat, "state must be fp32");

  auto f32 = [](const at::Tensor& t) { return t.to(at::kFloat).contiguous(); };
  auto qkv = f32(t_qkv);
  auto conv_w = f32(t_conv_w);
  auto z = f32(t_z);
  auto bb = f32(t_b);
  auto aa = f32(t_a);
  auto A_log = f32(t_A_log);
  auto dt_bias = f32(t_dt_bias);
  auto norm_w = f32(t_norm_w);
  TORCH_CHECK(t_conv_state.is_contiguous(), "conv state must be contiguous");

  const long B = t_conv_state.size(0);
  const long conv_dim = t_conv_state.size(1);
  const long state_len = t_conv_state.size(2);
  const long kernel = conv_w.size(1);
  const long HV = t_state.size(0) / B;
  const long K = t_state.size(1);
  const long V = t_state.size(2);
  const long key_dim = num_k_heads * K;
  const long rep = HV / num_k_heads;
  TORCH_CHECK(conv_dim == 2 * key_dim + HV * V, "conv_dim mismatch");
  TORCH_CHECK(state_len >= kernel - 1, "conv state shorter than kernel");

  auto t_mixed = at::empty({B, conv_dim}, qkv.options());
  auto t_out = at::empty({B, HV * V}, qkv.options());

  const float* qkv_p = qkv.data_ptr<float>();
  const float* w_p = conv_w.data_ptr<float>();
  float* mixed_p = t_mixed.data_ptr<float>();

  // causal_conv1d_update: append the new sample, shift the state, keep the last
  // output position, then silu. Runs in the cache's own dtype to stay in place.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, t_conv_state.scalar_type(), "conv_update", [&] {
        conv_update_impl<scalar_t>(
            qkv_p,
            t_conv_state.data_ptr<scalar_t>(),
            w_p,
            mixed_p,
            B,
            conv_dim,
            state_len,
            kernel);
      });

  const float* z_p = z.data_ptr<float>();
  const float* b_p = bb.data_ptr<float>();
  const float* a_p = aa.data_ptr<float>();
  const float* Al_p = A_log.data_ptr<float>();
  const float* dt_p = dt_bias.data_ptr<float>();
  const float* nw_p = norm_w.data_ptr<float>();
  float* st_p = t_state.data_ptr<float>();
  float* out_p = t_out.data_ptr<float>();
  const float inv_sqrt_k = 1.f / std::sqrt((float)K);

#pragma omp parallel
  {
    std::vector<float> qs(V), ks(V), dl(V), qn(K), kn(K);
#pragma omp for
    for (long bh = 0; bh < B * HV; bh++) {
      const long b = bh / HV;
      const long h = bh % HV;
      const long hk = h / rep; // q/k are repeat_interleaved over value heads
      const float* mixed = mixed_p + b * conv_dim;
      const float* qsrc = mixed + hk * K;
      const float* ksrc = mixed + key_dim + hk * K;
      const float* vsrc = mixed + 2 * key_dim + h * V;

      float qsum = 0.f, ksum = 0.f;
#pragma omp simd reduction(+ : qsum, ksum)
      for (long i = 0; i < K; i++) {
        qsum += qsrc[i] * qsrc[i];
        ksum += ksrc[i] * ksrc[i];
      }
      const float qr = 1.f / std::sqrt(qsum + 1e-6f) * inv_sqrt_k;
      const float kr = 1.f / std::sqrt(ksum + 1e-6f);
#pragma omp simd
      for (long i = 0; i < K; i++) {
        qn[i] = qsrc[i] * qr;
        kn[i] = ksrc[i] * kr;
      }

      const float beta = 1.f / (1.f + expf(-b_p[b * HV + h]));
      const float sp = log1pf(expf(a_p[b * HV + h] + dt_p[h]));
      const float g = expf(-expf(Al_p[h]) * sp);

      float* o = out_p + b * HV * V + h * V;
      delta_step_head(
          qn.data(),
          kn.data(),
          vsrc,
          g,
          beta,
          st_p + bh * K * V,
          o,
          K,
          V,
          qs.data(),
          ks.data(),
          dl.data());

      float var = 0.f;
#pragma omp simd reduction(+ : var)
      for (long j = 0; j < V; j++)
        var += o[j] * o[j];
      const float rs = 1.f / std::sqrt(var / (float)V + (float)eps);
      const float* zr = z_p + b * HV * V + h * V;
#pragma omp simd
      for (long j = 0; j < V; j++)
        o[j] = nw_p[j] * (o[j] * rs) * siluf(zr[j]);
    }
  }

  return t_out.to(t_z.scalar_type());
}

// x * rsqrt(mean(x^2) + eps) * (1 + w), fp32 accumulation, over the last dim.
template <typename T>
void rmsnorm_impl(
    const T* __restrict x,
    const float* __restrict w,
    T* __restrict y,
    long rows,
    long C,
    float eps) {
#pragma omp parallel for if (rows * C > PAR_MIN_ELEMS)
  for (long r = 0; r < rows; r++) {
    const T* __restrict xr = x + r * C;
    T* __restrict yr = y + r * C;
    float ss = 0.f;
#pragma omp simd reduction(+ : ss)
    for (long i = 0; i < C; i++) {
      const float v = (float)xr[i];
      ss += v * v;
    }
    const float rs = 1.f / std::sqrt(ss / (float)C + eps);
    for (long i = 0; i < C; i++)
      yr[i] = (T)((float)xr[i] * rs * (1.f + w[i]));
  }
}

at::Tensor rmsnorm(at::Tensor t_x, at::Tensor t_w, double eps) {
  RECORD_SCOPE(rmsnorm, {t_x});
  t_x = t_x.contiguous();
  auto w = t_w.to(at::kFloat).contiguous();
  const long C = t_x.size(-1);
  const long rows = t_x.numel() / C;
  auto t_y = at::empty_like(t_x);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, t_x.scalar_type(), "rmsnorm", [&] {
        rmsnorm_impl<scalar_t>(
            t_x.data_ptr<scalar_t>(),
            w.data_ptr<float>(),
            t_y.data_ptr<scalar_t>(),
            rows,
            C,
            (float)eps);
      });
  return t_y;
}

template <typename T>
void add_rmsnorm_impl(
    const T* __restrict x,
    const T* __restrict r,
    const float* __restrict w,
    T* __restrict s,
    T* __restrict y,
    long rows,
    long C,
    float eps) {
#pragma omp parallel for if (rows * C > PAR_MIN_ELEMS)
  for (long i = 0; i < rows; i++) {
    const T* __restrict xr = x + i * C;
    const T* __restrict rr = r + i * C;
    T* __restrict sr = s + i * C;
    T* __restrict yr = y + i * C;
    float ss = 0.f;
#pragma omp simd reduction(+ : ss)
    for (long j = 0; j < C; j++) {
      const float v = (float)xr[j] + (float)rr[j];
      sr[j] = (T)v;
      ss += v * v;
    }
    const float rs = 1.f / std::sqrt(ss / (float)C + eps);
    for (long j = 0; j < C; j++)
      yr[j] = (T)((float)sr[j] * rs * (1.f + w[j]));
  }
}

// Returns (x + residual, rmsnorm(x + residual)) so the residual stream is read
// once for both the sum and the norm.
std::vector<at::Tensor> add_rmsnorm(
    at::Tensor t_x,
    at::Tensor t_r,
    at::Tensor t_w,
    double eps) {
  RECORD_SCOPE(add_rmsnorm, {t_x});
  t_x = t_x.contiguous();
  t_r = t_r.contiguous();
  auto w = t_w.to(at::kFloat).contiguous();
  const long C = t_x.size(-1);
  const long rows = t_x.numel() / C;
  auto t_s = at::empty_like(t_x);
  auto t_y = at::empty_like(t_x);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, t_x.scalar_type(), "add_rmsnorm", [&] {
        add_rmsnorm_impl<scalar_t>(
            t_x.data_ptr<scalar_t>(),
            t_r.data_ptr<scalar_t>(),
            w.data_ptr<float>(),
            t_s.data_ptr<scalar_t>(),
            t_y.data_ptr<scalar_t>(),
            rows,
            C,
            (float)eps);
      });
  return {t_s, t_y};
}

template <typename T>
void silu_mul_impl(
    const T* __restrict g,
    const T* __restrict u,
    T* __restrict y,
    long n) {
#pragma omp parallel for if (n > PAR_MIN_ELEMS)
  for (long i = 0; i < n; i++)
    y[i] = (T)(siluf((float)g[i]) * (float)u[i]);
}

at::Tensor silu_mul(at::Tensor t_g, at::Tensor t_u) {
  RECORD_SCOPE(silu_mul, {t_g});
  t_g = t_g.contiguous();
  t_u = t_u.contiguous();
  auto t_y = at::empty_like(t_g);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, t_g.scalar_type(), "silu_mul", [&] {
        silu_mul_impl<scalar_t>(
            t_g.data_ptr<scalar_t>(),
            t_u.data_ptr<scalar_t>(),
            t_y.data_ptr<scalar_t>(),
            t_g.numel());
      });
  return t_y;
}

template <typename T>
void sigmoid_mul_impl(
    const T* __restrict x,
    const T* __restrict g,
    T* __restrict y,
    long n) {
#pragma omp parallel for if (n > PAR_MIN_ELEMS)
  for (long i = 0; i < n; i++)
    y[i] = (T)((float)x[i] / (1.f + expf(-(float)g[i])));
}

at::Tensor sigmoid_mul(at::Tensor t_x, at::Tensor t_g) {
  RECORD_SCOPE(sigmoid_mul, {t_x});
  t_x = t_x.contiguous();
  t_g = t_g.contiguous();
  auto t_y = at::empty_like(t_x);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, t_x.scalar_type(), "sigmoid_mul", [&] {
        sigmoid_mul_impl<scalar_t>(
            t_x.data_ptr<scalar_t>(),
            t_g.data_ptr<scalar_t>(),
            t_y.data_ptr<scalar_t>(),
            t_x.numel());
      });
  return t_y;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tpp_llm, m) {
  m.def(
      "gated_delta_step(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) state) -> Tensor",
      &gated_delta_step);
  m.def(
      "gated_delta_layer(Tensor qkv, Tensor(a!) conv_state, Tensor conv_w, Tensor z, Tensor b, Tensor a, Tensor A_log, Tensor dt_bias, Tensor(b!) state, Tensor norm_w, float eps, int num_k_heads) -> Tensor",
      &gated_delta_layer);
  m.def("rmsnorm(Tensor x, Tensor w, float eps) -> Tensor", &rmsnorm);
  m.def(
      "add_rmsnorm(Tensor x, Tensor r, Tensor w, float eps) -> Tensor[]",
      &add_rmsnorm);
  m.def("silu_mul(Tensor g, Tensor u) -> Tensor", &silu_mul);
  m.def("sigmoid_mul(Tensor x, Tensor g) -> Tensor", &sigmoid_mul);
}
