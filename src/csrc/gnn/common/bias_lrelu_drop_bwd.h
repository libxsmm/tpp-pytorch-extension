/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("bias_lrelu_drop_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_inp = inputs[i++];
auto t_lrelu_mask = inputs[i++];
auto t_dp_mask = (p > 0) ? inputs[i++] : at::empty(0, at::kShort);

auto in_sizes = t_grad_out.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;
long K = in_sizes[1];
auto dK = (K + 15) / 16;

auto t_grad_in = t_grad_out.new_empty({N, K});
auto t_grad_bias = at::empty(0);
if (dparam == 0)
  t_grad_bias = at::empty({K});
else if (dparam == 1)
  t_grad_bias = at::empty({K}, at::kBFloat16);

auto grad_out = GetVLAPtr<Tact>(t_grad_out, {bn, K});
auto inp = GetVLAPtr<Tact>(t_inp, {bn, K});
auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {bn, dK});

auto grad_in = GetVLAPtr<Tact>(t_grad_in, {bn, K});
auto grad_bias = GetVLAPtr<Tprm>(t_grad_bias, {K});

int threads = omp_get_max_threads();

if (p > 0) {
  auto leaky_relu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<Tact>(bn, K, alpha), ACT);
  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tact>(bn, K), BIAS);
  auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(K), EW_ZERO);
  auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<Tact>(bn, K, p), DROPOUT);

  auto dp_mask = GetVLAPtr<short>(t_dp_mask, {bn, dK});

  RECORD_SCOPE(gdo_bias_lrelu_drop, {t_grad_out});
  {
    tensor_set_zero(1, K, t_grad_bias);
    float* bias_ptrs[threads];
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[1][K];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for
      for (int n = 0; n < nn; n++) {
        dropout_bwd_tpp(grad_out[n][0], grad_in[n][0], dp_mask[n][0]);
        leaky_relu_bwd_tpp(
            grad_in[n][0], grad_in[n][0], inp[n][0], lrelu_mask[n][0]);
        grad_bias_tpp(grad_in[n][0], prv_grad_bias[0]);
      }
#pragma omp barrier
      omp_reduce_buf(threads, K, bias_ptrs, grad_bias[0]);
    }
    if (rem > 0) {
      auto leaky_relu_bwd_tpp =
          SCOPEIT(LeakyReLUBwdTPP<Tact>(1, K, alpha), ACT);
      auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tact>(1, K), BIAS);
      auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<Tact>(1, K, p), DROPOUT);

      auto dp_mask = GetVLAPtr<short>(t_dp_mask, {dK});
      auto grad_out = GetVLAPtr<Tact>(t_grad_out, {K});
      auto inp = GetVLAPtr<Tact>(t_inp, {K});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});
      auto grad_in = GetVLAPtr<Tact>(t_grad_in, {K});

      float prv_grad_bias[1][K];
      bias_ptrs[0] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);

      for (int n = nn * bn; n < nn * bn + rem; n++) {
        dropout_bwd_tpp(grad_out[n], grad_in[n], dp_mask[n]);
        leaky_relu_bwd_tpp(grad_in[n], grad_in[n], inp[n], lrelu_mask[n]);
        grad_bias_tpp(grad_in[n], prv_grad_bias[0]);
      }
      omp_reduce_buf(1, K, bias_ptrs, grad_bias[0], true);
    }
  }
} else {
  RECORD_SCOPE(gdo_bias_lrelu_drop, {t_grad_out});
  {
    auto leaky_relu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<Tact>(bn, K, alpha), ACT);
    auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tact>(bn, K), BIAS);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(K), EW_ZERO);

    tensor_set_zero(1, K, t_grad_bias);
    float* bias_ptrs[threads];
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[1][K];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for
      for (int n = 0; n < nn; n++) {
        leaky_relu_bwd_tpp(
            grad_out[n][0], grad_in[n][0], inp[n][0], lrelu_mask[n][0]);
        grad_bias_tpp(grad_in[n][0], prv_grad_bias[0]);
      }
#pragma omp barrier
      omp_reduce_buf(threads, K, bias_ptrs, grad_bias[0]);
    }
    if (rem > 0) {
      auto leaky_relu_bwd_tpp =
          SCOPEIT(LeakyReLUBwdTPP<Tact>(1, K, alpha), ACT);
      auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tact>(1, K), BIAS);

      auto grad_out = GetVLAPtr<Tact>(t_grad_out, {K});
      auto inp = GetVLAPtr<Tact>(t_inp, {K});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});
      auto grad_in = GetVLAPtr<Tact>(t_grad_in, {K});

      float prv_grad_bias[1][K];
      bias_ptrs[0] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);

      for (int n = nn * bn; n < nn * bn + rem; n++) {
        leaky_relu_bwd_tpp(grad_out[n], grad_in[n], inp[n], lrelu_mask[n]);
        grad_bias_tpp(grad_in[n], prv_grad_bias[0]);
      }
      omp_reduce_buf(1, K, bias_ptrs, grad_bias[0], true);
    }
  }
}

return {t_grad_in, t_grad_bias};
