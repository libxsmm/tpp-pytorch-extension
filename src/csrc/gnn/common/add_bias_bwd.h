/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("add_bias_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++].contiguous();

auto in_sizes = t_grad_out.sizes();
auto N = in_sizes[0];
long K = in_sizes[1];

auto grad_out = GetVLAPtr<Tact>(t_grad_out, {K});

auto t_grad_bias = at::empty(0);
if (dparam == 0)
  t_grad_bias = at::empty({K});
else if (dparam == 1)
  t_grad_bias = at::empty({K}, at::kBFloat16);

auto grad_bias = GetVLAPtr<Tprm>(t_grad_bias, {K});

auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tact>(1, K), BIAS);
auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(K), EW_ZERO);

int threads = omp_get_max_threads();

{
  RECORD_SCOPE(gdo_add_bias, {t_grad_out});
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
      for (int n = 0; n < N; n++) {
        grad_bias_tpp(grad_out[n], prv_grad_bias[0]);
      }
#pragma omp barrier
      omp_reduce_buf(threads, K, bias_ptrs, grad_bias[0]);
    }
  }
}

return t_grad_bias;
