/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("relu_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_relu_mask = inputs[i++];

auto N = t_grad_out.numel();

auto grad_out = t_grad_out.data_ptr<T>();
auto relu_mask = t_relu_mask.data_ptr<short>();

auto t_grad_in = at::empty_like(t_grad_out);
auto grad_in = t_grad_in.data_ptr<T>();

const int BS = 256; // Define the block size

auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<T>(BS, true), ACT);
{
  RECORD_SCOPE(gdo_relu, {t_grad_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    long n;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, BS); n += BS)
      relu_bwd_tpp(&grad_out[n], &grad_in[n], (T*)NULL, &relu_mask[n / 16]);
    if (n < N) {
      auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<T>(N - n, true), ACT);
      relu_bwd_tpp(&grad_out[n], &grad_in[n], (T*)NULL, &relu_mask[n / 16]);
    }
  }
}

return t_grad_in;
