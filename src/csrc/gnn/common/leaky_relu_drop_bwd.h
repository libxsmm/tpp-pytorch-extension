/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("lrelu_drop_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_inp = inputs[i++];
auto t_lrelu_mask = inputs[i++];
auto t_dp_mask = inputs[i++];

auto t_grad_in = at::empty_like(t_grad_out);

auto N = t_grad_out.numel();

auto grad_out = t_grad_out.data_ptr<T>();
auto inp = t_inp.data_ptr<T>();
auto grad_in = t_grad_in.data_ptr<T>();
auto dp_mask = t_dp_mask.data_ptr<short>();
auto lrelu_mask = t_lrelu_mask.data_ptr<short>();

const int BS = 256; // Define the block size

auto leaky_relu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(BS, alpha), ACT);
auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(BS, p), DROPOUT);
{
  RECORD_SCOPE(gdo_lrelu_drop, {t_grad_out});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    long n;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, BS); n += BS) {
      leaky_relu_bwd_tpp(
          &grad_out[n], &grad_in[n], &inp[i], &lrelu_mask[n / 16]);
      dropout_bwd_tpp(&grad_in[n], &grad_in[n], &dp_mask[n / 16]);
    }

    if (n < N) {
      auto leaky_relu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(N - n, true), ACT);
      auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(N - n, p), DROPOUT);
      leaky_relu_bwd_tpp(
          &grad_out[n], &grad_in[n], &inp[n], &lrelu_mask[n / 16]);
      dropout_bwd_tpp(&grad_in[n], &grad_in[n], &dp_mask[n / 16]);
    }
  }
}
return t_grad_in;
