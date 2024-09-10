/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("leakyrelu_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_inp = inputs[i++];
auto t_lrelu_mask = inputs[i++];

auto in_sizes = t_grad_out.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn= N / bn;
auto rem = N % bn;
long K = in_sizes[1];
auto dK = (K + 15) / 16;

auto t_grad_in = t_grad_out.new_empty({N, K});

auto grad_out = GetVLAPtr<T>(t_grad_out, {bn, K});
auto inp = GetVLAPtr<T>(t_inp, {bn, K});
auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {bn, dK});
auto grad_in = GetVLAPtr<T>(t_grad_in, {bn, K});

auto lrelu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(bn, K, alpha), ACT);

{
  RECORD_SCOPE(gdo_bias_lrelu, {t_grad_out});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
      lrelu_bwd_tpp(grad_out[n][0], grad_in[n][0], inp[n][0], lrelu_mask[n][0]);
    }
    if(rem > 0) {
      auto grad_out = GetVLAPtr<T>(t_grad_out, {K});
      auto inp = GetVLAPtr<T>(t_inp, {K});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});
      auto grad_in = GetVLAPtr<T>(t_grad_in, {K});

      auto lrelu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(1, K, alpha), ACT);

      for(int n = nn*bn; n < N; n++) {
        lrelu_bwd_tpp(grad_out[n], grad_in[n], inp[n], lrelu_mask[n]);
      }
    }
  }
}

return t_grad_in;
