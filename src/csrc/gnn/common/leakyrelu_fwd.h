/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("leakyrelu_fwd", std::vector<c10::IValue>());

auto t_in = inp;
auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;
auto K = in_sizes[1];
auto dK = (K + 15) / 16;
auto t_lrelu_mask = at::empty({N, dK}, at::kShort);
auto t_out = t_in.new_empty({N, K}); // [N,  K]

auto in = GetVLAPtr<T>(t_in, {bn, K});
auto out = GetVLAPtr<T>(t_out, {bn, K});
auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {bn, dK});

auto leaky_relu_fwd_tpp =
    SCOPEIT((LeakyReLUFwdTPP<T>(bn, K, alpha)), ACT);
{
  RECORD_SCOPE(go_bias_lrelu_drop, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
      leaky_relu_fwd_tpp(in[n][0], out[n][0], lrelu_mask[n][0]);
    }
    if (rem > 0) {
      auto leaky_relu_fwd_tpp =
          SCOPEIT((LeakyReLUFwdTPP<T>(1, K, alpha)), ACT);

      auto in = GetVLAPtr<T>(t_in, {K});
      auto out = GetVLAPtr<T>(t_out, {K});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});

      for (int n = nn * bn; n < nn * bn + rem; n++) {
        leaky_relu_fwd_tpp(in[n], out[n], lrelu_mask[n]);
      }
    }
  }
}

return {t_out.view({N,K,1}), t_lrelu_mask};
