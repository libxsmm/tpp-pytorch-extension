/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("bias_lrelu_fwd", std::vector<c10::IValue>());

int i = 0;
auto t_in = inputs[i++];
auto t_bias = inputs[i];

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;
auto K = in_sizes[1];
auto dK = (K + 15) / 16;
auto t_lrelu_mask = at::empty({N, dK}, at::kShort);
auto t_out = t_in.new_empty({N, K}); // [N,  K]

auto in = GetVLAPtr<Tact>(t_in, {bn, K});
auto bias = GetVLAPtr<Tprm>(t_bias, {K});
auto out = GetVLAPtr<Tact>(t_out, {bn, K});
auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {bn, dK});

auto cvt_f32_tpp = SCOPEIT((ConvertTPP<Tact, float>(bn, K)), EW_COPY);
auto add_bias_tpp = SCOPEIT(AddBiasTPP<Tprm>(bn, K), BIAS);
auto leaky_relu_fwd_tpp =
    SCOPEIT((LeakyReLUFwdTPP<float, Tact>(bn, K, alpha)), ACT);
{
  RECORD_SCOPE(go_bias_lrelu_drop, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      float tmp[bn][K];
#pragma omp for
      for (int n = 0; n < nn; n++) {
        cvt_f32_tpp(in[n][0], tmp[0]);
        add_bias_tpp(bias[0], tmp[0]);
        leaky_relu_fwd_tpp(tmp[0], out[n][0], lrelu_mask[n][0]);
      }
    }
    if (rem > 0) {
      auto cvt_f32_tpp = SCOPEIT((ConvertTPP<Tact, float>(1, K)), EW_COPY);
      auto add_bias_tpp = SCOPEIT(AddBiasTPP<Tprm>(1, K), BIAS);
      auto leaky_relu_fwd_tpp =
          SCOPEIT((LeakyReLUFwdTPP<float, Tact>(1, K, alpha)), ACT);

      auto in = GetVLAPtr<Tact>(t_in, {K});
      auto bias = GetVLAPtr<Tprm>(t_bias, {K});
      auto out = GetVLAPtr<Tact>(t_out, {K});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});
      float tmp[K];

      for (int n = nn * bn; n < nn * bn + rem; n++) {
        cvt_f32_tpp(in[n], tmp);
        add_bias_tpp(bias[0], tmp);
        leaky_relu_fwd_tpp(tmp, out[n], lrelu_mask[n]);
      }
    }
  }
}
return {t_out, t_lrelu_mask};
