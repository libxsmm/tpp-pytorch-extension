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
auto bk = 16;
auto nk = K / bk;
auto dK = (bk + 15) / 16;
auto t_lrelu_mask = at::empty({N, dK}, at::kShort);
auto t_out = t_in.new_empty({N, K}); // [N,  K]

auto in = GetVLAPtr<Tact>(t_in, {bn, nk, bk});
auto bias = GetVLAPtr<Tprm>(t_bias, {bk});
auto out = GetVLAPtr<Tact>(t_out, {bn, nk, bk});
auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {bn, dK});

auto cvt_f32_tpp = SCOPEIT((ConvertTPP<Tact, float>(bn, bk)), EW_COPY);
auto add_bias_tpp = SCOPEIT(AddBiasTPP<Tprm>(bn, bk), BIAS);
auto leaky_relu_fwd_tpp =
    SCOPEIT((LeakyReLUFwdTPP<float, Tact>(bn, bk, alpha)), ACT);
{
  RECORD_SCOPE(go_bias_lrelu_drop, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      float tmp[bn][bk];
#pragma omp for
      for (int n = 0; n < nn; n++) {
        for(int k = 0; k < nk; k++) {
          cvt_f32_tpp(in[n][0][k], tmp[0]);
          add_bias_tpp(bias[k], tmp[0]);
          leaky_relu_fwd_tpp(tmp[0], out[n][0][k], lrelu_mask[n][0]);
        }
      }
    }
    if (rem > 0) {
      auto cvt_f32_tpp = SCOPEIT((ConvertTPP<Tact, float>(1, bk)), EW_COPY);
      auto add_bias_tpp = SCOPEIT(AddBiasTPP<Tprm>(1, bk), BIAS);
      auto leaky_relu_fwd_tpp =
          SCOPEIT((LeakyReLUFwdTPP<float, Tact>(1, bk, alpha)), ACT);

      auto in = GetVLAPtr<Tact>(t_in, {nk, bk});
      auto bias = GetVLAPtr<Tprm>(t_bias, {bk});
      auto out = GetVLAPtr<Tact>(t_out, {nk, bk});
      auto lrelu_mask = GetVLAPtr<short>(t_lrelu_mask, {dK});
      float tmp[bk];

      for (int n = nn * bn; n < nn * bn + rem; n++) {
        for (int k=0; k < nk; k++) {
          cvt_f32_tpp(in[n][k], tmp);
          add_bias_tpp(bias[k], tmp);
          leaky_relu_fwd_tpp(tmp, out[n][k], lrelu_mask[n]);
        }
      }
    }
  }
}
return {t_out, t_lrelu_mask};
