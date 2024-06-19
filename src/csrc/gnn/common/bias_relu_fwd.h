/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("bias_relu_fwd", std::vector<c10::IValue>());

int i = 0;
auto t_in = inputs[i++].contiguous();
auto t_bias = inputs[i];

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto K = in_sizes[1];
auto dK = (K + 15) / 16;
auto t_relu_mask = at::empty({N, dK}, at::kShort);

auto t_out = t_in.new_empty({N, K}); // [N,  K]

auto in = GetVLAPtr<Tact>(t_in, {K});
auto bias = GetVLAPtr<Tprm>(t_bias, {K});
auto out = GetVLAPtr<Tact>(t_out, {K});
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {dK});

auto relu_fwd_tpp = SCOPEIT((ReLUFwdTPP<float, Tact>(1, K, true)), ACT);
auto add_bias_tpp = SCOPEIT(AddBiasTPP<Tprm>(1, K), BIAS);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<Tact, float>(1, K)), EW_COPY);
{
  RECORD_SCOPE(go_bias_relu, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      float tmp[K];
#pragma omp for
      for (int n = 0; n < N; n++) {
        cvt_f32_tpp(in[n], tmp);
        add_bias_tpp(bias[0], tmp);
        relu_fwd_tpp(tmp, out[n], relu_mask[n]);
      }
    }
  }
}
return {t_out, t_relu_mask};
