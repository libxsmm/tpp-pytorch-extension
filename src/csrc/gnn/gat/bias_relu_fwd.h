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

int i=0;
auto t_in = inputs[i++].contiguous();
auto t_bias = inputs[i];

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto K = in_sizes[1];
auto dK = (K + 15) / 16;
auto t_relu_mask = at::empty({N, dK}, at::kShort);

auto t_out = t_in.new_empty({N, K}); // [N,  K]
at::Tensor t_out_f32 = t_out;
if (t_in.dtype() == at::kBFloat16)
  t_out_f32 = at::empty({N, K});

auto in = GetVLAPtr<T>(t_in, {K});
auto bias = GetVLAPtr<float>(t_bias, {K});
auto out = GetVLAPtr<T>(t_out, {K});
auto out_f32 = GetVLAPtr<float>(t_out_f32, {K});
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {dK});

auto relu_fwd_tpp = SCOPEIT((ReLUFwdTPP<float, T>(1, K, true)), ACT);
auto add_bias_tpp = SCOPEIT(AddBiasTPP<float>(1, K), BIAS);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(1, K)), EW_COPY);
{
  RECORD_SCOPE(go_bias_relu, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
        cvt_f32_tpp(in[n], out_f32[n]);
        add_bias_tpp(bias[0], out_f32[n]);
        relu_fwd_tpp(out_f32[n], out[n], relu_mask[n]);
    }
  }
}
return {t_out, t_relu_mask};
