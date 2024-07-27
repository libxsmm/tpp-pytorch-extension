/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("mlp_attn", std::vector<c10::IValue>());

at::Tensor t_in, t_attn_3d, t_wt, t_bias;
int i = 0;

t_in = inputs[i++];
t_wt = inputs[i++];
t_attn_3d = inputs[i++];
if (add_bias)
  t_bias = inputs[i++];
else
  t_bias = at::empty(0,at::kBFloat16);

at::Tensor t_out_mlp = fc_plain<bfloat16>(t_in, t_wt, t_bias);

auto attn_sizes = t_attn_3d.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto N = t_out_mlp.sizes()[0];
auto H = attn_sizes[1]; // 4
auto F = attn_sizes[2]; // 128

auto t_out_attn = t_out_mlp.new_empty({N, H});
auto out_attn = GetVLAPtr<bfloat16>(t_out_attn, {H}); // N, H

auto t_attn = t_attn_3d.view({H * F});
auto attn = GetVLAPtr<bfloat16>(t_attn, {F}); // nk, bk

auto in_attn = GetVLAPtr<bfloat16>(t_out_mlp, {H, F});

auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<bfloat16, bfloat16, bfloat16>(H, F)), EW_MUL);
{
  RECORD_SCOPE(o_attn, {t_out_attn});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}

return {t_out_mlp, t_out_attn.view({N, H, 1})};
