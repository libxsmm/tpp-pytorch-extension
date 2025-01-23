/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gat_mlp_attn_fwd", std::vector<c10::IValue>());

at::Tensor t_in_mlp, t_attn_3d, t_wt, t_bias;
int i = 0;

t_in_mlp = inputs[i++];
t_wt = inputs[i++];
t_attn_3d = inputs[i++];
t_bias = inputs[i++];

at::Tensor t_out_mlp = fc_plain<T>(t_in_mlp, t_wt, t_bias);

auto attn_sizes = t_attn_3d.sizes();

auto N = attn_sizes[0];
auto H = attn_sizes[1];
auto F = attn_sizes[2];

auto t_out_attn = t_out_mlp.new_empty({N, H});
auto out_attn = GetVLAPtr<T>(t_out_attn, {H});

auto t_attn = t_attn_3d.view({H * F});
auto attn = GetVLAPtr<T>(t_attn, {F});

auto in_attn = GetVLAPtr<T>(t_out_mlp, {H, F});

auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<T, T, T>(H, F)), EW_MUL);
{
  RECORD_SCOPE(go_attn, {t_out_attn});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}

return {t_out_mlp, t_out_attn.view({N, H, 1})};
