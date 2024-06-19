/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gat_attn_fwd", std::vector<c10::IValue>());

at::Tensor t_in, t_attn_3d;
int i = 0;

t_in = inputs[i++];
t_attn_3d = inputs[i++]; // [1, H, F]

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];

auto attn_sizes = t_attn_3d.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto H = attn_sizes[1]; // 4
auto F = attn_sizes[2]; // 128

auto t_out_attn = t_in.new_empty({N, H});

auto t_attn = t_attn_3d.view({H * F});

auto in_attn = GetVLAPtr<Tact>(t_in, {H, F});
auto attn = GetVLAPtr<Tprm>(t_attn, {F}); // nk, bk
auto out_attn = GetVLAPtr<Tact>(t_out_attn, {H}); // N, H

auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<Tprm, Tact, Tact>(H, F)), EW_MUL);

{
  RECORD_SCOPE(go_attn, {t_out_attn});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}
return t_out_attn.view({N, H, 1});
