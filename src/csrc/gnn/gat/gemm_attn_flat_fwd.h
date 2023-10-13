/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gemm_attn_fwd", std::vector<c10::IValue>());

at::Tensor t_in, t_wt;
int i = 0;

t_in = inputs[i++];
t_wt = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];
auto bcp = bc;
auto K = nk * bk;

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

auto t_out = t_in.new_empty({N, K});
auto in = GetVLAPtr<T>(t_in, {bn, nc, bcp});
auto wt_V = GetVLAPtr<T>(t_wt_V, {nc, bcp* bk});
auto out = GetVLAPtr<T>(t_out, {bn, nk, bk});

{
  RECORD_SCOPE(gao_gemm_attn, {t_in, t_wt_V});
  {
    auto brgemm_tpp = SCOPEIT((BrgemmTPP<T, T>(
        bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n = 0; n < nn; n++) {
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[n][0][0], wt_V[k][0], out[n][0][k], nc);
      }
    }
    if (rem > 0) {
      auto in = GetVLAPtr<T>(t_in, {nc, bcp});
      auto out = GetVLAPtr<T>(t_out, {nk, bk});

      auto brgemm_tpp = SCOPEIT((BrgemmTPP<T, T>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));

#pragma omp parallel for
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[nn * bn][0], wt_V[k][0], out[nn * bn][k], nc);
      }
    }
  }
}

return t_out;
