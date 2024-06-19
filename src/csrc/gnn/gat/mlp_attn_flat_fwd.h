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

at::Tensor t_in_mlp, t_attn_3d, t_wt, t_bias = at::empty(0);
int i = 0;

t_in_mlp = inputs[i++];
t_wt = inputs[i++];
t_attn_3d = inputs[i++];
if (add_bias)
  t_bias = inputs[i++];

auto in_sizes = t_in_mlp.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dim() == 5)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];
auto bcp = bc;
auto K = nk * bk;

if (t_wt.dim() == 5) {
  auto lp = get_vnni_block_size(t_wt.dtype());
  auto d = bc % lp;
  bcp = d > 0 ? bc + (lp - d) : bc;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

auto t_out_mlp = t_in_mlp.new_empty({N, K}); // [N,  K]

if (add_bias) {
  auto in = GetVLAPtr<Tact>(t_in_mlp, {bn, nc, bcp});
  auto wt_V = GetVLAPtr<Tprm>(t_wt_V, {nc, bcp * bk});
  auto bias = GetVLAPtr<Tprm>(t_bias, {bk});
  auto out = GetVLAPtr<Tact>(t_out_mlp, {bn, nk, bk});

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
      bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));

  auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<Tprm, Tact>(bn, bk, K)), BIAS);

  {
    RECORD_SCOPE(gao_gemm, {t_in_mlp, t_wt_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int work = nn * nk;
        int chunk =
            (work % threads == 0) ? (work / threads) : (work / threads) + 1;
        int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
        int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

        brgemm_tpp.config();

        for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
          int n = n3k / nk;
          int k = n3k % nk;
          cpy_bias_tpp(bias[k], out[n][0][k]);
          brgemm_tpp(in[n][0][0], wt_V[k][0], out[n][0][k], nc);
        }
        brgemm_tpp.release();
      }
      if (rem > 0) {
        auto in = GetVLAPtr<Tact>(t_in_mlp, {nc, bcp});
        auto out = GetVLAPtr<Tact>(t_out_mlp, {nk, bk});

        auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));

        auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<Tprm, Tact>(1, bk, K)), BIAS);

        brgemm_tpp.config();

        for (int k = 0; k < nk; k++) {
          for (int r = nn * bn; r < nn * bn + rem; r++)
            cpy_bias_tpp(bias[k], out[r][k]);
          brgemm_tpp(in[nn * bn][0], wt_V[k][0], out[nn * bn][k], nc);
        }
        brgemm_tpp.release();
      }
    }
  }
} else {
  auto in = GetVLAPtr<Tact>(t_in_mlp, {bn, nc, bcp});
  auto wt_V = GetVLAPtr<Tprm>(t_wt_V, {nc, bcp * bk});
  auto out = GetVLAPtr<Tact>(t_out_mlp, {bn, nk, bk});

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
      bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));

  {
    RECORD_SCOPE(gao_gemm, {t_in_mlp, t_wt_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int work = nn * nk;
        int chunk =
            (work % threads == 0) ? (work / threads) : (work / threads) + 1;
        int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
        int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

        brgemm_tpp.config();

        for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
          int n = n3k / nk;
          int k = n3k % nk;
          brgemm_tpp(in[n][0][0], wt_V[k][0], out[n][0][k], nc);
        }

        brgemm_tpp.release();
      }
      if (rem > 0) {
        auto in = GetVLAPtr<Tact>(t_in_mlp, {nc, bcp});
        auto out = GetVLAPtr<Tact>(t_out_mlp, {nk, bk});

        auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));

        brgemm_tpp.config();
        for (int k = 0; k < nk; k++) {
          brgemm_tpp(in[nn * bn][0], wt_V[k][0], out[nn * bn][k], nc);
        }
        brgemm_tpp.release();
      }
    }
  }
}

auto attn_sizes = t_attn_3d.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto H = attn_sizes[1]; // 4
auto F = attn_sizes[2]; // 128

auto t_out_attn = t_out_mlp.new_empty({N, H});
auto out_attn = GetVLAPtr<Tact>(t_out_attn, {H}); // N, H

auto t_attn = t_attn_3d.view({H * F});
auto attn = GetVLAPtr<Tprm>(t_attn, {F}); // nk, bk

auto in_attn = GetVLAPtr<Tact>(t_out_mlp, {H, F});

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

return {t_out_mlp, t_out_attn.view({N, H, 1})};
