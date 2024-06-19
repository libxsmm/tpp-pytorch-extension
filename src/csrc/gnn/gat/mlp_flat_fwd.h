/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gat_mlp_fwd", std::vector<c10::IValue>());

at::Tensor t_in, t_wt, t_bias = at::empty(0);
int i = 0;

t_in = inputs[i++]; // [N, C]
t_wt = inputs[i++]; // [nk, nc, bc, bk]
if (add_bias)
  t_bias = inputs[i++]; // [K]

auto in_sizes = t_in.sizes();
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

auto t_out = t_in.new_empty({N, K}); // [N,  K]
auto in = GetVLAPtr<Tact>(t_in, {bn, nc, bcp});
auto wt_V = GetVLAPtr<Tprm>(t_wt_V, {nc, bcp* bk});
auto out = GetVLAPtr<Tact>(t_out, {bn, nk, bk});

{
  RECORD_SCOPE(gmo_gemm, {t_in, t_wt_V});
  {
    if (add_bias) {
      auto bias = GetVLAPtr<Tprm>(t_bias, {bk});
      auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
          bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));
      auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<Tprm, Tact>(bn, bk, K)), BIAS);

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
        auto in = GetVLAPtr<Tact>(t_in, {nc, bcp});
        auto out = GetVLAPtr<Tact>(t_out, {nk, bk});

        auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));

        auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<Tprm, Tact>(1, bk, K)), BIAS);

        brgemm_tpp.config();

        for (int k = 0; k < nk; k++) {
          for (int r = 0; r < rem; r++)
            cpy_bias_tpp(bias[k], out[nn * bn + r][k]);
          brgemm_tpp(in[nn * bn][0], wt_V[k][0], out[nn * bn][k], nc);
        }
        brgemm_tpp.release();
      }
    } else {
      auto brgemm_tpp = SCOPEIT((BrgemmTPP<Tact, Tact, Tprm>(
          bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
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
        auto in = GetVLAPtr<Tact>(t_in, {nc, bcp});
        auto out = GetVLAPtr<Tact>(t_out, {nk, bk});

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

return t_out;
