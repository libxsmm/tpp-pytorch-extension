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

at::Tensor t_in, t_attn_3d, t_wt, t_bias = at::empty(0);
int i = 0;

t_in = inputs[i++];
t_wt = inputs[i++];
t_attn_3d = inputs[i++];
if (add_bias)
  t_bias = inputs[i++];

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
auto C = in_sizes[1];

if (t_wt.dim() == 5) {
  auto lp = get_vnni_block_size(t_wt.dtype());
  auto d = bc % lp;
  bcp = d > 0 ? bc + (lp - d) : bc;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

auto t_out_mlp = at::empty({N, K}, at::kBFloat16); // [N,  K]

auto cvt_tpp = ConvertTPP<Tact, bfloat16>(bn, C);

if (add_bias) {
  auto in = GetVLAPtr<Tact>(t_in, {bn, nc, bcp});
  auto wt_V = GetVLAPtr<bfloat16>(t_wt_V, {nc, bcp * bk});
  auto bias = GetVLAPtr<bfloat16>(t_bias, {bk});
  auto out = GetVLAPtr<bfloat16>(t_out_mlp, {bn, nk, bk});

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<bfloat16, bfloat16, bfloat16>(
          bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));
  auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<bfloat16, bfloat16>(bn, bk, K)), BIAS);

  {
    RECORD_SCOPE(o_mlp_attn, {t_in, t_wt_V});
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

        bfloat16 tmp[bn][nc][bcp];

        brgemm_tpp.config();

        for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
          int n = n3k / nk;
          int k = n3k % nk;

          cpy_bias_tpp(bias[k], out[n][0][k]);
          cvt_tpp(in[n][0][0], tmp[0][0]);
          brgemm_tpp(tmp[0][0], wt_V[k][0], out[n][0][k], nc);
        }
        brgemm_tpp.release();
      }
      if (rem > 0) {
        auto in = GetVLAPtr<Tact>(t_in, {nc, bcp});
        auto out = GetVLAPtr<bfloat16>(t_out_mlp, {nk, bk});

        auto brgemm_tpp = SCOPEIT((BrgemmTPP<bfloat16, bfloat16, bfloat16>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));

        auto cpy_bias_tpp = SCOPEIT((CpyBiasTPP<bfloat16, bfloat16>(1, bk, K)), BIAS);
        auto cvt_tpp = ConvertTPP<Tact, bfloat16>(rem, C);

        bfloat16 tmp[rem][nc][bcp];

        brgemm_tpp.config();

        for (int k = 0; k < nk; k++) {
          for (int r = nn * bn; r < nn * bn + rem; r++)
            cpy_bias_tpp(bias[k], out[r][k]);
          cvt_tpp(in[nn * bn][0], tmp[0][0]);
          brgemm_tpp(tmp[0][0], wt_V[k][0], out[nn * bn][k], nc);
        }
        brgemm_tpp.release();
      }
    }
  }
} else {
  auto in = GetVLAPtr<Tact>(t_in, {bn, nc, bcp});
  auto wt_V = GetVLAPtr<bfloat16>(t_wt_V, {nc, bcp * bk});
  auto out = GetVLAPtr<bfloat16>(t_out_mlp, {bn, nk, bk});

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<bfloat16, bfloat16, bfloat16>(
          bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
  {
    RECORD_SCOPE(o_mlp_attn, {t_in, t_wt_V});
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

        bfloat16 tmp[bn][nc][bcp];

        brgemm_tpp.config();

        for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
          int n = n3k / nk;
          int k = n3k % nk;

          cvt_tpp(in[n][0][0], tmp[0][0]);
          brgemm_tpp(tmp[0][0], wt_V[k][0], out[n][0][k], nc);
        }

        brgemm_tpp.release();
      }
      if (rem > 0) {
        auto in = GetVLAPtr<Tact>(t_in, {nc, bcp});
        auto out = GetVLAPtr<bfloat16>(t_out_mlp, {nk, bk});

        auto brgemm_tpp = SCOPEIT((BrgemmTPP<bfloat16, bfloat16, bfloat16>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
        auto cvt_tpp = ConvertTPP<Tact, bfloat16>(rem, C);

        bfloat16 tmp[rem][nc][bcp];

        brgemm_tpp.config();
        for (int k = 0; k < nk; k++) {
          cvt_tpp(in[nn * bn][0], tmp[0][0]);
          brgemm_tpp(tmp[0][0], wt_V[k][0], out[nn * bn][k], nc);
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
