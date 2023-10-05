/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gemm_attn_bwd", std::vector<c10::IValue>());

at::Tensor t_in, t_wt;

int i = 0;

const int threads = omp_get_max_threads();

auto t_grad_out = inputs[i++].contiguous(); 

t_in = inputs[i++]; 
t_wt = inputs[i++]; 

auto in_sizes = t_in.sizes(); 
auto N = in_sizes[0];

auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto wt_sizes = t_wt.sizes();
auto C = in_sizes[1];

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];

auto bnp = bn;
auto bkp = bk;
auto bcp = bc;
auto remp = rem;

auto K = nk * bk;

if (t_in.dtype() == at::kBFloat16) {
  bnp = bn + bn % 2;
  remp = rem + rem % 2;
}

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
  bkp = bk + bk % 2;
}

//----------------------------
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);

auto t_wt_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt);

auto t_grad_in = t_in.new_empty({N, C});

auto t_grad_wt = at::empty_like(t_wt);
at::Tensor t_grad_wt_tmp;
if (t_wt.dtype() == at::kBFloat16)
  t_grad_wt_tmp = at::empty({nk, nc, bc, bk});
else
  t_grad_wt_tmp = t_grad_wt;

auto grad_out = GetVLAPtr<T>(t_grad_out, {bn, nk, bk});
auto grad_in = GetVLAPtr<T>(t_grad_in, {bn, nc, bc});

// del-weights and weights in blocked layout
auto grad_wt = GetVLAPtr<T>(t_grad_wt, {nc, bc* bk});
auto grad_wt_tmp = GetVLAPtr<float>(t_grad_wt_tmp, {nc, bc* bk});

auto wt_TV = GetVLAPtr<T>(t_wt_TV, {nc, bkp* bc});

auto in = GetVLAPtr<T>(t_in, {bn, nc, bc}); // flat layout for fp32

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(nk * bk), EW_ZERO);
auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(bn, 1, bkp), EW_ZERO);
auto n2v_tpp = SCOPEIT(
    XformExtTPP<T>(bn, bk, bnp, bk, nk* bk, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto n2v_wt_tpp = SCOPEIT(
    XformExtTPP<T>(bc, bk, bcp, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto cpy_tpp = SCOPEIT(CpyTPP<T>(bn, bk, bk, bkp), EW_COPY);
auto add_gwt_tpp = SCOPEIT((AddTPP<float, float>(bc, bk)), EW_ADD);

auto brgemm_di_tpp = SCOPEIT(
    (BrgemmTPP<
        T,
        T>(bn, bc, bkp, bkp, nc* bc* bkp, nk* bkp, bc, nc* bc, 0.0, 0, nk)));

auto brgemm_dw_f32_tpp = SCOPEIT((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    C* bnp,
    K* bnp,
    C,
    K,
    bk,
    0.0,
    input_trans_flag,
    16)));
auto brgemm_dw_f32_tpp_b1 = SCOPEIT((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    C* bnp,
    K* bnp,
    C,
    K,
    bk,
    1.0,
    input_trans_flag,
    16)));

// BF16 del-wt brgemms
auto brgemm_dw_bf16_tpp =
    SCOPEIT((BrgemmTPP<
             T,
             float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 0.0, 0, 16)));
auto brgemm_dw_bf16_tpp_b1 =
    SCOPEIT((BrgemmTPP<
             T,
             float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 1.0, 0, 16)));

//=================================================================================================================================
//==========================================Grad_bias & Grad_in fused
//=============================================================
//=================================================================================================================================

{
  RECORD_SCOPE(gadi_gemm_attn, {t_grad_out, t_grad_in});
  {
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
	T tmp[bn][nk][bkp];

#pragma omp for 
	for (int n = 0; n < nn; n++) {
	  // Grad_in Brgemm computation if bk != bkp
	  if (bk != bkp) {
	    for (int k = 0; k < nk; k++)
	      set_zero_col_tpp(&tmp[0][k][bk]);

	    brgemm_di_tpp.config();
	    for (int c = 0; c < nc; c++) {
	      for (int k = 0; k < nk; k++)
		cpy_tpp(grad_out[n][0][k], tmp[0][k]);
	      brgemm_di_tpp(tmp[0][0], wt_TV[0][c], grad_in[n][0][c], nk, true);
	    }
	    brgemm_di_tpp.release();
	  } else { // Grad_in Brgemm computation if bk == bkp
	    brgemm_di_tpp.config();
	    for (int c = 0; c < nc; c++) {
	      brgemm_di_tpp(
		  grad_out[n][0][0], wt_TV[0][c], grad_in[n][0][c], nk, true);
	    }
	    brgemm_di_tpp.release();
	  }
	} 
      } 

      if (rem > 0) {

	auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});

	// Grad_in-----------------------------------------------------
	{
	  auto grad_in = GetVLAPtr<T>(t_grad_in, {nc, bc});
	  auto wt_TV = GetVLAPtr<T>(t_wt_TV, {nc, bc * bkp});

	  auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(rem, 1, bkp), EW_ZERO);
	  auto cpy_tpp = SCOPEIT(CpyTPP<T>(rem, bk, bk, bkp), EW_COPY);
	  auto brgemm_di_tpp = SCOPEIT((BrgemmTPP<T, T>(
		  rem,
		  bc,
		  bkp,
		  bkp,
		  nc * bc * bkp,
		  nk * bkp,
		  bc,
		  nc * bc,
		  0.0,
		  0,
		  nk)));

	  if (bk != bkp) {
	    T tmp[rem][nk][bkp];

	    for (int k = 0; k < nk; k++)
	      set_zero_col_tpp(&tmp[0][k][bk]);

	    brgemm_di_tpp.config();

	    for (int c = 0; c < nc; c++) {
	      for (int k = 0; k < nk; k++)
		cpy_tpp(grad_out[nn * bn][k], tmp[0][k]);
	      brgemm_di_tpp(
		  tmp[0][0], wt_TV[0][c], grad_in[nn * bn][c], nk, true);
	    }
	    brgemm_di_tpp.release();
	  } else { // Grad_in Brgemm computation if bk == bkp
	    brgemm_di_tpp.config();
	    for (int c = 0; c < nc; c++) {
	      brgemm_di_tpp(
		  grad_out[nn * bn][0],
		  wt_TV[0][c],
		  grad_in[nn * bn][c],
		  nk,
		  true);
	    }
	    brgemm_di_tpp.release();
	  }
	}
      } // rem > 0
    }
  }
}

auto trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(bn, bc, bc, bnp, nc* bc, bnp, XformTPP::XFORM_XPOSE_TPP, true),
    XPOSE);
{
  RECORD_SCOPE(gadw_gemm_attn, {t_in, t_grad_out});
  {
    {
      int upd_n_weight_copies;
      int BF;

#if 1
      upd_n_weight_copies = nk * nc < 4 * threads ? threads : 1;
      BF = 32;
#else
      BF = atoi(getenv("BF"));
      upd_n_weight_copies = atoi(getenv("UPD_WEIGHT_COPIES"));
#endif
      const int fm_blocking = (bk % 16 == 0) ? 16 : bk;
      const int reduce_work = nk * nc * (bk / fm_blocking) * bc;
      const int reduce_chunksize = (reduce_work % threads == 0)
          ? (reduce_work / threads)
          : (reduce_work / threads) + 1;
      const int chunk0 = reduce_chunksize * fm_blocking;
      const int chunk1 =
          (reduce_work - (reduce_work / reduce_chunksize) * reduce_chunksize) *
          fm_blocking;

      int blocks_per_layer =
          (nn + upd_n_weight_copies - 1) / upd_n_weight_copies;
      // std::cout << " upd_n_weight_copies: " << upd_n_weight_copies << " nn "
      // << nn << " blocks_per_layer " << blocks_per_layer << std::endl;
      int reduce_rows = (nn % blocks_per_layer == 0)
          ? (nn / blocks_per_layer)
          : (nn / blocks_per_layer) + 1;

      auto wt_reduce_chunk0_tpp = SCOPEIT(
          (ReduceAddColTPP<float, float>(reduce_rows, chunk0, K * C, chunk0)),
          EW_RED);
      auto wt_reduce_chunk1_tpp = SCOPEIT(
          (ReduceAddColTPP<float, float>(reduce_rows, chunk1, K * C, chunk1)),
          EW_RED);
      auto setzero_delwt_tpp = SCOPEIT(SetZeroTPP<float>(bc * bk), EW_ZERO);

      at::Tensor t_grad_wt_priv =
          at::empty({upd_n_weight_copies, nk, nc, bc * bk});

      auto grad_wt_priv = GetVLAPtr<float>(t_grad_wt_priv, {nk, nc, bc * bk});

      at::Tensor t_global_tmp_go = at::empty(0);
      at::Tensor t_global_tmp_inT = at::empty(0);

      if (t_grad_out.dtype() == at::kBFloat16 &&
          t_wt.dtype() == at::kBFloat16) {
        t_global_tmp_go =
            at::empty({threads, (nn / BF + 1), bnp * bk}, at::kBFloat16);
        t_global_tmp_inT =
            at::empty({threads, nc, (nn / BF + 1), bnp * bc}, at::kBFloat16);
      }
      auto global_tmp_go =
          GetVLAPtr<T>(t_global_tmp_go, {(nn / BF + 1), bnp * bk});
      auto global_tmp_inT =
          GetVLAPtr<T>(t_global_tmp_inT, {nc, (nn / BF + 1), bnp * bc});

      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        unsigned long long blocks;
        int team_id = tid / (threads / upd_n_weight_copies);
        int in_team_id = tid % (threads / upd_n_weight_copies);
        int mb_blocks_chunksize = (nn % upd_n_weight_copies == 0)
            ? (nn / upd_n_weight_copies)
            : ((nn / upd_n_weight_copies) + 1);
        const int my_mb_start = (team_id * mb_blocks_chunksize < nn)
            ? (team_id * mb_blocks_chunksize)
            : nn;
        const int my_mb_end = ((team_id + 1) * mb_blocks_chunksize < nn)
            ? ((team_id + 1) * mb_blocks_chunksize)
            : nn;
        const int my_mb_blocks = my_mb_end - my_mb_start;
        const int threads_per_team = threads / upd_n_weight_copies;
        int ifm_chunk = (nc % threads_per_team == 0)
            ? nc / threads_per_team
            : nc / threads_per_team + 1;
        int my_ifm_start =
            (in_team_id * ifm_chunk < nc) ? in_team_id * ifm_chunk : nc;
        int my_ifm_end = ((in_team_id + 1) * ifm_chunk < nc)
            ? (in_team_id + 1) * ifm_chunk
            : nc;
        int mb_block_step = (my_mb_blocks % BF == 0)
            ? (my_mb_blocks / BF)
            : ((my_mb_blocks / BF) + 1);

        if (t_grad_out.dtype() == at::kBFloat16)
          brgemm_dw_bf16_tpp_b1.config();

        for (int bfn = my_mb_start; bfn < my_mb_end; bfn += mb_block_step) {
          blocks = (bfn + mb_block_step <= my_mb_end) ? mb_block_step
                                                      : my_mb_end - bfn;
          for (int ofm1 = 0; ofm1 < nk; ++ofm1) {
            if (t_in.dtype() == at::kBFloat16) {
              n2v_tpp(
                  blocks,
                  K * bnp,
                  bk * bnp,
                  grad_out[bfn][0][ofm1],
                  global_tmp_go[tid][0]);
            }
            for (int ifm1 = my_ifm_start; ifm1 < my_ifm_end; ++ifm1) {
              if (bfn == my_mb_start) {
                /* initiaize current work task to zero */
                setzero_delwt_tpp(grad_wt_priv[team_id][ofm1][ifm1]);
              }
              if (t_in.dtype() == at::kFloat) {
                brgemm_dw_f32_tpp_b1(
                    in[bfn][0][ifm1],
                    grad_out[bfn][0][ofm1],
                    grad_wt_priv[team_id][ofm1][ifm1],
                    blocks);
              } else if (t_in.dtype() == at::kBFloat16) {
                if (ofm1 == 0)
                  trans_tpp(
                      blocks,
                      C * bnp,
                      bc * bnp,
                      in[bfn][0][ifm1],
                      global_tmp_inT[tid][ifm1 - my_ifm_start][0]);

                brgemm_dw_bf16_tpp_b1(
                    global_tmp_inT[tid][ifm1 - my_ifm_start][0],
                    global_tmp_go[tid][0],
                    grad_wt_priv[team_id][ofm1][ifm1],
                    blocks,
                    true);
              }
            }
          }
        }

        if (t_grad_out.dtype() == at::kBFloat16)
          brgemm_dw_bf16_tpp_b1.release();

        const int reduce_thr_begin = (tid * reduce_chunksize < reduce_work)
            ? (tid * reduce_chunksize)
            : reduce_work;
        const int reduce_thr_end = ((tid + 1) * reduce_chunksize < reduce_work)
            ? ((tid + 1) * reduce_chunksize)
            : reduce_work;
#pragma omp barrier
        float* in = grad_wt_priv[0][0][0] + reduce_thr_begin * fm_blocking;
        float* out = grad_wt_tmp[0][0] + reduce_thr_begin * fm_blocking;

        if ((reduce_thr_end - reduce_thr_begin) == reduce_chunksize) {
          wt_reduce_chunk0_tpp(in, out);
        } else {
          if ((reduce_thr_end - reduce_thr_begin) > 0) {
            wt_reduce_chunk1_tpp(in, out);
          }
        }
      }
    } // nn > 0
    if (rem > 0) {
      auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});
      auto in = GetVLAPtr<T>(t_in, {nc, bc});

      auto brgemm_dw_f32_tpp_b1 = SCOPEITGEMM2((BrgemmTPP<T, float>(
          bc,
          bk,
          remp,
          C * remp,
          K * remp,
          C,
          K,
          bk,
          1.0,
          input_trans_flag,
          1)));
      auto brgemm_dw_bf16_tpp_b1 = SCOPEITGEMM((BrgemmTPP<T, float>(
          bc, bk, remp, bc * remp, bk * remp, remp, bk, bk, 1.0, 0, 1)));
      auto n2v_tpp = SCOPEIT(
          XformExtTPP<T>(
              rem, bk, remp, bk, nk * bk, bk, XformTPP::XFORM_N2V_TPP, true),
          VNNI);
      auto trans_tpp = SCOPEIT(
          XformExtTPP<T>(
              rem,
              bc,
              bc,
              remp,
              nc * bc,
              remp,
              XformTPP::XFORM_XPOSE_TPP,
              true),
          XPOSE);

      if (t_wt.dtype() == at::kFloat) {
#pragma omp parallel for collapse(2)
        for (int k = 0; k < nk; k++) {
          for (int c = 0; c < nc; c++) {
            brgemm_dw_f32_tpp_b1(
                in[nn * bn][c], grad_out[nn * bn][k], grad_wt_tmp[k][c], 1);
          }
        }
      } else if (t_wt.dtype() == at::kBFloat16) {
        T tmp_go[remp * bk], tmp_inT[remp * bc];
#pragma omp parallel
        {
          int tid = omp_get_thread_num();
          int threads = omp_get_num_threads();
          int work = nk * nc;
          int chunk =
              (work % threads == 0) ? (work / threads) : (work / threads) + 1;
          int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
          int chunk_end =
              ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

          brgemm_dw_bf16_tpp_b1.config();

          for (int kk = chunk_start; kk < chunk_end; kk++) {
            int k = kk / nc;
            int c = kk % nc;

            n2v_tpp(grad_out[nn * bn][k], tmp_go);
            trans_tpp(in[nn * bn][c], tmp_inT);
            brgemm_dw_bf16_tpp_b1(tmp_inT, tmp_go, grad_wt_tmp[k][c], 1, true);
          }
          brgemm_dw_bf16_tpp_b1.release();
        }
      }
    }
    if (t_wt.dtype() == at::kBFloat16) {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < nk; k++) {
        for (int c = 0; c < nc; c++) {
          n2v_wt_tpp(grad_wt_tmp[k][c], grad_wt[k][c]);
        }
      }
    }
  }
}

return {t_grad_in, t_grad_wt};
