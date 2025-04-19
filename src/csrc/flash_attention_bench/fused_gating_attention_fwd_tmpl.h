/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

// RECORD_FUNCTION(
//     "Gating attention forward",
//     std::vector<c10::IValue>({q_data, m_data})); // For recording time

int64_t Sp_t = S_t;

T* sfmask = new (std::align_val_t(64)) T[S_t - Sp_t];     /* create mask */
for (int i = 0; i < S_t - Sp_t; i++) {
  sfmask[i] = -30000;
}
auto sfmask_a = GetVLAPtr<T>(sfmask, {1L});

auto q_data_a = GetVLAPtr<T>(q_data, {S_t, HS_t});
auto m_data_a = GetVLAPtr<T>(m_data, {S_t, HS_t});
auto bias_a = GetVLAPtr<float>(bias, {S_t});
auto nonbatched_bias_a = GetVLAPtr<float>(nonbatched_bias, {N_t, S_t, S_t});

auto query_w_a = GetVLAPtr<T>(query_w, {N_t, H_t});
auto key_w_a = GetVLAPtr<T>(key_w, {N_t, H_t});
auto value_w_a = GetVLAPtr<T>(value_w, {N_t, H_t});
auto gating_w_a = GetVLAPtr<T>(gating_w, {N_t, H_t});
auto gating_b_a = GetVLAPtr<float>(gating_b, {H_t});
auto output_w_a = GetVLAPtr<T>(output_w, {H_t, HS_t});
auto output_b_a = GetVLAPtr<float>(output_b, {1L});

T* q = new (std::align_val_t(64)) T[B_t * S_t * N_t *H_t];
auto q_a = GetVLAPtr<T>(q, {S_t, N_t, H_t});

T* k = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto k_a = GetVLAPtr<T>(k, {S_t * N_t * H_t});

T* v = new (std::align_val_t(64)) T[B_t *S_t * N_t *H_t];
auto v_a = GetVLAPtr<T>(v, {S_t, N_t, H_t});

T* weighted_avg = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto weighted_avg_a = GetVLAPtr<T>(weighted_avg, {S_t, N_t, H_t});

auto output_a = GetVLAPtr<T>(output, {S_t, HS_t});


int lda = HS_t;
int ldb = N_t * H_t;
int ldc = N_t * H_t;

auto qkv_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKV_BLOCKSIZE, N_t* H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));

auto k_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKV_BLOCKSIZE,
        N_t* H_t,
        N_t* H_t,
        QKV_BLOCKSIZE,
        N_t* H_t,
        S_t,
        XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(QKV_BLOCKSIZE * HS_t)), EW_SCL);
auto zero_tpp = SCOPEIT(SetZeroTPP<T>(QKV_BLOCKSIZE * HS_t), EW_ZERO);
float alpha = (1.0 / sqrt(H_t));

// auto q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}),
// (1.0/sqrt(key_dim))) ;     /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] */

{
  // RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        T tmp[QKV_BLOCKSIZE * N_t * H_t];
        qkv_brgemm_tpp(&q_data_a[i][j][0], &query_w_a[0][0][0], &tmp[0], 1);
        scale_tpp(&tmp[0], &q_a[i][j][0][0], alpha);
      }
    }
  }
}

// auto k = at::einsum("bka,ahc->bkhc", {m_data, key_w}); /* [512, 764, 8, 32]
// = [512, 764, 256] * [256, 8, 32] */

{
  // RECORD_SCOPE(alpha_k_gemm, {k, m_data, key_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        T tmp[QKV_BLOCKSIZE * N_t * H_t];
        qkv_brgemm_tpp(&m_data_a[i][j][0], &key_w_a[0][0][0], &tmp[0], 1);
        k_trans_tpp(&tmp[0], &k_a[i][j]); // [ 0*H_t*S_t + 0*S_t + j]
      }
    }
  }
}

// auto v = at::einsum("bka,ahc->bkhc", {m_data, value_w}); /* [512, 764, 8, 32]
// = [512, 764, 256] * [256, 8, 32] */
{
  // RECORD_SCOPE(alpha_v_gemm, {v, m_data, value_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        qkv_brgemm_tpp(
            &m_data_a[i][j][0], &value_w_a[0][0][0], &v_a[i][j][0][0], 1);
      }
    }
  }
}

lda = H_t;
ldb = A_BLOCKSIZE;
ldc = S_t;

// logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias); /* [512, 8,
// 764, 764]  = [512, 764, 8, 32] * [512, 764, 8, 32] + [512, 1, 1, 764] */ if
// (nonbatched_bias.size(0) > 0)
//     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0)); /* [512, 8,
//     764, 764]  = [512, 8, 764, 764] + [1, 8, 764, 764] */
// weights = at::_softmax(logits, -1, false); /* [512, 8, 764, 764] = [512, 8,
// 764, 764] */ auto weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights,
// v}).contiguous();          /* [512, 764, 8, 32]  = [512, 8, 764, 764] * [512,
// 764, 8, 32] */

auto a_zero_tpp = SCOPEIT(SetZeroTPP<T>(A_BLOCKSIZE * H_t), EW_ZERO);
auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t, H_t, N_t* H_t), EW_COPY);

if (S_t < 2048) {
  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE, A_BLOCKSIZE, H_t, 0, 0, N_t * H_t, S_t, S_t, 0.0, 0, 1)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      A_BLOCKSIZE,
      A_BLOCKSIZE,
      A_BLOCKSIZE * N_t * H_t,
      S_t,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_tpp = SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, S_t, S_t), BIAS);
  auto a_add_nbbias_tpp =
      SCOPEIT((AddTPP<float, float>(A_BLOCKSIZE, S_t, S_t, S_t)), BIAS);

  auto a_add_sfmask_tpp =
      SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, S_t - Sp_t, S_t), BIAS);
  auto a_softmax_tpp =
      SCOPEIT((VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, S_t)), SOFTMAX);

  {
    // RECORD_SCOPE(alpha_ac_gemm, {q, k, bias});
    {
      // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int i = 0; i < B_t; i++) {
        for (int n = 0; n < N_t; n++) {
          for (int j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
            // T tmp_o[A_BLOCKSIZE * H_t];
            // T tmp_logits[A_BLOCKSIZE][S_t];
            LIBXSMM_ALIGNED(T tmp_o[A_BLOCKSIZE * H_t], 64);
            LIBXSMM_ALIGNED(T tmp_logits[A_BLOCKSIZE][S_t], 64);

            for (int j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
              a_brgemm_tpp(
                  &q_a[i][j1][n][0],
                  &k_a[i][n * H_t * S_t + j2],
                  &tmp_logits[0][j2],
                  1);
            }
            if (bias_flag)
              a_addbias_tpp(&bias_a[i][0], &tmp_logits[0][0]);

            if (flag)
              a_add_nbbias_tpp(
                  &nonbatched_bias_a[0][n][j1][0],
                  &tmp_logits[0][0],
                  &tmp_logits[0][0]);

            if (S_t == Sp_t) {
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits[0][0]);
            } else {
              a_add_sfmask_tpp(&sfmask_a[0][0], &tmp_logits[0][Sp_t]);
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits[0][0]);
            }

            c_brgemm_tpp(
                &tmp_logits[0][0],
                &v_a[i][0][n][0],
                &tmp_o[0],
                S_t / A_BLOCKSIZE);
            a_cpy_tpp(&tmp_o[0], &weighted_avg_a[i][j1][n][0]);
          }
        }
      }
    }
  }
} else {
  auto a_cpy2_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t), EW_COPY);

  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      Ak_BLOCKSIZE,
      H_t,
      0,
      0,
      N_t * H_t,
      S_t,
      Ak_BLOCKSIZE,
      0.0,
      0,
      1)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      Ak_BLOCKSIZE,
      0,
      0,
      Ak_BLOCKSIZE,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_online_tpp =
      SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, Ak_BLOCKSIZE, Ak_BLOCKSIZE), BIAS);
  auto a_add_nbbias_online_tpp = SCOPEIT(
      (AddTPP<float, float>(
          A_BLOCKSIZE, Ak_BLOCKSIZE, S_t, Ak_BLOCKSIZE, Ak_BLOCKSIZE)),
      BIAS);

  auto a_softmax_online_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, Ak_BLOCKSIZE, true)), SOFTMAX);
  auto a_softmax_fixup_online =
      SCOPEIT(SoftMaxFixUpTPP<T>(A_BLOCKSIZE, H_t, true), EW_RCP);
  auto a_softmax_scale_online =
      SCOPEIT(SoftMaxFlashScaleTPP<T>(A_BLOCKSIZE, H_t, true), EW_RCP);

  // if (S_t % Ak_BLOCKSIZE != 0){
  int lastBlockSize = S_t - (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
  if (lastBlockSize == 0)
    lastBlockSize = Ak_BLOCKSIZE; // handling the zero case
  auto a_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      lastBlockSize,
      H_t,
      0,
      0,
      N_t * H_t,
      S_t,
      lastBlockSize,
      0.0,
      0,
      1)));

  auto c_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      lastBlockSize,
      0,
      0,
      lastBlockSize,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_online_edge_tpp =
      SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, lastBlockSize, lastBlockSize), BIAS);
  auto a_add_nbbias_online_edge_tpp = SCOPEIT(
      (AddTPP<float, float>(
          A_BLOCKSIZE, lastBlockSize, S_t, lastBlockSize, lastBlockSize)),
      BIAS);

  auto a_add_sfmask_online_tpp =
      SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, S_t - Sp_t, lastBlockSize), BIAS);

  auto a_softmax_online_edge_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, lastBlockSize, true)), SOFTMAX);
  // }


  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float tmp_S[A_BLOCKSIZE * Ak_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float tmp_o1[A_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float tmp_o2[A_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float omax[A_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float osum[A_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float cmax[A_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float csum[A_BLOCKSIZE], 64);

    int64_t start_full = rdtsc_ordered();
      #pragma omp for collapse(3) nowait
      for (int i = 0; i < B_t; i++) {
        for (int n = 0; n < N_t; n++) {
          for (int j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
            for (int j2 = 0; j2 < (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
                 j2 += Ak_BLOCKSIZE) {

              a_brgemm_tpp(
                  &q_a[i][j1][n][0], &k_a[i][n * H_t * S_t + j2], tmp_S, 1);

              if (bias_flag)
                a_addbias_online_tpp(&bias_a[i][j2], tmp_S);

              if (flag)
                a_add_nbbias_online_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S, tmp_S);

              if (j2 == 0) {
                a_softmax_online_tpp(1, tmp_S, tmp_S, omax, osum, nullptr);
              } else {
                a_softmax_online_tpp(1, tmp_S, tmp_S, cmax, csum, omax);
              }

              c_brgemm_tpp(tmp_S, &v_a[i][j2][n][0], tmp_o1,
                           1); // O = P*V

              if (j2 == 0) {
                a_cpy2_tpp(tmp_o1, tmp_o2);
              } else {
                a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              }
            }

            if (S_t % Ak_BLOCKSIZE != 0) {
              T* tmp_S_edge = new (std::align_val_t(64)) T[A_BLOCKSIZE * lastBlockSize];
              int j2 = (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
              a_brgemm_edge_tpp(
                  &q_a[i][j1][n][0],
                  &k_a[i][n * H_t * S_t + j2],
                  tmp_S_edge,
                  1);

              a_addbias_online_edge_tpp(&bias_a[i][j2], tmp_S_edge);
              if (flag) {
                a_add_nbbias_online_edge_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S_edge, tmp_S_edge);
              }

              a_add_sfmask_online_tpp(&sfmask_a[0][0], &tmp_S_edge[Sp_t - j2]);
              a_softmax_online_edge_tpp(
                  1, tmp_S_edge, tmp_S_edge, cmax, csum, omax);

              c_brgemm_edge_tpp(tmp_S_edge, &v_a[i][j2][n][0], tmp_o1, 1);
              a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              delete[] tmp_S_edge;
            }

            a_softmax_scale_online(&tmp_o2[0], osum);
            a_cpy_tpp(&tmp_o2[0], &weighted_avg_a[i][j1][n][0]);
          }
        }
      }

    int64_t end = rdtsc_ordered();
    printf( "Time taken by thread %d is %0.2lf \n", omp_get_thread_num(), (end - start_full)/(double)1e9 );
  }
//   }
}
printf( "\n ----------------------------New threads starting -------------------------- \n" );

lda = HS_t;
ldb = N_t * H_t;
ldc = N_t * H_t;

auto g_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(C_BLOCKSIZE, N_t* H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
auto g_addbias_tpp = SCOPEIT(AddBiasTPP<float>(C_BLOCKSIZE, N_t* H_t, ldc), BIAS);
auto g_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<T>(C_BLOCKSIZE, N_t* H_t, ldc, ldc), EW_MUL);
auto g_mul_tpp = SCOPEIT((MulTPP<T, T>(C_BLOCKSIZE * N_t * H_t)), EW_MUL);

auto out_gemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(C_BLOCKSIZE, HS_t, N_t* H_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
auto out_addbias_tpp = SCOPEIT(AddBiasTPP<float>(C_BLOCKSIZE, HS_t, ldc), BIAS);

// gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data,
// gating_w}), gating_b));   /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] + [8, 32]*/ weighted_avg = at::mul(weighted_avg, gate_values); /* [512,
// 764, 8, 32]  = [512, 764, 8, 32] * [512, 764, 8, 32] */ output =
// at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b); /*
// [512, 764, 256]  = [512, 764, 8, 32] * [8, 32, 256] + [256] */

{
  // RECORD_SCOPE(alpha_o_gemm, {weighted_avg, v, q_data, gating_w, gating_b});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += C_BLOCKSIZE) {
        T tmp[C_BLOCKSIZE * N_t * H_t];
        T tmp_gate_values[C_BLOCKSIZE * N_t * H_t];

        g_brgemm_tpp(&q_data_a[i][j][0], &gating_w_a[0][0][0], &tmp[0], 1);
        g_addbias_tpp(&gating_b_a[0][0], &tmp[0]);

        g_sigmoid_tpp(&tmp[0], &tmp[0], &tmp_gate_values[0]);
        g_mul_tpp(
            &tmp_gate_values[0],
            &weighted_avg_a[i][j][0][0],
            &tmp_gate_values[0]);
        out_gemm_tpp(
            &tmp_gate_values[0], &output_w_a[0][0][0], &output_a[i][j][0], 1);
        out_addbias_tpp(&output_b_a[0][0], &output_a[i][j][0]);
      }
    }
  }
}

delete[] sfmask;
delete[] q;
delete[] k;
delete[] v;
delete[] weighted_avg;