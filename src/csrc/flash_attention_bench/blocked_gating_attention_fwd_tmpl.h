/******************************************************************************
 * Copyright (c) 2025 Intel Corporation - All rights reserved.                *
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

#ifdef __x86_64__
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

int64_t Sp_t = S_t;

T* sfmask = new (std::align_val_t(64)) T[S_t - Sp_t];     /* create mask */
for (int i = 0; i < S_t - Sp_t; i++) {
  sfmask[i] = -30000;
}
auto sfmask_a = GetVLAPtr<T>(sfmask, {1L});

auto q_data_a = GetVLAPtr<T>(q_data, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE, H_t});   // [B, Ns, N, Bs * H]
auto m_data_a = GetVLAPtr<T>(m_data, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE, H_t});
auto bias_a = GetVLAPtr<float>(bias, {S_t/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});
auto nonbatched_bias_a = GetVLAPtr<float>(nonbatched_bias, {N_t, S_t/QKVO_BLOCKSIZE, S_t/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});

auto query_w_a = GetVLAPtr<T>(query_w, {N_t, H_t, H_t});    // [Nc, Nk, Hc, Hk]
auto key_w_a = GetVLAPtr<T>(key_w, {N_t, H_t, H_t});
auto value_w_a = GetVLAPtr<T>(value_w, {N_t, H_t, H_t});
auto gating_w_a = GetVLAPtr<T>(gating_w, {N_t, H_t, H_t});
auto gating_b_a = GetVLAPtr<float>(gating_b, {H_t});
auto output_w_a = GetVLAPtr<T>(output_w, {N_t, H_t, H_t});
auto output_b_a = GetVLAPtr<float>(output_b, {H_t});

T* q = new (std::align_val_t(64)) T[B_t * S_t * N_t *H_t];
auto q_a = GetVLAPtr<T>(q, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE * H_t});    // [B, Ns, N, Bs * H]

T* k = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto k_a = GetVLAPtr<T>(k, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE * H_t});    // [B, Ns, N, H * Bs]

T* v = new (std::align_val_t(64)) T[B_t *S_t * N_t *H_t];
auto v_a = GetVLAPtr<T>(v, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE * H_t});    // [B, Ns, N, Bs * H]

T* weighted_avg = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto weighted_avg_a = GetVLAPtr<T>(weighted_avg, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE * H_t}); // [B, Ns, N, Bs * H]

auto output_a = GetVLAPtr<T>(output, {S_t/QKVO_BLOCKSIZE, N_t, QKVO_BLOCKSIZE, H_t});   // [B, Ns, N, Bs * H]

auto qkv_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKVO_BLOCKSIZE, H_t, H_t, QKVO_BLOCKSIZE*H_t, N_t*H_t*H_t, H_t, H_t, H_t, 0.0, 0, 1)));

auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(QKVO_BLOCKSIZE * H_t)), EW_SCL);
auto copy_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE * H_t), EW_COPY);
float alpha = (1.0 / sqrt(H_t));

auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < (S_t/QKVO_BLOCKSIZE); j++) {
        for (int k = 0; k < N_t; k++) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * H_t], 64);
          qkv_brgemm_tpp(&q_data_a[i][j][0][0][0], &query_w_a[0][k][0][0], &tmp[0], N_t);
          scale_tpp(&tmp[0], &tmp[0], alpha);
          copy_tpp(&tmp[0], &q_a[i][j][k][0]);
        }
      }
    }
  }
}

auto end_time = std::chrono::high_resolution_clock::now(); // End timing
auto q_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

auto k_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKVO_BLOCKSIZE,
        H_t,
        H_t,
        QKVO_BLOCKSIZE,
        H_t,
        QKVO_BLOCKSIZE,
        XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_k_gemm, {k, m_data, key_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < (S_t/QKVO_BLOCKSIZE); j++) {
        for (int k = 0; k < N_t; k++) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * H_t], 64);
          qkv_brgemm_tpp(&m_data_a[i][j][0][0][0], &key_w_a[0][k][0][0], &tmp[0], N_t);
          k_trans_tpp(&tmp[0], &k_a[i][j][k][0]); // [ 0*H_t*S_t + 0*S_t + j]
        }
      }
    }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto k_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();


start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_v_gemm, {v, m_data, value_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < (S_t/QKVO_BLOCKSIZE); j++) {
        for (int k = 0; k < N_t; k++) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * H_t], 64);
          qkv_brgemm_tpp(&m_data_a[i][j][0][0][0], &value_w_a[0][k][0][0], &tmp[0], N_t);
          copy_tpp(&tmp[0], &v_a[i][j][k][0]);
        }
      }
    }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto v_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

start_time = std::chrono::high_resolution_clock::now(); // Start timing

  auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE, H_t), EW_COPY);
  auto a_cpy2_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE, H_t), EW_COPY);

  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE,
      H_t,
      0,
      0,
      H_t,
      QKVO_BLOCKSIZE,
      Ak_BLOCKSIZE,
      0.0,
      0,
      1)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      QKVO_BLOCKSIZE,
      H_t,
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE,
      N_t*QKVO_BLOCKSIZE*H_t,
      Ak_BLOCKSIZE,
      H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_softmax_online_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(QKVO_BLOCKSIZE, Ak_BLOCKSIZE, true)), SOFTMAX);
  auto a_softmax_fixup_online =
      SCOPEIT(SoftMaxFixUpTPP<T>(QKVO_BLOCKSIZE, H_t, true), EW_RCP);
  auto a_softmax_scale_online =
      SCOPEIT(SoftMaxFlashScaleTPP<T>(QKVO_BLOCKSIZE, H_t, true), EW_RCP);

  auto a_addbias_online_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, Ak_BLOCKSIZE), BIAS);
  auto a_add_nbbias_online_tpp = SCOPEIT(
      (AddTPP<float, float>(
          QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, Ak_BLOCKSIZE, Ak_BLOCKSIZE)),
      BIAS);

  int lastBlockSize = S_t - (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
  if (lastBlockSize == 0)
    lastBlockSize = Ak_BLOCKSIZE; // handling the zero case

  auto a_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE,
      H_t,
      0,
      0,
      H_t,
      QKVO_BLOCKSIZE,
      lastBlockSize,
      0.0,
      0,
      1)));

  auto c_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      QKVO_BLOCKSIZE,
      H_t,
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE,
      N_t*QKVO_BLOCKSIZE*H_t,
      lastBlockSize,
      H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_online_edge_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, lastBlockSize), BIAS);
  auto a_add_nbbias_online_edge_tpp = SCOPEIT(
      (AddTPP<float, float>(
          QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, lastBlockSize, lastBlockSize)),
      BIAS);

  auto a_add_sfmask_online_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, S_t - Sp_t, lastBlockSize), BIAS);

  auto a_softmax_online_edge_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(QKVO_BLOCKSIZE, lastBlockSize, true)), SOFTMAX);

  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float tmp_S[QKVO_BLOCKSIZE * Ak_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float tmp_o1[QKVO_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float tmp_o2[QKVO_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float omax[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float osum[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float cmax[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float csum[QKVO_BLOCKSIZE], 64);

    #pragma omp for collapse(3) nowait
    for (int i = 0; i < B_t; i++) {
      for (int n = 0; n < N_t; n++) {
        for (int j1 = 0; j1 < (S_t/QKVO_BLOCKSIZE); j1++) {
          for (int j2 = 0; j2 < (S_t / Ak_BLOCKSIZE); j2++) {
            for (int f = 0; f < (Ak_BLOCKSIZE/QKVO_BLOCKSIZE); f++) {
              a_brgemm_tpp(
                  &q_a[i][j1][n][0], &k_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][n][0], &tmp_S[f*QKVO_BLOCKSIZE], 1);

              if (bias_flag)
                a_addbias_online_tpp(&bias_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][0], &tmp_S[f*QKVO_BLOCKSIZE]);

              if (nbbias_flag)
                a_add_nbbias_online_tpp(
                    &nonbatched_bias_a[0][n][j1][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][0][0], &tmp_S[f*QKVO_BLOCKSIZE], &tmp_S[f*QKVO_BLOCKSIZE]);
            }

            if (j2 == 0) {
              a_softmax_online_tpp(1, tmp_S, tmp_S, omax, osum, nullptr);
            } else {
              a_softmax_online_tpp(1, tmp_S, tmp_S, cmax, csum, omax);
            }

            // c_brgemm_tpp(tmp_S, &v_a[i][j2][n][0], tmp_o1,
            //               1); // O = P*V
            
            c_brgemm_tpp(tmp_S, &v_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE)][n][0], tmp_o1, (Ak_BLOCKSIZE/QKVO_BLOCKSIZE)); // O = P*V
            

            if (j2 == 0) {
              a_cpy2_tpp(tmp_o1, tmp_o2);
            } else {
              a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
            }
          }

          if (S_t % Ak_BLOCKSIZE != 0) {
            T* tmp_S_edge = new (std::align_val_t(64)) T[QKVO_BLOCKSIZE * lastBlockSize];
            int j2 = (S_t / Ak_BLOCKSIZE);
            for (int f = 0; f < (lastBlockSize/QKVO_BLOCKSIZE); f++) {
              a_brgemm_edge_tpp(
                  &q_a[i][j1][n][0],
                  &k_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][n][0],
                  &tmp_S_edge[f*QKVO_BLOCKSIZE],
                  1);

              if (bias_flag)
                a_addbias_online_edge_tpp(&bias_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][0], &tmp_S_edge[f*QKVO_BLOCKSIZE]);
              if (nbbias_flag)
                a_add_nbbias_online_edge_tpp(
                    &nonbatched_bias_a[0][n][j1][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE) + f][0][0], &tmp_S_edge[f*QKVO_BLOCKSIZE], &tmp_S_edge[f*QKVO_BLOCKSIZE]);
            }

            a_add_sfmask_online_tpp(&sfmask_a[0][0], &tmp_S_edge[Sp_t - j2]);
            a_softmax_online_edge_tpp(
                1, tmp_S_edge, tmp_S_edge, cmax, csum, omax);

            // c_brgemm_edge_tpp(tmp_S_edge, &v_a[i][j2][n * H_t], tmp_o1, 1);
            c_brgemm_edge_tpp(tmp_S_edge, &v_a[i][j2*(Ak_BLOCKSIZE/QKVO_BLOCKSIZE)][n][0], tmp_o1, (lastBlockSize/QKVO_BLOCKSIZE));
            a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
            delete[] tmp_S_edge;
          }

          a_softmax_scale_online(&tmp_o2[0], osum);
          a_cpy_tpp(&tmp_o2[0], &weighted_avg_a[i][j1][n][0]);
        }
      }
    }
  }

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto ac_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

auto g_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKVO_BLOCKSIZE, H_t, H_t, QKVO_BLOCKSIZE*H_t, N_t*H_t*H_t, H_t, H_t, H_t, 0.0, 0, 1)));

auto g_addbias_tpp = SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, H_t), BIAS);
auto g_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<T>(QKVO_BLOCKSIZE, H_t), EW_MUL);
auto g_mul_tpp = SCOPEIT((MulTPP<T, T>(QKVO_BLOCKSIZE, H_t)), EW_MUL);

auto out_gemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKVO_BLOCKSIZE, H_t, H_t, QKVO_BLOCKSIZE*H_t, N_t*H_t*H_t, H_t, H_t, H_t, 0.0, 0, 1)));
auto out_addbias_tpp = SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, H_t), BIAS);

start_time = std::chrono::high_resolution_clock::now(); // Start timing

{
  // RECORD_SCOPE(alpha_o_gemm, {weighted_avg, v, q_data, gating_w, gating_b});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    if (gate_flag){
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < (S_t/QKVO_BLOCKSIZE); j++) {
        LIBXSMM_ALIGNED(T tmp_gate_values[N_t][QKVO_BLOCKSIZE * H_t], 64);
        for (int k=0; k < N_t; k++) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * H_t], 64);
          LIBXSMM_ALIGNED(T tmp_sig[QKVO_BLOCKSIZE * H_t], 64);

          g_brgemm_tpp(&q_data_a[i][j][0][0][0], &gating_w_a[0][k][0][0], &tmp[0], N_t);
          g_addbias_tpp(&gating_b_a[k][0], &tmp[0]);

          g_sigmoid_tpp(&tmp[0], &tmp[0], &tmp_sig[0]);
          copy_tpp(&tmp_sig[0], &tmp_gate_values[k][0]);
          g_mul_tpp(
              &tmp_gate_values[k][0],
              &weighted_avg_a[i][j][k][0],
              &tmp_gate_values[k][0]);
        }
        for (int k=0; k < N_t; k++) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * H_t], 64);
          out_gemm_tpp(
              &tmp_gate_values[0][0], &output_w_a[0][k][0][0], &tmp[0], N_t);
          if (bias_flag)
            out_addbias_tpp(&output_b_a[k][0], &tmp[0]);
          copy_tpp(&tmp[0], &output_a[i][j][k][0][0]);
        }
      }
    }
  } else {
#pragma omp parallel for collapse(3)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < (S_t/QKVO_BLOCKSIZE); j++) {
          for (int k=0; k < N_t; k++) {
            LIBXSMM_ALIGNED(float tmp[QKVO_BLOCKSIZE * H_t], 64);
            out_gemm_tpp(
                &weighted_avg_a[i][j][0][0], &output_w_a[0][k][0][0], &tmp[0], N_t);
            if (bias_flag)
              out_addbias_tpp(&output_b_a[k][0], &tmp[0]);
            copy_tpp(&tmp[0], &output_a[i][j][k][0][0]);
          }
        }
      }
    }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto o_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

// make a std::vector of the times
std::vector<long int> times = {q_gemm_time, k_gemm_time, v_gemm_time, ac_gemm_time, o_gemm_time};

delete[] sfmask;
delete[] q;
delete[] k;
delete[] v;
delete[] weighted_avg;