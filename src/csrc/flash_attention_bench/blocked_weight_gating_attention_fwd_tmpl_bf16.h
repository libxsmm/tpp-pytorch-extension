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

float* sfmask = new (std::align_val_t(64)) float[S_t - Sp_t];     /* create mask */
for (int i = 0; i < S_t - Sp_t; i++) {
  sfmask[i] = -30000;
}
auto sfmask_a = GetVLAPtr<float>(sfmask, {1L});

auto q_data_a = GetVLAPtr<T>(q_data, {S_t, N_t * H_t});   // [B, Ns, N, Bs * H]
auto m_data_a = GetVLAPtr<T>(m_data, {S_t, N_t * H_t});
auto bias_a = GetVLAPtr<float>(bias, {S_t});
auto nonbatched_bias_a = GetVLAPtr<float>(nonbatched_bias, {N_t, S_t, S_t});

auto query_w_a = GetVLAPtr<T>(query_w, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});    // [Nc, Nk, Hc, Hk]
auto key_w_a = GetVLAPtr<T>(key_w, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});
auto value_w_a = GetVLAPtr<T>(value_w, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});
auto gating_w_a = GetVLAPtr<T>(gating_w, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});
auto gating_b_a = GetVLAPtr<float>(gating_b, {1L});
auto output_w_a = GetVLAPtr<T>(output_w, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});
auto output_b_a = GetVLAPtr<float>(output_b, {1L});

T* q = new (std::align_val_t(64)) T[B_t * S_t * N_t *H_t];
auto q_a = GetVLAPtr<T>(q, {S_t, N_t * H_t});    // [B, Ns, N, Bs * H]

T* k = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto k_a = GetVLAPtr<T>(k, {S_t * N_t * H_t});    // [B, Ns, N, H * Bs]

T* v = new (std::align_val_t(64)) T[B_t *S_t * N_t *H_t];
auto v_a = GetVLAPtr<T>(v, {S_t * N_t * H_t});    // [B, Ns, N, Bs * H]

T* weighted_avg = new (std::align_val_t(64)) T[B_t * S_t * N_t * H_t];
auto weighted_avg_a = GetVLAPtr<T>(weighted_avg, {S_t, N_t * H_t}); // [B, Ns, N, Bs * H]

auto output_a = GetVLAPtr<T>(output, {S_t, N_t * H_t});   // [B, Ns, N, Bs * H]

auto qgo_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE*QKVO_BLOCKSIZE, N_t*H_t, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, 0.0, 0, 1, b_vnni)));
auto q_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, N_t*H_t)), EW_ZERO);

auto kv_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE*QKVO_BLOCKSIZE, N_t*H_t, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, 0.0, 0, 1, b_vnni)));

auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(QKVO_BLOCKSIZE * QKVO_BLOCKSIZE)), EW_SCL);
auto copy_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE * H_t), EW_COPY);
float alpha = (1.0 / sqrt(H_t));

auto qkv_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(QKVO_BLOCKSIZE, 
          QKVO_BLOCKSIZE, 
          QKVO_BLOCKSIZE, 
          QKVO_BLOCKSIZE, 
          QKVO_BLOCKSIZE, 
          QKVO_BLOCKSIZE, 
          XformTPP::XFORM_N2V_TPP),
    VNNI);

T* qkv_w_vnni = new (std::align_val_t(64)) T[HS_t * N_t * H_t]; 
auto qkv_w_vnni_a = GetVLAPtr<T>(qkv_w_vnni, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});


auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    if(b_vnni) {
      #pragma omp parallel for collapse(2)
      for (int n1=0; n1 < (N_t*H_t)/QKVO_BLOCKSIZE; n1++){
        for (int n2=0; n2 < (N_t*H_t)/QKVO_BLOCKSIZE; n2++){
          qkv_vnni_trans_tpp(&query_w_a[n1][n2][0][0], &qkv_w_vnni_a[n1][n2][0][0]);
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKVO_BLOCKSIZE) {
        for (int k = 0; k < (N_t*H_t); k += QKVO_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          if (b_vnni)
            qgo_brgemm_tpp(&q_data_a[i][j][0], &qkv_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
          else
            qgo_brgemm_tpp(&q_data_a[i][j][0], &query_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
          scale_tpp(&tmp[0], &tmp[0], alpha);
          q_convert_tpp(&tmp[0], &q_a[i][j][k]);
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
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        S_t,
        (b_vnni) ? XformTPP::XFORM_XPOSE_N2V_TPP : XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

auto v_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        QKVO_BLOCKSIZE,
        N_t*H_t,
        XformTPP::XFORM_N2V_TPP),
    VNNI);


start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_k_gemm, {k, m_data, key_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    if(b_vnni) {
      #pragma omp parallel for collapse(2)
      for (int n1=0; n1 < (N_t*H_t)/QKVO_BLOCKSIZE; n1++){
        for (int n2=0; n2 < (N_t*H_t)/QKVO_BLOCKSIZE; n2++){
          qkv_vnni_trans_tpp(&key_w_a[n1][n2][0][0], &qkv_w_vnni_a[n1][n2][0][0]);
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKVO_BLOCKSIZE) {
        for (int k = 0; k < (N_t*H_t); k += QKVO_BLOCKSIZE) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          if (b_vnni){
            kv_brgemm_tpp(&m_data_a[i][j][0], &qkv_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            k_trans_tpp(&tmp[0], &k_a[i][k*S_t + 2 * j]);
          }
          else {
            kv_brgemm_tpp(&m_data_a[i][j][0], &key_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            k_trans_tpp(&tmp[0], &k_a[i][k*S_t + j]); // [ 0*H_t*S_t + 0*S_t + j]
          }
        }
      }
    }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto k_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

auto v_cpy_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, N_t* H_t), EW_COPY);
start_time = std::chrono::high_resolution_clock::now(); // Start timing
{
  // RECORD_SCOPE(alpha_v_gemm, {v, m_data, value_w});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    if(b_vnni) {
      #pragma omp parallel for collapse(2)
      for (int n1=0; n1 < (N_t*H_t)/QKVO_BLOCKSIZE; n1++){
        for (int n2=0; n2 < (N_t*H_t)/QKVO_BLOCKSIZE; n2++){
          qkv_vnni_trans_tpp(&value_w_a[n1][n2][0][0], &qkv_w_vnni_a[n1][n2][0][0]);
        }
      }
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKVO_BLOCKSIZE) {
        for (int k = 0; k < (N_t*H_t); k += QKVO_BLOCKSIZE) {
          LIBXSMM_ALIGNED(T tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          if(b_vnni) {
            kv_brgemm_tpp(&m_data_a[i][j][0], &qkv_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            v_vnni_trans_tpp(&tmp[0], &v_a[i][j * N_t * H_t + 2*k]);
          } else {
            kv_brgemm_tpp(&m_data_a[i][j][0], &value_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            v_cpy_tpp(&tmp[0], &v_a[i][j * N_t * H_t + k]);
          }
        }
      }
    }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto v_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

long lda = H_t;
long ldb = QKVO_BLOCKSIZE;
long ldc = S_t;

start_time = std::chrono::high_resolution_clock::now(); // Start timing

auto a_zero_tpp = SCOPEIT(SetZeroTPP<T>(QKVO_BLOCKSIZE * H_t), EW_ZERO);
auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(QKVO_BLOCKSIZE, H_t, H_t, N_t* H_t), EW_COPY);

if (S_t < 2560) {
  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
      QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, H_t, 1, 1, N_t * H_t, S_t, S_t, 0.0, 0, 1, b_vnni)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      QKVO_BLOCKSIZE,
      H_t,
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE,
      QKVO_BLOCKSIZE * N_t * H_t,
      S_t,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1,
      b_vnni)));

  auto a_addbias_tpp = SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, S_t, S_t), BIAS);
  auto a_add_nbbias_tpp =
      SCOPEIT((AddTPP<float, float>(QKVO_BLOCKSIZE, S_t, S_t, S_t)), BIAS);

  auto a_add_sfmask_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, S_t - Sp_t, ldc), BIAS);
  auto a_softmax_tpp =
      SCOPEIT((VarSoftMaxFwdTPP<float, T>(QKVO_BLOCKSIZE, S_t)), SOFTMAX);

  {
    // RECORD_SCOPE(alpha_ac_gemm, {q, k, bias});
    {
      // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());

#pragma omp parallel for collapse(3)
      for (int i = 0; i < B_t; i++) {
        for (int n = 0; n < N_t; n++) {
          for (int j1 = 0; j1 < S_t; j1 += QKVO_BLOCKSIZE) {
            LIBXSMM_ALIGNED(T tmp_o[QKVO_BLOCKSIZE * H_t], 64);
            LIBXSMM_ALIGNED(T tmp_logits_bf16[QKVO_BLOCKSIZE][S_t], 64);
            LIBXSMM_ALIGNED(float tmp_logits[QKVO_BLOCKSIZE][S_t], 64);

            a_brgemm_tpp.config();
            for (int j2 = 0; j2 < S_t; j2 += QKVO_BLOCKSIZE) {
              if (b_vnni) {
                a_brgemm_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + 2 * j2],
                    &tmp_logits[0][j2],
                    1,
                    true);
              } else {
                a_brgemm_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + j2],
                    &tmp_logits[0][j2],
                    1,
                    true);
              }
            }
            a_brgemm_tpp.release();
            if (bias_flag)
              a_addbias_tpp(&bias_a[i][0], &tmp_logits[0][0]);

            if (nbbias_flag)
              a_add_nbbias_tpp(
                  &nonbatched_bias_a[0][n][j1][0],
                  &tmp_logits[0][0],
                  &tmp_logits[0][0]);

            if (S_t == Sp_t) {
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
            } else {
              a_add_sfmask_tpp(&sfmask_a[0][0], &tmp_logits[0][Sp_t]);
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
            }
            
            if (b_vnni) {
              c_brgemm_tpp(
                  &tmp_logits_bf16[0][0],
                  &v_a[i][n * H_t * 2],
                  &tmp_o[0],
                  S_t / QKVO_BLOCKSIZE,
                  false);
            } else {
              c_brgemm_tpp(
                  &tmp_logits_bf16[0][0],
                  &v_a[i][n * H_t],
                  &tmp_o[0],
                  S_t / QKVO_BLOCKSIZE,
                  false);
            }
            a_cpy_tpp(&tmp_o[0], &weighted_avg_a[i][j1][n * H_t]);
          }
        }
      }
    }
  }
} else {
  // Flash attention implementation
  auto a_cpy2_tpp = SCOPEIT(CpyTPP<float>(QKVO_BLOCKSIZE, H_t), EW_COPY);

  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
      QKVO_BLOCKSIZE,
      Ak_BLOCKSIZE,
      H_t,
      1,
      1,
      N_t * H_t,
      S_t,
      Ak_BLOCKSIZE,
      0.0,
      0,
      1,
      b_vnni)));

  auto c_brgemm_online_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
      QKVO_BLOCKSIZE,
      H_t,
      Ak_BLOCKSIZE,
      1,
      1,
      Ak_BLOCKSIZE,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1,
      b_vnni)));

  auto a_addbias_online_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, Ak_BLOCKSIZE, Ak_BLOCKSIZE), BIAS);

  auto a_add_nbbias_online_tpp = SCOPEIT(
      (AddTPP<float, float>(
          QKVO_BLOCKSIZE, Ak_BLOCKSIZE, S_t, Ak_BLOCKSIZE, Ak_BLOCKSIZE)),
      BIAS);

  auto a_softmax_online_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(QKVO_BLOCKSIZE, Ak_BLOCKSIZE, true)), SOFTMAX);
  auto a_softmax_fixup_online =
      SCOPEIT(SoftMaxFixUpTPP<float>(QKVO_BLOCKSIZE, H_t, true), EW_RCP);
  auto a_softmax_scale_online =
      SCOPEIT(SoftMaxFlashScaleTPP<float>(QKVO_BLOCKSIZE, H_t, true), EW_RCP);

  auto a_convert_tpp = SCOPEIT(
      (ConvertTPP<float, T>(QKVO_BLOCKSIZE, H_t, H_t, N_t * H_t)), EW_ZERO);

  // if (S_t % Ak_BLOCKSIZE != 0){
  int lastBlockSize = S_t - (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
  if (lastBlockSize == 0)
    lastBlockSize = Ak_BLOCKSIZE; // handling the zero case
  auto a_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
      QKVO_BLOCKSIZE,
      lastBlockSize,
      H_t,
      1,
      1,
      N_t * H_t,
      S_t,
      lastBlockSize,
      0.0,
      0,
      1,
      b_vnni)));

  auto c_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
      QKVO_BLOCKSIZE,
      H_t,
      lastBlockSize,
      1,
      1,
      lastBlockSize,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1,
      b_vnni)));

  auto a_addbias_online_edge_tpp = SCOPEIT(
      AddBiasTPP<float>(QKVO_BLOCKSIZE, lastBlockSize, lastBlockSize), BIAS);
  auto a_add_nbbias_online_edge_tpp = SCOPEIT(
      (AddTPP<float, float>(
          QKVO_BLOCKSIZE, lastBlockSize, S_t, lastBlockSize, lastBlockSize)),
      BIAS);

  auto a_add_sfmask_online_tpp =
      SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, S_t - Sp_t, lastBlockSize), BIAS);

  auto a_softmax_online_edge_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(QKVO_BLOCKSIZE, lastBlockSize, true)), SOFTMAX);

  int max_threads = omp_get_max_threads();
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();

    LIBXSMM_ALIGNED(float tmp_S[QKVO_BLOCKSIZE * Ak_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(T tmp_S_bf16[QKVO_BLOCKSIZE * Ak_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float tmp_o1[QKVO_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float tmp_o2[QKVO_BLOCKSIZE * H_t], 64);
    LIBXSMM_ALIGNED(float omax[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float osum[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float cmax[QKVO_BLOCKSIZE], 64);
    LIBXSMM_ALIGNED(float csum[QKVO_BLOCKSIZE], 64);

    #pragma omp for collapse(3) nowait
      for (int i = 0; i < B_t; i++) {
        for (int n = 0; n < N_t; n++) {
          for (int j1 = 0; j1 < S_t; j1 += QKVO_BLOCKSIZE) {
            for (int j2 = 0; j2 < (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
                 j2 += Ak_BLOCKSIZE) {
              if (b_vnni) {
                a_brgemm_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + 2 * j2],
                    tmp_S,
                    1,
                    false);
              } else {
                a_brgemm_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + j2],
                    tmp_S,
                    1,
                    false);
              }

              if (bias_flag)
                a_addbias_online_tpp(&bias_a[i][j2], tmp_S);

              if (nbbias_flag) {
                a_add_nbbias_online_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S, tmp_S);
              }

              if (j2 == 0) {
                a_softmax_online_tpp(1, tmp_S, tmp_S_bf16, omax, osum, nullptr);
              } else {
                a_softmax_online_tpp(1, tmp_S, tmp_S_bf16, cmax, csum, omax);
              }

              if (b_vnni) {
                c_brgemm_online_tpp(
                    tmp_S_bf16,
                    &v_a[i][j2 * N_t * H_t + n * H_t * 2],
                    tmp_o1,
                    1,
                    false); // O = P*V
              } else {
                c_brgemm_online_tpp(
                    tmp_S_bf16,
                    &v_a[i][j2 * N_t * H_t + n * H_t],
                    tmp_o1,
                    1,
                    false); // O = P*V
              }

              if (j2 == 0) {
                a_cpy2_tpp(tmp_o1, tmp_o2);
              } else {
                a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              }
            }

            if (S_t % Ak_BLOCKSIZE != 0) {
              float* tmp_S_edge = new (std::align_val_t(64)) float[QKVO_BLOCKSIZE * lastBlockSize];
              T* tmp_S_bf16_edge = new (std::align_val_t(64)) T[QKVO_BLOCKSIZE * lastBlockSize];
              int j2 = (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
              if (b_vnni) {
                a_brgemm_edge_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + 2 * j2],
                    tmp_S_edge,
                    1);
              } else {
                a_brgemm_edge_tpp(
                    &q_a[i][j1][n * H_t],
                    &k_a[i][n * H_t * S_t + j2],
                    tmp_S_edge,
                    1);
              }

              if (bias_flag)
                a_addbias_online_edge_tpp(&bias_a[i][j2], tmp_S_edge);
              if (nbbias_flag) {
                a_add_nbbias_online_edge_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S_edge, tmp_S_edge);
              }

              a_add_sfmask_online_tpp(&sfmask_a[0][0], &tmp_S_edge[Sp_t - j2]);
              a_softmax_online_edge_tpp(
                  1, tmp_S_edge, tmp_S_bf16_edge, cmax, csum, omax);

              if (b_vnni) {
                c_brgemm_edge_tpp(
                    tmp_S_bf16_edge,
                    &v_a[i][j2 * N_t * H_t + n * H_t * 2],
                    tmp_o1,
                    1);
              } else {
                c_brgemm_edge_tpp(
                    tmp_S_bf16_edge,
                    &v_a[i][j2 * N_t * H_t + n * H_t],
                    tmp_o1,
                    1);
              }
              a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              delete[] tmp_S_edge;
              delete[] tmp_S_bf16_edge;
            }

            a_softmax_scale_online(&tmp_o2[0], osum);
            a_convert_tpp(&tmp_o2[0], &weighted_avg_a[i][j1][n * H_t]);
          }
        }
      }
  }
}

end_time = std::chrono::high_resolution_clock::now(); // End timing
auto ac_gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();


auto g_addbias_tpp = SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE), BIAS);
auto g_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE), EW_MUL);
auto g_mul_tpp = SCOPEIT((MulTPP<T, T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, N_t*H_t, N_t*H_t)), EW_MUL);
auto g_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, N_t*H_t)), EW_ZERO);

auto out_addbias_tpp = SCOPEIT(AddBiasTPP<float>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE), BIAS);

auto out_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, N_t*H_t)), EW_ZERO);

T* output_w_vnni = new (std::align_val_t(64)) T[N_t * H_t * HS_t];
auto output_w_vnni_a = GetVLAPtr<T>(output_w_vnni, {(N_t*H_t)/QKVO_BLOCKSIZE, QKVO_BLOCKSIZE, QKVO_BLOCKSIZE});

start_time = std::chrono::high_resolution_clock::now(); // Start timing

{
  // RECORD_SCOPE(alpha_o_gemm, {weighted_avg, v, q_data, gating_w, gating_b});
  {
    // RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    if(b_vnni) {
      #pragma omp parallel for collapse(2)
      for (int n1=0; n1 < (N_t*H_t)/QKVO_BLOCKSIZE; n1++){
        for (int n2=0; n2 < (N_t*H_t)/QKVO_BLOCKSIZE; n2++){
          qkv_vnni_trans_tpp(&gating_w_a[n1][n2][0][0], &qkv_w_vnni_a[n1][n2][0][0]);
        }
      }
      #pragma omp parallel for collapse(2)
      for (int n1=0; n1 < (N_t*H_t)/QKVO_BLOCKSIZE; n1++){
        for (int n2=0; n2 < (N_t*H_t)/QKVO_BLOCKSIZE; n2++){
          qkv_vnni_trans_tpp(&output_w_a[n1][n2][0][0], &output_w_vnni_a[n1][n2][0][0]);
        }
      }
    }
    if (gate_flag){
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += QKVO_BLOCKSIZE) {
        LIBXSMM_ALIGNED(T tmp_bf16[QKVO_BLOCKSIZE * N_t * H_t], 64);
        for (int k=0; k < (N_t*H_t); k += QKVO_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          LIBXSMM_ALIGNED(float tmp_gate_values[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          if (b_vnni)
            qgo_brgemm_tpp(&q_data_a[i][j][0], &qkv_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
          else
            qgo_brgemm_tpp(&q_data_a[i][j][0], &gating_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);

          g_addbias_tpp(&gating_b_a[0][k], &tmp[0]);

          g_sigmoid_tpp(&tmp[0], &tmp[0], &tmp_gate_values[0]);
          g_convert_tpp(&tmp_gate_values[0], &tmp_bf16[k]);
          g_mul_tpp(
              &tmp_bf16[k],
              &weighted_avg_a[i][j][k],
              &tmp_bf16[k]);
        }
        for (int k=0; k < (N_t * H_t); k += QKVO_BLOCKSIZE) {
          LIBXSMM_ALIGNED(float tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
          if (b_vnni)
            qgo_brgemm_tpp(
                &tmp_bf16[0], &output_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
          else
            qgo_brgemm_tpp(
                &tmp_bf16[0], &output_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
          if (bias_flag)
            out_addbias_tpp(&output_b_a[0][k], &tmp[0]);
          out_convert_tpp(&tmp[0], &output_a[i][j][k]);
        }
      }
    }
  } else {
#pragma omp parallel for collapse(3)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < S_t; j += QKVO_BLOCKSIZE) {
          for (int k=0; k < (N_t*H_t); k += QKVO_BLOCKSIZE) {
            LIBXSMM_ALIGNED(float tmp[QKVO_BLOCKSIZE * QKVO_BLOCKSIZE], 64);
            if (b_vnni)
              qgo_brgemm_tpp(&weighted_avg_a[i][j][0], &output_w_vnni_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            else
              qgo_brgemm_tpp(&weighted_avg_a[i][j][0], &output_w_a[k/QKVO_BLOCKSIZE][0][0][0], &tmp[0], (N_t*H_t)/QKVO_BLOCKSIZE);
            if (bias_flag)
              out_addbias_tpp(&output_b_a[0][k], &tmp[0]);
            out_convert_tpp(&tmp[0], &output_a[i][j][k]);
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