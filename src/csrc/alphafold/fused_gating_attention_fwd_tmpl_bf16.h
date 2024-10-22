/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION(
    "Gating attention forward",
    std::vector<c10::IValue>({q_data, m_data})); // For recording time

int64_t B_t = q_data.size(0); /* Batch (512) */
int64_t Sp_t = q_data.size(1); /* Query (764) */
int64_t HS_t = q_data.size(2); /* Channels (256) */

int64_t N_t = query_w.size(1); /* number of heads (8) */
int64_t H_t = query_w.size(2); /* head size (32) */

auto flag = nonbatched_bias.size(0) > 0;

int64_t S_t = Sp_t;
if (Sp_t % QKV_BLOCKSIZE != 0) {
  S_t = (Sp_t / QKV_BLOCKSIZE + 1) * QKV_BLOCKSIZE; // 768

  auto q_data_pad = q_data.new_zeros({B_t, S_t - Sp_t, HS_t});
  auto m_data_pad = m_data.new_zeros({B_t, S_t - Sp_t, HS_t});
  auto bias_pad = bias.new_zeros({B_t, 1, 1, S_t - Sp_t});
  auto nonbatched_bias_pad1 =
      nonbatched_bias.new_zeros({N_t, Sp_t, S_t - Sp_t});
  auto nonbatched_bias_pad2 = nonbatched_bias.new_zeros({N_t, S_t - Sp_t, S_t});

  q_data = at::cat({q_data, q_data_pad}, 1);
  m_data = at::cat({m_data, m_data_pad}, 1);
  bias = at::cat({bias, bias_pad}, 3);
  if (flag) {
    nonbatched_bias = at::cat({nonbatched_bias, nonbatched_bias_pad1}, 2);
    nonbatched_bias = at::cat({nonbatched_bias, nonbatched_bias_pad2}, 1);
  }
}

bias = bias.contiguous();
nonbatched_bias = nonbatched_bias.contiguous();
auto sfmask = -30000 * q_data.new_ones(S_t - Sp_t, at::kFloat).contiguous();
auto sfmask_a = GetVLAPtr<float>(sfmask, {1L});

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

auto q = q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 8, 32] */
auto q_a = GetVLAPtr<T>(q, {S_t, N_t, H_t});

auto k = q_data.new_empty({B_t, S_t * N_t * H_t}); /* [512, 764, 8, 32] */
auto k_a = GetVLAPtr<T>(k, {S_t * N_t * H_t});

auto v = q_data.new_empty({B_t, S_t * N_t * H_t}); /* [512, 764, 8, 32] */
auto v_a = GetVLAPtr<T>(v, {S_t * N_t * H_t});

auto weighted_avg =
    q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 8, 32] */
auto weighted_avg_a = GetVLAPtr<T>(weighted_avg, {S_t, N_t, H_t});

auto output = q_data.new_empty({B_t, S_t, HS_t}); /* [512, 764, 256] */
auto output_a = GetVLAPtr<T>(output, {S_t, HS_t});

int lda = HS_t;
int ldb = N_t * H_t;
int ldc = N_t * H_t;

auto q_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(QKV_BLOCKSIZE, N_t* H_t, HS_t, 1, 1, lda, ldb, ldc, 0.0, 0, 1)));
auto q_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(QKV_BLOCKSIZE * HS_t)), EW_ZERO);

auto scale_tpp =
    SCOPEIT((ScaleTPP<float, float>(QKV_BLOCKSIZE * HS_t)), EW_SCL);
auto zero_tpp = SCOPEIT(SetZeroTPP<float>(QKV_BLOCKSIZE * HS_t), EW_ZERO);
float alpha = (1.0 / sqrt(key_dim));

auto qkv_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(HS_t, N_t* H_t, HS_t, N_t* H_t, ldb, ldb, XformTPP::XFORM_N2V_TPP),
    VNNI);
auto qkv_w_vnni = q_data.new_empty({HS_t, N_t, H_t}); /* [256, 8, 32] */
auto qkv_w_vnni_a = GetVLAPtr<T>(qkv_w_vnni, {N_t, H_t});

// auto q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}),
// (1.0/sqrt(key_dim))) ;     /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] */
{
  RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w});
  {
    qkv_vnni_trans_tpp(&query_w_a[0][0][0], &qkv_w_vnni_a[0][0][0]);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// #pragma omp parallel for collapse(2)
  #pragma omp parallel
    {
      q_brgemm_tpp.config();
      #pragma omp for collapse(2)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
          float tmp[QKV_BLOCKSIZE][N_t][H_t];
          zero_tpp(&tmp[0][0][0]);
          q_brgemm_tpp(
              &q_data_a[i][j][0], &qkv_w_vnni_a[0][0][0], &tmp[0][0][0], 1, true);
          scale_tpp(&tmp[0][0][0], &tmp[0][0][0], alpha);
          q_convert_tpp(&tmp[0][0][0], &q_a[i][j][0][0]);
        }
      }
      q_brgemm_tpp.release();
    }
  }
}

auto k_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKV_BLOCKSIZE,
        N_t*H_t,
        N_t*H_t,
        QKV_BLOCKSIZE,
        N_t* H_t,
        S_t,
        XformTPP::XFORM_XPOSE_N2V_TPP),
    XPOSE);

auto v_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKV_BLOCKSIZE,
        N_t*H_t,
        QKV_BLOCKSIZE,
        N_t*H_t,
        N_t* H_t,
        N_t* H_t,
        XformTPP::XFORM_N2V_TPP),
    VNNI);

auto kv_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKV_BLOCKSIZE, N_t* H_t, HS_t, 1, 1, lda, ldb, ldc, 0.0, 0, 1)));
// auto k = at::einsum("bka,ahc->bkhc", {m_data, key_w});
// /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
{
  RECORD_SCOPE(alpha_k_gemm, {k, m_data, key_w});
  {
    qkv_vnni_trans_tpp(&key_w_a[0][0][0], &qkv_w_vnni_a[0][0][0]);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// #pragma omp parallel for collapse(2)
  #pragma omp parallel
    {
      kv_brgemm_tpp.config();
      #pragma omp for collapse(2)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
          T tmp[QKV_BLOCKSIZE*N_t*H_t];
          kv_brgemm_tpp(
              &m_data_a[i][j][0], &qkv_w_vnni_a[0][0][0], &tmp[0], 1, true);
          k_trans_tpp(&tmp[0], &k_a[i][2*j]);    // B, S, N, H ----> B, N, H, S (FP32)  ---> B, N, H/2, S, 2
        }
      }
      kv_brgemm_tpp.release();
    }
  }
}


// auto v = at::einsum("bka,ahc->bkhc", {m_data, value_w});
// /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
{
  RECORD_SCOPE(alpha_v_gemm, {v, m_data, value_w});
  {
    qkv_vnni_trans_tpp(&value_w_a[0][0][0], &qkv_w_vnni_a[0][0][0]);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// #pragma omp parallel for collapse(2)
    #pragma omp parallel
    {
      kv_brgemm_tpp.config();
      #pragma omp for collapse(2)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < S_t; j += QKV_BLOCKSIZE) {
          // kv_brgemm_tpp(
          //     &m_data_a[i][j][0], &qkv_w_vnni_a[0][0][0], &v_a[i][j][0][0], 1);
          T tmp[QKV_BLOCKSIZE*N_t*H_t];
          kv_brgemm_tpp(
              &m_data_a[i][j][0], &qkv_w_vnni_a[0][0][0], &tmp[0], 1, true);
          v_vnni_trans_tpp(&tmp[0], &v_a[i][j*N_t*H_t]);
        }
      }
      kv_brgemm_tpp.release();
    }
  }
}

lda = H_t;
ldb = A_BLOCKSIZE;
ldc = S_t;

auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t, N_t* H_t, H_t), EW_COPY);

auto a_zero_tpp = SCOPEIT(SetZeroTPP<T>(A_BLOCKSIZE * H_t), EW_ZERO);
auto a_cpy2_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t, H_t, N_t* H_t), EW_COPY);

auto a_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(A_BLOCKSIZE, A_BLOCKSIZE, H_t, 1, 1, H_t, S_t, S_t, 0.0, 0, 1)));

// auto a_brgemm2_tpp = SCOPEITGEMM(
//     (BrgemmTPP<
//         T,
//         T>(A_BLOCKSIZE, H_t, A_BLOCKSIZE, 1, 1, S_t, N_t*H_t, H_t, 1.0, 0, 1)));

auto a_brgemm2_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(A_BLOCKSIZE, H_t, A_BLOCKSIZE, A_BLOCKSIZE, A_BLOCKSIZE*N_t*H_t, S_t, N_t*H_t, H_t, 1.0, 0, 1)));

// auto a_addbias_tpp =
//     SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, A_BLOCKSIZE, S_t), BIAS);

// auto a_add_nbbias_tpp =
//     SCOPEIT((AddTPP<float, float>(A_BLOCKSIZE, A_BLOCKSIZE, S_t, S_t)), BIAS);

auto a_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        A_BLOCKSIZE,
        H_t,
        A_BLOCKSIZE,
        H_t,
        N_t* H_t,
        lda,
        XformTPP::XFORM_N2V_TPP),
    VNNI);

auto a_add_sfmask_tpp =
    SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, S_t - Sp_t, ldc), BIAS);
auto a_softmax_tpp =
    SCOPEIT((VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, S_t)), SOFTMAX);

// logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);
// /* [512, 8, 764, 764]  = [512, 764, 8, 32] * [512, 764, 8, 32] + [512, 1, 1,
// 764] */ if (nonbatched_bias.size(0) > 0)
//     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));
//     /* [512, 8, 764, 764]  = [512, 8, 764, 764] + [1, 8, 764, 764] */
// weights = at::_softmax(logits, -1, false);
// /* [512, 8, 764, 764] = [512, 8, 764, 764] */ auto weighted_avg =
// at::einsum("bhqk,bkhc->bqhc", {weights, v}).contiguous();          /* [512,
// 764, 8, 32]  = [512, 8, 764, 764] * [512, 764, 8, 32] */

auto a_addbias2_tpp =
    SCOPEIT(AddBiasTPP<float>(A_BLOCKSIZE, S_t, S_t), BIAS);
auto a_add_nbbias2_tpp =
    SCOPEIT((AddTPP<float, float>(A_BLOCKSIZE, S_t, S_t, S_t)), BIAS);

// auto a_scale_tpp =
//     SCOPEIT((ScaleTPP<float, float>(A_BLOCKSIZE * S_t)), EW_SCL);

{
  RECORD_SCOPE(alpha_a_gemm, {q, k, bias});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
    for (int i = 0; i < B_t; i++) {
      for (int n = 0; n < N_t; n++) {
        for (int j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
          T tmp_qv[A_BLOCKSIZE * H_t];
          T tmp_logits_bf16[A_BLOCKSIZE][S_t];
          float tmp_logits[A_BLOCKSIZE][S_t]; // Convert this into float
          a_cpy_tpp(&q_a[i][j1][n][0], &tmp_qv[0]);

          a_brgemm_tpp.config();
          for (int j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
            a_brgemm_tpp(&tmp_qv[0], &k_a[i][n*H_t*S_t + 2*j2], &tmp_logits[0][j2], 1, true);
            // a_addbias_tpp(&bias_a[i][j2], &tmp_logits[0][j2]);
            // if (flag) {
            //   a_add_nbbias_tpp(
            //       &nonbatched_bias_a[0][n][j1][j2],
            //       &tmp_logits[0][j2],
            //       &tmp_logits[0][j2]);
            // }
          }
          a_brgemm_tpp.release();
          a_addbias2_tpp(&bias_a[i][0], &tmp_logits[0][0]);
          if(flag)
            a_add_nbbias2_tpp(&nonbatched_bias_a[0][n][j1][0], &tmp_logits[0][0], &tmp_logits[0][0]);

          if (S_t == Sp_t) {
            a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
            // Have a new tmp for tmp_logits in bf16
          } else {
            a_add_sfmask_tpp(&sfmask_a[0][0], &tmp_logits[0][Sp_t]);
            a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
          }

          a_zero_tpp(&tmp_qv[0]);
          // for (int j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
            // a_brgemm2_tpp(&tmp_logits_bf16[0][j2], &v_a[i][j2*N_t*H_t + n*H_t*2], &tmp_qv[0], 1, true);
          // }
          a_brgemm2_tpp(&tmp_logits_bf16[0][0], &v_a[i][n*H_t*2], &tmp_qv[0], S_t/A_BLOCKSIZE, false);
          a_cpy2_tpp(&tmp_qv[0], &weighted_avg_a[i][j1][n][0]);
        }
      }
    }
  }
}

// {
//   RECORD_SCOPE(alpha_a_gemm, {q, k, bias});
//   {
//     RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// #pragma omp parallel for collapse(2)
//     for (int i = 0; i < B_t; i++) {
//       for (int j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
//         T tmp_qv[A_BLOCKSIZE * H_t];
//         T tmp_k[A_BLOCKSIZE * H_t];
//         T tmp_logits_bf16[A_BLOCKSIZE][S_t];
//         float tmp_logits[A_BLOCKSIZE][S_t]; // Convert this into float

//         for (int k = 0; k < N_t; k++) {
//           a_cpy_tpp(&q_a[i][j1][k][0], &tmp_qv[0]);

//           a_brgemm_tpp.config();
//           for (int j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
//             a_trans_tpp(
//                 &k_a[i][j2][k][0],
//                 &tmp_k[0]); // [A_BLOCKSIZE, 8, 32]  ----> [32, A_BLOCKSIZE]

//             a_brgemm_tpp(&tmp_qv[0], &tmp_k[0], &tmp_logits[0][j2], 1, true);
//             a_addbias_tpp(&bias_a[i][j2], &tmp_logits[0][j2]);
//             if (flag) {
//               a_add_nbbias_tpp(
//                   &nonbatched_bias_a[0][k][j1][j2],
//                   &tmp_logits[0][j2],
//                   &tmp_logits[0][j2]);
//             }
//           }
//           a_brgemm_tpp.release();
//           a_brgemm2_tpp.config();
//           if (S_t == Sp_t) {
//             a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
//             // Have a new tmp for tmp_logits in bf16
//           } else {
//             a_add_sfmask_tpp(&sfmask_a[0][0], &tmp_logits[0][Sp_t]);
//             a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits_bf16[0][0]);
//           }

//           a_zero_tpp(&tmp_qv[0]);
//           for (int j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
//             a_vnni_trans_tpp(&v_a[i][j2][k][0], &tmp_k[0]);
//             a_brgemm2_tpp(&tmp_logits_bf16[0][j2], &tmp_k[0], &tmp_qv[0], 1, true);
//           }
//           a_brgemm2_tpp.release();
//           a_cpy2_tpp(&tmp_qv[0], &weighted_avg_a[i][j1][k][0]);
//         }
//       }
//     }
//   }
// }

lda = HS_t;
ldb = N_t * H_t;
ldc = N_t * H_t;

auto c_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(C_BLOCKSIZE, N_t* H_t, HS_t, 1, 1, lda, ldb, ldc, 0.0, 0, 1)));
auto c_addbias_tpp =
    SCOPEIT(AddBiasTPP<float>(C_BLOCKSIZE, N_t* H_t, ldc), BIAS);
auto c_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<float>(C_BLOCKSIZE, N_t* H_t, ldc, ldc), EW_MUL);
auto c_mul_tpp = SCOPEIT((MulTPP<T, T>(C_BLOCKSIZE * N_t * H_t)), EW_MUL);

auto c_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(C_BLOCKSIZE * N_t * H_t)), EW_ZERO);

auto out_gemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(C_BLOCKSIZE, HS_t, N_t* H_t, 1, 1, lda, ldb, ldc, 0.0, 0, 1)));
auto out_addbias_tpp = SCOPEIT(AddBiasTPP<float>(C_BLOCKSIZE, HS_t, ldc), BIAS);

auto out_convert_tpp =
    SCOPEIT((ConvertTPP<float, T>(C_BLOCKSIZE * HS_t)), EW_ZERO);

auto output_vnni_trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(N_t * H_t, HS_t, N_t* H_t, HS_t, lda, lda, XformTPP::XFORM_N2V_TPP),
    VNNI);
auto output_w_vnni = q_data.new_empty({N_t, H_t, HS_t}); /* [8, 32, 256] */
auto output_w_vnni_a = GetVLAPtr<T>(output_w_vnni, {H_t, HS_t});

// gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data,
// gating_w}), gating_b));   /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] + [8, 32]*/ weighted_avg = at::mul(weighted_avg, gate_values);
// /* [512, 764, 8, 32]  = [512, 764, 8, 32] * [512, 764, 8, 32] */ output =
// at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b);
// /* [512, 764, 256]  = [512, 764, 8, 32] * [8, 32, 256] + [256] */
{
  RECORD_SCOPE(alpha_c_gemm, {weighted_avg, v, q_data, gating_w, gating_b});
  {
    qkv_vnni_trans_tpp(&gating_w_a[0][0][0], &qkv_w_vnni_a[0][0][0]);
    output_vnni_trans_tpp(&output_w_a[0][0][0], &output_w_vnni_a[0][0][0]);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// #pragma omp parallel for collapse(2)
  #pragma omp parallel
    {
      c_brgemm_tpp.config();
      #pragma omp for collapse(2)
      for (int i = 0; i < B_t; i++) {
        for (int j = 0; j < S_t; j += C_BLOCKSIZE) {
          float tmp[C_BLOCKSIZE * N_t * H_t]; // Should be in float for bf16
          float tmp_gate_values[C_BLOCKSIZE * N_t * H_t];
          T tmp_bf16[C_BLOCKSIZE * N_t * H_t];

          c_brgemm_tpp(&q_data_a[i][j][0], &qkv_w_vnni_a[0][0][0], &tmp[0], 1, true);
          c_addbias_tpp(&gating_b_a[0][0], &tmp[0]);

          c_sigmoid_tpp(&tmp[0], &tmp[0], &tmp_gate_values[0]);

          c_convert_tpp(&tmp_gate_values[0], &tmp_bf16[0]);
          c_mul_tpp(&tmp_bf16[0], &weighted_avg_a[i][j][0][0], &tmp_bf16[0]);

          out_gemm_tpp(&tmp_bf16[0], &output_w_vnni_a[0][0][0], &tmp[0], 1, true);
          out_addbias_tpp(&output_b_a[0][0], &tmp[0]);
          out_convert_tpp(&tmp[0], &output_a[i][j][0]);
        }
      }
      c_brgemm_tpp.release();
    }
  }
}

if (S_t != Sp_t) {
  output = output.narrow(1, 0, Sp_t);
}

return output;

// int64_t b_t = q_data.size(0);                    /* Batch (512) */
// int64_t q_t = q_data.size(1);                    /* Query (764) */
// int64_t k_t = m_data.size(1);                    /* Key (764) */
// int64_t a_t = q_data.size(2);                  /* Channels (256) */

// int64_t h_t = query_w.size(1);                  /* number of heads (8) */
// int64_t c_t = query_w.size(2);                  /* head channels (32) */

// auto output = q_data.new_empty({b_t,q_t,a_t});

// auto q = q_data.new_empty({b_t,q_t,h_t,c_t});
// auto k = q_data.new_empty({b_t,k_t,h_t,c_t});
// auto v = q_data.new_empty({b_t,k_t,h_t,c_t});

// auto logits = q_data.new_empty({b_t,h_t,q_t,k_t});
// auto weights = q_data.new_empty({b_t,h_t,q_t,k_t});
// auto weighted_avg = q_data.new_empty({b_t,q_t,h_t,c_t});

// auto gate_values = q_data.new_empty({b_t,q_t,h_t,value_dim});

// q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}),
// (1.0/sqrt(key_dim))) ; q = at::einsum("bqa,ahc->bqhc", {q_data, query_w}) ;
// k = at::einsum("bka,ahc->bkhc", {m_data, key_w});
// v = at::einsum("bka,ahc->bkhc", {m_data, value_w});

// logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);
// // logits = at::mul(at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias),
// (1.0/sqrt(key_dim)));

// if (nonbatched_bias.size(0) > 0)
//     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));

// weights = at::_softmax(logits, -1, false);

// weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights, v});

// gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data,
// gating_w}), gating_b));

// weighted_avg = at::mul(weighted_avg, gate_values);

// output = at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}),
// output_b);

// return output;
