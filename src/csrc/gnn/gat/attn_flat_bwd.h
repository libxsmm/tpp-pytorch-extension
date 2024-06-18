/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("attn_flat_bwd", std::vector<c10::IValue>());

int i = 0;

const int threads = omp_get_max_threads();

auto t_grad_out = inputs[i++].contiguous(); // 3D shape [N, H, 1]

auto t_in = inputs[i++];
auto t_attn = inputs[i++]; // [1, H, F]

auto in_sizes = t_in.sizes();
auto attn_sizes = t_attn.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto N = in_sizes[0];
auto H = attn_sizes[1];
auto F = attn_sizes[2];

auto t_grad_in = t_in.new_empty({N, H, F});

auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto t_grad_attn = t_attn.new_empty({H * F});
auto t_grad_out_int =
    t_grad_out.view({N, H}); // squeeze(-1); // (N, H, 1) -> (N, H)
auto t_attn_tmp = t_attn.view({H * F});

auto in = GetVLAPtr<Tact>(t_in, {bn, H, F});
auto grad_out = GetVLAPtr<Tact>(t_grad_out_int, {bn, H});
auto attn = GetVLAPtr<Tprm>(t_attn_tmp, {F});
auto grad_in_H_F = GetVLAPtr<Tact>(t_grad_in, {bn, H, F}); // (nn, bn, H, F)
auto grad_attn = GetVLAPtr<Tprm>(t_grad_attn, {H, F});

auto set_attn_zero_tpp = SCOPEIT(SetZeroTPP<Tprm>(H * F), EW_ZERO);
auto mul_bcast_tpp = SCOPEIT(
    (BCastMulTPP<Tact, Tprm, Tact>(H, F)),
    EW_MUL); // Eltwise Brtoadcast and Multiplication
auto mul_add_bcast_tpp = SCOPEIT((BCastMulAddTPP<Tact, Tprm>(H, F)), EW_ADD);

//==========================================Attn

{
  RECORD_SCOPE(ga_dattn, {t_grad_out, t_grad_in, t_grad_attn});
  {
    tensor_set_zero(H, F, t_grad_attn);
    Tprm* attn_ptrs[threads];

    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();

        Tprm prv_grad_attn[H][F];
        attn_ptrs[tid] = prv_grad_attn[0];
        set_attn_zero_tpp(prv_grad_attn[0]);

#pragma omp for
        for (int n = 0; n < nn; n++) {
          for (int b = 0; b < bn; b++) {
            mul_bcast_tpp(grad_out[n][b], attn[0], grad_in_H_F[n][b][0]);
          }
          mul_add_bcast_tpp(grad_out[n][0], in[n][0][0], prv_grad_attn[0]);
        } // nn
        omp_reduce_buf(threads, H * F, attn_ptrs, grad_attn[0][0]);
      } // omp parallel

      if (rem > 0) {
        // Grad_attn---------------------------------------------------
        auto in = GetVLAPtr<Tact>(t_in, {H, F}); // (nn, bn, H, F)
        auto attn = GetVLAPtr<Tprm>(t_attn_tmp, {F}); // (H, F)
        auto grad_out = GetVLAPtr<Tact>(t_grad_out_int, {H}); // (nn, bn, H)
        auto grad_in_H_F = GetVLAPtr<Tact>(t_grad_in, {H, F}); // (nn, bn, H, F)
        auto grad_attn = GetVLAPtr<Tprm>(t_grad_attn, {H, F});

        auto set_attn_zero_tpp = SCOPEIT(SetZeroTPP<Tprm>(H * F), EW_ZERO);
        auto mul_bcast_tpp =
            SCOPEIT((BCastMulTPP<Tact, Tprm, Tact>(H, F)), EW_MUL);
        auto mul_add_bcast_tpp =
            SCOPEIT((BCastMulAddTPP<Tact, Tprm>(H, F)), EW_ADD);

        int tid = omp_get_thread_num();
        Tprm prv_grad_attn[H][F];
        attn_ptrs[tid] = prv_grad_attn[0];
        set_attn_zero_tpp(prv_grad_attn[0]);
        for (int r = nn * bn; r < nn * bn + rem; r++) {
          mul_bcast_tpp(grad_out[r], attn[0], grad_in_H_F[r][0]); // N, H) ->
                                                                  // (N, HF) ->
                                                                  // (HF) * (N,
                                                                  // HF) -> (N,
                                                                  // HF)
          mul_add_bcast_tpp(
              grad_out[r], in[r][0], prv_grad_attn[0]); // (N, HF) *
                                                        // (N, HF) ->
                                                        // (N, HF) ->
                                                        // (N, HF) +
                                                        // (HF) ->
                                                        // (HF)
        }
        omp_reduce_buf(1, H * F, attn_ptrs, grad_attn[0][0], true);
      } // rem > 0
    }
  }
}

return {t_grad_in, t_grad_attn.view({1, H, F})};
